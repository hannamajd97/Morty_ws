"""Synchronised demonstration recorder for VLA fine-tuning.

Samples every source at a fixed rate (default 10 Hz) and writes the raw
episode format described in raw_dataset.py.

Where an episode lands (2026-09-24, Majd: "store each experiment's VLA data
inside its matching trial folder, next to bag/"):
    * a trial-bound episode (the GUI sends ``hint <absolute trial dir>`` before
      ``trial ...``) writes images/, meta.json and steps.npz INTO that trial
      folder; its episode id is the folder name (trial_064_LA_S4) and its
      episode_index the trial number, so trial and episode are one record;
    * an episode started without a trial folder (``start <task>`` from the CLI)
      goes to ~/morty_ws/vla_experiments/datasets/<dataset_name>/episode_NNNNNN
      as before.

Manual control over `/vla_recorder/cmd` (std_msgs/String) or the `vla_episode` CLI:
    start <task text> | stop [ok|fail|unknown|auto] | discard | task <text>
    trial <fallback task>   one MMHRI trial = one episode (see start_trial_episode)
    hint <trial dir>        where the next trial episode is written

Auto mode (parameter auto_mode:=true) records one episode per Morty trial without
touching the GUI or the MMHRI recorder, using the events the trial already emits:
    /speech/accepted_transcript  -> episode starts (task = the spoken command), provisional
    /robot/intent                -> JSON "action_intent"; if it is a motion intent the episode is
                                    confirmed, otherwise (question, chat) it is discarded
    /grasp_executor_lin_llm/cycle_done -> "done|<id>..." stops with success, "failed|..." with failure
    safety: no intent within intent_wait_s -> discard; longer than max_episode_s -> stop fail

What one step holds (see raw_dataset.STEP_KEYS): the ORIGINAL header stamp of
every stream next to the value (joint_stamp, img_stamp_*, tcp_stamp,
ur_tcp_stamp), so delays are measurable after the fact; the TCP pose from TF
is rejected (NaN) when its stamp is older than max_tf_age_s instead of
repeating the last transform (episode 2 of 09-22 recorded a frozen pose for
55 s that way); the driver's own /tcp_pose_broadcaster/pose is stored as an
independent witness (ur_tcp_*); the last gripper COMMAND and its age come
from the adapter's stamped command events, not from the finger joint that
follows a second later; and every step carries the index of the instruction
it belongs to, the executor phase, and a segment code (idle / task / reset)
so the export can drop chatter and resets but keep the holds a task needs.

Events with timestamps (meta.json "events"): accepted transcripts, intents,
gripper commands, executor phases, interactive status, cycle_done.
Instructions (meta.json "instructions"): one per motion intent — the
transcript that caused it plus a canonical wording built from the intent
(object, destination), which is what the export uses as the language
instruction; that is how a misheard "grass for me" still trains "pick up".

Outcome is explicit: success / failure / unknown. "stop auto" takes the
executor's cycle_done verdict from THIS episode, otherwise unknown — never
"true because nobody said no".

Before a trial episode opens, every required stream (joints, both cameras,
TF to the ee frame) must be fresh; otherwise the episode is refused and the
status says which stream is missing. At stop, meta.json "quality" lists
step gaps, image delays, stale poses and gripper-state lag; the converters
refuse such episodes unless told to keep them.
"""
from __future__ import annotations

import json
import queue
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from std_msgs.msg import String
from sensor_msgs.msg import Image, CameraInfo, CompressedImage, JointState


class _Jpeg:
    """A camera frame kept as its compressed bytes (see _on_cimg)."""
    __slots__ = ("data",)

    def __init__(self, data: bytes) -> None:
        self.data = data


from geometry_msgs.msg import PoseStamped
from control_msgs.msg import JointTrajectoryControllerState
from tf2_ros import Buffer, TransformException
from ur3e_fast_observation.tf_listener_lite import DecimatedTransformListener
from ur3e_fast_observation.latest_message import LatestMessage, joint_state_stream

from .raw_dataset import ARM_JOINTS, STEP_KEYS, SEGMENT_IDLE, SEGMENT_TASK, SEGMENT_RESET, canonical_instruction

try:
    from ur_msgs.msg import IOStates
except Exception:
    IOStates = None

import cv2
from cv_bridge import CvBridge

LATCHED = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
# Every intent that moves the arm or the gripper (2026-09-25: the LLM's answer
# to "where should I drop it?" is confirm_drop, and it was missing — trial 067's
# drop-and-hand-over was labelled with the pick instruction; the look / sleep /
# ready moves were missing too).
MOTION_INTENTS = ["pick_and_place", "pick", "place", "drop_held", "drop_left", "drop_right", "go_home",
                  "carry_release", "release_home", "release", "open_gripper", "close_gripper",
                  "confirm_drop", "look_left", "look_right", "look_up", "go_to_ready", "go_to_start",
                  "sleep",
                  # 2026-09-28 (Majd's trial 080: "Drop it on the right side. Give me
                  # the objects and go back home." came from the instant path as ONE
                  # compound action and got no instruction — the drop, the release and
                  # the way home were labelled "pick up the tape dispenser"): the
                  # compound names chat_node's instant layer publishes.
                  "drop_right_release", "drop_left_release", "drop_right_release_home",
                  "drop_left_release_home", "go_home_open"]
# 2026-09-28: the answers to the pointing question ("You pointed at the X.
# Should I grasp it?") — no motion of their own and no verdict: the pick
# they confirm arrives as the pipeline's selection record right after.
ANSWER_INTENTS = ("confirm_point", "reject_point")
OUTCOMES = ("success", "failure", "unknown")


def episode_task_instruction(instructions: list, events: dict) -> str:
    """The episode's task text: the last pick instruction before the arm's
    first motion, else the first instruction (2026-09-28, trial 082: "Grasp
    for me the vitamin tube." was refused — two tubes — and "The right most
    vitamin tube." is the pick that ran; the task was the refused one)."""
    if not instructions:
        return ""
    starts = sorted(e["stamp"] for e in events.get("executor_phase", []) if e.get("name") == "motion_start")
    picks = [i for i in instructions if i["action_intent"] in ("pick", "pick_and_place")
             and (not starts or i["stamp"] <= starts[0])]
    return (picks[-1] if picks else instructions[0])["instruction"]


def _stamp_s(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def _quat_mul(a, b):
    """xyzw quaternion product a*b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw,
                     aw * bw - ax * bx - ay * by - az * bz])


def _quat_rot(q, v):
    """Rotate vector v by xyzw quaternion q."""
    qv = np.array([v[0], v[1], v[2], 0.0])
    qc = np.array([-q[0], -q[1], -q[2], q[3]])
    return _quat_mul(_quat_mul(q, qv), qc)[:3]


class _ImageWriter:
    """Writes JPEGs on its own thread so a slow disk never stalls the tick
    (a stalled executor was one source of the 0.5-2 s step gaps seen in
    the 09-22 episodes: joint_states older than 0.2 s -> tick dropped)."""

    def __init__(self, logger) -> None:
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=400)
        self._log = logger
        self.failed = 0
        self._t = threading.Thread(target=self._run, name="vla-image-writer", daemon=True)
        self._t.start()

    def put(self, path: Path, sample, jpeg_q: int) -> None:
        try:
            self._q.put_nowait((path, sample, jpeg_q))
        except queue.Full:
            self.failed += 1

    def _run(self) -> None:
        while True:
            path, sample, jpeg_q = self._q.get()
            try:
                if isinstance(sample, _Jpeg):
                    path.write_bytes(sample.data)        # original JPEG, no decode/re-encode
                else:
                    cv2.imwrite(str(path), sample, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q])
            except Exception as e:  # noqa: BLE001
                self.failed += 1
                self._log.warn(f"image write failed: {e}")
            finally:
                self._q.task_done()

    def drain(self, timeout_s: float = 10.0) -> None:
        t0 = time.monotonic()
        while not self._q.empty() and time.monotonic() - t0 < timeout_s:
            time.sleep(0.02)


class EpisodeRecorder(Node):
    def __init__(self):
        super().__init__("vla_episode_recorder")
        self.declare_parameters("", [
            ("dataset_root", str(Path.home() / "morty_ws" / "vla_experiments" / "datasets")),
            ("dataset_name", "morty_ur3e"),
            ("fps", 10.0),
            ("exterior_image_topic", "/astra/color/image_raw"),
            ("exterior_compressed_topic", "/astra/color/compressed_slow"),
            ("exterior_info_topic", "/astra/color/camera_info"),
            ("wrist_image_topic", "/camera/camera/color/image_raw"),
            ("wrist_compressed_topic", "/camera/camera/color/image_raw/compressed"),
            ("wrist_info_topic", "/camera/camera/color/camera_info"),
            ("require_exterior", True),
            ("require_wrist", True),
            ("joint_states_topic", "/joint_states"),
            ("controller_state_topic", "/scaled_joint_trajectory_controller/controller_state"),
            ("tcp_pose_topic", "/tcp_pose_broadcaster/pose"),
            ("io_states_topic", "/io_and_status_controller/io_states"),
            ("gripper_cmd_topic", "/gripper_controller/command_event"),
            ("transcript_topic", "/speech/accepted_transcript"),
            ("task_from_transcript", False),
            ("auto_mode", False),
            ("motion_confirm_rad", 0.10),   # trial-bound: any arm joint moving this far confirms the episode
            ("intent_topic", "/robot/intent"),
            ("motion_intents", MOTION_INTENTS),
            ("cycle_done_topic", "/grasp_executor_lin_llm/cycle_done"),
            ("timing_topic", "/grasp_executor_lin_llm/timing"),
            ("selection_timing_topic", "/graspgen/timing"),   # the pipeline's per-pick record (label, source)
            ("interactive_status_topic", "/grasp_executor_lin_llm/interactive_status"),
            ("intent_wait_s", 25.0),
            ("max_episode_s", 240.0),
            ("mmhri_dir", str(Path.home() / "morty_ws" / "experiments")),   # day folders since 2026-09-18
            ("base_frame", "base_link"),
            ("ee_frame", "gripper_tcp"),
            ("tool_frame", "tool0"),
            ("gripper_joint", "gripper_close_open_joint"),
            ("gripper_open_pos", 0.23),
            ("grasp_pin", 16),
            ("max_image_age_s", 0.5),
            ("max_joint_age_s", 0.2),
            ("max_tf_age_s", 0.2),
            ("max_step_gap_s", 0.5),        # quality: a gap between steps longer than this is reported
            ("preflight", True),            # refuse to open an episode while a required stream is stale
            ("jpeg_quality", 95),
            ("astra_tf_xyz", [0.0, 0.0, 0.0]),
            ("astra_tf_rpy", [0.0, 0.0, 0.0]),
        ])
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.dataset_dir = Path(g("dataset_root")).expanduser() / g("dataset_name")
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        self.fps = float(g("fps"))
        self.require_ext, self.require_wrist = bool(g("require_exterior")), bool(g("require_wrist"))
        self.base_frame, self.ee_frame, self.tool_frame = g("base_frame"), g("ee_frame"), g("tool_frame")
        self.gripper_joint, self.gripper_open, self.grasp_pin = g("gripper_joint"), float(g("gripper_open_pos")), int(g("grasp_pin"))
        self.max_img_age, self.max_joint_age, self.jpeg_q = float(g("max_image_age_s")), float(g("max_joint_age_s")), int(g("jpeg_quality"))
        self.max_tf_age, self.max_step_gap = float(g("max_tf_age_s")), float(g("max_step_gap_s"))
        self.preflight = bool(g("preflight"))
        self.task_from_transcript, self.auto_mode = bool(g("task_from_transcript")), bool(g("auto_mode"))
        self.motion_confirm_rad = float(g("motion_confirm_rad"))
        self.motion_intents = set(str(x).lower() for x in g("motion_intents"))
        self.intent_wait_s, self.max_episode_s = float(g("intent_wait_s")), float(g("max_episode_s"))
        self.mmhri_dir = Path(g("mmhri_dir")).expanduser()
        self.astra_tf = {"xyz": list(g("astra_tf_xyz")), "rpy": list(g("astra_tf_rpy"))}

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        # Every live value is (receive_monotonic, header_stamp_ros_s, payload).
        self.img = {"exterior": None, "wrist": None}
        self.info = {"exterior": None, "wrist": None}
        self.img_source = {}
        self.joints = None            # (recv, stamp, pos, vel, gripper_norm)
        self.joint_cmd = None         # (recv, stamp, pos)
        self.ur_tcp = None            # (recv, stamp, frame, pos, quat)
        self.io_grasp = None          # (recv, ros_now, bool)  (IOStates has no header)
        self.gripper_cmd = None       # (recv, stamp, value, dict)
        self.base_from_ur = None      # static xyzw/xyz of base_frame <- ur_tcp frame, resolved once
        self.bound_to_trial = False; self.confirmed_by = ""; self.q_start = None
        self.task_fallback = ""
        self.transcript = None
        self.writer = _ImageWriter(self.get_logger())

        self.tf_buffer = Buffer()
        # 2026-09-18: the stock TransformListener deserialised the driver's
        # ~830 /tf messages per second on this node's executor and kept the
        # idle recorder at a full core — one of the CPU hogs that starved
        # the UR driver's real-time loop. The pipeline's decimating listener
        # (grasp_pipeline/tf_listener_lite.py) feeds the same buffer at 50 Hz.
        self.tf_listener = DecimatedTransformListener(self.tf_buffer, self, max_hz=50.0)

        for cam in ("exterior", "wrist"):
            ctopic = g(f"{cam}_compressed_topic"); rtopic = g(f"{cam}_image_topic")
            if ctopic:
                self.create_subscription(CompressedImage, ctopic, lambda m, c=cam: self._on_cimg(c, m), qos_profile_sensor_data)
                self.img_source[cam] = ctopic
            else:
                self.create_subscription(Image, rtopic, lambda m, c=cam: self._on_img(c, m), qos_profile_sensor_data)
                self.img_source[cam] = rtopic
        # 2026-09-29 (Majd: "the system is very slow"): the fast state topics
        # — /joint_states, the TCP pose and io_states at 500 Hz, the
        # controller state — were subscribed, >1500 Python callbacks a second
        # all day long that only overwrote "the latest value"; the idle
        # recorder took a full core. They are READ now (_poll_state: at each
        # tick while recording, and for the freshness check) from standing
        # readers nothing spins — the same handlers, the newest message.
        # /joint_states keeps the newest per stream (the arm and the
        # gripper adapter publish there) with room for a tick's messages.
        # The camera infos are read until the first one arrived.
        self._fast = [
            (LatestMessage(self, JointState, g("joint_states_topic"), depth=128,
                           stream_key=joint_state_stream), self._on_joints),
            (LatestMessage(self, JointTrajectoryControllerState, g("controller_state_topic")), self._on_ctrl),
            (LatestMessage(self, PoseStamped, g("tcp_pose_topic")), self._on_tcp),
        ]
        if IOStates is not None:
            self._fast.append((LatestMessage(self, IOStates, g("io_states_topic")), self._on_io))
        self._info_readers = {cam: LatestMessage(self, CameraInfo, g(f"{cam}_info_topic"))
                              for cam in ("exterior", "wrist")}
        self.create_subscription(String, g("gripper_cmd_topic"), self._on_gripper_cmd, 10)
        self.create_subscription(String, "/vla_recorder/gripper_cmd", self._on_gripper_cmd, 10)   # the CLI's manual note
        self.create_subscription(String, g("transcript_topic"), self._on_transcript, 10)
        self.create_subscription(String, g("intent_topic"), self._on_intent, 10)
        self.create_subscription(String, g("cycle_done_topic"), self._on_cycle_done, 10)
        self.create_subscription(String, g("timing_topic"), self._on_timing, 20)
        self.create_subscription(String, g("selection_timing_topic"), self._on_selection_timing, 20)
        self.create_subscription(String, g("interactive_status_topic"), self._on_interactive, 10)
        self.create_subscription(String, "/vla_recorder/cmd", self._on_cmd, 10)
        self.status_pub = self.create_publisher(String, "/vla_recorder/status", LATCHED)

        self.recording = False; self.provisional = False
        self.task = ""; self.intent = ""; self.cycle_done = ""
        self.ep_dir: Path | None = None; self.ep_index = -1; self.ep_id = ""; self.in_trial_dir = False
        self.rows: dict[str, list] = {}; self.t0 = 0.0; self.t0_ros = 0.0; self.dropped = 0; self.drop_reasons: dict[str, int] = {}
        self.events: dict[str, list] = {}
        self.instructions: list[dict] = []
        self.mmhri_hint = ""
        self.mmhri_hint_next = ""   # "hint <trial folder>" from the GUI, consumed by the next episode
        self.last_error = ""
        self.last_saved: dict = {}
        self.create_timer(1.0 / self.fps, self._tick)
        self.create_timer(1.0, self._housekeeping)
        self.get_logger().info(f"VLA recorder ready. dataset={self.dataset_dir} fps={self.fps} auto_mode={self.auto_mode} "
                               f"exterior={self.img_source['exterior']} wrist={self.img_source['wrist']}")
        self._publish_status()

    # ------------------------------------------------------------- inputs
    def _poll_state(self) -> None:
        """The newest message of every fast state topic, into the same
        handlers the subscriptions used (see __init__), each with the time
        DDS RECEIVED it — the ages and staleness checks (joint_age,
        max_joint_age_s ...) measure arrival as before, not this read."""
        for reader, handler in self._fast:
            for msg, age in reader.take_streams(with_age=True):
                handler(msg, time.monotonic() - age)
        for cam, reader in list(self._info_readers.items()):
            msg = reader.take()
            if msg is not None:
                self._on_info(cam, msg)
                reader.destroy()
                del self._info_readers[cam]

    def _ros_now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_img(self, cam, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self.lock:
            self.img[cam] = (time.monotonic(), _stamp_s(msg.header), bgr)

    def _on_cimg(self, cam, msg):
        # Keep the JPEG bytes as they are (2026-09-09): decoding every frame
        # of both cameras cost this node 52 % of a core while idle, and the
        # episode re-encoded them anyway. The writer thread puts the bytes
        # straight on disk; freshness only needs the timestamps.
        with self.lock:
            self.img[cam] = (time.monotonic(), _stamp_s(msg.header), _Jpeg(bytes(msg.data)))

    def _on_info(self, cam, msg):
        if self.info[cam] is None:
            self.info[cam] = {"width": msg.width, "height": msg.height, "K": list(msg.k), "D": list(msg.d),
                              "distortion_model": msg.distortion_model, "frame_id": msg.header.frame_id}

    def _on_joints(self, msg: JointState, recv: float | None = None):
        names = list(msg.name)
        try:
            idx = [names.index(j) for j in ARM_JOINTS]
        except ValueError:
            idx = None
        with self.lock:
            now = time.monotonic() if recv is None else recv
            if idx is not None:
                pos = np.array([msg.position[i] for i in idx], dtype=np.float64)
                vel = np.array([msg.velocity[i] if i < len(msg.velocity) else np.nan for i in idx], dtype=np.float64)
                grip = self.joints[4] if self.joints else np.nan
                self.joints = [now, _stamp_s(msg.header), pos, vel, grip]
            if self.gripper_joint in names and self.joints is not None:
                gpos = float(msg.position[names.index(self.gripper_joint)])
                self.joints[4] = float(np.clip(gpos / self.gripper_open, 0.0, 1.0)) if self.gripper_open > 0 else gpos

    def _on_ctrl(self, msg, recv: float | None = None):
        pt = msg.reference if len(msg.reference.positions) else msg.desired
        if not len(pt.positions):
            return
        names = list(msg.joint_names)
        try:
            pos = np.array([pt.positions[names.index(j)] for j in ARM_JOINTS], dtype=np.float64)
        except ValueError:
            return
        with self.lock:
            self.joint_cmd = (time.monotonic() if recv is None else recv, _stamp_s(msg.header), pos)

    def _on_tcp(self, msg: PoseStamped, recv: float | None = None):
        p, o = msg.pose.position, msg.pose.orientation
        with self.lock:
            self.ur_tcp = (time.monotonic() if recv is None else recv, _stamp_s(msg.header), msg.header.frame_id,
                           np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]))

    def _on_io(self, msg, recv: float | None = None):
        # IOStates has no header: its stamp is the receipt, on the ROS clock
        now = time.monotonic()
        recv = now if recv is None else recv
        for d in msg.digital_in_states:
            if d.pin == self.grasp_pin:
                with self.lock:
                    self.io_grasp = (recv, self._ros_now() - (now - recv), bool(d.state))

    def _on_gripper_cmd(self, msg):
        """Adapter event JSON {"cmd","target","stamp","backend"} or the CLI's
        plain "open"/"close" (stamped on arrival)."""
        text = msg.data.strip()
        info = {}
        if text.startswith("{"):
            try:
                info = json.loads(text)
            except ValueError:
                return
            v = str(info.get("cmd", "")).lower()
        else:
            v = text.lower()
            info = {"cmd": v, "source": "manual"}
        val = 1.0 if v in ("open", "1", "1.0", "true") else 0.0 if v in ("close", "closed", "0", "0.0", "false") else None
        if val is None:
            return
        stamp = float(info.get("stamp") or self._ros_now())
        info = {**info, "cmd": "open" if val > 0.5 else "close", "value": val, "stamp": stamp}
        with self.lock:
            self.gripper_cmd = (time.monotonic(), stamp, val, info)
        self._event("gripper_cmd", info, stamp)

    # ---------------------------------------------------- trial events
    def _event(self, kind: str, data: dict, stamp: float | None = None) -> None:
        if not self.recording:
            return
        stamp = self._ros_now() if stamp is None else stamp
        self.events.setdefault(kind, []).append({"t": stamp - self.t0_ros, "stamp": stamp, **data})

    def _on_transcript(self, msg):
        self.transcript = msg.data
        self._event("transcript", {"text": msg.data})
        if self.auto_mode:
            if self.recording:
                if self.provisional:
                    self.task = msg.data  # newer command replaces a still-unconfirmed one
                return
            self.start_episode(msg.data, provisional=True)
        elif self.bound_to_trial and self.recording:
            # The task is the LAST command before the arm moved (confirmation
            # freezes it) — a question asked first never becomes the instruction.
            if self.provisional:
                self.task = msg.data
        elif self.task_from_transcript and self.recording and not self.task:
            self.task = msg.data

    def _on_intent(self, msg):
        try:
            payload = json.loads(msg.data)
            action = str(payload.get("action_intent") or "").strip().lower()
        except Exception:
            payload, action = {}, ""
        if not self.recording:
            return
        self.intent = action
        stamp = self._ros_now()
        # 2026-09-28: chat_node stamps every intent it publishes ("stamp",
        # wall s) — the time it happened, not the time this callback ran.
        try:
            st = float(payload.get("stamp")) if payload.get("stamp") else None
            if st is not None and abs(st - stamp) < 30.0:
                stamp = st
        except (TypeError, ValueError):
            pass
        self._event("intent", {"action_intent": action, "object_intent": payload.get("object_intent"),
                               "object_attributes": payload.get("object_attributes")}, stamp)
        if action in self.motion_intents:
            self._add_instruction(payload, action, stamp)
        if self.provisional:
            if action in self.motion_intents:
                self._confirm("intent", action)
            elif not self.bound_to_trial and action not in ANSWER_INTENTS:
                self.get_logger().info(f"episode {self.ep_id} discarded: intent '{action or 'none'}' is not a motion command")
                self.discard_episode()
            # bound: a question mid-trial is no verdict — the arm may still
            # move on a later command; the trial's own Stop decides.
        self._publish_status()

    def _on_selection_timing(self, msg):
        """2026-09-28 (Majd's trial 084: a pick chosen by POINTING sends no
        intent, so its approach, grasp and lift were labelled idle and the
        episode's task became "drop it on the right side"): the pipeline's
        selection-cycle record names the picked object and where the choice
        came from. When the instruction active at that moment is not already
        a pick of this object (a spoken pick, possibly clarified by
        pointing), a pick instruction starts at the selection."""
        if not self.recording:
            return
        try:
            rec = json.loads(msg.data)
        except Exception:
            return
        if not isinstance(rec, dict) or rec.get("kind") != "selection_cycle":
            return
        label = str(rec.get("label") or "").strip()
        source = str(rec.get("source") or "")
        try:
            stamp = float(rec.get("select") or rec.get("published") or self._ros_now())
        except (TypeError, ValueError):
            stamp = self._ros_now()
        self._event("selection", {"label": label, "source": source, "cycle_id": rec.get("cycle_id")}, stamp)
        if not label:
            return
        # 2026-09-28: the active instruction being a pick of this cycle is
        # enough — the pipeline's label is not the spoken name ('vitamin_tube'
        # for "vitamin c"; see relabel._covered_by_spoken_pick).
        before = [i for i in self.instructions if i["stamp"] <= stamp]
        if before and before[-1]["action_intent"] in ("pick", "pick_and_place") and not any(
                before[-1]["stamp"] < c["stamp"] < stamp for c in self.events.get("cycle_done", [])):
            return
        self._add_instruction({"action_intent": "pick", "object_intent": label,
                               "object_attributes": {"source": source or "selection"}}, "pick", stamp,
                              use_transcript=False)
        self.instructions.sort(key=lambda i: i["stamp"])
        for k, ins in enumerate(self.instructions):
            ins["idx"] = k
        if self.provisional:
            self._confirm("selection", "pick")   # a pointed pick moves the arm like a spoken one
        self._publish_status()

    def _add_instruction(self, payload: dict, action: str, stamp: float, use_transcript: bool = True) -> None:
        """One instruction per motion intent: the transcript that caused it
        (the last accepted one, at most 30 s before) and the canonical text."""
        transcripts = self.events.get("transcript", [])
        tr = transcripts[-1] if use_transcript and transcripts and stamp - transcripts[-1]["stamp"] < 30.0 else None
        text = canonical_instruction(action, payload.get("object_intent"), payload.get("object_attributes"),
                                     fallback=(tr or {}).get("text", ""))
        self.instructions.append({
            "idx": len(self.instructions), "t": stamp - self.t0_ros, "stamp": stamp,
            "action_intent": action, "object": payload.get("object_intent"),
            "attributes": payload.get("object_attributes"),
            "transcript": (tr or {}).get("text"), "transcript_stamp": (tr or {}).get("stamp"),
            "instruction": text,
        })
        self.get_logger().info(f"instruction {len(self.instructions) - 1}: '{text}' ({action})")

    def _on_cycle_done(self, msg):
        self.cycle_done = msg.data
        parts = msg.data.split("|")
        self._event("cycle_done", {"ok": parts[0].lower() == "done", "cycle_id": parts[1] if len(parts) > 1 else "",
                                   "reason": "|".join(parts[2:]) if len(parts) > 2 else ""})
        if self.auto_mode and self.recording:
            ok = msg.data.lower().startswith("done")
            self.stop_episode("success" if ok else "failure", note=f"cycle_done: {msg.data}")

    def _on_timing(self, msg):
        """Executor phase marks with absolute ROS stamps (published at cycle end)."""
        if not self.recording:
            return
        try:
            d = json.loads(msg.data)
            phases = d.get("phases") or []
            cid = d.get("cycle_id")
        except Exception:
            return
        for ph in phases:
            st = float(ph.get("t", 0.0))
            if st >= self.t0_ros - 1.0:
                self.events.setdefault("executor_phase", []).append(
                    {"t": st - self.t0_ros, "stamp": st, "name": str(ph.get("name")), "cycle_id": cid})

    def _on_interactive(self, msg):
        self._event("interactive_status", {"status": msg.data})

    def _on_cmd(self, msg):
        text = msg.data.strip()
        head, _, rest = text.partition(" "); head = head.lower()
        if head == "start":
            self.start_episode(rest.strip())
        elif head == "trial":
            self.start_trial_episode(rest.strip())
        elif head == "stop":
            arg = rest.strip().lower()
            if arg == "auto":
                # No human verdict: the executor's own word from THIS
                # episode, else unknown (never "true because nobody said no").
                cds = self.events.get("cycle_done", [])
                if cds:
                    outcome = "success" if cds[-1]["ok"] else "failure"
                    note = "outcome: executor cycle_done"
                else:
                    outcome, note = "unknown", "outcome: unknown (no human verdict, no executor verdict)"
            elif arg in ("fail", "failed", "false", "0"):
                outcome, note = "failure", "outcome: human"
            elif arg in ("unknown", "?"):
                outcome, note = "unknown", "outcome: human said unknown"
            else:
                outcome, note = "success", "outcome: human" if arg else "outcome: default (stop without verdict)"
            self.stop_episode(outcome, note=note)
        elif head == "discard":
            self.discard_episode()
        elif head == "task":
            self.task = rest.strip()
        elif head == "hint":
            self.mmhri_hint_next = rest.strip()
            self.get_logger().info(f"next episode goes to '{self.mmhri_hint_next}'"
                                   + ("" if self._resolve_trial_dir(self.mmhri_hint_next) else " (not a folder: dataset dir + note only)"))
        elif text:
            self.get_logger().warn(f"unknown command '{text}'")
        self._publish_status()

    def _housekeeping(self):
        if self.recording:
            el = time.monotonic() - self.t0
            if self.provisional and not self.bound_to_trial and el > self.intent_wait_s:
                self.get_logger().info(f"episode {self.ep_id} discarded: no motion intent within {self.intent_wait_s:.0f}s")
                self.discard_episode()
            elif el > self.max_episode_s:
                self.get_logger().warn(f"episode {self.ep_id} stopped: exceeded {self.max_episode_s:.0f}s")
                self.stop_episode("unknown", note="timeout")
        self._publish_status()

    def _newest_trial(self) -> str:
        try:
            # Day folders since 2026-09-18 (experiments_DD_MM_YYYY/trial_*), the old root layout kept.
            dirs = sorted((p for p in list(self.mmhri_dir.glob("*/trial_*")) + list(self.mmhri_dir.glob("*/*/trial_*"))
                           if p.is_dir()), key=lambda p: p.stat().st_mtime)
            return str(dirs[-1].relative_to(self.mmhri_dir)) if dirs else ""
        except Exception:
            return ""

    # ------------------------------------------------------------ episode
    def _next_index(self) -> int:
        existing = [int(p.name.split("_")[1]) for p in self.dataset_dir.glob("episode_*") if p.name.split("_")[1].isdigit()]
        return (max(existing) + 1) if existing else 0

    def _required(self) -> list[str]:
        req = ["joints", "tf_ee"]
        if self.require_ext:
            req.append("exterior")
        if self.require_wrist:
            req.append("wrist")
        return req

    def _stale_sources(self) -> list[str]:
        fresh = self._freshness()
        return [k for k in self._required() if not fresh.get(k)]

    def _resolve_trial_dir(self, hint: str) -> Path | None:
        """An absolute trial folder, or one relative to mmhri_dir, that exists."""
        if not hint:
            return None
        p = Path(hint).expanduser()
        if not p.is_absolute():
            p = self.mmhri_dir / hint
        return p if p.is_dir() else None

    def start_episode(self, task: str, provisional: bool = False):
        if self.recording:
            self.get_logger().warn("already recording; stop or discard first"); return
        if not task and not (self.task_from_transcript or provisional):
            self.get_logger().error("start needs a task text"); return
        if self.preflight:
            stale = self._stale_sources()
            if stale:
                self.last_error = "not started: stale " + ", ".join(stale)
                self.mmhri_hint_next = ""
                self.get_logger().error(f"episode refused — required streams not fresh: {', '.join(stale)}")
                self._publish_status()
                return
        self.last_error = ""
        hint = self.mmhri_hint_next
        self.mmhri_hint_next = ""
        trial_dir = self._resolve_trial_dir(hint)
        if trial_dir is not None:
            # The trial folder IS the episode folder (next to bag/, audio.wav ...).
            self.ep_dir, self.in_trial_dir = trial_dir, True
            self.ep_id = trial_dir.name
            m = re.match(r"trial_(\d+)", trial_dir.name)
            self.ep_index = int(m.group(1)) if m else -1
            self._remove_episode_files(trial_dir)     # a restarted trial overwrites its own episode
            self.mmhri_hint = str(trial_dir)
        else:
            self.ep_index = self._next_index()
            self.ep_dir = self.dataset_dir / f"episode_{self.ep_index:06d}"
            self.ep_id, self.in_trial_dir = self.ep_dir.name, False
            self.mmhri_hint = hint or self._newest_trial()
        for cam in ("exterior", "wrist"):
            (self.ep_dir / "images" / cam).mkdir(parents=True, exist_ok=True)
        self.rows = {k: [] for k in STEP_KEYS}
        self.events = {}; self.instructions = []
        self.task, self.intent, self.cycle_done = task, "", ""
        self.t0 = time.monotonic(); self.t0_ros = self._ros_now(); self.started_unix = time.time()
        self.dropped = 0; self.drop_reasons = {}; self.writer.failed = 0
        self.provisional = provisional; self.recording = True
        self.bound_to_trial = False; self.confirmed_by = ""; self.q_start = None
        self.task_fallback = ""
        self.get_logger().info(f"episode {self.ep_id} started{' (provisional)' if provisional else ''} in {self.ep_dir}: '{task}'")

    def start_trial_episode(self, fallback_task: str):
        """One MMHRI trial = one episode.

        Opens PROVISIONAL and is confirmed by the robot itself — an arm
        joint moving more than motion_confirm_rad (or a motion intent,
        whichever comes first). A trial that ends without motion ("what
        do you see", an absent object) is discarded, so asking about
        objects never becomes training data, and a trial driven by a
        GUI click (no speech, no intent) still counts because the arm
        moved. The task is the last command heard before the motion;
        the trial's own description is the fallback when nothing was said.
        """
        if self.auto_mode:
            self.get_logger().warn("'trial' ignored: auto_mode owns the episode lifecycle"); return
        self.start_episode("", provisional=True)
        if self.recording:
            self.bound_to_trial = True
            self.task_fallback = fallback_task

    def _confirm(self, how: str, detail: str = ""):
        """A provisional episode becomes real: something moved the robot."""
        self.provisional = False; self.confirmed_by = how
        self.get_logger().info(
            f"episode {self.ep_id} confirmed by {how}{' (' + detail + ')' if detail else ''}: "
            f"task '{self.task or self.task_fallback}'")

    def _remove_episode_files(self, d: Path) -> None:
        """Only the recorder's own files — a trial folder also holds bag/, audio, video."""
        shutil.rmtree(d / "images", ignore_errors=True)
        for name in ("meta.json", "steps.npz"):
            try:
                (d / name).unlink()
            except FileNotFoundError:
                pass

    def stop_episode(self, outcome: str = "success", note: str = ""):
        if not self.recording:
            return
        if outcome not in OUTCOMES:
            outcome = "unknown"
        if self.provisional:
            self.get_logger().info(f"episode {self.ep_id} ended unconfirmed (the arm never moved) — discarded")
            self.discard_episode(); return
        self.recording = False
        n = len(self.rows["t"])
        if n == 0:
            self.get_logger().warn("no frames captured; discarding")
            self._remove_episode_files(self.ep_dir)
            if not self.in_trial_dir:
                shutil.rmtree(self.ep_dir, ignore_errors=True)
            return
        self.writer.drain()
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in self.rows.items()}
        self._label_steps(arrays)
        quality = self._quality(arrays)
        np.savez_compressed(self.ep_dir / "steps.npz", **arrays)
        # The language instruction of record: the first motion instruction's
        # canonical text; the raw transcript / trial form stay as fallbacks.
        first = episode_task_instruction(self.instructions, self.events)
        task = first or self.task or self.task_fallback
        task_from = "intent" if first else "transcript" if self.task else "trial form"
        meta = {
            "format": "morty_vla_raw_v2", "episode_id": self.ep_id, "episode_index": self.ep_index,
            "episode_dir": str(self.ep_dir), "in_trial_dir": self.in_trial_dir,
            "trial_id": (re.match(r"trial_(\d+)", self.ep_id).group(1) if self.in_trial_dir and re.match(r"trial_(\d+)", self.ep_id) else None),
            "task": task, "task_from": task_from,
            "outcome": outcome, "success": {"success": True, "failure": False, "unknown": None}[outcome],
            "intent": self.intent, "cycle_done": self.cycle_done, "note": note,
            "mmhri_trial_hint": self.mmhri_hint, "bound_to_trial": self.bound_to_trial,
            "confirmed_by": self.confirmed_by,
            "spoken_task": self.task, "trial_form_task": self.task_fallback,
            "transcripts": [e["text"] for e in self.events.get("transcript", [])],
            "instructions": list(self.instructions),
            "events": self.events,
            "fps": self.fps, "num_steps": n, "duration_s": float(arrays["t"][-1] - arrays["t"][0]) if n > 1 else 0.0,
            "dropped_ticks": self.dropped, "drop_reasons": self.drop_reasons, "started_unix": self.started_unix,
            "started_ros": self.t0_ros,
            "robot_type": "ur3e_gripper_3d", "arm_joints": ARM_JOINTS, "gripper_joint": self.gripper_joint,
            "gripper_open_pos": self.gripper_open,
            "frames": {"base": self.base_frame, "ee": self.ee_frame, "tool": self.tool_frame,
                       "ur_tcp_source_frame": self.ur_tcp[2] if self.ur_tcp else None,
                       "ur_tcp_expressed_in": self.base_frame if self.base_from_ur is not None else (self.ur_tcp[2] if self.ur_tcp else None)},
            "cameras": {"exterior": self.info["exterior"], "wrist": self.info["wrist"]},
            "image_sources": self.img_source,
            "astra_static_tf_base_to_astra_link": self.astra_tf, "step_keys": STEP_KEYS,
            "segment_codes": {"idle": SEGMENT_IDLE, "task": SEGMENT_TASK, "reset": SEGMENT_RESET},
            "phase_names": self._phase_names,
            "quality": quality,
        }
        with open(self.ep_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        self.last_saved = {"episode_id": self.ep_id, "episode_dir": str(self.ep_dir), "outcome": outcome,
                           "num_steps": n, "quality_ok": quality["ok"], "issues": quality["issues"]}
        self.get_logger().info(f"episode {self.ep_id} saved: {n} steps, outcome={outcome}, dropped={self.dropped}, "
                               f"instructions={len(self.instructions)} {note}")
        for issue in quality["issues"]:
            self.get_logger().warn(f"quality: {issue}")

    def discard_episode(self):
        self.recording = False; self.provisional = False
        self.bound_to_trial = False; self.confirmed_by = ""; self.q_start = None
        if self.ep_dir and self.ep_dir.exists():
            self.writer.drain()
            self._remove_episode_files(self.ep_dir)
            if not self.in_trial_dir:
                shutil.rmtree(self.ep_dir, ignore_errors=True)
        self.ep_dir = None

    # ------------------------------------------------- labels and quality
    def _label_steps(self, a: dict) -> None:
        """Per-step instruction index, executor phase and segment, from the
        stamped events collected during the episode."""
        st = a["stamp_ros"]; n = len(st)
        instr = np.full(n, -1.0); phase = np.full(n, -1.0); seg = np.full(n, float(SEGMENT_IDLE))
        # instruction: active from its intent until the next cycle_done / next instruction
        cds = sorted(e["stamp"] for e in self.events.get("cycle_done", []))
        starts = sorted(e["stamp"] for e in self.events.get("executor_phase", []) if e.get("name") == "motion_start")
        picks = ("pick", "pick_and_place")
        for k, ins in enumerate(self.instructions):
            t_on = ins["stamp"]
            t_off = self.instructions[k + 1]["stamp"] if k + 1 < len(self.instructions) else np.inf
            # 2026-09-28 (trials 082, 071): a pick replaced by another pick
            # before the arm moved was refused or clarified ("which tube?" —
            # then "the rightmost" / "the yellow vitamin c"; only one pick
            # runs, whatever the names) — its steps stay idle instead of
            # teaching "pick up the vitamin tube" = stand still.
            nxt = self.instructions[k + 1] if k + 1 < len(self.instructions) else None
            ins.pop("superseded", None)
            if (nxt is not None and ins["action_intent"] in picks and nxt["action_intent"] in picks
                    and not any(t_on <= s < t_off for s in starts)):
                ins["superseded"] = True
                continue
            ends = [c for c in cds if c > t_on]
            if ends:
                t_off = min(t_off, ends[0] + 0.5)      # half a second after the verdict: the arm has stopped
            m = (st >= t_on) & (st < t_off)
            instr[m] = k; seg[m] = SEGMENT_TASK
        # reset: from a failure verdict or an explicit reset_home until the next instruction
        resets = [e["stamp"] for e in self.events.get("cycle_done", []) if not e["ok"]]
        resets += [e["stamp"] for e in self.events.get("interactive_status", []) if str(e["status"]).startswith("reset_home")]
        for r in sorted(resets):
            nxt = [i["stamp"] for i in self.instructions if i["stamp"] > r]
            t_off = nxt[0] if nxt else np.inf
            m = (st >= r) & (st < t_off)
            seg[m] = SEGMENT_RESET; instr[m] = -1
        # executor phase: the last phase mark at or before the step
        names: list[str] = []
        marks = sorted(self.events.get("executor_phase", []), key=lambda e: e["stamp"])
        for e in marks:
            if e["name"] not in names:
                names.append(e["name"])
        if marks:
            mstamps = np.array([e["stamp"] for e in marks]); mcode = np.array([names.index(e["name"]) for e in marks])
            idx = np.searchsorted(mstamps, st, side="right") - 1
            phase = np.where(idx >= 0, mcode[np.maximum(idx, 0)], -1.0).astype(np.float64)
        self._phase_names = names
        a["instruction_idx"] = instr; a["phase"] = phase; a["segment"] = seg

    def _quality(self, a: dict) -> dict:
        n = len(a["t"]); issues = []
        dt = np.diff(a["t"]) if n > 1 else np.zeros(0)
        gaps = [(int(i) + 1, float(d)) for i, d in enumerate(dt) if d > self.max_step_gap]
        if gaps:
            issues.append(f"{len(gaps)} step gap(s) > {self.max_step_gap:.1f} s (max {max(d for _, d in gaps):.2f} s)")
        if self.dropped:
            issues.append(f"{self.dropped} dropped tick(s): {self.drop_reasons}")
        img = {}
        for cam in ("exterior", "wrist"):
            age = a[f"img_age_{cam}"]
            fin = age[np.isfinite(age)]
            img[cam] = {"age_max_s": float(fin.max()) if fin.size else None,
                        "age_p95_s": float(np.percentile(fin, 95)) if fin.size else None,
                        "missing": int(np.isnan(age).sum())}
            if fin.size and fin.max() > self.max_img_age:
                issues.append(f"{cam} image up to {fin.max():.2f} s old")
        tcp_stale = int(np.isnan(a["tcp_pos"][:, 0]).sum())
        if tcp_stale:
            issues.append(f"ee pose stale/missing in {tcp_stale} step(s)")
        ur_missing = int(np.isnan(a["ur_tcp_pos"][:, 0]).sum())
        if ur_missing == n:
            issues.append("driver TCP pose never received")
        moved = float(np.nanmax(np.abs(a["joint_pos"] - a["joint_pos"][0]))) if n else 0.0
        tcp_fin = a["tcp_pos"][np.isfinite(a["tcp_pos"][:, 0])]
        if moved > 0.2 and tcp_fin.shape[0] > 10 and float(tcp_fin.std(axis=0).max()) < 1e-4:
            issues.append("ee pose frozen while the joints moved")
        # gripper: every command must be followed by the finger state within 2 s
        lags = []
        cmds = self.events.get("gripper_cmd", [])
        for e in cmds:
            after = a["stamp_ros"] >= e["stamp"]
            want_open = e["value"] > 0.5
            reached = after & ((a["gripper"] > 0.5) if want_open else (a["gripper"] < 0.5))
            k = np.argmax(reached) if reached.any() else -1
            lag = float(a["stamp_ros"][k] - e["stamp"]) if k >= 0 else None
            lags.append({"t": e["t"], "cmd": e["cmd"], "state_lag_s": lag})
            if lag is None:
                issues.append(f"gripper state never followed '{e['cmd']}' at t={e['t']:.1f}")
            elif lag > 2.0:
                issues.append(f"gripper state {lag:.1f} s behind '{e['cmd']}' at t={e['t']:.1f}")
        state_changes = int((np.abs(np.diff((a["gripper"] > 0.5).astype(int))) > 0).sum()) if n > 1 else 0
        if state_changes and not cmds:
            issues.append(f"gripper state changed {state_changes}x but no command event was received (adapter not publishing?)")
        if not self.instructions:
            issues.append("no instruction (no motion intent during the episode)")
        if self.writer.failed:
            issues.append(f"{self.writer.failed} image(s) failed to write")
        task_steps = int((a["segment"] == SEGMENT_TASK).sum())
        return {"ok": not issues, "issues": issues, "num_steps": n, "task_steps": task_steps,
                "idle_steps": int((a["segment"] == SEGMENT_IDLE).sum()), "reset_steps": int((a["segment"] == SEGMENT_RESET).sum()),
                "step_gaps": gaps, "dt_max_s": float(dt.max()) if dt.size else 0.0, "images": img,
                "joint_age_max_s": float(np.nanmax(a["joint_age"])) if n else None,
                "tcp_stale_steps": tcp_stale, "ur_tcp_missing_steps": ur_missing, "gripper_commands": lags,
                "gripper_state_changes": state_changes, "n_transcripts": len(self.events.get("transcript", [])),
                "n_intents": len(self.events.get("intent", [])), "n_instructions": len(self.instructions),
                "n_executor_phases": len(self.events.get("executor_phase", []))}

    # --------------------------------------------------------------- tick
    def _lookup(self, child):
        """(pos, quat, stamp) of child in base_frame, or None when unavailable
        or older than max_tf_age_s — a frozen /tf never counts as a pose."""
        try:
            tr = self.tf_buffer.lookup_transform(self.base_frame, child, Time())
        except TransformException:
            return None
        stamp = _stamp_s(tr.header)
        if self._ros_now() - stamp > self.max_tf_age:
            return None
        t, r = tr.transform.translation, tr.transform.rotation
        return (np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w]), stamp)

    def _ur_tcp_in_base(self, ur):
        """The driver's TCP pose (frame 'base') expressed in base_frame via the
        static transform, resolved once; unchanged if that transform is unknown."""
        _, stamp, frame, pos, quat = ur
        if frame and frame != self.base_frame:
            if self.base_from_ur is None:
                try:
                    tr = self.tf_buffer.lookup_transform(self.base_frame, frame, Time())
                    t, r = tr.transform.translation, tr.transform.rotation
                    self.base_from_ur = (np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w]))
                except TransformException:
                    return pos, quat, stamp
            bt, bq = self.base_from_ur
            return _quat_rot(bq, pos) + bt, _quat_mul(bq, quat), stamp
        return pos, quat, stamp

    def _tick(self):
        if not self.recording:
            return
        self._poll_state()
        now = time.monotonic(); ros_now = self._ros_now()
        with self.lock:
            ext, wri, joints, jcmd, grasp, gcmd, ur = (self.img["exterior"], self.img["wrist"], self.joints,
                                                       self.joint_cmd, self.io_grasp, self.gripper_cmd, self.ur_tcp)
        reasons = []
        if joints is None or now - joints[0] > self.max_joint_age:
            reasons.append("joints")
        if self.require_ext and (ext is None or now - ext[0] > self.max_img_age):
            reasons.append("exterior")
        if self.require_wrist and (wri is None or now - wri[0] > self.max_img_age):
            reasons.append("wrist")
        if reasons:
            self.dropped += 1
            for r in reasons:
                self.drop_reasons[r] = self.drop_reasons.get(r, 0) + 1
            if self.dropped in (1, 10, 50) or self.dropped % 200 == 0:
                self.get_logger().warn(f"dropped tick (missing: {', '.join(reasons)}) total={self.dropped}")
            return
        tcp = self._lookup(self.ee_frame); tool = self._lookup(self.tool_frame)
        if tcp is None:
            self.drop_reasons["tf_stale"] = self.drop_reasons.get("tf_stale", 0) + 1
            tcp = (np.full(3, np.nan), np.full(4, np.nan), np.nan)
        if tool is None:
            tool = (np.full(3, np.nan), np.full(4, np.nan), np.nan)
        if self.provisional and self.bound_to_trial:
            q = np.asarray(joints[2], dtype=np.float64)
            if self.q_start is None:
                self.q_start = q
            else:
                moved = float(np.max(np.abs(q - self.q_start)))
                if moved > self.motion_confirm_rad:
                    self._confirm("motion", f"{moved:.2f} rad")
        i = len(self.rows["t"])
        for cam, sample in (("exterior", ext), ("wrist", wri)):
            if sample is not None:
                self.writer.put(self.ep_dir / "images" / cam / f"{i:06d}.jpg", sample[2], self.jpeg_q)
        r = self.rows
        r["t"].append(ros_now - self.t0_ros); r["stamp_ros"].append(ros_now)
        r["joint_pos"].append(joints[2]); r["joint_vel"].append(joints[3]); r["joint_stamp"].append(joints[1])
        r["joint_cmd"].append(jcmd[2] if jcmd and now - jcmd[0] < 0.5 else np.full(6, np.nan))
        r["tcp_pos"].append(tcp[0]); r["tcp_quat"].append(tcp[1]); r["tcp_stamp"].append(tcp[2])
        r["tool0_pos"].append(tool[0]); r["tool0_quat"].append(tool[1])
        if ur is not None and now - ur[0] < self.max_tf_age:
            up, uq, us = self._ur_tcp_in_base(ur)
        else:
            up, uq, us = np.full(3, np.nan), np.full(4, np.nan), np.nan
        r["ur_tcp_pos"].append(up); r["ur_tcp_quat"].append(uq); r["ur_tcp_stamp"].append(us)
        r["gripper"].append(joints[4])
        r["gripper_cmd"].append(gcmd[2] if gcmd else np.nan)
        r["gripper_cmd_age"].append(ros_now - gcmd[1] if gcmd else np.nan)
        r["grasp_detected"].append(float(grasp[2]) if grasp else -1.0)
        r["img_stamp_exterior"].append(ext[1] if ext else np.nan); r["img_stamp_wrist"].append(wri[1] if wri else np.nan)
        r["img_age_exterior"].append(now - ext[0] if ext else np.nan); r["img_age_wrist"].append(now - wri[0] if wri else np.nan)
        r["joint_age"].append(now - joints[0])
        r["instruction_idx"].append(-1.0); r["phase"].append(-1.0); r["segment"].append(float(SEGMENT_IDLE))  # labelled at stop

    def _freshness(self) -> dict:
        self._poll_state()
        now = time.monotonic()
        with self.lock:
            fresh = {k: (v is not None and now - v[0] < self.max_img_age) for k, v in self.img.items()}
            fresh["joints"] = self.joints is not None and now - self.joints[0] < self.max_joint_age
            fresh["controller_ref"] = self.joint_cmd is not None and now - self.joint_cmd[0] < 1.0
            fresh["io"] = self.io_grasp is not None and now - self.io_grasp[0] < 2.0
            fresh["ur_tcp"] = self.ur_tcp is not None and now - self.ur_tcp[0] < self.max_tf_age
        fresh["tf_ee"] = self._lookup(self.ee_frame) is not None
        fresh["tf_tool"] = self._lookup(self.tool_frame) is not None
        return fresh

    def _publish_status(self):
        fresh = self._freshness()
        st = {"recording": self.recording, "provisional": self.provisional, "auto_mode": self.auto_mode,
              "bound": self.bound_to_trial, "confirmed_by": self.confirmed_by,
              "episode": self.ep_index if self.recording else None,
              "episode_id": self.ep_id if self.recording else "",
              "episode_dir": str(self.ep_dir) if self.recording and self.ep_dir else "",
              "task": self.task if self.recording else "",
              "intent": self.intent, "frames": len(self.rows["t"]) if self.recording and self.rows else 0,
              "instructions": len(self.instructions) if self.recording else 0,
              "dropped": self.dropped if self.recording else 0, "sources": fresh,
              "required": self._required(), "stale": [k for k in self._required() if not fresh.get(k)],
              "last_error": self.last_error, "last_saved": self.last_saved, "hint_next": self.mmhri_hint_next,
              "dataset_dir": str(self.dataset_dir), "episodes_on_disk": len(list(self.dataset_dir.glob("episode_*")))}
        self.status_pub.publish(String(data=json.dumps(st)))


def main(args=None):
    rclpy.init(args=args)
    node = EpisodeRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.recording:
            node.stop_episode("unknown", note="recorder shut down")
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__":
    main()
