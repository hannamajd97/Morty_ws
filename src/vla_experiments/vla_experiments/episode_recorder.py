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

import collections
import json
import os
import queue
import subprocess
import uuid
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

from .raw_dataset import (ARM_JOINTS, STEP_KEYS, SEGMENT_IDLE, SEGMENT_TASK, SEGMENT_RESET, alignment_verdict,
                          canonical_instruction)
from . import labels as _labels
from .recording_contract import (COMMAND_EVENT_SCHEMA, RAW_FORMAT, SEGMENT_NAMES, parse_stop_arg,
                                 validate_command_event)

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
OUTCOMES = ("success", "failure", "unknown", "interrupted")
MANIFEST = "recording_manifest.json"
ANNOTATION_KINDS = ("assist", "rearranged", "camera_blocked", "abort", "handover", "note")
CAMERA_CHECK_VIEWS = ("home", "pickup", "left", "right", "handover")


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


def _software_revision() -> dict:
    """git revision and dirty state of the workspace this recorder runs from."""
    src = Path(__file__).resolve().parents[2]
    out = {"repo": str(src)}
    try:
        out["commit"] = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                                       text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(src), "status", "--porcelain", "--untracked-files=no", "."],
                               capture_output=True, text=True, timeout=5).stdout.splitlines()
        out["dirty_files"] = len(dirty)
        out["dirty"] = bool(dirty)
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
    return out


def _atomic_json(path: Path, obj) -> None:
    """Write JSON through a temporary file + rename: a crash leaves either
    the old file or the new one, never half of one (2026-09-30)."""
    tmp = path.with_name(f".{path.name}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _atomic_npz(path: Path, arrays: dict) -> None:
    tmp = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class _ImageWriter:
    """Writes JPEGs on its own thread so a slow disk never stalls the tick
    (a stalled executor was one source of the 0.5-2 s step gaps seen in
    the 09-22 episodes: joint_states older than 0.2 s -> tick dropped)."""

    def __init__(self, logger) -> None:
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=400)
        self._log = logger
        self.failed = 0
        self.failed_paths: list = []
        self._t = threading.Thread(target=self._run, name="vla-image-writer", daemon=True)
        self._t.start()

    def put(self, path: Path, sample, jpeg_q: int) -> None:
        try:
            self._q.put_nowait((path, sample, jpeg_q))
        except queue.Full:
            self._fail(path, "writer queue full")

    def _fail(self, path: Path, why: str) -> None:
        self.failed += 1
        if len(self.failed_paths) < 50:
            self.failed_paths.append(f"{path.name}: {why}")

    def _run(self) -> None:
        while True:
            path, sample, jpeg_q = self._q.get()
            try:
                if isinstance(sample, _Jpeg):
                    path.write_bytes(sample.data)        # original JPEG, no decode/re-encode
                # 2026-09-30: cv2.imwrite reports most failures by returning
                # False, not by raising — both count now
                elif not cv2.imwrite(str(path), sample, [cv2.IMWRITE_JPEG_QUALITY, jpeg_q]):
                    self._fail(path, "cv2.imwrite returned False")
                    self._log.warn(f"image write failed: {path}")
            except Exception as e:  # noqa: BLE001
                self._fail(path, f"{type(e).__name__}: {e}")
                self._log.warn(f"image write failed: {e}")
            finally:
                self._q.task_done()

    def drain(self, timeout_s: float = 10.0) -> bool:
        """Wait until every queued write has FINISHED — including the one the
        worker is still writing (2026-09-30: it used to return as soon as the
        queue was empty, while the last image could still be half on disk).
        False = the timeout expired with writes outstanding."""
        t_end = time.monotonic() + timeout_s
        with self._q.all_tasks_done:
            while self._q.unfinished_tasks:
                left = t_end - time.monotonic()
                if left <= 0:
                    return False
                self._q.all_tasks_done.wait(left)
        return True


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
            # 2026-09-30 (recording contract v1): lifecycle / alignment inputs
            ("command_event_topic", "/robot/command_event"),
            ("task_state_topic", "/grasp_executor_lin_llm/task_state"),
            ("point_proposal_topic", "/grasp_pipeline/point_proposal"),
            ("open_question_topic", "/robot/open_question"),
            ("utterance_meta_topic", "/speech/utterance_meta"),
            ("max_camera_skew_s", 0.35),     # quality: exterior vs wrist capture stamps
            ("max_gripper_age_s", 2.0),      # quality: gripper joint samples older than this
            ("max_clock_offset_s", 0.5),     # a camera clock further off than this is not comparable
            ("save_timeout_s", 20.0),        # image writes must finish within this at stop
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
        self.max_skew, self.max_gripper_age = float(g("max_camera_skew_s")), float(g("max_gripper_age_s"))
        self.max_clock_offset, self.save_timeout = float(g("max_clock_offset_s")), float(g("save_timeout_s"))
        # 2026-09-30: what this recorder IS, stored with every run (schema,
        # parameters, software revision and dirty state)
        self.provenance = {"raw_format": RAW_FORMAT, "command_event_schema": COMMAND_EVENT_SCHEMA,
                           "parameters": {p.name: p.value for p in self._parameters.values()},
                           "software": _software_revision(), "node_started_unix": time.time()}

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
        self.gripper_recv = None      # monotonic receipt of the last gripper joint sample
        # camera receipt-minus-source offsets (clock comparability, per camera)
        self.cam_offsets = {"exterior": collections.deque(maxlen=60), "wrist": collections.deque(maxlen=60)}
        self.last_task_state = None
        self.router_question: dict = {}
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
        # 2026-09-30 (recording contract v1): command lifecycle, the executor's
        # task state as it changes, pointing questions, utterance ids
        self.create_subscription(String, g("command_event_topic"), self._on_command_event,
                                 QoSProfile(depth=100, reliability=QoSReliabilityPolicy.RELIABLE))
        self.create_subscription(String, g("task_state_topic"), self._on_task_state,
                                 QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE,
                                            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL))
        self.create_subscription(String, g("point_proposal_topic"), self._on_point_question, 10)
        self.create_subscription(String, g("open_question_topic"), self._on_router_question, LATCHED)
        self.create_subscription(String, g("utterance_meta_topic"), self._on_utterance_meta, 10)
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
        # 2026-09-30: every episode is one RUN with its own id and state:
        # idle -> recording (provisional) -> saving -> complete | incomplete
        # | error, or discarded. The GUI sends "run <id>" before "trial" and
        # waits until the status names that id (an acknowledgment, not a
        # cached status).
        self.run_id = ""; self.next_run_id = ""; self.run_state = "idle"
        self.last_run: dict = {}
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
        src = _stamp_s(msg.header)
        with self.lock:
            prev = self.img[cam]
            self.img[cam] = (time.monotonic(), src, bgr)
            if prev is None or prev[1] != src:          # a repeated frame says nothing about the clock
                self.cam_offsets[cam].append(time.time() - src)

    def _on_cimg(self, cam, msg):
        # Keep the JPEG bytes as they are (2026-09-09): decoding every frame
        # of both cameras cost this node 52 % of a core while idle, and the
        # episode re-encoded them anyway. The writer thread puts the bytes
        # straight on disk; freshness only needs the timestamps.
        src = _stamp_s(msg.header)
        with self.lock:
            prev = self.img[cam]
            self.img[cam] = (time.monotonic(), src, _Jpeg(bytes(msg.data)))
            if prev is None or prev[1] != src:          # receipt - capture of NEW frames (clock comparability)
                self.cam_offsets[cam].append(time.time() - src)

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
            if self.gripper_joint in names:
                self.gripper_recv = now        # 2026-09-30: gripper freshness, apart from the arm's
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
        # "stamp" = when it happened (the source's own time where the message
        # carries one), "recv" = when this recorder received it (system clock)
        self.events.setdefault(kind, []).append({"t": stamp - self.t0_ros, "stamp": stamp,
                                                 "recv": round(time.time(), 4), **data})

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
        # 2026-09-30: the whole payload (command_id, utterance_id, question …),
        # kept even when the router later rejects the command
        self._event("intent", {**{k: v for k, v in payload.items() if k != "stamp"},
                               "action_intent": action, "object_intent": payload.get("object_intent"),
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
        self._event("selection", {**{k: v for k, v in rec.items() if k not in ("kind", "stamp")},
                                  "label": label, "source": source, "cycle_id": rec.get("cycle_id")}, stamp)
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

    # ---------------------------------------- recording contract v1 inputs
    def _on_command_event(self, msg):
        """The router's command lifecycle (recording_contract.py). Invalid
        events are kept too, marked — the evidence is never dropped."""
        try:
            e = json.loads(msg.data)
        except ValueError:
            return
        if not isinstance(e, dict):
            return
        bad = validate_command_event(e)
        if bad:
            e["invalid"] = bad
        try:
            stamp = float(e.get("stamp"))
        except (TypeError, ValueError):
            stamp = None
        self._event("command", {k: v for k, v in e.items() if k != "stamp"}, stamp)

    def _on_task_state(self, msg):
        """Every NEW executor task-state sample, as it happens (the detailed
        executor marks only arrive at cycle end)."""
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        if not isinstance(d, dict):
            return
        key = (d.get("boot"), d.get("seq"))
        last = self.last_task_state
        self.last_task_state = d
        if last is not None and (last.get("boot"), last.get("seq")) == key:
            return
        try:
            stamp = float(d.get("stamp"))
        except (TypeError, ValueError):
            stamp = None
        self._event("task_state", {k: d.get(k) for k in ("boot", "seq", "cycle_id", "phase", "question",
                                                         "question_id", "holding", "object", "destination")}
                    | {"last_outcome": (d.get("last") or {}).get("outcome")}, stamp)

    def _on_point_question(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(d, dict):
            try:
                stamp = float(d.get("stamp"))
            except (TypeError, ValueError):
                stamp = None
            self._event("point_question", {k: v for k, v in d.items() if k != "stamp"}, stamp)

    def _on_router_question(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(d, dict):
            self.router_question = d
            self._event("router_question", d)

    def _on_utterance_meta(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        if isinstance(d, dict):
            try:
                stamp = float(d.get("published"))
            except (TypeError, ValueError):
                stamp = None
            self._event("utterance", d, stamp)

    def _annotate(self, rest: str) -> None:
        """annotate <kind> [start|end|at] [note …] — an operator intervention
        (manual assistance, a rearranged object, a blocked camera, an abort,
        an intended hand-over) with its time; assist intervals are excluded
        from autonomous-grasp training (labels.py)."""
        parts = rest.split()
        if not parts or parts[0] not in ANNOTATION_KINDS:
            self.get_logger().warn(f"annotate: kind must be one of {ANNOTATION_KINDS}")
            return
        kind = parts[0]
        edge = parts[1] if len(parts) > 1 and parts[1] in ("start", "end", "at") else "at"
        note = " ".join(parts[2:] if len(parts) > 1 and parts[1] in ("start", "end", "at") else parts[1:])
        if not self.recording:
            self.get_logger().warn(f"annotation '{kind} {edge}' ignored: no episode is recording")
            self.last_error = "annotation ignored: not recording"
            return
        self._event("annotation", {"kind": kind, "edge": edge, "note": note})
        self.get_logger().info(f"annotation: {kind} {edge} {note}")

    def _camera_check(self, view: str) -> None:
        """Save both cameras' current frames for one workspace view (home,
        pickup, left, right, handover) with their stamps — the operator moves
        the arm; this never moves anything. Fresh streams are not proof of
        useful framing: the saved pictures are."""
        view = view.strip().lower() or "view"
        base = (self.ep_dir / "camera_check" if self.recording and self.ep_dir else
                self.dataset_dir.parent / "camera_checks" / time.strftime("%Y%m%d"))
        base.mkdir(parents=True, exist_ok=True)
        tag = time.strftime("%H%M%S") + "_" + re.sub(r"[^a-z0-9_]+", "_", view)
        rec = {"view": view, "unix": time.time(), "files": {}, "run_id": self.run_id}
        now = time.monotonic()
        with self.lock:
            frames = {cam: self.img[cam] for cam in ("exterior", "wrist")}
        for cam, fr in frames.items():
            if fr is None:
                rec["files"][cam] = None
                continue
            path = base / f"{tag}_{cam}.jpg"
            if isinstance(fr[2], _Jpeg):
                path.write_bytes(fr[2].data)
            elif not cv2.imwrite(str(path), fr[2]):
                path = None
            rec["files"][cam] = {"path": str(path) if path else None, "source_stamp": fr[1],
                                 "receipt_age_s": round(now - fr[0], 3)}
        log = base / "camera_check.json"
        try:
            items = json.loads(log.read_text()) if log.exists() else []
        except ValueError:
            items = []
        items.append(rec)
        _atomic_json(log, items)
        self._event("camera_check", rec)
        self.get_logger().info(f"camera check '{view}' saved in {base}")

    def _on_cmd(self, msg):
        text = msg.data.strip()
        head, _, rest = text.partition(" "); head = head.lower()
        if head == "start":
            self.start_episode(rest.strip())
        elif head == "trial":
            self.start_trial_episode(rest.strip())
        elif head == "stop":
            # 2026-09-30: a bare or misspelled argument is NOT a success
            # (it used to save outcome=success) — recording_contract.parse_stop_arg
            outcome, note, known = parse_stop_arg(rest)
            if not known:
                self.get_logger().warn(f"stop: {note}")
            if outcome == "auto":
                # No human verdict: the executor's own word from THIS
                # episode, else unknown (never "true because nobody said no").
                # The executor's cycle verdict is execution evidence only —
                # it does not prove the right object or task.
                cds = self.events.get("cycle_done", [])
                if cds:
                    outcome = "success" if cds[-1]["ok"] else "failure"
                    note = "outcome: executor cycle_done (execution only, not verified task success)"
                else:
                    outcome, note = "unknown", "outcome: unknown (no human verdict, no executor verdict)"
            self.stop_episode(outcome, note=note)
        elif head == "run":
            # the id the GUI gives the next run; its status acknowledges it
            self.next_run_id = rest.strip()[:64]
        elif head == "annotate":
            self._annotate(rest.strip())
        elif head == "camera_check":
            self._camera_check(rest.strip())
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
            self.get_logger().warn("already recording; stop or discard first")
            self.last_error = "not started: already recording"
            self._refuse(self.last_error)
            return
        if not task and not (self.task_from_transcript or provisional):
            self.get_logger().error("start needs a task text"); return
        if self.preflight:
            stale = self._stale_sources()
            if stale:
                self.last_error = "not started: stale " + ", ".join(stale)
                self.mmhri_hint_next = ""
                self.get_logger().error(f"episode refused — required streams not fresh: {', '.join(stale)}")
                self._refuse(self.last_error)
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
            # 2026-09-30: a reused trial folder never loses an earlier episode —
            # it moves to vla_previous/<its run>/ (it used to be deleted)
            self._preserve_previous(trial_dir)
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
        self.run_id = self.next_run_id or uuid.uuid4().hex
        self.next_run_id = ""
        self.run_state = "recording"
        self.prev_src = {"exterior": None, "wrist": None}
        # an interrupted run is recognisable: the manifest says "recording"
        # until the save writes "complete" / "incomplete"
        _atomic_json(self.ep_dir / MANIFEST, {"schema": "morty.recording_manifest.v1", "run_id": self.run_id,
                                              "state": "recording", "episode_id": self.ep_id,
                                              "started_unix": self.started_unix})
        self.get_logger().info(f"episode {self.ep_id} (run {self.run_id[:8]}) started"
                               f"{' (provisional)' if provisional else ''} in {self.ep_dir}: '{task}'")

    def _refuse(self, reason: str) -> None:
        """A start that did not happen — reported under the run id the GUI
        asked for, so its wait for an acknowledgment ends with the reason."""
        if self.next_run_id:
            self.last_run = {"run_id": self.next_run_id, "state": "refused", "reason": reason}
        self.next_run_id = ""

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
        shutil.rmtree(d / "camera_check", ignore_errors=True)
        for name in ("meta.json", "steps.npz", MANIFEST):
            try:
                (d / name).unlink()
            except FileNotFoundError:
                pass

    def _preserve_previous(self, d: Path) -> None:
        """Move an existing episode of this folder to vla_previous/<run>/."""
        names = [x for x in ("images", "meta.json", "steps.npz", MANIFEST, "camera_check") if (d / x).exists()]
        if not names:
            return
        prev = "unknown"
        try:
            prev = json.loads((d / MANIFEST).read_text()).get("run_id") or prev
        except (OSError, ValueError):
            pass
        dst = d / "vla_previous" / f"{time.strftime('%Y%m%d_%H%M%S')}_{prev[:8]}"
        dst.mkdir(parents=True, exist_ok=True)
        for x in names:
            shutil.move(str(d / x), str(dst / x))
        self.get_logger().warn(f"{d.name} already held an episode — moved to {dst.relative_to(d)} (not deleted)")

    def stop_episode(self, outcome: str = "unknown", note: str = ""):
        if not self.recording:
            return                     # a repeated stop changes nothing
        if outcome not in OUTCOMES:
            outcome = "unknown"
        if self.provisional:
            self.get_logger().info(f"episode {self.ep_id} ended unconfirmed (the arm never moved) — discarded")
            self.last_run = {"run_id": self.run_id, "state": "discarded", "episode_id": self.ep_id,
                             "reason": "the arm never moved"}
            self.discard_episode(); return
        self.recording = False
        self.run_state = "saving"
        self._publish_status()
        n = len(self.rows["t"])
        if n == 0:
            self.get_logger().warn("no frames captured; discarding")
            self._remove_episode_files(self.ep_dir)
            if not self.in_trial_dir:
                shutil.rmtree(self.ep_dir, ignore_errors=True)
            self.run_state = "error"
            self.last_run = {"run_id": self.run_id, "state": "error", "episode_id": self.ep_id,
                             "reason": "no frames captured"}
            return
        drained = self.writer.drain(self.save_timeout)
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in self.rows.items()}
        lab = self._label_steps(arrays)
        # A command still executing at the press is an interruption: the
        # operator's boundary is kept, the outcome is not silently success.
        # an answer to a pause has no router verdict: the executor reaching
        # its next pause / idle completes it (labels: end_evidence)
        done_by_evidence = {i["command_id"] for i in lab["instructions"] if i.get("outcome") == "completed"}
        open_cmds = [c for c in lab.get("commands", []) if c.get("accepted") and not c.get("terminal")
                     and c["command_id"] not in done_by_evidence]
        if open_cmds:
            if note.startswith("outcome: human") and outcome == "success":
                lab["flags"].append(f"stopped while {len(open_cmds)} command(s) still executed — human said success")
            else:
                note = (note + "; " if note else "") + f"interrupted: {len(open_cmds)} command(s) still executing at stop"
                outcome = "interrupted" if outcome != "failure" else outcome
        quality = self._quality(arrays, lab, drained)
        _atomic_npz(self.ep_dir / "steps.npz", arrays)
        # The language instruction of record: the first motion instruction's
        # canonical text; the raw transcript / trial form stay as fallbacks.
        first = episode_task_instruction(self.instructions, self.events)
        task = first or self.task or self.task_fallback
        task_from = "intent" if first else "transcript" if self.task else "trial form"
        meta = {
            "format": RAW_FORMAT, "run_id": self.run_id, "label_source": lab["label_source"],
            "episode_id": self.ep_id, "episode_index": self.ep_index,
            "episode_dir": str(self.ep_dir), "in_trial_dir": self.in_trial_dir,
            "trial_id": (re.match(r"trial_(\d+)", self.ep_id).group(1) if self.in_trial_dir and re.match(r"trial_(\d+)", self.ep_id) else None),
            "task": task, "task_from": task_from,
            "outcome": outcome, "success": {"success": True, "failure": False}.get(outcome),
            "outcome_provenance": note,
            "intent": self.intent, "cycle_done": self.cycle_done, "note": note,
            "mmhri_trial_hint": self.mmhri_hint, "bound_to_trial": self.bound_to_trial,
            "confirmed_by": self.confirmed_by,
            "spoken_task": self.task, "trial_form_task": self.task_fallback,
            "transcripts": [e["text"] for e in self.events.get("transcript", [])],
            "instructions": list(self.instructions),
            # the intents as they ARRIVED (v2 labels), kept as raw evidence
            "intent_instructions": list(getattr(self, "_intent_instructions", [])),
            "commands": lab.get("commands", []),
            "open_commands_at_stop": [c["command_id"] for c in open_cmds],
            "interventions": self.events.get("annotation", []),
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
            "segment_codes": {name: code for code, name in SEGMENT_NAMES.items()},
            "phase_names": self._phase_names,
            "task_phase_names": getattr(self, "_task_phase_names", []),
            "quality": quality,
            "training_eligible": quality["training_eligible"],
            "provenance": {**self.provenance, "saved_unix": time.time(),
                           "cameras": {"exterior": self.info["exterior"], "wrist": self.info["wrist"],
                                       "astra_static_tf": self.astra_tf, "sources": self.img_source}},
        }
        _atomic_json(self.ep_dir / "meta.json", meta)
        # The manifest comes LAST, after the files it describes are complete.
        state = "complete" if quality["integrity"]["ok"] else "incomplete"
        _atomic_json(self.ep_dir / MANIFEST, {
            "schema": "morty.recording_manifest.v1", "run_id": self.run_id, "state": state,
            "episode_id": self.ep_id, "started_unix": self.started_unix, "saved_unix": time.time(),
            "num_steps": n, "files": {k: (self.ep_dir / k).stat().st_size for k in ("steps.npz", "meta.json")},
            "images": quality["integrity"]["images_on_disk"], "integrity_issues": quality["integrity"]["issues"],
            "outcome": outcome, "training_eligible": quality["training_eligible"]})
        self.run_state = state
        self.last_run = {"run_id": self.run_id, "state": state, "episode_id": self.ep_id,
                         "episode_dir": str(self.ep_dir), "outcome": outcome,
                         "integrity_ok": quality["integrity"]["ok"], "alignment_ok": quality["alignment"]["ok"],
                         "review_required": quality["semantic"]["review_required"],
                         "training_eligible": quality["training_eligible"]}
        self.last_saved = {"episode_id": self.ep_id, "episode_dir": str(self.ep_dir), "outcome": outcome,
                           "num_steps": n, "quality_ok": quality["ok"], "issues": quality["issues"],
                           "run_id": self.run_id, "state": state}
        self.get_logger().info(f"episode {self.ep_id} saved: {n} steps, outcome={outcome}, dropped={self.dropped}, "
                               f"instructions={len(self.instructions)} {note}")
        for issue in quality["issues"]:
            self.get_logger().warn(f"quality: {issue}")

    def discard_episode(self):
        if self.recording and self.last_run.get("run_id") != self.run_id:
            self.last_run = {"run_id": self.run_id, "state": "discarded", "episode_id": self.ep_id}
        self.run_state = "idle"
        self.recording = False; self.provisional = False
        self.bound_to_trial = False; self.confirmed_by = ""; self.q_start = None
        if self.ep_dir and self.ep_dir.exists():
            self.writer.drain()
            self._remove_episode_files(self.ep_dir)
            if not self.in_trial_dir:
                shutil.rmtree(self.ep_dir, ignore_errors=True)
        self.ep_dir = None

    # ------------------------------------------------- labels and quality
    def _label_steps(self, a: dict) -> dict:
        """Per-step instruction, segment and phase — labels.label_episode,
        the one implementation relabel / validate / the tests use. From the
        command lifecycle when the router recorded it (v3), else the v2
        intent-arrival rule, flagged as unverified."""
        import copy
        self._intent_instructions = copy.deepcopy(self.instructions)
        lab = _labels.label_episode(self.events, a, legacy_instructions=copy.deepcopy(self.instructions))
        self.instructions = lab["instructions"]
        a["instruction_idx"] = lab["instruction_idx"]; a["segment"] = lab["segment"]
        st = a["stamp_ros"]
        # executor phase marks (detailed; they arrive at cycle end) ...
        names: list[str] = []
        marks = sorted(self.events.get("executor_phase", []), key=lambda e: e["stamp"])
        for e in marks:
            if e["name"] not in names:
                names.append(e["name"])
        a["phase"] = np.full(len(st), -1.0)
        if marks:
            mstamps = np.array([e["stamp"] for e in marks]); mcode = np.array([names.index(e["name"]) for e in marks])
            idx = np.searchsorted(mstamps, st, side="right") - 1
            a["phase"] = np.where(idx >= 0, mcode[np.maximum(idx, 0)], -1.0).astype(np.float64)
        self._phase_names = names
        # ... and the task-state phase, recorded as it happened (v3)
        tnames: list[str] = []
        tss = sorted(self.events.get("task_state", []), key=lambda e: e["stamp"])
        for e in tss:
            if str(e.get("phase")) not in tnames:
                tnames.append(str(e.get("phase")))
        a["task_phase"] = np.full(len(st), -1.0)
        if tss:
            tst = np.array([e["stamp"] for e in tss]); tcode = np.array([tnames.index(str(e.get("phase"))) for e in tss])
            idx = np.searchsorted(tst, st, side="right") - 1
            a["task_phase"] = np.where(idx >= 0, tcode[np.maximum(idx, 0)], -1.0).astype(np.float64)
        self._task_phase_names = tnames
        return lab

    def _quality(self, a: dict, lab: dict, drained: bool) -> dict:
        """Four separate verdicts (2026-09-30) — readable arrays are not a
        correct demonstration:
          integrity  files, images, arrays, finite values (can the run be read?)
          alignment  timing: step gaps, dropped ticks, image receipt / capture
                     age, camera skew, repeated frames, clock comparability,
                     stale poses, gripper freshness and lag
          semantic   do the labels rest on execution evidence? (the command
                     lifecycle, no open / overlapping / unresolved commands,
                     no annotated assistance) — review_required otherwise
          outcome    in meta "outcome" (human, executor or unknown)
        "ok" (kept for older readers) = integrity AND alignment only."""
        n = len(a["t"])
        # ---- integrity
        integ = []
        if not drained:
            integ.append(f"image writes still pending after {self.save_timeout:.0f} s")
        if self.writer.failed:
            integ.append(f"{self.writer.failed} image(s) failed to write: {self.writer.failed_paths[:5]}")
        on_disk = {}
        for cam in ("exterior", "wrist"):
            files = sorted((self.ep_dir / "images" / cam).glob("*.jpg"))
            on_disk[cam] = len(files)
            required = self.require_ext if cam == "exterior" else self.require_wrist
            if required and len(files) != n:
                integ.append(f"{cam}: {len(files)} images on disk for {n} steps")
            for f in (files[:1] + files[-1:]) if files else []:
                if cv2.imread(str(f)) is None:
                    integ.append(f"{cam}: {f.name} does not decode")
        lengths = {k: len(v) for k, v in a.items()}
        if len(set(lengths.values())) > 1:
            integ.append(f"step arrays of different lengths: {lengths}")
        if not np.isfinite(a["joint_pos"]).all():
            integ.append("non-finite joint positions")
        if n > 1 and (np.diff(a["t"]) <= 0).any():
            integ.append("step times not increasing")
        # ---- alignment (the v2 checks, plus capture stamps and skew)
        align = []
        dt = np.diff(a["t"]) if n > 1 else np.zeros(0)
        gaps = [(int(i) + 1, float(d)) for i, d in enumerate(dt) if d > self.max_step_gap]
        if gaps:
            align.append(f"{len(gaps)} step gap(s) > {self.max_step_gap:.1f} s (max {max(d for _, d in gaps):.2f} s)")
        if self.dropped:
            align.append(f"{self.dropped} dropped tick(s): {self.drop_reasons}")
        img = {}
        for cam in ("exterior", "wrist"):
            age = a[f"img_age_{cam}"]; fin = age[np.isfinite(age)]
            cap = a.get(f"img_src_age_{cam}", np.full(n, np.nan)); cfin = cap[np.isfinite(cap)]
            rep = a.get(f"img_repeat_{cam}", np.zeros(n))
            offs = list(self.cam_offsets[cam])
            off = float(np.median(offs)) if offs else None
            comparable = off is not None and abs(off) <= self.max_clock_offset
            img[cam] = {"age_max_s": float(fin.max()) if fin.size else None,
                        "age_p95_s": float(np.percentile(fin, 95)) if fin.size else None,
                        "capture_age_p95_s": float(np.percentile(cfin, 95)) if cfin.size and comparable else None,
                        "capture_age_max_s": float(cfin.max()) if cfin.size and comparable else None,
                        "repeated_frames": int(np.nansum(rep == 1)),
                        "clock_offset_s": off, "clock_offset_spread_s": float(np.std(offs)) if offs else None,
                        "clock_comparable": comparable, "missing": int(np.isnan(age).sum())}
            if fin.size and fin.max() > self.max_img_age:
                align.append(f"{cam} image received up to {fin.max():.2f} s before its step")
            if comparable and cfin.size and cfin.max() > self.max_img_age:
                align.append(f"{cam} image CAPTURED up to {cfin.max():.2f} s before its step")
            if off is not None and not comparable:
                align.append(f"{cam} camera clock {off:+.2f} s from the laptop: capture stamps not comparable")
            # a 10 Hz camera sampled at 10 Hz repeats frames now and then;
            # only a frame reused for several steps in a row is a frozen stream
            run = best = 0
            for r_ in rep:
                run = run + 1 if r_ == 1 else 0
                best = max(best, run)
            img[cam]["longest_repeat_steps"] = int(best)
            if best >= 4:
                align.append(f"{cam}: the same frame reused for {best + 1} steps in a row "
                             f"({(best + 1) / self.fps:.1f} s — frozen stream?)")
        skew = a.get("camera_skew", np.full(n, np.nan)); sfin = np.abs(skew[np.isfinite(skew)])
        if sfin.size and sfin.max() > self.max_skew:
            align.append(f"cameras up to {sfin.max():.2f} s apart (> {self.max_skew:.2f} s) in {int((sfin > self.max_skew).sum())} step(s)")
        tcp_stale = int(np.isnan(a["tcp_pos"][:, 0]).sum())
        if tcp_stale:
            align.append(f"ee pose stale/missing in {tcp_stale} step(s)")
        ur_missing = int(np.isnan(a["ur_tcp_pos"][:, 0]).sum())
        if ur_missing == n:
            align.append("driver TCP pose never received")
        moved = float(np.nanmax(np.abs(a["joint_pos"] - a["joint_pos"][0]))) if n else 0.0
        tcp_fin = a["tcp_pos"][np.isfinite(a["tcp_pos"][:, 0])]
        if moved > 0.2 and tcp_fin.shape[0] > 10 and float(tcp_fin.std(axis=0).max()) < 1e-4:
            align.append("ee pose frozen while the joints moved")
        ga = a.get("gripper_age", np.full(n, np.nan))
        gfin = ga[np.isfinite(ga)]
        if not gfin.size:
            align.append("gripper state never received")
        elif gfin.max() > self.max_gripper_age:
            align.append(f"gripper state up to {gfin.max():.1f} s old (> {self.max_gripper_age:.1f} s)")
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
                align.append(f"gripper state never followed '{e['cmd']}' at t={e['t']:.1f}")
            elif lag > 2.0:
                align.append(f"gripper state {lag:.1f} s behind '{e['cmd']}' at t={e['t']:.1f}")
        state_changes = int((np.abs(np.diff((a["gripper"] > 0.5).astype(int))) > 0).sum()) if n > 1 else 0
        if state_changes and not cmds:
            align.append(f"gripper state changed {state_changes}x but no command event was received (adapter not publishing?)")
        # ---- semantic
        flags = _labels.review_flags({**lab, "instructions": self.instructions})
        notes = list(lab.get("notes", []))
        verified = lab["label_source"] == "command_lifecycle" and not flags
        seg = a["segment"].astype(int)
        # 2026-09-30: steps outside the timing tolerances are left out of
        # training (raw_dataset.alignment_verdict); only episode-wide timing
        # problems block the episode. "details" keeps every check's wording.
        limits = {"max_image_age_s": self.max_img_age, "max_camera_skew_s": self.max_skew,
                  "max_gripper_age_s": self.max_gripper_age}
        av = alignment_verdict(a, seg, {"issues": align, "images": img, "limits": limits})
        integrity_ok, alignment_ok = not integ, av["ok"]
        issues = integ + av["issues"] + [f"review: {f}" for f in flags]
        return {"ok": integrity_ok and alignment_ok, "issues": issues,
                "integrity": {"ok": integrity_ok, "issues": integ, "images_on_disk": on_disk, "drained": drained},
                "alignment": {"ok": alignment_ok, "issues": av["issues"], "notes": av["notes"],
                              "details": align, "limits": limits,
                              "misaligned_steps": av["misaligned_steps"],
                              "misaligned_task_steps": av["misaligned_task_steps"],
                              "images": img, "step_gaps": gaps,
                              "dt_max_s": float(dt.max()) if dt.size else 0.0,
                              "joint_age_max_s": float(np.nanmax(a["joint_age"])) if n else None,
                              "gripper_age_max_s": float(gfin.max()) if gfin.size else None,
                              "tcp_stale_steps": tcp_stale, "ur_tcp_missing_steps": ur_missing,
                              "gripper_commands": lags, "gripper_state_changes": state_changes},
                "semantic": {"label_source": lab["label_source"], "verified": verified,
                             "review_required": not verified, "flags": flags, "notes": notes},
                "training_eligible": integrity_ok and alignment_ok and verified,
                "num_steps": n, "task_steps": int((seg == SEGMENT_TASK).sum()),
                "segments": {name: int((seg == code).sum()) for code, name in SEGMENT_NAMES.items()},
                "idle_steps": int((seg == SEGMENT_IDLE).sum()), "reset_steps": int((seg == SEGMENT_RESET).sum()),
                "n_transcripts": len(self.events.get("transcript", [])),
                "n_intents": len(self.events.get("intent", [])), "n_commands": len(lab.get("commands", [])),
                "n_instructions": len(self.instructions),
                "n_executor_phases": len(self.events.get("executor_phase", [])),
                "n_task_states": len(self.events.get("task_state", []))}

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
        # 2026-09-30: capture age (tick minus the frame's own stamp; valid
        # when the camera clock is comparable — see _quality), the frame
        # reused from the previous step, the two cameras' capture skew and
        # the gripper's own freshness
        for cam, fr in (("exterior", ext), ("wrist", wri)):
            r[f"img_src_age_{cam}"].append(ros_now - fr[1] if fr else np.nan)
            r[f"img_repeat_{cam}"].append(float(fr is not None and fr[1] == self.prev_src[cam]))
            self.prev_src[cam] = fr[1] if fr else None
        r["camera_skew"].append(ext[1] - wri[1] if ext and wri else np.nan)
        r["gripper_age"].append(now - self.gripper_recv if self.gripper_recv is not None else np.nan)
        r["task_phase"].append(-1.0)   # labelled at stop
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
        st = {"run_id": self.run_id if self.recording or self.run_state == "saving" else "",
              "state": self.run_state, "last_run": self.last_run, "next_run_id": self.next_run_id,
              "recording": self.recording, "provisional": self.provisional, "auto_mode": self.auto_mode,
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
