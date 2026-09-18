"""Synchronised demonstration recorder for VLA fine-tuning.

Samples every source at a fixed rate (default 10 Hz) and writes the raw
episode format described in raw_dataset.py, into its own folder
(~/morty_ws/vla_experiments/datasets/<dataset_name>), never into experiments/.

Manual control over `/vla_recorder/cmd` (std_msgs/String) or the `vla_episode` CLI:
    start <task text> | stop | stop fail | stop auto | discard | task <text>
    trial <fallback task>   one MMHRI trial = one episode (see start_trial_episode)

Auto mode (parameter auto_mode:=true) records one episode per Morty trial without
touching the GUI or the MMHRI recorder, using the events the trial already emits:
    /speech/accepted_transcript  -> episode starts (task = the spoken command), provisional
    /robot/intent                -> JSON "action_intent"; if it is a motion intent the episode is
                                    confirmed, otherwise (question, chat) it is discarded
    /grasp_executor_lin_llm/cycle_done -> "done|<id>..." stops with success, "failed|..." with failure
    safety: no intent within intent_wait_s -> discard; longer than max_episode_s -> stop fail
The MMHRI trial folder that was newest at that time is noted in meta.json to link both records.

Sources: exterior image (Astra, compressed JPEG relayed from the shuttle), wrist image
(RealSense, compressed), /joint_states, controller reference, TF base->ee, gripper command,
UR tool digital input (grasp detected). Status is latched on /vla_recorder/status as JSON.
"""
from __future__ import annotations

import json
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
from grasp_pipeline.tf_listener_lite import DecimatedTransformListener

from .raw_dataset import ARM_JOINTS, STEP_KEYS

try:
    from ur_msgs.msg import IOStates
except Exception:
    IOStates = None

import cv2
from cv_bridge import CvBridge

LATCHED = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
MOTION_INTENTS = ["pick_and_place", "place", "drop_held", "drop_left", "drop_right", "go_home",
                  "carry_release", "release_home", "release", "open_gripper", "close_gripper"]


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
            ("gripper_cmd_topic", "/vla_recorder/gripper_cmd"),
            ("transcript_topic", "/speech/accepted_transcript"),
            ("task_from_transcript", False),
            ("auto_mode", False),
            ("motion_confirm_rad", 0.10),   # trial-bound: any arm joint moving this far confirms the episode
            ("intent_topic", "/robot/intent"),
            ("motion_intents", MOTION_INTENTS),
            ("cycle_done_topic", "/grasp_executor_lin_llm/cycle_done"),
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
        self.task_from_transcript, self.auto_mode = bool(g("task_from_transcript")), bool(g("auto_mode"))
        self.motion_confirm_rad = float(g("motion_confirm_rad"))
        self.motion_intents = set(str(x).lower() for x in g("motion_intents"))
        self.intent_wait_s, self.max_episode_s = float(g("intent_wait_s")), float(g("max_episode_s"))
        self.mmhri_dir = Path(g("mmhri_dir")).expanduser()
        self.astra_tf = {"xyz": list(g("astra_tf_xyz")), "rpy": list(g("astra_tf_rpy"))}

        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.img = {"exterior": None, "wrist": None}      # (wall_time, bgr ndarray)
        self.info = {"exterior": None, "wrist": None}
        self.img_source = {}
        self.joints = None; self.joint_cmd = None; self.tcp_topic = None; self.io_grasp = None; self.gripper_cmd = None
        # Trial-bound episodes (2026-09-03, driven by the GUI's MMHRI recorder):
        # provisional until the arm is SEEN to move; the last command heard
        # before that is the task; every utterance is kept for the record.
        self.bound_to_trial = False; self.confirmed_by = ""; self.q_start = None
        self.task_fallback = ""; self.transcripts = []
        self.transcript = None

        self.tf_buffer = Buffer()
        # 2026-09-18: the stock TransformListener deserialised the driver's
        # ~830 /tf messages per second on this node's executor and kept the
        # idle recorder at a full core (main thread 85 %) — one of the CPU
        # hogs that starved the UR driver's real-time loop into "Connection
        # to reverse interface dropped" bursts. The pipeline's decimating
        # listener (grasp_pipeline/tf_listener_lite.py) feeds the same
        # buffer at 50 Hz; the tick samples the TCP pose at `fps` (10 Hz).
        self.tf_listener = DecimatedTransformListener(self.tf_buffer, self, max_hz=50.0)

        for cam in ("exterior", "wrist"):
            ctopic = g(f"{cam}_compressed_topic"); rtopic = g(f"{cam}_image_topic")
            if ctopic:
                self.create_subscription(CompressedImage, ctopic, lambda m, c=cam: self._on_cimg(c, m), qos_profile_sensor_data)
                self.img_source[cam] = ctopic
            else:
                self.create_subscription(Image, rtopic, lambda m, c=cam: self._on_img(c, m), qos_profile_sensor_data)
                self.img_source[cam] = rtopic
            self.create_subscription(CameraInfo, g(f"{cam}_info_topic"), lambda m, c=cam: self._on_info(c, m), qos_profile_sensor_data)
        self.create_subscription(JointState, g("joint_states_topic"), self._on_joints, qos_profile_sensor_data)
        self.create_subscription(JointTrajectoryControllerState, g("controller_state_topic"), self._on_ctrl, 10)
        self.create_subscription(PoseStamped, g("tcp_pose_topic"), self._on_tcp, 10)
        if IOStates is not None:
            self.create_subscription(IOStates, g("io_states_topic"), self._on_io, 10)
        self.create_subscription(String, g("gripper_cmd_topic"), self._on_gripper_cmd, 10)
        self.create_subscription(String, g("transcript_topic"), self._on_transcript, 10)
        self.create_subscription(String, g("intent_topic"), self._on_intent, 10)
        self.create_subscription(String, g("cycle_done_topic"), self._on_cycle_done, 10)
        self.create_subscription(String, "/vla_recorder/cmd", self._on_cmd, 10)
        self.status_pub = self.create_publisher(String, "/vla_recorder/status", LATCHED)

        self.recording = False; self.provisional = False
        self.task = ""; self.intent = ""; self.cycle_done = ""
        self.ep_dir: Path | None = None; self.ep_index = -1
        self.rows: dict[str, list] = {}; self.t0 = 0.0; self.dropped = 0; self.drop_reasons: dict[str, int] = {}
        self.mmhri_hint = ""
        self.mmhri_hint_next = ""   # "hint <trial folder>" from the GUI, consumed by the next episode
        self.create_timer(1.0 / self.fps, self._tick)
        self.create_timer(1.0, self._housekeeping)
        self.get_logger().info(f"VLA recorder ready. dataset={self.dataset_dir} fps={self.fps} auto_mode={self.auto_mode} "
                               f"exterior={self.img_source['exterior']} wrist={self.img_source['wrist']}")
        self._publish_status()

    # ------------------------------------------------------------- inputs
    def _on_img(self, cam, msg):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self.lock:
            self.img[cam] = (time.time(), bgr)

    def _on_cimg(self, cam, msg):
        # Keep the JPEG bytes as they are (2026-09-09): decoding every frame
        # of both cameras cost this node 52 % of a core while idle, and the
        # episode re-encoded them anyway. The tick writes the bytes straight
        # to disk; freshness only needs the timestamp.
        with self.lock:
            self.img[cam] = (time.time(), _Jpeg(bytes(msg.data)))

    def _on_info(self, cam, msg):
        if self.info[cam] is None:
            self.info[cam] = {"width": msg.width, "height": msg.height, "K": list(msg.k), "D": list(msg.d),
                              "distortion_model": msg.distortion_model, "frame_id": msg.header.frame_id}

    def _on_joints(self, msg: JointState):
        names = list(msg.name)
        try:
            idx = [names.index(j) for j in ARM_JOINTS]
        except ValueError:
            idx = None
        with self.lock:
            now = time.time()
            if idx is not None:
                pos = np.array([msg.position[i] for i in idx], dtype=np.float64)
                vel = np.array([msg.velocity[i] if i < len(msg.velocity) else np.nan for i in idx], dtype=np.float64)
                grip = self.joints[3] if self.joints else np.nan
                self.joints = [now, pos, vel, grip]
            if self.gripper_joint in names and self.joints is not None:
                gpos = float(msg.position[names.index(self.gripper_joint)])
                self.joints[3] = float(np.clip(gpos / self.gripper_open, 0.0, 1.0)) if self.gripper_open > 0 else gpos

    def _on_ctrl(self, msg):
        pt = msg.reference if len(msg.reference.positions) else msg.desired
        if not len(pt.positions):
            return
        names = list(msg.joint_names)
        try:
            pos = np.array([pt.positions[names.index(j)] for j in ARM_JOINTS], dtype=np.float64)
        except ValueError:
            return
        with self.lock:
            self.joint_cmd = (time.time(), pos)

    def _on_tcp(self, msg):
        with self.lock:
            self.tcp_topic = (time.time(), msg)

    def _on_io(self, msg):
        for d in msg.digital_in_states:
            if d.pin == self.grasp_pin:
                with self.lock:
                    self.io_grasp = (time.time(), bool(d.state))

    def _on_gripper_cmd(self, msg):
        v = msg.data.strip().lower()
        val = 1.0 if v in ("open", "1", "1.0", "true") else 0.0 if v in ("close", "closed", "0", "0.0", "false") else None
        if val is not None:
            with self.lock:
                self.gripper_cmd = (time.time(), val)

    # ---------------------------------------------------- trial events
    def _on_transcript(self, msg):
        self.transcript = msg.data
        if self.auto_mode:
            if self.recording:
                if self.provisional:
                    self.task = msg.data  # newer command replaces a still-unconfirmed one
                return
            self.start_episode(msg.data, provisional=True)
        elif self.bound_to_trial and self.recording:
            # The task is the LAST command before the arm moved (confirmation
            # freezes it) — a question asked first never becomes the instruction.
            self.transcripts.append(msg.data)
            if self.provisional:
                self.task = msg.data
        elif self.task_from_transcript and self.recording and not self.task:
            self.task = msg.data

    def _on_intent(self, msg):
        try:
            action = str(json.loads(msg.data).get("action_intent") or "").strip().lower()
        except Exception:
            action = ""
        if not self.recording:
            return
        self.intent = action
        if self.provisional:
            if action in self.motion_intents:
                self._confirm("intent", action)
            elif not self.bound_to_trial:
                self.get_logger().info(f"episode {self.ep_index} discarded: intent '{action or 'none'}' is not a motion command")
                self.discard_episode()
            # bound: a question mid-trial is no verdict — the arm may still
            # move on a later command; the trial's own Stop decides.
        self._publish_status()

    def _on_cycle_done(self, msg):
        self.cycle_done = msg.data
        if self.auto_mode and self.recording:
            ok = msg.data.lower().startswith("done")
            self.stop_episode(success=ok, note=f"cycle_done: {msg.data}")

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
                # No human verdict: the executor's own word, else "it moved
                # and nobody said it failed".
                ok = self.cycle_done.lower().startswith("done") if self.cycle_done else True
                note = "success: auto (executor verdict)" if self.cycle_done else "success: auto (no executor verdict)"
            else:
                ok = arg not in ("fail", "failed", "false", "0")
                note = f"success: {'human' if arg else 'default'}"
            self.stop_episode(success=ok, note=note)
        elif head == "discard":
            self.discard_episode()
        elif head == "task":
            self.task = rest.strip()
        elif head == "hint":
            self.mmhri_hint_next = rest.strip()
        elif text:
            self.get_logger().warn(f"unknown command '{text}'")
        self._publish_status()

    def _housekeeping(self):
        if self.recording:
            el = time.time() - self.t0
            if self.provisional and not self.bound_to_trial and el > self.intent_wait_s:
                self.get_logger().info(f"episode {self.ep_index} discarded: no motion intent within {self.intent_wait_s:.0f}s")
                self.discard_episode()
            elif el > self.max_episode_s:
                self.get_logger().warn(f"episode {self.ep_index} stopped: exceeded {self.max_episode_s:.0f}s")
                self.stop_episode(success=False, note="timeout")
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

    def start_episode(self, task: str, provisional: bool = False):
        if self.recording:
            self.get_logger().warn("already recording; stop or discard first"); return
        if not task and not (self.task_from_transcript or provisional):
            self.get_logger().error("start needs a task text"); return
        self.ep_index = self._next_index()
        self.ep_dir = self.dataset_dir / f"episode_{self.ep_index:06d}"
        for cam in ("exterior", "wrist"):
            (self.ep_dir / "images" / cam).mkdir(parents=True, exist_ok=True)
        self.rows = {k: [] for k in STEP_KEYS}
        self.task, self.intent, self.cycle_done = task, "", ""
        self.t0 = time.time(); self.dropped = 0; self.drop_reasons = {}
        self.provisional = provisional; self.recording = True
        self.bound_to_trial = False; self.confirmed_by = ""; self.q_start = None
        self.task_fallback = ""; self.transcripts = []
        # 2026-09-18: the GUI names the trial folder ("hint ..." before
        # "trial ..."); the newest-folder guess stays as the fallback for
        # episodes started outside a trial. (The guess was empty all day:
        # recorder.yaml still pointed at the old experiments/mmhri root.)
        self.mmhri_hint = self.mmhri_hint_next or self._newest_trial()
        self.mmhri_hint_next = ""
        self.get_logger().info(f"episode {self.ep_index} started{' (provisional)' if provisional else ''}: '{task}'")

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
            f"episode {self.ep_index} confirmed by {how}{' (' + detail + ')' if detail else ''}: "
            f"task '{self.task or self.task_fallback}'")

    def stop_episode(self, success: bool = True, note: str = ""):
        if not self.recording:
            return
        if self.provisional:
            self.get_logger().info(f"episode {self.ep_index} ended unconfirmed (the arm never moved) — discarded")
            self.discard_episode(); return
        self.recording = False
        n = len(self.rows["t"])
        if n == 0:
            self.get_logger().warn("no frames captured; discarding")
            shutil.rmtree(self.ep_dir, ignore_errors=True); return
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in self.rows.items()}
        np.savez_compressed(self.ep_dir / "steps.npz", **arrays)
        meta = {
            "format": "morty_vla_raw_v1", "episode_index": self.ep_index, "task": self.task or self.task_fallback,
            "success": bool(success), "intent": self.intent, "cycle_done": self.cycle_done, "note": note,
            "mmhri_trial_hint": self.mmhri_hint, "bound_to_trial": self.bound_to_trial,
            "confirmed_by": self.confirmed_by, "task_from": "transcript" if self.task else "trial form",
            "transcripts": list(self.transcripts),
            "fps": self.fps, "num_steps": n, "duration_s": float(arrays["t"][-1] - arrays["t"][0]) if n > 1 else 0.0,
            "dropped_ticks": self.dropped, "drop_reasons": self.drop_reasons, "started_unix": self.t0,
            "robot_type": "ur3e_gripper_3d", "arm_joints": ARM_JOINTS, "gripper_joint": self.gripper_joint,
            "gripper_open_pos": self.gripper_open,
            "frames": {"base": self.base_frame, "ee": self.ee_frame, "tool": self.tool_frame},
            "cameras": {"exterior": self.info["exterior"], "wrist": self.info["wrist"]},
            "image_sources": self.img_source,
            "astra_static_tf_base_to_astra_link": self.astra_tf, "step_keys": STEP_KEYS,
        }
        with open(self.ep_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        self.get_logger().info(f"episode {self.ep_index} saved: {n} steps, success={success}, dropped={self.dropped} {note}")

    def discard_episode(self):
        self.recording = False; self.provisional = False
        self.bound_to_trial = False; self.confirmed_by = ""; self.q_start = None
        if self.ep_dir and self.ep_dir.exists():
            shutil.rmtree(self.ep_dir, ignore_errors=True)
        self.ep_dir = None

    # --------------------------------------------------------------- tick
    def _lookup(self, child):
        try:
            tr = self.tf_buffer.lookup_transform(self.base_frame, child, Time())
        except TransformException:
            return None
        t, r = tr.transform.translation, tr.transform.rotation
        return (np.array([t.x, t.y, t.z]), np.array([r.x, r.y, r.z, r.w]))

    def _tick(self):
        if not self.recording:
            return
        now = time.time()
        with self.lock:
            ext, wri, joints, jcmd, grasp, gcmd = self.img["exterior"], self.img["wrist"], self.joints, self.joint_cmd, self.io_grasp, self.gripper_cmd
        reasons = []
        if joints is None or now - joints[0] > self.max_joint_age:
            reasons.append("joints")
        if self.require_ext and (ext is None or now - ext[0] > self.max_img_age):
            reasons.append("exterior")
        if self.require_wrist and (wri is None or now - wri[0] > self.max_img_age):
            reasons.append("wrist")
        tcp = self._lookup(self.ee_frame); tool = self._lookup(self.tool_frame)
        if tcp is None:
            reasons.append("tf")
        if reasons:
            self.dropped += 1
            for r in reasons:
                self.drop_reasons[r] = self.drop_reasons.get(r, 0) + 1
            if self.dropped in (1, 10, 50) or self.dropped % 200 == 0:
                self.get_logger().warn(f"dropped tick (missing: {', '.join(reasons)}) total={self.dropped}")
            return
        if self.provisional and self.bound_to_trial:
            q = np.asarray(joints[1], dtype=np.float64)
            if self.q_start is None:
                self.q_start = q
            else:
                moved = float(np.max(np.abs(q - self.q_start)))
                if moved > self.motion_confirm_rad:
                    self._confirm("motion", f"{moved:.2f} rad")
        i = len(self.rows["t"])
        for cam, sample in (("exterior", ext), ("wrist", wri)):
            if sample is not None:
                path = self.ep_dir / "images" / cam / f"{i:06d}.jpg"
                if isinstance(sample[1], _Jpeg):
                    path.write_bytes(sample[1].data)        # original JPEG, no decode/re-encode
                else:
                    cv2.imwrite(str(path), sample[1], [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
        r = self.rows
        r["t"].append(now - self.t0); r["stamp_ros"].append(self.get_clock().now().nanoseconds * 1e-9)
        r["joint_pos"].append(joints[1]); r["joint_vel"].append(joints[2])
        r["joint_cmd"].append(jcmd[1] if jcmd and now - jcmd[0] < 0.5 else np.full(6, np.nan))
        r["tcp_pos"].append(tcp[0]); r["tcp_quat"].append(tcp[1])
        if tool is None:
            tool = (np.full(3, np.nan), np.full(4, np.nan))
        r["tool0_pos"].append(tool[0]); r["tool0_quat"].append(tool[1])
        r["gripper"].append(joints[3]); r["gripper_cmd"].append(gcmd[1] if gcmd else np.nan)
        r["grasp_detected"].append(float(grasp[1]) if grasp else -1.0)
        r["img_age_exterior"].append(now - ext[0] if ext else np.nan); r["img_age_wrist"].append(now - wri[0] if wri else np.nan)
        r["joint_age"].append(now - joints[0])

    def _publish_status(self):
        now = time.time()
        with self.lock:
            fresh = {k: (v is not None and now - v[0] < self.max_img_age) for k, v in self.img.items()}
            fresh["joints"] = self.joints is not None and now - self.joints[0] < self.max_joint_age
            fresh["controller_ref"] = self.joint_cmd is not None and now - self.joint_cmd[0] < 1.0
            fresh["io"] = self.io_grasp is not None and now - self.io_grasp[0] < 2.0
        fresh["tf_ee"] = self._lookup(self.ee_frame) is not None
        st = {"recording": self.recording, "provisional": self.provisional, "auto_mode": self.auto_mode,
              "bound": self.bound_to_trial, "confirmed_by": self.confirmed_by,
              "episode": self.ep_index if self.recording else None, "task": self.task if self.recording else "",
              "intent": self.intent, "frames": len(self.rows["t"]) if self.recording and self.rows else 0,
              "dropped": self.dropped if self.recording else 0, "sources": fresh,
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
            node.stop_episode(success=False, note="recorder shut down")
        node.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__":
    main()
