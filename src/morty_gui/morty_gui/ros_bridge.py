"""ROS side of the Morty GUI.

One rclpy node, spun in a background thread, that mirrors everything the
GUI shows into thread-safe holders the Qt timer polls. The GUI never
imports grasp_pipeline code — topics are the whole contract.

A deliberate trick: the filter-chain table, utility plot and log tail are
all fed by parsing /rosout messages from the pipeline node, so the GUI
gains debug views without any pipeline changes.
"""
from __future__ import annotations

import json
import re
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
import rclpy.time
import rclpy.duration
from action_msgs.srv import CancelGoal
from control_msgs.action import GripperCommand
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseArray, PoseStamped
from rcl_interfaces.msg import Log, Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import GetParameters, SetParameters
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, JointState, PointCloud2
from std_msgs.msg import Bool, Float32, Float32MultiArray, Int64, String
from std_srvs.srv import Trigger
try:
    from ur_dashboard_msgs.srv import IsProgramRunning, IsInRemoteControl
except ImportError:  # fake-hardware boxes without the UR driver messages
    IsProgramRunning = None
    IsInRemoteControl = None
try:
    from controller_manager_msgs.srv import ListControllers, SwitchController
except ImportError:
    ListControllers = SwitchController = None

from .config import VOICE_FILTER_PARAMS as _VOICE_FILTER_PARAMS

PIPELINE_NODE_NAME = "yolo_live_select_grasp"
# communication_llm/launch/voice_robot.launch.py starts the STT node as
#   Node(package="communication_llm", executable="speech_input_node",
#        name="speech_input_node")   # no namespace, no PushRosNamespace
# so its fully-qualified name is /speech_input_node.
SPEECH_NODE_NAME = "speech_input_node"

_RE_FILTER_START = re.compile(
    r"\[filter-chain\] START: (\d+)(?: grasps| score-ranked candidates)"
    r"(?: of (\d+) generated)?"
)
_RE_FILTER_STAGE = re.compile(r"\[filter-chain\] after ([a-z-]+): (\d+)")
_RE_UTILITY = re.compile(r"\[grasp-utility\].*U=([0-9.]+)")
_RE_BEST = re.compile(r"Best grasp score=([0-9.]+)")


def _cloud_to_numpy(msg: PointCloud2) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """PointCloud2 → (xyz float32 Nx3, rgb01 float32 Nx3 or None)."""
    offs: Dict[str, int] = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"} <= offs.keys():
        return None, None
    n = msg.width * msg.height
    if n == 0:
        return None, None
    # frombuffer on msg.data directly — bytes() made a full copy of every
    # cloud message (~MB each) before this even started parsing.
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(n, msg.point_step)

    def f32(name: str) -> np.ndarray:
        o = offs[name]
        return raw[:, o : o + 4].copy().view(np.float32).reshape(n)

    pts = np.stack([f32("x"), f32("y"), f32("z")], axis=1)
    good = np.isfinite(pts).all(axis=1)
    cols = None
    if "rgb" in offs:
        o = offs["rgb"]
        rgb_u32 = raw[:, o : o + 4].copy().view(np.uint32).reshape(n)
        r = ((rgb_u32 >> 16) & 0xFF).astype(np.float32) / 255.0
        g = ((rgb_u32 >> 8) & 0xFF).astype(np.float32) / 255.0
        b = (rgb_u32 & 0xFF).astype(np.float32) / 255.0
        cols = np.stack([r, g, b], axis=1)[good]
    return pts[good].astype(np.float32), cols


class MortyGuiBridge(Node):
    def __init__(self) -> None:
        super().__init__("morty_gui")
        self._lock = threading.Lock()
        self._bridge = CvBridge()

        # Holders the Qt side polls (all guarded by _lock).
        self.image_bgr: Optional[np.ndarray] = None
        self.image_dirty = False
        self.cloud_pts: Optional[np.ndarray] = None
        self.cloud_cols: Optional[np.ndarray] = None
        self.cloud_dirty = False
        # Per-frame captured cloud (sweep frames / single-view capture).
        self.frame_pts: Optional[np.ndarray] = None
        self.frame_cols: Optional[np.ndarray] = None
        self.frame_dirty = False
        self.campose: Optional[Tuple[float, float, float]] = None
        self.poses: Dict[str, List[Any]] = {"kept": [], "rejected": [], "best": [], "exec": []}
        self.poses_dirty = False
        self.log_lines: deque = deque(maxlen=800)
        self.log_dirty = False
        self.filter_rows: List[Tuple[str, int]] = []
        self.filter_dirty = False
        self.utility_hist: deque = deque(maxlen=200)
        self.utility_dirty = False
        self.status_text = ""
        # Health monitoring (Majd's LED request): /joint_states freshness
        # says SOME hardware interface is alive (fake or real); the UR
        # driver's io_and_status topic only exists with a real robot link,
        # and its value is the pendant's program running (Play) state.
        self._joint_state_t = 0.0
        self._program_running: Optional[bool] = None
        # 2026-09-22: pendant PLAYING while the driver's flag is False = the
        # External Control link is down; only a stop+Play restart cures it.
        self._program_disconnected: bool = False
        self._program_t = 0.0
        self._camera_t = 0.0
        # Astra (fixed cam on the shuttle PC, relayed over the network) and
        # the VLA episode recorder — separate LEDs since 2026-09-03 (Majd:
        # "which camera is alive?" was unanswerable with one LED).
        self._astra_t = 0.0
        # Astra VIEW (2026-09-08): decode the relayed JPEG only while the
        # GUI's picture panel is switched to it.
        self._astra_view = False
        self.astra_bgr: Optional[np.ndarray] = None
        self.astra_dirty = False
        self._vla_status: Optional[Dict[str, Any]] = None
        self.joint_names: List[str] = []
        self.joint_positions: List[float] = []
        self._fast_status = ""
        self._fast_status_t = 0.0
        self.cycle_id: Optional[int] = None
        self.best_score: Optional[float] = None
        self.cycle_done = False  # set when a cycle's best-score line arrives
        self._executor_done_pending = ""
        # Voice layer (communication_llm): chat events + state.
        self.chat_events: deque = deque(maxlen=100)  # (who, text)
        self.voice_chat_state = ""
        self.voice_dirty = False
        self._voice_seen_t = 0.0

        self.create_subscription(Image, "/yolo/debug_image", self._on_image, 2)
        # The bare camera in the same panel while the pipeline is not
        # running (Majd 2026-09-15): the RealSense publishes RELIABLE, so
        # the subscription must too (a BEST_EFFORT one never matches it).
        # Shown only while no debug image arrived in the last second, and
        # every 3rd frame (10 Hz is plenty for a preview).
        self._debug_img_t = 0.0
        self._raw_count = 0
        self.create_subscription(
            Image, "/camera/camera/color/image_raw", self._on_raw_image,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE))
        # Executor pause states ("paused" = holding, waiting for a drop side).
        self._exec_interactive_pending = ""
        self.exec_interactive_state = ""
        self.create_subscription(
            String, "/grasp_executor_lin_llm/interactive_status", self._on_exec_interactive, 10)
        self.create_subscription(PointCloud2, "/graspgen/final_cloud", self._on_cloud, 2)
        self.create_subscription(
            PointCloud2, "/yolo/selected_cloud_rgb", self._on_frame_cloud, 2
        )
        self.create_subscription(
            PoseStamped, "/graspgen/viz/camera_pose", self._on_campose, 2
        )
        self.create_subscription(PoseArray, "/graspgen/viz/kept_poses", lambda m: self._on_poses("kept", m), 2)
        self.create_subscription(PoseArray, "/graspgen/viz/rejected_poses", lambda m: self._on_poses("rejected", m), 2)
        self.create_subscription(PoseArray, "/graspgen/viz/best_pose_raw", lambda m: self._on_poses("best", m), 2)
        self.create_subscription(PoseArray, "/graspgen/viz/executing_pose", lambda m: self._on_poses("exec", m), 2)
        self.create_subscription(Log, "/rosout", self._on_rosout, 20)
        self.create_subscription(JointState, "/joint_states", self._on_joint_state, 5)
        self.create_subscription(
            CameraInfo, "/camera/camera/color/camera_info", self._on_camera_info, 5
        )
        # Astra liveness: the shuttle relays compressed_slow at 10 Hz
        # best-effort — sensor-data QoS or it never matches. Image is NOT
        # decoded, only timestamped.
        self.create_subscription(
            CompressedImage, "/astra/color/compressed_slow", self._on_astra,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo, "/astra/color/camera_info", self._on_astra,
            qos_profile_sensor_data,
        )
        # VLA recorder status is latched (TRANSIENT_LOCAL) and only
        # re-published on change — subscriber must match the durability.
        self.create_subscription(
            String, "/vla_recorder/status", self._on_vla_status,
            QoSProfile(depth=1,
                       reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        # (Scene window removed 2026-08-11 — Majd: RViz is the better tool
        # for that. Its TF listener + cloud feeds had also been the GUI
        # slowdown, see log #76.)

        # Voice layer (communication_llm --voice-only). accepted_transcript
        # is what Morty actually ACTED on (post echo/junk gates) — the raw
        # /speech/transcript is too noisy for display. Typed text from the
        # GUI chat box is published ONTO /speech/transcript, so the chat
        # node treats typing exactly like speech.
        self.create_subscription(
            String, "/speech/accepted_transcript", self._on_voice_heard, 5
        )
        self.create_subscription(
            String, "/robot/assistant_text", self._on_assistant_text, 5
        )
        # EVERYTHING the speak server actually says, including narrations
        # from intent_router/executor ("The grasp failed: ..."). Those never
        # pass through assistant_text, so the chat window missed them
        # (Majd, 2026-08-11). Deduped against Morty's replies in the GUI.
        self.create_subscription(
            String, "/speech/ai_said", self._on_ai_said, 5
        )
        # Why an utterance produced nothing (echo filter, language gate,
        # low confidence, …) — shown as ⚠ lines in the chat.
        self.create_subscription(
            String, "/speech/rejected", self._on_speech_rejected, 5
        )
        # Per-candidate probe status ("grasp 7/50: rejected — no IK").
        self._cand_status: deque = deque(maxlen=80)
        self.create_subscription(
            String, "/grasp_executor_lin_llm/candidate_status",
            self._on_cand_status, 20
        )
        # Per-object readiness from the pipeline's precompute (GraspGen done /
        # verified in the background / failed): the ready LED and the
        # readiness panel (2026-09-18).
        self._readiness = None
        self._readiness_t = 0.0
        self.create_subscription(String, "/graspgen/readiness", self._on_readiness, 10)
        # Timing records (2026-09-18): every selection cycle, background job
        # and executor cycle, kept ~600 deep so a trial can include the
        # background work done for its object BEFORE the trial started.
        self.timing_records: deque = deque(maxlen=600)
        self.create_subscription(String, "/graspgen/timing", self._on_timing, 20)
        self.create_subscription(String, "/grasp_executor_lin_llm/timing", self._on_timing, 20)
        # Voice system-control: chat_node publishes start_driver /
        # start_pipeline / start_all / stop_* intents; the GUI owns the
        # launches, so it executes them (Majd, 2026-08-12).
        self._system_intents: deque = deque(maxlen=8)
        self.create_subscription(
            String, "/robot/intent", self._on_robot_intent, 10
        )
        self.create_subscription(String, "/chat/state", self._on_chat_state, 5)
        # ── Scenario-trial recorder feeds (2026-08-20, Majd's MMHRI ──
        # experiments PDF): stage anchors for intent/yolo/graspgen/exec
        # timings. Passive — recorded only while a trial is active.
        self.trial_active = False
        self.trial_events: List[Tuple[float, str, str]] = []
        self.create_subscription(
            String, "/speech/accepted_transcript",
            lambda m: self.trial_mark("transcript", str(m.data)), 10,
        )
        # The pick cycle id comes from the pipeline's own topic, published
        # BEFORE its candidates and viewer poses. It used to be scraped from
        # /rosout text ("cycle_id=N" in any pipeline log line), which broke
        # the manual EXECUTE button in two ways (2026-09-11): the depth-20
        # /rosout queue drops lines under load, so the id could be learned
        # from the later "Waiting for executor completion (cycle_id=N)"
        # line — AFTER the best pose had armed the button — and lines such
        # as "Ignoring executor status for different cycle_id=N-1" flipped
        # it back to the previous cycle. Both disarmed a ready grasp.
        self.create_subscription(
            Int64, "/graspgen/selection_cycle_id", self._on_cycle_id,
            QoSProfile(depth=1,
                       reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_subscription(
            String, "/grasp_executor_lin_llm/cycle_done",
            self._on_llm_cycle_done, 10,
        )
        # Live flag: the long-running LLM executor is mid-cycle (probing
        # or moving). Lets the EXECUTE button act as a STOP for voice-
        # driven grasps too (Majd, 2026-08-20: "stop should directly
        # stop whatever it does — button AND voice").
        self.llm_cycle_active = False
        self._pub_resume = self.create_publisher(String, "/grasp_executor_lin_llm/resume", 10)
        self._pub_llm_abort = self.create_publisher(
            String, "/grasp_executor_lin_llm/abort", 10
        )
        # The pipeline's own abort (2026-09-09): Reset/E-Stop used to reach
        # the pipeline only through the voice stack's intent router; with
        # voice off, a pipeline stuck "waiting for executor completion"
        # ignored every Reset press.
        self._pub_pipeline_abort = self.create_publisher(
            String, "/grasp_pipeline/abort", 10
        )
        self._dets_t = 0.0
        self.create_subscription(
            String, "/yolo/all_detections", self._on_all_detections, 5,
        )
        # Typed text gets its own topic (2026-08-27): the chat node must
        # NOT acoustic-echo-check keyboard input — deliberate typed
        # retries were being eaten as "echo of my own voice".
        self._pub_typed_text = self.create_publisher(String, "/speech/transcript_typed", 5)
        # VLA episode recorder command line (see vla_link.py).
        self._pub_vla_cmd = self.create_publisher(String, "/vla_recorder/cmd", 10)
        # Mic strip (replaces the separate "Morty mic" window): level bar,
        # speech state, mute. /speech/muted is BOTH state and toggle —
        # speech_input_node acts on it directly.
        self.mic_level = 0.0
        self.speech_state = ""
        self.muted = False
        self.create_subscription(Float32, "/speech/level", self._on_mic_level, 10)
        # Scrolling ring of signed peak amplitudes, exactly like the
        # original mic window (260 bars; chunks EXTEND the ring, silence
        # keeps it scrolling so the trace never freezes).
        self._wave_ring: deque = deque([0.0] * 260, maxlen=260)
        self._wave_recv_t = 0.0
        self.create_subscription(
            Float32MultiArray, "/speech/waveform", self._on_waveform, 5
        )
        self.create_subscription(String, "/speech/state", self._on_speech_state, 10)
        # TTS busy flag from speak_node (2026-08-25): the MMHRI recorder
        # delays trial finalization until Morty finished talking.
        self.tts_active = False
        self.create_subscription(
            Bool, "/speech/tts_active",
            lambda m: setattr(self, "tts_active", bool(m.data)), 10)
        self.create_subscription(Bool, "/speech/muted", self._on_muted, 10)
        self._pub_mute = self.create_publisher(Bool, "/speech/muted", 10)

        # Fast observation replay bridge (pipeline's cached-trajectory node).
        self._pub_fast_replay = self.create_publisher(
            String, "/ur3e_fast_observation_replay/replay", 5
        )
        # MoveIt's trajectory execution manager halts on "stop" here — this
        # is what actually stops the arm; killing the commander alone would
        # leave the trajectory running in the controller.
        self._pub_traj_event = self.create_publisher(
            String, "/trajectory_execution_event", 1
        )
        self._pub_fast_cancel = self.create_publisher(
            String, "/ur3e_fast_observation_replay/cancel", 10
        )
        # Cancel the action goals at the controllers too. This catches
        # direct fast-replay trajectories that never passed through MoveIt.
        self._arm_cancel_clients = [
            self.create_client(
                CancelGoal,
                f"/{controller}/follow_joint_trajectory/_action/cancel_goal",
            )
            for controller in (
                "scaled_joint_trajectory_controller",
                "joint_trajectory_controller",
                "passthrough_trajectory_controller",
            )
        ]
        # Strongest software stop on the real UR: stop the pendant program.
        # This is deliberately additive to action cancellation, since the
        # dashboard service is absent when running fake hardware.
        self._dashboard_stop = self.create_client(Trigger, "/dashboard_client/stop")
        # Pendant Play from the GUI (2026-09-08, Majd: "bypass the play
        # button on the teach pendant"). The dashboard only honours it while
        # the pendant is in REMOTE control mode; in Local mode PolyScope
        # answers "Failed to execute: play" and we say so.
        self._dashboard_play = self.create_client(Trigger, "/dashboard_client/play")
        # A dashboard socket opened while the pendant was in Local mode stays
        # "not allowed" after the switch to Remote until it reconnects
        # ("...reconnect to port 29999", seen 2026-09-08): quit + connect
        # then retry once.
        self._dashboard_quit = self.create_client(Trigger, "/dashboard_client/quit")
        self._dashboard_connect = self.create_client(Trigger, "/dashboard_client/connect")
        # The pendant's "connection to the remote PC ... refused" popup
        # (2026-09-16): External Control keeps running while the driver is
        # away and complains; it is closed before every Play.
        self._dashboard_close_popup = self.create_client(Trigger, "/dashboard_client/close_popup")
        # Pendant Play state on demand (2026-09-03): robot_program_running
        # only publishes on CHANGE, so a GUI started after the driver can
        # sit at "unknown" forever. The dashboard answers immediately.
        self._dashboard_prog_cli = (
            self.create_client(IsProgramRunning, "/dashboard_client/program_running")
            if IsProgramRunning is not None else None)
        self._prog_query_t = 0.0
        # Auto-Play (2026-09-08, Majd: "why don't we start directly if we are
        # in remote mode"). While the pendant is in REMOTE control mode the
        # pendant's own Play is inactive anyway, so a stopped program can
        # only be restarted from here: the periodic check sends Play itself.
        # In LOCAL mode the dashboard refuses Play and the human presses the
        # pendant. Never right after OUR OWN stop (E-Stop), and never more
        # than once per 10 s so a robot that refuses (protective stop, brakes)
        # is not hammered.
        self._dashboard_remote_cli = (
            self.create_client(IsInRemoteControl, "/dashboard_client/is_in_remote_control")
            if IsInRemoteControl is not None else None)
        self._remote_mode: Optional[bool] = None
        self._remote_query_t = 0.0
        self._last_stop_cmd_t = 0.0
        self._last_autoplay_t = 0.0
        # Dashboard self-heal (2026-09-09): a pendant Local/Remote toggle makes
        # the robot's dashboard server go silent for a few seconds; the driver's
        # client then disconnects ("Did not receive answer ... Disconnecting")
        # and NEVER reconnects, so every later query fails with "Are you
        # connected to the Dashboard Server?". Any such reply triggers one
        # /dashboard_client/connect, at most every 10 s.
        self._last_dash_reconnect_t = 0.0
        # Trajectory-controller watchdog (2026-09-08): after any program
        # stop (E-Stop, pendant Stop, protective stop) controller_stopper
        # deactivates scaled_joint_trajectory_controller; if it is still
        # inactive while the program runs again, every execution dies with
        # "Can't accept new action goals. Controller is not running."
        # (= the executor's CONTROL_FAILED). Re-activate it ourselves.
        self._cm_list_cli = (self.create_client(ListControllers, "/controller_manager/list_controllers")
                             if ListControllers is not None else None)
        self._cm_switch_cli = (self.create_client(SwitchController, "/controller_manager/switch_controller")
                               if SwitchController is not None else None)
        self._ctrl_check_t = 0.0
        self._ctrl_reactivated = False
        self.create_subscription(
            String, "/ur3e_fast_observation_replay/status", self._on_fast_status, 5
        )
        # robot_program_running is LATCHED (TRANSIENT_LOCAL) and publishes
        # only on change — a default volatile subscription is QoS-incompatible
        # and receives NOTHING (found live 2026-08-07: LEDs grey while the
        # robot was demonstrably executing).
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # Manual-pick approval gate for the long-running LLM executor. The
        # latched value survives an executor restart; voice picks leave it
        # open, while a GUI click can close it until Execute is pressed.
        self._pub_execution_enabled = self.create_publisher(
            Bool, "/grasp_executor_lin_llm/execution_enabled", latched
        )
        self._pub_execution_enabled.publish(Bool(data=True))
        self.create_subscription(
            Bool, "/io_and_status_controller/robot_program_running",
            self._on_program, latched,
        )

        # Same click topic click_select_viewer uses, so selection keeps
        # working identically whichever window the user clicks in.
        self._pub_click = self.create_publisher(Point, "/yolo/debug_image_mouse_left", 10)
        self._pub_exec_active = self.create_publisher(Bool, "/morty/executor_active", 10)
        self._pub_intent = self.create_publisher(String, "/robot/intent", 10)

        # Gripper: same GripperCommand action MoveIt uses (adapter node);
        # position 0.5 = open, 0.0 = closed (gripper_command_adapter).
        self._gripper_client = ActionClient(
            self, GripperCommand, "/gripper_controller/gripper_cmd"
        )
        self._gripper_param_cli = self.create_client(
            SetParameters, "/gripper_controller/gripper_command_adapter/set_parameters"
        )
        # Read-back client on the SAME adapter node the setter targets
        # (namespace gripper_controller, node gripper_command_adapter), so
        # the Advanced "Apply" button can prove the holds really landed.
        self._gripper_param_get_cli = self.create_client(
            GetParameters, "/gripper_controller/gripper_command_adapter/get_parameters"
        )
        # Pipeline filter parameters (read + write on the live node).
        self._pipe_get_cli = self.create_client(
            GetParameters, f"/{PIPELINE_NODE_NAME}/get_parameters"
        )
        self._pipe_set_cli = self.create_client(
            SetParameters, f"/{PIPELINE_NODE_NAME}/set_parameters"
        )
        self._exec_set_cli = self.create_client(
            SetParameters, "/grasp_executor_lin_llm/set_parameters"
        )
        self._exec_get_cli = self.create_client(
            GetParameters, "/grasp_executor_lin_llm/get_parameters")
        # Voice-filter parameters on the live STT node (Advanced ▸ "Voice
        # filters" Apply button, 2026-08-28). Same read+write pair as the
        # pipeline above — the node only exists while the voice stack runs,
        # so every call is allowed to come back "unreachable".
        self._speech_get_cli = self.create_client(
            GetParameters, f"/{SPEECH_NODE_NAME}/get_parameters"
        )
        self._speech_set_cli = self.create_client(
            SetParameters, f"/{SPEECH_NODE_NAME}/set_parameters"
        )
        self.pipeline_params: Dict[str, Any] = {}
        self.pipeline_params_dirty = False

    # ── callbacks (executor thread) ──────────────────────────────────
    def _on_joint_state(self, msg: JointState) -> None:
        self._joint_state_t = time.monotonic()
        if msg.name:
            with self._lock:
                self.joint_names = list(msg.name)
                self.joint_positions = list(msg.position)

    def _on_camera_info(self, _msg: CameraInfo) -> None:
        self._camera_t = time.monotonic()

    def _on_astra(self, msg) -> None:
        self._astra_t = time.monotonic()
        # CompressedImage (has .format); the CameraInfo feed shares this callback.
        if self._astra_view and hasattr(msg, "format"):
            try:
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            except Exception:
                return
            if img is not None:
                with self._lock:
                    self.astra_bgr = img
                    self.astra_dirty = True

    def set_astra_view(self, on: bool) -> None:
        self._astra_view = bool(on)

    def _on_vla_status(self, msg: String) -> None:
        try:
            st = json.loads(msg.data)
        except (ValueError, TypeError):
            return
        with self._lock:
            self._vla_status = st

    def _on_fast_status(self, msg: String) -> None:
        with self._lock:
            self._fast_status = str(msg.data)
            self._fast_status_t = time.monotonic()

    def _on_program(self, msg: Bool) -> None:
        self._program_running = bool(msg.data)
        self._program_t = time.monotonic()

    def _on_image(self, msg: Image) -> None:
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self._lock:
            self._debug_img_t = time.monotonic()
            self.image_bgr = img
            self.image_dirty = True

    def _on_raw_image(self, msg: Image) -> None:
        self._raw_count += 1
        if self._raw_count % 3 or time.monotonic() - self._debug_img_t < 1.0:
            return
        try:
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception:
            return
        with self._lock:
            if time.monotonic() - self._debug_img_t < 1.0:
                return
            self.image_bgr = img
            self.image_dirty = True

    def _on_all_detections(self, msg: String) -> None:
        self._dets_t = time.monotonic()
        self.trial_mark("detections", str(msg.data)[:400])

    def executor_alive(self) -> bool:
        return self.count_publishers("/grasp_executor_lin_llm/cycle_done") > 0

    def _on_exec_interactive(self, msg: String) -> None:
        with self._lock:
            self._exec_interactive_pending = str(msg.data)
            self.exec_interactive_state = str(msg.data)

    def executor_waiting_with_object(self) -> bool:
        """True while the executor holds an object and waits for a drop side."""
        return self.exec_interactive_state == "paused"

    def _on_cloud(self, msg: PointCloud2) -> None:
        pts, cols = _cloud_to_numpy(msg)
        if pts is None:
            return
        with self._lock:
            self.cloud_pts, self.cloud_cols = pts, cols
            self.cloud_dirty = True

    def _on_frame_cloud(self, msg: PointCloud2) -> None:
        pts, cols = _cloud_to_numpy(msg)
        if pts is None:
            return
        with self._lock:
            self.frame_pts, self.frame_cols = pts, cols
            self.frame_dirty = True

    def _on_campose(self, msg: PoseStamped) -> None:
        with self._lock:
            p = msg.pose.position
            self.campose = (float(p.x), float(p.y), float(p.z))

    def _on_voice_heard(self, msg: String) -> None:
        with self._lock:
            self.chat_events.append(("You", str(msg.data)))
            self.voice_dirty = True
            self._voice_seen_t = time.monotonic()

    def trial_mark(self, source: str, text: str) -> None:
        """Record one event into the active scenario trial (no-op when
        no trial runs). Timestamps are time.monotonic() so durations are
        immune to wall-clock jumps. 'detections' is throttled to 1 Hz —
        it's a 5 Hz firehose that would drown the event log."""
        if not self.trial_active:
            return
        now = time.monotonic()
        with self._lock:
            if source == "detections":
                last = getattr(self, "_trial_last_det_t", 0.0)
                if now - last < 1.0:
                    return
                self._trial_last_det_t = now
            self.trial_events.append((now, source, text))

    # ── VLA episode recorder ─────────────────────────────────────────
    def vla_cmd(self, text: str) -> None:
        """One line of the recorder's /vla_recorder/cmd protocol."""
        self._pub_vla_cmd.publish(String(data=text))

    def vla_status(self) -> Optional[Dict[str, Any]]:
        """The recorder's latched status (episode index, recording flag…)."""
        with self._lock:
            return dict(self._vla_status) if self._vla_status else None

    def vla_node_up(self) -> bool:
        try:
            return "vla_episode_recorder" in self.get_node_names()
        except Exception:
            return False

    # ── pendant Play state ───────────────────────────────────────────
    def stop_program(self) -> bool:
        """Stop the pendant program through the dashboard (before the driver
        goes away, so External Control does not lose its PC mid-run and
        throw the popup). Auto-Play holds off for 10 s afterwards."""
        if not self._dashboard_stop.service_is_ready():
            return False
        try:
            self._dashboard_stop.call_async(Trigger.Request())
            self._last_stop_cmd_t = time.monotonic()
            return True
        except Exception:
            return False

    def close_popup(self) -> None:
        """Dismiss a pendant popup (fire-and-forget)."""
        try:
            if self._dashboard_close_popup.service_is_ready():
                self._dashboard_close_popup.call_async(Trigger.Request())
        except Exception:
            pass

    def play_program(self, _retry: bool = True) -> bool:
        """Start the pendant program through the UR dashboard (needs Remote
        control mode). Result lands in status_text; the Play LED follows
        from the latched robot_program_running topic within a second.
        If the dashboard socket predates the switch to Remote mode it is
        reconnected (quit, connect) and Play is retried once."""
        if not self._dashboard_play.service_is_ready():
            self.status_text = "Play: UR dashboard not available (driver down?)"
            return False
        self.status_text = "Play sent to the pendant..."
        self.close_popup()
        fut = self._dashboard_play.call_async(Trigger.Request())

        def _reconnect_then_retry():
            self.status_text = "Play: reconnecting the dashboard for Remote mode..."

            def _after_quit(_f):
                cf = self._dashboard_connect.call_async(Trigger.Request())
                cf.add_done_callback(lambda _g: self.play_program(_retry=False))
            qf = self._dashboard_quit.call_async(Trigger.Request())
            qf.add_done_callback(_after_quit)

        def _done(f):
            try:
                resp = f.result()
            except Exception as exc:
                self.status_text = f"Play failed: {exc}"[:120]
                return
            if resp is not None and resp.success:
                self.status_text = "Pendant program started (dashboard Play)"
                # Force a fresh dashboard query so the LED cannot stay stale.
                self._program_running = None
                self._prog_query_t = 0.0
                return
            msg = (resp.message if resp is not None else "no reply").strip()
            if self._dashboard_lost(resp):
                return
            if _retry and "29999" in msg:
                _reconnect_then_retry()
            elif "Remote Control" in msg:
                self.status_text = "Play refused: put the pendant in REMOTE control mode"
            elif msg == "Failed to execute: play":
                # "Already running" is the usual meaning — and then the
                # driver's own view (robot_program_running false) says the
                # program's External Control node lost its link to the
                # driver (2026-09-11 19:52: the robot reconnected, the old
                # socket was dropped, every trajectory failed with
                # CONTROL_FAILED while the dashboard reported PLAYING).
                # Only a program restart reconnects it: stop, then Play.
                if _retry and self.restart_program(
                        "Play refused: program already running but not linked "
                        "to the driver — restarting it (stop, then Play)"):
                    pass
                else:
                    self.status_text = ("Play refused by the robot: program already running, "
                                        "or robot not ready (safety / brakes)")
            else:
                self.status_text = f"Play refused: {msg}"[:120]
        fut.add_done_callback(_done)
        return True

    def restart_program(self, why: str = "") -> bool:
        """Stop the pendant program and Play it again 1.5 s later: the only
        way to reconnect an External Control node whose link to the driver
        dropped (pendant PLAYING, robot_program_running False, every goal
        accepted and then aborted on state tolerance). Needs Remote mode.
        Returns True when the stop was sent."""
        stop_cli = getattr(self, "_dashboard_stop", None)
        if stop_cli is None or not stop_cli.service_is_ready():
            return False
        self.status_text = why or "Restarting the pendant program (stop, then Play)"
        self._last_stop_cmd_t = 0.0
        sf = stop_cli.call_async(Trigger.Request())

        def _after_stop(_f):
            # the robot needs a moment to unload the program
            self._restart_timer = self.create_timer(1.5, _play_again)

        def _play_again():
            try:
                self._restart_timer.cancel()
            except Exception:
                pass
            self.play_program(_retry=False)
        sf.add_done_callback(_after_stop)
        return True

    def _dashboard_lost(self, resp) -> bool:
        """True when a dashboard reply reports a dead socket; reconnects once
        per 10 s as a side effect."""
        # The dashboard answers carry the text in `message` (Trigger) OR in
        # `answer` (IsProgramRunning, IsInRemoteControl); reading only
        # `message` meant the two 5-s polls never triggered the reconnect
        # and the driver log filled with "non-connected socket" (2026-09-16,
        # 316 times in one session). Any failed answer counts.
        if resp is None or getattr(resp, "success", True):
            return False
        msg = str(getattr(resp, "message", "") or getattr(resp, "answer", "") or "")
        if "Dashboard Server" not in msg and "socket" not in msg.lower() and msg:
            return False
        now = time.monotonic()
        if now - self._last_dash_reconnect_t >= 10.0 and self._dashboard_connect.service_is_ready():
            self._last_dash_reconnect_t = now
            self._dashboard_connect.call_async(Trigger.Request())
            self.status_text = "UR dashboard connection was lost — reconnecting"
        return True

    def query_remote_mode(self) -> None:
        """Ask the dashboard every 5 s whether the pendant is in Remote
        control mode (the only mode in which Play can be sent from here)."""
        now = time.monotonic()
        if (self._dashboard_remote_cli is None or now - self._joint_state_t > 1.5
                or now - self._remote_query_t < 5.0):
            return
        self._remote_query_t = now
        if not self._dashboard_remote_cli.wait_for_service(timeout_sec=0.2):
            return
        fut = self._dashboard_remote_cli.call_async(IsInRemoteControl.Request())

        def _done(f):
            try:
                resp = f.result()
            except Exception:
                return
            if self._dashboard_lost(resp):
                return
            if resp is not None and resp.success:
                self._remote_mode = bool(resp.remote_control)
        fut.add_done_callback(_done)

    def auto_play_tick(self, stop_latched: bool) -> bool:
        """Send Play by itself when the program is stopped and the pendant is
        in Remote mode. Returns True when a Play was sent this tick. Skipped
        while the GUI's software stop is latched, within 10 s of our own
        dashboard stop, and more often than every 10 s."""
        now = time.monotonic()
        if (stop_latched or self._program_running is not False or self._remote_mode is not True
                or now - self._joint_state_t > 1.5
                or now - self._last_stop_cmd_t < 10.0
                or now - self._last_autoplay_t < 10.0):
            return False
        self._last_autoplay_t = now
        if self._program_disconnected:
            return self.restart_program(
                "Program PLAYING on the pendant but not linked to the driver — "
                "restarting it automatically (stop, then Play)")
        sent = self.play_program()
        if sent:
            self.status_text = "Program stopped, pendant in Remote mode — Play sent automatically"
        return sent

    def query_program_running(self) -> None:
        """Ask the dashboard for the Play state every 5 s while the driver is
        alive; result lands in the same field the latched topic feeds.
        2026-09-08: no longer only while unknown — the latched topic misses
        a change that happens while the driver restarts, and the LED then
        stayed green on a stopped program (and the Play button refused)."""
        now = time.monotonic()
        if (self._dashboard_prog_cli is None
                or now - self._joint_state_t > 1.5 or now - self._prog_query_t < 5.0):
            return
        self._prog_query_t = now
        if not self._dashboard_prog_cli.wait_for_service(timeout_sec=0.2):
            return
        fut = self._dashboard_prog_cli.call_async(IsProgramRunning.Request())

        def _done(f):
            try:
                resp = f.result()
            except Exception:
                return
            if self._dashboard_lost(resp):
                return
            if resp is not None and resp.success:
                val = bool(resp.program_running)
                if (val and self._program_running is False and self.count_publishers(
                        "/io_and_status_controller/robot_program_running") > 0):
                    # 2026-09-22 (18:47): the link dropped, the pendant kept
                    # PLAYING, and this poll overwrote the driver's False every
                    # 5 s — the Play LED stayed green and auto_play_tick (which
                    # needs False) never restarted the program. The driver's
                    # flag wins; the poll only records the disagreement.
                    self._program_disconnected = True
                else:
                    self._program_disconnected = False
                    self._program_running = val
                self._program_t = time.monotonic()
        fut.add_done_callback(_done)

    def ensure_trajectory_controller_active(self) -> bool:
        """Every ~5 s while the driver is up and the program runs: list the
        controllers; if the trajectory controller is not active, activate
        it. Returns True once when a re-activation was sent (for the GUI
        status line)."""
        flag, self._ctrl_reactivated = self._ctrl_reactivated, False
        now = time.monotonic()
        if (self._cm_list_cli is None or self._cm_switch_cli is None
                or now - self._joint_state_t > 1.5 or self._program_running is not True
                or now - self._ctrl_check_t < 5.0):
            return flag
        self._ctrl_check_t = now
        if not self._cm_list_cli.wait_for_service(timeout_sec=0.2):
            return flag
        fut = self._cm_list_cli.call_async(ListControllers.Request())

        def _listed(f):
            try:
                ctrls = f.result().controller
            except Exception:
                return
            for c in ctrls:
                if c.name == "scaled_joint_trajectory_controller":
                    self._traj_ctrl_active = (c.state == "active")
                    self._traj_ctrl_t = time.monotonic()
                    if c.state != "active":
                        req = SwitchController.Request()
                        req.activate_controllers = ["scaled_joint_trajectory_controller"]
                        req.strictness = SwitchController.Request.BEST_EFFORT
                        req.timeout = rclpy.duration.Duration(seconds=2.0).to_msg()
                        self._cm_switch_cli.call_async(req)
                        self._ctrl_reactivated = True
                    return
        fut.add_done_callback(_listed)
        return flag

    def trial_start(self) -> None:
        with self._lock:
            self.trial_events = []
            self._trial_last_det_t = 0.0
            self.trial_active = True
            self.trial_start_wall = time.time()
            self.trial_mono_to_wall = time.time() - time.monotonic()
        self.trial_mark("trial", "start")

    def trial_stop(self) -> List[Tuple[float, str, str]]:
        self.trial_mark("trial", "stop")
        with self._lock:
            self.trial_active = False
            return list(self.trial_events)

    def _on_robot_intent(self, msg: String) -> None:
        self.trial_mark("intent", str(msg.data))
        try:
            action = str(json.loads(msg.data).get("action_intent") or "").lower()
        except Exception:
            return
        if action in ("start_driver", "stop_driver", "start_pipeline",
                      "stop_pipeline", "start_all", "stop_all"):
            with self._lock:
                self._system_intents.append(action)
        elif action == "stop":
            # Voice "stop" must also halt a BUTTON-launched executor —
            # that process doesn't listen on the voice abort topic, so
            # until now voice stop only froze the LLM executor while a
            # button execution kept moving (Majd, 2026-08-20: "they
            # aren't synchronised").
            with self._lock:
                self._voice_stop_pending = True

    def _on_ai_said(self, msg: String) -> None:
        self.trial_mark("morty_said", str(msg.data))
        with self._lock:
            self.chat_events.append(("Spoken", str(msg.data)))
            self.voice_dirty = True

    def _on_speech_rejected(self, msg: String) -> None:
        with self._lock:
            self.chat_events.append(("Rejected", str(msg.data)))
            self.voice_dirty = True

    def _on_cycle_id(self, msg: Int64) -> None:
        self.trial_mark("selection_cycle_id", str(msg.data))
        with self._lock:
            self.cycle_id = int(msg.data)

    def _on_llm_cycle_done(self, msg: String) -> None:
        self.trial_mark("cycle_done", str(msg.data))
        self.llm_cycle_active = False
        with self._lock:
            self._executor_done_pending = str(msg.data)

    def publish_resume(self, token: str) -> None:
        """Answer the executor's interactive pauses the way the voice
        router does (2026-09-11): after a grasp it waits for a destination
        ('drop_left_side' / 'drop_right_side' / 'home'), at the drop pose
        for a release ('release_home'). Without voice the GUI had no way
        to answer, so every pick ended in a Reset with the object still in
        the gripper."""
        self._pub_resume.publish(String(data=str(token)))
        self.get_logger().info(f"Resume token '{token}' sent to the executor.")

    def publish_llm_abort(self) -> None:
        """Abort the pick everywhere: the executor freezes in place, the
        pipeline ends its cycle at the next phase boundary (or leaves its
        wait for the executor)."""
        self._pub_llm_abort.publish(String(data="abort"))
        self._pub_pipeline_abort.publish(String(data="abort"))
        self.llm_cycle_active = False

    def _on_timing(self, msg) -> None:
        try:
            self.timing_records.append((time.time(), json.loads(msg.data)))
        except Exception:
            pass

    def timing_snapshot(self):
        with self._lock:
            return list(self.timing_records)

    def _on_readiness(self, msg) -> None:
        try:
            self._readiness = json.loads(msg.data)
            self._readiness_t = time.monotonic()
        except Exception:
            pass

    def _on_cand_status(self, msg: String) -> None:
        self.trial_mark("cand_status", str(msg.data))
        self.llm_cycle_active = True
        with self._lock:
            self._cand_status.append(str(msg.data))

    def _on_assistant_text(self, msg: String) -> None:
        with self._lock:
            self.chat_events.append(("Morty", str(msg.data)))
            self.voice_dirty = True
            self._voice_seen_t = time.monotonic()

    def _on_mic_level(self, msg: Float32) -> None:
        self.mic_level = float(msg.data)

    def _on_waveform(self, msg: Float32MultiArray) -> None:
        with self._lock:
            self._wave_ring.extend(float(v) for v in msg.data)
            self._wave_recv_t = time.monotonic()

    def _on_speech_state(self, msg: String) -> None:
        self.speech_state = str(msg.data)

    def _on_muted(self, msg: Bool) -> None:
        self.muted = bool(msg.data)

    def set_muted(self, muted: bool) -> None:
        self._pub_mute.publish(Bool(data=bool(muted)))

    def chat_alive(self) -> bool:
        return self.count_publishers("/chat/state") > 0

    def send_chat_text(self, text: str) -> None:
        """Typed input → same topic the STT feeds; chat node can't tell
        the difference."""
        self._pub_typed_text.publish(String(data=str(text)))
        self.get_logger().info(f"[chat] typed message sent ({len(text)} chars)")

    def _on_chat_state(self, msg: String) -> None:
        with self._lock:
            self.voice_chat_state = str(msg.data)
            self.voice_dirty = True
            self._voice_seen_t = time.monotonic()

    def _on_poses(self, kind: str, msg: PoseArray) -> None:
        if kind == "kept":
            self.trial_mark("grasp_poses", f"{len(msg.poses)} kept")
        with self._lock:
            self.poses[kind] = list(msg.poses)
            if kind == "kept":
                # New cycle: clear the previous executed-orange, mirroring
                # the Open3D viewer's persistence semantics.
                self.poses["exec"] = []
            self.poses_dirty = True

    def _on_rosout(self, msg: Log) -> None:
        if msg.name != PIPELINE_NODE_NAME:
            return
        text = str(msg.msg)
        with self._lock:
            self.log_lines.append(text)
            self.log_dirty = True
            m = _RE_FILTER_START.search(text)
            if m:
                entered = int(m.group(1))
                generated = int(m.group(2)) if m.group(2) else entered
                self.filter_rows = [("generated", generated)]
                if entered != generated:
                    self.filter_rows.append(("score top-k", entered))
                self.filter_dirty = True
            m = _RE_FILTER_STAGE.search(text)
            if m:
                self.filter_rows.append((m.group(1), int(m.group(2))))
                self.filter_dirty = True
            m = _RE_UTILITY.search(text)
            if m:
                self.utility_hist.append(float(m.group(1)))
                self.utility_dirty = True
            m = _RE_BEST.search(text)
            if m:
                self.best_score = float(m.group(1))
                self.cycle_done = True
            for key in ("Waiting for selection", "[multiframe]", "Selected index",
                        "Best grasp score", "[registration]", "Waiting for executor"):
                if key in text:
                    self.status_text = text[:120]
                    break

    # ── GUI-thread API ───────────────────────────────────────────────
    def publish_hover(self, x_px: float, y_px: float) -> None:
        """Mouse over the camera image: the pipeline computes that object first."""
        if not hasattr(self, "_pub_hover"):
            self._pub_hover = self.create_publisher(Point, "/yolo/hover_pixel", 10)
        p = Point()
        p.x, p.y, p.z = float(x_px), float(y_px), 0.0
        self._pub_hover.publish(p)

    def publish_click(self, x_px: float, y_px: float) -> None:
        p = Point()
        p.x, p.y, p.z = float(x_px), float(y_px), 0.0
        self._pub_click.publish(p)
        self.get_logger().info(f"GUI click at image px ({int(x_px)},{int(y_px)}) published.")

    def publish_executor_active(self, active: bool) -> None:
        self._pub_exec_active.publish(Bool(data=bool(active)))

    def set_grasp_execution_enabled(self, enabled: bool) -> None:
        """Open/close the persistent executor gate for manual GUI picks."""
        self._pub_execution_enabled.publish(Bool(data=bool(enabled)))
        self.get_logger().info(
            f"Grasp execution gate {'OPEN' if enabled else 'CLOSED'}."
        )

    @staticmethod
    def _pv_to_py(pv: ParameterValue):
        t = pv.type
        if t == ParameterType.PARAMETER_BOOL:
            return bool(pv.bool_value)
        if t == ParameterType.PARAMETER_INTEGER:
            return int(pv.integer_value)
        if t == ParameterType.PARAMETER_DOUBLE:
            return float(pv.double_value)
        if t == ParameterType.PARAMETER_STRING:
            return str(pv.string_value)
        # Arrays (2026-09-16): grasp2tcp_offset_m is a double array; the
        # read-back returned None for it, so Apply reported "no answer".
        if t == ParameterType.PARAMETER_DOUBLE_ARRAY:
            return [float(x) for x in pv.double_array_value]
        if t == ParameterType.PARAMETER_INTEGER_ARRAY:
            return [int(x) for x in pv.integer_array_value]
        if t == ParameterType.PARAMETER_BOOL_ARRAY:
            return [bool(x) for x in pv.bool_array_value]
        if t == ParameterType.PARAMETER_STRING_ARRAY:
            return [str(x) for x in pv.string_array_value]
        return None

    @staticmethod
    def _py_to_pv(v) -> ParameterValue:
        # bool must be checked before int (bool is an int subclass).
        if isinstance(v, bool):
            return ParameterValue(type=ParameterType.PARAMETER_BOOL, bool_value=v)
        # A list was sent as its str() before, which the node rejected as a
        # type mismatch: the GUI's -0.010 offset never reached the pipeline.
        if isinstance(v, (list, tuple)):
            if all(isinstance(x, bool) for x in v):
                return ParameterValue(type=ParameterType.PARAMETER_BOOL_ARRAY, bool_array_value=list(v))
            if all(isinstance(x, int) for x in v):
                return ParameterValue(type=ParameterType.PARAMETER_INTEGER_ARRAY, integer_array_value=list(v))
            if all(isinstance(x, (int, float)) for x in v):
                return ParameterValue(type=ParameterType.PARAMETER_DOUBLE_ARRAY,
                                      double_array_value=[float(x) for x in v])
            return ParameterValue(type=ParameterType.PARAMETER_STRING_ARRAY, string_array_value=[str(x) for x in v])
        if isinstance(v, int):
            return ParameterValue(type=ParameterType.PARAMETER_INTEGER, integer_value=v)
        if isinstance(v, float):
            return ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=v)
        return ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=str(v))

    def fetch_pipeline_params(self, names: List[str]) -> bool:
        """Async-read filter params off the live pipeline node."""
        if not self._pipe_get_cli.service_is_ready():
            if not self._pipe_get_cli.wait_for_service(timeout_sec=0.5):
                return False
        req = GetParameters.Request(names=list(names))
        fut = self._pipe_get_cli.call_async(req)

        def _done(f, names=list(names)):
            try:
                vals = f.result().values
            except Exception:
                return
            with self._lock:
                self.pipeline_params = {
                    n: self._pv_to_py(v) for n, v in zip(names, vals)
                }
                self.pipeline_params_dirty = True

        fut.add_done_callback(_done)
        return True

    def set_executor_params(self, mapping: Dict[str, Any]) -> bool:
        """Runtime parameters of the grasp executor (drop side, speed, ...);
        it re-reads them at the start of every pick (2026-09-11)."""
        if not self._exec_set_cli.service_is_ready():
            if not self._exec_set_cli.wait_for_service(timeout_sec=0.5):
                return False
        req = SetParameters.Request()
        for name, val in mapping.items():
            req.parameters.append(Parameter(name=name, value=self._py_to_pv(val)))
        self._exec_set_cli.call_async(req)
        self.get_logger().info(f"Executor params set: {list(mapping.keys())}")
        return True

    def set_pipeline_params(self, mapping: Dict[str, Any]) -> bool:
        if not self._pipe_set_cli.service_is_ready():
            if not self._pipe_set_cli.wait_for_service(timeout_sec=0.5):
                return False
        req = SetParameters.Request()
        for name, val in mapping.items():
            req.parameters.append(Parameter(name=name, value=self._py_to_pv(val)))
        self._pipe_set_cli.call_async(req)
        self.get_logger().info(f"Pipeline params set: {list(mapping.keys())}")
        return True

    def _get_params_sync(self, client, names, label: str) -> Optional[Dict[str, Any]]:
        """Bounded, synchronous GetParameters read for the Qt thread.

        Added 2026-08-28 for the Advanced "Apply" buttons: the operator
        wants proof the running node took the value, so the button reads
        the values back instead of echoing the widget.

        Never spins (the background thread owns the executor) — it only
        polls the future, so the worst case is ~1.0 s (0.5 s for the
        service + 0.5 s for the reply) and the UI cannot hang.
        Returns name->value, or None when the node did not answer.
        """
        if not client.service_is_ready():
            if not client.wait_for_service(timeout_sec=0.5):
                self.get_logger().warn(
                    f"{label} get_parameters not ready — read-back skipped")
                return None
        names = list(names)
        fut = client.call_async(GetParameters.Request(names=names))
        deadline = time.monotonic() + 0.5
        while not fut.done() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not fut.done():
            try:
                client.remove_pending_request(fut)
            except Exception:
                pass
            self.get_logger().warn(f"{label} read-back timed out")
            return None
        try:
            vals = fut.result().values
        except Exception as exc:
            self.get_logger().warn(f"{label} read-back failed: {exc}")
            return None
        out = {n: self._pv_to_py(v) for n, v in zip(names, vals)}
        self.get_logger().info(f"{label} read back: {out}")
        return out

    def get_pipeline_params(self, names: List[str]) -> Optional[Dict[str, Any]]:
        """Synchronous read-back off the live pipeline node (Apply button).

        Sibling of the async fetch_pipeline_params, which stores into
        self.pipeline_params for the Filters tab; this one RETURNS the
        values so the caller can show what the node actually holds.
        """
        return self._get_params_sync(self._pipe_get_cli, names, "Pipeline")

    def get_executor_params(self, names: List[str]) -> Optional[Dict[str, Any]]:
        """Synchronous read-back off the grasp executor (Apply button)."""
        return self._get_params_sync(self._exec_get_cli, names, "Executor")

    def get_gripper_holds(self) -> Optional[Dict[str, Any]]:
        """Synchronous read-back of the adapter's pulse holds (Apply button)."""
        return self._get_params_sync(
            self._gripper_param_get_cli,
            ["pulse_hold_s_open", "pulse_hold_s_close"],
            "Gripper adapter",
        )

    # ── voice filters on the live STT node (2026-08-28) ──────────────
    # These four used to be launch-argument-only, so retuning the mic gate
    # meant restarting the voice stack. speech_input_node now has a
    # set-parameters callback that pushes them onto the running listener.
    # The names live in config.py (ROS-free) so the settings-profile code
    # can share them without importing rclpy — see config.VOICE_FILTER_PARAMS.
    VOICE_FILTER_PARAMS = list(_VOICE_FILTER_PARAMS)

    def set_voice_filters(self, mapping: Dict[str, Any]) -> bool:
        """Push voice-filter params onto the live speech node (async).

        Returns False when the node's param service is unreachable — i.e.
        the voice stack is not running — so the GUI can say so instead of
        pretending it applied. Mirrors set_pipeline_params exactly.
        """
        if not self._speech_set_cli.service_is_ready():
            if not self._speech_set_cli.wait_for_service(timeout_sec=0.5):
                self.get_logger().warn(
                    "speech_input_node param service not ready — voice "
                    "filters NOT applied (is the voice stack running?)"
                )
                return False
        req = SetParameters.Request()
        for name, val in mapping.items():
            req.parameters.append(Parameter(name=name, value=self._py_to_pv(val)))
        self._speech_set_cli.call_async(req)
        self.get_logger().info(f"Voice filters set: {list(mapping.keys())}")
        return True

    def get_voice_filters(self) -> Optional[Dict[str, Any]]:
        """Synchronous read-back of the four filters off the live STT node."""
        return self._get_params_sync(
            self._speech_get_cli, self.VOICE_FILTER_PARAMS, "Speech input"
        )

    def stop_motion(self) -> None:
        self._pub_traj_event.publish(String(data="stop"))
        # Fast replay bypasses MoveIt, so trajectory_execution_event alone
        # cannot stop it. Cancel both the replay node and the controller goal.
        self._pub_fast_cancel.publish(String(data=""))
        self._cancel_arm_controller_goals()
        self.get_logger().warn(
            "STOP sent to MoveIt, fast replay, and arm trajectory controllers."
        )

    def _cancel_arm_controller_goals(self) -> None:
        # A zero goal id + zero timestamp is the action protocol's
        # cancel-all request.
        for client in self._arm_cancel_clients:
            if client.service_is_ready():
                try:
                    client.call_async(CancelGoal.Request())
                except Exception as exc:
                    self.get_logger().error(f"Controller cancel failed: {exc}")

    def reset_cycle(self) -> None:
        """Soft per-cycle reset (Majd, 2026-08-25): abort the current
        pick cycle and clear every queued/latched selection so the
        detection loop starts fresh — WITHOUT stopping any process, the
        motion controller, or the pendant program (that is the E-Stop's
        job). After this the pipeline returns to "Waiting for selection"
        as if no detection had been chosen yet.
        """
        if self.executor_waiting_with_object():
            # Holding an object and waiting for Left / Right (2026-09-15):
            # "abort" would freeze the arm in the air (the voice rule for
            # "stop"); reset means go home with the object and be ready.
            # Nothing is moving, so no motion stop is sent.
            self.publish_resume("reset_home")
            self.exec_interactive_state = ""
            self.llm_cycle_active = False
        else:
            self.publish_llm_abort()
            # ALSO stop the arm (2026-08-25, Majd: "he can't reset the pick"
            # — the reset felt dead while a trajectory was executing, because
            # the abort only takes effect at the next phase boundary).
            self.stop_motion()
        try:
            self.set_pipeline_params({"auto_select_label": ""})
        except Exception as exc:
            self.get_logger().warn(f"reset: auto_select clear failed: {exc}")
        try:
            self._pub_intent.publish(String(data='{"action_intent":"stop"}'))
        except Exception:
            pass
        self.llm_cycle_active = False
        self.get_logger().info(
            "CYCLE RESET: pick aborted, selections cleared — "
            "detection restarts fresh."
        )

    def emergency_stop(self) -> bool:
        """Best-effort immediate stop across every software motion path.

        Returns whether the real-robot dashboard stop request was sent. This
        is a software stop, not a replacement for the pendant's safety-rated
        emergency-stop circuit.
        """
        self.set_grasp_execution_enabled(False)
        self.publish_llm_abort()
        self.stop_motion()
        # Break the PICK LOOP too, not just the motion: clear the
        # pipeline's auto-pick label directly (works even when the voice
        # router is dead) and publish a stop intent so a living router
        # clears its queued picks. Without this the pipeline restarted a
        # new cycle right after every E-stop press and the button felt
        # dead (Majd, 2026-08-21: cycles 1..4 unkillable).
        try:
            self.set_pipeline_params({"auto_select_label": ""})
        except Exception as exc:
            self.get_logger().warn(f"E-stop: auto_select clear failed: {exc}")
        try:
            self._pub_intent.publish(String(
                data='{"action_intent":"stop"}'
            ))
        except Exception:
            pass

        dashboard_sent = False
        if self._dashboard_stop.service_is_ready():
            try:
                self._dashboard_stop.call_async(Trigger.Request())
                self._last_stop_cmd_t = time.monotonic()   # auto-Play must not undo this
                dashboard_sent = True
            except Exception as exc:
                self.get_logger().error(f"UR dashboard stop failed: {exc}")
        self.get_logger().fatal(
            "SOFTWARE E-STOP: MoveIt stopped, executor/replay aborted, "
            "controller goals cancelled%s.",
            ", UR program stop requested" if dashboard_sent else "",
        )
        return dashboard_sent

    def named_pose_server_available(self) -> bool:
        """True when /move_to_named_pose has an action server (ours, or the
        voice stack's)."""
        try:
            if getattr(self, "_named_pose_probe", None) is None:
                from rclpy.action import ActionClient
                from morty_interfaces.action import MoveToNamedPose
                self._named_pose_probe = ActionClient(self, MoveToNamedPose, "/move_to_named_pose")
            return bool(self._named_pose_probe.server_is_ready())
        except Exception:
            return True          # cannot probe: never start a duplicate blindly

    def fast_replay_available(self) -> bool:
        """True when the pipeline's replay node is subscribed and listening."""
        return self._pub_fast_replay.get_subscription_count() > 0

    def request_fast_replay(self, target_name: str, duration_s: float = 2.0) -> None:
        import json
        with self._lock:
            self._fast_status = ""
            self._fast_status_t = 0.0
        # JSON payload → per-request duration (the node stretches the cached
        # trajectory over this time; bigger = slower, calmer motion).
        payload = json.dumps({"label": str(target_name), "duration_s": float(duration_s)})
        self._pub_fast_replay.publish(String(data=payload))
        self.get_logger().info(
            f"[fast-replay] requested '{target_name}' over {duration_s:.1f}s from GUI"
        )

    def send_gripper(self, open_: bool) -> bool:
        """Fire-and-forget gripper open/close. Returns False if no server."""
        if not self._gripper_client.server_is_ready():
            if not self._gripper_client.wait_for_server(timeout_sec=0.5):
                return False
        goal = GripperCommand.Goal()
        goal.command.position = 0.5 if open_ else 0.0
        goal.command.max_effort = 0.0
        self._gripper_client.send_goal_async(goal)
        self.get_logger().info(f"Gripper {'OPEN' if open_ else 'CLOSE'} sent.")
        return True

    def set_gripper_holds(self, open_s: float, close_s: float,
                          quiet: bool = False) -> bool:
        """Push pulse hold durations onto the adapter node (async).
        Returns False when the adapter's param service is unreachable —
        the old silent skip made the Advanced spinboxes look dead
        (Majd 2026-08-25: "the time doesn't work when I change it").
        quiet=True suppresses the not-ready warn — used by the GUI's
        automatic startup retry, which knocks every ~1.5 s until the
        driver (and with it the adapter) exists (2026-08-26 spam)."""
        if not self._gripper_param_cli.wait_for_service(timeout_sec=0.5):
            if not quiet:
                self.get_logger().warn(
                    "gripper adapter param service not ready — hold values NOT applied")
            return False
        req = SetParameters.Request()
        for name, val in (("pulse_hold_s_open", open_s), ("pulse_hold_s_close", close_s)):
            p = Parameter(name=name)
            p.value = ParameterValue(
                type=ParameterType.PARAMETER_DOUBLE, double_value=float(val)
            )
            req.parameters.append(p)
        self._gripper_param_cli.call_async(req)
        if not quiet:
            # quiet=True is the GUI's periodic self-heal re-push (every
            # 5 s) — logging every one of those drowned the GUI log
            # (2026-08-31). Operator-initiated pushes still log.
            self.get_logger().info(
                f"Gripper holds set: open={open_s:.2f}s close={close_s:.2f}s"
            )
        return True

    def take_updates(self) -> Dict[str, Any]:
        """Return dirty data and clear the flags (called from the Qt timer)."""
        out: Dict[str, Any] = {}
        with self._lock:
            if self.image_dirty:
                out["image"] = self.image_bgr
            if self.astra_dirty:
                self.astra_dirty = False
                out["astra_image"] = self.astra_bgr
                self.image_dirty = False
            if self.cloud_dirty:
                out["cloud"] = (self.cloud_pts, self.cloud_cols)
                self.cloud_dirty = False
            if self.frame_dirty:
                out["frame_cloud"] = (self.frame_pts, self.frame_cols)
                self.frame_dirty = False
            out["campose"] = self.campose
            out["cycle_id"] = self.cycle_id
            if self.poses_dirty:
                out["poses"] = {k: list(v) for k, v in self.poses.items()}
                self.poses_dirty = False
            if self.log_dirty:
                out["log"] = list(self.log_lines)
                self.log_lines.clear()
                self.log_dirty = False
            if self.filter_dirty:
                out["filter"] = list(self.filter_rows)
                self.filter_dirty = False
            if self.utility_dirty:
                out["utility"] = list(self.utility_hist)
                self.utility_dirty = False
            out["status"] = self.status_text
            if getattr(self, "_voice_stop_pending", False):
                out["voice_stop"] = True
                self._voice_stop_pending = False
            now = time.monotonic()
            out["hw_alive"] = (now - self._joint_state_t) < 1.5
            out["pipeline_alive"] = (now - self._dets_t) < 2.5
            out["readiness"] = self._readiness if (now - self._readiness_t) < 30.0 else None
            # None = unknown (no listing yet / driver down); False = seen inactive within 15 s
            out["traj_ctrl_active"] = (getattr(self, "_traj_ctrl_active", None)
                                       if now - getattr(self, "_traj_ctrl_t", 0.0) < 15.0 else None)
            out["camera_alive"] = (now - self._camera_t) < 1.5
            # 3 s, not 1.5: 10 Hz over the shuttle network link.
            out["astra_alive"] = (now - self._astra_t) < 3.0
            # No freshness timeout for the recorder (latched, change-only);
            # liveness comes from the ROS graph instead.
            out["vla_node_up"] = self.vla_node_up()
            out["vla_status"] = self._vla_status
            # Latched state topic: valid until changed — BUT only while its
            # publisher (the UR driver) is alive. Without this check the
            # robot/play LEDs stayed green after the driver died (Majd's
            # bug report, 2026-08-11).
            if self.count_publishers(
                "/io_and_status_controller/robot_program_running"
            ) > 0:
                out["program"] = self._program_running
            else:
                out["program"] = None
            out["program_disconnected"] = self._program_disconnected
            out["joints"] = (list(self.joint_names), list(self.joint_positions))
            out["fast_status"] = (self._fast_status, self._fast_status_t)
            if self.pipeline_params_dirty:
                out["pipeline_params"] = dict(self.pipeline_params)
                self.pipeline_params_dirty = False
            if self.voice_dirty:
                out["chat_events"] = list(self.chat_events)
                self.chat_events.clear()
                out["chat_state"] = self.voice_chat_state
                self.voice_dirty = False
            if self._system_intents:
                out["system_intents"] = list(self._system_intents)
                self._system_intents.clear()
            if self._cand_status:
                out["cand_status"] = list(self._cand_status)
                self._cand_status.clear()
            out["voice_alive"] = (
                self.count_publishers("/chat/state") > 0
            )
            out["mic"] = (self.mic_level, self.speech_state, self.muted)
            out["chat_state_now"] = self.voice_chat_state
            # Keep the ring scrolling during silence (the original's trick),
            # and always hand the GUI the current snapshot.
            if time.monotonic() - self._wave_recv_t > 0.15:
                self._wave_ring.extend([0.0, 0.0, 0.0])
            out["waveform"] = np.fromiter(self._wave_ring, dtype=np.float32)
            if self.cycle_done:
                out["cycle_done"] = {
                    "cycle": self.cycle_id,
                    "best_score": self.best_score,
                    "utility": self.utility_hist[-1] if self.utility_hist else None,
                    "filter": list(self.filter_rows),
                }
                self.cycle_done = False
            if self._executor_done_pending:
                out["executor_done"] = self._executor_done_pending
                self._executor_done_pending = ""
            if self._exec_interactive_pending:
                out["exec_interactive"] = self._exec_interactive_pending
                self._exec_interactive_pending = ""
        return out


def start_bridge() -> Tuple[MortyGuiBridge, threading.Thread]:
    rclpy.init()
    node = MortyGuiBridge()

    def _paced_spin() -> None:
        # Paced spin (2026-08-27): plain rclpy.spin() burned ~a full core
        # rebuilding the wait set per message under the GUI's many
        # subscriptions. 50 ms poll + 10 ms breath adds at most ~20 ms to
        # inbound DISPLAY data — invisible at the GUI's 10 Hz poll.
        # Safety paths are unaffected: E-stop / stop_motion / abort
        # publish directly from the Qt thread, not through this executor.
        while rclpy.ok():
            try:
                rclpy.spin_once(node, timeout_sec=0.05)
            except Exception:
                break
            time.sleep(0.01)

    thread = threading.Thread(target=_paced_spin, daemon=True)
    thread.start()
    return node, thread
