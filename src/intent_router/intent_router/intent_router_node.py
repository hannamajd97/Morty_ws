"""Route /robot/intent JSON into grasp_pipeline parameter changes.

When chat_node publishes a structured pick/grasp intent, this node sets
the ``auto_select_label`` parameter on the grasp pipeline's selection node
so the pipeline auto-picks that object on its next cycle and bypasses its
built-in voice menu entirely. No changes to grasp_pipeline are required.
"""

from __future__ import annotations

import collections
import difflib
import json
import re
import threading
import time
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.wait_for_message import wait_for_message
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, Int64, String
from vision_msgs.msg import Detection2D

from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from controller_manager_msgs.srv import ListControllers, SwitchController

from morty_interfaces.action import MoveToNamedPose, OperateGripper, Speak


PICK_INTENTS = {"pick", "pickup", "pick_up", "pick up", "grasp", "take", "grab"}
GRIPPER_OPEN_INTENTS = {"open", "open_gripper", "release", "let_go", "let go", "drop_held"}
GRIPPER_CLOSE_INTENTS = {"close", "close_gripper", "grip", "clamp"}
GO_HOME_INTENTS = {"go_home", "home", "return_home", "return"}
READY_INTENTS = {
    "go_to_ready",
    "ready",
    "ready_position",
    "go_to_start",
    "start",
    "start_position",
    "go_to_the_start_position",
    "go to the start position",
    "go to start",
    "go to ready",
}
CLEAR_INTENTS = {"stop", "cancel", "abort"}

# Intents whose ONLY job, while the executor is paused, is to answer that
# pause (they are handled by the bypass branches in _on_intent). Outside a
# pause each of them means something quite different — a standalone gripper
# move, a standalone named-pose move — which is why a straggler must be
# swallowed rather than allowed to fall through to the queue.
RESUME_STYLE_INTENTS = {
    "open_gripper", "open", "release", "release_home", "drop_held",
    "let_go", "let go", "close_gripper", "close", "stay", "hold",
    "keep_holding", "confirm_drop", "yes",
    "drop_right", "drop_left",
    "drop_right_release", "drop_left_release",
    "drop_right_release_home", "drop_left_release_home",
    "go_home", "home",
    # v2 atomic carry+release (2026-08-31): only meaningful at the
    # release pause; a straggler with no pause open must be swallowed
    # like the others, never queued as a standalone action.
    "carry_release",
}

# How long after a resume token goes out a second resume-style intent is
# still treated as "the same breath" and swallowed.
#
# 2026-08-28: the losing intent arrived 4 ms after the winner. A straggler
# from the LLM path in the same turn can trail the FAST layer by ~2 s, so
# the window is 3 s. It is BOUNDED on purpose — if the executor dies and
# never opens another pause (and never sends cycle_done), the window
# expires and plain "open the gripper" works normally again instead of
# being swallowed forever.
RESUME_SWALLOW_WINDOW_S = 3.0

# Maps a "look_X" intent to the named pose loaded by ur3e_fast_observation_replay.
# Keep both underscore and space forms because LLM/ASR output varies.
LOOK_INTENTS: dict[str, str] = {
    "look_left": "detect_left",
    "look left": "detect_left",
    "view_left": "detect_left",
    "look_right": "detect_right",
    "look right": "detect_right",
    "view_right": "detect_right",
    "look_up": "detect_higher",
    "look up": "detect_higher",
    "look_higher": "detect_higher",
    "view_up": "detect_higher",
}

# "Sleep" / stow / park intents — the safe parked pose far from work area.
SLEEP_INTENTS = {
    "sleep", "go_to_sleep", "go to sleep", "sleep_home", "sleep home",
    "stow", "park", "rest",
}

# Drop-pose intents. Maps directly to named poses for the drop locations.
DROP_INTENTS: dict[str, str] = {
    # generic drop
    "drop_position": "drop",
    "drop_pose": "drop",
    "go_to_drop": "drop",
    "drop_location": "drop",
    # right-side drop
    "drop_right": "drop_right_side",
    "drop_right_side": "drop_right_side",
    "go_to_drop_right": "drop_right_side",
    "drop right": "drop_right_side",
    # left-side drop
    "drop_left": "drop_left_side",
    "drop_left_side": "drop_left_side",
    "go_to_drop_left": "drop_left_side",
    "drop left": "drop_left_side",
}


def _normalise_label(name: str) -> str:
    return "_".join(name.strip().lower().split())


class IntentRouterNode(Node):
    def __init__(self) -> None:
        super().__init__("intent_router_node")

        self.declare_parameter("intent_topic", "/robot/intent")
        self.declare_parameter("target_node", "yolo_live_select_grasp")
        self.declare_parameter("label_param_name", "auto_select_label")
        self.declare_parameter(
            "known_labels",
            [
                "tape_dispenser",
                "cd_case",
                "toy_faucet",
                "kitchen_towel",
                "nutmeg_jar",
                "retail_box",
                "small_bottle",
                "wooden_maze_puzzle",
                "vitamin_c",
            ],
        )
        self.declare_parameter("fuzzy_cutoff", 0.6)
        self.declare_parameter("clear_label_on_stop", True)
        self.declare_parameter("speak_feedback", True)
        self.declare_parameter("speak_action", "/speak")
        self.declare_parameter("cycle_id_topic", "/graspgen/selection_cycle_id")
        # morty_full.launch.py runs grasp_executor_lin_llm, which publishes
        # on /grasp_executor_lin_llm/cycle_done. The plain grasp_executor_lin
        # binary still exists but isn't spawned by the umbrella.
        self.declare_parameter("cycle_done_topic", "/grasp_executor_lin_llm/cycle_done")
        self.declare_parameter(
            "executor_resume_topic", "/grasp_executor_lin_llm/resume"
        )
        self.declare_parameter("narrate_lifecycle", True)
        # Special demo path for the vitamin_c (vitamin tube) label:
        # shorter first close pulse, skips the TI0 no-grasp check, fires a
        # second harder close pulse. Off by default — turn on with
        # enable_vitamin_mode:=true for the slip / re-grip demonstration.
        self.declare_parameter("enable_vitamin_mode", False)
        # Topic the executor publishes slip lifecycle on. We subscribe so
        # Morty can narrate "Slip detected" / "Grip restored".
        self.declare_parameter("slip_event_topic", "/executor/slip_event")
        self.declare_parameter("gripper_action", "/gripper")
        self.declare_parameter("gripper_force_n", 50.0)
        self.declare_parameter("named_pose_action", "/move_to_named_pose")
        # 'home' is the working / ready-to-grasp pose (was previously
        # 'initgrasp'). 'sleep_home' is the stowed pose (was previously
        # 'home'). See SRDF + observation_poses.yaml.
        self.declare_parameter("home_pose_name", "home")
        # The "ready" / "start position" pose. Defaults to the same pose as
        # home today, but it has its own parameter so you can point it at
        # a different pose later.
        self.declare_parameter("ready_pose_name", "home")
        # Spatial-pick wiring: subscribe to detections so we can resolve
        # "leftmost" / "rightmost" / "middle" qualifiers, and publish the
        # chosen detection index to manual_select_topic.
        self.declare_parameter("detections_topic", "/yolo/all_detections")
        self.declare_parameter("manual_select_topic", "/yolo/select_index")
        self.declare_parameter("scene_max_age_s", 3.0)
        # Robot health monitoring: avoid sending motion goals to a dead
        # robot, and tell the user clearly when something's offline.
        self.declare_parameter("joint_states_topic", "/joint_states")
        self.declare_parameter(
            "robot_program_running_topic",
            "/io_and_status_controller/robot_program_running",
        )
        self.declare_parameter("joint_states_max_age_s", 2.0)
        # /joint_states is SAMPLED, not subscribed to; see
        # _sample_joint_states_liveness(). The default period leaves 4x
        # margin against joint_states_max_age_s (2.0 s), so even a couple
        # of dropped samples in a row cannot make a healthy driver look
        # frozen, while a driver that really dies is still noticed inside
        # the same 2.0 s window as before.
        self.declare_parameter("liveness_sample_period_s", 0.5)
        self.declare_parameter("liveness_sample_timeout_s", 0.3)

        self._target_node = str(self.get_parameter("target_node").value)
        self._label_param = str(self.get_parameter("label_param_name").value)
        self._known_labels = list(self.get_parameter("known_labels").value or [])
        self._fuzzy_cutoff = float(self.get_parameter("fuzzy_cutoff").value)
        self._clear_on_stop = bool(self.get_parameter("clear_label_on_stop").value)
        self._speak_feedback = bool(self.get_parameter("speak_feedback").value)

        # Service client to set parameters on grasp_selection_node.
        self._set_params_client = self.create_client(
            SetParameters, f"/{self._target_node}/set_parameters"
        )

        # Optional Speak action client for narration.
        self._speak_client: Optional[ActionClient] = None
        if self._speak_feedback:
            self._speak_client = ActionClient(
                self, Speak, str(self.get_parameter("speak_action").value)
            )

        # OperateGripper client for direct gripper open/close/release intents.
        self._gripper_force = float(self.get_parameter("gripper_force_n").value)
        self._gripper_client = ActionClient(
            self, OperateGripper, str(self.get_parameter("gripper_action").value)
        )

        # MoveToNamedPose client for go_home / named-pose intents.
        self._home_pose = str(self.get_parameter("home_pose_name").value)
        self._ready_pose = str(self.get_parameter("ready_pose_name").value)
        self._named_pose_client = ActionClient(
            self,
            MoveToNamedPose,
            str(self.get_parameter("named_pose_action").value),
        )

        # Active goal handles so a "stop" intent can cancel mid-flight.
        self._active_gripper_handle = None
        self._active_named_pose_handle = None

        # Multi-action queue. New intents append; the dispatcher pops the
        # next one only when the previous operation has completed.
        # Operations and their completion signals:
        #   gripper        -> _on_gripper_result
        #   named_pose     -> _on_named_pose_result
        #   pick (label-)  -> /grasp_executor_lin/cycle_done
        #   pick (idx-)    -> /grasp_executor_lin/cycle_done
        #   stop           -> instant
        # The pick path also gets a watchdog timeout so a missing executor
        # cannot wedge the queue.
        self.declare_parameter("queue_pick_timeout_s", 90.0)
        self._queue_lock = threading.Lock()
        self._intent_queue: collections.deque[dict] = collections.deque()
        self._busy = False
        self._busy_kind: str = ""  # "gripper" | "named_pose" | "pick" | ""
        # Track the action_intent of the currently-dispatched goal so we can
        # dedup repeat requests for the same action while it's still running.
        self._currently_dispatched_action: str = ""
        self._pick_deadline: float = 0.0
        # Periodic watchdog so a hung pick doesn't block the queue forever.
        self._pick_timeout_s = float(self.get_parameter("queue_pick_timeout_s").value)
        self.create_timer(1.0, self._queue_watchdog)

        # ── Controller auto-recovery ──────────────────────────────────
        # The UR controller_stopper deactivates scaled_joint_trajectory_
        # controller when the robot program stops or the reverse interface
        # drops. The auto-restore on reconnect is unreliable on this
        # driver build — we've seen the controller stay 'inactive' after
        # a fast drop+reconnect cycle, and every subsequent move fails
        # with CONTROL_FAILED ("Can't accept new action goals. Controller
        # is not running."). This watchdog polls the controller manager
        # at 2 Hz and re-activates it whenever it's found inactive, so
        # the user never has to manually run set_controller_state.
        self._traj_controller_name = "scaled_joint_trajectory_controller"
        self._controller_manager_ns = "/controller_manager"
        self._list_controllers_cli = self.create_client(
            ListControllers,
            f"{self._controller_manager_ns}/list_controllers",
        )
        self._switch_controller_cli = self.create_client(
            SwitchController,
            f"{self._controller_manager_ns}/switch_controller",
        )
        self._last_controller_state: str = ""
        self.create_timer(2.0, self._controller_watchdog)

        # Detection cache for spatial picks.
        self._scene_max_age_s = float(self.get_parameter("scene_max_age_s").value)
        self._detections: list[dict] = []
        self._detections_at: float = 0.0
        self.create_subscription(
            String,
            str(self.get_parameter("detections_topic").value),
            self._on_detections,
            10,
        )
        self._select_index_pub = self.create_publisher(
            Int32, str(self.get_parameter("manual_select_topic").value), 10
        )
        # Label+bbox companion for spatial picks. A raw index computed from
        # OUR detection snapshot can point at a different object by the time
        # the pipeline consumes it (YOLO reorders detections between frames
        # — 2026-08-11 'small_bottle → wooden_maze_puzzle' bug). This topic
        # carries the matched detection so the pipeline re-resolves it
        # against its live list, same as the debug-window click path.
        self._select_det_pub = self.create_publisher(
            String, "/yolo/select_detection", 10
        )
        # Resume signal for the LLM grasp executor's interactive drop
        # pause. After Steps 1-3 (open, LIN approach, close) the executor
        # holds at the grasp pose and waits for one of:
        #   "drop_right" / "drop_left" / "home"  → continue with lift + drop
        #   "stop" / "cancel"                    → go home with gripper as-is
        # We publish here when the LLM emits a confirm_drop intent during
        # the paused window.
        self._executor_resume_pub = self.create_publisher(
            String, str(self.get_parameter("executor_resume_topic").value), 10
        )
        # Hard abort for the executor's pick cycle. "stop" used to cancel
        # only router-owned goals — the executor kept planning/moving
        # while Morty said "stopping" (2026-08-11).
        self._executor_abort_pub = self.create_publisher(
            String, "/grasp_executor_lin_llm/abort", 10
        )
        # Immediately halts whatever trajectory MoveIt is executing (same
        # signal as the GUI's ■ STOP button).
        self._traj_stop_pub = self.create_publisher(
            String, "/trajectory_execution_event", 10
        )
        self._pipeline_abort_pub = self.create_publisher(
            String, "/grasp_pipeline/abort", 10
        )
        # True between "pick dispatched in wait-mode" and "cycle_done
        # received", so we know whether a confirm_drop should be routed
        # to the executor resume topic (vs. ignored because nothing is
        # waiting on it).
        self._pick_awaiting_destination: bool = False
        # Second interactive pause: True after arrival at the drop pose
        # and before the gripper opens. Set by the 'awaiting_release'
        # lifecycle event from the executor, cleared by 'released' or
        # 'release_skipped'. While True, an open_gripper intent (or
        # affirmative phrases mapped to it) is routed to the executor's
        # resume topic instead of triggering a standalone gripper open.
        self._pick_awaiting_release: bool = False
        # True when THIS cycle's release pause resolved as KEEP-HOLDING
        # (executor published 'release_skipped' — the operator answered
        # the "should I open the gripper?" question with stop/stay/close,
        # or the pause timed out, and the object rides home in the
        # gripper). Reset at every new release pause and at cycle_done.
        # This is the ONLY signal that a finished cycle really ended with
        # something in the gripper: a normal cycle is pick -> drop ->
        # RELEASE -> home, i.e. EMPTY at cycle_done (2026-08-28: cycle_done
        # 'done|4' at 1787941443.35 was immediately followed by
        # "Held-object memory: 'small_bottle'" and Morty then insisted
        # "I'm still holding the small bottle" after the operator said it
        # was not).
        self._release_skipped_this_cycle: bool = False
        # ONE-RESUME-PER-PAUSE guard (2026-08-28). The executor has at
        # most ONE interactive pause open at a time and ANY resume token
        # consumes it. Until now every bypass branch published straight
        # to the resume topic and left the awaiting-flags set — they were
        # only cleared later by the executor's async status event. Two
        # intents from one utterance therefore both published:
        #   1787943598.921  plain 'go_home'     -> resume 'home'
        #   1787943598.925  open_gripper        -> resume 'release'
        # The executor consumed 'home' and left the pause, so 'release'
        # reached a robot that was no longer waiting and did NOTHING —
        # while the voice had already said "Opening the gripper."
        # _claim_executor_resume() now closes the pause at publish time;
        # the loser is SWALLOWED (never queued — a stale resume arriving
        # after the pause closed is exactly this bug) and narrated
        # honestly.
        self._pause_lock = threading.Lock()
        # True once a resume token has been published for the pause the
        # executor most recently opened. Reset by _on_interactive_status
        # whenever the executor announces a NEW pause, and at cycle end.
        self._resume_claimed: bool = False
        self._last_resume_dest: str = ""
        self._last_resume_mono: float = 0.0
        # Which OperateGripper command is in flight, so _on_gripper_result
        # can narrate what ACTUALLY happened rather than what was asked.
        self._last_gripper_command: int = -1

        # Held-object memory. Latched so a late subscriber (chat_node
        # restarting mid-session) gets the last known value. chat_node
        # consumes this and feeds it to the LLM as context, so "drop it on
        # the left" / "stell es nach links" can resolve "it" without the
        # user having to repeat the object name.
        held_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self._held_object_pub = self.create_publisher(
            String, "/robot/held_object", held_qos
        )
        self._held_object_label: str = ""
        # Last named pose successfully reached. chat_node feeds this to
        # the LLM as a developer message so it knows where the arm is.
        self._current_pose_pub = self.create_publisher(
            String, "/robot/current_pose", held_qos
        )
        self._current_pose_name: str = ""
        # Last named pose REQUESTED (may still be in flight). Used for
        # the completion handler to know what pose we just finished.
        self._pending_pose_name: str = ""

        # Robot health: track joint_states freshness + program-running flag.
        self._js_max_age_s = float(self.get_parameter("joint_states_max_age_s").value)
        self._joint_states_at: float = 0.0
        self._robot_program_running: bool | None = None  # None = unknown
        # /joint_states publishes at ~494 Hz and this node reads NOTHING
        # from it but "did a message arrive recently". A standing
        # subscription cost ~0.9 of a CPU core (2026-08-28) even though
        # the callback only stamped a float, and that load is what made
        # scaled_joint_trajectory_controller abort a real grasp with
        # PATH_TOLERANCE_VIOLATED (0.2007 rad lag against a 0.2000 rad
        # tolerance). We now take ONE message per tick through a
        # momentary subscription; _joint_states_at and every threshold
        # that reads it are unchanged.
        self._joint_states_topic = str(
            self.get_parameter("joint_states_topic").value
        )
        self._liveness_sample_period_s = float(
            self.get_parameter("liveness_sample_period_s").value
        )
        self._liveness_sample_timeout_s = float(
            self.get_parameter("liveness_sample_timeout_s").value
        )
        self.create_timer(
            max(0.05, self._liveness_sample_period_s),
            self._sample_joint_states_liveness,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("robot_program_running_topic").value),
            self._on_robot_program_running,
            10,
        )

        self.create_subscription(
            String,
            str(self.get_parameter("intent_topic").value),
            self._on_intent,
            10,
        )

        # Lifecycle narration: speak when a grasp starts, finishes, or fails.
        # Tracks the most recent target so end-of-cycle narration can reference it.
        # The cycle_done subscription is ALWAYS on (regardless of narrate flag)
        # because the multi-action queue depends on it to know when a pick is
        # complete. Narration itself is gated by the flag inside the handler.
        self._narrate_lifecycle = bool(self.get_parameter("narrate_lifecycle").value)
        self._last_target_label: str = ""
        self._last_cycle_id: int = 0
        # Cycle id at which _last_target_label was deliberately set by an
        # intent (vs. opportunistically picked up from /yolo/selected_detection).
        # While _last_target_label_lock_cycle == self._last_cycle_id, the
        # on-selected-detection callback refuses to overwrite the label,
        # so the cycle-start narration says the object the user actually
        # asked for, not the previous pick's stale detection.
        self._last_target_label_lock_cycle: int = -1
        # When the lock was taken. The lock also EXPIRES after 45 s: if
        # the pipeline was restarted, cycle ids reset and the ==-cycle
        # comparison above can stay true forever, freezing a stale label.
        self._last_target_label_lock_t: float = 0.0
        self.create_subscription(
            Int64,
            str(self.get_parameter("cycle_id_topic").value),
            self._on_cycle_id,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("cycle_done_topic").value),
            self._on_cycle_done,
            10,
        )
        # Lifecycle status for the interactive drop: 'paused' fires when
        # the executor has finished Step 3 (close gripper) and is now
        # blocking on the resume topic; 'resumed:<dest>' fires when we
        # send a destination and the executor continues. The 'paused'
        # event is our cue to ask the user "where should I drop it?"
        # because we know the grasp succeeded.
        self.create_subscription(
            String,
            "/grasp_executor_lin_llm/interactive_status",
            self._on_interactive_status,
            10,
        )
        # Pipeline publishes the selected object's class on this topic
        # whenever any selection method commits — voice, mouse click,
        # or mediapipe pointing. We use it to keep _last_target_label
        # in sync with NON-VOICE selections, so the "Starting grasp on
        # the X" narration still names the object when the user picks
        # by pointing.
        self.create_subscription(
            Detection2D,
            "/yolo/selected_detection",
            self._on_selected_detection,
            10,
        )
        # Slip lifecycle from the LIN executor's slip watchdog. Two
        # values are published: "slip_start" (TI1 went HIGH → executor
        # dropped to slow speed) and "slip_clear" (TI1 went LOW → speed
        # restored). We narrate both so the user hears what the robot
        # is doing.
        self.create_subscription(
            String,
            str(self.get_parameter("slip_event_topic").value),
            self._on_slip_event,
            10,
        )

        self.get_logger().info(
            f"intent_router_node listening on {self.get_parameter('intent_topic').value}; "
            f"will set {self._label_param} on /{self._target_node}; "
            f"known labels={self._known_labels}; "
            f"narrate_lifecycle={self._narrate_lifecycle}"
        )

    # ── intent handling ──────────────────────────────────────────────────

    def _on_intent(self, msg: String) -> None:
        raw = (msg.data or "").strip()
        if not raw:
            return
        try:
            intent = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Could not parse intent JSON: {exc}; raw={raw!r}")
            return

        action = str(intent.get("action_intent") or "").strip().lower()

        # Stop is special: it must execute immediately and clear the queue,
        # not wait behind whatever's currently running.
        if action in CLEAR_INTENTS:
            # If a pick is paused at the grasp pose OR the release
            # pose, tell the executor to unblock with 'stop' — gripper
            # stays as-is and arm goes home.
            if self._pick_awaiting_destination or self._pick_awaiting_release:
                self.get_logger().info(
                    "Stop received while pick is paused → publishing "
                    "'stop' on executor resume topic."
                )
                self._claim_executor_resume(
                    "stop", "stop received while pick is paused")
            elif getattr(self, "_pick_awaiting_home", False):
                # Home-confirm pause. 'stop' reaches CLEAR_INTENTS before
                # the _pick_awaiting_home block further down, so that
                # block's stop branch never ran and the pause stayed
                # UNANSWERED until the executor's 1800 s timeout — then
                # it went home anyway. Answer it with 'stay'.
                # NOTE: the flag is cleared by _claim_executor_resume
                # (which closes the pause atomically), not here.
                self._claim_executor_resume(
                    "stay",
                    "stop received at the home-confirm pause; arm stays put",
                )
            with self._queue_lock:
                cleared = len(self._intent_queue)
                self._intent_queue.clear()
            if cleared:
                self.get_logger().info(f"Stop received: cleared {cleared} queued intent(s).")
            self._handle_stop()
            self._busy = False
            self._busy_kind = ""
            self._try_dispatch_next()
            return

        # ── ONE RESUME PER PAUSE: swallow the loser ──────────────────
        # A resume-style intent that arrives just after the pause it was
        # meant to answer has already been answered must NOT fall through
        # to the queue, where it would be re-interpreted as a standalone
        # action (an open_gripper here would fire the gripper mid-motion
        # and drop the object). It is dropped, logged, and narrated
        # honestly. Deliberately NOT queued for later: a stale resume
        # replayed after the pause closed is the 2026-08-28 bug itself.
        if (
            action in RESUME_STYLE_INTENTS
            and self._resume_claimed
            and not self._executor_pause_open()
            and (time.monotonic() - self._last_resume_mono)
            <= RESUME_SWALLOW_WINDOW_S
        ):
            age_ms = (time.monotonic() - self._last_resume_mono) * 1000.0
            self.get_logger().warn(
                f"SWALLOWED intent {action!r}: no executor pause is open — "
                f"the pause it targets was already answered {age_ms:.0f} ms "
                f"ago by resume {self._last_resume_dest!r} (two intents from "
                f"one utterance both aimed at the same single pause; the "
                f"first one closed it). Dropping it, NOT queued for later: "
                f"a stale resume arriving after the pause closed reaches an "
                f"executor that is no longer waiting and does nothing, which "
                f"is exactly the bug this guard fixes. Say it again once the "
                f"robot has stopped, or use the compound form "
                f"(\'open the gripper and go back home\')."
            )
            self._narrate_ignored_resume(action, "")
            return

        # Release-pause bypass: arm is at the drop pose, gripper still
        # closed, waiting for the user to say "release" / "open". We
        # route open_gripper (and the safety-net affirmation phrases
        # that map to it) directly to the executor's resume topic. Same
        # deadlock-avoidance reason as the destination pause below.
        if self._pick_awaiting_release:
            if action == "carry_release":
                # v2 (2026-08-31): ATOMIC "take it to <dest> and give it
                # to me". One token, because the executor's resume
                # mailbox is single-slot — publishing a destination and
                # a release as two messages back-to-back would overwrite
                # the destination and open the gripper AT THE CURRENT
                # POSE (the exact failure class the 08-27 review caught).
                _attrs = intent.get("object_attributes") or {}
                if not isinstance(_attrs, dict):
                    _attrs = {}
                _d = str(_attrs.get("destination", "")).lower()
                _tgt = {"home": "home", "left": "drop_left_side",
                        "right": "drop_right_side"}.get(_d, "")
                if _tgt:
                    if not self._claim_executor_resume(
                        f"carry_release:{_tgt}",
                        f"release pause: carry+release to {_tgt}",
                    ):
                        self._narrate_ignored_resume(action, "carry_release")
                else:
                    self.get_logger().warn(
                        f"carry_release with unknown destination "
                        f"{_d!r} — ignoring.")
                return
            if action == "release_home":
                # Compound "open the gripper and go back home" — the
                # executor opens and goes straight home, no question.
                if not self._claim_executor_resume(
                    "release_home", "release pause: compound release_home"
                ):
                    self._narrate_ignored_resume(action, "release_home")
                return
            if action in ("open_gripper", "release"):
                # Was: log "publishing 'release'" then publish blindly.
                # The log line itself was a lie once the pause had already
                # been consumed by another token 4 ms earlier — the claim
                # helper logs the truth either way now.
                if not self._claim_executor_resume(
                    "release", f"bypass queue (release paused): {action}"
                ):
                    self._narrate_ignored_resume(action, "release")
                return
            if action in ("close_gripper", "stay", "hold", "keep_holding"):
                # User wants to keep holding — tell executor to skip.
                # 'stay'/'hold' come from the chat node's deterministic
                # NEGATIVE-answer layer ("no", "don't open it", "not
                # yet"): before 2026-08-28 a negative answer here matched
                # nothing and fell through to the LLM, which re-emitted
                # "I'm at the home position. Should I open the gripper?"
                # forever.
                if not self._claim_executor_resume(
                    "stop",
                    f"bypass queue (release paused, keep holding): {action}",
                ):
                    self._narrate_ignored_resume(action, "stop")
                return

        # Interactive-drop resume: confirm_drop and the safety-net
        # redirects (drop_right / drop_left / go_home while a pick is
        # paused) must execute IMMEDIATELY rather than queuing behind the
        # in-flight pick. Otherwise we deadlock: the pick is busy waiting
        # for the executor, the executor is paused waiting for the resume
        # signal, and the resume signal is sitting in the queue waiting
        # for busy to clear. We just shove the destination on the resume
        # topic — the pick cycle resumes and emits cycle_done as usual.
        # Also forward confirm_drop when a pick is in flight even if the
        # awaiting flag is somehow desynced. The executor will no-op
        # the resume if it isn't paused — safe fallback.
        # Home-confirm pause (2026-08-25): the executor released the
        # object and asked "Should I go back home?" — the answer must go
        # to the resume topic, never spawn a named-pose move mid-cycle.
        if getattr(self, "_pick_awaiting_home", False):
            if action in ("go_home", "confirm_drop", "yes"):
                if not self._claim_executor_resume(
                    "home", f"home-confirm pause: {action}"
                ):
                    self._narrate_ignored_resume(action, "home")
                return
            if action in ("stop", "stay", "hold", "keep_holding",
                          "close_gripper"):
                # NOTE: plain "stop" is intercepted by the CLEAR_INTENTS
                # block above (which now answers this pause too); the
                # remaining actions arrive from the chat node's
                # deterministic NEGATIVE-answer layer.
                if not self._claim_executor_resume(
                    "stay",
                    f"home-confirm pause: {action!r} — arm remains at the "
                    "drop pose",
                ):
                    self._narrate_ignored_resume(action, "stay")
                return
        # Synthetic compound intents from the chat FAST layer
        # (2026-08-26): destination + release(+home) in one deterministic
        # action, immune to the LLM forgetting the 'then' attribute.
        _compound = {
            "drop_right_release": ("drop_right", "+release"),
            "drop_left_release": ("drop_left", "+release"),
            "drop_right_release_home": ("drop_right", "+release_home"),
            "drop_left_release_home": ("drop_left", "+release_home"),
        }
        pick_in_flight = self._pick_awaiting_destination or self._busy_kind == "pick"
        if action in _compound:
            base, suffix = _compound[action]
            if pick_in_flight:
                if not self._claim_executor_resume(
                    base + suffix, f"compound {action!r}"
                ):
                    self._narrate_ignored_resume(action, base + suffix)
                return
            # No pick running: behave like the plain drop-side move.
            action = base
            intent["action_intent"] = base
        if pick_in_flight:
            attrs = intent.get("object_attributes") or {}
            dest = ""
            if isinstance(attrs, dict):
                dest = str(attrs.get("destination") or "").strip().lower()
            if action == "confirm_drop":
                dest = dest or "drop_right"
                # Compound plan (2026-08-25): "drop right, open the
                # gripper and go home" in ONE sentence — carry the
                # release/home authorization through as a suffix so the
                # executor never pauses again for it.
                then_s = ""
                if isinstance(attrs, dict):
                    then_s = str(attrs.get("then") or "").strip().lower()
                if then_s == "release_home":
                    dest += "+release_home"
                elif then_s == "release":
                    dest += "+release"
                if not self._claim_executor_resume(
                    dest, "bypass queue (pick in flight): confirm_drop"
                ):
                    self._narrate_ignored_resume(action, dest)
                return
            if action in ("drop_right", "drop_left", "go_home"):
                redirect = {"drop_right": "drop_right",
                            "drop_left": "drop_left",
                            "go_home": "home"}[action]
                if not self._claim_executor_resume(
                    redirect,
                    f"bypass queue (pick paused): plain {action!r}",
                ):
                    self._narrate_ignored_resume(action, redirect)
                return

        # Deduplicate: don't queue the same action twice in a row, and don't
        # queue an action that matches what is currently running. This stops
        # the queue from piling up when the user repeats themselves while a
        # move is still in flight (e.g. "go to ready ... going to ready" from
        # mic-feedback or a slow LLM round-trip).
        with self._queue_lock:
            duplicate = False
            if self._busy and self._currently_dispatched_action == action:
                duplicate = True
            elif self._intent_queue:
                last = self._intent_queue[-1]
                if str(last.get("action_intent") or "").strip().lower() == action:
                    # Same action already at the tail — collapse.
                    duplicate = True
            if duplicate:
                self.get_logger().info(
                    f"Dedup: dropping repeat intent {action!r} "
                    "(already running or already queued)."
                )
                return
            self._intent_queue.append(intent)
            queued = len(self._intent_queue)
        self.get_logger().info(
            f"Queued intent {action!r}; queue size now {queued}, busy={self._busy}"
        )
        self._try_dispatch_next()

    def _try_dispatch_next(self) -> None:
        with self._queue_lock:
            if self._busy or not self._intent_queue:
                return
            intent = self._intent_queue.popleft()
            self._busy = True
        self._dispatch(intent)

    def _mark_busy_done(self, kind: str) -> None:
        self._currently_dispatched_action = ""
        if self._busy_kind and self._busy_kind != kind:
            # Mismatch — somebody else completed first; ignore this signal so
            # we don't double-pop the queue.
            return
        self._busy = False
        self._busy_kind = ""
        self._try_dispatch_next()

    def _dispatch(self, intent: dict) -> None:
        action = str(intent.get("action_intent") or "").strip().lower()
        self._currently_dispatched_action = action
        obj = str(intent.get("object_intent") or "").strip()
        attrs = intent.get("object_attributes") or {}
        colour = shape = position = destination = ""
        ordinal = 0
        ordinal_from = "left"
        speed = "slow"
        if isinstance(attrs, dict):
            colour = str(attrs.get("colour") or "").strip()
            shape = str(attrs.get("shape") or "").strip()
            position = str(attrs.get("position") or "").strip().lower()
            try:
                ordinal = int(attrs.get("ordinal") or 0)
            except (TypeError, ValueError):
                ordinal = 0
            if str(attrs.get("ordinal_from") or "").strip().lower() == "right":
                ordinal_from = "right"
            destination = str(attrs.get("destination") or "").strip().lower()
            raw_speed = str(attrs.get("speed") or "").strip().lower()
            # Normalise variations (with/without underscore).
            raw_speed_n = raw_speed.replace(" ", "_")
            if raw_speed_n in {
                "very_slow", "slow", "normal", "fast", "very_fast",
            }:
                speed = raw_speed_n
        # Map symbolic speed to velocity_scaling for the MoveToNamedPose
        # action. Capped at 50% — anything higher overspeeds the UR3e
        # trajectory controller and triggers a protective-stop crash.
        # Tiers cluster under that ceiling so even "very fast" stays
        # safe. named_pose_server picks a matching duration bucket.
        speed_to_vs = {
            "very_slow": 0.05,
            "slow":      0.20,
            "normal":    0.30,
            "fast":      0.40,
            "very_fast": 0.50,   # hard ceiling
        }
        velocity_scaling = speed_to_vs.get(speed, speed_to_vs["slow"])

        self.get_logger().info(
            f"dispatching: action={action!r}, object={obj!r}, "
            f"colour={colour!r}, shape={shape!r}, position={position!r}, "
            f"speed={speed!r}"
        )

        # ── confirm_drop: resume a paused interactive pick ─────────────
        # This intent fires after the executor has paused at the grasp
        # pose (post Step 3) and the user has said where to drop ("drop
        # right", "drop left", "home", or "stop"). We publish the
        # destination string on the executor's resume topic and DO NOT
        # touch _busy_kind — the pick is still in flight on the executor
        # side; we're just feeding it the destination it asked for.
        if action == "confirm_drop":
            # Normally this is handled in the bypass path of _on_intent,
            # but keep this branch as a fallback in case the bypass
            # didn't fire (e.g. _pick_awaiting_destination flipped between
            # _on_intent and _dispatch).
            # Permissive: if there's ANY pick busy_kind in flight (even
            # if our internal flag is desynced from the executor), just
            # forward the resume. The executor will ignore the resume
            # if it isn't actually paused — that's a safe no-op — but
            # an over-cautious refusal here strands the user with the
            # executor paused forever.
            if not (self._pick_awaiting_destination or self._busy_kind == "pick"):
                self.get_logger().info(
                    "confirm_drop received but no pick is busy AND no "
                    "pause flag set; ignoring."
                )
                self._mark_busy_done("")
                return
            dest = destination or "drop_right"
            # force=True keeps the documented anti-stranding behaviour
            # (forward even when our flag desynced from the executor) but
            # still refuses to publish a SECOND token for a pause that
            # already got one.
            if not self._claim_executor_resume(
                dest,
                f"confirm_drop (in-queue, busy_kind={self._busy_kind!r}, "
                f"awaiting={self._pick_awaiting_destination})",
                force=True,
            ):
                self._narrate_ignored_resume(action, dest)
            self._mark_busy_done("")
            return

        # Safety-net fallback for plain drop_*/go_home that slipped past
        # the bypass in _on_intent. Should be rare.
        if self._pick_awaiting_destination and action in (
            "drop_right", "drop_left", "go_home"
        ):
            redirect_map = {
                "drop_right": "drop_right",
                "drop_left": "drop_left",
                "go_home": "home",
            }
            dest = redirect_map[action]
            if not self._claim_executor_resume(
                dest, f"in-queue fallback; redirecting {action!r}"
            ):
                self._narrate_ignored_resume(action, dest)
            self._mark_busy_done("")
            return

        # ── stop while a pick is paused: open gripper? NO — user wants
        # the gripper LEFT as-is and the arm to go home. Send 'stop' on
        # the resume topic so the executor unblocks and runs the PTP-home
        # path. We still honour normal stop semantics afterwards.
        if action == "stop" and self._pick_awaiting_destination:
            # Was a RAW publish that bypassed every guard — routed
            # through the claim helper so it closes the pause like any
            # other resume token (2026-08-28).
            self._claim_executor_resume(
                "stop", "in-queue stop received while pick is paused")
            self._mark_busy_done("")
            return

        if action in GRIPPER_OPEN_INTENTS:
            reason = self._check_robot_ready()
            if reason:
                self._narrate(reason)
                self._mark_busy_done("gripper")
                return
            self._busy_kind = "gripper"
            # No start narration — the LLM reply already confirms.
            self._send_gripper(OperateGripper.Goal.OPEN, "")
            return

        if action in GRIPPER_CLOSE_INTENTS:
            reason = self._check_robot_ready()
            if reason:
                self._narrate(reason)
                self._mark_busy_done("gripper")
                return
            self._busy_kind = "gripper"
            self._send_gripper(OperateGripper.Goal.CLOSE, "")
            return

        if action in GO_HOME_INTENTS:
            reason = self._check_robot_ready()
            if reason:
                self._narrate(reason)
                self._mark_busy_done("named_pose")
                return
            self._busy_kind = "named_pose"
            # No start narration — chat_node's LLM reply already says
            # something like "Going home" and we don't want double TTS.
            self._send_named_pose(self._home_pose, "", velocity_scaling)
            return

        if action in READY_INTENTS:
            reason = self._check_robot_ready()
            if reason:
                self._narrate(reason)
                self._mark_busy_done("named_pose")
                return
            self._busy_kind = "named_pose"
            self._send_named_pose(self._ready_pose, "", velocity_scaling)
            return

        if action in LOOK_INTENTS:
            reason = self._check_robot_ready()
            if reason:
                self._narrate(reason)
                self._mark_busy_done("named_pose")
                return
            self._busy_kind = "named_pose"
            pose = LOOK_INTENTS[action]
            self._send_named_pose(pose, "", velocity_scaling)
            return

        if action in SLEEP_INTENTS:
            reason = self._check_robot_ready()
            if reason:
                self._narrate(reason)
                self._mark_busy_done("named_pose")
                return
            self._busy_kind = "named_pose"
            self._send_named_pose("sleep_home", "", velocity_scaling)
            return

        if action in DROP_INTENTS:
            reason = self._check_robot_ready()
            if reason:
                self._narrate(reason)
                self._mark_busy_done("named_pose")
                return
            self._busy_kind = "named_pose"
            self._send_named_pose(DROP_INTENTS[action], "", velocity_scaling)
            return

        # "place" intent: when the user says "place it on the right" or
        # similar, route to the right drop pose based on position.
        if action in {"place", "drop_held"}:
            reason = self._check_robot_ready()
            if reason:
                self._narrate(reason)
                self._mark_busy_done("named_pose")
                return
            # Pick the drop pose by position attribute.
            if position in ("right", "rightmost"):
                pose = "drop_right_side"
            elif position in ("left", "leftmost"):
                pose = "drop_left_side"
            else:
                pose = "drop"
            self._busy_kind = "named_pose"
            self._send_named_pose(pose, "", velocity_scaling)
            return

        # ── pick_and_place compound intent ───────────────────────────────
        # Treat as a normal pick, but first instruct the C++ grasp executor
        # to use the requested drop pose (drop_right_side / drop_left_side /
        # drop / home) instead of its default. The executor handles the full
        # ── multi-view toggle ────────────────────────────────────────────
        if action == "enable_multiview":
            self._set_pipeline_bool("registration_enabled", True)
            self._mark_busy_done("")
            return
        if action == "disable_multiview":
            self._set_pipeline_bool("registration_enabled", False)
            self._mark_busy_done("")
            return

        # pick → move → release → return-home cycle internally; we just
        # parameterise it for this run. Empty/unknown destination falls back
        # to the executor default.
        # Decide the pick MODE first (so the executor knows whether to
        # pause) but don't SEND anything to the executor yet — we still
        # need to run the readiness + label checks below. If those fail
        # we abort silently without leaving wait_for_destination=True
        # stuck on the executor.
        is_pick_and_place = (action == "pick_and_place")
        chosen_drop = ""
        if is_pick_and_place:
            drop_map = {
                "drop_right": "drop_right_side",
                "drop_left": "drop_left_side",
                "drop_center": "drop",
                "home": "home",
            }
            chosen_drop = drop_map.get(destination) or ""
            if not chosen_drop:
                self.get_logger().info(
                    f"pick_and_place has no/unknown destination "
                    f"({destination!r}); using executor default."
                )
            action = "pick"

        if action not in PICK_INTENTS:
            self.get_logger().info(f"Action {action!r} not recognised; skipping.")
            self._mark_busy_done("")
            return

        if not obj:
            self.get_logger().warn("Pick intent has no object_intent; skipping.")
            self._narrate("I didn't catch which object to pick.")
            self._mark_busy_done("")
            return

        reason = self._check_robot_ready()
        if reason:
            self._narrate(reason)
            self._mark_busy_done("")
            return

        label = self._resolve_label(obj, colour, shape)
        if label is None:
            self.get_logger().warn(f"Could not match object {obj!r} to a known label.")
            self._narrate(f"I don't recognise {obj} as something I can pick.")
            self._mark_busy_done("")
            return

        # NAME → VARIANT inference (Majd 2026-08-31: "Vitamin C" /
        # "B12" ARE disambiguations — the system asked "leftmost or
        # rightmost?" three times at a user who had already named the
        # exact tube). The two tubes share the label vitamin_c and
        # differ by cap colour; the debug overlay itself maps
        # yellow/orange→C and red/purple→B12. If the operator's words
        # name a variant and no colour qualifier was extracted, derive
        # it here so the existing cap-colour matcher (with its lighting
        # aliases) can pin the right tube instead of re-asking.
        if label is not None and label.lower() == "vitamin_c" and not colour:
            _oname = (obj or "").lower()
            if re.search(r"\bb\s*-?\s*12\b", _oname):
                colour = "red"       # alias group {red, purple} = B12 cap
            elif re.search(r"(?:^|[^a-z])c(?:[^a-z]|$)", _oname):
                colour = "yellow"    # alias group {yellow, orange} = C cap
            if colour:
                self.get_logger().info(
                    f"Inferred tube variant from name {obj!r} -> "
                    f"colour {colour!r}.")

        # Hard backstop: refuse plain-pick of vitamin_c when 2+ are
        # visible AND the user didn't pin one down by position. The LLM
        # is told to ask for clarification via system prompt, but in
        # practice it sometimes hedges by firing BOTH a pick tool call
        # AND a clarifying reply. Catch the pick here and force the
        # clarification path.
        if label.lower() == "vitamin_c":
            n_vit = self._count_visible_for_label(label)
            self.get_logger().info(
                f"Vitamin-tube pick check: position={position!r}, "
                f"visible_count={n_vit}, scene_age_s="
                f"{(self.get_clock().now().nanoseconds/1e9 - self._detections_at):.2f}"
            )
            if not position and not ordinal and not colour and n_vit >= 2:
                self.get_logger().info(
                    "Refusing plain-pick of vitamin_c: 2+ visible and no "
                    "position/ordinal/colour qualifier. Asking user to clarify."
                )
                self._narrate(
                    "I see two vitamin tubes. Could you say leftmost or "
                    "rightmost, or point at the one you want?"
                )
                self._mark_busy_done("")
                return

        # All checks passed — now commit to the executor. Order matters:
        # set drop_target FIRST (so pick_and_place's destination lands),
        # THEN wait_for_destination, THEN the awaiting flag for our own
        # bypass routing.
        if is_pick_and_place and chosen_drop:
            self._set_executor_drop_target(chosen_drop)
        if is_pick_and_place:
            self._set_executor_wait_for_destination(False)
        else:
            self._set_executor_wait_for_destination(True)
            self._pick_awaiting_destination = True

        # Toggle the executor's vitamin-tube special mode. Only enabled
        # for vitamin_c picks AND only if the operator has opted in via
        # the enable_vitamin_mode parameter (default False). Without
        # opt-in, vitamin_c picks follow the standard pick path like any
        # other object, so the user is not surprised by a different
        # close pulse / no-TI0-check behaviour.
        enable_vitamin_mode = bool(
            self.get_parameter("enable_vitamin_mode").value
        )
        is_vitamin = enable_vitamin_mode and (label.lower() == "vitamin_c")
        self._set_executor_param(
            "vitamin_mode_active", is_vitamin, kind="bool"
        )

        self._busy_kind = "pick"
        self._pick_deadline = time.monotonic() + self._pick_timeout_s

        # The pick cycle moves the arm through several non-named poses
        # (grasp → lift → drop → home). Clear the latched current-pose so
        # the LLM doesn't keep saying "I'm at home" while the arm is mid
        # pick. The replay_node will re-publish 'home' (or whichever
        # named pose the executor lands on) when the cycle finishes.
        try:
            self._current_pose_name = ""
            self._current_pose_pub.publish(String(data=""))
        except Exception:
            pass

        if position or ordinal or colour:
            index = self._resolve_index_by_position(
                label, position,
                colour=colour, ordinal=ordinal, ordinal_from=ordinal_from,
            )
            if index is not None:
                if ordinal:
                    qual = f"number {ordinal} from the {ordinal_from}"
                elif position:
                    qual = position
                else:
                    qual = colour
                self._narrate(
                    f"Picking the {qual} "
                    f"{self._spoken_name_for_index(index, label)}."
                )
                # 2026-08-25 (Majd: "selecting the wrong object even if
                # he detected the right one"): do NOT also set
                # auto_select_label here — the label path independently
                # picks ANY same-label detection and raced/overrode the
                # precise spatial selection (narrated Multi, grabbed C).
                # A qualified pick lives or dies by its det re-resolution;
                # if that refuses, the pipeline re-prompts instead of
                # blind-grabbing a sibling.
                self._last_target_label = label
                self._open_execution_gate()
                self._publish_select_index(index)
                return
            self.get_logger().info(
                f"Qualifier (position={position!r}, ordinal={ordinal}, "
                f"colour={colour!r}) given but no matching detection; "
                "falling back to label-only selection."
            )

        # Label-only path: set auto_select_label on the pipeline and let
        # it match against its live names list in the next selection cycle.
        # We deliberately do NOT publish a /yolo/select_index from our own
        # scene cache here — detection indices can shift between the
        # router's snapshot and the pipeline's, so an index from a stale
        # cache picks the wrong object (e.g. 'small_bottle' → index 0 in
        # the router's snapshot, but index 0 in the live pipeline cache is
        # 'tape_dispenser'). The label is invariant; the pipeline matches
        # it against the live `names` list in its selection cycle, which
        # re-runs every ~1s thanks to selection_manual_wait_s=1.0.
        self._set_target_label(label)

    # ── label resolution ─────────────────────────────────────────────────

    def _resolve_label(self, obj: str, colour: str, shape: str) -> Optional[str]:
        # Build candidate strings the LLM might have spoken, in priority order.
        candidates = [obj]
        if colour:
            candidates.append(f"{colour} {obj}")
        if shape:
            candidates.append(f"{shape} {obj}")
        normalised = [_normalise_label(c) for c in candidates if c]

        # Exact match first.
        for cand in normalised:
            if cand in self._known_labels:
                return cand

        # Substring match (handles "tube" -> "vitamin_tube", "towel" -> "kitchen_towel").
        for cand in normalised:
            tokens = cand.split("_")
            for label in self._known_labels:
                if any(tok and tok in label.split("_") for tok in tokens):
                    return label

        # Fuzzy fallback.
        match = difflib.get_close_matches(
            normalised[0], self._known_labels, n=1, cutoff=self._fuzzy_cutoff
        )
        return match[0] if match else None

    # ── robot health ─────────────────────────────────────────────────────

    def _sample_one_message(self, msg_type, topic: str):
        """Take ONE message from *topic* through a MOMENTARY subscription.

        Returns the message, or None if nothing arrived inside
        ``liveness_sample_timeout_s``. Never raises: a liveness probe must
        not be able to take the node down, and a failed probe is
        indistinguishable from a silent topic, which the staleness
        thresholds already handle.

        Why not a standing subscription: rclpy pays a per-message wait-set
        cost even when the callback body is trivial, so subscribing to a
        ~494 Hz topic burns most of a CPU core to do nothing.

        Why rclpy's wait_for_message() and not the create/destroy busy-wait
        in ur3e_fast_observation's replay_node._capture_current_joints():
        that node runs a MultiThreadedExecutor, so a subscription in a
        dedicated ReentrantCallbackGroup can still be delivered while the
        caller blocks. THIS node runs rclpy.spin() — a SingleThreadedExecutor
        — where no callback can ever be delivered while another callback is
        on the stack, so that pattern would time out every single time.
        wait_for_message() builds its OWN private wait set and takes the
        message directly, bypassing the executor, and destroys the
        subscription in a finally: block so nothing can leak.
        """
        try:
            # Cheap local graph query. With no publisher at all (driver
            # not started — the common case) there is nothing to wait for,
            # so skip the bounded wait entirely rather than blocking this
            # node's only executor thread for nothing.
            if self.count_publishers(topic) == 0:
                return None
            ok, msg = wait_for_message(
                msg_type,
                self,
                topic,
                time_to_wait=max(0.0, self._liveness_sample_timeout_s),
            )
            return msg if ok else None
        except Exception:
            return None

    def _sample_joint_states_liveness(self) -> None:
        """Refresh ``_joint_states_at`` from ONE sampled /joint_states
        message. Replaces the old standing 494 Hz subscription; see the
        note where this timer is created.

        On failure the timestamp simply stops advancing — exactly what a
        dead driver always looked like — so _check_robot_ready() reports
        precisely what it reported before.
        """
        now_mono = time.monotonic()
        if now_mono < getattr(self, "_liveness_backoff_until", 0.0):
            return
        if self._sample_one_message(JointState, self._joint_states_topic) is None:
            # Same half-up guard as chat_node (2026-08-31): a present-but-
            # silent publisher would otherwise cost the full sample timeout
            # every tick on this node's only thread.
            if self.count_publishers(self._joint_states_topic) > 0:
                prev = getattr(self, "_liveness_backoff_s", 0.0)
                self._liveness_backoff_s = min(10.0, max(1.0, prev * 2.0))
                self._liveness_backoff_until = (
                    now_mono + self._liveness_backoff_s)
            return
        self._liveness_backoff_s = 0.0
        self._liveness_backoff_until = 0.0
        self._joint_states_at = self.get_clock().now().nanoseconds / 1e9

    def _on_robot_program_running(self, msg: Bool) -> None:
        self._robot_program_running = bool(msg.data)

    def _check_robot_ready(self) -> Optional[str]:
        """Return None if the robot can accept motion right now, otherwise
        a short narration explaining why it can't."""
        now = self.get_clock().now().nanoseconds / 1e9
        if not self._joint_states_at:
            return "I don't see the robot driver. Is the UR driver running?"
        age = now - self._joint_states_at
        if age > self._js_max_age_s:
            return (
                f"The robot driver looks frozen. No joint state for "
                f"{age:.1f} seconds."
            )
        # Controller auto-heal (2026-08-26, Majd: "it didn't move
        # because the controller wasn't working" — the E-Stop and
        # protective stops leave scaled_joint_trajectory_controller
        # INACTIVE while the program runs; a human had to notice).
        # Fire-and-forget activate: harmless when already active.
        try:
            if not hasattr(self, "_ctrl_switch_cli"):
                from controller_manager_msgs.srv import SwitchController
                self._SwitchController = SwitchController
                self._ctrl_switch_cli = self.create_client(
                    SwitchController, "/controller_manager/switch_controller")
            now_heal = now
            if (self._ctrl_switch_cli.service_is_ready()
                    and now_heal - getattr(self, "_ctrl_heal_at", 0.0) > 10.0):
                self._ctrl_heal_at = now_heal
                req = self._SwitchController.Request()
                req.activate_controllers = ["scaled_joint_trajectory_controller"]
                req.strictness = 1  # BEST_EFFORT — no-op if already active
                self._ctrl_switch_cli.call_async(req)
                self.get_logger().info(
                    "controller auto-heal: best-effort activate of "
                    "scaled_joint_trajectory_controller dispatched.")
        except Exception as exc:
            self.get_logger().warn(f"controller auto-heal failed: {exc}")
        if self._robot_program_running is False:
            # Only report this if we've actually heard False (the topic
            # exists). If we've never heard the topic at all
            # (_robot_program_running is None) we don't know, so we let
            # motion proceed.
            return (
                "The robot program isn't running. Press play on the teach "
                "pendant or remote control."
            )
        return None

    def _spoken_name_for_index(self, index: int, fallback_label: str) -> str:
        """Human name for a detection: the cap-colour product for vitamin
        tubes ("Multivitamin", "Vitamin B12", "Vitamin C"), else the
        label. Majd 2026-08-25: narrating the raw detector class said
        "vitamin c" for every tube."""
        try:
            det = self._detections[index]
            cap = str(det.get("cap_colour") or "").lower()
            product = {
                "yellow": "Vitamin C", "orange": "Vitamin C",
                "red": "Vitamin B12", "purple": "Vitamin B12",
                "blue": "Multivitamin",
            }.get(cap)
            if product:
                return product
        except Exception:
            pass
        return fallback_label.replace("_", " ")

    # ── detections cache + spatial selection ─────────────────────────────

    def _on_detections(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data or "[]")
        except json.JSONDecodeError:
            return
        if not isinstance(payload, list):
            return
        self._detections = payload
        self._detections_at = self.get_clock().now().nanoseconds / 1e9

    def _scene_is_fresh(self) -> bool:
        if not self._detections_at:
            return False
        now = self.get_clock().now().nanoseconds / 1e9
        return (now - self._detections_at) <= self._scene_max_age_s

    def _resolve_index_by_position(
        self, label: str, position: str,
        colour: str = "", ordinal: int = 0, ordinal_from: str = "left",
    ) -> Optional[int]:
        """Return the detection index matching the requested qualifiers
        among detections sharing the resolved label, or None.

        Accepted qualifiers (Hamid-schema integration, 2026-08-20):
            spatial:  "left", "leftmost", "right", "rightmost", "middle"
            size:     "biggest", "largest", "smallest"
            colour:   matches the pipeline's per-detection colour name
                      ("the red one") — needs the pipeline to publish
                      'colour' in /yolo/all_detections
            ordinal:  1-based count along x ("the second from the left"),
                      direction from ordinal_from ("left"/"right")
        """
        if not self._scene_is_fresh():
            return None

        # Detection labels in /yolo/all_detections are SIMPLIFIED display
        # names ('vitamin_tube'), but the label we're handed may be the
        # raw class name ('vitamin_c'). Translate to compare apples to
        # apples.
        wanted = self._to_display_label(label).lower()
        # Collect (original_index, x_centre, area) tuples for matches.
        matches: list[tuple[int, float, float]] = []
        for i, det in enumerate(self._detections):
            det_label = str(det.get("label") or det.get("display") or "").lower()
            if det_label != wanted:
                continue
            if colour:
                want = colour.lower()
                cap = str(det.get("cap_colour") or "").lower()
                body = str(det.get("colour") or "").lower()
                # Cap colour outranks body colour: the vitamin tubes are
                # identical except for their caps (Majd, 2026-08-21).
                # Lighting aliases (2026-08-25, measured live): in dim
                # light the yellow C-cap reads orange (hue ~10) and the
                # pink B12-cap can drift purple — accept the neighbour
                # colour so the match survives lighting changes.
                _alias = {
                    "yellow": {"yellow", "orange"},
                    "orange": {"orange", "yellow"},
                    "red": {"red", "purple"},
                    "purple": {"purple", "red"},
                }
                accept = _alias.get(want, {want})
                if cap or body:
                    if cap not in accept and body not in accept:
                        continue
            bbox = det.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            try:
                x1, y1, x2, y2 = (float(bbox[k]) for k in range(4))
            except (TypeError, ValueError, IndexError):
                continue
            x_centre = (x1 + x2) * 0.5
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            matches.append((i, x_centre, area))

        if not matches:
            return None
        if len(matches) == 1:
            return matches[0][0]

        if ordinal:
            matches.sort(key=lambda p: p[1],
                         reverse=(ordinal_from == "right"))
            if 1 <= ordinal <= len(matches):
                return matches[ordinal - 1][0]
            return None

        size_qualifiers = {"biggest", "largest", "smallest"}
        if position in size_qualifiers:
            matches.sort(key=lambda p: p[2])  # ascending area
            if position == "smallest":
                return matches[0][0]
            return matches[-1][0]  # biggest / largest

        matches.sort(key=lambda p: p[1])  # ascending x = left to right
        if position in ("left", "leftmost"):
            return matches[0][0]
        if position in ("right", "rightmost"):
            return matches[-1][0]
        if position in ("middle", "centre", "center"):
            return matches[len(matches) // 2][0]
        # Colour-only qualifier with several same-colour matches: take the
        # first (leftmost) rather than failing the pick.
        if colour:
            matches.sort(key=lambda p: p[1])
            return matches[0][0]
        return None

    def _count_visible_for_label(self, label: str) -> int:
        """Return how many detections match the (display) translated
        `label`. Used as a hard backstop for the multi-instance
        disambiguation rule — the LLM is unreliable about NOT firing
        the tool when it should ask first.

        Uses a generous freshness window (3× the normal scene_max_age_s)
        because the pipeline can pause publishing /yolo/all_detections
        when it's in selection-suspend mode, leaving us with a slightly
        stale cache right at the moment the user fires the pick. If
        even the generous window misses, we err on the side of refusing
        the pick — better one false refusal than a wrong-tube pick.
        """
        wanted = self._to_display_label(label).lower()
        now = self.get_clock().now().nanoseconds / 1e9
        age = (now - self._detections_at) if self._detections_at else 1e9
        if age > max(self._scene_max_age_s * 3.0, 10.0):
            # Truly stale (>10s old). We don't know — bail with 0 so
            # the normal pick path runs. The pipeline's own selection
            # cycle will get to do the right thing.
            return 0
        n = 0
        for det in self._detections:
            det_label = str(det.get("label") or det.get("display") or "").lower()
            if det_label == wanted:
                n += 1
        return n

    def _resolve_index_for_label(self, label: str) -> Optional[int]:
        """Return the first detection index matching `label`, or None if
        the scene is stale or no detection matches. Used for label-only
        picks (no spatial qualifier) so we can publish an index that
        unblocks the grasp pipeline's manual-wait loop immediately."""
        if not self._scene_is_fresh():
            return None
        for i, det in enumerate(self._detections):
            det_label = str(det.get("label") or det.get("display") or "").lower()
            if det_label == label.lower():
                return i
        return None

    def _executor_pause_open(self) -> bool:
        """True while the executor is parked at one of its three
        interactive pauses: destination ('paused'), release
        ('awaiting_release') or home-confirm ('awaiting_home_confirm').
        These flags are already maintained by _on_interactive_status."""
        return bool(
            self._pick_awaiting_destination
            or self._pick_awaiting_release
            or getattr(self, "_pick_awaiting_home", False)
        )

    def _claim_executor_resume(
        self, dest: str, why: str, *, force: bool = False
    ) -> bool:
        """Publish EXACTLY ONE resume token per open executor pause.

        Returns True if the token went out, False if it was swallowed.

        The pause is CLOSED the instant a token is published — locally,
        under a lock, without waiting for the executor's async status
        event. A second resume-style intent arriving milliseconds later
        therefore finds no open pause and is dropped with a logged
        reason instead of being fired into a robot that already moved on.

        ``force`` is only for the documented in-queue anti-stranding
        fallback (a confirm_drop while busy_kind=='pick' but no flag set,
        i.e. the router missed the status event). It bypasses the
        flags-are-clear test but NOT the already-claimed test: once a
        token has gone out for this pause, nothing publishes a second.
        """
        # DISARMED 2026-08-28 after adversarial review. The guard closed
        # the pause locally at claim time, which broke two real flows:
        #  * the executor's OWN documented early-home path (it deliberately
        #    consumes a 'home' that arrives during the 3 s post-release
        #    sleep, i.e. "open the gripper" then "go home" as two
        #    utterances) became unreachable at every timing — the operator
        #    was told "I did not do that" while the robot was in fact
        #    still going to act;
        #  * for POINTING / debug-window picks _busy_kind is "" , so
        #    clearing _pick_awaiting_destination at claim time turned a
        #    later utterance into a REAL MoveToNamedPose goal while the
        #    executor was holding the object at the drop pose — two motion
        #    commanders on one arm. Measured with the real dispatch code.
        # The original double-token bug it targeted is real but far less
        # dangerous (one resume is ignored), so we take that instead until
        # the guard can be rewritten to key on the executor's own status
        # events rather than local flags.
        # DISARMED: publish exactly as the pre-patch code did — no local
        # pause closing, no swallowing. The flags stay owned by the
        # executor's status events, which is what the pointing-pick and
        # early-home flows depend on.
        _guard_disarmed = True
        if _guard_disarmed:
            self.get_logger().info(f"Resume {dest!r} published ({why}).")
            self._publish_executor_resume(dest)
            return True
        with self._pause_lock:
            open_now = self._executor_pause_open()
            if not open_now and not (force and not self._resume_claimed):
                self.get_logger().warn(
                    f"SWALLOWED resume {dest!r} ({why}): no executor pause "
                    f"is open "
                    f"(awaiting dest={self._pick_awaiting_destination} "
                    f"release={self._pick_awaiting_release} "
                    f"home={getattr(self, '_pick_awaiting_home', False)}"
                    f", already claimed={self._resume_claimed} by "
                    f"{self._last_resume_dest!r} "
                    f"{max(0.0, time.monotonic() - self._last_resume_mono):.3f}s "
                    f"ago). Publishing it would reach an executor that is "
                    f"no longer waiting and do nothing, so it is dropped "
                    f"— NOT queued: a stale resume replayed later is the "
                    f"bug this guard exists for."
                )
                return False
            # Close the pause BEFORE publishing so a concurrent intent on
            # another callback thread cannot slip a second token through.
            self._pick_awaiting_destination = False
            self._pick_awaiting_release = False
            self._pick_awaiting_home = False
            self._resume_claimed = True
            self._last_resume_dest = dest
            self._last_resume_mono = time.monotonic()
        self.get_logger().info(
            f"Resume {dest!r} published ({why}"
            f"{'; FORCED, flags were desynced' if (force and not open_now) else ''})"
            f"; executor pause now CLOSED."
        )
        self._publish_executor_resume(dest)
        return True

    def _narrate_ignored_resume(self, action: str, dest: str) -> None:
        """Say something TRUE after a resume token was swallowed.

        Fix (A) backstop: the system must never imply it did something it
        did not do — but silence is just as bad here, because the
        operator asked for something and nothing happened. So we say
        exactly that, and name the object still in the gripper when we
        know of one.
        """
        wants_open = ("release" in (dest or "")) or action in (
            "open_gripper", "open", "release", "release_home", "drop_held",
            "let_go", "let go",
        )
        if wants_open:
            held = (self._held_object_label or "").replace("_", " ")
            if held:
                self._narrate(
                    "I did not open the gripper. That came in right after "
                    "another command and the robot was no longer waiting. "
                    f"I'm still holding the {held}."
                )
            else:
                self._narrate(
                    "I did not open the gripper. There was nothing "
                    "waiting to be released."
                )
            return
        self._narrate(
            "I did not do that. The robot was no longer waiting for an "
            "answer, so nothing happened. Say it again and I'll run it."
        )

    def _publish_executor_resume(self, dest: str) -> None:
        """Push a destination string to the LLM grasp executor's resume
        topic. Called from the immediate (bypass-queue) path in
        _on_intent when a pick is paused, AND from the legacy in-queue
        path in _dispatch for backward compat."""
        try:
            self._executor_resume_pub.publish(String(data=dest))
        except Exception as exc:
            self.get_logger().warn(f"Failed to publish resume: {exc}")

    def _publish_select_index(self, index: int) -> None:
        """Publish a spatial-pick selection to the pipeline.

        When the detection behind `index` is still in our cache, send it as
        label+bbox JSON on /yolo/select_detection so the pipeline re-resolves
        it against its CURRENT detection list (indices are not stable across
        frames). Raw Int32 index only as a fallback."""
        det = None
        try:
            if 0 <= int(index) < len(self._detections):
                det = self._detections[int(index)]
        except Exception:
            det = None
        label = str((det or {}).get("label") or (det or {}).get("display") or "")
        bbox = (det or {}).get("bbox") or []
        if label and len(bbox) == 4:
            payload = json.dumps(
                {"index": int(index), "label": label, "bbox": bbox}
            )
            self._select_det_pub.publish(String(data=payload))
            self.get_logger().info(
                f"Published spatial pick det (index={index}, label='{label}') "
                "to /yolo/select_detection for live re-resolution."
            )
            return
        msg = Int32()
        msg.data = int(index)
        self._select_index_pub.publish(msg)
        self.get_logger().info(
            f"Published index={index} to /yolo/select_index for spatial pick."
        )

    # ── parameter setting ────────────────────────────────────────────────

    def _set_pipeline_bool(self, param_name: str, value: bool) -> None:
        """Set a bool parameter on /yolo_live_select_grasp at runtime.

        Used by the enable_multiview / disable_multiview voice intents
        to toggle ``registration_enabled`` live — no pipeline restart.
        """
        cli_name = f"/{self._target_node}/set_parameters"
        cli = self.create_client(SetParameters, cli_name)
        if not cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(
                f"{cli_name} not available; cannot set {param_name}."
            )
            return
        request = SetParameters.Request()
        param = ParameterMsg()
        param.name = param_name
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_BOOL
        param.value.bool_value = bool(value)
        request.parameters = [param]
        future = cli.call_async(request)

        def _done(fut, name=param_name, val=value):
            try:
                resp = fut.result()
            except Exception as exc:
                self.get_logger().warn(f"{name} set failed: {exc}")
                return
            if resp and resp.results and resp.results[0].successful:
                self.get_logger().info(
                    f"Set {name}={val} on /{self._target_node}."
                )
            else:
                reason = (
                    resp.results[0].reason if resp and resp.results else "?"
                )
                self.get_logger().warn(
                    f"{name} refused by pipeline: {reason}"
                )

        future.add_done_callback(_done)

    def _set_executor_drop_target(self, drop_pose: str) -> None:
        """Set drop_target_name on the LLM grasp executor so the next pick
        cycle releases at the requested destination. Used by
        pick_and_place compound intents. Soft-fails if the executor is
        down — pick still runs at executor default."""
        self._set_executor_param("drop_target_name", drop_pose, kind="string")

    def _set_executor_wait_for_destination(self, wait: bool) -> None:
        """Toggle the interactive-drop pause on the LLM executor. True =
        executor pauses after Step 3 (close gripper) and waits for a
        destination on the resume topic. False = executes lift+drop
        straight through using the configured drop_target_name."""
        self._set_executor_param("wait_for_destination", wait, kind="bool")

    def _set_executor_param(self, name: str, value, kind: str) -> None:
        cli_name = "/grasp_executor_lin_llm/set_parameters"
        cli = self.create_client(SetParameters, cli_name)
        if not cli.wait_for_service(timeout_sec=0.5):
            self.get_logger().warn(
                f"{cli_name} not available; cannot set {name}={value!r}."
            )
            return
        request = SetParameters.Request()
        param = ParameterMsg()
        param.name = name
        param.value = ParameterValue()
        if kind == "string":
            param.value.type = ParameterType.PARAMETER_STRING
            param.value.string_value = str(value)
        elif kind == "bool":
            param.value.type = ParameterType.PARAMETER_BOOL
            param.value.bool_value = bool(value)
        else:
            self.get_logger().warn(f"Unknown executor param kind: {kind}")
            return
        request.parameters = [param]
        future = cli.call_async(request)

        def _done(fut, k=name, v=value):
            try:
                resp = fut.result()
            except Exception as exc:
                self.get_logger().warn(f"{k} set failed: {exc}")
                return
            if resp and resp.results and resp.results[0].successful:
                self.get_logger().info(
                    f"Set {k}={v!r} on grasp_executor_lin_llm."
                )
            else:
                reason = (
                    resp.results[0].reason if resp and resp.results else "?"
                )
                self.get_logger().warn(f"{k} refused by executor: {reason}")

        future.add_done_callback(_done)

    # Some YOLO class names get remapped to friendlier display labels in
    # grasp_pipeline.geometry_utils._simplify_object_name. The pipeline's
    # selection cycle matches auto_select_label against those DISPLAY
    # names, not the raw class names. We have to apply the same mapping
    # here or the pipeline will warn "auto_select_label='vitamin_c' not
    # found in ['vitamin_tube',...]" and silently fall back to interactive.
    # Keep this dict in sync with geometry_utils._simplify_object_name.
    _LABEL_DISPLAY_MAP = {
        "this is food pink": "pink food",
        "green pringles": "pringles",
        "astra": "red astra",
        "this is food": "orange food",
        "red bull": "lightblue redbull",
        "vitamin_c": "vitamin_tube",
    }

    def _to_display_label(self, label: str) -> str:
        return self._LABEL_DISPLAY_MAP.get(label.lower().strip(), label)

    def _set_target_label(self, label: str) -> None:
        if label:
            self._open_execution_gate()
        # Service may not be up yet (grasp_pipeline still starting). Don't block forever.
        if not self._set_params_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                f"/{self._target_node}/set_parameters not available; cannot set label."
            )
            # Only narrate the failure if this is an actual pick request
            # (i.e. label is non-empty). Clearing on stop should be silent
            # so we don't trigger an echo cascade.
            if label:
                self._narrate(
                    "I can't pick anything because the grasp pipeline isn't running."
                )
            return

        # Pipeline matches against simplified DISPLAY names, not raw class
        # names. Translate before sending or the pipeline will reject e.g.
        # 'vitamin_c' even though it sees 'vitamin_tube' on the table.
        display_label = self._to_display_label(label) if label else ""

        # SYNCHRONOUS state update — must happen BEFORE the async
        # set-parameter call dispatches, because the pipeline can emit a
        # new cycle_id (and the cycle-start narration can fire) before
        # the async callback returns. If we waited for the callback, the
        # cycle-start narration would still use the previous pick's label
        # and Morty would say the wrong object name.
        if display_label:
            self._last_target_label = display_label
            self._last_target_label_lock_cycle = self._last_cycle_id
            self._last_target_label_lock_t = time.monotonic()
        else:
            self._last_target_label = ""
            self._last_target_label_lock_cycle = -1

        request = SetParameters.Request()
        param = ParameterMsg()
        param.name = self._label_param
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_STRING
        param.value.string_value = display_label
        request.parameters = [param]

        future = self._set_params_client.call_async(request)
        future.add_done_callback(
            lambda fut, lbl=display_label: self._on_set_param_done(fut, lbl)
        )

    def _on_set_param_done(self, fut, label: str) -> None:
        """Confirmation-only callback. _last_target_label was already
        updated synchronously in _set_target_label so the cycle-start
        narration could use it. Here we just log success/failure and
        narrate user-facing errors."""
        try:
            response = fut.result()
        except Exception as exc:
            self.get_logger().error(f"set_parameters call failed: {exc}")
            self._narrate("I could not set the target on the grasp pipeline.")
            return
        if response is None or not response.results:
            self.get_logger().error("set_parameters returned no result.")
            return
        ok = bool(response.results[0].successful)
        if not ok:
            reason = response.results[0].reason
            self.get_logger().warn(f"set_parameters refused: {reason!r}")
            self._narrate("The grasp pipeline refused that target.")
            return
        if label:
            self.get_logger().info(f"Set {self._label_param}={label!r} on /{self._target_node}.")
        else:
            self.get_logger().info(f"Cleared {self._label_param} on /{self._target_node}.")

    # ── stop / cancel ────────────────────────────────────────────────────

    def _handle_stop(self) -> None:
        """Cancel everything intent_router currently owns AND abort the
        executor's pick cycle (candidate evaluation, approach, pauses)."""
        try:
            self._executor_abort_pub.publish(String(data="stop"))
            self._traj_stop_pub.publish(String(data="stop"))
            # Kill an in-flight PIPELINE cycle too (segmentation/GraspGen
            # phase) — the executor abort alone let those finish and the
            # arriving candidates cleared the stop (trial 1001).
            self._pipeline_abort_pub.publish(String(data="stop"))
        except Exception as exc:
            self.get_logger().warn(f"Executor abort publish failed: {exc}")
        cancelled_anything = False
        if self._active_gripper_handle is not None:
            try:
                self._active_gripper_handle.cancel_goal_async()
                self.get_logger().info("Cancelled in-flight gripper goal.")
                cancelled_anything = True
            except Exception as exc:
                self.get_logger().warn(f"Gripper cancel failed: {exc}")
        if self._active_named_pose_handle is not None:
            try:
                self._active_named_pose_handle.cancel_goal_async()
                self.get_logger().info("Cancelled in-flight named-pose goal.")
                cancelled_anything = True
            except Exception as exc:
                self.get_logger().warn(f"Named-pose cancel failed: {exc}")
        if self._clear_on_stop:
            self._set_target_label("")

        if cancelled_anything:
            self._narrate("Cancelled the motion.", interrupt=True)
        else:
            # chat_node already speaks an instant "Stopping." when the
            # intent fires — narrating "no motion in progress" here too
            # stacked three spoken stop lines per command (Majd,
            # 2026-08-12). Log-only.
            self.get_logger().info("Stop: nothing was in progress.")

    # ── gripper ──────────────────────────────────────────────────────────

    def _send_gripper(self, command: int, narration: str) -> None:
        if not self._gripper_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("/gripper action server not available.")
            self._narrate(
                "I can't move the gripper because the robot driver isn't running."
            )
            self._mark_busy_done("gripper")
            return
        if narration:
            self._narrate(narration)
        # Remembered so _on_gripper_result can narrate the OUTCOME.
        self._last_gripper_command = int(command)
        self.get_logger().info(f"Sending OperateGripper command={command}")

        goal = OperateGripper.Goal()
        goal.command = int(command)
        goal.force_n = float(self._gripper_force)

        send_future = self._gripper_client.send_goal_async(goal)

        def _on_response(fut):
            handle = fut.result()
            if handle is None or not handle.accepted:
                self._narrate("The gripper rejected the command.")
                return
            self._active_gripper_handle = handle
            handle.get_result_async().add_done_callback(self._on_gripper_result)

        send_future.add_done_callback(_on_response)

    def _on_gripper_result(self, fut) -> None:
        self._active_gripper_handle = None
        try:
            wrapper = fut.result()
        except Exception as exc:
            self.get_logger().error(f"Gripper result failed: {exc}")
            self._mark_busy_done("gripper")
            return
        if wrapper is None:
            self._mark_busy_done("gripper")
            return
        result = wrapper.result
        self.get_logger().info(
            f"Gripper done: success={result.success} width={result.final_width_m:.3f} "
            f"object_detected={result.object_detected}"
        )
        # Keep narration short to avoid drowning chat.
        # 2026-08-28: this is now the ONLY place a standalone gripper move
        # gets announced. chat_node used to speak "Opening the gripper."
        # the instant the intent matched — before anything knew whether
        # the gripper would move — which is how Morty announced an open
        # that never happened. The adapter's OperateGripper result is the
        # ground truth, so it does the talking.
        cmd = getattr(self, "_last_gripper_command", -1)
        self._last_gripper_command = -1
        if not result.success:
            self._narrate(f"Gripper command failed. {result.error}".strip())
        elif cmd == OperateGripper.Goal.OPEN:
            self._narrate("The gripper is open.")
        elif cmd == OperateGripper.Goal.CLOSE:
            if getattr(result, "object_detected", False):
                self._narrate("The gripper is closed on the object.")
            else:
                self._narrate("The gripper is closed. There was nothing in it.")
        self._mark_busy_done("gripper")

    # ── named pose ───────────────────────────────────────────────────────

    def _send_named_pose(
        self, pose_name: str, narration: str, velocity_scaling: float = 0.0
    ) -> None:
        if not self._named_pose_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("/move_to_named_pose action server not available.")
            self._narrate(
                "I can't move the arm because the motion server isn't running."
            )
            self._mark_busy_done("named_pose")
            return
        if narration:
            self._narrate(narration)
        self.get_logger().info(
            f"Sending MoveToNamedPose -> {pose_name!r} (velocity_scaling={velocity_scaling:.2f})"
        )

        goal = MoveToNamedPose.Goal()
        goal.pose_name = pose_name
        goal.velocity_scaling = float(velocity_scaling)
        # Remember what we're going to so the completion handler can
        # update /robot/current_pose with the right label.
        self._pending_pose_name = pose_name
        # Clear the latched pose for the DURATION of the move (2026-08-26:
        # "go home" during a look move got "I'm already at home" — the
        # ledger only updated on completion, so mid-motion it actively
        # LIED). Empty label = "in motion / unknown"; chat_node words it
        # accordingly. Completion below publishes the real label.
        self._current_pose_name = ""
        try:
            self._current_pose_pub.publish(String(data=""))
        except Exception:
            pass

        send_future = self._named_pose_client.send_goal_async(goal)
        # IN-FLIGHT WATCHDOG (2026-09-02): a named-pose goal that never
        # returns (the motion server wedged on 'home' today) left the
        # router "busy" for minutes — every later "go home" was dropped
        # as "Dedup: already running" while the arm sat at look-left.
        # No move legitimately takes this long (worst taught leg ~12 s
        # very_slow, blends ~9 s): after the budget, cancel the goal,
        # tell the operator, and free the queue.
        _budget_s = float(getattr(self, "_named_pose_budget_s", 40.0))
        _ticket = getattr(self, "_named_pose_ticket", 0) + 1
        self._named_pose_ticket = _ticket

        def _watchdog():
            if getattr(self, "_named_pose_ticket", 0) != _ticket:
                return  # a newer goal superseded this one
            handle = self._active_named_pose_handle
            if handle is None:
                return  # already finished
            self.get_logger().error(
                f"MoveToNamedPose -> {pose_name!r} gave no result within "
                f"{_budget_s:.0f}s — cancelling and freeing the queue.")
            try:
                handle.cancel_goal_async()
            except Exception:
                pass
            self._active_named_pose_handle = None
            self._pending_pose_name = ""
            self._narrate(
                "That move is taking too long — I'm giving up on it.")
            self._mark_busy_done("named_pose")

        _t = threading.Timer(_budget_s, _watchdog)
        _t.daemon = True
        _t.start()

        def _on_response(fut):
            handle = fut.result()
            if handle is None or not handle.accepted:
                self._narrate("The motion server rejected the request.")
                return
            self._active_named_pose_handle = handle
            handle.get_result_async().add_done_callback(self._on_named_pose_result)

        send_future.add_done_callback(_on_response)

    def _on_named_pose_result(self, fut) -> None:
        self._active_named_pose_handle = None
        try:
            wrapper = fut.result()
        except Exception as exc:
            self.get_logger().error(f"MoveToNamedPose result failed: {exc}")
            self._mark_busy_done("named_pose")
            return
        if wrapper is None:
            self._mark_busy_done("named_pose")
            return
        result = wrapper.result
        if result.success:
            # Silent on success — the LLM already confirmed at the start
            # ("Going home"), and an extra "Arrived" creates extra TTS that
            # the mic re-hears. Failures still narrate so the user knows.
            self.get_logger().info("MoveToNamedPose completed.")
            # Update the latched current-pose topic so chat_node can
            # feed the new location to the LLM next turn.
            if self._pending_pose_name:
                self._current_pose_name = self._pending_pose_name
                try:
                    self._current_pose_pub.publish(
                        String(data=self._current_pose_name)
                    )
                except Exception:
                    pass
                self._pending_pose_name = ""
        else:
            self.get_logger().warn(f"MoveToNamedPose failed: {result.error}")
            self._narrate(f"Could not reach the pose. {result.error}".strip())
            self._pending_pose_name = ""
        self._mark_busy_done("named_pose")

    # ── lifecycle handlers ───────────────────────────────────────────────

    def _on_cycle_id(self, msg: Int64) -> None:
        cycle = int(msg.data)
        if cycle == self._last_cycle_id:
            return
        if cycle < self._last_cycle_id:
            # Cycle ids only ever grow within one pipeline run — a smaller
            # id means the pipeline was restarted. Whatever label lock was
            # taken against the OLD numbering is meaningless now.
            self.get_logger().info(
                "pipeline restarted — clearing the stale target-label lock"
            )
            self._last_target_label_lock_cycle = -1
        self._last_cycle_id = cycle
        if not self._narrate_lifecycle:
            return
        # Don't narrate immediately: the pipeline publishes the chosen
        # label and the cycle_id back-to-back on different topics, and
        # delivery order between topics is not guaranteed — measured
        # 2026-08-20: the label lost the race by 7 ms and Morty said
        # "Starting grasp." with no name. Read the label after a short
        # grace instead; 0.6 s is imperceptible next to the TTS itself.
        def _say_start() -> None:
            # destroy, not cancel (2026-08-27): cancelled Humble timers stay
            # "ready" forever and hot-loop the spin — this leaked one
            # core-burning timer PER GRASP CYCLE.
            self.destroy_timer(timer)
            target = (
                self._last_target_label.replace("_", " ")
                if self._last_target_label else ""
            )
            self.get_logger().info(
                f"cycle_id={cycle} (target={target!r}) — narrating start"
            )
            self._narrate(
                f"Starting grasp on the {target}." if target else "Starting grasp."
            )

        timer = self.create_timer(0.6, _say_start)

    def _on_selected_detection(self, msg: Detection2D) -> None:
        """Update _last_target_label whenever the pipeline emits the
        selected detection. Covers selections that didn't come from a
        voice intent (mediapipe pointing, debug-window click) so the
        cycle-start narration still names the object.

        IMPORTANT: when the user just gave an explicit pick intent, the
        previous pick's detection may still be on the wire. We must NOT
        let it clobber the freshly-set label, or the next cycle-start
        narration will say the wrong object name. _last_target_label is
        "locked" against this overwrite until a new cycle_id arrives.
        """
        try:
            if not msg.results:
                return
            cls = str(msg.results[0].hypothesis.class_id).strip()
            if not cls or cls == self._last_target_label:
                return
            # If the label was locked by a recent intent and no new cycle
            # has started yet, ignore the incoming detection. The lock
            # expires after 45 s — a pipeline restart resets cycle ids,
            # which can leave the ==-comparison true indefinitely.
            lock_fresh = (
                time.monotonic() - self._last_target_label_lock_t
            ) < 45.0
            if lock_fresh and \
               self._last_target_label_lock_cycle == self._last_cycle_id and \
               self._last_target_label:
                self.get_logger().debug(
                    f"Ignoring stale selected detection {cls!r}; current "
                    f"intent label {self._last_target_label!r} is locked "
                    f"until next cycle (cycle_id={self._last_cycle_id})."
                )
                return
            self.get_logger().info(
                f"Selected detection: label={cls!r} (updating "
                f"_last_target_label for narration)."
            )
            self._last_target_label = cls
        except Exception as exc:
            self.get_logger().debug(f"_on_selected_detection: {exc}")

    def _on_slip_event(self, msg: String) -> None:
        """Narrate slip onset / clearance from the LIN executor's slip
        watchdog. Payloads: 'slip_start' / 'slip_clear'. Anything else
        is ignored (forwards-compatible with future event names).
        """
        event = (msg.data or "").strip().lower()
        if not self._narrate_lifecycle:
            return
        if event == "slip_start":
            self._narrate("Slip detected, slowing down.", interrupt=True)
        elif event == "slip_clear":
            self._narrate("Grip restored, resuming.")

    def _on_interactive_status(self, msg: String) -> None:
        """The LLM grasp executor broadcasts its interactive-drop
        lifecycle here. We only act on 'paused' — that's the moment the
        arm is holding the object at the grasp pose and waiting for a
        destination. We speak the prompt then so the user hears it at
        the right instant (not 3s earlier when we DISPATCHED the pick)."""
        data = (msg.data or "").strip()
        if not data:
            return
        if data == "paused":
            # The executor is paused waiting for a destination — this
            # is the authoritative signal that we're in the drop-pause
            # window, regardless of how the pick started (voice intent,
            # mediapipe pointing, or debug-window click). Make sure the
            # router's own flag is in sync so the confirm_drop bypass
            # routes the user's reply correctly.
            self._pick_awaiting_destination = True
            # A NEW pause is open: the one-resume-per-pause window reopens.
            self._resume_claimed = False
            label = self._last_target_label or self._held_object_label or ""
            # The pause fires AFTER a confirmed close — the gripper IS
            # holding the object NOW. Publishing held_object only at
            # cycle end left the LLM claiming "I'm not holding anything"
            # mid-pause and refusing the drop command (Majd, 2026-08-12).
            if label:
                self._set_held_object(label)
            if label:
                human = label.replace("_", " ")
                prompt = (
                    f"Got the {human}. Where should I drop it? "
                    f"Left, right, or home?"
                )
            else:
                prompt = (
                    "Got it. Where should I drop it? Left, right, or home?"
                )
            self._narrate(prompt)
        elif data == "awaiting_release":
            # Make sure the release-pause flag is set even for pointed-
            # at picks that bypassed intent_router's dispatch path.
            self._pick_awaiting_release = True
            # A NEW pause is open: the one-resume-per-pause window reopens.
            self._resume_claimed = False
            # A NEW release pause: the keep-holding verdict is not in yet.
            self._release_skipped_this_cycle = False
            if self._last_target_label:
                self._set_held_object(self._last_target_label)
            # The arm has ARRIVED at the drop pose (executor-driven move)
            # — record it so "which position are you?" answers truthfully.
            dest = getattr(self, "_executor_drop_dest", "")
            self._publish_current_pose(dest or "the drop position")
            # Arm is at the drop pose, gripper still closed. Wait for
            # user to say "release" / "open it" / "give it to me".
            self._pick_awaiting_release = True
            _dest = getattr(self, "_executor_drop_dest", "")
            _spoken = {
                "drop_left_side": "drop left",
                "drop_right_side": "drop right",
                "home": "home",
            }.get(_dest, "drop")
            self._stayed_pose = ""
            # Name what we are holding (Majd 2026-08-31: "should i open
            # the gripper, I'm still holding the object? ... more
            # nature"). Also re-spoken after every REDIRECT, naming the
            # NEW pose, because the executor re-publishes this status on
            # arrival.
            _holding = (self._last_target_label
                        or self._held_object_label or "").replace("_", " ")
            if _holding:
                self._narrate(
                    f"I'm at the {_spoken} position, still holding the "
                    f"{_holding}. Should I open the gripper?"
                )
            else:
                self._narrate(
                    f"I'm at the {_spoken} position. Should I open the gripper?"
                )
        elif data == "releasing":
            # The executor is ABOUT to command the open (v2 ordering,
            # 2026-08-31): progressive tense only — "The gripper is
            # open." waits for the hardware to actually report success.
            self._narrate("Opening the gripper.")
        elif data == "release_failed":
            # The open was commanded and FAILED — the object is still
            # held. Without this line the operator heard "Opening the
            # gripper." and then silence while nothing happened.
            # The pause is over (the executor is in failure recovery),
            # so clear the flag or the next utterance would still be
            # treated as a release answer.
            self._pick_awaiting_release = False
            self._narrate(
                "The gripper did not open — I still have the object."
            )
        elif data.startswith("redirect:"):
            # v2: the executor is carrying the held object to a new
            # destination. Deliberately NO narration (chat_node already
            # acknowledged the command — review finding 14 measured a
            # double announcement) and NO pose bookkeeping here: this
            # status fires BEFORE the motion, and a failed redirect
            # would leave the books claiming a pose the arm never
            # reached (finding 6). The executor publishes an
            # authoritative resumed:<pose> after the motion settles.
            self.get_logger().info(f"Executor redirect in progress: {data!r}")
        elif data == "redirect_failed":
            self._narrate(
                "I couldn't reach that spot — I'm staying here, still "
                "holding the object."
            )
        elif data == "keeping":
            # v2 keep-holding: the arm STAYS at the drop pose with the
            # question open. chat_node already spoke the acknowledgement
            # at intent time, so this is bookkeeping only.
            _kept2 = self._last_target_label or self._held_object_label
            if _kept2:
                self._set_held_object(_kept2)
        elif data == "stopped_holding":
            # Abort-freeze at the release pause: arm parked at the drop
            # pose, gripper closed on the object. Do NOT claim "home".
            self._pick_awaiting_release = False
            _kept3 = self._last_target_label or self._held_object_label
            if _kept3:
                self._set_held_object(_kept3)
            # Adversarial review 2026-08-31 (finding 4): the cycle ends
            # as failed|stopped by your command, and _on_cycle_done's
            # keep-memory guard keys on this flag — without it the held
            # object was wiped while the gripper physically holds it.
            self._release_skipped_this_cycle = True
            dest3 = getattr(self, "_executor_drop_dest", "")
            if dest3:
                self._publish_current_pose(dest3)
        elif data == "released":
            self._pick_awaiting_release = False
            # Majd 2026-08-31: "Opening the gripper." + "The gripper is
            # open." — "one of them not both". The progressive line (on
            # 'releasing') stays as the immediate response; success is
            # implied by the completion narration that follows seconds
            # later, and FAILURE still speaks loudly via
            # 'release_failed'. This branch keeps the bookkeeping only.
            # The gripper OPENED — this cycle ends empty, so cycle_done
            # must NOT re-arm the held-object memory (2026-08-28 loop).
            self._release_skipped_this_cycle = False
            # The object left the gripper — clear the held-object memory
            # (2026-08-25: without this the LLM kept believing it was
            # holding the tube after dropping it). The executor now
            # returns to home (reset target).
            self._set_held_object("")
            # NOTE: do NOT publish "home" here — the arm is still AT the
            # drop pose when the gripper opens, and may even stay there
            # (2026-08-25: the LLM claimed "you're already home" while
            # the arm stood at drop-left). cycle_done/stayed: publish
            # the true pose.
        elif data == "release_skipped":
            # Keep-holding carry-home (release-pause timeout, or an
            # unrecognized answer): gripper stays closed while the arm
            # returns home. Narrate BEFORE the ride — review finding 10:
            # v2's stay-open pause made this a delayed autonomous move
            # up to 30 min after the last word, with zero warning.
            self._pick_awaiting_release = False
            self._release_skipped_this_cycle = True
            _kept = self._last_target_label or self._held_object_label
            if _kept:
                self._set_held_object(_kept)
            self._narrate(
                "I'll keep holding it and take it back home."
            )
            self._publish_current_pose("home")
        elif data == "no_grasp":
            self._pick_awaiting_destination = False
            # The executor now lifts clear and ASKS before homing
            # (2026-08-26) — don't promise "going back home" here.
            self._narrate(
                "The object was moved. I'm not grasping anything."
            )
        elif data == "vitamin_retighten":
            # Mid-drop slip-prevention re-tightening (vitamin tube only).
            self._narrate(
                "Slip is getting detected. I'm holding the object tighter.",
                interrupt=True,
            )
        elif data == "awaiting_home_confirm":
            self._pick_awaiting_home = True
            # A NEW pause is open: the one-resume-per-pause window reopens.
            self._resume_claimed = False
            self._narrate("Should I go back home?")
        elif data.startswith("stayed:"):
            # Arm stayed at the drop pose per the operator's wish.
            self._pick_awaiting_home = False
            self._stayed_pose = data.split(":", 1)[1].strip()
            self._publish_current_pose(self._stayed_pose)
        elif data.startswith("resumed:"):
            # The executor is now moving to the chosen drop pose — track
            # it so /robot/current_pose is true for EXECUTOR-driven moves
            # too (2026-08-25, Majd: "Morty must know what position he
            # is" — it claimed 'home' while standing at drop-left because
            # only router-commanded moves updated the pose).
            self._executor_drop_dest = data.split(":", 1)[1].strip()
        # The LLM's confirm_drop reply already narrates the destination.

    def _on_cycle_done(self, msg: String) -> None:
        data = (msg.data or "").strip()
        if not data:
            return
        outcome = data.split("|", 1)[0].lower()
        self.get_logger().info(f"cycle_done received: {data!r} (outcome={outcome})")
        if self._narrate_lifecycle:
            # Silent on success — same reasoning as MoveToNamedPose: avoid
            # extra TTS that the mic loops back. Failure still narrates,
            # WITH the executor's reason when it sent one
            # ("failed|<cycle_id>|<reason>", 2026-08-11).
            if outcome == "failed":
                fields = data.split("|", 2)
                reason = fields[2].strip() if len(fields) > 2 else ""
                if reason == "stopped by your command":
                    # Operator STOP at a pause: the arm froze in place,
                    # nothing resets — saying "Resetting." here was a
                    # lie (2026-08-31 narration audit).
                    self._narrate("Stopped. I'm holding position.")
                elif reason:
                    self._narrate(f"The grasp failed: {reason}. Resetting.")
                else:
                    self._narrate("The grasp failed. Resetting.")
            elif outcome == "done":
                self.get_logger().info("Grasp cycle completed successfully.")
                # END-OF-TASK narration (Majd 2026-08-31: "when he go
                # back home he didnt say im at home what should i do
                # next"). Safe to speak now: the word-time echo cut
                # (#272/#274) keeps the mic from looping this back.
                # F3 (review): prefer the stayed pose CARRIED IN THE
                # PAYLOAD ("done|id|stayed:<pose>") — the separate
                # stayed: status message races this one near-
                # deterministically (subscription creation order decides
                # delivery order within a batch) and used to lose.
                stayed = ""
                for _f in data.split("|")[2:]:
                    if _f.startswith("stayed:"):
                        stayed = _f.split(":", 1)[1].strip()
                        break
                if not stayed:
                    stayed = getattr(self, "_stayed_pose", "")
                if stayed:
                    _sp = {
                        "drop_left_side": "drop left",
                        "drop_right_side": "drop right",
                        "home": "home",
                    }.get(stayed, stayed.replace("_", " "))
                    self._narrate(
                        f"Okay — staying at {_sp}. What should I do next?"
                    )
                    self._stayed_pose = ""
                elif self._release_skipped_this_cycle:
                    self._narrate(
                        "I've brought the object back home — still "
                        "holding it. What should I do next?"
                    )
                else:
                    # "back at" reads wrong when the release happened AT
                    # home — the arm never left it (Majd 2026-08-31).
                    if getattr(self, "_executor_drop_dest", "") == "home":
                        self._narrate(
                            "I'm at the home position. What's next?"
                        )
                    else:
                        self._narrate(
                            "Done — I'm back at the home position. "
                            "What's next?"
                        )
            else:
                self.get_logger().info(f"Unknown cycle_done outcome: {data!r}")
        # Update held-object memory so the LLM can resolve "it" / "that
        # one" / "drop it on the left" in the next turn.
        # 2026-08-28: this used to SET held on every successful cycle,
        # which is backwards. The pipeline goes pick → drop → RELEASE →
        # home, so a completed cycle normally ends with an EMPTY gripper
        # ("cycle_done received: 'done|4' (outcome=done)" at
        # 1787941443.35 was immediately followed by "Held-object memory:
        # 'small_bottle'", after which Morty kept insisting "I'm still
        # holding the small bottle"). The only cycle that really ends
        # holding is the KEEP-HOLDING path, where the executor published
        # 'release_skipped' and carried the object home.
        if outcome == "done" and self._release_skipped_this_cycle:
            _kept = self._last_target_label or self._held_object_label
            if _kept:
                self._set_held_object(_kept)
        elif self._release_skipped_this_cycle:
            # An operator STOP at the release pause also publishes
            # 'release_skipped' and then fails the cycle — but the gripper
            # is genuinely STILL CLOSED on the object at the drop pose.
            # Clearing here would recreate exactly the bug Majd hit from
            # the other side ("you are not holding it anymore" / "I'm
            # still holding it"), so keep the memory (2026-08-28).
            _kept = self._last_target_label or self._held_object_label
            if _kept:
                self._set_held_object(_kept)
        elif outcome in ("done", "failed"):
            # done  ⇒ the gripper opened at the drop pose; it is empty.
            # failed ⇒ the arm did not end up holding anything either.
            self._set_held_object("")
        # Position bookkeeping (2026-08-25): a normal cycle end leaves the
        # arm at the reset pose (home). The one exception is the operator
        # freeze-stop, which leaves the arm wherever it was — publish
        # empty so the LLM falls back to the nearest-joint-pose lookup.
        try:
            _reason = data.split("|", 2)[2].strip().lower() if data.count("|") >= 2 else ""
        except Exception:
            _reason = ""
        if "stopped by your command" in _reason:
            self._publish_current_pose("")
        elif getattr(self, "_stayed_pose", ""):
            self._publish_current_pose(self._stayed_pose)
            self._stayed_pose = ""
        else:
            self._publish_current_pose("home")
        # Clear auto_select_label on the pipeline so it doesn't auto-pick
        # the SAME object again the moment the pipeline starts its next
        # selection cycle (after the executor reset). Without this, the
        # robot picks the small_bottle, drops it, goes home, sees it back
        # on the table, and immediately picks it AGAIN — endless loop.
        self._set_target_label("")
        # Interactive drop is over (success, fail, or timeout).
        self._pick_awaiting_destination = False
        self._pick_awaiting_release = False
        self._pick_awaiting_home = False
        self._resume_claimed = False
        self._release_skipped_this_cycle = False
        # The finished cycle's label must not name the NEXT pick. The
        # _set_target_label("") call above already clears it — but only
        # when the pipeline's set_parameters service answered; it early-
        # returns when the pipeline is down/restarting. Clear here
        # unconditionally.
        self._last_target_label = ""
        self._last_target_label_lock_cycle = -1
        # Pick completed (success or fail) — release the queue so the
        # next queued intent can proceed.
        self._mark_busy_done("pick")

    def _open_execution_gate(self) -> None:
        """Voice picks are autonomous by definition (2026-08-26: with the
        GUI's auto-execute unchecked, voice cycles buffered 'waiting for
        manual approval' forever while the pipeline re-fired the label —
        the small-bottle loop). The GUI checkbox governs CLICK picks;
        voice picks open the gate themselves."""
        try:
            if not hasattr(self, "_gate_pub"):
                from rclpy.qos import QoSProfile, DurabilityPolicy
                self._gate_pub = self.create_publisher(
                    Bool, "/grasp_executor_lin_llm/execution_enabled",
                    QoSProfile(depth=1,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL))
            self._gate_pub.publish(Bool(data=True))
            self.get_logger().info(
                "Voice pick: execution gate OPENED (auto-execute checkbox "
                "applies to GUI clicks only).")
        except Exception as exc:
            self.get_logger().warn(f"gate open failed: {exc}")

    def _publish_current_pose(self, pose_name: str) -> None:
        """Update the latched /robot/current_pose topic (2026-08-25):
        executor-driven moves (drop poses, post-release reset) now feed
        the same channel router-commanded moves always used, so the LLM
        can answer "which position are you?" truthfully at any time."""
        try:
            self._current_pose_name = pose_name or ""
            self._current_pose_pub.publish(String(data=self._current_pose_name))
        except Exception as exc:
            self.get_logger().warn(f"current_pose publish failed: {exc}")

    def _set_held_object(self, label: str) -> None:
        """Publish the currently-held object label (or "" if empty) so
        chat_node can hand it to the LLM as context for pronoun
        resolution. Topic is latched so a late subscriber still sees the
        last value."""
        try:
            msg = String()
            msg.data = label or ""
            self._held_object_pub.publish(msg)
            self._held_object_label = label or ""
            self.get_logger().info(f"Held-object memory: {label!r}")
        except Exception as exc:
            self.get_logger().warn(f"Held-object publish failed: {exc}")

    def _controller_watchdog(self) -> None:
        """Poll the controller manager; auto-activate the trajectory
        controller whenever it's found inactive. Fixes the UR driver
        regression where controller_stopper deactivates the controller
        on reverse-interface drop and never restores it, leaving the
        operator with 'Controller is not running' errors until they
        manually run `ros2 control set_controller_state`."""
        if not self._list_controllers_cli.service_is_ready():
            return  # controller manager not up yet; quiet poll
        req = ListControllers.Request()
        future = self._list_controllers_cli.call_async(req)
        future.add_done_callback(self._on_list_controllers_done)

    def _on_list_controllers_done(self, fut) -> None:
        try:
            resp = fut.result()
        except Exception as exc:
            self.get_logger().debug(f"list_controllers failed: {exc}")
            return
        if resp is None:
            return
        target_state = ""
        for ctrl in resp.controller:
            if ctrl.name == self._traj_controller_name:
                target_state = ctrl.state
                break
        if not target_state:
            return
        # Log state TRANSITIONS so the operator can see when an
        # auto-recovery actually happened, without flooding on every
        # 2-second poll.
        if target_state != self._last_controller_state:
            self.get_logger().info(
                f"Controller '{self._traj_controller_name}' state: "
                f"{self._last_controller_state!r} -> {target_state!r}"
            )
            self._last_controller_state = target_state
        if target_state != "active":
            self._activate_traj_controller()

    def _activate_traj_controller(self) -> None:
        if not self._switch_controller_cli.service_is_ready():
            return
        req = SwitchController.Request()
        # ros2_humble field names. activate_controllers = list of names to
        # activate; strictness BEST_EFFORT so a momentary inconsistency
        # doesn't error out the whole call.
        req.activate_controllers = [self._traj_controller_name]
        req.deactivate_controllers = []
        req.strictness = SwitchController.Request.BEST_EFFORT
        req.activate_asap = True
        req.timeout = rclpy.duration.Duration(seconds=1.0).to_msg()
        future = self._switch_controller_cli.call_async(req)

        def _done(f, name=self._traj_controller_name):
            try:
                r = f.result()
            except Exception as exc:
                self.get_logger().warn(
                    f"Auto-recovery: switch_controller failed: {exc}"
                )
                return
            if r and r.ok:
                # Log-only: this fires routinely on every driver start
                # (controller comes up inactive) — announcing internal
                # plumbing out loud confused the operator (Majd, 2026-08-12).
                self.get_logger().info(
                    f"Auto-recovery: activated '{name}'."
                )
            else:
                self.get_logger().warn(
                    f"Auto-recovery: switch_controller refused to "
                    f"activate '{name}'."
                )

        future.add_done_callback(_done)

    def _queue_watchdog(self) -> None:
        """Time-out picks that never receive a cycle_done so the queue
        cannot wedge if the executor is offline."""
        if (
            self._busy
            and self._busy_kind == "pick"
            and self._pick_deadline > 0.0
            and time.monotonic() > self._pick_deadline
        ):
            self.get_logger().warn(
                f"Pick watchdog: no cycle_done within {self._pick_timeout_s}s; "
                "advancing the queue."
            )
            self._mark_busy_done("pick")

    # ── narration ────────────────────────────────────────────────────────

    def _narrate(self, text: str, *, interrupt: bool = False) -> None:
        """Speak a status line. Polite by default: waits for whatever is
        currently being spoken (chat reply, other narration) to finish —
        interrupt_current=True chopped Morty's sentences mid-word whenever
        the executor reached a lifecycle event (Majd, 2026-08-12). Pass
        interrupt=True only for time-critical lines (stop, slip)."""
        if not self._speak_feedback or self._speak_client is None or not text:
            return
        if not self._speak_client.server_is_ready():
            if not self._speak_client.wait_for_server(timeout_sec=0.3):
                return
        goal = Speak.Goal()
        goal.text = text
        goal.interrupt_current = bool(interrupt)
        self._speak_client.send_goal_async(goal)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = IntentRouterNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
