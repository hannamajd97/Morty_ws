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
import uuid
from typing import Optional

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from ur3e_fast_observation.latest_message import LatestMessage
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Int32, Int64, String
from vision_msgs.msg import Detection2D

from rcl_interfaces.msg import Parameter as ParameterMsg, ParameterValue, ParameterType
from rcl_interfaces.srv import SetParameters
from controller_manager_msgs.srv import ListControllers, SwitchController
from ur_dashboard_msgs.srv import GetProgramState

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
# 2026-09-25: "reset" is the GUI's "↻ Reset pick" said aloud — end the pick
# and go home keeping any held object, gripper untouched. It is NOT a stop
# (a stop freezes the arm), so it has its own set and its own handler
# (_handle_reset) instead of riding on CLEAR_INTENTS.
RESET_INTENTS = {"reset"}

# 2026-09-25: resume tokens a destination intent maps to, for the early
# destination latch (_destination_token). Values are what the executor's
# destination pause accepts; the keys are every spelling the chat node and
# the LLM tool use for the three answers to "Left, right, or home?".
# 2026-09-29: destinations that mean "release where the arm is" (the
# executor's "here" at the destination question).
_HERE_WORDS = {"here", "in_place", "in place", "same_place", "same place",
               "the same place", "grasp_spot", "where it was", "back"}

_DESTINATION_WORDS = {
    "drop_right": "drop_right", "right": "drop_right",
    "drop right": "drop_right", "drop_right_side": "drop_right",
    "drop_left": "drop_left", "left": "drop_left",
    "drop left": "drop_left", "drop_left_side": "drop_left",
    "home": "home", "go_home": "home",
}

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


def _then_home(action: str, attrs) -> bool:
    """2026-09-28: the intent means "open, then go home" — release_home, or
    an open_gripper / release whose ``then`` is release_home (how the model
    sends "Give me the object and go back home.")."""
    if action == "release_home":
        return True
    then = str(attrs.get("then") or "").strip().lower() if isinstance(attrs, dict) else ""
    return action in ("open_gripper", "release") and then == "release_home"


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
        self.declare_parameter("default_speed", "slow")   # voice moves without a speed word; GUI Motion Apply sets it
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
        # 2026-09-23: a follow-up ("give me the object, open the gripper,
        # then go back home") that arrives while the pick is still moving to
        # the drop pose. The executor clears its resume flag when a pause
        # OPENS, so a token sent earlier is lost and a queued intent came
        # back after the cycle as "not recognised". The plan is kept here
        # and answers the release pause the moment it opens.
        self._followup_plan: str = ""   # "", "release" or "release_home"
        # True from the executor's "paused" (Where should I drop it?) until it
        # starts moving to a destination ("resumed:"), reaches a release pause
        # or ends the cycle. _pick_awaiting_destination stays set through the
        # release pause too, so it cannot tell the two questions apart.
        self._destination_open: bool = False
        # 2026-09-25: the executor's latched task state
        # (/grasp_executor_lin_llm/task_state, JSON, see _on_task_state).
        # None until the first message: while it is None every read below
        # falls back to the legacy flags, so an old executor that never
        # publishes it sees exactly the router of 3f512258.
        self._task: Optional[dict] = None
        self._task_boot: float = -1.0
        self._task_seq: int = -1
        # 2026-09-25: EARLY DESTINATION LATCH (same pattern as
        # _followup_plan). "Put it on the right" said while the grasp is
        # still approaching used to be published at once — and the
        # executor clears its resume slot when the destination pause OPENS,
        # so the answer was lost and Morty asked again after saying "Going
        # to drop right.". The token is kept here instead and answers the
        # pause the moment it opens. Dict {token, cycle, seen, t}; cycle is
        # None until the pick's cycle starts (see _on_task_state).
        self._early_dest: Optional[dict] = None
        # 2026-09-25: True from a destination answer the ROUTER put on the
        # resume topic until the executor's resumed:<t> — that event (not
        # the intent) is what makes the router say "Going to drop right.",
        # so the line is only spoken for an answer the executor accepted.
        self._dest_answer_pending: bool = False
        # 2026-09-25: the newest cycle id that ended (cycle_done), and the
        # cycle id current at the last stop/reset: a "Starting grasp on …"
        # scheduled 0.6 s after a cycle start must not be spoken for a
        # cycle that already ended or was stopped inside that grace.
        self._last_done_cycle_id: int = -1
        self._start_suppressed_upto: int = -1
        # 2026-09-25: True while the router's own plain voice pick (it set
        # wait_for_destination=True) is dispatched — lets a destination said
        # before the executor's cycle has even started (phase still idle)
        # be latched. pick_and_place sets it False: its destination is fixed.
        self._pick_waits_for_destination: bool = False
        # 2026-09-25 (review R1): the side a dispatched pick_and_place was
        # given ("drop_right" / "drop_left" / "home"; "" for a plain pick or
        # an unknown / executor-default destination). A destination said
        # during its approach is never published into no open pause; this
        # tells "the same side again" (nothing changes) from a change the
        # executor can no longer take (see _destination_without_pause).
        self._pick_fixed_side: str = ""
        # 2026-09-25 (review R6): the line the next 'releasing' speaks when
        # a follow-up plan answered the release pause ("Opening the gripper
        # and going home."); "" = the usual "Opening the gripper.". Spoken
        # once, on 'releasing' (the claim used to say it too).
        self._releasing_line: str = ""
        # 2026-09-25 (Majd, live: "Now back home." while the arm was still on
        # its way to the drop-left pose after "Should I go back home?"): a
        # go-home / drop-side move said while the cycle's LAST move runs (all
        # questions answered, gripper empty) is kept here and carried out
        # when that cycle ends — it used to be published to a pause that
        # never opened again and was lost. See _defer_after_cycle.
        self._after_cycle_intent: Optional[dict] = None
        # 2026-09-25: the drop pose (and speed) to go to once the separate
        # move home that _send_drop_pose started has arrived; see there.
        self._home_then: Optional[tuple] = None
        # 2026-09-25: when "I'm still holding the X — where should I put it
        # first?" was last said (monotonic); the executor's refusals of
        # pointing / GUI picks repeat it at most every 15 s.
        self._still_holding_mono: float = -1e9
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
        # 2026-09-25: the executor's reset request (payload = source,
        # "voice" here; the GUI publishes "gui" itself). A reset ends the
        # pick and takes the arm home keeping any held object; the abort
        # above freezes it. See _handle_reset.
        self._executor_reset_topic = "/grasp_executor_lin_llm/reset"
        self._executor_reset_pub = self.create_publisher(
            String, self._executor_reset_topic, 10
        )
        # 2026-09-25: ONE cached client for task_object_label (see
        # _set_task_object_label). _set_executor_param builds a client per
        # call and waits up to 0.5 s for it, which is fine at dispatch but
        # not on every selected detection or cycle_done.
        self._task_label_cli = self.create_client(
            SetParameters, "/grasp_executor_lin_llm/set_parameters"
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
        # 2026-09-18: ONE owner for "can the robot move": this node's
        # _check_robot_ready verdict (latched program flag + the live
        # witnesses that outrank it) is broadcast every watchdog tick so
        # chat_node's spoken 'press Play' reminder and its status block
        # for the LLM say the same thing as the router instead of reading
        # the raw flag themselves.
        self._robot_status_pub = self.create_publisher(
            String, "/intent_router/robot_status", 1)
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
        # Live witnesses that outrank a stale False on that latched topic
        # (see _check_robot_ready): dashboard program state, polled by the
        # controller watchdog only while the flag says False.
        self._dashboard_state: str = ""
        self._dashboard_at: float = 0.0
        self._stale_flag_warned_at: float = 0.0
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
        # 2026-09-22: the pipeline's own verdict when it gives a label up
        # before any cycle starts (object not seen / three failed starts);
        # same grammar as cycle_done, so the same handler frees the pick.
        self.create_subscription(
            String, "/grasp_pipeline/selection_result", self._on_cycle_done, 10)
        # 2026-09-28 (Majd): pointing confirmation. The pipeline holds a
        # POINTED choice and proposes it here; this node asks about it
        # (_on_point_proposal) and forwards the answer (confirm_point /
        # reject_point, or a plain pick of the same kind). The open question
        # is published latched on /robot/open_question, so chat_node answers
        # a yes / no instantly, as it does the executor's own questions.
        self._point_q: Optional[dict] = None
        # (display label, monotonic) of this node's last which-one question
        # (_two_of_line) — a pointed answer to it is worded against it.
        self._clarify: Optional[tuple] = None
        self._point_answer_pub = self.create_publisher(
            String, "/grasp_pipeline/point_answer", 10)
        self._open_q_pub = self.create_publisher(
            String, "/robot/open_question",
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                       reliability=ReliabilityPolicy.RELIABLE,
                       durability=DurabilityPolicy.TRANSIENT_LOCAL))
        self._open_q_pub.publish(String(data=json.dumps({"question": ""})))
        self.create_subscription(
            String, "/grasp_pipeline/point_proposal", self._on_point_proposal, 10)
        # 2026-09-30 (recording contract v1, vla_experiments/recording_contract.py):
        # every command's LIFECYCLE — received, queued / deferred / rejected,
        # accepted (the executing component took it), cycle, completed /
        # failed / cancelled — at the moment this node knows it. The VLA
        # recorder labels steps from the command actually executing instead
        # of from intent arrival (trial 114: look_right and go_home arrived
        # 8 ms apart; go_home took the look's motion). Events only; no
        # routing decision reads them.
        self._cmd_pub = self.create_publisher(
            String, "/robot/command_event",
            QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=100,
                       reliability=ReliabilityPolicy.RELIABLE))
        self._cmd_seq = 0
        self._cmd_boot = round(time.time(), 3)
        self._cmds: "collections.OrderedDict[str, dict]" = collections.OrderedDict()
        self._routing_cmd: Optional[dict] = None   # the command _on_intent is routing now
        self._active_cmd: Optional[dict] = None    # the command _dispatch runs (queue head)
        self._pick_cmd: Optional[dict] = None      # the command whose grasp cycle runs
        self._followup_cmd: Optional[dict] = None  # the command behind _followup_plan
        self._dispatch_narr: Optional[list] = None # lines said while _dispatch runs
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
        # 2026-09-25: the executor's task state — the one owner of "which
        # question is open, what is held, which cycle, how it ended". Latched
        # (the executor publishes it reliable + transient_local), read as a
        # level; see _on_task_state.
        # 2026-09-25 (review R4): kept by name — _live_task() trusts the
        # latched sample only while this topic still has a publisher.
        self._task_state_topic = "/grasp_executor_lin_llm/task_state"
        self.create_subscription(
            String,
            self._task_state_topic,
            self._on_task_state,
            QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
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
        """Entry of /robot/intent: the command's "received" event, then the
        routing below (_route_intent), then — when no route marked the
        command at all — a "rejected" event, so a silently dropped intent
        never looks like an executed one (recording contract v1)."""
        try:
            intent = json.loads(msg.data or "")
        except (TypeError, ValueError):
            intent = None
        if not isinstance(intent, dict):
            self._route_intent(msg)
            return
        cmd = self._cmd_from_intent(intent)
        self._cmd_event(cmd, "received")
        self._routing_cmd = cmd
        try:
            self._route_intent(String(data=json.dumps(intent)))
        finally:
            self._routing_cmd = None
            if cmd.get("state") == "received":
                self._cmd_event(cmd, "rejected", "dropped by the router without a route")

    def _route_intent(self, msg: String) -> None:
        raw = (msg.data or "").strip()
        if not raw:
            return
        try:
            intent = json.loads(raw)
        except json.JSONDecodeError as exc:
            self.get_logger().warn(f"Could not parse intent JSON: {exc}; raw={raw!r}")
            return

        action = str(intent.get("action_intent") or "").strip().lower()

        # 2026-09-25: reset (voice, the LLM, or the GUI's "↻ Reset pick"
        # with the new executor) goes home keeping the object — checked
        # BEFORE CLEAR_INTENTS because it must never become a stop (the GUI
        # used to send "stop" after its reset; the router turned that into
        # an abort and the arm froze instead of going home).
        if action in RESET_INTENTS:
            _rattrs = intent.get("object_attributes")
            _src = str(_rattrs.get("source") or "") if isinstance(_rattrs, dict) else ""
            self._cmd_event(self._routing_cmd, "accepted", route="reset")
            self._handle_reset(_src or "voice")
            self._cmd_event(self._routing_cmd, "completed", "reset sent to the executor")
            return

        # Stop is special: it must execute immediately and clear the queue,
        # not wait behind whatever's currently running.
        if action in CLEAR_INTENTS:
            # 2026-09-25 (review R7): the GUI's E-stop publishes its own
            # executor /abort and tags its stop intent source "gui"; this
            # path then sends no second abort (see _handle_stop).
            _sattrs = intent.get("object_attributes")
            _ssrc = str(_sattrs.get("source") or "") if isinstance(_sattrs, dict) else ""
            self._cmd_event(self._routing_cmd, "accepted", route="stop")
            self._stop_path(_ssrc)
            self._cmd_event(self._routing_cmd, "completed", "stop sent")
            return

        # 2026-09-28: the operator confirmed the gripper is empty although
        # the executor's record says holding (an empty tight close sets the
        # grasp pin too). _not_holding_action says what that means now; an
        # open_gripper continues down the normal path.
        if action == "not_holding":
            action = self._not_holding_action()
            if not action:
                return
            intent = {"action_intent": action, "object_intent": None, "object_attributes": None}

        # 2026-09-28 (Majd): the answer to a pointing question (see
        # _on_point_proposal). A plain pick of the proposed kind answers it
        # too; any other pick goes the normal way and the pipeline drops the
        # proposal for it (superseded).
        if action in ("confirm_point", "reject_point"):
            if not self._answer_point(action == "confirm_point", "answer"):
                self.get_logger().info(f"{action}: no pointing question is open — ignored.")
                self._cmd_event(self._routing_cmd, "rejected", "no pointing question is open")
            return
        if (self._point_q is not None and action in PICK_INTENTS
                and self._pick_confirms_point(intent)):
            self._answer_point(True, f"a pick of the {self._point_q['label']}")
            return
        if (self._point_q is not None and action == "pick_and_place"
                and self._pick_confirms_point(intent, with_destination=True)):
            # "Yes, grab it and put it on the left." — the pick carries a
            # destination, so it goes the normal way (the pipeline drops the
            # proposal for its label); the pointed instance's position is
            # added so the label does not take the first of its kind.
            _pa = dict(intent.get("object_attributes") or {})
            _where = self._rank_word(self._point_q["label"], self._point_q.get("bbox"))
            if _where:
                _pa["position"] = _where
            intent = {**intent, "object_attributes": _pa,
                      "object_intent": intent.get("object_intent") or self._point_q["label"]}
            self.get_logger().info(
                f"pick_and_place at the pointing question: the pointed "
                f"{self._point_q['label']} ({_where or 'only one'}).")

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
        # 2026-09-25: the question comes from the executor's task state
        # when it has reported (_question_open), else from the legacy flag.
        if self._question_open("release"):
            # 2026-09-25 (Majd, live test): a SIDE named at "Should I open the
            # gripper?" is a REDIRECT — "No, go to drop left." (confirm_drop
            # drop_left, no then) was sent as drop_left+release and the
            # executor carried the object there AND OPENED without asking.
            # Only an explicit release (then = release / release_home, or
            # open_gripper / release WITH a side) becomes drop_<side>+release
            # (carry_release: opens on arrival, or in place when already
            # there); a bare side is the plain redirect token drop_<side>:
            # the executor carries it there and asks the release question
            # again. See _release_side_token. Only with a live task state
            # (review R4); without one today's paths below run unchanged.
            _tok = (self._release_side_token(action, intent)
                    if self._live_task() is not None else "")
            if _tok:
                if "+" not in _tok and self._arm_at_side(_tok):
                    # 2026-09-25: the executor ignores a redirect to where the
                    # arm already is (stale-echo rule) and keeps waiting in
                    # silence — say so and re-ask instead of claiming.
                    _sp = {"drop_left": "drop left", "drop_right": "drop right"}[_tok]
                    self.get_logger().info(
                        f"release pause: {action!r} -> {_tok!r}, but the arm is "
                        "already there — nothing published, question re-asked.")
                    self._narrate(
                        f"I'm already at the {_sp} position. Should I open the gripper?")
                    return
                # 2026-09-25 (review R5): the router answered, so the router
                # narrates what the executor does with it (chat_node no
                # longer speaks it): "Going to drop <side>." on its
                # redirect:<target>, or just "Opening the gripper." on
                # 'releasing' when an explicit release opens in place. See
                # _on_interactive_status.
                self._dest_answer_pending = True
                if not self._claim_executor_resume(
                    _tok,
                    f"release pause: {action!r} names a side -> "
                    + ("carry and release" if "+release" in _tok else "redirect, ask again"),
                ):
                    self._narrate_ignored_resume(action, _tok)
                return
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
                # 2026-09-25: left/right go out as drop_<side>+release, which
                # the executor maps to its CONFIGURED drop targets (Majd's
                # vla_drop_left / vla_drop_right); the hard-coded
                # carry_release:drop_*_side named other poses.
                _tok = {"home": "carry_release:home", "left": "drop_left+release",
                        "right": "drop_right+release"}.get(_d, "")
                _tgt = _tok
                if _tgt:
                    if not self._claim_executor_resume(
                        _tok,
                        f"release pause: carry+release ({_tok})",
                    ):
                        self._narrate_ignored_resume(action, "carry_release")
                else:
                    self.get_logger().warn(
                        f"carry_release with unknown destination "
                        f"{_d!r} — ignoring.")
                return
            if action == "release_home" or (
                    action in ("open_gripper", "release")
                    and _then_home(action, intent.get("object_attributes"))):
                # Compound "open the gripper and go back home" — the
                # executor opens and goes straight home, no question.
                # 2026-09-28: also open_gripper / release with
                # then=release_home (the model's form of the same sentence).
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
        # 2026-09-25: question from the task state when known (legacy flag
        # otherwise), see _question_open.
        if self._question_open("home"):
            # 2026-09-25 (Majd, live test ~1790347093): "Don't go back home,
            # go to the left side." at "Should I go back home?" was sent as
            # drop_left through the pick-in-flight branch; the executor went
            # HOME and "Going to drop left." was said anyway. A side here is
            # the plain token drop_<side> — the executor moves to its
            # configured drop pose instead of home and ends
            # done|id|stayed:<target> — and "Going to drop <side>." waits for
            # its resumed:<target> (_dest_answer_pending). Already at that
            # side: "stay". Only with a live task state (the executor that
            # knows this answer); legacy runs the paths below unchanged.
            _hside = (self._home_question_side(action, intent)
                      if self._live_task() is not None else "")
            if _hside:
                if self._arm_at_side(_hside):
                    if not self._claim_executor_resume(
                        "stay", f"home-confirm pause: {action!r} names the side "
                        "the arm is already at — staying"):
                        self._narrate_ignored_resume(action, "stay")
                    else:
                        self._narrate("Okay, I'll stay here.")
                    return
                self._dest_answer_pending = True
                if not self._claim_executor_resume(
                    _hside, f"home-confirm pause: {action!r} -> {_hside} "
                    "(configured drop pose instead of home)"):
                    self._narrate_ignored_resume(action, _hside)
                return
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
        # 2026-09-25: _pick_in_flight() also counts pointing / GUI picks once
        # the task state is known (phase != idle) — those used to turn a
        # "put it on the left" into a real MoveToNamedPose goal while the
        # executor was driving the arm.
        pick_in_flight = self._pick_in_flight()
        # 2026-09-25: EARLY DESTINATION LATCH. A destination said while the
        # grasp is still on its way (no destination question open yet) is
        # kept, not published — the executor clears its resume slot when
        # the pause opens, so a token sent now was lost. It answers the
        # pause from _on_interactive_status('paused'). See _latch_allowed
        # for exactly when; everything else takes the paths below.
        if pick_in_flight and self._latch_allowed():
            _tok = self._destination_token(action, intent.get("object_attributes"))
            if _tok:
                self._latch_destination(_tok, action)
                return
        # 2026-09-25 (review R1): a destination that was NOT latched above —
        # a pick_and_place (its side is fixed, no destination question), a
        # cycle that will not ask (asks_destination false), or confirm_drop
        # without a side — used to be published as a resume while NO pause
        # was open; the executor's open_pause_mailbox then erased it and,
        # chat_node having withheld its line for these phases, the operator
        # heard nothing at all. With a live task state, no question open and
        # the pick still before its grasp (idle-but-dispatched, probing,
        # approaching — the phases chat_node withholds in), nothing is
        # published: see _destination_without_pause. Legacy (no task state)
        # and every other phase keep the paths below unchanged (the
        # releasing phase must keep its early "home": the executor consumes
        # a home that lands during its post-release wait).
        if pick_in_flight and self._defer_after_cycle(action, intent):
            return
        if (pick_in_flight
                and action in ("confirm_drop", "drop_right", "drop_left", "go_home")
                and self._live_task() is not None
                and self._task_q() == ""
                and str((self._live_task() or {}).get("phase") or "idle")
                in ("idle", "probing", "approaching")):
            self._destination_without_pause(action, intent.get("object_attributes"))
            return
        if action in _compound:
            base, suffix = _compound[action]
            if pick_in_flight:
                self._note_dest_answer()
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
            # "Right side, open the gripper, after that go home" while the
            # destination question is open: the model sometimes sends
            # release_home WITH the side instead of confirm_drop + then
            # (1 of 10 phrasings, 2026-09-24). Same meaning — the side is the
            # answer, the release is the plan.
            # 2026-09-25: the destination question from the task state when
            # known, else the legacy _destination_open.
            if (action in ("release_home", "release", "open_gripper") and dest
                    and self._question_open("destination")):
                _plan = "release_home" if action == "release_home" else "release"
                self.get_logger().info(
                    f"{action!r} with destination {dest!r} at the destination question "
                    f"-> confirm_drop {dest} then {_plan}")
                attrs = dict(attrs) if isinstance(attrs, dict) else {}
                attrs["then"] = _plan
                action = "confirm_drop"
            # 2026-09-29 (Majd, live: "I don't want you to move right now. I
            # just want you to open the gripper, give me the object." at
            # "Where should I drop it?" was put off below to a release pause
            # that never came — nothing opened, nothing was said, and the
            # model then claimed "The gripper just opened"): an open /
            # release with no side AT the destination question, or a
            # destination meaning "here" ("the same place", "where it was"),
            # releases WHERE THE ARM IS — the executor's "here" (no
            # transport, open, rise clear, then "Should I go back home?").
            if self._question_open("destination") and (
                    (action in ("release_home", "release", "open_gripper") and not dest)
                    or (action == "confirm_drop" and dest in _HERE_WORDS)):
                _tok = "here+release_home" if (_then_home(action, attrs) or (
                    isinstance(attrs, dict) and str(attrs.get("then") or "").strip().lower()
                    == "release_home")) else "here+release"
                if not self._claim_executor_resume(
                        _tok, f"{action!r} at the destination question: release here"):
                    self._narrate_ignored_resume(action, _tok)
                return
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
                self._note_dest_answer()
                if not self._claim_executor_resume(
                    dest, "bypass queue (pick in flight): confirm_drop"
                ):
                    self._narrate_ignored_resume(action, dest)
                return
            if action in ("drop_right", "drop_left", "go_home"):
                redirect = {"drop_right": "drop_right",
                            "drop_left": "drop_left",
                            "go_home": "home"}[action]
                # "Drop it home, open the gripper" (2026-09-18): the chat
                # node sends ONE go_home with then=release instead of a
                # second open_gripper — the second intent used to fire the
                # gripper standalone, mid-carry, and the executor still
                # asked at home. The suffix makes the executor open on
                # arrival, no question.
                then_s = str((attrs or {}).get("then") or "").strip().lower() if isinstance(attrs, dict) else ""
                if then_s == "release_home":
                    redirect += "+release_home"
                elif then_s == "release":
                    redirect += "+release"
                self._note_dest_answer()
                if not self._claim_executor_resume(
                    redirect,
                    f"bypass queue (pick paused): plain {action!r}"
                    + (f" then {then_s}" if then_s else ""),
                ):
                    self._narrate_ignored_resume(action, redirect)
                return
            if action in ("release_home", "open_gripper", "release"):
                # 2026-09-28 (Majd's trial 084: "Give me the object and go back
                # home." while carrying — the model sent open_gripper with
                # then=release_home, the plan became "release" and the
                # executor asked "Should I go back home?" after opening).
                plan = "release_home" if _then_home(action, attrs) else "release"
                if plan == "release" and self._followup_plan == "release_home":
                    plan = "release_home"   # never downgrade an earlier "then home"
                self._followup_plan = plan
                self._followup_cmd = self._routing_cmd
                self._cmd_event(self._routing_cmd, "deferred",
                                "kept for the release pause of the running pick",
                                parent_id=(self._pick_cmd or {}).get("command_id", ""))
                self.get_logger().info(
                    f"Follow-up {action!r} noted while the pick is in flight — "
                    f"it answers the release pause when it opens (plan={plan}).")
                # 2026-09-29: said, not silent — the model filled the silence
                # with "The gripper just opened" (Majd's live test).
                self._narrate("Okay — I'll open the gripper once I'm there."
                              if (self._live_task() or {}).get("holding") else
                              "Okay — I'll give it to you once I have it.")
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
                self._cmd_event(self._routing_cmd, "rejected", "duplicate of a running or queued command")
                self.get_logger().info(
                    f"Dedup: dropping repeat intent {action!r} "
                    "(already running or already queued)."
                )
                # 2026-09-25 (Majd, live: "The small bottle." while the bottle
                # was still in the gripper at the release question was dropped
                # here in silence): a pick repeated while the running pick holds
                # its object gets the still-holding line instead of nothing.
                _t = self._live_task() or {}
                if action == "pick" and _t.get("holding"):
                    self._say_still_holding(rate_limited=True)
                return
            self._intent_queue.append(intent)
            queued = len(self._intent_queue)
        self._cmd_event(self._routing_cmd, "queued", f"position {queued}, router busy={self._busy}")
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
        """Run the queue head (_dispatch_inner). A command that finished
        synchronously without being accepted was refused by one of the
        checks (robot not ready, which one?, still holding ...): its
        "rejected" event carries what was said (recording contract v1)."""
        cmd = self._cmd_from_intent(intent)
        self._active_cmd = cmd
        self._dispatch_narr = []
        try:
            self._dispatch_inner(intent)
        finally:
            said, self._dispatch_narr = self._dispatch_narr or [], None
            if cmd.get("state") in ("received", "queued") and not self._busy:
                self._cmd_event(cmd, "rejected", " / ".join(said) or "refused by the router")

    def _dispatch_inner(self, intent: dict) -> None:
        action = str(intent.get("action_intent") or "").strip().lower()
        self._currently_dispatched_action = action
        obj = str(intent.get("object_intent") or "").strip()
        attrs = intent.get("object_attributes") or {}
        colour = shape = position = destination = ""
        ordinal = 0
        ordinal_from = "left"
        # Default speed for voice moves without a speed word (2026-09-23):
        # set live from the GUI's Motion window Apply (parameter
        # default_speed = very_slow|slow|normal|fast|very_fast).
        try:
            speed = str(self.get_parameter("default_speed").value).strip().lower() or "slow"
        except Exception:
            speed = "slow"
        if speed not in {"very_slow", "slow", "normal", "fast", "very_fast"}:
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
            # 2026-09-25: _pick_in_flight() — the same test, plus the task
            # state's phase once the executor reports one.
            if not self._pick_in_flight():
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
        # 2026-09-25: _pick_in_flight() (in _dispatch _busy_kind is always
        # "" here, so without a task state this is the old flag test).
        if self._pick_in_flight() and action in (
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
        # 2026-09-25: only without a task state — with the new executor a
        # stop is ONE abort (_handle_stop), which resolves every pause.
        if action == "stop" and self._task is None and self._pick_awaiting_destination:
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
            # 2026-09-25: left / right go to the executor's configured drop
            # poses when the live task state names them (drop_targets);
            # otherwise today's table.
            _pose = DROP_INTENTS[action]
            _side = {"drop_left_side": "left", "drop_right_side": "right"}.get(_pose, "")
            if _side:
                _pose = self._configured_drop_pose(_side) or _pose
            self._send_drop_pose(_pose, velocity_scaling)
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
            # 2026-09-25: the configured drop poses when known (as above).
            if position in ("right", "rightmost"):
                pose = self._configured_drop_pose("right") or "drop_right_side"
            elif position in ("left", "leftmost"):
                pose = self._configured_drop_pose("left") or "drop_left_side"
            else:
                pose = "drop"
            self._busy_kind = "named_pose"
            self._send_drop_pose(pose, velocity_scaling)   # 2026-09-25: home first, see there
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
            # 2026-09-25: the executor's configured drop pose for that side
            # when the live task state names it (drop_targets).
            _cside = {"drop_right_side": "right", "drop_left_side": "left"}.get(chosen_drop, "")
            if _cside:
                chosen_drop = self._configured_drop_pose(_cside) or chosen_drop
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

        # 2026-09-25 (Majd approved): never start a pick while the gripper is
        # closed on something — the executor's live task state says holding
        # (closed on an object, not opened since). Checked here, at DISPATCH,
        # not at intake: a pick queued behind a running cycle is judged when
        # that cycle is over (it normally ends released). Nothing is sent to
        # the executor or the pipeline. Legacy (no live task state): as before.
        _held_task = self._live_task()
        if _held_task is not None and _held_task.get("holding"):
            self.get_logger().warn(
                f"Pick {obj!r} refused: the executor is still holding "
                f"{_held_task.get('object')!r}.")
            self._say_still_holding()
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
            # 2026-09-28 (Majd: say what you can SEE, not what you can grasp)
            self._narrate(self._not_seen_line(obj))
            self._mark_busy_done("")
            return

        # 2026-09-29: the answer to this node's which-one question keeps the
        # destination of the pick_and_place it refused (the model often
        # re-sends only "pick the leftmost vitamin tube").
        _plan = None if is_pick_and_place else self._clarify_plan_for(label)
        if _plan:
            is_pick_and_place = True
            chosen_drop, destination = _plan.get("drop") or "", _plan.get("destination") or ""
            if _plan.get("then"):
                attrs = {**(attrs if isinstance(attrs, dict) else {}), "then": _plan["then"]}
            self.get_logger().info(
                f"Which-one answer {obj!r}: carrying the refused pick_and_place's "
                f"destination {destination!r} (then={_plan.get('then') or ''!r}).")
        # a which-one question below keeps this pick's destination plan
        _refused_plan = ({"drop": chosen_drop, "destination": destination,
                          "then": str(attrs.get("then") or "").strip().lower()
                          if isinstance(attrs, dict) else ""}
                         if is_pick_and_place else None)

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
                # 2026-09-25: the real count and its positions (_two_of_line).
                self._narrate(self._two_of_line(label, n_vit))
                self._clarify = (self._to_display_label(label).lower(), time.monotonic(),
                                 _refused_plan)
                self._mark_busy_done("")
                return

        # 2026-09-25 (Majd, live): with exactly TWO of the requested class in
        # view, "the middle one" / "the one in the centre" names neither —
        # _resolve_index_by_position took matches[len // 2], i.e. silently
        # the right-hand one. Ask with the two-tubes line instead (named for
        # the class; identical for the vitamin tubes). An ordinal or a colour
        # still decides (the colour filter runs first), and three or more
        # keep today's middle.
        if (position in ("middle", "centre", "center", "central")
                and not ordinal and not colour
                and self._count_visible_for_label(label) == 2):
            self.get_logger().info(
                f"Refusing {position!r} with exactly two {label!r} visible: "
                "neither is in the middle. Asking user to clarify.")
            self._narrate(self._two_of_line(label))
            self._clarify = (self._to_display_label(label).lower(), time.monotonic(),
                             _refused_plan)
            self._mark_busy_done("")
            return

        # All checks passed — now commit to the executor. Order matters:
        # set drop_target FIRST (so pick_and_place's destination lands),
        # THEN wait_for_destination, THEN the awaiting flag for our own
        # bypass routing.
        _then = str(attrs.get("then") or "").strip().lower() if isinstance(attrs, dict) else ""
        self._apply_destination_plan(
            {"drop": chosen_drop, "destination": destination, "then": _then}
            if is_pick_and_place else None)
        self._clarify = None   # 2026-09-29: a committed pick ends any which-one question
        # 2026-09-30: the pick is committed — its command is accepted, with
        # the ACCEPTED form (a which-one answer carries the refused
        # pick_and_place's destination) and the resolved target.
        self._pick_cmd = self._active_cmd
        self._cmd_event(
            self._active_cmd, "accepted", route="pick",
            action="pick_and_place" if is_pick_and_place else "pick",
            attributes={k: v for k, v in (attrs if isinstance(attrs, dict) else {}).items() if v not in (None, "")}
            | ({"destination": destination} if is_pick_and_place and destination else {}),
            target={"label": self._to_display_label(label), "colour": colour or "",
                    "position": position or "", "ordinal": ordinal or 0,
                    "ordinal_from": ordinal_from, "source": "voice"})

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
        # 2026-09-25: the executor knows no object names; it reads this at
        # the cycle start and freezes it at the grasp verdict, so the task
        # state (held object, reset line, the LLM's context) can name it.
        # The display label, as _set_target_label stores it.
        self._set_task_object_label(self._to_display_label(label))
        # 2026-09-25: a destination latched for an earlier pick must never
        # answer this one.
        self._drop_early_dest("new pick dispatched")
        self._after_cycle_intent = None   # 2026-09-25: a kept after-cycle move ends here too
        self._home_then = None             # 2026-09-25: and a drop pose waiting for its move home

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
        """The newest message on *topic* since the previous call, or None.

        2026-09-29: from a standing reader (ur3e_fast_observation's
        LatestMessage — a private node nothing spins, drained with
        take_message). It used to be rclpy's wait_for_message(): a
        subscription created and destroyed every 0.5 s on 500-830 Hz
        topics, which with SHM off meant a discovery round per sample and
        the driver streaming to readers already gone — this node's busy
        Fast DDS thread and its UDP drops (see latest_message.py). Never
        raises; a missing publisher simply gives None."""
        try:
            readers = self.__dict__.setdefault("_latest_readers", {})
            reader = readers.get(topic)
            if reader is None:
                reader = readers[topic] = LatestMessage(self, msg_type, topic)
            return reader.take()
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
        # 2026-09-29: no backoff any more — the read never blocks (it used to
        # cost up to the 0.3 s sample timeout while the driver was silent).
        if self._sample_one_message(JointState, self._joint_states_topic) is None:
            return
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
                    and self._robot_program_running is not False
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
            #
            # 2026-09-22: the flag is AUTHORITATIVE. On 09-18 two "live
            # witnesses" (trajectory controller active, dashboard PLAYING)
            # were allowed to outrank a False flag; both are false witnesses
            # when the External Control link drops: the pendant program
            # stays PLAYING but disconnected, and this node's own auto-heal
            # re-activates the controller the stopper had switched off.
            # Goals were then accepted and aborted on state tolerance while
            # the operator was told the robot was fine. The witness code is
            # gone; the reason names what a False flag means.
            if now - self._stale_flag_warned_at > 30.0:
                self._stale_flag_warned_at = now
                self.get_logger().warn(
                    "robot_program_running is False: the External Control "
                    f"link is down (controller {self._last_controller_state or '?'}, "
                    f"dashboard {self._dashboard_state or '?'}).")
            return (
                "The robot program isn't connected to the driver. Press Stop "
                "and Play on the teach pendant, or switch it to Remote control "
                "so I can restart it."
            )
        return None

    def _program_running_witness(self, now: float) -> str:
        """Retired 2026-09-22 (see _check_robot_ready): nothing outranks the
        driver's own flag. Kept only so the status broadcast can say what
        the other sources claim."""
        return ""

    def _poll_dashboard_program_state(self) -> None:
        """Ask the UR dashboard for the program state (async; result lands
        in _dashboard_state/_dashboard_at). Called by the controller
        watchdog only while the latched program flag says False."""
        if not hasattr(self, "_dashboard_cli"):
            self._dashboard_cli = self.create_client(
                GetProgramState, "/dashboard_client/program_state")
        if not self._dashboard_cli.service_is_ready():
            return

        def _done(fut) -> None:
            try:
                resp = fut.result()
            except Exception as exc:
                self.get_logger().debug(f"dashboard program_state failed: {exc}")
                return
            if resp is not None and resp.success:
                self._dashboard_state = str(resp.state.state)
                self._dashboard_at = self.get_clock().now().nanoseconds / 1e9

        self._dashboard_cli.call_async(GetProgramState.Request()).add_done_callback(_done)

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
        These flags are already maintained by _on_interactive_status.

        2026-09-25: once the executor's task state has arrived, its
        ``question`` decides (non-empty exactly while a pause waits); the
        flags below are the legacy fallback for an old executor (and, since
        review R4, for a dead one: _live_task)."""
        if self._live_task() is not None:
            return self._task_q() != ""
        return bool(
            self._pick_awaiting_destination
            or self._pick_awaiting_release
            or getattr(self, "_pick_awaiting_home", False)
        )

    # ── executor task state (2026-09-25) ─────────────────────────────────
    # Every helper below reads /grasp_executor_lin_llm/task_state and falls
    # back to the legacy flags ONLY until the first task state arrives:
    # with an old executor that never publishes it, the router behaves
    # exactly as at 3f512258.

    def _on_task_state(self, msg: String) -> None:
        """2026-09-25: store the executor's latched task state.

        Read as a LEVEL, not as events (SPEC §1.1): a message whose seq is
        not newer than the last one is dropped unless ``boot`` changed (the
        executor restarted); the latched first sample is adopted silently —
        nothing is narrated from task state, the held-object topic is only
        brought in line with it. Then the early destination latch is bound
        to the cycle it was said in, or dropped when that cycle is gone.
        """
        try:
            task = json.loads(msg.data or "")
        except (json.JSONDecodeError, TypeError) as exc:
            self.get_logger().warn(f"task_state: unparsable ({exc}); ignored.")
            return
        if not isinstance(task, dict):
            return
        try:
            boot = float(task.get("boot") or 0.0)
            seq = int(task.get("seq") or 0)
        except (TypeError, ValueError):
            return
        if boot == self._task_boot and seq <= self._task_seq:
            return
        first = self._task is None
        prev = self._task or {}
        self._task, self._task_boot, self._task_seq = task, boot, seq
        if first or boot != float(prev.get("boot") or -1.0) \
                or task.get("phase") != prev.get("phase") \
                or task.get("question") != prev.get("question"):
            self.get_logger().info(
                f"task_state{' (first)' if first else ''}: cycle "
                f"{task.get('cycle_id')} phase={task.get('phase')!r} "
                f"question={task.get('question')!r} "
                f"holding={task.get('holding')} object={task.get('object')!r}")
        self._sync_held_object()
        latch = self._early_dest
        if latch is not None:
            phase = str(task.get("phase") or "")
            try:
                cid = int(task.get("cycle_id") or 0)
            except (TypeError, ValueError):
                cid = 0
            if latch["cycle"] is None:
                # Said before the pick's cycle began: bind to the first NEW
                # cycle that starts (not the one that was idle at the time).
                if phase in ("probing", "approaching") and cid != latch["seen"]:
                    latch["cycle"] = cid
                    self.get_logger().info(
                        f"Early destination {latch['token']!r} bound to cycle {cid}.")
            elif latch["cycle"] != cid:
                self._drop_early_dest(
                    f"cycle {latch['cycle']} is over (task at cycle {cid})")

    def _live_task(self) -> Optional[dict]:
        """2026-09-25 (review R4): the task state, but only while the
        executor that published it is alive — None when no publisher is left
        on the task-state topic. The sample is latched (transient_local), so
        a crashed or stopped executor's last state (a question it will never
        ask again, a phase it left) would otherwise keep steering routing;
        with None every reader falls back to the legacy flags, exactly as
        before the first task state. A restarted executor publishes its idle
        state at start (new boot), which is adopted by _on_task_state."""
        task = self._task
        if task is None:
            return None
        try:
            if self.count_publishers(self._task_state_topic) == 0:
                return None
        except Exception:
            pass   # graph query failed: keep trusting the last level
        return task

    def _not_holding_action(self) -> str:
        """2026-09-28 (Majd, live: the grasp missed, the empty fingers closed
        tight and set the grasp pin, and every pick was refused "still
        holding the small bottle"): the operator confirmed the gripper is
        empty. Any open clears the executor's holding record
        (note_gripper_command), so the answer is an open, where the robot is
        allowed to open now:

          * release question — "release_home": open here, then home;
          * destination question — "home+release": home, open there;
          * idle — "open_gripper", returned for the normal intent path;
          * record already empty — nothing to do, said so;
          * any other phase (the arm moves) — nothing, asked to repeat.

        Returns the action to continue with, or "" when handled here."""
        task = self._live_task()
        if task is not None and not task.get("holding"):
            self._narrate("Okay — my gripper is already marked empty.")
            return ""
        q = self._task_q()
        if q in ("release", "destination"):
            tok = "release_home" if q == "release" else "home+release"
            self.get_logger().info(
                f"not_holding at the {q} question -> resume {tok!r} (open, forget the object)")
            if q == "destination":
                self._note_dest_answer()
            if self._claim_executor_resume(tok, f"not_holding at the {q} question"):
                self._narrate("Okay — I'll open the gripper and forget the object.")
            else:
                self._narrate_ignored_resume("not_holding", tok)
            return ""
        phase = str((task or {}).get("phase") or "idle")
        if task is None or phase == "idle":
            self.get_logger().info("not_holding while idle -> open_gripper (clears the record)")
            self._narrate("Okay — I'm not holding anything.")   # the open result says the rest
            return "open_gripper"
        self.get_logger().info(f"not_holding ignored: the arm is busy (phase={phase!r})")
        self._narrate("Let me finish this move first — then tell me again.")
        return ""

    def _task_q(self) -> str:
        """2026-09-25: the question the executor has open — "",
        "destination", "release" or "home". From the task state when it has
        arrived (and its publisher lives, review R4: _live_task), else
        inferred from the legacy flags."""
        task = self._live_task()
        if task is not None:
            return str(task.get("question") or "")
        if self._destination_open:
            return "destination"
        if self._pick_awaiting_release:
            return "release"
        if getattr(self, "_pick_awaiting_home", False):
            return "home"
        return ""

    def _question_open(self, question: str) -> bool:
        """2026-09-25: is THIS question open? The task state's answer when
        known; without it, exactly the flag each call site tested before
        (_destination_open / _pick_awaiting_release / _pick_awaiting_home),
        so an old executor keeps today's routing bit for bit. Review R4: a
        dead executor's latched state counts as unknown (_live_task)."""
        if self._live_task() is not None:
            return self._task_q() == question
        if question == "destination":
            return bool(self._destination_open)
        if question == "release":
            return bool(self._pick_awaiting_release)
        if question == "home":
            return bool(getattr(self, "_pick_awaiting_home", False))
        return False

    def _pick_in_flight(self) -> bool:
        """2026-09-25: is a pick cycle running? The router's own pick, or —
        once the task state is known — ANY executor cycle (pointing and GUI
        picks included: phase != idle). Legacy: today's flag test — also
        when the executor is gone (review R4: _live_task)."""
        task = self._live_task()
        if task is not None:
            return (self._busy_kind == "pick"
                    or str(task.get("phase") or "idle") != "idle")
        return bool(self._pick_awaiting_destination or self._busy_kind == "pick")

    def _sync_held_object(self) -> None:
        """2026-09-25: /robot/held_object derived from the task state (the
        executor's own "closed on something, not opened since"), published
        only when it changes. The legacy writes in _on_interactive_status
        and _on_cycle_done run only while no task state has arrived.
        Review R4: nothing is taken from a dead executor's latched state."""
        task = self._live_task()
        if task is None:
            return
        if task.get("holding"):
            label = (str(task.get("object") or "") or self._last_target_label
                     or self._held_object_label or "object")
        else:
            label = ""
        if label != self._held_object_label:
            self._set_held_object(label)

    def _latch_allowed(self) -> bool:
        """2026-09-25: may a destination intent be LATCHED right now?

        Only with the task state known, no destination question open, and
        either the executor is probing / approaching a pick that will ask
        (asks_destination), or it is still idle while the router's own plain
        voice pick (which will ask) is dispatched. Any other time the
        existing paths run unchanged. Review R4: only a LIVE task state
        (_live_task) — a dead executor's last phase allows no latch."""
        task = self._live_task()
        if task is None or self._task_q() == "destination":
            return False
        phase = str(task.get("phase") or "")
        if phase in ("probing", "approaching"):
            return bool(task.get("asks_destination"))
        if phase == "idle":
            return self._busy_kind == "pick" and self._pick_waits_for_destination
        return False

    @staticmethod
    def _destination_token(action: str, attrs) -> Optional[str]:
        """2026-09-25: the resume token a destination intent stands for, or
        None. Pure. confirm_drop (with a destination), drop_right /
        drop_left / go_home, and the compounds drop_*_release[_home] map to
        drop_right / drop_left / home plus +release / +release_home from
        ``then``; confirm_drop WITHOUT a destination is None (today's path
        defaults it to the right side only at the question itself)."""
        compound = {
            "drop_right_release": "drop_right+release",
            "drop_left_release": "drop_left+release",
            "drop_right_release_home": "drop_right+release_home",
            "drop_left_release_home": "drop_left+release_home",
        }
        if action in compound:
            return compound[action]
        attrs = attrs if isinstance(attrs, dict) else {}
        if action == "confirm_drop":
            base = _DESTINATION_WORDS.get(
                str(attrs.get("destination") or "").strip().lower())
        elif action in ("drop_right", "drop_left", "go_home"):
            base = {"drop_right": "drop_right", "drop_left": "drop_left",
                    "go_home": "home"}[action]
        else:
            return None
        if not base:
            return None
        then_s = str(attrs.get("then") or "").strip().lower()
        if then_s == "release_home":
            return base + "+release_home"
        if then_s == "release":
            return base + "+release"
        return base

    @staticmethod
    def _release_side(action: str, intent: dict) -> str:
        """2026-09-25: "drop_right" / "drop_left" when an intent answering
        the release question names a SIDE — confirm_drop / drop_right /
        drop_left, or open_gripper / release carrying a destination side —
        else "". Home is not a side (go_home keeps today's redirect)."""
        attrs = intent.get("object_attributes")
        attrs = attrs if isinstance(attrs, dict) else {}
        if action in ("drop_right", "drop_left"):
            return action
        # 2026-09-25: release_home carrying a side too — without it "drop it
        # on the left, give it to me and go home" at the release question fell
        # through to the plain release_home token and opened WHERE THE ARM IS.
        if action in ("confirm_drop", "open_gripper", "release", "release_home"):
            side = _DESTINATION_WORDS.get(
                str(attrs.get("destination") or "").strip().lower(), "")
            return side if side in ("drop_right", "drop_left") else ""
        return ""

    def _release_side_token(self, action: str, intent: dict) -> str:
        """2026-09-25 (Majd, live test): the resume token for a SIDE named at
        the release question, or "" when no side is named. An explicit
        release — open_gripper / release with a side, or a ``then`` of
        release / release_home — gives drop_<side>+release; a bare side
        (confirm_drop / drop_left / drop_right) gives the plain redirect
        drop_<side>, after which the executor asks again. Never opens on a
        side alone."""
        side = self._release_side(action, intent)
        if not side:
            return ""
        attrs = intent.get("object_attributes")
        attrs = attrs if isinstance(attrs, dict) else {}
        then_s = str(attrs.get("then") or "").strip().lower()
        # release_home with a side: carry there and open (the executor's
        # carry_release then asks "Should I go back home?" — the home part is
        # asked, never assumed, rather than opening at the wrong side).
        if (action in ("open_gripper", "release", "release_home")
                or then_s in ("release", "release_home")):
            return side + "+release"
        return side

    @staticmethod
    def _home_question_side(action: str, intent: dict) -> str:
        """2026-09-25: "drop_left" / "drop_right" when an answer to "Should I
        go back home?" names a side — drop_left / drop_right and their
        DROP_INTENTS aliases (go_to_drop_left, …), or confirm_drop with a
        side — else "" (go_home, yes, a sideless confirm_drop: home)."""
        pose = DROP_INTENTS.get(action, "")
        if pose == "drop_left_side":
            return "drop_left"
        if pose == "drop_right_side":
            return "drop_right"
        if action == "confirm_drop":
            attrs = intent.get("object_attributes")
            attrs = attrs if isinstance(attrs, dict) else {}
            side = _DESTINATION_WORDS.get(
                str(attrs.get("destination") or "").strip().lower(), "")
            return side if side in ("drop_left", "drop_right") else ""
        return ""

    def _configured_drop_pose(self, side: str) -> str:
        """2026-09-25: the executor's configured drop pose for ``side``
        ("left" / "right") from the live task state's drop_targets (Majd's
        vla_drop_left / vla_drop_right), or "" — the caller then keeps its
        hard-coded drop_*_side name, exactly as before."""
        task = self._live_task() or {}
        targets = task.get("drop_targets")
        if isinstance(targets, dict):
            return str(targets.get(side) or "").strip()
        return ""

    @staticmethod
    def _spoken_pose(name: str, default: str) -> str:
        """2026-09-25: how a drop pose is said — by "left" / "right" in its
        name, so the configured vla_drop_* poses are named too (the lookup
        knew only drop_*_side: "I'm at the drop position, …" live)."""
        n = (name or "").lower()
        if n == "home":
            return "home"
        if "left" in n:
            return "drop left"
        if "right" in n:
            return "drop right"
        return default

    def _arm_at_side(self, side: str) -> bool:
        """2026-09-25: is the arm already at ``side`` ("drop_left" /
        "drop_right")? From the task state's destination (the executor's
        drop target while it waits at the release question), else the last
        resumed:<pose>. Matched by "left" / "right" in the target name, so
        configured target names work too; unknown -> False (claim as usual)."""
        task = self._live_task() or {}
        where = (str(task.get("destination") or "")
                 or str(getattr(self, "_executor_drop_dest", "") or "")).lower()
        if not where:
            return False
        if side == "drop_left":
            return "left" in where
        if side == "drop_right":
            return "right" in where
        return False

    def _say_still_holding(self, fallback: str = "", *, rate_limited: bool = False) -> None:
        """2026-09-25 (Majd approved): the answer to a pick while the gripper
        is closed on something — "I'm still holding the {object} — where
        should I put it first?", the object from the live task state (else
        ``fallback``, else "object"). ``rate_limited`` (the executor's
        refusal of a pointing / GUI pick) speaks at most once per 15 s so
        repeated pointing cannot spam it; a spoken pick always gets it."""
        now = time.monotonic()
        if rate_limited and now - self._still_holding_mono < 15.0:
            self.get_logger().info(
                "Still-holding line suppressed (said "
                f"{now - self._still_holding_mono:.1f} s ago).")
            return
        self._still_holding_mono = now
        task = self._live_task() or {}
        what = (str(task.get("object") or "") or fallback or "object").replace("_", " ")
        self._narrate(f"I'm still holding the {what} — where should I put it first?")

    def _two_of_line(self, label: str, n: int = 2) -> str:
        """2026-09-25: the several-of-a-kind clarification, named for the
        class. For vitamin_c (display "vitamin_tube") and n == 2 it is exactly
        the existing two-tubes line. 2026-09-25 (Majd, live: "I see two
        vitamin tubes" with THREE on the table — the count was hard-coded):
        the sentence says the real count and offers the positions that
        exist (three: leftmost, middle or rightmost)."""
        name = self._to_display_label(label).replace("_", " ").strip() or "object"
        plural = name + ("es" if name.endswith(("s", "x", "z", "ch", "sh")) else "s")
        count = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}.get(n, str(n))
        if n > 3:
            return (f"I see {count} {plural}. Could you say which one, for example "
                    "the leftmost or the rightmost, or point at it?")
        choice = "leftmost or rightmost" if n <= 2 else "leftmost, middle or rightmost"
        return (f"I see {count} {plural}. Could you say {choice}, "
                "or point at the one you want?")

    # ── command lifecycle events (recording contract v1) ─────────────────

    def _cmd_from_intent(self, intent: dict) -> dict:
        """The command record of an intent — the same record for every event
        of that command. An intent without a command_id (GUI, a test) gets
        one here; it travels with the intent through the queue."""
        cid = str(intent.get("command_id") or "") or uuid.uuid4().hex
        intent["command_id"] = cid
        cmd = self._cmds.get(cid)
        if cmd is None:
            attrs = intent.get("object_attributes")
            cmd = {"command_id": cid, "parent_id": "",
                   "action": str(intent.get("action_intent") or "").strip().lower(),
                   "object": intent.get("object_intent"),
                   "attributes": dict(attrs) if isinstance(attrs, dict) else {},
                   "utterance_id": str(intent.get("utterance_id") or ""),
                   "utterance": str(intent.get("utterance") or ""),
                   "plan_id": str(intent.get("plan_id") or ""),
                   "plan_index": intent.get("plan_index"),
                   "question": str(intent.get("question") or ""),
                   "question_id": intent.get("question_id"),
                   "cycle_id": None, "target": None, "route": "", "state": ""}
            self._cmds[cid] = cmd
            while len(self._cmds) > 300:
                self._cmds.popitem(last=False)
        return cmd

    def _cmd_event(self, cmd: Optional[dict], state: str, reason: str = "", **extra) -> None:
        """Publish one lifecycle event of ``cmd`` (None = nothing to report).
        A command that already reached a terminal state gets no second one."""
        if not cmd:
            return
        terminal = ("completed", "failed", "rejected", "cancelled", "superseded")
        if cmd.get("state") in terminal and state in terminal:
            return
        for k in ("route", "target", "cycle_id", "parent_id", "action", "attributes"):
            if extra.get(k) not in (None, ""):
                cmd[k] = extra.pop(k)
            else:
                extra.pop(k, None)
        cmd["state"] = state
        self._cmd_seq += 1
        ev = {"schema": "morty.command_event.v1", "event_id": uuid.uuid4().hex,
              "seq": self._cmd_seq, "source": "intent_router", "boot": self._cmd_boot,
              "command_id": cmd["command_id"], "parent_id": cmd.get("parent_id", ""),
              "plan_id": cmd.get("plan_id", ""), "plan_index": cmd.get("plan_index"),
              "utterance_id": cmd.get("utterance_id", ""), "utterance": cmd.get("utterance", ""),
              "question": cmd.get("question", ""), "question_id": cmd.get("question_id"),
              "cycle_id": cmd.get("cycle_id"), "state": state, "action": cmd.get("action", ""),
              "attributes": cmd.get("attributes") or {}, "target": cmd.get("target"),
              "route": cmd.get("route", ""), "reason": reason, "stamp": round(time.time(), 4)}
        ev.update(extra)
        try:
            self._cmd_pub.publish(String(data=json.dumps(ev, default=str)))
        except Exception as exc:  # bookkeeping never breaks routing
            self.get_logger().debug(f"command event failed: {exc}")

    def _cancel_cmds(self, reason: str, queued: list) -> None:
        """A stop / reset: the queued intents and the executing commands end."""
        for it in queued:
            self._cmd_event(self._cmds.get(str(it.get("command_id") or "")), "cancelled", reason)
        for c in (self._active_cmd, self._pick_cmd, self._followup_cmd):
            self._cmd_event(c, "cancelled", reason)
        self._pick_cmd = None
        self._followup_cmd = None

    def _on_point_proposal(self, msg: String) -> None:
        """2026-09-28 (Majd: "ask like 'you pointed to this object, should I
        grasp it?' … and when the object selected is wrong he should say 'you
        selected the tape dispenser, do you want this object?'"): the pipeline
        holds a POINTED choice until it is confirmed. The question:

          * after this node's which-one question for the same kind
            (_two_of_line): "Do you mean this vitamin tube, the leftmost one?";
          * after it for ANOTHER kind: "You pointed at the tape dispenser, not
            a vitamin tube. Do you want the tape dispenser?";
          * otherwise: "You pointed at the small bottle. Should I grasp it?".

        While the gripper still holds an object nothing is asked: the
        still-holding line (rate-limited, as for the executor's refusal of a
        pointed pick) and a no. The pipeline's closing state (yes, no,
        superseded, abort, timeout) closes the question."""
        try:
            d = json.loads(msg.data or "{}")
            pid = int(d.get("id"))
        except (TypeError, ValueError, json.JSONDecodeError):
            return
        state = str(d.get("state") or "")
        label = str(d.get("label") or "")
        if state != "open":
            if self._point_q is not None and self._point_q["id"] == pid:
                self._close_point_question(state)
            return
        task = self._live_task()
        if task is not None and task.get("holding"):
            self.get_logger().info(
                f"Point proposal {pid} ({label}): still holding "
                f"{task.get('object')!r} — not asked, answered no.")
            self._point_answer_pub.publish(String(data=f"no|{pid}"))
            self._say_still_holding(rate_limited=True)
            return
        line = self._point_question_line(label, d.get("bbox"))
        self._point_q = {"id": pid, "label": label, "since": time.time(), "bbox": d.get("bbox")}
        self._open_q_pub.publish(String(data=json.dumps({
            "question": "point", "id": pid, "since": round(self._point_q["since"], 3),
            "object": label, "line": line})))
        self.get_logger().info(f"Point proposal {pid} ({label}): asking {line!r}")
        self._narrate(line)

    def _not_seen_line(self, obj: str) -> str:
        """2026-09-28 (Majd: "not telling what kind of object we can grasp —
        let him only say what object he can see"): "I don't see a mouse. I
        can see a tape dispenser, a small bottle and two vitamin tubes." —
        the second sentence from the live detections, left out when the
        scene is stale or empty."""
        name = re.sub(r"^(?:the|a|an) ", "", str(obj or "").strip().lower()) or "that"
        art = "" if name == "that" else ("an " if name[:1] in "aeiou" else "a ")
        seen = self._visible_objects_phrase()
        return f"I don't see {art}{name}." + (f" I can see {seen}." if seen else "")

    def _visible_objects_phrase(self) -> str:
        """'a tape dispenser, a small bottle and two vitamin tubes' from the
        live detections, left to right; '' when the scene is stale or empty."""
        now = self.get_clock().now().nanoseconds / 1e9
        if not self._detections_at or now - self._detections_at > max(self._scene_max_age_s * 3.0, 10.0):
            return ""

        def _x(det) -> float:
            try:
                b = det.get("bbox")
                return 0.5 * (float(b[0]) + float(b[2]))
            except (TypeError, ValueError, IndexError):
                return 0.0

        counts: dict = {}
        for det in sorted(self._detections, key=_x):
            name = str(det.get("label") or det.get("display") or "").replace("_", " ").strip()
            if name:
                counts[name] = counts.get(name, 0) + 1
        words = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}
        parts = []
        for name, n in counts.items():
            if n == 1:
                parts.append(("an " if name[:1] in "aeiou" else "a ") + name)
            else:
                plural = name + ("es" if name.endswith(("s", "x", "z", "ch", "sh")) else "s")
                parts.append(f"{words.get(n, str(n))} {plural}")
        if len(parts) <= 1:
            return parts[0] if parts else ""
        return ", ".join(parts[:-1]) + " and " + parts[-1]

    def _point_question_line(self, label: str, bbox) -> str:
        """The question for a pointed ``label`` (see _on_point_proposal)."""
        name = label.replace("_", " ").strip() or "object"
        ctx = self._clarify
        if ctx is None or time.monotonic() - ctx[1] > 120.0:
            return f"You pointed at the {name}. Should I grasp it?"
        if ctx[0] == label.lower():
            where = self._rank_word(label, bbox)
            return f"Do you mean this {name}" + (f", the {where} one" if where else "") + "?"
        want = ctx[0].replace("_", " ")
        art = "an" if want[:1] in "aeiou" else "a"
        return f"You pointed at the {name}, not {art} {want}. Do you want the {name}?"

    def _rank_word(self, label: str, bbox) -> str:
        """leftmost / middle / rightmost for the detection of ``label`` at
        ``bbox`` among the visible ones of its kind — the order
        _resolve_index_by_position reads those words in (ascending image x);
        "" with one, or more than three, of the kind or no fresh scene."""
        try:
            cx = 0.5 * (float(bbox[0]) + float(bbox[2]))
        except (TypeError, ValueError, IndexError):
            return ""
        now = self.get_clock().now().nanoseconds / 1e9
        if not self._detections_at or now - self._detections_at > max(self._scene_max_age_s * 3.0, 10.0):
            return ""
        xs = []
        for det in self._detections:
            if str(det.get("label") or det.get("display") or "").lower() != label.lower():
                continue
            try:
                b = det.get("bbox")
                xs.append(0.5 * (float(b[0]) + float(b[2])))
            except (TypeError, ValueError, IndexError):
                continue
        if not 2 <= len(xs) <= 3:
            return ""
        xs.sort()
        rank = min(range(len(xs)), key=lambda i: abs(xs[i] - cx))
        return ["leftmost", "middle", "rightmost"][rank] if len(xs) == 3 else ["leftmost", "rightmost"][rank]

    def _apply_destination_plan(self, plan: Optional[dict]) -> None:
        """The executor's destination settings for the pick about to start
        (moved out of _dispatch 2026-09-29, now shared with the pointing yes).
        ``plan`` = {"drop", "destination", "then"} of a pick_and_place — the
        executor carries the object there without asking (drop_target_name
        FIRST, then wait_for_destination) and a "then" of release /
        release_home answers the release pause on arrival ("Pick the bottle,
        put it on the right, then give it to me", 2026-09-24); None = a plain
        pick, which asks "Where should I drop it?"."""
        if plan:
            if plan.get("drop"):
                self._set_executor_drop_target(plan["drop"])
            self._set_executor_wait_for_destination(False)
            # 2026-09-25: pick_and_place carries its destination; nothing to latch.
            self._pick_waits_for_destination = False
            # 2026-09-25 (review R1): the side it goes to, when known
            # (drop_center / the executor default -> "").
            self._pick_fixed_side = (_DESTINATION_WORDS.get(plan.get("destination") or "", "")
                                     if plan.get("drop") else "")
        else:
            self._set_executor_wait_for_destination(True)
            self._pick_awaiting_destination = True
            # 2026-09-25: this pick will ask "Where should I drop it?", so a
            # destination said before its cycle starts may be latched.
            self._pick_waits_for_destination = True
            self._pick_fixed_side = ""   # 2026-09-25 (review R1)
        then = str((plan or {}).get("then") or "")
        self._followup_plan = then if then in ("release", "release_home") else ""
        if self._followup_plan:
            self.get_logger().info(
                f"pick_and_place with then={then!r}: the release pause will be answered on arrival.")

    def _clarify_plan_for(self, label: str) -> Optional[dict]:
        """2026-09-29 (Majd, live: "Pick up the vitamin tube and put it on the
        left side." -> which one? -> pointed, yes -> "Where should I drop
        it?"): the destination plan of the pick_and_place this node's
        which-one question refused, while that question is fresh and about
        ``label``'s kind — the answer (a pointed yes, or a spoken qualifier)
        carries it on. None otherwise."""
        ctx = self._clarify
        if (ctx is None or len(ctx) < 3 or not ctx[2]
                or time.monotonic() - ctx[1] > 120.0
                or ctx[0] != self._to_display_label(label).lower()):
            return None
        return ctx[2]

    def _pick_confirms_point(self, intent: dict, with_destination: bool = False) -> bool:
        """A plain pick of the proposed kind while its question is open ("Yes,
        grasp the vitamin tube.", or a pick without an object) confirms THAT
        object — sent as a label it would take the first of its kind, not
        necessarily the one pointed at. A qualifier or another object is a
        new pick; a destination too, unless ``with_destination``."""
        attrs = intent.get("object_attributes") or {}
        if not isinstance(attrs, dict):
            attrs = {}
        keys = ("position", "ordinal", "colour") + (() if with_destination else ("destination",))
        if any(attrs.get(k) for k in keys):
            return False
        obj = str(intent.get("object_intent") or "").strip()
        if not obj:
            return True
        label = self._resolve_label(obj, "", str(attrs.get("shape") or ""))
        return (label is not None and self._point_q is not None
                and self._to_display_label(label).lower() == self._point_q["label"].lower())

    def _answer_point(self, yes: bool, how: str) -> bool:
        """Send the operator's answer to the open pointing question; False
        when none is open. A yes opens the execution gate as a spoken pick
        does (the operator approved it by voice) and ends the which-one
        context; a no invites the next choice."""
        q = self._point_q
        if q is None:
            return False
        if yes:
            # 2026-09-29: BEFORE the yes (the executor reads these at its
            # cycle start): the destination of the pick_and_place this
            # pointing answers, else a plain pick that asks — pointed picks
            # used to inherit whatever the previous voice pick had set.
            plan = self._clarify_plan_for(q["label"])
            self._apply_destination_plan(plan)
            if plan:
                self.get_logger().info(
                    f"Pointed yes carries the refused pick_and_place's destination "
                    f"{plan.get('destination')!r}.")
        self._point_answer_pub.publish(String(data=f"{'yes' if yes else 'no'}|{q['id']}"))
        self.get_logger().info(
            f"Point proposal {q['id']} ({q['label']}): {'YES' if yes else 'NO'} ({how}).")
        # 2026-09-30: a yes starts the pointed pick — its command (the "yes",
        # or a plain pick of the same kind) is that pick, target = the
        # proposal; a no is a completed answer that starts nothing.
        _c = self._routing_cmd
        if _c is not None:
            _c["question"], _c["question_id"] = "point", q["id"]
        if yes:
            _plan = plan or {}
            self._pick_cmd = _c
            self._cmd_event(
                _c, "accepted", how, route="pick",
                action="pick_and_place" if _plan else "pick",
                attributes={k: v for k, v in (("destination", _plan.get("destination")),
                                              ("then", _plan.get("then"))) if v},
                target={"label": q["label"], "position": self._rank_word(q["label"], q.get("bbox")),
                        "bbox": q.get("bbox"), "proposal_id": q["id"], "source": "pointing"})
        else:
            self._cmd_event(_c, "completed", "answered no to the pointing question", route="none")
        self._close_point_question("yes" if yes else "no")
        if yes:
            self._clarify = None
            self._open_execution_gate()
        else:
            self._narrate("Okay. Point at the one you want, or tell me which one.")
        return True

    def _close_point_question(self, state: str) -> None:
        q, self._point_q = self._point_q, None
        self._open_q_pub.publish(String(data=json.dumps({
            "question": "", "id": q["id"] if q else 0, "closed": state})))
        self.get_logger().info(f"Point question {q['id'] if q else '-'} closed: {state}.")

    def _latch_destination(self, token: str, action: str) -> None:
        """2026-09-25: keep ``token`` for the destination pause of this pick
        (a newer one replaces it) and say what will happen — a stated plan,
        not "Going to drop right.", which waits for the executor's
        resumed:. Nothing is published now."""
        task = self._task or {}
        phase = str(task.get("phase") or "")
        try:
            cid = int(task.get("cycle_id") or 0)
        except (TypeError, ValueError):
            cid = 0
        old = self._early_dest
        self._early_dest = {
            "token": token,
            "cycle": cid if phase in ("probing", "approaching") else None,
            "seen": cid,
            "t": time.monotonic(),
            "cmd": self._routing_cmd,   # 2026-09-30: the command it answers later
        }
        self._cmd_event(self._routing_cmd, "deferred", "latched for the destination pause")
        self.get_logger().info(
            f"Early destination LATCHED: {action!r} -> {token!r} (phase={phase!r}, "
            f"cycle={self._early_dest['cycle']}, seen={cid})"
            + (f", replacing {old['token']!r}" if old else "")
            + " — it answers the destination pause when it opens.")
        where = {"drop_right": "the right side", "drop_left": "the left side",
                 "home": "the home position"}.get(token.split("+", 1)[0], "")
        if where:
            self._narrate(f"Okay — {where} once I have it.")

    def _destination_without_pause(self, action: str, attrs) -> None:
        """2026-09-25 (review R1): a destination said before the grasp while
        no pause is open and it could not be latched (_latch_allowed).
        Publishes NOTHING — a resume now would land in no open pause and be
        erased when the next one opens. Instead:

        * confirm_drop without a side: nothing to change; the destination
          question (if this cycle asks one) is still to come. Log only.
        * the side a pick_and_place is already going to (_pick_fixed_side):
          nothing changes — say the latch line (true: it goes there once
          held); a "then release [home]" becomes the follow-up plan, which
          answers the release pause when it opens, exactly as an
          open_gripper said during the approach does.
        * any other side: the executor cannot take it now (its drop target
          is read at the cycle start, and this cycle asks no destination),
          so say so honestly. At the release question a side is still
          taken (drop_<side>+release)."""
        tok = self._destination_token(action, attrs)
        task = self._live_task() or {}
        if not tok:
            self.get_logger().info(
                f"{action!r} without a destination while no pause is open "
                f"(phase={task.get('phase')!r}): nothing published — a resume "
                "now would be erased when the next pause opens.")
            return
        side, _, plan = tok.partition("+")
        if self._pick_fixed_side and side == self._pick_fixed_side:
            if plan in ("release", "release_home"):
                if not (plan == "release" and self._followup_plan == "release_home"):
                    self._followup_plan = plan   # never downgrade "then home"
            self.get_logger().info(
                f"{action!r} -> {tok!r} during the approach: the pick already "
                f"goes to {side!r} — nothing published"
                + (f", follow-up plan={self._followup_plan!r}." if plan else "."))
            where = {"drop_right": "the right side", "drop_left": "the left side",
                     "home": "the home position"}.get(side, "")
            if where:
                self._narrate(f"Okay — {where} once I have it.")
            return
        self.get_logger().warn(
            f"{action!r} -> {tok!r} while no pause is open and this cycle asks "
            f"no destination (phase={task.get('phase')!r}, "
            f"asks_destination={task.get('asks_destination')}, "
            f"pick_and_place side={self._pick_fixed_side!r}): NOT published "
            "(it would be erased when the next pause opens); said so.")
        self._narrate(
            "I can't change the drop side now — I'll put it where you said first.")

    def _drop_early_dest(self, why: str) -> None:
        """2026-09-25: forget the latched destination (log only)."""
        if self._early_dest is not None:
            self.get_logger().info(
                f"Early destination {self._early_dest['token']!r} dropped: {why}.")
            self._early_dest = None

    def _note_dest_answer(self) -> None:
        """2026-09-25: the router is about to put a destination on the
        resume topic. When that answers the destination question of a known
        task, the executor's resumed:<t> will say "Going to drop …" (see
        _on_interactive_status) — chat_node no longer says it at intent
        time, before anything accepted it."""
        if self._live_task() is not None and self._task_q() == "destination":
            self._dest_answer_pending = True

    def _set_task_object_label(self, label: str) -> None:
        """2026-09-25: set the executor's task_object_label parameter.

        Only once the task state is known (an old executor has no such
        parameter), through one cached client that never blocks this node's
        only thread: when the service is not ready the value is skipped with
        a log line (the executor re-reads the parameter at every cycle
        start, so the next set catches up)."""
        if self._task is None:
            return
        cli = self._task_label_cli
        if not cli.service_is_ready():
            self.get_logger().info(
                f"task_object_label={label!r} not set: executor parameter "
                "service not ready.")
            return
        request = SetParameters.Request()
        param = ParameterMsg()
        param.name = "task_object_label"
        param.value = ParameterValue()
        param.value.type = ParameterType.PARAMETER_STRING
        param.value.string_value = str(label or "")
        request.parameters = [param]

        def _done(fut, v=label):
            try:
                resp = fut.result()
            except Exception as exc:
                self.get_logger().warn(f"task_object_label set failed: {exc}")
                return
            if resp and resp.results and resp.results[0].successful:
                self.get_logger().info(f"Set task_object_label={v!r} on grasp_executor_lin_llm.")
            else:
                reason = resp.results[0].reason if resp and resp.results else "?"
                self.get_logger().warn(f"task_object_label refused by executor: {reason}")

        cli.call_async(request).add_done_callback(_done)

    def _claim_executor_resume(
        self, dest: str, why: str, *, force: bool = False, cmd: Optional[dict] = None
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
            # 2026-09-30: the answer went to the executor's pause. Whether the
            # executor acted on it is its next task_state phase (recorded).
            _t = self._live_task() or {}
            _c = cmd or self._routing_cmd
            if _c is not None:
                _c["question"] = _c.get("question") or str(_t.get("question") or "")
                if _c.get("question_id") in (None, ""):
                    _c["question_id"] = _t.get("question_id")
            self._cmd_event(_c, "accepted", why, route="executor_answer", resume_token=dest,
                            cycle_id=_t.get("cycle_id"),
                            parent_id=(self._pick_cmd or {}).get("command_id", ""))
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
            self._clarify = None    # 2026-09-28: a spoken pick answered any which-one question
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

    def _stop_path(self, src: str = "") -> None:
        """Stop is special: it must execute immediately and clear the queue,
        not wait behind whatever's currently running.

        2026-09-25 (review R7): ``src`` is the stop intent's
        object_attributes.source; "gui" (the E-stop, which already sent its
        own executor abort) makes _handle_stop skip the executor /abort.

        2026-09-25: moved out of _on_intent unchanged so that a reset can
        fall back to it against an old executor (_handle_reset). With the
        task state known no resume 'stop' / 'stay' is claimed first: the
        new executor resolves every pause from the ONE abort _handle_stop
        sends (a resume 'stop' used to send the arm home holding, and a
        second answer for the same pause is exactly what froze the GUI
        reset)."""
        if self._task is None:
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
            _dropped = list(self._intent_queue)
            self._intent_queue.clear()
        self._cancel_cmds("stop", _dropped)
        if cleared:
            self.get_logger().info(f"Stop received: cleared {cleared} queued intent(s).")
        self._clarify = None   # 2026-09-29: a which-one question and its destination end too
        self._handle_stop(src)
        self._busy = False
        self._busy_kind = ""
        self._try_dispatch_next()

    def _handle_reset(self, src: str) -> None:
        """2026-09-25: RESET — what the GUI's "↻ Reset pick" means: end the
        pick and go home keeping any held object, gripper untouched; clear
        the queue, the plan, the latch and the selection. NOT a freeze.

        The executor does the motion itself on /grasp_executor_lin_llm/reset
        (at a pause, the pause resolves; during motion it stops, then goes
        home; idle away from home it goes home). This node therefore sends
        NO abort and no resume — an abort next to the reset is what made
        the arm freeze at the grasp pose (the GUI race). The GUI publishes
        its own reset and pipeline abort (src == "gui"); for voice this
        node publishes both. Says nothing: chat_node already said
        "Resetting." and the cycle end says where the arm is.

        Old executor (nobody subscribes to the reset topic): today's stop
        path, which was the only thing a spoken "reset" ever did — for a
        VOICE reset only (2026-09-25, review R2): the GUI decided its own
        path from its own subscriber count and already sent /reset; an
        abort from here after it (a discovery lag can show this node 0
        subscribers) would freeze the arm the reset is taking home."""
        if src != "gui" and self.count_subscribers(self._executor_reset_topic) == 0:
            self.get_logger().info(
                f"Reset ({src}): no executor listens on "
                f"{self._executor_reset_topic} — old executor, running the stop path.")
            self._stop_path()
            return
        # 2026-09-25 (review R3): the executor reset and the pipeline abort
        # go out FIRST — _set_target_label below can block up to 1 s in
        # wait_for_service, and the arm must not wait for that.
        if src != "gui":
            try:
                self._executor_reset_pub.publish(String(data="voice"))
                self._pipeline_abort_pub.publish(String(data="stop"))
                self.get_logger().info(
                    "Reset: published executor reset 'voice' + pipeline abort.")
            except Exception as exc:
                self.get_logger().warn(f"Reset publish failed: {exc}")
        with self._queue_lock:
            cleared = len(self._intent_queue)
            _dropped = list(self._intent_queue)
            self._intent_queue.clear()
        self._cancel_cmds("reset", _dropped)
        self._followup_plan = ""
        self._after_cycle_intent = None   # 2026-09-25: a kept after-cycle move ends here too
        self._home_then = None             # 2026-09-25: and a drop pose waiting for its move home
        self._drop_early_dest(f"reset ({src})")
        self._clarify = None   # 2026-09-29: a which-one question and its destination end too
        self._dest_answer_pending = False
        self._releasing_line = ""   # 2026-09-25 (review R6): the plan is gone
        self._start_suppressed_upto = self._last_cycle_id
        self.get_logger().info(
            f"Reset ({src}): cleared {cleared} queued intent(s), the follow-up "
            "plan and the destination latch.")
        # The router's own goals: cancel, log only (no "Cancelled the motion.").
        for _name, _handle in (("gripper", self._active_gripper_handle),
                               ("named-pose", self._active_named_pose_handle)):
            if _handle is not None:
                try:
                    _handle.cancel_goal_async()
                    self.get_logger().info(f"Reset: cancelled the in-flight {_name} goal.")
                except Exception as exc:
                    self.get_logger().warn(f"Reset: {_name} cancel failed: {exc}")
        self._set_target_label("")
        self._busy = False
        self._busy_kind = ""
        self._try_dispatch_next()

    def _handle_stop(self, src: str = "") -> None:
        """Cancel everything intent_router currently owns AND abort the
        executor's pick cycle (candidate evaluation, approach, pauses).

        2026-09-25: also forgets what the stopped pick was going to do —
        the follow-up plan, the latched destination, a pending "Going to
        drop …" — and suppresses a "Starting grasp on …" still waiting in
        its 0.6 s grace for the cycle that was just stopped. Still exactly
        one abort.

        2026-09-25 (review R7): with ``src == "gui"`` (the E-stop's stop
        intent) the executor /abort is NOT sent here — the GUI already sent
        it (publish_llm_abort), and two aborts per press was one too many.
        Everything else stays: the trajectory stop (harmless), the pipeline
        abort, the cancels and the clears."""
        self._followup_plan = ""
        self._after_cycle_intent = None   # 2026-09-25: a kept after-cycle move ends here too
        self._home_then = None             # 2026-09-25: and a drop pose waiting for its move home
        self._drop_early_dest("stop")
        self._dest_answer_pending = False
        self._releasing_line = ""   # 2026-09-25 (review R6): the plan is gone
        self._start_suppressed_upto = self._last_cycle_id
        try:
            if src == "gui":
                self.get_logger().info(
                    "Stop (gui): the GUI sent the executor abort itself — not repeated.")
            else:
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
        _cmd = self._gripper_cmd = self._active_cmd   # 2026-09-30: whose goal this is

        def _on_response(fut):
            handle = fut.result()
            if handle is None or not handle.accepted:
                self._cmd_event(_cmd, "failed", "the gripper rejected the goal", route="gripper")
                self._narrate("The gripper rejected the command.")
                return
            self._cmd_event(_cmd, "accepted", "gripper goal accepted", route="gripper",
                            target={"gripper": "open" if int(command) == OperateGripper.Goal.OPEN else "close"})
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
        self._cmd_event(getattr(self, "_gripper_cmd", None), "completed" if result.success else "failed",
                        "" if result.success else str(result.error))
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
            # 2026-09-25: say only what is known. object_detected is the
            # shim's "stalled", and the adapter hard-codes stalled=False, so
            # "There was nothing in it." was said while holding the object
            # (and "closed on the object" could never be measured). What IS
            # known is whether the robot was already holding something —
            # the held-object memory, derived from the executor's task state
            # when it reports one.
            _held = (self._held_object_label or "").replace("_", " ")
            if _held:
                self._narrate(f"The gripper is closed. I'm still holding the {_held}.")
            else:
                self._narrate("The gripper is closed.")
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
        _cmd = self._named_pose_cmd = self._active_cmd   # 2026-09-30: whose goal this is
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
            self._cmd_event(_cmd, "failed", f"no result within {_budget_s:.0f} s (cancelled)")
            self._narrate(
                "That move is taking too long — I'm giving up on it.")
            self._mark_busy_done("named_pose")

        _t = threading.Timer(_budget_s, _watchdog)
        _t.daemon = True
        _t.start()

        def _on_response(fut):
            handle = fut.result()
            if handle is None or not handle.accepted:
                self._cmd_event(_cmd, "failed", f"the motion server rejected {pose_name!r}", route="named_pose")
                self._narrate("The motion server rejected the request.")
                return
            # accepted = the goal was taken, NOT that the arm moves yet
            self._cmd_event(_cmd, "accepted", f"goal {pose_name!r} accepted", route="named_pose",
                            target={"pose": pose_name, "leg": pose_name})
            self._active_named_pose_handle = handle
            handle.get_result_async().add_done_callback(self._on_named_pose_result)

        send_future.add_done_callback(_on_response)

    def _send_drop_pose(self, pose: str, velocity_scaling: float) -> None:
        """2026-09-25 (Majd, live: "Go to the drop right position.", then
        "Now go to the drop left." — the named-pose server blended
        vla_drop_right -> vla_via_right -> home -> vla_drop_left into ONE
        rounded spline that never really stopped at home, the arm ran into
        the table / mount and the pendant program stopped): a drop pose is
        reached from anywhere but home in TWO separate moves — home first,
        stopping there, then the drop pose (sent from _on_named_pose_result
        once home is reached; a failed or stopped home move drops it). When
        the router does not know where the arm is ("" after a start, a pick
        or a failed move) it goes home first as well — at worst one short
        extra move, never a sweep across the table."""
        here = str(getattr(self, "_current_pose_name", "") or "").strip()
        if pose == "home" or here in ("home", pose):
            self._send_named_pose(pose, "", velocity_scaling)
            return
        self._home_then = (pose, velocity_scaling)
        self.get_logger().info(
            f"drop pose {pose!r} from {here or 'an unknown pose'!r}: home first, "
            "as its own move, then the drop pose")
        self._send_named_pose("home", "", velocity_scaling)

    def _on_named_pose_result(self, fut) -> None:
        self._active_named_pose_handle = None
        # 2026-09-25: the drop pose waiting for this move home (see
        # _send_drop_pose) — taken now, so every failure path below drops it.
        _then = self._home_then
        self._home_then = None
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
            if _then and self._current_pose_name == "home":
                # 2026-09-25: home reached as its own move — now the drop
                # pose (_send_drop_pose); the router stays busy with it.
                self.get_logger().info(f"home reached — now {_then[0]!r}")
                self._send_named_pose(_then[0], "", _then[1])
                return
            self._cmd_event(getattr(self, "_named_pose_cmd", None), "completed",
                            f"reached {self._current_pose_name!r}")
        else:
            self._cmd_event(getattr(self, "_named_pose_cmd", None), "failed", str(result.error))
            self.get_logger().warn(f"MoveToNamedPose failed: {result.error}")
            self._narrate(f"Could not reach the pose. {result.error}".strip())
            self._pending_pose_name = ""
        self._mark_busy_done("named_pose")

    # ── lifecycle handlers ───────────────────────────────────────────────

    def _on_cycle_id(self, msg: Int64) -> None:
        cycle = int(msg.data)
        if cycle == self._last_cycle_id:
            return
        # 2026-09-30: the grasp cycle that executes the committed pick
        if self._pick_cmd is not None and self._pick_cmd.get("cycle_id") in (None, ""):
            self._cmd_event(self._pick_cmd, "cycle", cycle_id=cycle)
        if cycle < self._last_cycle_id:
            # Cycle ids only ever grow within one pipeline run — a smaller
            # id means the pipeline was restarted. Whatever label lock was
            # taken against the OLD numbering is meaningless now.
            self.get_logger().info(
                "pipeline restarted — clearing the stale target-label lock"
            )
            self._last_target_label_lock_cycle = -1
            # 2026-09-25: the same for the start-line suppression marks —
            # against the new numbering they would silence every start.
            self._last_done_cycle_id = -1
            self._start_suppressed_upto = -1
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
            # 2026-09-25: a cycle that already ended (cycle_done inside the
            # 0.6 s grace — e.g. no candidate, or a stop) or that a stop /
            # reset cut off gets no "Starting grasp" after the fact.
            if cycle <= self._last_done_cycle_id or cycle <= self._start_suppressed_upto:
                self.get_logger().info(
                    f"cycle_id={cycle}: start narration suppressed (ended "
                    f"{self._last_done_cycle_id}, stopped/reset at "
                    f"{self._start_suppressed_upto}).")
                return
            # 2026-09-25: while the gripper holds an object the executor
            # refuses the pick (its cycle_done says "still holding …", which
            # _on_cycle_done answers) — no "Starting grasp" before that, and
            # no start line per pointing attempt.
            _t = self._live_task()
            if _t is not None and _t.get("holding"):
                self.get_logger().info(
                    f"cycle_id={cycle}: start narration suppressed (still holding "
                    f"{_t.get('object')!r}; the executor refuses this pick).")
                return
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
            # 2026-09-25: pointing / GUI selections name the object for the
            # executor's task state too (it re-reads the label while probing
            # and approaching, and freezes it at the grasp verdict).
            self._set_task_object_label(cls)
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
            self._destination_open = True
            # A NEW pause is open: the one-resume-per-pause window reopens.
            self._resume_claimed = False
            label = self._last_target_label or self._held_object_label or ""
            # The pause fires AFTER a confirmed close — the gripper IS
            # holding the object NOW. Publishing held_object only at
            # cycle end left the LLM claiming "I'm not holding anything"
            # mid-pause and refusing the drop command (Majd, 2026-08-12).
            # 2026-09-25: legacy only — with the task state known the held
            # object follows the executor's own holding (_sync_held_object).
            if label and self._task is None:
                self._set_held_object(label)
            # 2026-09-25: a destination said during the approach (latched)
            # answers this pause now, and the question is not asked. Only
            # for the cycle it was said in (a latch bound to another cycle
            # is dropped and the question is asked as usual).
            latch, self._early_dest = self._early_dest, None
            if latch is not None:
                try:
                    _cid = int((self._task or {}).get("cycle_id") or 0)
                except (TypeError, ValueError):
                    _cid = 0
                if latch["cycle"] is None or self._task is None or latch["cycle"] == _cid:
                    self._dest_answer_pending = True
                    self._claim_executor_resume(latch["token"], "early destination",
                                                cmd=latch.get("cmd"))
                    self._narrate(
                        f"Got the {label.replace('_', ' ')}." if label else "Got it.")
                    return
                self.get_logger().info(
                    f"Early destination {latch['token']!r} dropped at the pause: "
                    f"latched for cycle {latch['cycle']}, pause of cycle {_cid}.")
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
            self._destination_open = False
            # A NEW pause is open: the one-resume-per-pause window reopens.
            self._resume_claimed = False
            # A NEW release pause: the keep-holding verdict is not in yet.
            self._release_skipped_this_cycle = False
            # 2026-09-25: legacy only (task state owns the held object).
            if self._last_target_label and self._task is None:
                self._set_held_object(self._last_target_label)
            # The arm has ARRIVED at the drop pose (executor-driven move)
            # — record it so "which position are you?" answers truthfully.
            dest = getattr(self, "_executor_drop_dest", "")
            self._publish_current_pose(dest or "the drop position")
            # Arm is at the drop pose, gripper still closed. Wait for
            # user to say "release" / "open it" / "give it to me".
            self._pick_awaiting_release = True
            _dest = getattr(self, "_executor_drop_dest", "")
            # 2026-09-25: named by left/right in the pose name (vla_drop_*).
            _spoken = self._spoken_pose(_dest, "drop")
            self._stayed_pose = ""
            if self._followup_plan:
                # The operator already said what comes next (2026-09-23):
                # answer the pause instead of asking.
                plan, self._followup_plan = self._followup_plan, ""
                token = "release_home" if plan == "release_home" else "release"
                _fcmd, self._followup_cmd = self._followup_cmd, None
                if self._claim_executor_resume(token, f"follow-up said earlier: {plan}", cmd=_fcmd):
                    # 2026-09-25 (review R6): said ONCE, on the executor's
                    # 'releasing' (it used to be said here AND there).
                    self._releasing_line = ("Opening the gripper and going home."
                                            if plan == "release_home" else "")
                    return
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
            # 2026-09-25 (review R6): a follow-up plan's line replaces it
            # (spoken once, here). Review R5: a side claimed at the release
            # question that opens IN PLACE ends here too — the arm was
            # already there, so no "Going to drop …".
            line, self._releasing_line = (self._releasing_line or "Opening the gripper."), ""
            self._dest_answer_pending = False
            self._narrate(line)
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
            # 2026-09-25 (review R5): except for a side the ROUTER claimed at
            # the release question (drop_<side>+release): chat_node no
            # longer acknowledges it, and this redirect is the executor
            # accepting it — the carry+release has no resumed: on success.
            if self._dest_answer_pending:
                self._dest_answer_pending = False
                self._say_going_to(data.split(":", 1)[1].strip())
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
            # 2026-09-25: legacy only (task state owns the held object).
            if _kept2 and self._task is None:
                self._set_held_object(_kept2)
        elif data == "stopped_holding":
            # Abort-freeze at the release pause: arm parked at the drop
            # pose, gripper closed on the object. Do NOT claim "home".
            self._pick_awaiting_release = False
            _kept3 = self._last_target_label or self._held_object_label
            # 2026-09-25: legacy only (task state owns the held object).
            if _kept3 and self._task is None:
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
            # 2026-09-25: legacy only (task state owns the held object).
            if self._task is None:
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
            # 2026-09-25: legacy only (task state owns the held object).
            if _kept and self._task is None:
                self._set_held_object(_kept)
            self._narrate(
                "I'll keep holding it and take it back home."
            )
            self._publish_current_pose("home")
        elif data == "no_grasp":
            self._pick_awaiting_destination = False
            # The executor now lifts clear and ASKS before homing
            # (2026-08-26) — don't promise "going back home" here.
            # 2026-09-22 (Majd: "he must always check before saying such a
            # thing"): the executor only knows the fingers closed and met
            # nothing (grasp pin low, no stall). Whether the object moved,
            # was pushed by the early close or the pose missed is not
            # known — say what was measured.
            self._narrate(
                "I closed the gripper but caught nothing."
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
            self._destination_open = False
            # The executor is now moving to the chosen drop pose — track
            # it so /robot/current_pose is true for EXECUTOR-driven moves
            # too (2026-08-25, Majd: "Morty must know what position he
            # is" — it claimed 'home' while standing at drop-left because
            # only router-commanded moves updated the pose).
            self._executor_drop_dest = data.split(":", 1)[1].strip()
            # 2026-09-25: the executor ACCEPTED a destination the router
            # itself answered (the latch, or a claim at the question) — only
            # now is "Going to drop right." true. chat_node withholds that
            # line at intent time once the task state is known; a GUI
            # button's resumed: finds the flag clear and stays silent.
            if self._dest_answer_pending:
                self._dest_answer_pending = False
                self._say_going_to(self._executor_drop_dest)
        elif data == "reset_home":
            # 2026-09-25: the executor is resetting (going home, keeping any
            # held object): whatever question was open is gone. Bookkeeping
            # only — the cycle end says where the arm ended up.
            self._destination_open = False
            self._pick_awaiting_release = False
            self._pick_awaiting_home = False
        # The LLM's confirm_drop reply already narrates the destination.

    def _say_going_to(self, target: str) -> None:
        """2026-09-25: the line for a destination the executor accepted from
        the router — on resumed:<t> (destination question) and, since review
        R5, on redirect:<t> (a side claimed at the release question)."""
        if target == "grasp_spot":
            return   # 2026-09-29: release here — "Opening the gripper." follows at once
        if "right" in target:
            self._narrate("Going to drop right.")
        elif "left" in target:
            self._narrate("Going to drop left.")
        elif target in ("home", self._home_pose):
            self._narrate("Going home.")
        else:
            self._narrate(f"Going to the {target.replace('_', ' ')}.")

    def _defer_after_cycle(self, action: str, intent: dict) -> bool:
        """2026-09-25 (Majd, live): a go-home / drop-side move said while the
        cycle's LAST move runs — live task state, no question open, gripper
        empty, phase carrying (the home question's move to a drop side) or
        returning_home — is kept and carried out when the cycle ends
        (_on_cycle_done). It used to go out as a resume into a pause that
        never opened again, so "Now back home." was lost and the model later
        claimed it was "already heading home". True when handled here."""
        task = self._live_task()
        if task is None or self._task_q() != "" or task.get("holding"):
            return False
        phase = str(task.get("phase") or "")
        if phase not in ("carrying", "returning_home"):
            return False
        attrs = intent.get("object_attributes")
        attrs = attrs if isinstance(attrs, dict) else {}
        move = action
        if action == "confirm_drop":
            move = _DESTINATION_WORDS.get(str(attrs.get("destination") or "").strip().lower(), "")
            move = "go_home" if move == "home" else move
        if move not in ("go_home", "drop_left", "drop_right"):
            return False
        if move == "go_home" and phase == "returning_home":
            self._narrate("I'm already on my way home.")
            return True
        self._cmd_event(self._routing_cmd, "deferred", "kept until the running cycle ends")
        self._after_cycle_intent = {"command_id": (self._routing_cmd or {}).get("command_id", ""),
                                    "action_intent": move, "object_intent": None,
                                    "object_attributes": None}
        where = {"go_home": "go home", "drop_left": "go to the drop left position",
                 "drop_right": "go to the drop right position"}[move]
        self.get_logger().info(
            f"after-cycle move kept: {move!r} (phase={phase!r}, said as {action!r})")
        self._narrate(f"Okay — I'll {where} when this move ends.")
        return True

    def _run_after_cycle_intent(self) -> None:
        """2026-09-25: carry out the move kept by _defer_after_cycle once the
        cycle has ended — 0.6 s later (the executor's idle task state and
        cycle_done travel on different topics), through the normal intent
        path, announced here (no chat line was spoken for it)."""
        intent = self._after_cycle_intent
        self._after_cycle_intent = None
        if not intent:
            return

        def _go() -> None:
            self.destroy_timer(timer)   # destroy, not cancel (see _say_start)
            move = intent["action_intent"]
            self._narrate({"go_home": "Going home.", "drop_left": "Going to drop left.",
                           "drop_right": "Going to drop right."}[move])
            self._on_intent(String(data=json.dumps(intent)))

        timer = self.create_timer(0.6, _go)

    def _on_cycle_done(self, msg: String) -> None:
        data = (msg.data or "").strip()
        if not data:
            return
        outcome = data.split("|", 1)[0].lower()
        # 2026-09-30: the pick command of THIS cycle ends ("failed|0|..." =
        # the pipeline gave the label up before any cycle started)
        _parts = data.split("|")
        try:
            _cyc = int(_parts[1]) if len(_parts) > 1 else None
        except ValueError:
            _cyc = None
        _pc = self._pick_cmd
        if _pc is not None and (_cyc == _pc.get("cycle_id") or (_cyc == 0 and _pc.get("cycle_id") is None)):
            self._cmd_event(_pc, "completed" if outcome == "done" else "failed",
                            "|".join(_parts[2:]), cycle_id=_cyc if _cyc else None)
            self._pick_cmd = None
        self.get_logger().info(f"cycle_done received: {data!r} (outcome={outcome})")
        # 2026-09-25: a move kept for after this cycle (_defer_after_cycle)
        # runs only after a normal end; a failed / stopped / reset cycle
        # drops it (stop means freeze). While it is pending the usual
        # "…What's next?" line is not said — the move announces itself.
        _after_pending = outcome == "done" and self._after_cycle_intent is not None
        if _after_pending:
            self._run_after_cycle_intent()    # fires 0.6 s later, see there
        else:
            self._after_cycle_intent = None
        self._followup_plan = ""
        self._destination_open = False
        # 2026-09-25: the cycle is over — remember its id (a late "Starting
        # grasp" for it is suppressed in _say_start), drop a destination
        # latched for it and a pending "Going to drop …", and clear the
        # executor's object label so the next cycle cannot inherit it.
        # Also runs for /grasp_pipeline/selection_result (same grammar).
        try:
            _done_id = int(data.split("|")[1])
        except (IndexError, ValueError):
            _done_id = 0
        if _done_id > 0:
            self._last_done_cycle_id = _done_id
        self._drop_early_dest(f"cycle_done {data!r}")
        self._dest_answer_pending = False
        # 2026-09-25 (reviews R1, R6): per-pick memory ends with the cycle.
        self._releasing_line = ""
        self._pick_fixed_side = ""
        self._set_task_object_label("")
        if self._narrate_lifecycle:
            # Silent on success — same reasoning as MoveToNamedPose: avoid
            # extra TTS that the mic loops back. Failure still narrates,
            # WITH the executor's reason when it sent one
            # ("failed|<cycle_id>|<reason>", 2026-08-11).
            if outcome == "failed":
                fields = data.split("|", 2)
                reason = fields[2].strip() if len(fields) > 2 else ""
                if reason.lower().startswith("still holding"):
                    # 2026-09-25 (Majd approved): the executor refused a
                    # pointing / GUI pick because the gripper still holds an
                    # object ("still holding the <label> — tell me where to
                    # drop it first"). Not a failed grasp; rate-limited so
                    # repeated pointing cannot spam it.
                    _m = re.match(r"still holding the (.+?)(?:\s+[—–-]+\s|,|$)", reason,
                                  flags=re.IGNORECASE)
                    self._say_still_holding(_m.group(1).strip() if _m else "",
                                            rate_limited=True)
                elif reason == "stopped by your command":
                    # Operator STOP at a pause: the arm froze in place,
                    # nothing resets — saying "Resetting." here was a
                    # lie (2026-08-31 narration audit).
                    self._narrate("Stopped. I'm holding position.")
                elif reason == "reset by the operator" and self._task is not None:
                    # 2026-09-25: a reset is not a failed grasp ("The grasp
                    # failed: reset by the operator. Resetting." was said
                    # while the arm stood still). Say where the arm ended
                    # and what it holds — from the task state's `last`,
                    # which is published before cycle_done but on another
                    # topic, hence the short grace (see _say_reset_outcome).
                    self._say_reset_outcome(_done_id)
                elif fields[1].strip() == "0" and reason:
                    # No cycle ever ran (pipeline gave the label up): the
                    # reason is already a sentence; nothing is resetting.
                    self._narrate(reason)
                elif reason.startswith("tried "):
                    # The executor's reason is a planner report ("tried 10
                    # grasps: 8 had no IK …"); the operator needs the
                    # conclusion, the log keeps the report (2026-09-23).
                    what = (self._last_target_label or self._held_object_label
                            or "it").replace("_", " ")
                    self._narrate(
                        f"I couldn't find a way to reach the {what} from here. "
                        "Move it a bit closer, or ask for another object.")
                elif reason:
                    self._narrate(f"The grasp failed: {reason}. Resetting.")
                else:
                    self._narrate("The grasp failed. Resetting.")
            elif outcome == "done" and _after_pending:
                self.get_logger().info("Grasp cycle completed successfully "
                                       "(a kept move follows — end line skipped).")
                self._stayed_pose = ""
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
                    _sp = self._spoken_pose(stayed, stayed.replace("_", " "))
                    # 2026-09-25 (Majd): at a drop pose (left / right in its
                    # name, configured vla_drop_* included) say where the arm
                    # is; other stayed poses keep today's line.
                    if _sp in ("drop left", "drop right"):
                        self._narrate(f"I'm at the {_sp} position. What's next?")
                    else:
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
        # 2026-09-25: with the task state known the executor's own holding
        # decides (_sync_held_object) — a stop or reset that ends the cycle
        # still holding no longer wipes the memory. The chain below is the
        # legacy fallback — also for a dead executor (review R4: _live_task).
        # 2026-09-25: a pick the executor refused because it still holds
        # an object never touches the held-object memory or the pose.
        _refused_holding = outcome == "failed" and (
            data.split("|", 2)[2].strip().lower().startswith("still holding")
            if data.count("|") >= 2 else False)
        if self._live_task() is not None:
            self._sync_held_object()
        elif _refused_holding:
            pass
        elif outcome == "done" and self._release_skipped_this_cycle:
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
        # 2026-09-25: the stayed pose from the PAYLOAD first — the narration
        # above clears _stayed_pose, so a stayed end (the arm left at a drop
        # pose, e.g. a side named at "Should I go back home?") used to be
        # booked as "home" here.
        _payload_stayed = ""
        for _f in data.split("|")[2:]:
            if _f.startswith("stayed:"):
                _payload_stayed = _f.split(":", 1)[1].strip()
                break
        if _refused_holding:
            pass   # 2026-09-25: the refused pick never moved the arm
        elif "stopped by your command" in _reason:
            self._publish_current_pose("")
        elif _payload_stayed or getattr(self, "_stayed_pose", ""):
            self._publish_current_pose(_payload_stayed or self._stayed_pose)
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

    def _say_reset_outcome(self, cycle_id: int) -> None:
        """2026-09-25: the spoken end of a reset, from the task state.

        The executor publishes its idle task state (with ``last`` filled)
        BEFORE cycle_done, but on another topic, so the order of arrival is
        not guaranteed: wait 0.3 s (a one-shot timer, destroyed in its own
        callback like _say_start — cancelled Humble timers hot-loop the
        spin), then say where the arm is and what it still holds. When
        ``last`` does not describe this cycle: "Pick reset."."""
        def _say() -> None:
            self.destroy_timer(timer)
            task = self._task or {}
            last = task.get("last") if isinstance(task.get("last"), dict) else {}
            try:
                last_id = int(last.get("cycle_id"))
            except (TypeError, ValueError):
                last_id = None
            if last_id is None or last_id != cycle_id:
                self.get_logger().info(
                    f"Reset of cycle {cycle_id}: task state's last cycle is "
                    f"{last_id} — saying the fallback line.")
                self._narrate("Pick reset.")
                return
            tail = ""
            if task.get("holding"):
                what = (str(task.get("object") or "") or self._held_object_label
                        or "object").replace("_", " ")
                tail = f", still holding the {what}"
            if last.get("at_home"):
                self._narrate(f"Reset — I'm back home{tail}.")
            else:
                self._narrate(f"Reset — but I couldn't get back home{tail}.")

        timer = self.create_timer(0.3, _say)

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
        if self._robot_program_running is False:
            self._poll_dashboard_program_state()
        self._publish_robot_status()
        if not self._list_controllers_cli.service_is_ready():
            return  # controller manager not up yet; quiet poll
        req = ListControllers.Request()
        future = self._list_controllers_cli.call_async(req)
        future.add_done_callback(self._on_list_controllers_done)

    def _publish_robot_status(self) -> None:
        """program_running: null while the flag was never heard, else the
        flag OR a live witness (see _check_robot_ready)."""
        now = self.get_clock().now().nanoseconds / 1e9
        flag = self._robot_program_running
        witness = self._program_running_witness(now)
        running = None if flag is None else bool(flag)
        self._robot_status_pub.publish(String(data=json.dumps({
            "program_running": running,
            "flag": flag,
            "witness": witness,
            "controller": self._last_controller_state,
            "dashboard": self._dashboard_state,
        })))

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
        # 2026-09-22: controller_stopper deactivates the controller when the
        # program stops and re-activates it when the program reconnects.
        # Re-activating it ourselves while the program is down only makes
        # goals get accepted and aborted; leave it to the stopper then.
        if target_state != "active" and self._robot_program_running is not False:
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
        # 2026-09-25: a latched destination older than a whole pick may be
        # (queue_pick_timeout_s) belongs to nothing any more.
        if (self._early_dest is not None
                and time.monotonic() - self._early_dest["t"] > self._pick_timeout_s):
            self._drop_early_dest(f"older than {self._pick_timeout_s:.0f} s")
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
            self._cmd_event(self._pick_cmd, "failed", f"no cycle_done within {self._pick_timeout_s:.0f} s")
            self._pick_cmd = None
            self._mark_busy_done("pick")

    # ── narration ────────────────────────────────────────────────────────

    def _narrate(self, text: str, *, interrupt: bool = False) -> None:
        """Speak a status line. Polite by default: waits for whatever is
        currently being spoken (chat reply, other narration) to finish —
        interrupt_current=True chopped Morty's sentences mid-word whenever
        the executor reached a lifecycle event (Majd, 2026-08-12). Pass
        interrupt=True only for time-critical lines (stop, slip)."""
        if self._dispatch_narr is not None and text:
            self._dispatch_narr.append(text)   # 2026-09-30: the reason of a refused command
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
