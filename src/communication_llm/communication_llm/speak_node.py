"""
speak_node.py
=============

ROS action server for text-to-speech. Exposes the
``morty_interfaces/action/Speak`` action on ``/speak``: any node that wants
Morty to say something sends a goal with the text, and this node synthesizes
it to audio and plays it through the system speakers.

The actual TTS engine lives in :mod:`tts` (Kokoro neural TTS by default, with
a pyttsx3 + espeak fallback for German / fast / low-resource paths). This
file is the ROS interface around it:

* ``ActionServer`` with cancel support — speaking can be interrupted mid-
  sentence by sending a new goal (the previous one is cancelled).
* Phase callbacks (e.g. ``"started"``, ``"finished"``) are also published as
  ``std_msgs/String`` so the chat layer can co-ordinate (e.g. mute the mic
  while speaking to avoid self-echo).
* Uses a ``ReentrantCallbackGroup`` so a cancel can be processed while a
  ``Speak`` goal is still streaming audio.

This is one of the heaviest nodes on startup because Kokoro loads its neural
voice model. Once loaded, synthesis is near real-time.
"""
from __future__ import annotations

import threading
import time

import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.action.server import ServerGoalHandle
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_msgs.msg import Bool, String

from morty_interfaces.action import Speak

from .tts import KokoroTTS


class SpeakActionServer(Node):
    def __init__(self) -> None:
        super().__init__("speak_action_server")

        self.declare_parameter("server_action_name", "/speak")
        self.declare_parameter("voice", "bf_emma")
        self.declare_parameter("volume_gain", 1.6)
        self.declare_parameter("lang_code", "a")
        self.declare_parameter("repo_id", "hexgrad/Kokoro-82M")
        self.declare_parameter("sample_rate", 24000)
        # Skip Kokoro entirely and use pyttsx3 (espeak-ng) as the only TTS.
        # ~10x faster on CPU at the cost of a robotic voice. Useful for
        # debugging on a VM with no GPU.
        self.declare_parameter("force_pyttsx3", False)

        self._tts = KokoroTTS(
            voice=str(self.get_parameter("voice").value),
            volume_gain=float(self.get_parameter("volume_gain").value),
            lang_code=str(self.get_parameter("lang_code").value),
            repo_id=str(self.get_parameter("repo_id").value),
            sample_rate=int(self.get_parameter("sample_rate").value),
            logger=self.get_logger(),
            force_pyttsx3=bool(self.get_parameter("force_pyttsx3").value),
        )

        if not self._tts.available:
            self.get_logger().error(
                "No TTS backend available (neither Kokoro nor pyttsx3 initialised). "
                "Speak goals will be rejected."
            )

        self._cb = ReentrantCallbackGroup()
        self._goal_lock = threading.Lock()
        self._active_goal: ServerGoalHandle | None = None
        # Set when an in-flight _execute returns (i.e. speak_blocking
        # released the TTS lock and the audio stream is closed). A new
        # interrupt-style goal waits on this event so we never have two
        # speak_blocking calls overlapping in the audio device.
        self._idle_event = threading.Event()
        self._idle_event.set()  # start in "idle" state
        # FIFO tickets (2026-08-26): several polite goals waiting on the
        # same idle event woke in ARBITRARY thread order — "Should I go
        # back home?" played before the "object was moved" line queued
        # 2 s earlier. Arrival order is restored via a ticket line;
        # interrupt-style goals may jump it.
        self._ticket_counter = 0
        self._ticket_waiting: list = []

        # Broadcast every accepted Speak text on /speech/ai_said so
        # echo-suppression filters in OTHER nodes (chat_node) can see
        # what was spoken — including narrations originating from
        # intent_router, the pipeline, or anywhere else. Without this
        # the filter only catches chat_node's own utterances and the
        # other ones loop back through the mic.
        self._ai_said_pub = self.create_publisher(
            String, "/speech/ai_said", 10
        )
        # TTS busy flag (2026-08-25, Majd: trial videos cut Morty's
        # speech mid-sentence) — True while a Speak goal is playing,
        # False when playback finished. The MMHRI recorder waits on it.
        self._tts_active_pub = self.create_publisher(
            Bool, "/speech/tts_active", 10
        )
        # Barge-in (2026-09-23): speech_input_node says a HUMAN started
        # talking while we play (speaker fingerprint, not Morty's echo).
        # Stop the sentence within one 200 ms write window and listen.
        self.create_subscription(Bool, "/speech/barge_in", self._on_barge_in, 10)
        # The operator has the floor (2026-09-24, Majd: "he keeps answering
        # the old question"). Stopping the sentence was not enough: the rest
        # of the answer was already queued here, one goal per sentence, and
        # played on while the operator was still talking. On barge-in every
        # queued goal is now dropped and new speech waits while
        # /speech/state says the operator is still speaking (at most
        # 10 s), so the next thing Morty says answers what was just said.
        self._flush_mark = 0            # tickets below this were queued before the barge-in
        self._hold_until = 0.0          # monotonic deadline of the hold ...
        self._operator_talking = False  # ... which also ends when the operator stops
        self.create_subscription(String, "/speech/state", self._on_listener_state, 10)

        self._server = ActionServer(
            self,
            Speak,
            str(self.get_parameter("server_action_name").value),
            execute_callback=self._execute,
            goal_callback=self._goal_cb,
            cancel_callback=self._cancel_cb,
            callback_group=self._cb,
        )

        self.get_logger().info(
            f"Speak action server ready on {self.get_parameter('server_action_name').value}"
        )

    def _on_barge_in(self, msg: Bool) -> None:
        if not bool(msg.data):
            return
        with self._goal_lock:
            active = self._active_goal is not None
            queued = len(self._ticket_waiting)
            self._flush_mark = self._ticket_counter
            self._hold_until = time.monotonic() + 10.0
        if active or queued:
            self.get_logger().info(
                "Barge-in: the operator is talking — stopping playback"
                + (f", dropping {queued} queued sentence(s)" if queued else "")
                + "; new speech waits until they finish.")
        if active:
            self._tts.stop()

    def _on_listener_state(self, msg: String) -> None:
        """speech_input_node's 'speech' / 'idle' / 'muted'. Anything but
        'speech' after a barge-in means the operator finished: release."""
        talking = str(msg.data) == "speech"
        self._operator_talking = talking
        if not talking:
            self._hold_until = 0.0

    def _goal_cb(self, _request) -> GoalResponse:
        if not self._tts.available:
            return GoalResponse.REJECT
        interrupt = bool(_request.interrupt_current)
        with self._goal_lock:
            in_flight = self._active_goal is not None
            if in_flight and interrupt:
                self.get_logger().info("Interrupting in-flight Speak per goal flag.")
                self._tts.stop()
                self._idle_event.clear()
        # Two cases, both ACCEPT:
        #   - interrupt-style goal: wait briefly for the old _execute to
        #     unwind so we don't double-up on the audio device.
        #   - polite (non-interrupt) goal: accept; the new _execute will
        #     block on the TTS lock until the previous speak_blocking
        #     finishes. This effectively queues short follow-up replies
        #     (e.g. the LLM joke after the "Going to ready" narration)
        #     instead of dropping them, which is what the user wants.
        if interrupt:
            if not self._idle_event.wait(timeout=2.0):
                self.get_logger().warn(
                    "Previous Speak did not idle within 2 s; accepting new goal anyway."
                )
        elif in_flight:
            self.get_logger().info(
                "Queueing Speak behind in-flight goal (interrupt_current=false)."
            )
        return GoalResponse.ACCEPT

    def _cancel_cb(self, goal_handle) -> CancelResponse:
        # Stop the audio only when the cancelled goal is the one PLAYING; a
        # queued goal is skipped when its turn comes (_execute). Stopping on
        # every cancel would cut a router status line whenever chat_node
        # drops the queued sentences of an interrupted answer.
        with self._goal_lock:
            playing = (self._active_goal is not None
                       and bytes(self._active_goal.goal_id.uuid) == bytes(goal_handle.goal_id.uuid))
        if playing:
            self._tts.stop()
        return CancelResponse.ACCEPT

    def _publish_phase(self, goal_handle: ServerGoalHandle, phase: str, progress: float) -> None:
        fb = Speak.Feedback()
        fb.phase = phase
        fb.progress = float(progress)
        try:
            goal_handle.publish_feedback(fb)
        except Exception:
            pass

    def _execute(self, goal_handle: ServerGoalHandle) -> Speak.Result:
        # If a previous goal is still running (the queued / polite case),
        # wait for it to finish before we touch the TTS. Cap the wait so
        # a wedged backend can't deadlock us forever. The 30-s ceiling
        # is well over the longest reasonable utterance.
        with self._goal_lock:
            my_ticket = self._ticket_counter
            self._ticket_counter += 1
            self._ticket_waiting.append(my_ticket)
        jump = False
        try:
            jump = bool(goal_handle.request.interrupt_current)
        except Exception:
            pass

        def _claim() -> None:
            self._active_goal = goal_handle
            try:
                self._ticket_waiting.remove(my_ticket)
            except ValueError:
                pass
            self._idle_event.clear()

        deadline = 30.0
        dropped = ""
        while True:
            held = False
            with self._goal_lock:
                if not jump and my_ticket < self._flush_mark:
                    dropped = "dropped: the operator started talking"
                elif goal_handle.is_cancel_requested:
                    dropped = "cancelled"
                if dropped:
                    try:
                        self._ticket_waiting.remove(my_ticket)
                    except ValueError:
                        pass
                    break
                held = (not jump and self._operator_talking
                        and time.monotonic() < self._hold_until)
                if not held and self._active_goal is None and (
                        jump or (self._ticket_waiting
                                 and self._ticket_waiting[0] == my_ticket)):
                    _claim()
                    break
            if held:
                time.sleep(0.05)        # the operator still has the floor
                continue
            if not self._idle_event.wait(timeout=deadline):
                self.get_logger().warn(
                    "Queue wait exceeded 30 s; running this Speak anyway."
                )
                with self._goal_lock:
                    _claim()
                break
            # Event is set but it's not our turn yet — yield so the
            # rightful head of the line can claim and clear it.
            time.sleep(0.02)
        if dropped:
            result = Speak.Result()
            result.success = False
            result.error = dropped
            try:
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                else:
                    goal_handle.abort()
            except Exception:
                pass
            return result

        try:
            self._tts_active_pub.publish(Bool(data=True))
        except Exception:
            pass
        try:
            if goal_handle.is_cancel_requested:   # cancelled while it waited its turn
                goal_handle.canceled()
                result = Speak.Result()
                result.success = False
                result.error = "cancelled"
                return result
            goal: Speak.Goal = goal_handle.request

            if goal.voice:
                self._tts.set_voice(goal.voice)
            if goal.volume_gain:
                self._tts.set_volume_gain(goal.volume_gain)

            self.get_logger().info(f"Speak: {goal.text!r}")
            # Broadcast to /speech/ai_said so any echo-suppression
            # filter elsewhere can record what we're about to say.
            try:
                self._ai_said_pub.publish(String(data=str(goal.text)))
            except Exception:
                pass

            phase_progress = {"synthesizing": 0.25, "playing": 0.6, "finished": 1.0}

            def on_phase(phase: str) -> None:
                self._publish_phase(goal_handle, phase, phase_progress.get(phase, 0.0))

            ok = self._tts.speak_blocking(goal.text, on_phase=on_phase)

            result = Speak.Result()

            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                result.success = False
                result.error = "cancelled"
                return result

            if ok:
                goal_handle.succeed()
                result.success = True
                result.error = ""
            else:
                goal_handle.abort()
                result.success = False
                result.error = "TTS failed"
            return result
        finally:
            try:
                self._tts_active_pub.publish(Bool(data=False))
            except Exception:
                pass
            with self._goal_lock:
                self._active_goal = None
                # Audio stream is closed at this point; the next
                # interrupt-style goal may proceed.
                self._idle_event.set()

    def destroy_node(self) -> bool:
        self._tts.close()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SpeakActionServer()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
