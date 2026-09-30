"""ROS integration check of the recording path on an ISOLATED domain (no robot).

    ROS_DOMAIN_ID=77 python3 test/ros_integration_lifecycle.py <work dir>

Runs the REAL intent router (in this process; its robot-readiness check
and executor-parameter calls stubbed — no driver here) and the REAL VLA
recorder (its own process), against fakes of the robot and cameras, the
named-pose and gripper action servers and the grasp executor's pause /
resume protocol. The scenario replays the reviewed failure patterns and
then reads the saved episode:

  1. look_right and go_home published 8 ms apart (trial 114);
  2. a duplicate go_home (rejected) and a queued look_left cancelled by stop;
  3. the executor paused at "Should I open the gripper?" and the operator's
     "open the gripper and go back home" (open_gripper + then=release_home,
     trial 126) — released, returned home, then an automatic close;
  4. stop with a bare "stop" (must not become success), a repeated stop,
     then the same trial folder started again (the first episode must be
     preserved, not deleted).
Exit code 0 = every check passed; the checks print as they run.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import rclpy
import rclpy.duration
from rclpy.action import ActionServer
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

from control_msgs.msg import JointTrajectoryControllerState
from geometry_msgs.msg import PoseStamped, TransformStamped
from morty_interfaces.action import MoveToNamedPose, OperateGripper
from sensor_msgs.msg import CameraInfo, CompressedImage, JointState
from std_msgs.msg import Bool, String
from tf2_msgs.msg import TFMessage
from ur_msgs.msg import Digital, IOStates

ARM = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
POSES = {"home": [0.0] * 6, "detect_right": [0.6, -0.2, 0.3, 0.0, 0.1, 0.0], "detect_left": [-0.6, -0.2, 0.3, 0.0, 0.1, 0.0],
         "vla_drop_right": [0.4, 0.3, -0.3, 0.2, 0.0, 0.0]}
LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
FAILED = []


def check(ok, what):
    print(("  PASS " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILED.append(what)


class FakeWorld(Node):
    """Robot, cameras, named-pose + gripper servers, executor pause protocol."""

    def __init__(self):
        super().__init__("fake_world_claude")
        self.q = np.zeros(6)
        self.grip = 1.0
        self.lock = threading.Lock()
        self.js = self.create_publisher(JointState, "/joint_states", 10)
        self.tcp = self.create_publisher(PoseStamped, "/tcp_pose_broadcaster/pose", 10)
        self.io = self.create_publisher(IOStates, "/io_and_status_controller/io_states", 10)
        self.ctrl = self.create_publisher(JointTrajectoryControllerState,
                                          "/scaled_joint_trajectory_controller/controller_state", 10)
        self.tf = self.create_publisher(TFMessage, "/tf", 10)
        self.tfs = self.create_publisher(TFMessage, "/tf_static", LATCHED)
        self.cam = {"wrist": self.create_publisher(CompressedImage, "/camera/camera/color/image_raw/compressed", 10),
                    "exterior": self.create_publisher(CompressedImage, "/astra/color/compressed_slow", 10)}
        self.info = {"wrist": self.create_publisher(CameraInfo, "/camera/camera/color/camera_info", 10),
                     "exterior": self.create_publisher(CameraInfo, "/astra/color/camera_info", 10)}
        self.gcmd = self.create_publisher(String, "/gripper_controller/command_event", 10)
        self.task = self.create_publisher(String, "/grasp_executor_lin_llm/task_state", LATCHED)
        self.status = self.create_publisher(String, "/grasp_executor_lin_llm/interactive_status", 10)
        self.done = self.create_publisher(String, "/grasp_executor_lin_llm/cycle_done", 10)
        self.prog = self.create_publisher(Bool, "/io_and_status_controller/robot_program_running", LATCHED)
        self.prog.publish(Bool(data=True))
        self.create_subscription(String, "/grasp_executor_lin_llm/resume", self._on_resume, 10)
        ActionServer(self, MoveToNamedPose, "/move_to_named_pose", self._named)
        ActionServer(self, OperateGripper, "/gripper", self._gripper)
        self.seq = 0
        self.boot = time.time()
        self.cycle = 0
        self.phase("idle")
        import cv2
        self.jpg = cv2.imencode(".jpg", np.full((120, 160, 3), 90, np.uint8))[1].tobytes()
        self.k = 0
        # timing faults for the alignment checks (scenario 5)
        self.ext_delay = 0.0        # exterior frames stamped this much before they are sent
        self.wrist_frozen = None    # a stamp the wrist camera keeps repeating
        self.grip_stream = True     # the gripper adapter's joint messages
        self.create_timer(0.002, self._tick)
        t = TransformStamped(); t.header.frame_id = "base_link"; t.child_frame_id = "base"; t.transform.rotation.w = 1.0
        self.tfs.publish(TFMessage(transforms=[t]))

    # executor task state
    def phase(self, name, question="", holding=False):
        self.seq += 1
        self.task.publish(String(data=json.dumps({
            "v": 1, "boot": self.boot, "seq": self.seq, "stamp": time.time(), "cycle_id": self.cycle, "phase": name,
            "question": question, "question_id": self.seq if question else 0, "question_since": time.time(),
            "holding": holding, "object": "small_bottle" if holding else "", "destination": "", "last": {}})))

    def move_to(self, target, dur=2.0):
        with self.lock:
            q0 = self.q.copy()
        t0 = time.time()
        while time.time() - t0 < dur:
            a = (time.time() - t0) / dur
            with self.lock:
                self.q = q0 + (np.asarray(target) - q0) * (0.5 - 0.5 * math.cos(math.pi * a))
            time.sleep(0.01)
        with self.lock:
            self.q = np.asarray(target, float)

    def set_grip(self, cmd):
        self.gcmd.publish(String(data=json.dumps({"cmd": cmd, "stamp": time.time(), "backend": "fake"})))
        time.sleep(0.5)
        self.grip = 1.0 if cmd == "open" else 0.0

    def _named(self, gh):
        self.move_to(POSES.get(gh.request.pose_name, [0.0] * 6))
        gh.succeed()
        r = MoveToNamedPose.Result(); r.success = True
        return r

    def _gripper(self, gh):
        self.set_grip("open" if gh.request.command == OperateGripper.Goal.OPEN else "close")
        gh.succeed()
        r = OperateGripper.Result(); r.success = True; r.final_width_m = 0.05
        return r

    def pause_release(self):
        """The executor holding the object at the drop pose, asking to open."""
        self.cycle += 1
        self.move_to(POSES["vla_drop_right"], 1.5)
        self.grip = 0.0
        self.phase("awaiting_release", "release", holding=True)
        self.status.publish(String(data="awaiting_release"))

    def _on_resume(self, msg):
        def run():
            tok = msg.data
            if tok.startswith("release"):
                self.phase("releasing", holding=True)
                self.status.publish(String(data="releasing"))
                self.set_grip("open")
                self.status.publish(String(data="released"))
                if tok == "release_home":
                    self.phase("returning_home")
                    self.move_to(POSES["home"])
                    self.set_grip("close")          # the executor's automatic close at home
                else:
                    self.phase("awaiting_home_confirm", "home")
                    return
                self.phase("idle")
                self.done.publish(String(data=f"done|{self.cycle}"))
        threading.Thread(target=run, daemon=True).start()

    def _tick(self):
        self.k += 1
        now = self.get_clock().now().to_msg()
        with self.lock:
            q = self.q.copy()
        m = JointState(); m.header.stamp = now; m.name = ARM; m.position = list(q); m.velocity = [0.0] * 6
        m.effort = [0.0] * 6
        self.js.publish(m)
        p = PoseStamped(); p.header.stamp = now; p.header.frame_id = "base"; p.pose.position.x = float(q[0])
        p.pose.orientation.w = 1.0
        self.tcp.publish(p)
        io = IOStates(); io.digital_in_states = [Digital(pin=16, state=self.grip < 0.5)]
        self.io.publish(io)
        if self.k % 5 == 0:
            c = JointTrajectoryControllerState(); c.header.stamp = now; c.joint_names = ARM
            c.reference.positions = list(q)
            self.ctrl.publish(c)
            tfm = []
            for child, z in (("gripper_tcp", 0.3), ("tool0", 0.2)):
                t = TransformStamped(); t.header.stamp = now; t.header.frame_id = "base_link"
                t.child_frame_id = child; t.transform.translation.x = float(q[0]); t.transform.translation.z = z
                t.transform.rotation.w = 1.0
                tfm.append(t)
            self.tf.publish(TFMessage(transforms=tfm))
        if self.k % 25 == 0 and self.grip_stream:
            g = JointState(); g.header.stamp = now; g.name = ["gripper_close_open_joint", "gripper_finger_right_joint"]
            g.position = [0.23 * self.grip, 0.0]
            self.js.publish(g)
        if self.k % 17 == 0:
            for cam in ("wrist",) + (("exterior",) if self.k % 51 == 0 else ()):
                img = CompressedImage(); img.header.stamp = now; img.format = "jpeg"; img.data = self.jpg
                if cam == "exterior" and self.ext_delay:
                    img.header.stamp = (self.get_clock().now() - rclpy.duration.Duration(seconds=self.ext_delay)).to_msg()
                if cam == "wrist" and self.wrist_frozen is not None:
                    img.header.stamp = self.wrist_frozen
                self.cam[cam].publish(img)
                info = CameraInfo(); info.header.stamp = now; info.width, info.height = 160, 120
                self.info[cam].publish(info)


def main(work: Path) -> int:
    work.mkdir(parents=True, exist_ok=True)
    trial = work / "experiments_30_09_2026" / "trial_900_LA_S7"
    trial.mkdir(parents=True, exist_ok=True)
    rclpy.init()
    world = FakeWorld()
    from intent_router.intent_router_node import IntentRouterNode
    router = IntentRouterNode()
    router._check_robot_ready = lambda: None                   # no driver on this domain
    router._set_executor_param = lambda *a, **k: None
    router._speak_feedback = False             # nothing is spoken; refusal reasons are still recorded
    tester = rclpy.create_node("lifecycle_tester")
    intent_pub = tester.create_publisher(String, "/robot/intent", 10)
    cmd_pub = tester.create_publisher(String, "/vla_recorder/cmd", 10)
    status = {}
    tester.create_subscription(String, "/vla_recorder/status", lambda m: status.update(json.loads(m.data)), LATCHED)
    events = []
    tester.create_subscription(String, "/robot/command_event", lambda m: events.append(json.loads(m.data)),
                               QoSProfile(depth=100, reliability=ReliabilityPolicy.RELIABLE))
    ex = MultiThreadedExecutor(num_threads=6)
    for n in (world, router, tester):
        ex.add_node(n)
    threading.Thread(target=ex.spin, daemon=True).start()

    rec = subprocess.Popen([sys.executable, "-m", "vla_experiments.episode_recorder", "--ros-args",
                            "-p", f"dataset_root:={work / 'datasets'}", "-p", f"mmhri_dir:={work}"],
                           stdout=open(work / "recorder.log", "w"), stderr=subprocess.STDOUT)
    try:
        def wait(pred, timeout, what):
            t0 = time.time()
            while time.time() - t0 < timeout:
                if pred():
                    return True
                time.sleep(0.1)
            check(False, f"timeout waiting for {what}")
            return False

        def intent(action, **kw):
            d = {"action_intent": action, "object_intent": kw.pop("obj", None),
                 "object_attributes": kw.pop("attrs", None), "stamp": time.time(),
                 "command_id": uuid.uuid4().hex, "utterance_id": kw.pop("utt", "u-" + action),
                 "utterance": kw.pop("text", action)}
            d.update(kw)
            intent_pub.publish(String(data=json.dumps(d)))
            return d["command_id"]

        wait(lambda: status.get("state") == "idle" and not status.get("stale"), 40, "recorder ready with fresh streams")
        run_id = uuid.uuid4().hex
        for c in (f"run {run_id}", f"hint {trial}", "trial pointing test"):
            cmd_pub.publish(String(data=c)); time.sleep(0.2)
        wait(lambda: status.get("run_id") == run_id and status.get("recording"), 10, "the run acknowledgment")
        check(status.get("run_id") == run_id, "the recorder acknowledged THIS run id")

        print("1. look_right and go_home 8 ms apart", flush=True)
        a = intent("look_right"); time.sleep(0.008); b = intent("go_home")
        time.sleep(0.3)
        d = intent("go_home")                                       # duplicate while queued
        wait(lambda: any(e["command_id"] == b and e["state"] == "completed" for e in events), 15, "go_home done")

        print("2. queued look_left cancelled by stop", flush=True)
        intent("look_right"); time.sleep(0.05); ll = intent("look_left"); time.sleep(0.3); intent("stop")
        wait(lambda: any(e["command_id"] == ll and e["state"] == "cancelled" for e in events), 5, "look_left cancelled")
        time.sleep(2.5)
        world.move_to(POSES["home"], 1.0)

        print("3. release pause answered with open + go back home", flush=True)
        world.pause_release()
        time.sleep(1.0)
        c = intent("open_gripper", attrs={"then": "release_home"}, text="Open the gripper and go back home.")
        wait(lambda: status.get("recording") and world.grip > 0.5 and world.q.max() < 0.01, 15, "released and home")
        time.sleep(1.5)

        print("4. bare stop, repeated stop, the same trial again", flush=True)
        cmd_pub.publish(String(data="stop")); time.sleep(0.2); cmd_pub.publish(String(data="stop"))
        wait(lambda: (status.get("last_run") or {}).get("run_id") == run_id
             and (status.get("last_run") or {}).get("state") in ("complete", "incomplete"), 30, "the save")
        man = json.loads((trial / "recording_manifest.json").read_text())
        meta = json.loads((trial / "meta.json").read_text())
        z = np.load(trial / "steps.npz")
        st, idx, seg = z["t"], z["instruction_idx"].astype(int), z["segment"].astype(int)

        states = {}
        for e in events:
            states.setdefault(e["command_id"], []).append(e["state"])
        check(states.get(a, [])[-1:] == ["completed"] and "accepted" in states.get(a, []), "look_right accepted and completed")
        check("queued" in states.get(b, []) and states[b].index("queued") < states[b].index("accepted"),
              "go_home was queued before it was accepted")
        check(states.get(d, [])[-1:] == ["rejected"], "the duplicate go_home was rejected")
        check(states.get(ll, [])[-1:] == ["cancelled"], "the queued look_left was cancelled by stop")
        check("accepted" in states.get(c, []), "open + go home accepted as the answer to the release pause")

        check(man["state"] == "complete" and man["run_id"] == run_id, f"manifest complete for this run ({man['state']})")
        check(meta["format"] == "morty_vla_raw_v3" and meta["label_source"] == "command_lifecycle",
              "raw v3, labels from the command lifecycle")
        check(meta["outcome"] == "unknown", f"a bare stop is not a success (outcome={meta['outcome']})")
        texts = [i["instruction"] for i in meta["instructions"]]
        print("     instructions:", texts, flush=True)
        for want in ("look to the right", "go back home", "open the gripper and go back home"):
            check(want in texts, f"instruction '{want}' present")
        check("look to the left" not in texts, "the cancelled look_left labels nothing")

        def steps_of(text):
            k = next(i["idx"] for i in meta["instructions"] if i["instruction"] == text)
            return st[(idx == k) & (seg == 1)]
        look, home = steps_of("look to the right"), steps_of("go back home")
        check(look.size >= 10 and home.size >= 10 and look.max() < home.min(),
              f"look_right ({look.size} steps) and go_home ({home.size} steps) each got their own motion, in order")
        opn = next(i for i in meta["instructions"] if i["instruction"] == "open the gripper and go back home")
        check(opn.get("auto_cleanup_from") is not None, "the automatic close after the hand-over is cleanup, not the command")
        hold = st[seg == 3]
        check(hold.size > 0, f"the release pause before the answer is a hold ({hold.size} steps)")
        q = meta["quality"]
        check(q["integrity"]["ok"], f"integrity ok ({q['integrity']['issues']})")
        print("     alignment issues:", q["alignment"]["issues"], flush=True)
        print("     semantic flags:", q["semantic"]["flags"], flush=True)
        check(set(("command", "task_state")) <= set(meta["events"]), "command and task_state events recorded")

        run2 = uuid.uuid4().hex
        for cmd in (f"run {run2}", f"hint {trial}", "trial again"):
            cmd_pub.publish(String(data=cmd)); time.sleep(0.2)
        wait(lambda: status.get("run_id") == run2 and status.get("recording"), 10, "second run on the same folder")
        prev = list((trial / "vla_previous").glob("*/meta.json")) if (trial / "vla_previous").exists() else []
        check(len(prev) == 1 and json.loads(prev[0].read_text())["run_id"] == run_id,
              "the first episode was preserved in vla_previous/, not deleted")
        print("5. timing faults: old exterior capture stamps, a frozen wrist frame, no gripper stream", flush=True)
        world.ext_delay = 1.2
        world.wrist_frozen = world.get_clock().now().to_msg()
        world.grip_stream = False
        world.move_to(POSES["detect_left"], 2.0)          # confirms the provisional episode by motion
        time.sleep(3.0)
        cmd_pub.publish(String(data="stop ok"))
        wait(lambda: (status.get("last_run") or {}).get("run_id") == run2
             and (status.get("last_run") or {}).get("state") in ("complete", "incomplete"), 30, "the second save")
        meta2 = json.loads((trial / "meta.json").read_text())
        # every check's wording is in "details" (2026-09-30); "issues" holds
        # what blocks the episode (here the frozen stream), the late steps are
        # counted and left out of training
        align = meta2["quality"]["alignment"]
        al = " | ".join(align.get("details", align["issues"]))
        print("     alignment details:", align.get("details"), "| blocking:", align["issues"], flush=True)
        check(align.get("misaligned_steps", 0) >= 40,
              f"the late steps are counted ({align.get('misaligned_steps')} of {meta2['num_steps']})")
        check("exterior camera clock" in al or "exterior image CAPTURED" in al,
              "old exterior capture stamps detected (capture age, or the clock not comparable)")
        check("wrist: the same frame reused" in al, "the frozen wrist frame detected")
        check("gripper state up to" in al or "gripper state never received" in al, "the stale gripper stream detected")
        check(not meta2["quality"]["ok"] and not meta2["training_eligible"], "such an episode is not training-eligible")
        check(meta2["outcome"] == "success", "an explicit 'stop ok' is the human's success")
        world.ext_delay, world.wrist_frozen, world.grip_stream = 0.0, None, True
    finally:
        rec.terminate()
        try:
            rec.wait(10)
        except subprocess.TimeoutExpired:
            rec.kill()
        for n in (tester, router, world):
            try:
                n.destroy_node()
            except Exception:
                pass
        ex.shutdown()
        rclpy.try_shutdown()
    print(f"{'ALL PASSED' if not FAILED else str(len(FAILED)) + ' FAILED'}", flush=True)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    if os.environ.get("ROS_DOMAIN_ID") in (None, "", "0"):
        sys.exit("set ROS_DOMAIN_ID to an unused domain (e.g. 77): this must not talk to the live robot stack")
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/morty_lifecycle_check")))
