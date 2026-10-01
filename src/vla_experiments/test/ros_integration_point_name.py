"""The pointing question names the exact tube — on an ISOLATED domain (2026-10-01).

    ROS_DOMAIN_ID=77 python3 test/ros_integration_point_name.py

Majd's trial 197: a small bottle and three vitamin tubes (B12 left, Multi
middle, C right); he pointed at one and Morty said "You pointed at the vitamin
tube. Should I grasp it?". The real IntentRouterNode (speech and the executor
stubbed) gets the pipeline's proposal WITH the cap colour and must name the
tube in the question, in /robot/open_question (chat_node's LLM context) and,
after the yes, in that pick's start and destination lines — and only there:
the next pick is "the vitamin tube" again.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Int64, String

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
FAILED = []


def check(ok, what):
    print(("  PASS " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILED.append(what)


def main() -> int:
    rclpy.init()
    from intent_router.intent_router_node import IntentRouterNode
    r = IntentRouterNode()
    said = []
    _orig = r._narrate
    r._speak_feedback = False

    def _narr(text, interrupt=False):
        said.append(text)
        _orig(text, interrupt=interrupt)
    r._narrate = _narr
    r._check_robot_ready = lambda: None
    r._set_executor_param = lambda name, value, kind: None
    r._set_target_label = lambda label: None
    r._set_task_object_label = lambda label: None
    t = rclpy.create_node("router_point_name_tester")
    questions, answers = [], []
    t.create_subscription(String, "/robot/open_question", lambda m: questions.append(json.loads(m.data)), LATCHED)
    t.create_subscription(String, "/grasp_pipeline/point_answer", lambda m: answers.append(m.data), 10)
    prop = t.create_publisher(String, "/grasp_pipeline/point_proposal", 10)
    intent_pub = t.create_publisher(String, "/robot/intent", 10)
    task = t.create_publisher(String, "/grasp_executor_lin_llm/task_state", LATCHED)
    cyc = t.create_publisher(Int64, str(r.get_parameter("cycle_id_topic").value), 10)
    ex = MultiThreadedExecutor()
    ex.add_node(r); ex.add_node(t)
    threading.Thread(target=ex.spin, daemon=True).start()
    time.sleep(1.5)
    seq = [0]

    def state(**kw):
        seq[0] += 1
        d = {"v": 1, "boot": 1.0, "seq": seq[0], "cycle_id": 0, "phase": "idle", "question": "",
             "question_id": 0, "holding": False, "object": ""}
        d.update(kw)
        task.publish(String(data=json.dumps(d)))
    state()
    scene = [{"label": "vitamin_tube", "bbox": [383, 205, 457, 512], "cap_colour": "blue"},
             {"label": "small_bottle", "bbox": [0, 345, 67, 530], "cap_colour": ""},
             {"label": "vitamin_tube", "bbox": [501, 200, 570, 456], "cap_colour": "orange"},
             {"label": "vitamin_tube", "bbox": [212, 205, 277, 462], "cap_colour": "purple"}]
    r._on_detections(String(data=json.dumps(scene)))
    time.sleep(0.5)

    print("1. a pointed Vitamin C tube (proposal with its cap colour)", flush=True)
    prop.publish(String(data=json.dumps({"id": 1, "state": "open", "label": "vitamin_tube",
                                         "bbox": [501, 200, 570, 456], "cap_colour": "orange"})))
    time.sleep(0.6)
    check(said[-1:] == ["You pointed at the Vitamin C tube. Should I grasp it?"], f"asked: {said[-1:]!r}")
    q = questions[-1] if questions else {}
    check(q.get("question") == "point" and q.get("name") == "Vitamin C tube" and q.get("object") == "vitamin_tube",
          f"/robot/open_question carries the name ({q})")
    prop.publish(String(data=json.dumps({"id": 1, "state": "timeout"})))
    time.sleep(0.4)

    print("2. a proposal WITHOUT cap colour (older pipeline): named from the live scene", flush=True)
    prop.publish(String(data=json.dumps({"id": 2, "state": "open", "label": "vitamin_tube",
                                         "bbox": [214, 207, 275, 460]})))
    time.sleep(0.6)
    check(said[-1:] == ["You pointed at the Vitamin B12 tube. Should I grasp it?"], f"asked: {said[-1:]!r}")

    print("3. yes -> the pick of that tube names it in its start and destination lines", flush=True)
    intent_pub.publish(String(data=json.dumps({"action_intent": "confirm_point", "object_intent": None,
                                               "object_attributes": None})))
    time.sleep(0.6)
    check(answers[-1:] == ["yes|2"], f"answered the pipeline: {answers[-1:]}")
    r._last_target_label = "vitamin_tube"
    cyc.publish(Int64(data=int(time.time() * 1000)))
    time.sleep(1.2)
    check("Starting grasp on the Vitamin B12 tube." in said, f"start line: {said[-2:]!r}")
    r._on_interactive_status(String(data="paused"))
    time.sleep(0.3)
    check(any(s.startswith("Got the Vitamin B12 tube. Where should I drop it?") for s in said),
          f"destination line: {said[-1:]!r}")

    print("4. the next (voice) pick of a tube is 'the vitamin tube' again", flush=True)
    cyc.publish(Int64(data=int(time.time() * 1000) + 5))
    time.sleep(1.2)
    check(said[-1:] == ["Starting grasp on the vitamin tube."], f"start line: {said[-1:]!r}")

    ex.shutdown()
    for n in (t, r):
        n.destroy_node()
    rclpy.try_shutdown()
    print(f"{'ALL PASSED' if not FAILED else str(len(FAILED)) + ' FAILED'}", flush=True)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    if os.environ.get("ROS_DOMAIN_ID") in (None, "", "0"):
        sys.exit("set ROS_DOMAIN_ID to an unused domain (e.g. 77)")
    sys.exit(main())
