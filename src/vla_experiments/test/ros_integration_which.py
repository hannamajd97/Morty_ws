"""The router's which-one question on an ISOLATED domain (2026-10-01).

    ROS_DOMAIN_ID=77 python3 test/ros_integration_which.py

Majd's live case: a small bottle and three vitamin tubes (B12 left, Multi
middle, C right in the camera picture); "Give me the vitamin tube." -> the
router asks which one and now PUBLISHES that question on /robot/open_question;
the instant answer chat_node builds from "The middle one."
(voice_grammar.match_which_answer -> pick, position=middle) must select the
MIDDLE tube (the Multivitamin, not the B12 the model guessed) and close the
question. The real IntentRouterNode, with speech and the executor stubbed.
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
from std_msgs.msg import String

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
FAILED = []


def check(ok, what):
    print(("  PASS " if ok else "  FAIL ") + what, flush=True)
    if not ok:
        FAILED.append(what)


def main() -> int:
    rclpy.init()
    from communication_llm import voice_grammar
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
    r._set_target_label = lambda label: setattr(r, "_clarify", None) if label else None
    r._set_task_object_label = lambda label: None
    t = rclpy.create_node("router_which_tester")
    questions, selections = [], []
    t.create_subscription(String, "/robot/open_question", lambda m: questions.append(json.loads(m.data)), LATCHED)
    t.create_subscription(String, "/yolo/select_detection", lambda m: selections.append(json.loads(m.data)), 10)
    intent_pub = t.create_publisher(String, "/robot/intent", 10)
    task = t.create_publisher(String, "/grasp_executor_lin_llm/task_state", LATCHED)
    ex = MultiThreadedExecutor()
    ex.add_node(r); ex.add_node(t)
    threading.Thread(target=ex.spin, daemon=True).start()
    time.sleep(1.5)
    task.publish(String(data=json.dumps({"v": 1, "boot": 1.0, "seq": 1, "cycle_id": 0, "phase": "idle",
                                         "question": "", "question_id": 0, "holding": False, "object": ""})))
    scene = [{"label": "vitamin_tube", "bbox": [383, 205, 457, 512], "cap_colour": "blue"},
             {"label": "small_bottle", "bbox": [0, 345, 67, 530], "cap_colour": ""},
             {"label": "vitamin_tube", "bbox": [501, 200, 570, 456], "cap_colour": "orange"},
             {"label": "vitamin_tube", "bbox": [212, 205, 277, 462], "cap_colour": "purple"}]
    r._on_detections(String(data=json.dumps(scene)))
    time.sleep(0.5)

    print("1. a plain pick with three tubes -> the which-one question is asked AND published", flush=True)
    intent_pub.publish(String(data=json.dumps({"action_intent": "pick", "object_intent": "vitamin_tube",
                                               "object_attributes": None})))
    time.sleep(0.8)
    check(any("three vitamin tubes" in s for s in said), f"asked: {said[-1:]!r}")
    q = questions[-1] if questions else {}
    check(q.get("question") == "which" and q.get("object") == "vitamin_tube",
          f"/robot/open_question says which vitamin_tube ({q})")

    print("2. 'The middle one.' -> chat_node's instant answer -> the MIDDLE tube (Multivitamin)", flush=True)
    ans = voice_grammar.match_which_answer("The middle one.")
    check(ans == {"position": "middle"}, f"grammar: {ans}")
    r._on_detections(String(data=json.dumps(scene)))
    intent_pub.publish(String(data=json.dumps({"action_intent": "pick", "object_intent": q.get("object"),
                                               "object_attributes": ans})))
    time.sleep(1.0)
    sel = selections[-1] if selections else {}
    bbox = sel.get("bbox") or sel.get("det", {}).get("bbox")
    check(bbox == [383, 205, 457, 512], f"selected the middle tube's box ({sel})")
    check(any("Multivitamin" in s for s in said[-3:]), f"narrated the Multivitamin: {said[-3:]!r}")
    check(questions[-1].get("question") == "", f"the question is closed again ({questions[-1]})")

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
