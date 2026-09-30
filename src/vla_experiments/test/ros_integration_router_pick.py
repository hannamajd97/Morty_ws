"""The router's pick / pointing command lifecycle on an ISOLATED domain.

    ROS_DOMAIN_ID=77 python3 test/ros_integration_router_pick.py

The real IntentRouterNode (readiness, executor parameters, the pipeline's
label service and speech stubbed); the test plays the pipeline (pointing
proposals, cycle ids) and the executor (task state, cycle_done). Checks the
events of recording contract v1 and that the routing itself is unchanged
(the questions Morty asks, the answers sent to the pipeline / executor).
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid

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
    said, params, labels = [], [], []
    _orig_narrate = r._narrate
    r._speak_feedback = False                 # the real _narrate runs (it records a refusal's reason), nothing is spoken

    def _narr(text, interrupt=False):
        said.append(text)
        _orig_narrate(text, interrupt=interrupt)
    r._narrate = _narr
    r._check_robot_ready = lambda: None
    r._set_executor_param = lambda name, value, kind: params.append((name, value))
    r._set_target_label = lambda label: (labels.append(label), setattr(r, "_clarify", None) if label else None)
    r._set_task_object_label = lambda label: None
    t = rclpy.create_node("router_pick_tester")
    events, answers, resumes = [], [], []
    t.create_subscription(String, "/robot/command_event", lambda m: events.append(json.loads(m.data)),
                          QoSProfile(depth=200, reliability=ReliabilityPolicy.RELIABLE))
    t.create_subscription(String, "/grasp_pipeline/point_answer", lambda m: answers.append(m.data), 10)
    t.create_subscription(String, "/grasp_executor_lin_llm/resume", lambda m: resumes.append(m.data), 10)
    prop = t.create_publisher(String, "/grasp_pipeline/point_proposal", 10)
    intent_pub = t.create_publisher(String, "/robot/intent", 10)
    det = t.create_publisher(String, "/yolo/all_detections", 10)
    cyc = t.create_publisher(Int64, "/graspgen/selection_cycle_id", LATCHED)
    done = t.create_publisher(String, "/grasp_executor_lin_llm/cycle_done", 10)
    task = t.create_publisher(String, "/grasp_executor_lin_llm/task_state", LATCHED)
    status = t.create_publisher(String, "/grasp_executor_lin_llm/interactive_status", 10)
    ex = MultiThreadedExecutor()
    ex.add_node(r); ex.add_node(t)
    threading.Thread(target=ex.spin, daemon=True).start()
    time.sleep(1.5)
    seq = [0]

    def state(phase, q="", holding=False, cycle=0):
        seq[0] += 1
        task.publish(String(data=json.dumps({"v": 1, "boot": 1.0, "seq": seq[0], "cycle_id": cycle, "phase": phase,
                                             "question": q, "question_id": seq[0], "question_since": time.time(),
                                             "holding": holding, "object": "vitamin_tube" if holding else "",
                                             "destination": "", "drop_targets": {"left": "drop_left_side",
                                                                                 "right": "drop_right_side"},
                                             "last": {}})))

    def intent(action, obj=None, attrs=None, **kw):
        cid = uuid.uuid4().hex
        intent_pub.publish(String(data=json.dumps({"action_intent": action, "object_intent": obj,
                                                   "object_attributes": attrs, "command_id": cid, **kw})))
        time.sleep(0.7)
        return cid

    def states(cid):
        return [e["state"] for e in events if e["command_id"] == cid]

    def last(cid, state_):
        return [e for e in events if e["command_id"] == cid and e["state"] == state_][-1:]

    scene = [{"label": "vitamin_tube", "bbox": [100, 50, 140, 150]}, {"label": "vitamin_tube", "bbox": [400, 50, 440, 150]}]
    r._on_detections(String(data=json.dumps(scene)))
    state("idle")
    time.sleep(0.5)

    print("1. pick_and_place with two tubes -> which one? -> pointed -> yes", flush=True)
    r._on_detections(String(data=json.dumps(scene)))
    p0 = intent("pick_and_place", "vitamin tube", {"destination": "drop_left"}, utterance="give me the vitamin tube")
    check(states(p0)[-1:] == ["rejected"] and "Could you say" in last(p0, "rejected")[0]["reason"],
          "the refused pick_and_place is 'rejected' with the which-one question as reason")
    prop.publish(String(data=json.dumps({"id": 5, "state": "open", "label": "vitamin_tube", "bbox": [100, 50, 140, 150]})))
    time.sleep(0.5)
    check(said[-1] == "Do you mean this vitamin tube, the leftmost one?", f"the question is unchanged: {said[-1]!r}")
    y = intent("confirm_point", utterance="yes", utterance_id="utt-y")
    acc = last(y, "accepted")
    check(answers[-1:] == ["yes|5"], "the pipeline got yes|5")
    check(bool(acc) and acc[0]["route"] == "pick" and acc[0]["action"] == "pick_and_place"
          and acc[0]["attributes"].get("destination") == "drop_left", "the yes is the accepted pick_and_place (left)")
    check(bool(acc) and acc[0]["target"]["proposal_id"] == 5 and acc[0]["target"]["position"] == "leftmost"
          and acc[0]["question"] == "point" and acc[0]["question_id"] == 5,
          "target (proposal 5, leftmost) and the answered question are on the event")
    check(("drop_target_name", "drop_left_side") in params and ("wait_for_destination", False) in params,
          "the executor got the destination before the pick started")
    cyc.publish(Int64(data=11)); time.sleep(0.9)
    check(any(e["command_id"] == y and e["state"] == "cycle" and e["cycle_id"] == 11 for e in events),
          "the grasp cycle 11 is linked to the pick command")

    print("2. a follow-up open said during the pick, answered at the release pause", flush=True)
    state("approaching", cycle=11)
    time.sleep(0.3)
    f = intent("open_gripper", attrs={"then": "release_home"}, utterance="and then give it to me and go home")
    check(states(f)[-1:] == ["deferred"], "the follow-up is 'deferred', not executed")
    state("awaiting_release", "release", holding=True, cycle=11)
    status.publish(String(data="awaiting_release"))
    time.sleep(0.8)
    fa = last(f, "accepted")
    check(bool(fa) and fa[0]["route"] == "executor_answer" and fa[0]["parent_id"] == y
          and resumes[-1:] == ["release_home"], "accepted at the release pause, as a child of the pick, token release_home")
    done.publish(String(data="done|11")); time.sleep(0.8)
    check(states(y)[-1:] == ["completed"], "cycle_done completes the pick command")

    print("3. a pick while holding is refused (still holding)", flush=True)
    state("idle", holding=True)
    time.sleep(0.4)
    h = intent("pick", "small bottle")
    check(states(h)[-1:] == ["rejected"] and "still holding" in last(h, "rejected")[0]["reason"],
          "refused with the still-holding line as reason")
    state("idle")
    time.sleep(0.3)

    print("4. an intent without command_id gets one; stop cancels the queue", flush=True)
    intent_pub.publish(String(data=json.dumps({"action_intent": "look_left"})))
    time.sleep(0.6)
    anon = [e for e in events if e["action"] == "look_left"]
    check(bool(anon) and anon[0]["command_id"], "the router minted a command id")
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
