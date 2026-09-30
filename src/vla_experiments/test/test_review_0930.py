"""The review of the 30 Sep recordings (deterministic fixtures, no ROS):
what the evidence settles is a note, not a review flag; timing problems of
single steps leave those steps out instead of the episode; a bag whose
rosbag2 hung only after "Recording stopped" is complete; post-processing
waits for the recorder's exit; instruction wording; Morty's action claims."""
import json
import sqlite3
from pathlib import Path

import numpy as np

from vla_experiments.labels import label_episode
from vla_experiments.recording_contract import COMMAND_EVENT_SCHEMA, SEGMENT_IDLE, SEGMENT_RESET, SEGMENT_TASK

T0 = 1000.0


def _steps(duration=20.0, moving=()):
    t = np.arange(0.0, duration, 0.1)
    q = np.zeros((len(t), 6))
    for i in range(1, len(t)):
        q[i] = q[i - 1] + (0.03 if any(a <= t[i] < b for a, b in moving) else 0.0)
    return {"t": t, "stamp_ros": T0 + t, "joint_pos": q}


def _cmd(events, cid, state, t, **kw):
    e = {"schema": COMMAND_EVENT_SCHEMA, "event_id": f"{cid}{state}{t}", "seq": len(events), "source": "test",
         "command_id": cid, "state": state, "stamp": T0 + t, "action": kw.pop("action", ""),
         "attributes": kw.pop("attributes", {}), "route": kw.pop("route", ""), "reason": kw.pop("reason", "")}
    e.update(kw)
    events.append(e)


def _ts(*items):
    return [{"stamp": T0 + t, "seq": k, "boot": 1, "phase": ph, "cycle_id": 5, "question": q, "holding": h}
            for k, (t, ph, q, h) in enumerate(items)]


def _handover(with_open=True, move_after_close=False):
    """Trial 143's pattern: pick, carried home, "open the gripper" at the
    release pause; the executor opens, waits for the hand, closes, ends."""
    moving = [(1.0, 5.0), (7.0, 10.0)] + ([(13.0, 14.5)] if move_after_close else [])
    st = _steps(20.0, moving)
    c = []
    _cmd(c, "P", "received", 0.4, action="pick")
    _cmd(c, "P", "accepted", 0.5, action="pick", route="pick")
    _cmd(c, "P", "cycle", 0.9, action="pick", route="pick", cycle_id=5)
    _cmd(c, "H", "accepted", 6.9, action="go_home", route="executor_answer", parent_id="P", cycle_id=5)
    _cmd(c, "O", "accepted", 11.9, action="open_gripper", route="executor_answer", parent_id="P", cycle_id=5)
    _cmd(c, "P", "completed", 15.0, action="pick", route="pick", cycle_id=5)
    ts = _ts((1.0, "approaching", "", False), (5.5, "awaiting_destination", "destination", True),
             (7.0, "carrying", "", True), (10.5, "awaiting_release", "release", True),
             (12.0, "releasing", "", True), (15.0, "idle", "", False))
    grip = ([{"stamp": T0 + 12.2, "cmd": "open"}] if with_open else []) + [{"stamp": T0 + 13.0, "cmd": "close"}]
    return st, {"command": c, "task_state": ts, "gripper_cmd": grip}


def test_the_release_routine_close_is_a_note_not_a_review_flag():
    st, ev = _handover()
    lab = label_episode(ev, st)
    assert not lab["flags"], lab["flags"]
    assert any("release routine" in n for n in lab["notes"])
    k = next(i["idx"] for i in lab["instructions"] if i["instruction"] == "open the gripper")
    task = st["t"][(lab["instruction_idx"] == k) & (lab["segment"] == SEGMENT_TASK)]
    reset = st["t"][lab["segment"] == SEGMENT_RESET]
    assert task.max() < 13.0 and reset.min() >= 13.0 - 1e-9   # the close is cleanup, not the command


def test_a_close_without_an_open_or_with_motion_after_it_is_flagged():
    st, ev = _handover(with_open=False)
    assert any("review" in f for f in label_episode(ev, st)["flags"])
    st, ev = _handover(with_open=False)                 # the open was the command BEFORE (trial 145)
    ev["gripper_cmd"].insert(0, {"stamp": T0 + 11.0, "cmd": "open"})
    assert not label_episode(ev, st)["flags"]
    st, ev = _handover(move_after_close=True)
    assert any("review" in f for f in label_episode(ev, st)["flags"])


def test_a_pick_that_failed_before_moving_is_a_note():
    """Trial 144's first attempt: no reachable grasp, the arm never moved."""
    st = _steps(10.0)
    c = []
    _cmd(c, "P", "accepted", 0.5, action="pick", route="pick")
    _cmd(c, "P", "cycle", 0.9, action="pick", route="pick", cycle_id=5)
    _cmd(c, "P", "failed", 3.0, action="pick", route="pick", reason="tried 10 grasps: 6 had no IK")
    ev = {"command": c, "task_state": _ts((1.0, "probing", "", False), (3.0, "idle", "", False))}
    lab = label_episode(ev, st)
    assert not lab["flags"] and not lab["instructions"]
    assert any("failed before moving" in n for n in lab["notes"])
    # without the executor's verdict it stays a review flag (trial 142: the stuck pick)
    c2 = [e for e in c if e["state"] != "failed"]
    lab2 = label_episode({"command": c2, "task_state": ev["task_state"][:1]}, st)
    assert any("never started moving" in f for f in lab2["flags"])


def test_a_drop_pose_reached_with_an_empty_gripper_is_not_a_drop():
    """Trial 147 (S7): "Drop right position." with nothing held."""
    st = _steps(10.0, [(1.0, 4.0)])
    c = []
    _cmd(c, "D", "accepted", 0.9, action="drop_right", route="named_pose")
    _cmd(c, "D", "completed", 4.5, action="drop_right", route="named_pose")
    lab = label_episode({"command": c}, st)
    assert [i["instruction"] for i in lab["instructions"]] == ["go to the right side"]


def test_the_detector_class_vitamin_c_is_a_vitamin_tube():
    from vla_experiments.raw_dataset import canonical_instruction
    assert canonical_instruction("pick", "vitamin_c", {"colour": "blue"}) == "pick up the blue vitamin tube"
    assert canonical_instruction("pick", "vitamin c", {"position": "leftmost"}) == "pick up the leftmost vitamin tube"
    assert canonical_instruction("drop_right", None, {}) == "drop it on the right side"


# ------------------------------------------------------------ alignment ----
class _Ep:
    """The attributes training_segments reads."""

    def __init__(self, steps, seg, idx, alignment):
        self.steps, self.n = steps, len(seg)
        self._seg, self._idx = np.asarray(seg), np.asarray(idx)
        self.meta = {"quality": {"alignment": alignment}}
        self.instructions = [{"idx": 0, "instruction": "a"}]
        self.task = "a"

    def segment_codes(self):
        return self._seg

    def instruction_idx(self):
        return self._idx


def test_single_late_frames_leave_their_steps_out_not_the_episode():
    from vla_experiments.raw_dataset import alignment_verdict, step_misaligned, training_segments
    n = 40
    seg = np.array([SEGMENT_IDLE] * 10 + [SEGMENT_TASK] * 30)
    idx = np.array([-1] * 10 + [0] * 30)
    steps = {"t": np.arange(n) / 10.0, "img_src_age_wrist": np.full(n, 0.1), "camera_skew": np.full(n, 0.05),
             "gripper_age": np.full(n, 0.2), "segment": seg.astype(float)}
    steps["img_src_age_wrist"][3] = 0.64          # trial 150: outside every task
    steps["img_src_age_wrist"][20] = np.nan       # a missing value is no violation
    steps["camera_skew"][25] = 0.47               # one bad task step
    al = {"issues": ["wrist image CAPTURED up to 0.64 s before its step",
                     "cameras up to 0.47 s apart (> 0.35 s) in 1 step(s)", "2 step gap(s) > 0.5 s (max 0.7 s)"],
          "images": {"wrist": {"clock_comparable": True}}}
    assert list(np.nonzero(step_misaligned(steps, al))[0]) == [3, 25]
    v = alignment_verdict(steps, seg, al)
    assert v["issues"] == ["2 step gap(s) > 0.5 s (max 0.7 s)"]        # the episode-wide problem stays
    assert v["misaligned_steps"] == 2 and v["misaligned_task_steps"] == 1 and v["notes"]
    al_ok = dict(al, issues=al["issues"][:2])
    assert alignment_verdict(steps, seg, al_ok)["ok"]
    sel = dict(training_segments(_Ep(steps, seg, idx, al_ok)))["a"]
    assert 25 not in sel and len(sel) == 29
    sel_all = dict(training_segments(_Ep(steps, seg, idx, al_ok), drop_misaligned=False))["a"]
    assert len(sel_all) == 30
    steps["camera_skew"][11:20] = 0.9              # 10 of 30 task steps: too many to leave out
    assert not alignment_verdict(steps, seg, al_ok)["ok"]


# ------------------------------------------------------------------ bag ----
def _bag(d: Path, n=5):
    d.mkdir(parents=True)
    con = sqlite3.connect(d / "bag_0.db3")
    con.execute("create table topics(id integer primary key, name text, type text, serialization_format text,"
                " offered_qos_profiles text)")
    con.execute("create table messages(id integer primary key, topic_id integer, timestamp integer, data blob)")
    con.execute("insert into topics values (1, '/joint_states', 'sensor_msgs/msg/JointState', 'cdr', '')")
    for i in range(n):
        con.execute("insert into messages(topic_id, timestamp, data) values (1, ?, x'00')", (int((2000 + i) * 1e9),))
    con.commit()
    con.close()
    (d / "metadata.yaml").write_text(
        "rosbag2_bagfile_information:\n  relative_file_paths:\n    - bag_0.db3\n"
        f"  message_count: {n}\n  topics_with_message_count:\n"
        f"    - topic_metadata:\n        name: /joint_states\n      message_count: {n}\n")


def test_a_bag_killed_after_recording_stopped_is_complete(tmp_path):
    from vla_experiments.bag_check import verify_bag
    _bag(tmp_path / "bag")
    (tmp_path / "bag_status.json").write_text(json.dumps(
        {"state": "exited", "exit_code": -9, "stop_stage": 3, "stop_requested_unix": 2004.2}))
    (tmp_path / "bag_record.log").write_text("[rosbag2_recorder]: Recording stopped\n")
    r = verify_bag(tmp_path / "bag")
    assert r["ok"] and r["warnings"] and "data is complete" in r["warnings"][0]
    (tmp_path / "bag_record.log").write_text("[rosbag2_recorder]: Recording...\n")    # never finished
    assert not verify_bag(tmp_path / "bag")["ok"]
    (tmp_path / "bag_record.log").write_text("[rosbag2_recorder]: Recording stopped\n")
    (tmp_path / "bag_status.json").write_text(json.dumps(                          # messages missing at the end
        {"state": "exited", "exit_code": -9, "stop_stage": 3, "stop_requested_unix": 2010.0}))
    assert not verify_bag(tmp_path / "bag")["ok"]


def test_postprocess_waits_for_the_recorder_exit_not_the_status_file(tmp_path):
    from vla_experiments.postprocess import wait_ready
    _bag(tmp_path / "bag")
    (tmp_path / "bag_status.json").write_text(json.dumps({"state": "stopping"}))
    r = wait_ready(tmp_path, timeout_s=1.5)
    assert r["timeout"] and not r["bag_exit_recorded"]
    (tmp_path / "bag_status.json").write_text(json.dumps({"state": "exited", "exit_code": 0, "stop_stage": 1}))
    assert not wait_ready(tmp_path, timeout_s=1.5)["timeout"]


# ------------------------------------------------------------- the voice ----
def test_morty_never_speaks_an_action_claim_worded_like_trial_147():
    import pytest
    cn = pytest.importorskip("communication_llm.chat_node")
    for s in ("Going to look left, then home.", "Going back home.", "Okay, moving to the drop right position.",
              "Dropping right and heading home."):
        assert cn._is_action_claim(s), s
    for s in ("I'm going to need the name of the object.", "I can see a small bottle.", "Which vitamin do you mean?"):
        assert not cn._is_action_claim(s), s


def test_reconstruction_pairs_each_answer_with_the_intent_the_router_names(tmp_path):
    """Trial 137: close_gripper and drop_right 72 ms apart at the release
    pause; the reconstruction gave the drop's intent to the close."""
    from vla_experiments.validate import reconstruct_commands
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "voice_gui.log").write_text(
        "[intent_router_node-6] [INFO] [1790781734.211] [intent_router_node]: Resume 'stop' published "
        "(bypass queue (release paused, keep holding): close_gripper).\n"
        "[intent_router_node-6] [INFO] [1790781734.283] [intent_router_node]: Resume 'drop_right' published "
        "(release pause: 'drop_right' names a side -> redirect, ask again).\n")
    meta = {"started_ros": 1790781720.0, "duration_s": 30.0,
            "events": {"intent": [{"stamp": 1790781734.210, "action_intent": "close_gripper"},
                                  {"stamp": 1790781734.282, "action_intent": "drop_right"}]}}
    evs = reconstruct_commands(tmp_path, meta)
    acc = [(e["stamp"], e["action"]) for e in evs if e["state"] == "accepted"]
    assert [a for _, a in acc] == ["close_gripper", "drop_right"]
