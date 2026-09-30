"""labels.label_episode on deterministic fixtures (no ROS) — the lifecycle
rules of recording_contract.py. Each fixture is a timeline the router, the
executor and the gripper adapter could publish; stamps are seconds."""
import random

import numpy as np
import pytest

from vla_experiments.labels import label_episode
from vla_experiments.recording_contract import (COMMAND_EVENT_SCHEMA, SEGMENT_ASSIST, SEGMENT_CLARIFY, SEGMENT_HOLD,
                                                SEGMENT_IDLE, SEGMENT_RESET, SEGMENT_TASK)

T0 = 1000.0
FPS = 10.0


def steps(duration=30.0, moving=()):
    """10 Hz steps; the arm moves (0.3 rad/s on every joint) inside the
    given (t_on, t_off) windows, relative to T0."""
    t = np.arange(0.0, duration, 1.0 / FPS)
    q = np.zeros((len(t), 6))
    for i in range(1, len(t)):
        dq = 0.03 if any(a <= t[i] < b for a, b in moving) else 0.0
        q[i] = q[i - 1] + dq
    return {"t": t, "stamp_ros": T0 + t, "joint_pos": q}


class Cmds:
    """Builds command lifecycle events like the router publishes them."""

    def __init__(self):
        self.events, self.seq = [], 0

    def ev(self, cid, state, t, **kw):
        self.seq += 1
        e = {"schema": COMMAND_EVENT_SCHEMA, "event_id": f"e{self.seq}", "seq": self.seq, "source": "test",
             "command_id": cid, "state": state, "stamp": T0 + t, "action": kw.pop("action", ""),
             "attributes": kw.pop("attributes", {}), "route": kw.pop("route", ""), "reason": kw.pop("reason", "")}
        e.update(kw)
        self.events.append(e)
        return self


def ts(*items):
    """task_state samples: (t, phase, cycle_id[, question])."""
    out = []
    for k, it in enumerate(items):
        t, phase, cyc = it[:3]
        out.append({"stamp": T0 + t, "seq": k, "boot": 1, "phase": phase, "cycle_id": cyc,
                    "question": it[3] if len(it) > 3 else ""})
    return out


def idx_of(lab, text):
    return next(i["idx"] for i in lab["instructions"] if i["instruction"] == text)


def steps_of(lab, st, text, seg=SEGMENT_TASK):
    k = idx_of(lab, text)
    sel = np.nonzero((lab["instruction_idx"] == k) & (lab["segment"] == seg))[0]
    return st["t"][sel]


def test_two_commands_ms_apart_each_get_their_own_motion():
    """Trial 114's pattern: look_right and go_home arrive 8 ms apart; the
    router queues go_home until look_right completes."""
    st = steps(30.0, moving=[(1.0, 4.0), (5.0, 8.0)])
    c = Cmds()
    c.ev("A", "received", 0.97, action="look_right").ev("A", "queued", 0.971, action="look_right")
    c.ev("B", "received", 0.978, action="go_home").ev("B", "queued", 0.979, action="go_home")
    c.ev("A", "accepted", 0.98, action="look_right", route="named_pose").ev("A", "completed", 4.5, action="look_right")
    c.ev("B", "accepted", 4.6, action="go_home", route="named_pose").ev("B", "completed", 8.5, action="go_home")
    lab = label_episode({"command": c.events}, st)
    assert lab["label_source"] == "command_lifecycle"
    look = steps_of(lab, st, "look to the right")
    home = steps_of(lab, st, "go back home")
    assert look.size > 20 and look.min() >= 0.8 and look.max() <= 4.1
    assert home.size > 20 and home.min() >= 4.8
    assert not lab["flags"]


def test_queued_rejected_duplicate_deferred_cancelled_never_take_a_label():
    st = steps(20.0, moving=[(2.0, 5.0)])
    c = Cmds()
    c.ev("A", "received", 1.0, action="go_home").ev("A", "accepted", 1.1, action="go_home", route="named_pose")
    c.ev("A", "completed", 5.5, action="go_home")
    c.ev("D", "received", 1.2, action="go_home").ev("D", "rejected", 1.2, reason="duplicate")
    c.ev("Q", "received", 1.3, action="look_left").ev("Q", "queued", 1.3).ev("Q", "cancelled", 3.0, reason="stop")
    c.ev("F", "received", 2.0, action="open_gripper").ev("F", "deferred", 2.0, reason="kept for the release pause")
    c.ev("R", "received", 2.5, action="pick").ev("R", "rejected", 2.5, reason="I don't see the robot driver")
    lab = label_episode({"command": c.events}, st)
    assert [i["instruction"] for i in lab["instructions"]] == ["go back home"]
    states = {x["command_id"]: x["states"] for x in lab["commands"]}
    assert states["Q"][-1] == "cancelled" and states["D"][-1] == "rejected" and states["F"][-1] == "deferred"


def test_pending_release_keeps_the_pick_label_during_the_grasp():
    """Trial 121's pattern: "open the gripper and go back home" said while a
    pick runs is deferred to the release pause — the approach, close and
    lift stay "pick up the vitamin tube"."""
    st = steps(40.0, moving=[(5.0, 9.0), (14.0, 17.0), (20.0, 24.0)])
    c = Cmds()
    c.ev("P", "received", 3.0, action="pick").ev("P", "accepted", 3.1, action="pick", route="pick",
                                                 target={"label": "vitamin_tube", "source": "voice"})
    c.ev("P", "cycle", 3.5, cycle_id=7)
    c.ev("F", "received", 6.0, action="open_gripper", attributes={"then": "release_home"})
    c.ev("F", "deferred", 6.0, action="open_gripper", attributes={"then": "release_home"})
    c.ev("H", "received", 12.0, action="drop_right").ev("H", "accepted", 12.1, action="drop_right",
                                                         route="executor_answer", cycle_id=7)
    c.ev("F", "accepted", 18.0, action="open_gripper", attributes={"then": "release_home"},
         route="executor_answer", cycle_id=7)
    c.ev("P", "completed", 25.0, cycle_id=7)
    tstates = ts((3.5, "probing", 7), (5.0, "approaching", 7), (9.5, "awaiting_destination", 7, "destination"),
                 (12.2, "carrying", 7), (17.5, "awaiting_release", 7, "release"), (18.1, "releasing", 7),
                 (20.0, "returning_home", 7), (24.5, "idle", 7))
    lab = label_episode({"command": c.events, "task_state": tstates}, st)
    texts = [i["instruction"] for i in lab["instructions"]]
    assert texts == ["pick up the vitamin tube", "drop it on the right side", "open the gripper and go back home"]
    pick = steps_of(lab, st, "pick up the vitamin tube")
    assert pick.min() >= 4.9 and pick.max() < 9.5          # approach .. lift, nothing of the answers
    opn = steps_of(lab, st, "open the gripper and go back home")
    assert opn.min() >= 18.0 and opn.max() >= 23.0        # release AND the requested way home
    hold = st["t"][lab["segment"] == SEGMENT_HOLD]
    # both pauses are holds: "Where should I drop it?" and "Should I open the gripper?"
    assert hold.size and set(np.round(hold, 1)) <= set(np.round(np.r_[np.arange(9.5, 12.2, 0.1),
                                                                      np.arange(17.5, 18.1, 0.1)], 1))


def test_compound_whose_second_part_did_not_run_is_labelled_as_executed():
    """Trial 084's pattern: release_home, but the executor asked "Should I go
    back home?" — the label is what the robot did, and it is flagged."""
    st = steps(20.0, moving=[(3.0, 5.0)])
    c = Cmds()
    c.ev("F", "received", 2.0, action="open_gripper", attributes={"then": "release_home"})
    c.ev("F", "accepted", 2.1, action="open_gripper", attributes={"then": "release_home"},
         route="executor_answer", cycle_id=3)
    tstates = ts((1.0, "awaiting_release", 3, "release"), (2.2, "releasing", 3),
                 (5.0, "awaiting_home_confirm", 3, "home"))
    lab = label_episode({"command": c.events, "task_state": tstates}, st)
    assert [i["instruction"] for i in lab["instructions"]] == ["open the gripper"]
    assert lab["instructions"][0]["requested_instruction"] == "open the gripper and go back home"
    assert any("executed only as" in f for f in lab["flags"])


def test_home_and_handover_compound_keeps_carry_hold_and_release():
    st = steps(20.0, moving=[(2.0, 6.0)])
    c = Cmds()
    c.ev("G", "received", 1.0, action="go_home", attributes={"then": "release"})
    c.ev("G", "accepted", 1.1, action="go_home", attributes={"then": "release"}, route="executor_answer", cycle_id=2)
    tstates = ts((0.5, "awaiting_destination", 2, "destination"), (1.2, "carrying", 2), (6.5, "releasing", 2),
                 (8.0, "awaiting_home_confirm", 2, "home"))
    lab = label_episode({"command": c.events, "task_state": tstates}, st)
    assert [i["instruction"] for i in lab["instructions"]] == ["bring it home and hand it over"]
    s = steps_of(lab, st, "bring it home and hand it over")
    assert s.min() >= 1.1 and s.max() >= 7.5


def test_pointing_target_and_question_survive():
    st = steps(20.0, moving=[(6.0, 9.0)])
    c = Cmds()
    c.ev("P0", "received", 1.0, action="pick", utterance="give me the vitamin tube")
    c.ev("P0", "rejected", 1.0, reason="I see two vitamin tubes. Could you say leftmost or rightmost, or point at it?")
    c.ev("Y", "received", 4.0, action="confirm_point", question="point", question_id=3, utterance="yes")
    c.ev("Y", "accepted", 4.0, action="pick", route="pick", question="point", question_id=3,
         target={"label": "vitamin_tube", "position": "leftmost", "bbox": [1, 2, 3, 4], "proposal_id": 3,
                 "source": "pointing"})
    c.ev("Y", "cycle", 4.5, cycle_id=9).ev("Y", "completed", 12.0, cycle_id=9)
    tstates = ts((4.5, "probing", 9), (6.0, "approaching", 9), (9.5, "awaiting_destination", 9, "destination"))
    events = {"command": c.events, "task_state": tstates,
              "point_question": [{"stamp": T0 + 3.0, "id": 3, "state": "open", "label": "vitamin_tube"},
                                 {"stamp": T0 + 4.0, "id": 3, "state": "yes"}]}
    lab = label_episode(events, st)
    ins = lab["instructions"][0]
    assert ins["instruction"] == "pick up the leftmost vitamin tube"
    assert ins["target_resolution"] == "pointing (confirmed)" and ins["question_id"] == 3
    assert ins["target"]["proposal_id"] == 3
    clar = st["t"][lab["segment"] == SEGMENT_CLARIFY]
    assert clar.size and clar.min() >= 1.0 and clar.max() < 6.0
    assert (lab["instruction_idx"][lab["segment"] == SEGMENT_CLARIFY] == ins["idx"]).all()


def test_automatic_close_after_handover_is_cleanup_not_the_open_command():
    st = steps(20.0, moving=[(3.0, 8.0)])
    c = Cmds()
    c.ev("O", "received", 2.0, action="release_home").ev("O", "accepted", 2.1, action="release_home",
                                                         route="executor_answer", cycle_id=4)
    tstates = ts((1.0, "awaiting_release", 4, "release"), (2.2, "releasing", 4), (3.0, "returning_home", 4),
                 (9.0, "idle", 4))
    events = {"command": c.events, "task_state": tstates,
              "gripper_cmd": [{"stamp": T0 + 2.3, "cmd": "open"}, {"stamp": T0 + 8.5, "cmd": "close"}]}
    lab = label_episode(events, st)
    k = lab["instructions"][0]["idx"]
    after = (st["t"] >= 8.5) & (st["t"] < 9.0)
    assert (lab["segment"][after] == SEGMENT_RESET).all()
    assert ((lab["instruction_idx"] == k) & (lab["segment"] == SEGMENT_TASK))[st["t"] < 8.4].any()


def test_assist_annotation_is_excluded():
    st = steps(20.0, moving=[(2.0, 6.0)])
    c = Cmds()
    c.ev("A", "received", 1.0, action="go_home").ev("A", "accepted", 1.1, action="go_home", route="named_pose")
    c.ev("A", "completed", 7.0)
    events = {"command": c.events, "annotation": [{"stamp": T0 + 3.0, "kind": "assist", "edge": "start"},
                                                  {"stamp": T0 + 4.0, "kind": "assist", "edge": "end"}]}
    lab = label_episode(events, st)
    band = (st["t"] >= 3.0) & (st["t"] < 4.0)
    assert (lab["segment"][band] == SEGMENT_ASSIST).all()


def test_reset_and_failed_cycle_are_cleanup():
    st = steps(20.0, moving=[(2.0, 4.0), (6.0, 9.0)])
    c = Cmds()
    c.ev("P", "received", 1.0, action="pick").ev("P", "accepted", 1.1, action="pick", route="pick",
                                                 target={"label": "small_bottle"})
    c.ev("P", "cycle", 1.5, cycle_id=5).ev("P", "failed", 10.0, cycle_id=5, reason="empty gripper")
    tstates = ts((1.5, "probing", 5), (2.0, "approaching", 5), (5.0, "no_grasp", 5, "home"),
                 (6.0, "returning_home", 5), (9.5, "idle", 5))
    lab = label_episode({"command": c.events, "task_state": tstates}, st)
    assert lab["instructions"][0]["outcome"] == "failed"
    rec = st["t"][lab["segment"] == SEGMENT_RESET]
    assert rec.size and rec.min() >= 5.0 and rec.max() < 9.5


def test_out_of_order_and_duplicated_delivery_give_the_same_labels():
    st = steps(30.0, moving=[(1.0, 4.0), (5.0, 8.0)])
    c = Cmds()
    c.ev("A", "received", 0.97, action="look_right").ev("A", "accepted", 0.98, action="look_right", route="named_pose")
    c.ev("A", "completed", 4.5).ev("B", "received", 0.978, action="go_home")
    c.ev("B", "queued", 0.979).ev("B", "accepted", 4.6, action="go_home", route="named_pose").ev("B", "completed", 8.5)
    ref = label_episode({"command": list(c.events)}, st)
    shuffled = list(c.events) + list(c.events[:3])           # duplicates
    random.Random(4).shuffle(shuffled)
    got = label_episode({"command": shuffled}, st)
    assert np.array_equal(ref["instruction_idx"], got["instruction_idx"])
    assert np.array_equal(ref["segment"], got["segment"])


def test_accepted_without_execution_evidence_is_flagged_not_invented():
    st = steps(10.0)
    c = Cmds()
    c.ev("A", "received", 1.0, action="go_home").ev("A", "accepted", 1.1, action="go_home", route="named_pose")
    c.ev("A", "completed", 3.0)
    lab = label_episode({"command": c.events}, st)
    assert lab["instructions"] == []
    assert any("did not move" in f for f in lab["flags"])
    assert (lab["segment"] == SEGMENT_IDLE).all()


def test_legacy_episode_keeps_old_labels_and_is_flagged():
    st = steps(10.0)
    ins = [{"idx": 0, "stamp": T0 + 1.0, "action_intent": "look_right", "instruction": "look to the right"},
           {"idx": 1, "stamp": T0 + 1.008, "action_intent": "go_home", "instruction": "go back home"}]
    lab = label_episode({"intent": []}, st, legacy_instructions=ins)
    assert lab["label_source"] == "legacy_intent_arrival"
    assert any("8 ms apart" in f for f in lab["flags"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
