"""chat_node tags every intent with its command / utterance / question ids
(recording contract v1) — the publisher wrapper and the utterance lookup,
without a running chat node."""
import json
import threading
import time

import pytest

cn = pytest.importorskip("communication_llm.chat_node")


class _Pub:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(json.loads(msg.data))


def _wrapper(ctx_holder):
    from std_msgs.msg import String  # noqa: F401 — the wrapper builds String messages
    pub = _Pub()
    w = cn._ActCountingPublisher(pub, lambda: None, context=lambda: ctx_holder.get("ctx"))
    return w, pub


def _send(w, action):
    from std_msgs.msg import String
    w.publish(String(data=json.dumps({"action_intent": action})))


def test_every_intent_gets_a_command_id_and_its_utterance_plan():
    holder = {"ctx": {"utterance_id": "utt-1", "text": "go home and close the gripper", "n_intents": 0}}
    w, pub = _wrapper(holder)
    _send(w, "go_home")
    _send(w, "close_gripper")
    a, b = pub.msgs
    assert a["command_id"] and b["command_id"] and a["command_id"] != b["command_id"]
    assert a["utterance_id"] == b["utterance_id"] == a["plan_id"] == "utt-1"
    assert (a["plan_index"], b["plan_index"]) == (0, 1)
    assert a["utterance"] == "go home and close the gripper" and "stamp" in a


def test_question_is_carried_and_no_context_still_gets_an_id():
    holder = {"ctx": {"utterance_id": "u", "text": "yes", "n_intents": 0, "question": "point", "question_id": 7}}
    w, pub = _wrapper(holder)
    _send(w, "confirm_point")
    assert pub.msgs[0]["question"] == "point" and pub.msgs[0]["question_id"] == 7
    holder["ctx"] = None
    _send(w, "go_home")
    assert pub.msgs[1]["command_id"] and "utterance_id" not in pub.msgs[1]


def test_utterance_id_lookup_matches_the_meta_of_the_same_text():
    class N:
        _utt_meta_lock = threading.Lock()
        _utt_meta = [[time.time(), "Yes.", 0.0, None, False, "id-from-speech"]]
    assert cn.ChatIntentNode._utterance_id_for(N, "Yes.", None) == "id-from-speech"
    assert cn.ChatIntentNode._utterance_id_for(N, "No.", None).startswith("voice-")
    assert cn.ChatIntentNode._utterance_id_for(N, "Yes.", time.time()).startswith("typed-")
