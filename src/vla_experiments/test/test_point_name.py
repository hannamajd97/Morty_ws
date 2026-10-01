"""The pointing question names the exact tube (2026-10-01, Majd, trial 197:
three vitamin tubes, "You pointed at the vitamin tube. Should I grasp it?"),
"Yes, this one. I don't know the name of it." is the yes, and a "Grasp it."
right after the yes does not ask "Which object do you mean?"."""
import threading
import time
import types

import pytest

from communication_llm import voice_grammar as g

ir = pytest.importorskip("intent_router.intent_router_node")
cn = pytest.importorskip("communication_llm.chat_node")
R = ir.IntentRouterNode

SCENE = [{"label": "vitamin_tube", "bbox": [383, 205, 457, 512], "cap_colour": "blue"},
         {"label": "small_bottle", "bbox": [0, 345, 67, 530], "cap_colour": ""},
         {"label": "vitamin_tube", "bbox": [501, 200, 570, 456], "cap_colour": "orange"},
         {"label": "vitamin_tube", "bbox": [212, 205, 277, 462], "cap_colour": "purple"}]


# ── grammar: a consent followed by a remark ─────────────────────────────────
@pytest.mark.parametrize("text", [
    "Yes, this one. I don't know the name of it.",
    "Yes, this one, I don't know the name of it.",
    "Yes. I don't remember what it is called.",
    "This one. I don't know its name.",
    "Yes please. Thank you.",
])
def test_point_consent_with_remark_confirms(text):
    assert g.classify(text, "point", True) == ("answer", "confirm_point")


@pytest.mark.parametrize("text", [
    "Yes. No, the other one.", "Yes, the left one.", "Yes. Give me the small bottle.",
    "Yes, but not that one.", "Yes. Wait.", "Yes, the blue one.", "Yes. I don't want it.",
    "Yes, this one. Actually no.", "Yes. Then go home.", "Yes, this one, and drop it on the right.",
    "Yes, the vitamin C.", "Yes this one I don't know the name of it",   # one clause: whole patterns only
    "I don't know. Yes.", "Grab it. I'm not sure what it is called though",
])
def test_point_consent_with_action_or_choice_is_the_models(text):
    assert g.classify(text, "point", True) != ("answer", "confirm_point")


def test_point_consent_needs_the_point_question_and_fresh():
    t = "Yes, this one. I don't know the name of it."
    assert g.classify(t, "point", False) == g.LLM
    assert g.classify(t, "", True) == g.LLM
    assert g.classify(t, "release", True) != ("answer", "confirm_point")


@pytest.mark.parametrize("text,want", [
    ("Yes.", ("answer", "confirm_point")), ("Grab it.", ("answer", "confirm_point")),
    ("That one.", ("answer", "confirm_point")), ("No.", ("answer", "reject_point")),
    ("Not that one.", ("answer", "reject_point")), ("Stop.", ("stop", "stop")),
    ("Give me the small bottle.", g.LLM)])
def test_point_answers_unchanged(text, want):
    assert g.classify(text, "point", True) == want


# ── router: the name ────────────────────────────────────────────────────────
def _router(**kw):
    n = types.SimpleNamespace(_LABEL_DISPLAY_MAP=R._LABEL_DISPLAY_MAP, _CAP_PRODUCT=R._CAP_PRODUCT,
                              _detections=list(SCENE), _clarify=None, _cycle_object_name=None,
                              _last_cycle_id=0)
    n.__dict__.update(kw)
    for m in ("_to_display_label", "_pointed_object_name", "_point_question_line", "_object_words"):
        setattr(n, m, getattr(R, m).__get__(n))
    return n


def test_question_names_the_tube_from_the_proposal_cap():
    n = _router()
    for cap, want in (("orange", "Vitamin C tube"), ("yellow", "Vitamin C tube"),
                      ("purple", "Vitamin B12 tube"), ("blue", "Multivitamin tube")):
        name = n._pointed_object_name("vitamin_tube", cap, [0, 0, 1, 1])
        assert name == want
        assert n._point_question_line("vitamin_tube", None, name) == f"You pointed at the {want}. Should I grasp it?"


def test_name_from_the_live_scene_when_the_proposal_has_no_cap():
    n = _router()
    assert n._pointed_object_name("vitamin_tube", "", [214, 207, 275, 460]) == "Vitamin B12 tube"
    assert n._pointed_object_name("vitamin_tube", None, [900, 0, 950, 50]) == "vitamin tube"   # none near
    assert n._pointed_object_name("small_bottle", "", [0, 345, 67, 530]) == "small bottle"
    assert n._pointed_object_name("vitamin_tube", "green", [0, 0, 1, 1]) == "vitamin tube"


def test_other_kind_line_names_the_tube():
    n = _router(_clarify=("small bottle", time.monotonic(), None))
    assert n._point_question_line("vitamin_tube", None, "Vitamin C tube") == \
        "You pointed at the Vitamin C tube, not a small bottle. Do you want the Vitamin C tube?"


def test_object_words_only_for_the_pointed_cycle():
    n = _router(_cycle_object_name=(7, "vitamin_tube", "Vitamin C tube"), _last_cycle_id=7)
    assert n._object_words("vitamin_tube") == "Vitamin C tube"
    assert n._object_words("small_bottle") == "small bottle"
    assert n._object_words("") == ""
    n._last_cycle_id = 8                                  # the next pick
    assert n._object_words("vitamin_tube") == "vitamin tube"


# ── chat_node: the two utterances of trial 197 ─────────────────────────────
def _chat(router_q, task=None, yes_at=-1e9):
    calls = []
    log = types.SimpleNamespace(info=lambda *a, **k: None, debug=lambda *a, **k: None,
                                warning=lambda *a, **k: None, warn=lambda *a, **k: None)
    now = time.time()
    n = types.SimpleNamespace(
        _router_q=router_q, _task=task, _point_yes_at=yes_at, _held_object_label="",
        _pointed_object_label="vitamin_tube", _voice_ext_dest=True, _utt_meta_seen=True,
        _utt_local=threading.local(), _stop_epoch=0,
        get_logger=lambda: log,
        _live_task=lambda: task,
        _speech_start_for=lambda *a, **k: now,
        _fire_fast_command=lambda action, obj, text: calls.append(("command", action)),
        _fire_fast_pick=lambda *a, **k: calls.append(("pick",) + a[:1]),
        _fire_pause_negative=lambda *a: calls.append(("negative",) + a),
        _ask_which_object=lambda text: calls.append(("which-object", text)),
        _speak=lambda text: calls.append(("speak", text)),
        _note_instant=lambda *a: None,
        _route_legacy=lambda *a, **k: calls.append(("legacy",)),
    )
    n._on_router_question = cn.ChatIntentNode._on_router_question.__get__(n)
    return n, calls


def _route(n, text):
    try:
        cn.ChatIntentNode._route_minimal(n, text, None, time.monotonic(), None)
    except AttributeError as exc:     # reached the LLM path: not a grammar route
        return f"llm ({exc})"
    return None


def test_yes_with_remark_answers_the_open_pointing_question():
    q = {"question": "point", "id": 4, "since": time.time() - 3, "object": "vitamin_tube",
         "name": "Vitamin C tube"}
    n, calls = _chat(dict(q))
    assert _route(n, "Yes, this one. I don't know the name of it.") is None
    assert calls == [("command", "confirm_point")]


def test_grasp_it_right_after_the_yes_is_absorbed():
    q = {"question": "point", "id": 4, "since": time.time() - 3, "object": "vitamin_tube"}
    n, calls = _chat(dict(q))
    from std_msgs.msg import String
    import json
    n._on_router_question(String(data=json.dumps({"question": "", "id": 4, "closed": "yes"})))
    assert time.monotonic() - n._point_yes_at < 1.0
    for task in (None, {"phase": "approaching", "question": "", "holding": False}):
        n._task = task
        n._live_task = lambda task=task: task
        assert _route(n, "Grasp it.") is None      # handled here, not by the model
    assert ("which-object", "Grasp it.") not in calls and not any(c[0] == "speak" for c in calls), calls


def test_grasp_it_without_a_recent_yes_still_asks_which_object():
    n, calls = _chat({"question": "", "id": 3, "closed": "no"})
    assert _route(n, "Grasp it.") is None
    assert calls == [("which-object", "Grasp it.")]
