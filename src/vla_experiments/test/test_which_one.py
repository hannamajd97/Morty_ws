"""Which tube is "the middle one" (2026-10-01, Majd: a small bottle and three
vitamin tubes; "Give me the middle vitamin tube." picked the B12): the scene
text counts positions among objects of the SAME kind, and a bare answer to
the router's which-one question is an instant qualified pick."""
import threading
import types

import pytest

cn = pytest.importorskip("communication_llm.chat_node")
from communication_llm import voice_grammar as g  # noqa: E402

SCENE = [{"label": "vitamin_tube", "bbox": [383, 205, 457, 512], "cap_colour": "blue"},
         {"label": "small_bottle", "bbox": [0, 345, 67, 530], "cap_colour": ""},
         {"label": "vitamin_tube", "bbox": [501, 200, 570, 456], "cap_colour": "orange"},
         {"label": "vitamin_tube", "bbox": [212, 205, 277, 462], "cap_colour": "purple"}]


def _summary(scene):
    clock = types.SimpleNamespace(now=lambda: types.SimpleNamespace(nanoseconds=100e9))
    n = types.SimpleNamespace(_scene_awareness=True, _scene_lock=threading.Lock(), _latest_scene_at=99.9,
                              _latest_scene=scene, _scene_max_age_s=5.0, get_clock=lambda: clock)
    return cn.ChatIntentNode._scene_summary(n)


def test_positions_are_counted_among_the_same_kind():
    s = _summary(SCENE)
    assert "Vitamin B12] (leftmost of the 3 vitamin tubes)" in s
    assert "Multivitamin] (middle of the 3 vitamin tubes)" in s
    assert "Vitamin C] (rightmost of the 3 vitamin tubes)" in s
    assert s.count("(middle") == 1 and "small_bottle (" not in s      # the bottle is not a tube position
    two = _summary([SCENE[0], SCENE[3], SCENE[1]])
    assert "(left of the 2 vitamin tubes)" in two and "(right of the 2 vitamin tubes)" in two


@pytest.mark.parametrize("text,ans", [
    ("The middle one.", {"position": "middle"}), ("Middle.", {"position": "middle"}),
    ("the left one", {"position": "leftmost"}), ("The one on the right.", {"position": "rightmost"}),
    ("In the middle.", {"position": "middle"}), ("Purple cup.", {"colour": "purple"}),
    ("The blue one please", {"colour": "blue"}), ("No, the middle one.", {"position": "middle"}),
    ("The middle one and go home.", None), ("How many vitamin tubes do you see", None),
    ("left side of the table is dirty", None), ("Okay.", None)])
def test_which_answer_grammar(text, ans):
    assert g.match_which_answer(text) == ans
