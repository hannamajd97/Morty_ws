"""Settings profiles for the Advanced window (Majd 2026-08-28).

"make a Save button for the whole program, or Load if I have saved old
stuff" — node parameters are runtime-only (they die with the pipeline
process), so a tuned setup used to be unreproducible. This module owns
the on-disk format; app.py owns the ROS traffic.

Deliberately ROS-free so it stays importable — and testable — headless.

Defaults are NOT duplicated here. ``declared_defaults()`` parses the
``declare_parameter("name", default)`` calls out of a node's own source
at runtime, so the node's declarations stay the single source of truth:
change a default there and "Reset to defaults" follows it with no edit
here. ``pipeline_defaults()`` does that for
grasp_pipeline/selection/params.py, ``voice_defaults()`` for
communication_llm/speech_input_node.py (the four Voice-filter values —
the GUI's own spinbox startup values are a mirror of those declarations,
NOT a second source of truth). GUI-only defaults come from MotionWindow's
own widget constructors (see ``MotionWindow.gui_defaults``), for the same
reason.
"""
from __future__ import annotations

import ast
import os
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

from .config import (ALL_TUNABLE_PARAMS, FILTER_PARAMS, PERCEPTION_PARAMS,
                     VOICE_FILTER_PARAMS)

# ── on-disk format ───────────────────────────────────────────────────
SCHEMA_KEY = "morty_gui_profile"
SCHEMA_VERSION = 1

SECTION_GRIPPER = "gripper"
SECTION_COLLISION = "collision"
SECTION_FILTERS = "filters"
SECTION_PERCEPTION = "perception"
SECTION_MOTION = "motion_gui"
#: live parameters of the STT node (/speech_input_node) — a DIFFERENT node
#: from the pipeline, which is why they are not merged into
#: ``pipeline_values`` (2026-08-28: "the whole program").
SECTION_VOICE = "voice"

#: top-level keys that carry values (everything else is metadata)
VALUE_SECTIONS = (
    SECTION_GRIPPER,
    SECTION_COLLISION,
    SECTION_FILTERS,
    SECTION_PERCEPTION,
    SECTION_MOTION,
    SECTION_VOICE,
)
#: top-level keys that are metadata, not values
META_KEYS = (SCHEMA_KEY, "saved_at", "notes")

GRIPPER_KEYS: Tuple[str, ...] = ("pulse_hold_s_open", "pulse_hold_s_close")
#: the two pipeline params the Motion tab's "Collision avoidance" group owns.
#: They are ALSO in FILTER_PARAMS — see ``pipeline_values`` for the tie-break.
COLLISION_KEYS: Tuple[str, ...] = (
    "neighbour_obstacles_enabled",
    "neighbour_obstacle_padding_m",
)
#: the four Voice-filter params of the STT node. Same list ros_bridge
#: writes/reads (config.VOICE_FILTER_PARAMS is the one copy).
VOICE_KEYS: Tuple[str, ...] = tuple(VOICE_FILTER_PARAMS)
#: pure GUI state — no node holds these, they only steer what the GUI
#: launches. Names are profile keys, not ROS parameter names.
MOTION_KEYS: Tuple[str, ...] = (
    "executor_mode",
    "fast_replay",
    "fast_move_duration_s",
    "planning_library",
    "planner_id",
    "planning_time_s",
    "planning_attempts",
    "velocity_scaling",
    "acceleration_scaling",
)

_HEADER = """\
# Morty GUI settings profile.
# Written by the Advanced window's File menu -> "Save settings profile".
# Load it back with "Load...", or press "Reset to defaults" to go back to
# the values declared in grasp_pipeline/selection/params.py (and, for the
# "voice" section, in communication_llm/speech_input_node.py).
#
# "filters" and "perception" are LIVE parameters of the running pipeline
# node; "gripper" is the gripper_command_adapter; "voice" is the four
# mic/transcript gates on the speech_input_node (a different node — the
# voice stack can be down while the pipeline runs, and vice versa);
# "motion_gui" is pure GUI state (nothing holds it but the widgets).
# null = the value could not be read off the node when this file was
# saved; those keys are skipped on load. A profile written before the
# "voice" section existed simply has no such section, and loads fine.
# The two "collision" keys also appear under "filters" (they are ordinary
# pipeline parameters); on load "collision" wins and any disagreement is
# reported. Unknown keys are ignored with a warning, never fatal.
"""

PROFILE_DIR = os.path.join(
    os.environ.get(
        "WS_DIR", os.path.join(os.environ.get("HOME", os.path.expanduser("~")),
                               "morty_ws")),
    "config", "gui_profiles",
)


def ensure_profile_dir() -> str:
    """The profile directory, created on first use. Falls back to $HOME so
    the file dialog always opens somewhere writable."""
    try:
        os.makedirs(PROFILE_DIR, exist_ok=True)
        return PROFILE_DIR
    except OSError:
        return os.environ.get("HOME", os.path.expanduser("~"))


def default_filename() -> str:
    return time.strftime("morty_profile_%Y-%m-%d_%H%M.yaml")


# ── code defaults, parsed out of the nodes' own declarations ─────────
def _ws_dir() -> str:
    home = os.environ.get("HOME", os.path.expanduser("~"))
    return os.environ.get("WS_DIR", os.path.join(home, "morty_ws"))


def params_source_path() -> str:
    """Where the pipeline declares its parameters. Resolved from $WS_DIR /
    $HOME exactly the way params.py resolves its own workspace dir."""
    return os.path.join(_ws_dir(), "src", "grasp_pipeline", "grasp_pipeline",
                        "selection", "params.py")


def speech_source_path() -> str:
    """Where the STT node declares the four Voice-filter parameters.

    THE source of truth for "Reset all to defaults" on the voice section
    (2026-08-28). The GUI's own vf_* spinbox startup values are only a
    mirror of these declarations, so resetting to the node's own
    declare_parameter() values keeps one copy instead of two: retune
    speech_input_node.py and Reset follows it with no edit here.
    """
    return os.path.join(_ws_dir(), "src", "communication_llm",
                        "communication_llm", "speech_input_node.py")


_DEFAULTS_CACHE: Dict[str, Tuple[Dict[str, Any], str]] = {}


def declared_defaults(path: str) -> Tuple[Dict[str, Any], str]:
    """(name -> default, human-readable provenance note) for ONE source file.

    Reads every ``<obj>.declare_parameter("name", <literal>)`` call in the
    file with ast — works the same for the pipeline's params.py and for
    speech_input_node.py's ``self.declare_parameter`` calls. Non-literal
    defaults (computed paths, f-strings) are skipped — none of the values
    the GUI shows use one. A missing or unparsable file yields ({}, why)
    instead of raising: Reset then refuses honestly rather than pushing a
    hardcoded guess.
    """
    hit = _DEFAULTS_CACHE.get(path)
    if hit is not None:
        return hit
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=path)
    except (OSError, SyntaxError, ValueError) as exc:
        out = ({}, f"could not read defaults from {path}: {exc}")
        _DEFAULTS_CACHE[path] = out
        return out
    found: Dict[str, Any] = {}
    skipped: List[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not (isinstance(func, ast.Attribute)
                and func.attr == "declare_parameter"):
            continue
        if len(node.args) < 2:
            continue
        name_node = node.args[0]
        if not (isinstance(name_node, ast.Constant)
                and isinstance(name_node.value, str)):
            continue
        try:
            found[name_node.value] = ast.literal_eval(node.args[1])
        except (ValueError, SyntaxError):
            skipped.append(name_node.value)
    note = (f"parsed {len(found)} declare_parameter() defaults from "
            f"{os.path.basename(path)}")
    if skipped:
        note += f"; {len(skipped)} non-literal default(s) skipped"
    out = (found, note)
    _DEFAULTS_CACHE[path] = out
    return out


def pipeline_defaults(path: Optional[str] = None
                      ) -> Tuple[Dict[str, Any], str]:
    """Code defaults of the pipeline node (grasp_pipeline params.py)."""
    return declared_defaults(path or params_source_path())


def voice_defaults(path: Optional[str] = None
                   ) -> Tuple[Dict[str, Any], List[str], str]:
    """(defaults for VOICE_KEYS, keys with no default found, provenance).

    Same shape as ``tunable_defaults`` — parsed straight out of
    speech_input_node.py's ``declare_parameter`` calls (see
    ``speech_source_path``). When the file cannot be read the caller gets
    ({}, all four names, why) and must leave the voice filters ALONE
    rather than invent values.
    """
    found, note = declared_defaults(path or speech_source_path())
    have = {n: found[n] for n in VOICE_KEYS if n in found}
    missing = [n for n in VOICE_KEYS if n not in found]
    return have, missing, note


def tunable_defaults(names: Sequence[str] = ALL_TUNABLE_PARAMS,
                     path: Optional[str] = None
                     ) -> Tuple[Dict[str, Any], List[str], str]:
    """(defaults for `names`, names with no default found, provenance note)."""
    found, note = pipeline_defaults(path)
    have = {n: found[n] for n in names if n in found}
    missing = [n for n in names if n not in found]
    return have, missing, note


# ── serialise / parse ────────────────────────────────────────────────
def build_profile(gripper: Dict[str, Any], collision: Dict[str, Any],
                  filters: Dict[str, Any], perception: Dict[str, Any],
                  motion_gui: Dict[str, Any],
                  voice: Optional[Dict[str, Any]] = None,
                  notes: Sequence[str] = ()) -> Dict[str, Any]:
    """Assemble a profile document in display order.

    ``voice=None`` omits the "voice" section entirely — which is exactly
    what a pre-2026-08-28 profile looks like, so the old shape stays
    writable (and loadable) on purpose.
    """
    doc: Dict[str, Any] = {
        SCHEMA_KEY: SCHEMA_VERSION,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if notes:
        doc["notes"] = list(notes)
    doc[SECTION_GRIPPER] = dict(gripper)
    doc[SECTION_COLLISION] = dict(collision)
    doc[SECTION_MOTION] = dict(motion_gui)
    if voice is not None:
        doc[SECTION_VOICE] = dict(voice)
    doc[SECTION_FILTERS] = dict(filters)
    doc[SECTION_PERCEPTION] = dict(perception)
    return doc


def dump_profile(doc: Dict[str, Any]) -> str:
    """Profile dict -> YAML text (with the explanatory header). Raises
    ValueError when a value is not representable, so the caller reports a
    red status line instead of leaking a yaml traceback."""
    try:
        body = yaml.safe_dump(doc, default_flow_style=False, sort_keys=False,
                              allow_unicode=True)
    except yaml.YAMLError as exc:
        raise ValueError(f"could not serialise the profile — {exc}") from exc
    return _HEADER + body


def parse_profile(text: str) -> Dict[str, Any]:
    """YAML text -> profile dict. Raises ValueError on anything malformed
    (the caller turns that into a red status line, never a traceback)."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"not valid YAML — {exc}") from exc
    if doc is None:
        raise ValueError("file is empty")
    if not isinstance(doc, dict):
        raise ValueError(
            f"expected a YAML mapping at the top level, got "
            f"{type(doc).__name__}")
    return doc


def split_profile(doc: Dict[str, Any]
                  ) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """(section -> {key: value}, warnings).

    Every unknown top-level key, non-mapping section and unknown parameter
    name becomes a warning line; nothing raises.
    """
    warnings: List[str] = []
    out: Dict[str, Dict[str, Any]] = {s: {} for s in VALUE_SECTIONS}
    version = doc.get(SCHEMA_KEY)
    if version is None:
        warnings.append(
            f"no '{SCHEMA_KEY}' marker — loading it anyway as version "
            f"{SCHEMA_VERSION}")
    elif version != SCHEMA_VERSION:
        warnings.append(
            f"profile is format version {version!r}, this GUI writes "
            f"{SCHEMA_VERSION} — unknown keys will be skipped")
    allowed = {
        SECTION_GRIPPER: set(GRIPPER_KEYS),
        SECTION_COLLISION: set(COLLISION_KEYS),
        SECTION_FILTERS: set(FILTER_PARAMS),
        SECTION_PERCEPTION: set(PERCEPTION_PARAMS),
        SECTION_MOTION: set(MOTION_KEYS),
        SECTION_VOICE: set(VOICE_KEYS),
    }
    for key, value in doc.items():
        if key in META_KEYS:
            continue
        if key not in VALUE_SECTIONS:
            warnings.append(f"ignored unknown section '{key}'")
            continue
        if not isinstance(value, dict):
            warnings.append(
                f"ignored section '{key}': expected a mapping, got "
                f"{type(value).__name__}")
            continue
        for name, val in value.items():
            if not isinstance(name, str):
                warnings.append(f"ignored non-text key {name!r} in '{key}'")
                continue
            if name not in allowed[key]:
                warnings.append(f"ignored unknown key '{key}.{name}'")
                continue
            if val is None:
                warnings.append(
                    f"skipped '{key}.{name}': saved as null (unreadable when "
                    f"the profile was written)")
                continue
            out[key][name] = val
    return out, warnings


def pipeline_values(sections: Dict[str, Dict[str, Any]]
                    ) -> Tuple[Dict[str, Any], List[str]]:
    """filters + perception + collision -> one ROS parameter mapping.

    Everything this returns goes to the PIPELINE node in one
    SetParameters call, so "voice" is deliberately NOT merged in: those
    four belong to /speech_input_node and are applied separately by
    app._profile_apply. Sending them here would make the pipeline reject
    the whole batch as unknown parameters.

    The two neighbour_* keys live in both "collision" and "filters" (the
    Motion tab owns them, but they are ordinary pipeline parameters). A
    hand-edit can therefore disagree with itself: "collision" wins, and the
    disagreement is reported rather than silently resolved.
    """
    warnings: List[str] = []
    merged: Dict[str, Any] = {}
    merged.update(sections.get(SECTION_FILTERS, {}))
    merged.update(sections.get(SECTION_PERCEPTION, {}))
    for name, val in sections.get(SECTION_COLLISION, {}).items():
        if name in merged and merged[name] != val:
            warnings.append(
                f"'{name}' disagrees: collision={val!r} vs "
                f"filters={merged[name]!r} — using the collision value")
        merged[name] = val
    return merged, warnings
