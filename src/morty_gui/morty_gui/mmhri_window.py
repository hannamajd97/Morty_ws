"""MMHRI experiment trial recorder — the doc's table, live in the program.

Implements the protocol from "MMHRI Paper Experiments.docx" (Hamid/Majd):
one row per trial with the exact 15 columns, auto-filled where the system
knows the answer (command used, interaction time, total time), and a
per-trial evidence folder:

    ~/morty_ws/experiments/mmhri/trial_NNN_LX_SN/     (e.g. trial_008_LA_S1)
        camera.mp4          annotated camera stream during the trial
        cloud.ply           the object's 3D cloud (colored, any viewer opens it)
        grasps.json         kept / best / executed grasp poses
        view3d.png          snapshot of the 3D panel
        audio.wav           microphone recording (parecord)
        conversation.txt    timestamped spoken/typed/replied lines
        trial.json          all fields + timings

Master tables (append-only):
    mmhri_experiments.csv   the doc's main table, column-for-column
    mmhri_meta.csv          Trial ID / Date / Scenario (the doc's meta table)

Timing (Majd's choice): ⏺ Start press is t=0. Interaction Time = t0 → object
selection published; Total Time = t0 → executor finished.

VLA (2026-09-03): the same ⏺ / ⏹ / ✖ also drive one training episode per
trial through vla_link.VlaLink — recorded to vla_experiments/datasets/,
never into the MMHRI folder, and kept only if the arm actually moved.
The "Record:" row chooses MMHRI evidence, VLA episode, or both (default).
"""
from __future__ import annotations

import csv
import datetime
import glob
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtWidgets

from .html_viewer import write_view3d_html
from .vla_link import VlaLink
from .bag_link import BagLink
from .managed_launch import die_with_parent_preexec

MMHRI_DIR = os.path.expanduser("~/morty_ws/experiments/mmhri")      # campaign CSVs + the trials before 2026-09-18
EXPERIMENTS_DIR = os.path.expanduser("~/morty_ws/experiments")      # day folders live directly here
VLA_DATASET_DIR = os.path.expanduser("~/morty_ws/vla_experiments/datasets/morty_ur3e")   # every VLA episode, one dataset


def day_dir(now: Optional[datetime.date] = None) -> str:
    """Today's folder: ~/morty_ws/experiments/experiments_18_09_2026 (Majd
    2026-09-18: directly in the experiments folder, named by the day)."""
    d = now or datetime.date.today()
    return os.path.join(EXPERIMENTS_DIR, d.strftime("experiments_%d_%m_%Y"))


def all_trial_dirs() -> List[str]:
    """Every trial folder: experiments/<session>/trial_*, plus the layouts
    before 2026-09-18 (experiments/mmhri/trial_* and mmhri/<session>/trial_*)."""
    out: List[str] = []

    def _scan(base: str, depth: int) -> None:
        try:
            for d in sorted(os.listdir(base)):
                p = os.path.join(base, d)
                if not os.path.isdir(p):
                    continue
                if d.startswith("trial_"):
                    out.append(p)
                elif depth > 0 and not d.startswith("."):
                    _scan(p, depth - 1)
        except OSError:
            pass
    _scan(EXPERIMENTS_DIR, 2)
    return sorted(set(out))

# The 7 experiment scenarios (Majd's MMHRI Paper Experiments PDF +
# chat spec, 2026-08-20). 3 layouts x 5 reps each = 105 trials.
# SINGLE SOURCE for scenarios: scenario_experiments.SCENARIOS.
# The two recorder windows used to carry divergent copies (S8-S10
# landed in only one of them, 2026-08-25) — never duplicate this list.
from .scenario_experiments import SCENARIOS as _SCENARIO_TUPLES
from .scenario_experiments import MODIFIERS
SCENARIOS = [f"{sid}: {txt}" for sid, txt in _SCENARIO_TUPLES]
LAYOUTS = ["Layout A", "Layout B", "Layout C"]
REPS_PER_CELL = 5
LEVELS = [
    "L1 single-step, explicit",
    "L2 single-step, deictic/multimodal",
    "L3 two-object relation",
    "L4 parameterized action",
    "L5 multi-step command",
    "L6 ambiguous natural command",
]
OBJECTS = [
    "tape_dispenser", "cd_case", "toy_faucet", "kitchen_towel", "nutmeg_jar",
    "retail_box", "small_bottle", "wooden_maze_puzzle", "vitamin_c",
]
CSV_COLUMNS = [
    "Trial ID", "Task ID", "Scene Layout", "Command Used", "Intended Target",
    "Intended Destination", "User Intent understood correctly?",
    "Object Detected?", "Correct Object Selected?", "Grasp Success?",
    "Action Executed Correctly?", "Full Task Success?",
    "Interaction Time (s)", "Total Time (s)", "Failure Category / Notes",
    # 2026-09-03: appended AFTER the doc's 15 columns (positions kept).
    "Modifiers",
]
YESNO_FIELDS = [
    ("intent", "User Intent understood correctly?"),
    ("detected", "Object Detected?"),
    ("correct_obj", "Correct Object Selected?"),
    ("grasp", "Grasp Success?"),
    ("action", "Action Executed Correctly?"),
    ("full", "Full Task Success?"),
]


def _write_ply(path: str, pts: np.ndarray, cols01: Optional[np.ndarray]) -> None:
    n = len(pts)
    has_c = cols01 is not None and len(cols01) == n
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if has_c:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        if has_c:
            c8 = np.clip(cols01 * 255.0, 0, 255).astype(np.uint8)
            for p, c in zip(pts, c8):
                f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]}\n")
        else:
            for p in pts:
                f.write(f"{p[0]:.5f} {p[1]:.5f} {p[2]:.5f}\n")


_GRASP_COLORS = {"kept": (46, 204, 64), "rejected": (211, 58, 58),
                 "best": (58, 176, 255), "exec": (243, 156, 18)}


def _quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Quaternion → 3×3 rotation matrix."""
    n = (x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n**0.5, y / n**0.5, z / n**0.5, w / n**0.5
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _write_grasps_ply(path: str, poses_by_kind: Dict) -> None:
    """Grasp poses as a colored line-glyph PLY (vertex+edge elements —
    MeshLab renders it on top of cloud.ply). Each grasp: an approach
    segment along +z and a finger-width bar along x. Colors: kept green,
    rejected red, best blue, executed orange."""
    verts: List[tuple] = []
    edges: List[tuple] = []
    for kind, poses in poses_by_kind.items():
        col = _GRASP_COLORS.get(kind, (200, 200, 200))
        for p in poses:
            o = np.array([p.position.x, p.position.y, p.position.z])
            r = _quat_to_mat(p.orientation.x, p.orientation.y,
                             p.orientation.z, p.orientation.w)
            zax, xax = r[:, 2], r[:, 0]
            base = len(verts)
            for v in (o, o + 0.05 * zax, o - 0.03 * xax, o + 0.03 * xax):
                verts.append((float(v[0]), float(v[1]), float(v[2]), *col))
            edges.append((base, base + 1))
            edges.append((base + 2, base + 3))
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\n")
        f.write("property int vertex1\nproperty int vertex2\nend_header\n")
        for v in verts:
            f.write(f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f} {v[3]} {v[4]} {v[5]}\n")
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")


def _poses_to_list(poses) -> List[Dict]:
    out = []
    for p in poses:
        out.append({
            "position": [p.position.x, p.position.y, p.position.z],
            "orientation": [p.orientation.x, p.orientation.y,
                            p.orientation.z, p.orientation.w],
        })
    return out


class MMHRIWindow(QtWidgets.QMainWindow):
    """The experiment table + per-trial recorder."""

    def __init__(self, bridge=None) -> None:
        super().__init__()
        # Optional ROS bridge: enables the passive event recorder that
        # yields the per-stage timings (intent/yolo/graspgen/execution).
        self._bridge = bridge
        self.setWindowTitle("MMHRI Experiments — trial recorder")
        self.resize(560, 720)
        os.makedirs(MMHRI_DIR, exist_ok=True)
        self._active = False
        # One trial = one save (2026-09-25): see _on_stop_clicked / _on_stop_save_final.
        self._stop_pressed = False
        self._saved = False
        self._cancelled = False
        self._rec_header: Dict[str, str] = {}
        self._t0 = 0.0
        self._trial_root = ""   # the trial's one folder, always (VLA episode + bag + MMHRI evidence)
        self._trial_dir = ""    # the same folder when MMHRI evidence is recorded, else "" 
        self._video: Optional["object"] = None
        self._audio_proc: Optional[subprocess.Popen] = None
        self._conv_f = None
        self._interaction_s: Optional[float] = None
        self._total_s: Optional[float] = None
        self._command_seen = False
        self._cloud_saved = False
        self._conv_entries: List[tuple] = []
        self._vla = VlaLink(bridge, self)
        self._bag = BagLink(self)          # one ROS bag per trial (Hamid 2026-09-09)
        self._vla_ended = False

        central = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(central)

        # Where today's trials go — one folder per day, editable (Majd
        # 2026-09-18: "I have to be allowed to change it"). The name is
        # proposed from the date; once edited it is kept for this window.
        folder_row = QtWidgets.QHBoxLayout()
        folder_row.addWidget(QtWidgets.QLabel("Session folder:"))
        self.folder_edit = QtWidgets.QLineEdit()
        self.folder_edit.setToolTip("Sub-folder of ~/morty_ws/experiments that receives this session's trials\n"
                                    "(video, audio, clouds, grasps, events, timeline, bag/). Proposed from today's\n"
                                    "date; type another name to change it. The campaign CSVs stay in experiments/mmhri/.\n"
                                    f"VLA episodes are one training dataset and always go to\n{VLA_DATASET_DIR}/episode_NNN;\n"
                                    "each trial's trial.json names its episode, each episode's meta.json names its trial.")
        self.folder_edit.textEdited.connect(lambda _t: setattr(self, "_folder_edited", True))
        folder_row.addWidget(self.folder_edit, stretch=1)
        self.folder_hint = QtWidgets.QLabel()
        folder_row.addWidget(self.folder_hint)
        self._folder_edited = False
        self._refresh_folder_label()
        lay.addLayout(folder_row)

        form_box = QtWidgets.QGroupBox("Trial")
        form = QtWidgets.QFormLayout(form_box)
        self.trial_id = QtWidgets.QLineEdit(self._next_trial_id())
        form.addRow("Trial ID:", self.trial_id)
        # Task ID field removed from the UI (Majd 2026-08-25: "delete
        # Task ID") — the CSV column stays for the doc's table shape but
        # is written empty.
        self.layout_edit = QtWidgets.QComboBox()
        self.layout_edit.addItems(LAYOUTS)
        form.addRow("Scene Layout:", self.layout_edit)
        self.scenario = QtWidgets.QComboBox()
        self.scenario.addItems(SCENARIOS)
        form.addRow("Scenario:", self.scenario)
        # Interaction modifiers (2026-09-03, Majd): the former S8-S10,
        # now optional tick-boxes that can apply to ANY scenario trial.
        # Multi-select on purpose — one trial may use several.
        mod_box = QtWidgets.QWidget()
        mod_lay = QtWidgets.QHBoxLayout(mod_box)
        mod_lay.setContentsMargins(0, 0, 0, 0)
        self.modifier_boxes: Dict[str, QtWidgets.QCheckBox] = {}
        for mid, mtxt, old_sid, integrated in MODIFIERS:
            short = mtxt.split(" — ")[0]
            cb = QtWidgets.QCheckBox(
                f"{mid} {short}" + ("" if integrated else " (not integrated)"))
            cb.setToolTip(
                f"{mid} (was {old_sid}): {mtxt}\n"
                + ("Tick if you USED this during the trial."
                   if integrated else
                   "Not implemented in Morty yet — kept for the record only."))
            self.modifier_boxes[mid] = cb
            mod_lay.addWidget(cb)
        mod_lay.addStretch(1)
        mod_box.setToolTip(
            "Optional behaviours you may add on top of the scenario\n"
            "(the former S8-S10). Tick whichever you actually used —\n"
            "none, one, or several. Saved as e.g. 'M1;M2'.")
        form.addRow("Modifiers:", mod_box)
        # Level (L1-L6 difficulty table) removed from the form (Majd,
        # 2026-08-20: superseded by the 7 scenarios; the PDF itself said
        # "maybe for now we skip"). Hidden widget kept so trial.json /
        # mmhri_meta.csv keep their column alignment.
        self.level = QtWidgets.QComboBox()
        self.level.addItems(LEVELS)
        self.level.setVisible(False)
        self.target = QtWidgets.QComboBox()
        self.target.setEditable(True)
        self.target.addItems(OBJECTS)
        form.addRow("Intended Target:", self.target)
        self.destination = QtWidgets.QComboBox()
        self.destination.setEditable(True)
        self.destination.addItems(["—", "drop_left", "drop_right", "home"])
        self.destination.setToolTip(
            "Where the object should be DROPPED after the pick\n"
            "(only meaningful for scenarios that grasp something)."
        )
        form.addRow("Intended Destination:", self.destination)
        # What ⏺ records (Majd 2026-09-03: "on default both are on"):
        # MMHRI = this window's evidence folder + CSV row; VLA = one
        # training episode (both cameras + joints) per trial, which the
        # recorder keeps only if the arm moves. Either alone is legal —
        # MMHRI off is a pure demonstration-collection session.
        # Sits ABOVE the Trial box (Majd 2026-09-03: "normally they put
        # them on top") — you decide what a session records before you
        # describe any trial.
        rec_box = QtWidgets.QGroupBox("Record")
        rec_lay = QtWidgets.QHBoxLayout(rec_box)
        self.rec_mmhri = QtWidgets.QCheckBox("MMHRI evidence")
        self.rec_mmhri.setChecked(True)
        self.rec_mmhri.setToolTip(
            "The paper's record: trial folder (video, audio, clouds,\n"
            "conversation, logs) + one row in mmhri_experiments.csv.")
        self.rec_vla = QtWidgets.QCheckBox("VLA episode")
        self.rec_vla.setChecked(True)
        self.rec_vla.setToolTip(
            "One training episode per trial (Astra + RealSense + joints),\n"
            "saved under vla_experiments/datasets/ — separate from MMHRI.\n"
            "Kept ONLY if the arm moves; a 'what do you see' trial is\n"
            "dropped by itself. The recorder starts with this window.")
        self.rec_vla.toggled.connect(self._vla.set_enabled)
        self.rec_bag = QtWidgets.QCheckBox("ROS bag")
        self.rec_bag.setChecked(True)
        self.rec_bag.setToolTip(
            "ros2 bag record for the span of the trial: both cameras\n"
            "(compressed), joints, TCP pose, gripper I/O, controller state,\n"
            "transcript, intents, detections, executor status, /tf.\n"
            "In the trial folder (bag/), or vla_experiments/bags/ for a\n"
            "VLA-only trial. ~2.5 MB/s.")
        rec_lay.addWidget(self.rec_mmhri)
        rec_lay.addWidget(self.rec_vla)
        rec_lay.addWidget(self.rec_bag)
        rec_lay.addStretch(1)
        rec_box.setToolTip("What ⏺ Start records. Both on by default.")
        # Target/destination only make sense for some scenarios:
        # S1 (scene question) and S7 (move only) have neither; S2 has a
        # target (the absent object!) but nothing to drop. Grey out what
        # doesn't apply so the form can't confuse (Majd, 2026-08-20;
        # the table is scenario_experiments.SCENARIO_FIELDS).
        self.scenario.currentIndexChanged.connect(self._on_scenario_changed)
        # Command field removed from the form (Majd, 2026-08-20: the
        # whole conversation is recorded anyway) — the hidden widget
        # stays so the auto-fill wiring and the CSV column keep working.
        self.command = QtWidgets.QLineEdit()
        self.command.setVisible(False)
        lay.addWidget(rec_box)
        lay.addWidget(form_box)

        out_box = QtWidgets.QGroupBox("Outcome (the doc's yes/no columns)")
        out_form = QtWidgets.QFormLayout(out_box)
        self.chips: Dict[str, QtWidgets.QComboBox] = {}
        for key, label in YESNO_FIELDS:
            cb = QtWidgets.QComboBox()
            cb.addItems(["", "yes", "no"])
            self.chips[key] = cb
            out_form.addRow(label, cb)
        # Both time columns are MEASURED automatically at ⏹ Stop (Majd,
        # 2026-08-20: "should be auto"). Read-only so it's obvious.
        self.t_interact = QtWidgets.QLineEdit()
        self.t_interact.setReadOnly(True)
        self.t_interact.setPlaceholderText("auto — measured at ⏹ Stop")
        self.t_interact.setToolTip(
            "How long it took to CONVEY the command:\n"
            "from your spoken/typed words until Morty extracted the intent."
        )
        out_form.addRow("Interaction Time (s):", self.t_interact)
        self.t_total = QtWidgets.QLineEdit()
        self.t_total.setReadOnly(True)
        self.t_total.setPlaceholderText("auto — measured at ⏹ Stop")
        self.t_total.setToolTip(
            "Whole task: from your command until the robot finished the\n"
            "final action (cycle done)."
        )
        out_form.addRow("Total Time (s):", self.t_total)
        # Per-stage breakdown of the total, measured from the live
        # topics — shows WHERE the time went inside one trial.
        self.stage_lbl = QtWidgets.QLabel("auto — filled at ⏹ Stop")
        self.stage_lbl.setStyleSheet("font-family:monospace; color:#2a6;")
        self.stage_lbl.setToolTip(
            "The Total Time split into the 4 pipeline stages:\n"
            "  understanding: your words → intent extracted (LLM)\n"
            "  find object:   intent → object chosen in YOLO\n"
            "  grasp gen:     object cloud → grasp poses (GraspGen)\n"
            "  execution:     grasp poses → robot finished moving"
        )
        out_form.addRow("Where the time went:", self.stage_lbl)
        self.notes = QtWidgets.QLineEdit()
        out_form.addRow("Failure / Notes:", self.notes)
        lay.addWidget(out_box)

        btn_row = QtWidgets.QHBoxLayout()
        self.start_btn = QtWidgets.QPushButton("⏺ Start trial")
        self.start_btn.setStyleSheet(
            "QPushButton {background:#1d7a2c; color:white; font-weight:bold;}"
        )
        self.start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(self.start_btn)
        self.stop_btn = QtWidgets.QPushButton("⏹ Stop && save")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop_clicked)
        btn_row.addWidget(self.stop_btn)
        self.discard_btn = QtWidgets.QPushButton("✖ discard")
        self.discard_btn.setEnabled(False)
        self.discard_btn.clicked.connect(self._on_discard)
        btn_row.addWidget(self.discard_btn)
        # Delete = erase a trial FROM THE RECORDS (Majd 2026-09-03): its
        # folder if any, its rows in both CSVs (modifiers included), the
        # number free again. Discard only aborts the running one.
        self.delete_btn = QtWidgets.QPushButton("🗑 delete")
        self.delete_btn.clicked.connect(self._on_delete)
        btn_row.addWidget(self.delete_btn)
        # 2026-09-28 (Majd): pick trials and send them to the Excel workbook
        # (excel_window.ExcelExportWindow; refreshed after every trial).
        self.excel_btn = QtWidgets.QPushButton("📊 Excel export")
        self.excel_btn.setToolTip("Choose trials and send everything recorded about them to\n"
                                  "experiments/MMHRI_experiments.xlsx")
        self.excel_btn.clicked.connect(self._open_excel_window)
        btn_row.addWidget(self.excel_btn)
        # 2026-09-28 (Majd: "a button just for fast sending the last record")
        self.excel_last_btn = QtWidgets.QPushButton("📊 Send last trial")
        self.excel_last_btn.setToolTip("Adds the trial you saved last to the Excel file (only if it is not in it yet;\n"
                                       "nothing already in the file changes). Waits for its final times first.")
        self.excel_last_btn.clicked.connect(self._send_last_trial)
        btn_row.addWidget(self.excel_last_btn)
        lay.addLayout(btn_row)

        self.status = QtWidgets.QLabel("idle — fill the header, then ⏺ Start")
        self.status.setStyleSheet("color:#888;")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        # 7×3×5 progress grid, counted from the saved trial.json files.
        self.progress_lbl = QtWidgets.QLabel("")
        self.progress_lbl.setStyleSheet("font-family:monospace; color:#888;")
        lay.addWidget(self.progress_lbl)
        lay.addStretch(1)
        # Hover documentation for every field (Majd 2026-08-25: "when I
        # stop my mouse on everything it should explain what it does").
        self.trial_id.setToolTip(
            "Unique number of THIS run. Auto-increments; one folder\n"
            "experiments/mmhri/trial_NNN_LX_SN/ is created per trial\n"
            "(layout + scenario in the name, e.g. trial_008_LA_S1).")
        self.layout_edit.setToolTip(
            "Which physical arrangement of the objects is on the table\n"
            "(Layout A / B / C from the experiment plan). 5 repetitions\n"
            "of every scenario are recorded per layout.")
        self.scenario.setToolTip(
            "The experiment scenario being tested in this trial\n"
            "(S1-S7). Determines what you say/do and which outcome\n"
            "fields matter. Optional extras go in Modifiers below.")
        self.level.setToolTip(
            "Command-complexity level from the doc (L1 simple explicit\n"
            "... L6 ambiguous natural language). Hidden metadata.")
        self.target.setToolTip(
            "Intended Target = the object YOU plan to ask for in this\n"
            "trial (ground truth). Used to judge 'Correct Object\n"
            "Selected?' — it is NOT sent to the robot.")
        self.destination.setToolTip(
            "Intended Destination = where YOU plan to tell the robot to\n"
            "drop it (ground truth for 'Action Executed Correctly?').\n"
            "'—' for scenarios without a drop.")
        self.command.setToolTip(
            "The exact words you actually said/typed — auto-filled from\n"
            "the first accepted transcript of the trial.")
        _chip_docs = {
            "intent": "Did the LLM extract the right intent from your words\n(regardless of what happened after)?",
            "detected": "Did YOLO detect the intended object at all\n(a box on it, any label)?",
            "correct_obj": "Did the system SELECT the intended object\n(not a sibling/wrong one)?",
            "grasp": "Did the gripper physically hold the object\n(lifted without dropping)?",
            "action": "Was the commanded action performed as asked\n(right destination, right behavior)?",
            "full": "Everything end-to-end as intended?\nYes only if ALL previous are Yes.",
        }
        for _k, _doc in _chip_docs.items():
            if _k in self.chips:
                self.chips[_k].setToolTip(_doc)
        self.t_interact.setToolTip(
            "Measured automatically: your command -> object selection\n"
            "published (the human-interaction part of the trial).")
        self.notes.setToolTip(
            "Free text: why it failed / anything unusual\n"
            "(goes into the CSV's last column).")
        self.start_btn.setToolTip(
            "t=0 of the trial. Starts camera+audio recording and the\n"
            "event logger. Fill the header fields FIRST.")
        self.stop_btn.setToolTip(
            "Ends the trial: waits for Morty to finish speaking, then\n"
            "saves video/audio/events/CSV row. Fill the outcome\n"
            "dropdowns before or right after.")
        self.discard_btn.setToolTip(
            "Abort the RUNNING trial: nothing is saved, the trial number is\n"
            "reused. (A folder already opened stays — use delete.)")
        self.delete_btn.setToolTip(
            "While a trial RECORDS: stop it and erase everything it recorded\n"
            "(audio, video, bag, VLA images) — nothing is saved, the number is reused.\n"
            "Otherwise: erase the trial whose number is in 'Trial ID' from the records:\n"
            "its folder (if any), its rows in mmhri_experiments.csv and\n"
            "mmhri_meta.csv (modifiers included). Asks first; CSV backups\n"
            "go to backups/. VLA episodes are NOT touched — only reported.")
        # The LABELS need the tooltip too — hovering the text "Intended
        # Target:" hits a separate QLabel, not the input box (Majd
        # 2026-08-25: "no it doesn't").
        for _form in (form, out_form):
            for _i in range(_form.rowCount()):
                _field = _form.itemAt(_i, QtWidgets.QFormLayout.FieldRole)
                _lab = _form.itemAt(_i, QtWidgets.QFormLayout.LabelRole)
                if _field is not None and _lab is not None                         and _field.widget() is not None                         and _lab.widget() is not None                         and _field.widget().toolTip():
                    _lab.widget().setToolTip(_field.widget().toolTip())
        self.setCentralWidget(central)
        self._refresh_progress()

    def _on_scenario_changed(self, idx: int) -> None:
        """Enable target/destination only where the scenario has them —
        scenario_experiments.SCENARIO_FIELDS, kept next to SCENARIOS (since
        the 2026-09-25 renumbering: S1 neither, S2 target only, S3-S6 both,
        S7 moving only = neither)."""
        from .scenario_experiments import SCENARIO_FIELDS
        sid = SCENARIOS[idx].split(":")[0]
        has_target, has_dest = SCENARIO_FIELDS.get(sid, (True, True))
        self.target.setEnabled(has_target)
        self.destination.setEnabled(has_dest)
        if not has_target:
            self.target.setCurrentText("—")
        elif self.target.currentText().strip() in ("", "—"):
            self.target.setCurrentIndex(0)
        if not has_dest:
            self.destination.setCurrentText("—")

    # ── trial lifecycle ──────────────────────────────────────────────
    def _refresh_folder_label(self) -> None:
        if not getattr(self, "_folder_edited", False):
            self.folder_edit.setText(os.path.basename(day_dir()))
        self.folder_hint.setText("/ trial_NNN_LX_SN/      VLA → vla_experiments/datasets/morty_ur3e/")

    def session_dir(self) -> str:
        """~/morty_ws/experiments/<name typed or today's>."""
        name = (self.folder_edit.text().strip() or os.path.basename(day_dir())).replace("/", "_")
        return os.path.join(EXPERIMENTS_DIR, name)

    def _next_trial_id(self) -> str:
        """Trial IDs keep counting across days (the tally is campaign-wide)."""
        n = 1
        try:
            ids = [int(os.path.basename(d).split("_")[1]) for d in all_trial_dirs()]
            if ids:
                n = max(ids) + 1
        except (OSError, ValueError):
            pass
        return f"{n:03d}"

    # ── VLA coupling ─────────────────────────────────────────────────
    def showEvent(self, ev) -> None:
        super().showEvent(ev)
        # The recorder comes up WITH this window so it is warm by the
        # first ⏺ (Majd 2026-09-03: "no need for vla record here").
        self._vla.set_enabled(self.rec_vla.isChecked())

    def vla_shutdown(self) -> None:
        self._vla.shutdown()
        self._bag.shutdown()

    def _poll_bag_ready(self) -> None:
        """2026-09-28: switch the status to "speak now" once the bag has
        subscribed to every topic (see _on_start); marks the moment in the
        trial's events ("trial bag ready") so the timeline shows it."""
        if not self._active or self._bag_ended:
            self._bag_ready_timer.stop()
            return
        ready = False
        try:
            with open(self._bag_log, "rb") as f:
                ready = b"All requested topics are subscribed" in f.read()
        except OSError:
            pass
        waited = time.monotonic() - self._bag_wait_t0
        if not ready and waited < 10.0:
            return
        self._bag_ready_timer.stop()
        if self._bridge is not None:
            try:
                self._bridge.trial_mark("trial", "bag ready" if ready else "bag NOT confirmed after 10 s")
            except Exception:
                pass
        self.status.setText(self._bag_go_text if ready else
                            self._bag_go_text + "  (bag start not confirmed after 10 s — check bag_record.log)")

    def _open_excel_window(self) -> None:
        if getattr(self, "_excel_win", None) is None:
            from .excel_window import ExcelExportWindow
            self._excel_win = ExcelExportWindow(self, is_busy=lambda: bool(getattr(self, "_active", False)))
        self._excel_win.refresh()
        self._excel_win.show()
        self._excel_win.raise_()
        self._excel_win.activateWindow()

    def _last_trial_folder(self) -> str:
        """The trial saved last in this window, else the newest trial.json."""
        f = getattr(self, "_last_saved_trial", "")
        if f and os.path.exists(os.path.join(f, "trial.json")):
            return f
        cands = [p for p in all_trial_dirs() if os.path.exists(os.path.join(p, "trial.json"))]
        return max(cands, key=lambda p: os.path.getmtime(os.path.join(p, "trial.json"))) if cands else ""

    def _send_last_trial(self) -> None:
        if getattr(self, "_active", False):
            self.status.setText("✖ not while a trial is recording")
            return
        folder = self._last_trial_folder()
        if not folder:
            self.status.setText("✖ no saved trial to send")
            return
        if getattr(self, "_excel_sender", None) is None:
            from .excel_window import ExcelSender
            self._excel_sender = ExcelSender(self)
            self._excel_sender.progress.connect(lambda t: self.status.setText(t))
            self._excel_sender.finished.connect(self._on_last_trial_sent)
        if self._excel_sender.running:
            return
        self.excel_last_btn.setEnabled(False)
        self.status.setText(f"⏳ sending {os.path.basename(folder)} to Excel …")
        self._excel_sender.send([folder])

    def _on_last_trial_sent(self, res: dict) -> None:
        from .excel_window import ExcelSender
        self.excel_last_btn.setEnabled(True)
        self.status.setText(ExcelSender.describe(res))
        self._refresh_excel_window()

    def _refresh_excel_window(self) -> None:
        """After a saved or deleted trial: the export list shows it."""
        w = getattr(self, "_excel_win", None)
        if w is not None:
            try:
                w.refresh()
            except Exception:
                pass

    def _start_postprocess(self, trial_root: str) -> None:
        """2026-09-28: after a saved trial, a DETACHED process
        (morty_gui.trial_postprocess) makes the full-rate joints
        (joints_full.npz/.csv, joint_acc + joint_current in steps.npz) and
        viewer.html, then rebuilds the stage times from exact event times
        (trial_times) into the trial files and this trial's CSV rows. It waits
        for the bag and the episode, needs no GUI, and a GUI closed meanwhile
        does not stop it. Output: postprocess.log."""
        if not trial_root or not os.path.isdir(trial_root):
            return
        try:
            log = open(os.path.join(trial_root, "postprocess.log"), "ab")
            subprocess.Popen([sys.executable, "-m", "morty_gui.trial_postprocess", trial_root],
                             stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True, env=os.environ.copy())
            log.close()
        except Exception as exc:  # never let bookkeeping break the trial window
            print(f"[mmhri] post-processing not started for {trial_root}: {exc}")

    def close_active_trial(self) -> None:
        """2026-09-28 (trial 081: the GUI was closed with "Keep running"
        while the trial recorded; its parecord and ros2 bag ran on for 32
        minutes, unowned, and wrote 3.9 GB into 081 while trials 082-085
        were recorded): "Keep running" keeps the robot infrastructure, never
        a trial's own recorders. The VLA episode, the bag and the audio end
        here; no trial.json is written (the trial was not stopped by ⏹), so
        the folder stays marked incomplete. A GUI crash is covered by
        die_with_parent_preexec on both recorders."""
        if not getattr(self, "_active", False):
            return
        self._cancelled = True           # a ⏹ chain still waiting must not save it
        if not getattr(self, "_vla_ended", True):
            self._vla_ended = True
            self._vla.end(None)
        if not getattr(self, "_bag_ended", True):
            self._bag_ended = True
            self._bag.end()
        self._finish_recorders()
        if self._bridge is not None:
            try:
                self._bridge.trial_stop()
            except Exception:
                pass

    def _fallback_task(self) -> str:
        """The episode's instruction if nothing was said (clicked trials)."""
        sid, _, text = self.scenario.currentText().partition(": ")
        target = self.target.currentText().strip()
        dest = self.destination.currentText().strip()
        if target in ("", "—"):
            return f"{sid}: {text}"
        task = f"pick up the {target.replace('_', ' ')}"
        task += {"drop_left": " and drop it on the left",
                 "drop_right": " and drop it on the right",
                 "home": " and bring it home"}.get(dest, "")
        return task

    def _vla_success(self) -> Optional[bool]:
        """The human verdict if already given; None lets the executor's
        cycle_done decide on the recorder side."""
        v = self.chips["full"].currentText()
        return True if v == "yes" else False if v == "no" else None

    def _on_start(self) -> None:
        tid = self.trial_id.text().strip() or self._next_trial_id()
        # Folder = number + layout + scenario (Majd 2026-09-03: "atm we
        # have only the trial number in the folder"), e.g. trial_008_LA_S1.
        # _next_trial_id() still parses the number from split("_")[1].
        want_mmhri, want_vla = self.rec_mmhri.isChecked(), self.rec_vla.isChecked()
        want_bag = self.rec_bag.isChecked()
        if not (want_mmhri or want_vla or want_bag):
            self.status.setText("nothing to record — tick MMHRI, VLA and/or ROS bag first")
            return
        la = self.layout_edit.currentText().strip()[-1:]
        sid = self.scenario.currentText().split(":")[0]
        # One trial = one folder, whatever is recorded (2026-09-25, Majd:
        # "everything in the same package, not in 2, VLA and experiments").
        # MMHRI off only silences the evidence writers (every feed_* checks
        # _trial_dir); the VLA episode and the bag still go into the folder
        # instead of vla_experiments/datasets and vla_experiments/bags.
        self._refresh_folder_label()
        self._trial_root = os.path.join(self.session_dir(), f"trial_{tid}_L{la}_{sid}")
        os.makedirs(self._trial_root, exist_ok=True)
        self._trial_dir = self._trial_root if want_mmhri else ""
        self._t0 = time.monotonic()
        self._interaction_s = None
        self._total_s = None
        self._command_seen = False
        self._cloud_saved = False
        self._conv_entries = []
        self._last_cloud = None  # this trial's cloud only, not the previous one's
        # Byte offsets of the component logs at ⏺ — the slice written
        # during the trial is copied into the trial folder at ⏹ (Majd,
        # 2026-08-12: "the logs should be inside the experiments folder").
        self._log_offsets = {}
        for name in ("voice_gui.log", "pipeline_gui.log",
                     "driver_gui.log", "executor_gui.log"):
            path = os.path.expanduser(f"~/morty_ws/logs/{name}")
            try:
                self._log_offsets[path] = os.path.getsize(path)
            except OSError:
                self._log_offsets[path] = 0
        self._active = True
        # The record's header is fixed at ⏺ (2026-09-25): the save read the
        # form again at ⏹, and trial 067 was written as "068" after the form
        # had moved on to the next trial. Outcome fields stay live until ⏹.
        self._rec_header = {"tid": tid, "layout": self.layout_edit.currentText().strip(),
                            "scenario": self.scenario.currentText(), "level": self.level.currentText()}
        self._stop_pressed = self._saved = self._cancelled = False
        # Locked while it records: the record keeps these values, and a
        # combo box changed by a stray mouse wheel no longer disagrees with
        # the folder name (5 saved trials did, 2026-09-25).
        for w in (self.trial_id, self.layout_edit, self.scenario, self.level):
            w.setEnabled(False)
        # Wall-clock video pacer (10 Hz): duplicates the newest frame so
        # camera.mp4 duration always equals real time (audio sync).
        if getattr(self, "_video_timer", None) is None:
            self._video_timer = QtCore.QTimer(self)
            self._video_timer.timeout.connect(self._video_tick)
        self._video_t0 = 0.0
        self._video_frames = 0
        self._latest_bgr = None
        if self._trial_dir:
            self._video_timer.start(100)
        # Passive event recorder → stage timings + events.json.
        if self._bridge is not None:
            try:
                self._bridge.trial_start()
            except Exception:
                pass
        # audio (best-effort; missing parecord must not block the trial)
        self._audio_proc = None
        self._conv_f = None
        if self._trial_dir:
            try:
                self._audio_proc = subprocess.Popen(
                    ["parecord", "--file-format=wav",
                     os.path.join(self._trial_dir, "audio.wav")],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    preexec_fn=die_with_parent_preexec,   # 2026-09-28: see close_active_trial
                )
            except OSError:
                self._audio_proc = None
            self._conv_f = open(os.path.join(self._trial_dir, "conversation.txt"), "w")
            self._conv_f.write(f"# trial {tid} started {datetime.datetime.now().isoformat()}\n")
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.discard_btn.setEnabled(True)
        self.delete_btn.setEnabled(True)      # stops and erases the running trial
        # One provisional VLA episode per trial; the recorder keeps it
        # only if the arm moves (vla_link.py has the reasoning).
        self._vla_ended = False
        vla_note = ""
        if want_vla:
            # The episode is written into this trial's folder (next to bag/).
            why = self._vla.begin(self._fallback_task(), self._trial_root)
            vla_note = " + VLA episode" if not why else f" — VLA episode NOT started ({why})"
        # One ROS bag per trial, inside the trial folder.
        self._bag_ended = True
        bag_note = ""
        if want_bag:
            bag_dir = self._bag.begin(self._trial_root)
            if bag_dir:
                self._bag_ended = False
                bag_note = " + bag"
        go = (f"⏺ RECORDING trial {tid}{'' if want_mmhri else ' (VLA only)'}"
              f"{vla_note}{bag_note} — speak/click your command now.")
        if self._bag_ended:
            self.status.setText(go)
        else:
            # 2026-09-28: the bag needs 2-6 s to subscribe to its topics; the
            # operator spoke before it was ready in 5 of 28 trials and trial
            # 085's first arm motion is missing from its bag. "Speak now"
            # waits for the recorder's own "All requested topics are
            # subscribed" line (at most 10 s) — the start-up load spike (late
            # joints, GUI stamps up to 3 s late) is over by then too.
            self.status.setText(f"⏳ trial {tid}: starting the ROS bag — wait for “speak now”…")
            self._bag_go_text = go
            self._bag_wait_t0 = time.monotonic()
            self._bag_log = os.path.join(self._trial_root, "bag_record.log")
            if getattr(self, "_bag_ready_timer", None) is None:
                self._bag_ready_timer = QtCore.QTimer(self)
                self._bag_ready_timer.timeout.connect(self._poll_bag_ready)
            self._bag_ready_timer.start(200)

    def _finish_recorders(self) -> None:
        self._active = False
        try:
            if getattr(self, "_video_timer", None) is not None:
                self._video_timer.stop()
        except Exception:
            pass
        self._latest_bgr = None
        if self._video is not None:
            try:
                self._video.release()
            except Exception:
                pass
            self._video = None
        if self._audio_proc is not None:
            try:
                self._audio_proc.send_signal(signal.SIGINT)
                # WAIT for parecord to actually exit and flush the wav
                # (2026-08-26, Majd: trial.mp4 ended before Morty's
                # answer while audio.wav on disk looked complete —
                # ffmpeg raced a still-flushing wav and -shortest
                # clipped everything to the truncated audio).
                self._audio_proc.wait(timeout=3.0)
            except Exception:
                pass
            self._audio_proc = None
        if self._conv_f is not None:
            try:
                self._conv_f.close()
            except Exception:
                pass
            self._conv_f = None

    def _on_discard(self) -> None:
        self._cancelled = True           # a ⏹ still waiting for Morty must not save it after all
        self._vla.abort()
        self._bag.abort()
        self._bag_ended = True
        self._finish_recorders()
        if self._bridge is not None:
            try:
                self._bridge.trial_stop()
            except Exception:
                pass
        self.status.setText("trial discarded (its folder is still on disk — 🗑 delete removes it).")
        self._reset_buttons()

    # ── delete a trial from the records ──────────────────────────────
    def _trial_records(self, tid: str) -> dict:
        """Everything on disk that belongs to trial `tid`."""
        folders = [p for p in all_trial_dirs()
                   if os.path.basename(p) == f"trial_{tid}" or os.path.basename(p).startswith(f"trial_{tid}_")]
        rows = {}
        for name in ("mmhri_experiments.csv", "mmhri_meta.csv"):
            path = os.path.join(MMHRI_DIR, name)
            if os.path.exists(path):
                with open(path, newline="") as f:
                    rows[name] = sum(1 for r in csv.DictReader(f) if str(r.get("Trial ID", "")).strip() == tid)
        # VLA episodes that point at one of these folders (reported, never deleted here)
        episodes = []
        # Episodes recorded since 2026-09-24 live inside the trial folder and go with it;
        # the older dataset episodes only NAME the trial and are listed for the record.
        for meta in glob.glob(os.path.expanduser("~/morty_ws/vla_experiments/datasets/*/episode_*/meta.json")):
            try:
                with open(meta) as f:
                    hint = str(json.load(f).get("mmhri_trial_hint") or "")
                    if hint and any(hint == os.path.basename(p) or hint == os.path.relpath(p, EXPERIMENTS_DIR) for p in folders):
                        episodes.append(os.path.dirname(meta))
            except (OSError, ValueError):
                continue
        return {"folders": folders, "rows": rows, "episodes": episodes}

    def _on_delete(self) -> None:
        if self._active or (self._stop_pressed and not self._saved and not self._cancelled):
            self._delete_running_trial()
            return
        tid = self.trial_id.text().strip()
        if not tid:
            return
        rec = self._trial_records(tid)
        n_rows = sum(rec["rows"].values())
        if not rec["folders"] and not n_rows:
            self.status.setText(f"nothing recorded for trial {tid} — nothing to delete.")
            return
        lines = [f"Trial {tid} will be erased from the records:"]
        lines += [f"  • folder {os.path.relpath(d, EXPERIMENTS_DIR)}/"
                  + (" (with its VLA episode)" if os.path.exists(os.path.join(d, "steps.npz")) else "")
                  for d in rec["folders"]]
        lines += [f"  • {n} row(s) in {name}" for name, n in rec["rows"].items() if n]
        if rec["episodes"]:
            lines.append("VLA episodes referencing it (kept, delete by hand if wanted):")
            lines += [f"  • {e}" for e in rec["episodes"]]
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Delete trial " + tid)
        box.setText("\n".join(lines))
        box.setInformativeText("This cannot be undone (CSV copies go to backups/).")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        if box.exec_() != QtWidgets.QMessageBox.Yes:
            self.status.setText("delete cancelled.")
            return
        for d in rec["folders"]:
            shutil.rmtree(d if os.path.isabs(d) else os.path.join(EXPERIMENTS_DIR, d), ignore_errors=True)
        bdir = os.path.expanduser("~/morty_ws/backups")
        os.makedirs(bdir, exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        from .trial_times import csv_lock    # 2026-09-28: shared with trial_times
        for name, n in rec["rows"].items():
            if not n:
                continue
            path = os.path.join(MMHRI_DIR, name)
            with csv_lock():
                shutil.copy2(path, os.path.join(bdir, f"{name}.{stamp}.pre_delete_{tid}"))
                with open(path, newline="") as f:
                    rows = list(csv.reader(f))
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(rows[0])
                    w.writerows(r for r in rows[1:] if not (r and r[0].strip() == tid))
        self.status.setText(
            f"🗑 trial {tid} erased: {len(rec['folders'])} folder(s), {n_rows} CSV row(s)"
            + (f"; {len(rec['episodes'])} VLA episode(s) still reference it" if rec["episodes"] else ""))
        self._refresh_progress()
        self._refresh_excel_window()
        self.trial_id.setText(self._next_trial_id())

    def _delete_running_trial(self) -> None:
        """🗑 while a trial records, or while its ⏹ waits for Morty to finish
        talking (Majd 2026-09-25: "delete doesn't work while recording, only
        discard" — discard stops but leaves the folder). Stops everything like
        ✖ discard, cancels a pending save, then erases what the trial wrote.
        No CSV row exists yet; the trial number is free again."""
        tid = self._rec_header.get("tid") or self.trial_id.text().strip()
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Delete running trial " + tid)
        box.setText(f"Stop trial {tid} and erase everything it recorded so far "
                    "(audio, video, bag, VLA images)? Nothing is saved.")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.Cancel)
        if box.exec_() != QtWidgets.QMessageBox.Yes:
            self.status.setText("delete cancelled — the trial goes on.")
            return
        self._cancelled = True           # a pending ⏹ save must not write it back
        trial_dir = self._trial_root or self._trial_dir
        self._on_discard()               # VLA discard, bag stopped + removed, recorders closed
        self._trial_dir = self._trial_root = ""
        self._erase_when_idle(trial_dir, tid, time.monotonic() + 5.0)

    def _erase_when_idle(self, trial_dir: str, tid: str, deadline: float) -> None:
        # The VLA recorder drains its image writer and removes its files when
        # the discard arrives over ROS; the folder goes only after it has
        # stopped, else a late image would bring the folder back.
        st = {}
        if self._bridge is not None:
            try:
                st = self._bridge.vla_status() or {}
            except Exception:
                st = {}
        if st.get("recording") and time.monotonic() < deadline:
            QtCore.QTimer.singleShot(200, lambda: self._erase_when_idle(trial_dir, tid, deadline))
            return
        if trial_dir and os.path.isdir(trial_dir):
            shutil.rmtree(trial_dir, ignore_errors=True)
        if trial_dir and os.path.exists(trial_dir):
            self.status.setText(f"trial {tid} stopped, but its folder could not be removed: {trial_dir}")
        else:
            self.status.setText(f"🗑 trial {tid} stopped and erased — nothing saved; number {tid} is free again.")
        self._refresh_progress()
        self.trial_id.setText(self._next_trial_id())

    def _reset_buttons(self) -> None:
        for w in (self.trial_id, self.layout_edit, self.scenario, self.level):
            w.setEnabled(True)
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.discard_btn.setEnabled(False)
        self.delete_btn.setEnabled(True)
        self.trial_id.setText(self._next_trial_id())
        self.command.clear()
        self.t_interact.clear()
        self.t_total.clear()
        self.notes.clear()
        for cb in self.chips.values():
            cb.setCurrentIndex(0)
        for cb in self.modifier_boxes.values():
            cb.setChecked(False)

    def _modifiers_str(self) -> str:
        """Ticked modifier ids as 'M1;M2' ('' if none) — CSV-safe."""
        return ";".join(
            mid for mid in self.modifier_boxes if self.modifier_boxes[mid].isChecked())

    def _mux_trial_video(self) -> None:
        """camera.mp4 + audio.wav + conversation subtitles → trial.mp4
        (one shareable file; Majd 2026-08-12). Background ffmpeg; the
        separate files stay on disk untouched."""
        d = self._trial_dir
        if not d:
            return
        cam = os.path.join(d, "camera.mp4")
        wav = os.path.join(d, "audio.wav")
        if not (os.path.isfile(cam) and os.path.isfile(wav)):
            return
        srt = os.path.join(d, "captions.srt")
        try:
            with open(srt, "w") as f:
                for i, (t, who, text) in enumerate(self._conv_entries, 1):
                    def _ts(sec: float) -> str:
                        sec = max(0.0, sec)
                        h, rem = divmod(int(sec), 3600)
                        m, s = divmod(rem, 60)
                        ms = int((sec - int(sec)) * 1000)
                        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
                    f.write(f"{i}\n{_ts(t)} --> {_ts(t + 4.0)}\n"
                            f"{who}: {text}\n\n")
        except OSError:
            srt = ""
        cmd = ["ffmpeg", "-y", "-i", cam, "-i", wav]
        if srt:
            cmd += ["-i", srt, "-c:s", "mov_text", "-map", "2:s"]
        # No -shortest (2026-08-26): the LONGEST track defines trial.mp4
        # — a frozen last video frame under continuing audio beats ever
        # clipping Morty's speech again.
        cmd += ["-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "aac",
                os.path.join(d, "trial.mp4")]
        try:
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        except Exception:
            pass

    def _save_log_slices(self) -> None:
        """Copy the part of each component log written during the trial
        into <trial>/logs/ so every experiment is self-contained."""
        if not self._trial_dir:
            return
        out_dir = os.path.join(self._trial_dir, "logs")
        try:
            os.makedirs(out_dir, exist_ok=True)
        except OSError:
            return
        for path, offset in self._log_offsets.items():
            try:
                with open(path, "rb") as f:
                    f.seek(offset)
                    data = f.read()
                if data:
                    with open(os.path.join(
                            out_dir, os.path.basename(path)), "wb") as f:
                        f.write(data)
            except OSError:
                continue

    def _harvest_stage_times(self) -> dict:
        """Pull the bridge's event log, compute the four stage times,
        write events.json into the trial folder, show the summary."""
        times: dict = {}
        if self._bridge is None:
            return times
        try:
            events = self._bridge.trial_stop()
        except Exception:
            return times
        from .scenario_experiments import compute_stage_times, build_timeline
        times = compute_stage_times(events)
        # Full timeline (2026-09-18): pipeline + executor + background
        # records, incl. the GraspGen job that ran for this object BEFORE
        # the trial started; stage times come from the records when present.
        try:
            t_start_wall = float(getattr(self._bridge, "trial_start_wall", time.time()))
            m2w = float(getattr(self._bridge, "trial_mono_to_wall", time.time() - time.monotonic()))
            text, data, stage = build_timeline(events, self._bridge.timing_snapshot(), t_start_wall, time.time(), m2w)
            for k, v in stage.items():
                if v is not None:
                    times[k] = v
            if self._trial_dir:
                with open(os.path.join(self._trial_dir, "timeline.txt"), "w") as f:
                    f.write(text)
                with open(os.path.join(self._trial_dir, "timeline.json"), "w") as f:
                    json.dump(data, f, indent=1, default=str)
        except Exception as exc:
            self.status.setText(f"timeline not written: {exc}")
        if self._trial_dir and events:
            t0 = events[0][0]
            try:
                with open(os.path.join(self._trial_dir, "events.json"), "w") as f:
                    json.dump(
                        {"timings": times,
                         "events": [
                             {"t": round(t - t0, 3), "source": s, "text": x}
                             for t, s, x in events
                         ]}, f, indent=1)
            except OSError:
                pass
        names = [
            ("interaction_s", "command → arm moves"),
            ("intent_s", "understanding"),
            ("choice_s", "object chosen"),
            ("yolo_s", "candidates"),
            ("execution_s", "execution"),
        ]
        parts = [
            f"{label} {times[k]}s" for k, label in names
            if times.get(k) is not None
        ]
        self.stage_lbl.setText(
            " → ".join(parts) if parts
            else "n/a — no robot stages in this trial"
        )
        # The doc's two time columns, measured (not typed).
        if times.get("total_s") is not None:
            self.t_total.setText(str(times["total_s"]))
        # 2026-09-28 (Majd): Interaction Time = command given -> arm starts
        # moving (the doc's definition); the understanding time is intent_s.
        if times.get("interaction_s") is not None:
            self.t_interact.setText(str(times["interaction_s"]))
        return times

    def _on_stop_clicked(self) -> None:
        """⏹ counts once per trial (2026-09-25): a second press started a
        second save chain that re-wrote trial 067 from the reset form 20 s
        later (trial.json "068", a phantom CSV row)."""
        if self._stop_pressed:
            return
        self._stop_pressed = True
        self.stop_btn.setEnabled(False)
        self._on_stop_save()

    def _on_stop_save(self) -> None:
        if self._cancelled:              # 🗑 pressed while ⏹ waited for Morty
            return
        # The VLA episode ends at the PRESS — the speech tail below is
        # Morty talking, not the robot acting. (Re-entered by timers,
        # hence the flag.)
        if not self._vla_ended:
            self._vla_ended = True
            self._vla.end(self._vla_success())
        # The bag also ends at the PRESS: after it only Morty talks.
        if not getattr(self, "_bag_ended", True):
            self._bag_ended = True
            self._bag.end()
        # Wait for Morty to FINISH TALKING before ending the recording
        # (Majd 2026-08-25: trial.mp4 cut the speech mid-sentence). Poll
        # the TTS busy flag; cap the wait at 20 s; add a short tail so
        # the audio device flushes the last syllable.
        import time as _time
        from PyQt5 import QtCore as _QtCore
        speaking = bool(self._bridge is not None and (
            getattr(self._bridge, "tts_active", False)
            or getattr(self._bridge, "voice_chat_state", "") == "responding"))
        deadline = getattr(self, "_stop_speech_deadline", None)
        if speaking and deadline is None:
            self._stop_speech_deadline = _time.time() + 20.0
            self.status.setText("⏳ waiting for Morty to finish speaking…")
            _QtCore.QTimer.singleShot(300, self._on_stop_save)
            return
        if deadline is not None:
            if speaking and _time.time() < deadline:
                _QtCore.QTimer.singleShot(300, self._on_stop_save)
                return
            self._stop_speech_deadline = None
            if not speaking:
                # tail: let the last audio buffer play out
                _QtCore.QTimer.singleShot(
                    800, lambda: self._on_stop_save_final())
                return
        self._on_stop_save_final()

    def _on_stop_save_final(self) -> None:
        if self._saved or self._cancelled:
            return
        self._saved = True
        self._finish_recorders()
        stage_times = self._harvest_stage_times()
        if self._trial_dir:
            self._mux_trial_video()
            self._save_log_slices()
        hdr = self._rec_header
        tid = hdr.get("tid") or self.trial_id.text().strip()
        row = {
            "Trial ID": tid,
            "Task ID": "",
            "Scene Layout": hdr.get("layout") or self.layout_edit.currentText().strip(),
            "Command Used": self.command.text().strip(),
            "Intended Target": self.target.currentText().strip(),
            "Intended Destination": self.destination.currentText().strip(),
            "User Intent understood correctly?": self.chips["intent"].currentText(),
            "Object Detected?": self.chips["detected"].currentText(),
            "Correct Object Selected?": self.chips["correct_obj"].currentText(),
            "Grasp Success?": self.chips["grasp"].currentText(),
            "Action Executed Correctly?": self.chips["action"].currentText(),
            "Full Task Success?": self.chips["full"].currentText(),
            "Interaction Time (s)": self.t_interact.text().strip(),
            "Total Time (s)": self.t_total.text().strip(),
            "Failure Category / Notes": self.notes.text().strip(),
            "Modifiers": self._modifiers_str(),
        }
        meta = {
            "Trial ID": tid,
            "Date": datetime.date.today().isoformat(),
            "Scenario": hdr.get("scenario") or self.scenario.currentText(),
            "Modifiers": self._modifiers_str(),
            "Level": hdr.get("level") or self.level.currentText(),
            "Stage Times": stage_times,
        }
        if not self._trial_dir:
            # VLA-only trial: none of the MMHRI record is written.
            self.status.setText(
                f"✔ trial {tid} ended (no MMHRI evidence) → {self._trial_root}\n"
                "the VLA episode and the bag are in that folder (the vla-rec LED shows whether the episode was kept)")
            self._start_postprocess(self._trial_root)
            self._reset_buttons()
            return
        with open(os.path.join(self._trial_dir, "trial.json"), "w") as f:
            json.dump({**meta, **row, "bag": self._bag.last_dir,
                       "vla_episode": self._vla.last_episode}, f, indent=2)
        self._append_csv(os.path.join(MMHRI_DIR, "mmhri_experiments.csv"),
                         CSV_COLUMNS, [row.get(c, "") for c in CSV_COLUMNS])
        # Stage times as CSV columns too (Majd 2026-08-25: "I need
        # those logs as well") — one row per trial, ready for plotting.
        _st = stage_times or {}
        self._append_csv(os.path.join(MMHRI_DIR, "mmhri_meta.csv"),
                         ["Trial ID", "Date", "Scenario", "Modifiers", "Level",
                          "intent_s", "yolo_s", "graspgen_s",
                          "execution_s", "total_s"],
                         [meta["Trial ID"], meta["Date"], meta["Scenario"],
                          meta["Modifiers"], meta["Level"],
                          _st.get("intent_s", ""), _st.get("yolo_s", ""),
                          _st.get("graspgen_s", ""),
                          _st.get("execution_s", ""),
                          _st.get("total_s", "")])
        self.status.setText(
            f"✔ trial {tid} saved → {self._trial_dir}\n"
            "row appended to mmhri_experiments.csv"
        )
        self._last_saved_trial = self._trial_root
        self._start_postprocess(self._trial_root)
        self._reset_buttons()
        self._refresh_progress()
        self._refresh_excel_window()

    def _refresh_progress(self) -> None:
        """Count saved trials per scenario × layout from trial.json files."""
        counts: Dict[tuple, int] = {}
        try:
            for tdir in all_trial_dirs():
                p = os.path.join(tdir, "trial.json")
                if not os.path.isfile(p):
                    continue
                try:
                    with open(p) as f:
                        t = json.load(f)
                    sc = str(t.get("Scenario", "")).split(":")[0]
                    la = str(t.get("Scene Layout", ""))[-1:]
                    counts[(sc, la)] = counts.get((sc, la), 0) + 1
                except (OSError, ValueError):
                    continue
        except OSError:
            pass
        lines = ["       " + "   ".join(f"L{c[-1]}" for c in LAYOUTS)]
        total = 0
        for s in SCENARIOS:
            sid = s.split(":")[0]
            row = [counts.get((sid, c[-1]), 0) for c in LAYOUTS]
            total += sum(row)
            lines.append(
                f"{sid + chr(58):<5}  " + "  ".join(f"{n}/{REPS_PER_CELL}" for n in row))
        goal = len(SCENARIOS) * len(LAYOUTS) * REPS_PER_CELL
        lines.append(f"total: {total}/{goal}")
        self.progress_lbl.setText("\n".join(lines))

    @staticmethod
    def _append_csv(path: str, header: List[str], row: List[str]) -> None:
        # 2026-09-28: under the lock trial_times takes when it corrects a row.
        from .trial_times import csv_lock
        with csv_lock():
            MMHRIWindow._append_csv_locked(path, header, row)

    @staticmethod
    def _append_csv_locked(path: str, header: List[str], row: List[str]) -> None:
        # Header migration (2026-09-03, Modifiers column): a file written
        # with an OLDER header is rewritten once with the new one and its
        # rows padded — otherwise every new row misaligns against the old
        # header. Only additive changes migrate; a copy goes to backups/.
        if os.path.exists(path):
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
            if rows and rows[0] != header and set(rows[0]) <= set(header):
                bdir = os.path.expanduser("~/morty_ws/backups")
                os.makedirs(bdir, exist_ok=True)
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                shutil.copy2(path, os.path.join(
                    bdir, f"{os.path.basename(path)}.{stamp}.pre_migration"))
                idx = {h: i for i, h in enumerate(rows[0])}
                migrated = [
                    [r[idx[h]] if h in idx and idx[h] < len(r) else ""
                     for h in header]
                    for r in rows[1:]
                ]
                with open(path, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(header)
                    w.writerows(migrated)
        new = not os.path.exists(path)
        with open(path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(header)
            w.writerow(row)

    # ── feeds from the Operator window (called from its poll) ────────
    def feed_camera(self, bgr: np.ndarray) -> None:
        """Store the newest frame; the wall-clock pacer writes it.

        2026-08-25 (Majd: video/audio out of sync, speech chopped):
        writing one video frame PER ARRIVING detector frame at a fixed
        10 fps stamp made camera.mp4 shorter than wall time whenever the
        detector ran below 10 Hz (19.2s of video for 25.9s of audio) —
        and ffmpeg -shortest then chopped the audio. The pacer below
        writes the LATEST frame at a true 10 Hz regardless of arrivals.
        """
        if not self._active or bgr is None:
            return
        self._latest_bgr = bgr.copy()

    def _video_tick(self) -> None:
        bgr = getattr(self, "_latest_bgr", None)
        if not self._active or bgr is None:
            return
        import cv2
        if self._video is None:
            h, w = bgr.shape[:2]
            self._video = cv2.VideoWriter(
                os.path.join(self._trial_dir, "camera.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (w, h),
            )
            self._video_t0 = time.monotonic()
            self._video_frames = 0
        # Wall-clock pacing with catch-up: write as many copies of the
        # newest frame as the elapsed time demands, so duration always
        # matches reality even if this timer ticks late.
        want = int((time.monotonic() - self._video_t0) * 10.0) + 1
        while self._video_frames < want:
            self._video.write(bgr)
            self._video_frames += 1

    def feed_chat(self, who: str, text: str) -> None:
        if not self._active:
            return
        t = time.monotonic() - self._t0
        if self._conv_f is not None:
            self._conv_f.write(f"[{t:7.2f}s] {who}: {text}\n")
            self._conv_f.flush()
        # Kept in memory too → subtitles for trial.mp4 at stop.
        self._conv_entries.append((t, who, text))
        if who == "You" and not self._command_seen:
            self._command_seen = True
            self.command.setText(text)

    def feed_grasps(self, poses_by_kind: Dict) -> None:
        """Latest grasp PoseArrays → overwrite grasps.json + grasps.ply.
        Called on every poses update while recording: the cloud arrives
        BEFORE the grasps, so the old save-once-with-the-cloud approach
        froze empty arrays (trials 1-3, 2026-08-12)."""
        if not self._active or not self._trial_dir:
            return
        if not any(len(v) for v in poses_by_kind.values()):
            return
        try:
            with open(os.path.join(self._trial_dir, "grasps.json"), "w") as f:
                json.dump({k: _poses_to_list(v)
                           for k, v in poses_by_kind.items()}, f, indent=1)
            _write_grasps_ply(
                os.path.join(self._trial_dir, "grasps.ply"), poses_by_kind
            )
            if getattr(self, "_last_cloud", None) is not None:
                pts, cols = self._last_cloud
                write_view3d_html(self._trial_dir, pts, cols, poses_by_kind)
        except OSError:
            pass

    def feed_selection(self) -> None:
        """Object selection published. 2026-09-28 (Majd, trial 095: the field
        showed 9.3 s, the time since ⏺ Start): the Interaction Time field is
        filled only at ⏹, from the stage times (command given -> arm moves);
        the since-Start time is kept internally, never shown or saved."""
        if self._active and self._interaction_s is None:
            self._interaction_s = time.monotonic() - self._t0

    def feed_exec_done(self, rc: Optional[int]) -> None:
        """Executor finished. 2026-09-28: like feed_selection — the Total Time
        field is filled only at ⏹ (first command -> cycle end); the
        since-Start time stays internal."""
        if self._active and self._total_s is None:
            self._total_s = time.monotonic() - self._t0
            if self._conv_f is not None:
                self._conv_f.write(f"[{self._total_s:7.2f}s] executor exit={rc}\n")

    def feed_cloud(self, pts: Optional[np.ndarray],
                   cols: Optional[np.ndarray], poses: Dict) -> None:
        """Final cloud + grasp poses arrived → save the real 3D data."""
        if not self._active or self._cloud_saved or pts is None or not len(pts):
            return
        self._cloud_saved = True
        self._last_cloud = (pts, cols)
        try:
            _write_ply(os.path.join(self._trial_dir, "cloud.ply"), pts, cols)
            with open(os.path.join(self._trial_dir, "grasps.json"), "w") as f:
                json.dump({k: _poses_to_list(v) for k, v in poses.items()}, f,
                          indent=1)
            write_view3d_html(self._trial_dir, pts, cols, poses)
        except OSError as e:
            self.status.setText(f"3D save failed: {e}")

    def feed_view3d_snapshot(self, grab_fn) -> None:
        if self._active and self._trial_dir:
            try:
                grab_fn().save(os.path.join(self._trial_dir, "view3d.png"))
            except Exception:
                pass
