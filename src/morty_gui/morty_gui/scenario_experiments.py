"""Scenario-trial recorder for Majd's MMHRI experiments.

Design source: ~/Downloads/"MMHRI Paper Experiments.pdf" + chat spec
(2026-08-20): 7 scenarios x 3 layouts x 5 repetitions. Per trial the
bridge passively records every relevant topic event (transcript, intent,
detections, selection cycle, grasp poses, candidate probes, cycle done,
narrations); on save the four stage times are derived:

    intent_s    accepted transcript  -> /robot/intent
    yolo_s      intent               -> selection_cycle_id
    graspgen_s  selection_cycle_id   -> kept grasp poses
    execution_s kept grasp poses     -> cycle_done
    total_s     accepted transcript  -> cycle_done

plus the PDF's manual judgment columns. Raw events are stored in full so
anything can be recomputed later ("record as much as possible, delete
later" — Majd).

Output: ~/morty_ws/experiments/scenario_logs/
    S<sc>_L<layout>_R<rep>_<stamp>.json   (full trial)
    summary.csv                           (one row per trial)
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime

from PyQt5 import QtWidgets

SCENARIOS = [
    ("S1", "Available objects — no physical action"),
    ("S2", "Object NOT in scene ('give me the book')"),
    ("S3", "Moving without grasping"),
    ("S4", "One of a kind among others — ask for it"),
    ("S5", "Two same kind — clarify by words (left/right/colour)"),
    ("S6", "Two same kind — clarify by pointing"),
    ("S7", "Pointing only (with confirmation)"),
]
# Interaction MODIFIERS (2026-09-03, Majd): the former S8-S10 are not
# scenarios of their own — they are optional behaviours that may or may
# not happen DURING any S1-S7 trial ("I told the robot to grasp, then I
# can use them but it isn't a must"). Recorded per trial as a set, not
# as a row of the 7x3x5 grid. (id, text, former scenario id, integrated?)
MODIFIERS = [
    ("M1", "Compound one-shot — 'pick X and drop it right' (no pauses)", "S8", True),
    ("M2", "Chat while commanding — small talk + impossible request mid-task", "S9", True),
    ("M3", "Change of mind — stop mid-motion, redirect to another object", "S10", False),
]
LAYOUTS = ["A", "B", "C"]
REPS_PER_CELL = 5

LOG_DIR = os.path.expanduser("~/morty_ws/experiments/scenario_logs")

# Manual judgment columns from the PDF's experiments table.
JUDGMENTS = [
    ("understood", "User intent understood?"),
    ("detected", "Object detected (YOLO)?"),
    ("correct_object", "Correct object selected?"),
    ("grasp_success", "Grasp success?"),
    ("action_correct", "Action executed correctly?"),
    ("full_success", "Full task success?"),
]
JUDGMENT_CHOICES = ["-", "yes", "no", "n/a"]


def _first(events, source, after=None):
    for t, src, _txt in events:
        if src == source and (after is None or t > after):
            return t
    return None


def compute_stage_times(events) -> dict:
    """Derive the four PDF stage durations from the raw event list."""
    t_cmd = _first(events, "transcript")
    t_int = _first(events, "intent", after=t_cmd)
    # Fall back: typed/clicked trials have no transcript.
    anchor = t_cmd if t_cmd is not None else _first(events, "trial")
    t_sel = _first(events, "selection_cycle_id", after=t_int or anchor)
    t_pose = _first(events, "grasp_poses", after=t_sel)
    t_done = _first(events, "cycle_done", after=t_pose or t_sel)

    def d(a, b):
        return round(b - a, 3) if (a is not None and b is not None) else None

    return {
        "intent_s": d(t_cmd, t_int),
        "yolo_s": d(t_int, t_sel),
        "graspgen_s": d(t_sel, t_pose),
        "execution_s": d(t_pose, t_done),
        "total_s": d(t_cmd if t_cmd is not None else t_int, t_done),
    }


class ScenarioTrialsWindow(QtWidgets.QMainWindow):
    """7x3x5 trial recorder driven by the bridge's passive event feed."""

    def __init__(self, bridge, parent=None) -> None:
        super().__init__(parent)
        self.bridge = bridge
        self.setWindowTitle("Morty — Scenario trials (7×3×5)")
        os.makedirs(LOG_DIR, exist_ok=True)

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)

        row = QtWidgets.QHBoxLayout()
        self.scenario_box = QtWidgets.QComboBox()
        for sid, txt in SCENARIOS:
            self.scenario_box.addItem(f"{sid}: {txt}")
        self.layout_box = QtWidgets.QComboBox()
        self.layout_box.addItems([f"Layout {c}" for c in LAYOUTS])
        self.rep_spin = QtWidgets.QSpinBox()
        self.rep_spin.setRange(1, 99)
        row.addWidget(self.scenario_box, stretch=2)
        row.addWidget(self.layout_box)
        row.addWidget(QtWidgets.QLabel("rep:"))
        row.addWidget(self.rep_spin)
        v.addLayout(row)

        self.start_btn = QtWidgets.QPushButton("▶ START TRIAL")
        self.start_btn.setStyleSheet(
            "QPushButton{background:#1d7a2c;color:white;font-weight:bold;"
            "font-size:15px;padding:8px;}"
        )
        self.start_btn.clicked.connect(self._on_start_stop)
        v.addWidget(self.start_btn)

        self.state_lbl = QtWidgets.QLabel("idle")
        v.addWidget(self.state_lbl)

        form = QtWidgets.QFormLayout()
        self.judgment_boxes = {}
        for key, label in JUDGMENTS:
            box = QtWidgets.QComboBox()
            box.addItems(JUDGMENT_CHOICES)
            self.judgment_boxes[key] = box
            form.addRow(label, box)
        self.notes_edit = QtWidgets.QLineEdit()
        form.addRow("Failure category / notes", self.notes_edit)
        self.command_edit = QtWidgets.QLineEdit()
        form.addRow("Command used (auto from voice)", self.command_edit)
        self.target_edit = QtWidgets.QLineEdit()
        form.addRow("Intended target", self.target_edit)
        v.addLayout(form)

        self.save_btn = QtWidgets.QPushButton("💾 SAVE TRIAL")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self._on_save)
        v.addWidget(self.save_btn)

        self.progress_lbl = QtWidgets.QLabel("")
        self.progress_lbl.setStyleSheet("font-family:monospace;")
        v.addWidget(self.progress_lbl)

        self.setCentralWidget(central)
        self._events = None
        self._t_started = None
        self._refresh_progress()
        self._suggest_rep()
        self.scenario_box.currentIndexChanged.connect(self._suggest_rep)
        self.layout_box.currentIndexChanged.connect(self._suggest_rep)

    # ── trial lifecycle ──────────────────────────────────────────────
    def _on_start_stop(self) -> None:
        if not self.bridge.trial_active:
            self.bridge.trial_start()
            self._t_started = time.monotonic()
            self._events = None
            self.start_btn.setText("■ END TRIAL")
            self.start_btn.setStyleSheet(
                "QPushButton{background:#a03030;color:white;font-weight:bold;"
                "font-size:15px;padding:8px;}"
            )
            self.state_lbl.setText("RECORDING — run the scenario now.")
            self.save_btn.setEnabled(False)
        else:
            self._events = self.bridge.trial_stop()
            dur = time.monotonic() - (self._t_started or time.monotonic())
            times = compute_stage_times(self._events)
            # Auto-fill the command from the first accepted transcript.
            for _t, src, txt in self._events:
                if src == "transcript":
                    self.command_edit.setText(txt)
                    break
            self.start_btn.setText("▶ START TRIAL")
            self.start_btn.setStyleSheet(
                "QPushButton{background:#1d7a2c;color:white;font-weight:bold;"
                "font-size:15px;padding:8px;}"
            )
            pretty = ", ".join(
                f"{k.replace('_s','')}={v}s" for k, v in times.items()
                if v is not None
            ) or "no stage anchors captured"
            self.state_lbl.setText(
                f"ended after {dur:.1f}s, {len(self._events)} events — "
                f"{pretty}. Fill the judgments and SAVE."
            )
            self.save_btn.setEnabled(True)

    def _on_save(self) -> None:
        if self._events is None:
            return
        sc = SCENARIOS[self.scenario_box.currentIndex()][0]
        layout = LAYOUTS[self.layout_box.currentIndex()]
        rep = int(self.rep_spin.value())
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        times = compute_stage_times(self._events)
        t0 = self._events[0][0] if self._events else 0.0
        trial = {
            "trial_id": f"{sc}_L{layout}_R{rep}",
            "scenario": sc,
            "scenario_text": SCENARIOS[self.scenario_box.currentIndex()][1],
            "layout": layout,
            "rep": rep,
            "wallclock": stamp,
            "command_used": self.command_edit.text(),
            "intended_target": self.target_edit.text(),
            "timings": times,
            "judgments": {
                k: self.judgment_boxes[k].currentText()
                for k, _ in JUDGMENTS
            },
            "notes": self.notes_edit.text(),
            "events": [
                {"t": round(t - t0, 3), "source": src, "text": txt}
                for t, src, txt in self._events
            ],
        }
        path = os.path.join(LOG_DIR, f"{sc}_L{layout}_R{rep}_{stamp}.json")
        with open(path, "w") as f:
            json.dump(trial, f, indent=1)

        csv_path = os.path.join(LOG_DIR, "summary.csv")
        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(
                    ["trial_id", "wallclock", "scenario", "layout", "rep",
                     "command", "target",
                     "intent_s", "yolo_s", "graspgen_s", "execution_s",
                     "total_s"]
                    + [k for k, _ in JUDGMENTS] + ["notes"]
                )
            w.writerow(
                [trial["trial_id"], stamp, sc, layout, rep,
                 trial["command_used"], trial["intended_target"],
                 times["intent_s"], times["yolo_s"], times["graspgen_s"],
                 times["execution_s"], times["total_s"]]
                + [trial["judgments"][k] for k, _ in JUDGMENTS]
                + [trial["notes"]]
            )

        self.state_lbl.setText(f"saved {os.path.basename(path)}")
        self.save_btn.setEnabled(False)
        self._events = None
        # Reset judgments for the next trial.
        for box in self.judgment_boxes.values():
            box.setCurrentIndex(0)
        self.notes_edit.clear()
        self.command_edit.clear()
        self._refresh_progress()
        self._suggest_rep()

    # ── bookkeeping ──────────────────────────────────────────────────
    def _count_done(self, sc: str, layout: str) -> int:
        try:
            names = os.listdir(LOG_DIR)
        except OSError:
            return 0
        return sum(
            1 for n in names
            if n.startswith(f"{sc}_L{layout}_R") and n.endswith(".json")
        )

    def _suggest_rep(self) -> None:
        sc = SCENARIOS[self.scenario_box.currentIndex()][0]
        layout = LAYOUTS[self.layout_box.currentIndex()]
        self.rep_spin.setValue(self._count_done(sc, layout) + 1)

    def _refresh_progress(self) -> None:
        lines = ["      " + "  ".join(f"L{c}" for c in LAYOUTS)]
        total = 0
        for sid, _txt in SCENARIOS:
            counts = [self._count_done(sid, c) for c in LAYOUTS]
            total += sum(counts)
            lines.append(
                f"{sid}:   " + "  ".join(
                    f"{n}/{REPS_PER_CELL}" for n in counts
                )
            )
        goal = len(SCENARIOS) * len(LAYOUTS) * REPS_PER_CELL
        lines.append(f"total: {total}/{goal}")
        self.progress_lbl.setText("\n".join(lines))
