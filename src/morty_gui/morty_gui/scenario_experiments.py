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


def build_timeline(events, records, t_start_wall, t_stop_wall, mono_to_wall):
    """The trial's timeline (2026-09-18, Majd: "how fast the system is from
    top to bottom, every tiny thing, background and non-background").

    events:  the bridge's trial events, (monotonic, source, text)
    records: (wall, dict) timing records from /graspgen/timing and
             /grasp_executor_lin_llm/timing, incl. background jobs done
             BEFORE the trial for the object that was then picked.
    Returns (text, data, stage_times)."""
    import datetime as _dt

    def rel(t):
        return t - t_start_wall

    ev = [(mono_to_wall + t, src, txt) for t, src, txt in events]
    recs = [r for _w, r in records]
    lo, hi = t_start_wall - 2.0, t_stop_wall + 2.0

    def first_phase_t(r):
        ph = r.get("phases") or []
        return ph[0]["t"] if ph else None

    exec_cycles = [r for r in recs if r.get("kind") == "executor_cycle"
                   and first_phase_t(r) is not None and lo <= first_phase_t(r) <= hi]
    sel_cycles = [r for r in recs if r.get("kind") == "selection_cycle"
                  and lo <= float(r.get("select") or r.get("published") or 0.0) <= hi]
    labels = {r.get("label") for r in sel_cycles}
    bg = [r for r in recs if r.get("kind") in ("background_job", "background_verdict")
          and r.get("label") in labels
          and float(r.get("job_done") or r.get("verified_at") or 0.0) <= hi]

    items = []  # (wall_t, text)
    for t, src, txt in ev:
        if src == "transcript":
            items.append((t, f'you said: "{txt[:90]}"'))
        elif src == "intent":
            try:
                j = json.loads(txt)
                items.append((t, f"intent: {j.get('action_intent')} {j.get('object_intent') or ''}".strip()))
            except Exception:
                items.append((t, f"intent: {txt[:80]}"))
        elif src == "morty_said":
            items.append((t, f'Morty said: "{txt[:80]}"'))
        elif src == "selection_cycle_id":
            items.append((t, f"selection cycle {txt} started"))
        elif src == "cand_status":
            items.append((t, f"probe: {txt[:100]}"))
        elif src == "cycle_done":
            items.append((t, f"executor reported: {txt[:60]}"))
        elif src == "trial":
            items.append((t, f"trial {txt}"))
    for r in bg:
        if r["kind"] == "background_job":
            items.append((float(r.get("job_done") or 0.0),
                          f"background: GraspGen for '{r.get('label')}' done "
                          f"(job {float(r.get('job_s') or 0):.2f} s, {r.get('n_grasps')} grasps)"))
        else:
            ok = int(r.get("feasible", -1)) >= 0
            items.append((float(r.get("verified_at") or 0.0),
                          f"background: '{r.get('label')}' "
                          + (f"verified for execution via the {r.get('planner')} " if ok else "no feasible grasp ")
                          + f"(executor probe {float(r.get('probe_s') or 0):.2f} s, {r.get('tried')} tried)"))
    for r in sel_cycles:
        sel = r.get("select")
        if r.get("path") == "precomputed":
            pc = r.get("precompute") or {}
            lead = (sel - float(pc.get("job_done") or sel)) if sel else 0.0
            items.append((float(r.get("hit") or sel or 0.0),
                          f"pipeline: '{r.get('label')}' precomputed — GraspGen ran {lead:.1f} s before the click "
                          f"(job {float(pc.get('job_s') or 0):.2f} s), executor verified it via the {pc.get('planner') or '?'} "
                          f"(probe {float(pc.get('probe_s') or 0):.2f} s)"))
        else:
            for key, label in (("detect_done", "detection refreshed"), ("cloud_done", "segmentation + cloud done"),
                               ("graspgen_start", "GraspGen request sent"), ("graspgen_done", "GraspGen answered")):
                if r.get(key):
                    items.append((float(r[key]), f"pipeline: {label}"))
        if r.get("published"):
            dt = (float(r["published"]) - sel) if sel else None
            items.append((float(r["published"]), f"pipeline: {r.get('n_candidates')} candidates published"
                          + (f" ({dt:.2f} s after the click, {r.get('path')})" if dt is not None else "")))
    for r in exec_cycles:
        prev = None
        for ph in r.get("phases") or []:
            name, t = ph["name"], float(ph["t"])
            delta = f" (+{t - prev:.2f} s)" if prev is not None else ""
            items.append((t, f"executor: {name.replace('_', ' ')}{delta}"))
            prev = t
        if r.get("planner"):
            items.append((float((r.get("phases") or [{}])[-1].get("t", 0.0)) + 1e-3,
                          f"executor: cycle {r.get('cycle_id')} {'done' if r.get('ok') else 'FAILED'} via the {r.get('planner')}"))
    items.sort(key=lambda x: x[0])

    lines = [
        f"TRIAL TIMELINE  start {_dt.datetime.fromtimestamp(t_start_wall):%Y-%m-%d %H:%M:%S}  "
        f"stop {_dt.datetime.fromtimestamp(t_stop_wall):%H:%M:%S}  ({t_stop_wall - t_start_wall:.1f} s)",
        "seconds after the trial start; negative = work already done in the background before the trial",
        "",
    ]
    for t, txt in items:
        lines.append(f"t={rel(t):+8.2f}  {txt}")

    # stage times, from the records where they exist, else from the events
    t_cmd = _first(events, "transcript"); t_int = _first(events, "intent", after=t_cmd)
    t_cmd_w = mono_to_wall + t_cmd if t_cmd is not None else None
    t_int_w = mono_to_wall + t_int if t_int is not None else None
    sel = sel_cycles[-1] if sel_cycles else None
    ex = exec_cycles[-1] if exec_cycles else None
    phases = {ph["name"]: float(ph["t"]) for ph in (ex.get("phases") if ex else [])}

    def d(a, b):
        return round(b - a, 3) if (a is not None and b is not None) else None

    t_sel = float(sel["select"]) if sel and sel.get("select") else None
    t_pub = float(sel["published"]) if sel and sel.get("published") else None
    perception = None; graspgen = None; bg_lead = None
    if sel:
        if sel.get("path") == "precomputed":
            perception = d(t_sel, t_pub); graspgen = 0.0
            pc = sel.get("precompute") or {}
            bg_lead = d(float(pc.get("job_done") or 0.0), t_sel) if pc.get("job_done") and t_sel else None
        else:
            perception = d(t_sel, float(sel.get("cloud_done") or 0.0)) if sel.get("cloud_done") else None
            graspgen = d(float(sel.get("graspgen_start") or 0.0), float(sel.get("graspgen_done") or 0.0)) if sel.get("graspgen_done") else None
    end_t = phases.get("cycle_done") or phases.get("home_done") or phases.get("cycle_failed")
    stage = {
        "intent_s": d(t_cmd_w, t_int_w),
        "yolo_s": perception,
        "graspgen_s": graspgen,
        "planning_s": d(phases.get("candidates_received"), phases.get("motion_start")),
        "approach_s": d(phases.get("motion_start"), phases.get("arrival")),
        "grasp_s": d(phases.get("arrival"), phases.get("grasp_verdict")),
        "wait_destination_s": d(phases.get("wait_destination_start"), phases.get("destination_received")),
        "transport_s": d(phases.get("transport_start"), phases.get("transport_done")),
        "execution_s": d(phases.get("motion_start"), end_t),
        "click_to_motion_s": d(t_sel, phases.get("motion_start")),
        "background_lead_s": bg_lead,
        "total_s": d(t_cmd_w if t_cmd_w is not None else t_sel, end_t),
        "planner": (ex or {}).get("planner"),
        "path": (sel or {}).get("path"),
    }
    lines += ["", "SUMMARY"]
    for k, lab in (("intent_s", "speech -> intent"), ("yolo_s", "click -> perception done"),
                   ("graspgen_s", "GraspGen at the click (0 = precomputed)"), ("background_lead_s", "precomputed this long before the click"),
                   ("planning_s", "executor: candidates -> motion start"), ("click_to_motion_s", "click -> motion start"),
                   ("approach_s", "approach"), ("grasp_s", "arrival -> grasp verdict"), ("wait_destination_s", "waiting for the drop side"),
                   ("transport_s", "transport to the drop"), ("execution_s", "motion start -> home"), ("total_s", "total")):
        v = stage.get(k)
        if v is not None:
            lines.append(f"  {lab:42s} {v:7.2f} s")
    if stage.get("planner"):
        lines.append(f"  {'approach planned by':42s} {stage['planner']} ({stage.get('path')})")
    data = {"start_wall": t_start_wall, "stop_wall": t_stop_wall, "stage_times": stage,
            "selection_cycles": sel_cycles, "executor_cycles": exec_cycles, "background": bg,
            "events": [{"t": round(rel(t), 3), "source": s_, "text": x} for t, s_, x in ev]}
    return "\n".join(lines) + "\n", data, stage


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
