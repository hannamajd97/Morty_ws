"""Split from app.py 2026-08-25 (paper-artifact stage 1): mechanical
move, zero behavior change. Backup: ~/morty_ws/backups/app.py.2026-08-25.pre_split
"""
from __future__ import annotations

import re
import math
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from .config import *  # noqa: F401,F403

class MotionWindow(QtWidgets.QMainWindow):
    """RViz-MotionPlanning-style motion configuration + joint goal panel."""

    move_to_joints = QtCore.pyqtSignal(list)  # radians, UR_JOINTS order

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Morty Motion — planning & gripper")
        self.resize(560, 720)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        lib_box = QtWidgets.QGroupBox("Planning Library (Send robot to / joint goals)")
        lib_form = QtWidgets.QFormLayout(lib_box)
        self.lib_label = QtWidgets.QLabel("Pilz Industrial Motion Planner")
        self.lib_label.setStyleSheet(f"color:{GREEN}; font-weight:bold;")
        lib_form.addRow(self.lib_label)
        self.pipeline_combo = QtWidgets.QComboBox()
        self.pipeline_combo.addItems(list(PLANNER_LIBS.keys()))
        self.pipeline_combo.currentTextChanged.connect(self._on_lib_change)
        lib_form.addRow("Library:", self.pipeline_combo)
        self.planner_combo = QtWidgets.QComboBox()
        self.planner_combo.addItems(PLANNER_LIBS["pilz_industrial_motion_planner"])
        lib_form.addRow("Planner ID:", self.planner_combo)
        layout.addWidget(lib_box)

        opt_box = QtWidgets.QGroupBox("Options")
        opt_form = QtWidgets.QFormLayout(opt_box)
        self.ptime_spin = QtWidgets.QDoubleSpinBox()
        self.ptime_spin.setRange(0.1, 30.0); self.ptime_spin.setSingleStep(0.5); self.ptime_spin.setValue(5.0)
        opt_form.addRow("Planning Time (s):", self.ptime_spin)
        self.attempts_spin = QtWidgets.QSpinBox()
        self.attempts_spin.setRange(1, 50); self.attempts_spin.setValue(10)
        opt_form.addRow("Planning Attempts:", self.attempts_spin)
        self.vel_spin = QtWidgets.QDoubleSpinBox()
        self.vel_spin.setRange(0.01, 1.0); self.vel_spin.setSingleStep(0.05); self.vel_spin.setValue(0.20)
        opt_form.addRow("Velocity Scaling:", self.vel_spin)
        self.acc_spin = QtWidgets.QDoubleSpinBox()
        self.acc_spin.setRange(0.01, 1.0); self.acc_spin.setSingleStep(0.05); self.acc_spin.setValue(0.20)
        opt_form.addRow("Accel. Scaling:", self.acc_spin)
        self.fast_cb = QtWidgets.QCheckBox(
            "⚡ fast replay for Send-robot-to (cached trajectory — BYPASSES the planner)"
        )
        self.fast_cb.setToolTip(
            "Replays a recorded trajectory directly; no planning happens.\n"
            "Planner settings above only matter for joint goals and for the\n"
            "automatic fallback when a pose is not in the cache."
        )
        self.fast_cb.setChecked(True)
        self.fast_cb.toggled.connect(self._on_fast_toggle)
        opt_form.addRow(self.fast_cb)
        self.fast_dur_spin = QtWidgets.QDoubleSpinBox()
        self.fast_dur_spin.setRange(1.0, 10.0)
        self.fast_dur_spin.setSingleStep(0.5)
        self.fast_dur_spin.setValue(3.0)
        self.fast_dur_spin.setSuffix(" s")
        self.fast_dur_spin.setToolTip(
            "The cached trajectory is stretched over this many seconds.\n"
            "Bigger = slower, calmer motion. Applies per GO ⚡ press."
        )
        opt_form.addRow("⚡ move duration:", self.fast_dur_spin)
        layout.addWidget(opt_box)

        mode_box = QtWidgets.QGroupBox("Executor mode (EXECUTE button uses this)")
        mode_lay = QtWidgets.QHBoxLayout(mode_box)
        self.mode_group = QtWidgets.QButtonGroup(self)
        for name, label in [("LIN", "LIN (verified)"), ("PTP", "PTP (try)"), ("PILZ", "PILZ (try)")]:
            rb = QtWidgets.QRadioButton(label)
            rb.setProperty("mode", name)
            if name == "LIN":
                rb.setChecked(True)
            self.mode_group.addButton(rb)
            mode_lay.addWidget(rb)
        layout.addWidget(mode_box)

        # Collision avoidance (2026-08-28): live mirror of the pipeline's
        # neighbour-obstacle knobs. Defaults MUST match
        # grasp_pipeline/selection/params.py (True / 0.005 — padding was
        # halved by the 2026-08-28 tight-box fix) so the GUI never
        # contradicts the running node on startup.
        neigh_box = QtWidgets.QGroupBox("Collision avoidance")
        neigh_box.setToolTip(NEIGH_HELP)
        neigh_form = self._compact_form(neigh_box)
        # Majd 2026-08-28 round 2: "why this too big box when it can be
        # smaller — the Apply button is so far away every time". The long
        # grey explanations moved into tooltips (same words, no vertical
        # space) and Apply now sits directly under the last field.
        self.neighbour_cb = QtWidgets.QCheckBox("Avoid other objects")
        self.neighbour_cb.setToolTip(NEIGH_HELP)
        self.neighbour_cb.setChecked(True)
        neigh_form.addRow(self.neighbour_cb)
        self.neighbour_pad_spin = QtWidgets.QDoubleSpinBox()
        self.neighbour_pad_spin.setRange(0.000, 0.100)
        self.neighbour_pad_spin.setDecimals(3)
        self.neighbour_pad_spin.setSingleStep(0.005)
        self.neighbour_pad_spin.setValue(0.005)
        self.neighbour_pad_spin.setToolTip(
            "Safety cushion added around each neighbour box, per side (m).\n"
            "Bigger = safer, but harder to find a plan in a crowded scene."
        )
        neigh_form.addRow("Padding (m):", self.neighbour_pad_spin)
        # Apply + verify (Majd 2026-08-28: "people like to feel in control,
        # they like to check that they are really applied or not"). The
        # widgets still apply LIVE on change; this button re-sends and then
        # READS THE VALUES BACK off the pipeline node, so the label proves
        # what the running node holds — not what the widget shows.
        self.neigh_status = QtWidgets.QLabel("")
        self.neigh_status.setWordWrap(True)
        self.neigh_status.setStyleSheet(f"color:{GREY}; font-size:10px;")
        self.neigh_apply_btn = self._small_apply_button(
            "Re-send these values to the pipeline node, then read them back\n"
            "to verify the running node really holds them."
        )
        neigh_form.addRow("", self._apply_row(self.neigh_apply_btn,
                                              self.neigh_status))
        self.neighbour_cb.toggled.connect(self._mark_neigh_stale)
        self.neighbour_pad_spin.valueChanged.connect(self._mark_neigh_stale)
        layout.addWidget(neigh_box)

        grip_box = QtWidgets.QGroupBox("Gripper pulse holds")
        grip_box.setToolTip(GRIP_HELP)
        grip_form = self._compact_form(grip_box)
        # Adapter defaults/maxima per gripper_command_adapter source:
        # open 0.5 s (max 1.0), close 1.2 s (max 1.5).
        self.grip_open_spin = QtWidgets.QDoubleSpinBox()
        self.grip_open_spin.setRange(0.1, 1.0); self.grip_open_spin.setSingleStep(0.05); self.grip_open_spin.setValue(0.20)  # 0.20 s open pulse (Majd 2026-09-11 19:20: 0.30/0.50 open too wide; note 0.20 failed to open once at 18:47)
        # dont open very wide crashing with other objects". His measured table puts
        # 0.2-0.3 s at ~13 deg, still wide enough for every object on the table.
        self.grip_open_spin.setToolTip(GRIP_HELP)
        grip_form.addRow("Open hold (s):", self.grip_open_spin)
        self.grip_close_spin = QtWidgets.QDoubleSpinBox()
        self.grip_close_spin.setRange(0.1, 1.5); self.grip_close_spin.setSingleStep(0.05); self.grip_close_spin.setValue(1.00)  # Majd 2026-08-26: default 1.00
        self.grip_close_spin.setToolTip(GRIP_HELP)
        grip_form.addRow("Close hold (s):", self.grip_close_spin)
        # Same Apply + read-back verification as the collision group above.
        self.grip_status = QtWidgets.QLabel("")
        self.grip_status.setWordWrap(True)
        self.grip_status.setStyleSheet(f"color:{GREY}; font-size:10px;")
        self.grip_apply_btn = self._small_apply_button(
            "Re-send the hold times to the gripper adapter, then read them\n"
            "back to verify the running node really holds them."
        )
        grip_form.addRow("", self._apply_row(self.grip_apply_btn,
                                             self.grip_status))
        self.grip_open_spin.valueChanged.connect(self._mark_grip_stale)
        self.grip_close_spin.valueChanged.connect(self._mark_grip_stale)
        layout.addWidget(grip_box)

        # Pick speeds on the grasp executor (Majd 2026-09-16: "add those to
        # the Advanced motion config, with an Apply button"). Apply pushes
        # them onto the running executor and reads them back; they are also
        # passed as launch arguments when the executor next starts.
        speed_box = QtWidgets.QGroupBox("Pick speeds (grasp executor)")
        speed_form = self._compact_form(speed_box)
        qs = QtCore.QSettings("morty", "morty_gui")
        self.speed_vel_spin = QtWidgets.QDoubleSpinBox()
        self.speed_vel_spin.setRange(0.10, 0.50); self.speed_vel_spin.setSingleStep(0.05)
        self.speed_vel_spin.setValue(qs.value("pick_speed/velocity_scaling", 0.40, type=float))
        self.speed_vel_spin.setToolTip("Home → grasp straight line: fraction of the arm's joint speed limits. 0.5 is the ceiling.")
        speed_form.addRow("Approach speed:", self.speed_vel_spin)
        self.speed_acc_spin = QtWidgets.QDoubleSpinBox()
        self.speed_acc_spin.setRange(0.10, 0.50); self.speed_acc_spin.setSingleStep(0.05)
        self.speed_acc_spin.setValue(qs.value("pick_speed/acceleration_scaling", 0.50, type=float))
        self.speed_acc_spin.setToolTip("Home → grasp: fraction of the acceleration limits. Refusals 'joint would turn X rad per cm' come from here.")
        speed_form.addRow("Approach accel.:", self.speed_acc_spin)
        self.speed_transport_combo = QtWidgets.QComboBox()
        self._transport_tiers = [("slow  0.2 — 8.0 s, ~30 % of joint limits", 0.2),
                                 ("normal  0.3 — 5.4 s, 45 %", 0.3),
                                 ("fast  0.4 — 4.0 s, 61 %", 0.4),
                                 ("very fast  0.5 — 3.2 s, 76 % (ceiling)", 0.5)]
        for label, _v in self._transport_tiers:
            self.speed_transport_combo.addItem(label)
        self._select_tier(self.speed_transport_combo, self._transport_tiers,
                          qs.value("pick_speed/blend_velocity_scaling", 0.3, type=float))
        self.speed_transport_combo.setToolTip("Lift → home → drop, one blended arc (blend_velocity_scaling). Above 0.5 the arm protective-stops.")
        speed_form.addRow("Transport (arc):", self.speed_transport_combo)
        self.speed_return_combo = QtWidgets.QComboBox()
        self._return_tiers = [("slow — 4.0 s", 4.0), ("normal — 2.7 s", 2.7), ("fast — 2.0 s", 2.0), ("very fast — 1.6 s", 1.6)]
        for label, _v in self._return_tiers:
            self.speed_return_combo.addItem(label)
        self._select_tier(self.speed_return_combo, self._return_tiers,
                          qs.value("pick_speed/fast_replay_duration_s", 2.7, type=float))
        self.speed_return_combo.setToolTip("Drop → home, and every other named-pose move of the executor (fast_replay_duration_s).")
        speed_form.addRow("Return home:", self.speed_return_combo)
        self.speed_open_spin = QtWidgets.QDoubleSpinBox()
        self.speed_open_spin.setRange(0.3, 2.0); self.speed_open_spin.setSingleStep(0.1)
        self.speed_open_spin.setValue(qs.value("pick_speed/gripper_open_time_s", 1.2, type=float))
        self.speed_open_spin.setToolTip("Time the fingers need to be fully apart after the open pulse (measured 1.0–1.3 s).")
        speed_form.addRow("Gripper open time (s):", self.speed_open_spin)
        self.speed_margin_spin = QtWidgets.QDoubleSpinBox()
        self.speed_margin_spin.setRange(0.0, 1.0); self.speed_margin_spin.setSingleStep(0.1)
        self.speed_margin_spin.setValue(qs.value("pick_speed/open_before_arrival_margin_s", 0.5, type=float))
        self.speed_margin_spin.setToolTip("Extra time before the arm may arrive: open time + margin. Covers the 0.25 s until the pulse really starts.")
        speed_form.addRow("Open margin (s):", self.speed_margin_spin)
        self.speed_status = QtWidgets.QLabel("")
        self.speed_status.setWordWrap(True)
        self.speed_status.setStyleSheet(f"color:{GREY}; font-size:10px;")
        self.speed_apply_btn = self._small_apply_button(
            "Send these speeds to the running grasp executor, then read\n"
            "them back to verify. They are also used when the executor next starts."
        )
        speed_form.addRow("", self._apply_row(self.speed_apply_btn, self.speed_status))
        for _w in (self.speed_vel_spin, self.speed_acc_spin, self.speed_open_spin, self.speed_margin_spin):
            _w.valueChanged.connect(self._mark_speed_stale)
        for _c in (self.speed_transport_combo, self.speed_return_combo):
            _c.currentIndexChanged.connect(self._mark_speed_stale)
        layout.addWidget(speed_box)

        joints_box = QtWidgets.QGroupBox("Group joints of goal state (like RViz Joints tab)")
        joints_lay = QtWidgets.QVBoxLayout(joints_box)
        self._joint_spins: List[QtWidgets.QDoubleSpinBox] = []
        self._joint_sliders: List[QtWidgets.QSlider] = []
        for jname in UR_JOINTS:
            row = QtWidgets.QHBoxLayout()
            lbl = QtWidgets.QLabel(jname.replace("_joint", ""))
            lbl.setMinimumWidth(100)
            row.addWidget(lbl)
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(-360, 360)
            row.addWidget(slider, stretch=1)
            spin = QtWidgets.QDoubleSpinBox()
            spin.setRange(-360.0, 360.0); spin.setSuffix("°"); spin.setDecimals(1)
            row.addWidget(spin)
            slider.valueChanged.connect(lambda v, s=spin: s.setValue(float(v)))
            spin.valueChanged.connect(lambda v, s=slider: s.setValue(int(round(v))))
            self._joint_sliders.append(slider)
            self._joint_spins.append(spin)
            joints_lay.addLayout(row)
        btn_row = QtWidgets.QHBoxLayout()
        self.read_btn = QtWidgets.QPushButton("Read current joints")
        btn_row.addWidget(self.read_btn)
        self.move_joints_btn = QtWidgets.QPushButton("Plan && Execute joint goal")
        self.move_joints_btn.clicked.connect(self._emit_move)
        btn_row.addWidget(self.move_joints_btn)
        joints_lay.addLayout(btn_row)
        layout.addWidget(joints_box)

        # (The "Settings profile" group that used to sit here moved to the
        # MAIN window's "File ▸" menu on 2026-08-28 — "make like file on
        # the main window and put them here". app.py owns those actions,
        # the handlers and the profile read-out now.)

        layout.addStretch(1)
        self.setCentralWidget(central)
        self._on_fast_toggle(self.fast_cb.isChecked())
        # Snapshot the GUI-only defaults straight off the widgets that were
        # just built, so "Reset to defaults" has a single source of truth
        # here too (no second hardcoded copy to drift).
        self._gui_defaults = self.gui_state()
        self._grip_defaults = self.gripper_holds()

    # ── Apply / verify status labels (2026-08-28) ────────────────────
    # Live-apply stays; these only tell the operator whether what he sees
    # has been verified against the node since the last edit.
    STALE_TEXT = "\u2026 unsaved changes \u2014 press Apply to verify"

    @staticmethod
    def _set_status(label: QtWidgets.QLabel, text: str, color: str) -> None:
        label.setText(text)
        label.setStyleSheet(f"color:{color}; font-size:10px;")

    @staticmethod
    def _apply_status_text(kind: str, detail: str):
        """(text, colour) for an Apply outcome. Never claims verified
        unless the values came back off the node."""
        ts = time.strftime("%H:%M:%S")
        if kind == "ok":
            return f"\u2713 applied {ts} \u2014 {detail}", GREEN
        if kind == "sent":
            return f"\u2713 sent {ts}, but read-back failed \u2014 {detail}", ORANGE
        if kind == "mismatch":
            return f"\u26a0 sent {ts}, node reports {detail}", ORANGE
        return f"\u2717 NOT applied \u2014 {detail}", RED

    # ── compact-group helpers (2026-08-28 round 2) ───────────────────
    @staticmethod
    def _compact_form(box: QtWidgets.QGroupBox) -> QtWidgets.QFormLayout:
        """A QFormLayout with the vertical padding squeezed out."""
        form = QtWidgets.QFormLayout(box)
        form.setContentsMargins(8, 4, 8, 4)
        form.setVerticalSpacing(3)
        form.setHorizontalSpacing(6)
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        return form

    @staticmethod
    def _small_apply_button(tooltip: str) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton("Apply")
        btn.setMaximumWidth(90)
        btn.setToolTip(tooltip)
        return btn

    @staticmethod
    def _apply_row(btn: QtWidgets.QPushButton,
                   status: QtWidgets.QLabel) -> QtWidgets.QHBoxLayout:
        """Apply button hard against the fields above it, status to its
        right — one row instead of the old note + button + status stack."""
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        row.addWidget(btn)
        row.addWidget(status, stretch=1)
        return row

    def _mark_neigh_stale(self, *_a) -> None:
        self._set_status(self.neigh_status, self.STALE_TEXT, GREY)

    def _mark_grip_stale(self, *_a) -> None:
        self._set_status(self.grip_status, self.STALE_TEXT, GREY)

    def show_neigh_apply(self, kind: str, detail: str) -> None:
        self._set_status(self.neigh_status, *self._apply_status_text(kind, detail))

    def show_grip_apply(self, kind: str, detail: str) -> None:
        self._set_status(self.grip_status, *self._apply_status_text(kind, detail))

    # ── pick speeds (2026-09-16) ─────────────────────────────────────
    @staticmethod
    def _select_tier(combo: QtWidgets.QComboBox, tiers, value: float) -> None:
        best = min(range(len(tiers)), key=lambda i: abs(tiers[i][1] - float(value)))
        combo.setCurrentIndex(best)

    def _mark_speed_stale(self, *_a) -> None:
        self._set_status(self.speed_status, self.STALE_TEXT, GREY)

    def show_speed_apply(self, kind: str, detail: str) -> None:
        self._set_status(self.speed_status, *self._apply_status_text(kind, detail))

    def pick_speed_values(self) -> Dict[str, float]:
        """The six executor parameters as the tab shows them."""
        return {
            "velocity_scaling": float(self.speed_vel_spin.value()),
            "acceleration_scaling": float(self.speed_acc_spin.value()),
            "blend_velocity_scaling": float(self._transport_tiers[self.speed_transport_combo.currentIndex()][1]),
            "fast_replay_duration_s": float(self._return_tiers[self.speed_return_combo.currentIndex()][1]),
            "gripper_open_time_s": float(self.speed_open_spin.value()),
            "open_before_arrival_margin_s": float(self.speed_margin_spin.value()),
        }

    def pick_speed_launch_args(self) -> List[str]:
        return [f"{k}:={v:.2f}" for k, v in self.pick_speed_values().items()]

    def save_pick_speeds(self) -> None:
        qs = QtCore.QSettings("morty", "morty_gui")
        for k, v in self.pick_speed_values().items():
            qs.setValue(f"pick_speed/{k}", float(v))

    def gui_state(self) -> Dict[str, object]:
        """The Motion tab's pure-GUI tunables (nothing on a node holds
        these). Keys match profiles.MOTION_KEYS."""
        return {
            "executor_mode": self.executor_mode(),
            "fast_replay": bool(self.fast_cb.isChecked()),
            "fast_move_duration_s": float(self.fast_dur_spin.value()),
            "planning_library": self.pipeline_combo.currentText(),
            "planner_id": self.planner_combo.currentText(),
            "planning_time_s": float(self.ptime_spin.value()),
            "planning_attempts": int(self.attempts_spin.value()),
            "velocity_scaling": float(self.vel_spin.value()),
            "acceleration_scaling": float(self.acc_spin.value()),
        }

    def gui_defaults(self) -> Dict[str, object]:
        """The values the widget constructors above ship with — captured at
        construction so there is no second hardcoded copy."""
        return dict(self._gui_defaults)

    def gripper_holds(self) -> Dict[str, float]:
        return {
            "pulse_hold_s_open": float(self.grip_open_spin.value()),
            "pulse_hold_s_close": float(self.grip_close_spin.value()),
        }

    def gripper_defaults(self) -> Dict[str, float]:
        """The hold times this GUI ships with (0.35 / 1.00, Majd
        2026-08-26) — captured off the spinboxes at construction.

        NOTE these are the GUI's defaults, not gripper_command_adapter's
        own declared 0.5 / 1.2: the GUI deliberately overrides the adapter
        and pushes its values on startup, so its own constructors are the
        right source of truth for "put it back the way it ships".
        """
        return dict(self._grip_defaults)

    def set_gui_state(self, values: Dict[str, object]) -> List[str]:
        """Apply GUI-only values onto the widgets. Returns one problem
        string per key that could not be applied; unknown/awkward values
        are reported, never raised."""
        problems: List[str] = []
        if not isinstance(values, dict):
            return ["motion_gui: not a mapping"]
        # Library first: changing it repopulates planner_combo.
        lib = values.get("planning_library")
        if lib is not None:
            if lib in PLANNER_LIBS:
                self.pipeline_combo.setCurrentText(str(lib))
            else:
                problems.append(f"planning_library: unknown library {lib!r}")
        pid = values.get("planner_id")
        if pid is not None:
            if self.planner_combo.findText(str(pid)) >= 0:
                self.planner_combo.setCurrentText(str(pid))
            else:
                problems.append(
                    f"planner_id: {pid!r} is not offered by library "
                    f"'{self.pipeline_combo.currentText()}'")
        mode = values.get("executor_mode")
        if mode is not None:
            hit = False
            for btn in self.mode_group.buttons():
                if str(btn.property("mode")) == str(mode):
                    btn.setChecked(True)
                    hit = True
                    break
            if not hit:
                problems.append(f"executor_mode: unknown mode {mode!r}")
        if "fast_replay" in values:
            try:
                self.fast_cb.setChecked(bool(values["fast_replay"]))
            except Exception as exc:            # pragma: no cover - defensive
                problems.append(f"fast_replay: {exc}")
        for key, spin in (("fast_move_duration_s", self.fast_dur_spin),
                          ("planning_time_s", self.ptime_spin),
                          ("planning_attempts", self.attempts_spin),
                          ("velocity_scaling", self.vel_spin),
                          ("acceleration_scaling", self.acc_spin)):
            if key not in values:
                continue
            raw = values[key]
            try:
                num = float(raw)
            except (TypeError, ValueError):
                problems.append(f"{key}: {raw!r} is not a number")
                continue
            lo, hi = spin.minimum(), spin.maximum()
            if not lo <= num <= hi:
                problems.append(
                    f"{key}: {num:g} outside the allowed {lo:g}..{hi:g} "
                    f"(clamped)")
            spin.setValue(int(round(num)) if isinstance(spin, QtWidgets.QSpinBox)
                          else num)
        return problems

    def set_motion_widgets(self, gripper: Dict[str, object],
                           collision: Dict[str, object]) -> None:
        """Push loaded gripper/collision values onto the widgets WITHOUT
        re-triggering the live-apply signals — the caller has already sent
        them to the nodes and is about to verify by read-back."""
        pairs = (
            (gripper.get("pulse_hold_s_open"), self.grip_open_spin),
            (gripper.get("pulse_hold_s_close"), self.grip_close_spin),
            (collision.get("neighbour_obstacle_padding_m"),
             self.neighbour_pad_spin),
        )
        for val, spin in pairs:
            if val is None:
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                continue
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(num)
            finally:
                spin.blockSignals(blocked)
        enabled = collision.get("neighbour_obstacles_enabled")
        if enabled is not None:
            blocked = self.neighbour_cb.blockSignals(True)
            try:
                self.neighbour_cb.setChecked(bool(enabled))
            finally:
                self.neighbour_cb.blockSignals(blocked)

    def _on_fast_toggle(self, on: bool) -> None:
        # Fast replay bypasses planning entirely (Majd's point) — grey the
        # planner controls out so the UI says so. They still silently back
        # the cache-miss fallback and the joint-goal panel.
        for w in (self.pipeline_combo, self.planner_combo, self.ptime_spin,
                  self.attempts_spin, self.vel_spin, self.acc_spin):
            w.setEnabled(not on)
        self.fast_dur_spin.setEnabled(on)
        self.lib_label.setStyleSheet(
            f"color:{'#888' if on else GREEN}; font-weight:bold;"
        )
        self.lib_label.setText(
            "bypassed by ⚡ fast replay (fallback + joint goals only)" if on
            else ("OMPL" if self.pipeline_combo.currentText() == "ompl"
                  else "Pilz Industrial Motion Planner")
        )

    def _on_lib_change(self, lib: str) -> None:
        self.planner_combo.clear()
        self.planner_combo.addItems(PLANNER_LIBS[lib])
        if lib == "ompl":
            self.lib_label.setText("OMPL")
        else:
            self.lib_label.setText("Pilz Industrial Motion Planner")

    def _emit_move(self) -> None:
        self.move_to_joints.emit(
            [math.radians(s.value()) for s in self._joint_spins]
        )

    def set_joints_deg(self, values_deg: List[float]) -> None:
        for spin, v in zip(self._joint_spins, values_deg):
            spin.setValue(float(v))

    def executor_mode(self) -> str:
        btn = self.mode_group.checkedButton()
        return str(btn.property("mode")) if btn else "LIN"

    def motion_args(self) -> List[str]:
        """Extra -p args for observation_ptp_commander from the settings."""
        return [
            "-p", f"planning_pipeline:={self.pipeline_combo.currentText()}",
            "-p", f"planner_id:={self.planner_combo.currentText()}",
            "-p", f"velocity_scaling:={self.vel_spin.value():.2f}",
            "-p", f"acceleration_scaling:={self.acc_spin.value():.2f}",
            "-p", f"planning_time:={self.ptime_spin.value():.1f}",
            "-p", f"num_attempts:={self.attempts_spin.value()}",
        ]


