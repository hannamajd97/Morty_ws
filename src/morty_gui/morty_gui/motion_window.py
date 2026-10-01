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
            "Planner settings above only matter for joint goals and when no\n"
            "replay node runs. A failed fast move is NOT re-planned automatically:\n"
            "untick ⚡ and press GO to use the planner."
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
            "Seconds for a direct fast move; the named-pose server picks the\n"
            "matching speed tier: 1.6 s = fastest allowed, 2 s fast, 2.7 s normal,\n"
            "4 s slow, above 6 s very slow. A move through a via pose takes\n"
            "proportionally longer. Read when you press GO — no Apply needed."
        )
        opt_form.addRow("⚡ move duration:", self.fast_dur_spin)
        # Apply (Majd 2026-09-23, "still no apply here?"): the seconds field
        # already steers GO ⚡ live; Apply makes the same tier the default
        # for VOICE moves (router parameter default_speed). The executor's
        # own drop/return speed is the "transport" field in Pick speeds.
        self.fast_status = QtWidgets.QLabel("")
        self.fast_status.setWordWrap(True)
        self.fast_status.setStyleSheet(f"color:{GREY}; font-size:10px;")
        self.fast_apply_btn = self._small_apply_button(
            "GO ⚡ already uses the seconds above. Apply also makes that tier the\n"
            "default for voice moves (\"go home\", \"look left\" …) on the running router."
        )
        opt_form.addRow("", self._apply_row(self.fast_apply_btn, self.fast_status))
        hint = QtWidgets.QLabel("Planner fields and the ⚡ seconds are read when you press GO. "
                                "Apply pushes the ⚡ tier to voice moves as well.")
        hint.setStyleSheet("color:#888; font-size:11px;")
        hint.setWordWrap(True)
        opt_form.addRow(hint)
        self.fast_dur_spin.valueChanged.connect(lambda *_: self._set_status(self.fast_status, self.STALE_TEXT, GREY))
        layout.addWidget(opt_box)

        # Executor-mode radios (LIN / PTP / PILZ) removed 2026-09-23 (Majd:
        # "we don't need them anymore"): the PTP and PILZ executors are
        # retired; the LIN executor chooses its own approach chain (curve,
        # blended lines, direct line). executor_mode() stays for profiles.

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
        # 2026-09-28 (Majd): off by default, like the pipeline's
        # neighbour_obstacles_enabled — tick it to route around neighbours.
        self.neighbour_cb.setChecked(False)
        neigh_form.addRow(self.neighbour_cb)
        self.neighbour_pad_spin = QtWidgets.QDoubleSpinBox()
        self.neighbour_pad_spin.setRange(0.000, 0.100)
        self.neighbour_pad_spin.setDecimals(3)
        self.neighbour_pad_spin.setSingleStep(0.005)
        self.neighbour_pad_spin.setValue(0.001)
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
        self.speed_vel_spin.setToolTip("Home → grasp, the straight line to the object.\nHow fast the arm moves along it, as a fraction of the joints' speed limits:\n0.5 = half of the maximum (the ceiling, above it the arm protective-stops), 0.4 = a bit calmer,\n0.2 = slow. Typical approach: 1.3–3 s. A plan shorter than open time + margin is stretched.")
        speed_form.addRow("Approach speed:", self.speed_vel_spin)
        self.speed_acc_spin = QtWidgets.QDoubleSpinBox()
        self.speed_acc_spin.setRange(0.10, 1.00); self.speed_acc_spin.setSingleStep(0.05)
        self.speed_acc_spin.setValue(qs.value("pick_speed/acceleration_scaling", 1.00, type=float))
        self.speed_acc_spin.setToolTip("Home → grasp: how hard the arm may accelerate on that line, fraction of the limits.\nLower it if the log says 'joint would turn X rad per cm — speed limit exceeded';\nthe executor already retries a refused line once at half of this.")
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
        self.speed_transport_combo.setToolTip("Lift → home → drop, one blended arc with the object in the gripper.\nEach tier is a fixed time per segment, measured against the joints' limits on the taught poses:\nslow 8.0 s (~30 %), normal 5.4 s (45 %), fast 4.0 s (61 %), very fast 3.2 s (76 %).\nAbove 0.5 the elbow overspeeds and the robot protective-stops.")
        speed_form.addRow("Transport (arc):", self.speed_transport_combo)
        self.speed_return_combo = QtWidgets.QComboBox()
        self._return_tiers = [("slow — 4.0 s", 4.0), ("normal — 2.7 s", 2.7), ("fast — 2.0 s", 2.0), ("very fast — 1.6 s", 1.6)]
        for label, _v in self._return_tiers:
            self.speed_return_combo.addItem(label)
        self._select_tier(self.speed_return_combo, self._return_tiers,
                          qs.value("pick_speed/fast_replay_duration_s", 2.7, type=float))
        self.speed_return_combo.setToolTip('Drop → home after the release, and any other named-pose move of the executor\n(recovery to home too). One fixed duration per tier.')
        speed_form.addRow("Return home:", self.speed_return_combo)
        self.speed_open_spin = QtWidgets.QDoubleSpinBox()
        self.speed_open_spin.setRange(0.3, 2.0); self.speed_open_spin.setSingleStep(0.1)
        self.speed_open_spin.setValue(qs.value("pick_speed/gripper_open_time_s", 1.0, type=float))
        self.speed_open_spin.setToolTip('How long the fingers need to be fully apart after the open pulse is sent.\nMeasured 2026-09-16: still opening at 1.0 s, open at 1.3 s. The arm may not reach the object\nbefore open time + margin has passed, and the close waits for it as well.')
        speed_form.addRow("Gripper open time (s):", self.speed_open_spin)
        self.speed_margin_spin = QtWidgets.QDoubleSpinBox()
        self.speed_margin_spin.setRange(0.0, 1.0); self.speed_margin_spin.setSingleStep(0.1)
        self.speed_margin_spin.setValue(qs.value("pick_speed/open_before_arrival_margin_s", 0.1, type=float))
        self.speed_margin_spin.setToolTip('Safety on top of the open time. It also covers the ~0.25 s between sending the\nopen and the pulse really starting on the robot. Arrival ≥ open time + margin after the send.')
        speed_form.addRow("Open margin (s):", self.speed_margin_spin)
        self.speed_dur_spin = QtWidgets.QDoubleSpinBox()
        self.speed_dur_spin.setRange(0.5, 6.0); self.speed_dur_spin.setSingleStep(0.1)
        self.speed_dur_spin.setValue(qs.value("pick_speed/approach_duration_s", 2.5, type=float))
        self.speed_dur_spin.setToolTip('One duration for every approach: a plan shorter than this is slowed to it, whatever\n'
                                       'planner won (curve, blended lines, swing) and at whatever scaling. Longer plans stay.\n'
                                       'Keeps the fingers\' opening inside the approach and stops the fast hits on the object.')
        speed_form.addRow("Approach duration (s):", self.speed_dur_spin)
        self.speed_hold_spin = QtWidgets.QDoubleSpinBox()
        self.speed_hold_spin.setRange(0.0, 5.0); self.speed_hold_spin.setSingleStep(0.1)
        self.speed_hold_spin.setValue(qs.value("pick_speed/drop_hold_s", 0.5, type=float))
        self.speed_hold_spin.setToolTip("At the drop pose: how long the arm stands still BEFORE the gripper opens (a settle). Was 5 s.")
        speed_form.addRow("Hold at drop (s):", self.speed_hold_spin)
        self.speed_after_spin = QtWidgets.QDoubleSpinBox()
        self.speed_after_spin.setRange(0.0, 10.0); self.speed_after_spin.setSingleStep(0.5)
        self.speed_after_spin.setValue(qs.value("pick_speed/post_release_wait_s", 2.0, type=float))
        self.speed_after_spin.setToolTip("AFTER the gripper opens: how long the arm stays at the drop pose before going home,\nso you can take the object without the arm swinging at you.")
        speed_form.addRow("Wait after release (s):", self.speed_after_spin)
        self.cb_grasp_sensor = QtWidgets.QCheckBox("Verify the grasp with the finger sensor")
        self.cb_grasp_sensor.setChecked(qs.value("pick_speed/check_grasp_sensor", True, type=bool))
        self.cb_grasp_sensor.setToolTip("On: after the close, pin 16 must say 'object inside', otherwise no-grasp recovery.\n"
                                        "Off: the object is assumed held; no recovery, no 'still holding' refusal, no fingers-apart check.")
        speed_form.addRow("", self.cb_grasp_sensor)
        self.cb_slip = QtWidgets.QCheckBox("Watch for slip after the grasp")
        self.cb_slip.setChecked(qs.value("pick_speed/check_slip", True, type=bool))
        self.cb_slip.setToolTip("On: the slip pin slows the arm while carrying. Off: the slip pin is ignored.")
        speed_form.addRow("", self.cb_slip)
        # Drop poses (2026-09-23): which named poses "drop right/left" mean
        # for the executor. VLA = vla_drop_right/left, which carry their
        # own via in observation_poses.yaml, so the carry goes lift ->
        # via -> drop without passing home (one spline, pre-checked).
        self.drop_poses_combo = QtWidgets.QComboBox()
        self._drop_pose_sets = [("standard — drop_right_side / drop_left_side", ("drop_right_side", "drop_left_side")),
                                ("VLA — vla_drop_right / vla_drop_left", ("vla_drop_right", "vla_drop_left"))]
        for label, _ in self._drop_pose_sets:
            self.drop_poses_combo.addItem(label)
        _saved_right = str(qs.value("pick_speed/drop_right_target", "drop_right_side"))
        self.drop_poses_combo.setCurrentIndex(1 if _saved_right == "vla_drop_right" else 0)
        self.drop_poses_combo.setToolTip("Apply sends the pair to the running executor; it is used from the next pick.")
        speed_form.addRow("Drop poses:", self.drop_poses_combo)
        self.speed_status = QtWidgets.QLabel("")
        self.speed_status.setWordWrap(True)
        self.speed_status.setStyleSheet(f"color:{GREY}; font-size:10px;")
        self.speed_apply_btn = self._small_apply_button(
            "Send these values to the running grasp executor and read them back.\n"
            "Green = the executor confirmed them. They are remembered and used again\n"
            "when the executor next starts, so a restart keeps them."
        )
        speed_form.addRow("", self._apply_row(self.speed_apply_btn, self.speed_status))
        for _w in (self.speed_vel_spin, self.speed_acc_spin, self.speed_transport_combo,
                   self.speed_return_combo, self.speed_open_spin, self.speed_margin_spin, self.speed_dur_spin,
                   self.speed_hold_spin, self.speed_after_spin):
            _lab = speed_form.labelForField(_w)
            if _lab is not None:
                _lab.setToolTip(_w.toolTip())
        for _w in (self.speed_vel_spin, self.speed_acc_spin, self.speed_open_spin, self.speed_margin_spin,
                   self.speed_dur_spin, self.speed_hold_spin, self.speed_after_spin):
            _w.valueChanged.connect(self._mark_speed_stale)
        for _c in (self.speed_transport_combo, self.speed_return_combo):
            _c.currentIndexChanged.connect(self._mark_speed_stale)
        for _cb in (self.cb_grasp_sensor, self.cb_slip):
            _cb.toggled.connect(self._mark_speed_stale)
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

    def show_fast_apply(self, kind: str, detail: str) -> None:
        self._set_status(self.fast_status, *self._apply_status_text(kind, detail))

    @staticmethod
    def speed_tier_for_seconds(seconds: float) -> str:
        """The named-pose server tier a direct move of ``seconds`` maps to."""
        d = float(seconds)
        return ("very_fast" if d <= 1.8 else "fast" if d <= 2.3 else "normal" if d <= 3.3
                else "slow" if d <= 6.0 else "very_slow")

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
            "approach_duration_s": float(self.speed_dur_spin.value()),
            "drop_hold_s": float(self.speed_hold_spin.value()),
            "post_release_wait_s": float(self.speed_after_spin.value()),
            "check_grasp_sensor": bool(self.cb_grasp_sensor.isChecked()),
            "check_slip": bool(self.cb_slip.isChecked()),
            "drop_right_target": self._drop_pose_sets[self.drop_poses_combo.currentIndex()][1][0],
            "drop_left_target": self._drop_pose_sets[self.drop_poses_combo.currentIndex()][1][1],
        }

    def pick_speed_launch_args(self) -> List[str]:
        return [f"{k}:={str(v).lower() if isinstance(v, bool) else (v if isinstance(v, str) else f'{v:.2f}')}"
                for k, v in self.pick_speed_values().items()]

    def save_pick_speeds(self) -> None:
        qs = QtCore.QSettings("morty", "morty_gui")
        for k, v in self.pick_speed_values().items():
            qs.setValue(f"pick_speed/{k}", v if isinstance(v, str) else float(v))

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
        # "executor_mode" in old profiles is accepted and ignored (LIN only).
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
        return "LIN"      # the only executor since 2026-09-23 (radios removed)

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


