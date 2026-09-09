#!/usr/bin/env python3
"""Morty GUI: Operator window + Motion window + Experiments window.

Window 1 "Morty Operator": system launcher (driver/pipeline with health
LEDs, live camera/RViz toggles), clickable camera view, embedded 3D result
(cloud + kept/rejected/best/ORANGE-executed grasps), EXECUTE/STOP, gripper
open/close, send-robot-to-place.

Window 2 "Morty Motion" (toolbar): RViz-MotionPlanning-style panel —
planning library + planner, velocity/accel/time/attempts, fast-replay
toggle, executor mode (LIN/PTP/PILZ), gripper pulse holds, and a joint-goal
panel with per-joint sliders.

Window 3 "Morty Experiments" (toolbar): experiment recorder, filter-chain
table, U(v) plot, pipeline log tail — all fed from /rosout parsing.
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

import cv2  # noqa: F401  — import FIRST so its bundled Qt path can be scrubbed
# pip-opencv's bundled Qt hijacks the plugin path and breaks system PyQt5
# (root-caused 2026-08-05; same workaround as click_select_viewer).
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)
# pyqtgraph's GL widget needs a GLX context — run through XWayland, never
# the native wayland QPA (same family of lesson as the Open3D viewers).
os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
# Everything the GUI spawns (pipeline!) resolves models/caches via $WS_DIR,
# so they always point inside this workspace.
os.environ.setdefault("WS_DIR", os.path.expanduser("~/morty_ws"))

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from .gl_view import CloudGLView
from .html_viewer import write_view3d_html
from .mmhri_window import MMHRIWindow
from .ros_bridge import start_bridge


from .config import *  # noqa: F401,F403 — split 2026-08-25
from .managed_launch import ManagedLaunch
from .camera_panel import CameraPanel

from .motion_window import MotionWindow
from .experiments_window import ExperimentsWindow
from . import profiles

class OperatorWindow(QtWidgets.QMainWindow):
    def __init__(self, bridge) -> None:
        super().__init__()
        self.bridge = bridge
        self.setWindowTitle("Morty")
        self.resize(1520, 800)
        self.motion_win = MotionWindow()
        self.motion_win.move_to_joints.connect(self._on_move_joints)
        self.motion_win.read_btn.clicked.connect(self._on_read_joints)
        # Hold spinboxes apply LIVE on change (Majd 2026-08-25).
        self.motion_win.grip_open_spin.valueChanged.connect(
            lambda _v: self._push_gripper_holds())
        self.motion_win.grip_close_spin.valueChanged.connect(
            lambda _v: self._push_gripper_holds())
        # Collision-avoidance widgets apply LIVE on the pipeline node,
        # same as the hold spinboxes do on the gripper adapter.
        self.motion_win.neighbour_cb.toggled.connect(
            lambda _v: self._push_neighbour_obstacles())
        self.motion_win.neighbour_pad_spin.valueChanged.connect(
            lambda _v: self._push_neighbour_obstacles())
        # Explicit Apply + read-back verification (Majd 2026-08-28: the
        # live apply is silent, so the operator had no way to check).
        self.motion_win.neigh_apply_btn.clicked.connect(
            self._apply_neighbour_obstacles)
        self.motion_win.grip_apply_btn.clicked.connect(
            self._apply_gripper_holds)
        self.exp_win = ExperimentsWindow(bridge)
        self._exec_proc: Optional[subprocess.Popen] = None
        self._move_proc: Optional[subprocess.Popen] = None
        self._fast_pending: Optional[tuple] = None  # (target, deadline)
        self._last_joints: Dict[str, float] = {}
        self._grasp_ready = False
        self._seen_cycle: Optional[int] = None
        self._manual_execution_pending = False
        self._manual_execution_released = False
        self._software_stop_latched = False
        self._rviz_seen = False
        self._poll_count = 0
        self._ping_proc: Optional[subprocess.Popen] = None
        self._robot_pingable = False
        self._ping_shuttle_proc: Optional[subprocess.Popen] = None
        self._shuttle_pingable = False
        self._program_running: Optional[bool] = None   # pendant Play state, from the LED update

        toolbar = self.addToolBar("main")
        toolbar.setMovable(False)
        # ── File ▸ (Majd 2026-08-28 round 3) ──────────────────────────
        # "make like file on the main window and put them here": the
        # settings-profile actions used to be three buttons at the bottom
        # of the Advanced window's Motion tab. Built with the same
        # QAction-on-the-toolbar pattern as "Advanced ▸"/"Experiments ▸"
        # below, so it is the same toolbar button — only with a menu
        # attached (InstantPopup: one click drops it, no arrow to hit).
        self.file_menu = QtWidgets.QMenu("File", self)
        self.prof_save_action = self.file_menu.addAction("Save settings\u2026")
        self.prof_save_action.setToolTip(
            "Write every tunable (gripper holds, collision, all pipeline "
            "filter/perception parameters, the four voice filters, motion "
            "GUI state) to a .yaml file.")
        self.prof_save_action.triggered.connect(self._profile_save)
        self.prof_load_action = self.file_menu.addAction("Load settings\u2026")
        self.prof_load_action.setToolTip(
            "Read a saved .yaml profile, apply it, then read the values back "
            "and show what actually changed (and what failed).")
        self.prof_load_action.triggered.connect(self._profile_load)
        self.file_menu.addSeparator()
        self.prof_reset_action = self.file_menu.addAction("Reset all to defaults")
        self.prof_reset_action.setToolTip(
            "Put EVERY parameter back to the value declared in "
            "grasp_pipeline/selection/params.py — and the four voice "
            "filters to the values declared in "
            "communication_llm/speech_input_node.py (asks first).")
        self.prof_reset_action.triggered.connect(self._profile_reset)
        self.file_menu.addSeparator()
        # Read-out only (disabled on purpose): which profile is active.
        self.prof_name_action = self.file_menu.addAction("")
        self.prof_name_action.setEnabled(False)
        self._set_profile_name(None)
        self.file_action = QtWidgets.QAction("File \u25b8", self)
        self.file_action.setToolTip(PROFILE_HELP)
        self.file_action.setMenu(self.file_menu)
        toolbar.addAction(self.file_action)
        _file_button = toolbar.widgetForAction(self.file_action)
        if _file_button is not None:
            _file_button.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        # "Reset 3D view" belongs UNDER the 3D view, not next to File
        # (Majd 2026-09-03) — the QAction is created here because
        # view3d.reset_view is wired to it below; its button is added to
        # the 3D panel's bar.
        reset_view = QtWidgets.QAction("Reset 3D view", self)

        # Main window sidebar = the MORTY panel only (Majd, round 2: Motion
        # and Experiments go back into a separate Advanced window).
        self.adv_win = QtWidgets.QMainWindow()
        self.adv_win.setWindowTitle("Morty Advanced — motion · experiments")
        self.adv_tabs = QtWidgets.QTabWidget()
        motion_scroll0 = QtWidgets.QScrollArea()
        motion_scroll0.setWidgetResizable(True)
        motion_scroll0.setWidget(self.motion_win.takeCentralWidget())
        self.adv_tabs.addTab(motion_scroll0, "Motion")
        self.adv_tabs.addTab(self.exp_win.takeCentralWidget(), "Filters")
        # Voice-filter tunables (Majd, 2026-08-11): SEPARATE from the grasp
        # filters. Since 2026-08-28 they apply LIVE: Apply pushes them onto
        # the running speech_input_node (which retunes its listener on the
        # next audio chunk) and then READS THEM BACK off that node. They are
        # STILL passed as launch := overrides at Start voice, so a cold
        # start comes up on the numbers shown here.
        voice_tab = QtWidgets.QWidget()
        vf = QtWidgets.QFormLayout(voice_tab)
        self.vf_min_rms = QtWidgets.QDoubleSpinBox()
        self.vf_min_rms.setDecimals(3)
        self.vf_min_rms.setRange(0.001, 0.200); self.vf_min_rms.setSingleStep(0.005)
        self.vf_min_rms.setValue(0.025)
        self.vf_min_rms.setToolTip(
            "Speech energy gate: buffers quieter than this RMS are ignored as "
            "background noise (−32 dB default). Lower it for quiet mics."
        )
        vf.addRow("Min speech RMS:", self.vf_min_rms)
        self.vf_silence = QtWidgets.QSpinBox()
        self.vf_silence.setRange(100, 3000); self.vf_silence.setSingleStep(50)
        self.vf_silence.setValue(500); self.vf_silence.setSuffix(" ms")
        self.vf_silence.setToolTip("Silence that ends an utterance. Lower = snappier, higher = tolerates pauses.")
        vf.addRow("Silence to finalise:", self.vf_silence)
        self.vf_max_utt = QtWidgets.QDoubleSpinBox()
        self.vf_max_utt.setRange(3.0, 30.0); self.vf_max_utt.setValue(12.0)
        self.vf_max_utt.setSuffix(" s")
        vf.addRow("Max utterance:", self.vf_max_utt)
        self.vf_min_chars = QtWidgets.QSpinBox()
        self.vf_min_chars.setRange(1, 10); self.vf_min_chars.setValue(2)
        self.vf_min_chars.setToolTip("Transcripts shorter than this are dropped as junk.")
        vf.addRow("Min transcript chars:", self.vf_min_chars)
        vf_note = QtWidgets.QLabel(
            "Apply pushes these onto the RUNNING voice node (live, no "
            "restart). They are also used when voice next starts."
        )
        vf_note.setWordWrap(True)
        vf_note.setStyleSheet("color:#888; font-size:10px;")
        vf.addRow("", vf_note)
        # Apply + read-back verification, same compact row and the same
        # honesty rules as the Motion tab's two Apply buttons: green only
        # when the values came BACK off /speech_input_node.
        self.vf_status = QtWidgets.QLabel("")
        self.vf_status.setWordWrap(True)
        self.vf_status.setStyleSheet(f"color:{GREY}; font-size:10px;")
        self.vf_apply_btn = MotionWindow._small_apply_button(
            "Send these values to the running speech_input_node, then read\n"
            "them back to verify the live listener really holds them."
        )
        self.vf_apply_btn.clicked.connect(self._apply_voice_filters)
        vf.addRow("", MotionWindow._apply_row(self.vf_apply_btn,
                                              self.vf_status))
        for _w in (self.vf_min_rms, self.vf_silence,
                   self.vf_max_utt, self.vf_min_chars):
            _w.valueChanged.connect(self._mark_vf_stale)
        self.adv_tabs.addTab(voice_tab, "Voice filters")
        self.adv_win.setCentralWidget(self.adv_tabs)
        self.adv_win.resize(980, 800)
        self.adv_action = QtWidgets.QAction("Advanced ▸", self)
        self.adv_action.setCheckable(True)
        self.adv_action.toggled.connect(
            lambda on: self.adv_win.show() if on else self.adv_win.hide()
        )
        toolbar.addAction(self.adv_action)
        self.mmhri_win = MMHRIWindow(bridge)
        self.mmhri_action = QtWidgets.QAction("Experiments ▸", self)
        self.mmhri_action.setCheckable(True)
        self.mmhri_action.toggled.connect(
            lambda on: self.mmhri_win.show() if on else self.mmhri_win.hide()
        )
        toolbar.addAction(self.mmhri_action)
        # (The separate 7×3×5 window was merged INTO the MMHRI recorder
        # on Majd's request, 2026-08-20 — one window: scenarios, 3-layout
        # dropdown, auto stage times, progress grid, video/sync intact.)

        chat_tab = QtWidgets.QWidget()
        chat_lay = QtWidgets.QVBoxLayout(chat_tab)
        # Read-only provider + exact model display (settings are in Motion).
        self._model_names = read_model_names()
        self.model_label = QtWidgets.QLabel("—")
        self.model_label.setStyleSheet("color:#6cf; font-weight:bold;")
        self.model_label.setWordWrap(True)
        chat_lay.addWidget(self.model_label)
        # Mic strip — the old "Morty mic" window's essentials, embedded.
        mic_row = QtWidgets.QHBoxLayout()
        self.mute_btn = QtWidgets.QPushButton("🎙 live")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setToolTip("Toggle microphone mute (publishes /speech/muted).")
        self.mute_btn.toggled.connect(self._on_mute_toggle)
        self._mute_suppress = False
        mic_row.addWidget(self.mute_btn)
        self.speech_chip = QtWidgets.QLabel("IDLE")
        self.speech_chip.setStyleSheet(
            "background:#333; color:#aaa; padding:3px 10px; border-radius:8px; font-weight:bold;"
        )
        mic_row.addWidget(self.speech_chip)
        self.mic_bar = QtWidgets.QProgressBar()
        self.mic_bar.setRange(0, 100)
        self.mic_bar.setTextVisible(False)
        self.mic_bar.setMaximumHeight(12)
        self.mic_bar.setToolTip("Mic level (/speech/level) — keep system mic gain ~40%.")
        mic_row.addWidget(self.mic_bar, stretch=1)
        self.db_label = QtWidgets.QLabel("−∞ dB")
        self.db_label.setStyleSheet("font-family: monospace; color:#888; min-width:64px;")
        self.db_label.setToolTip(
            "Live mic level in dBFS. Speech typically lands around −30…−10 dB; "
            "below the STT threshold (−32 dB ≈ rms 0.025) nothing is transcribed."
        )
        mic_row.addWidget(self.db_label)
        chat_lay.addLayout(mic_row)
        # Waveform — BIG mirrored envelope like a classic audio wave
        # (Majd's reference picture): symmetric ±|amplitude| filled bars.
        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setBackground("k")
        self.wave_plot.setMinimumHeight(140)
        self.wave_plot.setMaximumHeight(170)
        self.wave_plot.hideAxis("bottom")
        self.wave_plot.hideAxis("left")
        self.wave_plot.setYRange(-1.0, 1.0, padding=0)
        self.wave_plot.setXRange(0, 259, padding=0)  # fixed ring width
        self.wave_plot.setMouseEnabled(x=False, y=False)
        self.wave_plot.hideButtons()
        wave_pen = pg.mkPen((70, 235, 120), width=1)
        wave_brush = (70, 235, 120, 110)
        self.wave_upper = self.wave_plot.plot(
            pen=wave_pen, fillLevel=0.0, brush=wave_brush
        )
        self.wave_lower = self.wave_plot.plot(
            pen=wave_pen, fillLevel=0.0, brush=wave_brush
        )
        chat_lay.addWidget(self.wave_plot)
        # Audio devices stay HERE on the Morty panel (Majd, round 2).
        dev_row = QtWidgets.QHBoxLayout()
        dev_row.addWidget(QtWidgets.QLabel("IN:"))
        self.in_combo = QtWidgets.QComboBox()
        self.in_combo.activated.connect(
            lambda _i: _pactl_set_default("source", self.in_combo.currentText())
        )
        dev_row.addWidget(self.in_combo, stretch=1)
        dev_row.addWidget(QtWidgets.QLabel("OUT:"))
        self.out_combo = QtWidgets.QComboBox()
        self.out_combo.activated.connect(
            lambda _i: _pactl_set_default("sink", self.out_combo.currentText())
        )
        dev_row.addWidget(self.out_combo, stretch=1)
        dev_btn = QtWidgets.QPushButton("⟳")
        dev_btn.setMaximumWidth(30)
        dev_btn.clicked.connect(self._refresh_devices)
        dev_row.addWidget(dev_btn)
        chat_lay.addLayout(dev_row)
        self.chat_log = QtWidgets.QPlainTextEdit()
        self.chat_log.setReadOnly(True)
        self.chat_log.setPlaceholderText(
            "Morty conversation — spoken and typed messages appear here.\n"
            "Start voice in the System bar for speech; typing works either way\n"
            "(the chat node must be running)."
        )
        self.chat_log.setStyleSheet("font-size: 13px;")
        chat_lay.addWidget(self.chat_log, stretch=1)
        self.chat_state_label = QtWidgets.QLabel("state: —")
        self.chat_state_label.setStyleSheet("color:#888;")
        chat_lay.addWidget(self.chat_state_label)
        input_row = QtWidgets.QHBoxLayout()
        self.chat_input = QtWidgets.QLineEdit()
        self.chat_input.setPlaceholderText("type to Morty…  (Enter to send)")
        self.chat_input.returnPressed.connect(self._on_chat_send)
        input_row.addWidget(self.chat_input, stretch=1)
        send_btn = QtWidgets.QPushButton("Send")
        send_btn.clicked.connect(self._on_chat_send)
        input_row.addWidget(send_btn)
        chat_lay.addLayout(input_row)
        # Live pipeline-status lines (Majd: "show some kind of debug") —
        # hearing state and model state, under the talking window.
        self.hear_status = QtWidgets.QLabel("👂 —")
        self.hear_status.setStyleSheet("color:#888;")
        chat_lay.addWidget(self.hear_status)
        self.model_status = QtWidgets.QLabel("🤖 —")
        self.model_status.setStyleSheet("color:#888;")
        chat_lay.addWidget(self.model_status)
        from collections import deque as _deque
        self._level_hist = _deque(maxlen=30)  # ~3 s of levels
        self._recent_morty_texts = _deque(maxlen=8)  # dedupe 🔊 vs 🤖 lines
        self._was_speech = False
        self._processing_until = 0.0
        self.sidebar = chat_tab  # the Morty panel IS the sidebar now
        self.sidebar.setMinimumWidth(380)

        # ── System bar ────────────────────────────────────────────────
        self._driver = ManagedLaunch(self)
        self._pipeline = ManagedLaunch(self)
        self._graspgen = ManagedLaunch(self)   # GraspGen service, lives with the pipeline (2026-09-09)
        self._rviz_extra = ManagedLaunch(self)
        self._camera_extra = ManagedLaunch(self)
        self._replay_extra = ManagedLaunch(self)
        self._voice = ManagedLaunch(self)
        self._executor = ManagedLaunch(self)
        sys_bar = QtWidgets.QWidget()
        sys_lay = QtWidgets.QHBoxLayout(sys_bar)
        sys_lay.setContentsMargins(8, 2, 8, 2)
        sys_lay.addWidget(QtWidgets.QLabel("System:"))
        self.start_all_btn = QtWidgets.QPushButton("▶ Start ALL")
        self.start_all_btn.setToolTip(
            "Starts the driver now and the pipeline automatically as soon as "
            "the robot is up (joint states flowing). One press, whole stack."
        )
        self.start_all_btn.setStyleSheet("font-weight:bold;")
        self.start_all_btn.clicked.connect(self._on_start_all)
        sys_lay.addWidget(self.start_all_btn)
        self._auto_pipeline = False
        self._auto_pipeline_deadline = 0.0
        sys_lay.addSpacing(12)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["real robot", "offline (fake)"])
        sys_lay.addWidget(self.mode_combo)
        self.driver_btn = QtWidgets.QPushButton("Start driver")
        self.driver_btn.clicked.connect(self._on_driver_btn)
        sys_lay.addWidget(self.driver_btn)
        self.cb_camera = QtWidgets.QCheckBox("realsense")
        self.cb_camera.setChecked(True)
        self.cb_camera.toggled.connect(self._on_camera_toggle)
        sys_lay.addWidget(self.cb_camera)
        self.cb_rviz = QtWidgets.QCheckBox("RViz")
        self.cb_rviz.setChecked(False)  # off by default (Majd, 2026-08-11)
        self.cb_rviz.toggled.connect(self._on_rviz_toggle)
        sys_lay.addWidget(self.cb_rviz)
        # Which camera the picture panel shows (Majd 2026-09-08: "see if
        # the [Astra] camera sees clear or not"). Click-to-select stays
        # RealSense-only; MMHRI keeps recording the RealSense stream.
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["view: RealSense", "view: Astra"])
        self.view_combo.setToolTip(
            "Picture panel source. Astra = the fixed camera on the shuttle\n"
            "(relayed JPEG, ~10 Hz) — to check what it sees. Clicking an\n"
            "object works only in the RealSense view.")
        self.view_combo.currentIndexChanged.connect(
            lambda i: self.bridge.set_astra_view(i == 1))
        sys_lay.addWidget(self.view_combo)
        sys_lay.addSpacing(24)
        self.pipeline_btn = QtWidgets.QPushButton("Start pipeline")
        self.pipeline_btn.clicked.connect(self._on_pipeline_btn)
        sys_lay.addWidget(self.pipeline_btn)
        # multiview belongs to the PIPELINE (Majd: "why is it after voice?")
        self.cb_multiview = QtWidgets.QCheckBox("multiview")
        self.cb_multiview.setToolTip(
            "Live toggle: while the pipeline runs, switching applies to the "
            "NEXT object selection (registration_enabled is read per cycle)."
        )
        self.cb_multiview.toggled.connect(self._on_multiview_toggle)
        self._mv_suppress = False
        self._mv_synced = False
        sys_lay.addWidget(self.cb_multiview)
        # ↻ Reset (Majd, 2026-08-25): soft per-cycle reset — aborts the
        # current pick, clears queued/latched selections, detection
        # starts over "like it didn't start yet". No process restarts,
        # no controller/pendant involvement (that's the E-Stop's job).
        # "Reset" alone was unclear (Majd, 2026-08-25) — name what it
        # resets: the current PICK/selection, never the pipeline process.
        self.reset_btn = QtWidgets.QPushButton("↻ Reset pick")
        self.reset_btn.setToolTip(
            "Reset the CURRENT cycle only: abort the running pick, clear "
            "the selected/queued object, and return to 'Waiting for "
            "selection' — without restarting the pipeline or touching "
            "the robot program."
        )
        self.reset_btn.clicked.connect(self._on_reset_cycle)
        sys_lay.addWidget(self.reset_btn)
        sys_lay.addSpacing(24)
        self.voice_btn = QtWidgets.QPushButton("Start voice")
        self.voice_btn.clicked.connect(self._on_voice_btn)
        sys_lay.addWidget(self.voice_btn)
        self.voice_combo = QtWidgets.QComboBox()
        self.voice_combo.addItems(list(VOICE_BASE.keys()))
        self.voice_combo.setToolTip("LLM provider: claude (debug/cheap) or openai (production).")
        self.voice_combo.currentTextChanged.connect(self._update_model_label)
        self._refresh_devices()
        self._update_model_label()
        sys_lay.addWidget(self.voice_combo)
        sys_lay.addStretch(1)
        # Health LEDs. driver=/joint_states, realsense=camera_info topic,
        # astra=shuttle relay frames, rviz=rviz2 process, robot=real UR
        # link, play=pendant program. (astra split out of the single
        # camera LED 2026-09-03.) They sit right-aligned on the FILE row,
        # not the System row (Majd 2026-09-03: "on the same line of file").
        _led_spacer = QtWidgets.QWidget()
        _led_spacer.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        toolbar.addWidget(_led_spacer)
        self.led_driver = QtWidgets.QLabel("● driver")
        self.led_camera = QtWidgets.QLabel("● realsense")
        self.led_astra = QtWidgets.QLabel("● astra")
        self.led_shuttle = QtWidgets.QLabel("● shuttle")
        self.led_rviz = QtWidgets.QLabel("● rviz")
        self.led_robot = QtWidgets.QLabel("● robot")
        self.led_play = QtWidgets.QLabel("● play")
        self.led_voice = QtWidgets.QLabel("● voice")
        for led in (self.led_driver, self.led_camera, self.led_astra,
                    self.led_shuttle, self.led_rviz,
                    self.led_robot, self.led_play, self.led_voice):
            led.setStyleSheet(f"color:{GREY}; font-weight:bold; padding:0 5px;")
            toolbar.addWidget(led)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.camera = CameraPanel()
        self.camera.clicked_px.connect(self._on_camera_click)
        split.addWidget(self.camera)
        self.view3d = CloudGLView()
        split.addWidget(self.view3d)
        split.addWidget(self.sidebar)
        # Equal, square-ish camera and 3D panels (Majd: "quadratisch").
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)
        split.setStretchFactor(2, 0)
        split.setSizes([560, 560, 400])
        reset_view.triggered.connect(self.view3d.reset_view)

        bar = QtWidgets.QWidget()
        bar_lay = QtWidgets.QHBoxLayout(bar)
        bar_lay.setContentsMargins(8, 4, 8, 4)

        # Layout (Majd 2026-09-03): EXECUTE with auto-execute UNDER it;
        # "Move to" row with the grip buttons UNDER it; then the 3D legend
        # with "Reset 3D view" beside it; E-Stop alone at the right end.
        exec_col = QtWidgets.QVBoxLayout()
        exec_col.setSpacing(2)
        self.manual_auto_execute = QtWidgets.QCheckBox("auto-execute")
        self.manual_auto_execute.setChecked(False)
        self.manual_auto_execute.setToolTip(
            "Unchecked: clicking an object prepares its grasp and waits for "
            "EXECUTE. Checked: GUI clicks execute immediately. Voice picks "
            "always execute automatically."
        )
        self.manual_auto_execute.toggled.connect(
            self._on_manual_auto_execute_toggled
        )

        self.exec_btn = QtWidgets.QPushButton("▶ EXECUTE (no grasp)")
        self.exec_btn.setMinimumHeight(40)
        self.exec_btn.setEnabled(False)
        self.exec_btn.setToolTip(
            "Enabled once the pipeline has a grasp ready for the selected "
            "object; disabled again when you select a new object."
        )
        self.exec_btn.setStyleSheet(
            "QPushButton {background:#1d7a2c; color:white; font-weight:bold; font-size:14px;}"
            "QPushButton:disabled {background:#3a4a3d; color:#9a9a9a;}"
        )
        self.exec_btn.clicked.connect(self._on_execute_stop)
        exec_col.addWidget(self.exec_btn)
        exec_col.addWidget(self.manual_auto_execute, alignment=QtCore.Qt.AlignHCenter)
        bar_lay.addLayout(exec_col)

        bar_lay.addSpacing(18)
        move_col = QtWidgets.QVBoxLayout()
        move_col.setSpacing(4)
        move_row = QtWidgets.QHBoxLayout()
        move_row.addWidget(QtWidgets.QLabel("Move to:"))
        self.place_combo = QtWidgets.QComboBox()
        self.place_combo.addItems(NAMED_PLACES)
        move_row.addWidget(self.place_combo)
        self.go_btn = QtWidgets.QPushButton("GO")
        self.go_btn.clicked.connect(self._on_go)
        move_row.addWidget(self.go_btn)
        move_col.addLayout(move_row)
        grip_row = QtWidgets.QHBoxLayout()
        self.grip_open_btn = QtWidgets.QPushButton("Open grip")
        self.grip_open_btn.clicked.connect(lambda: self._on_gripper(True))
        grip_row.addWidget(self.grip_open_btn)
        self.grip_close_btn = QtWidgets.QPushButton("Close grip")
        self.grip_close_btn.clicked.connect(lambda: self._on_gripper(False))
        grip_row.addWidget(self.grip_close_btn)
        move_col.addLayout(grip_row)
        bar_lay.addLayout(move_col)

        # Legend + Reset sit UNDER THE 3D PANE, not glued to the grips
        # (Majd 2026-09-08): one part of the free width before, two after.
        bar_lay.addStretch(1)
        # The checkbox text is painted in the glyph's own colour (same RGB
        # as gl_view.COLORS), so the colour word is redundant.
        for kind, label, col in [("kept", "kept", "#1acc33"),
                                 ("rejected", "rejected", "#d92626"),
                                 ("best", "best", "#2659f2")]:
            cb = QtWidgets.QCheckBox(label)
            cb.setStyleSheet(f"QCheckBox {{color:{col}; font-weight:bold;}}")
            cb.setChecked(True)
            cb.toggled.connect(lambda on, k=kind: self.view3d.set_kind_visible(k, on))
            bar_lay.addWidget(cb)
        # Same widget type as kept/rejected/best, so the four sit on one
        # baseline (Majd 2026-09-08: the label looked off the line) — and
        # the executed glyph can be hidden like the others.
        orange_legend = QtWidgets.QCheckBox("executed pick")
        orange_legend.setStyleSheet(f"QCheckBox {{color:{ORANGE}; font-weight:bold;}}")
        orange_legend.setChecked(True)
        orange_legend.toggled.connect(lambda on: self.view3d.set_kind_visible("exec", on))
        orange_legend.setToolTip(
            "The orange gripper glyph is the grasp the EXECUTOR actually "
            "chose and ran — it stays until you select the next object."
        )
        bar_lay.addWidget(orange_legend)
        # Reset 3D view belongs WITH the 3D legend (Majd 2026-09-03, third
        # placement: not on the File row, not in a row of its own under
        # the view — beside kept/rejected/best/executed).
        bar_lay.addSpacing(10)
        _reset_view_btn = QtWidgets.QToolButton()
        _reset_view_btn.setDefaultAction(reset_view)
        _reset_view_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)   # no icon → show the text
        _reset_view_btn.setToolTip("Re-frame the 3D view from the camera side.")
        bar_lay.addWidget(_reset_view_btn)
        bar_lay.addStretch(2)
        # E-Stop lives at the RIGHT END OF THE BOTTOM BAR — Majd's final
        # call (2026-08-24) after the menu-corner attempt rendered as a
        # clipped sliver twice: "it was down on the right side before".
        self.estop_btn = QtWidgets.QPushButton("🛑 E-Stop")
        self.estop_btn.setMinimumHeight(32)
        self.estop_btn.setToolTip(
            "Immediately stops MoveIt, aborts the grasp executor and fast "
            "replay, cancels controller goals, stops the pipeline, and "
            "requests UR dashboard program stop. This is NOT safety-rated; "
            "use the physical pendant E-stop when a person or equipment "
            "is in danger."
        )
        self.estop_btn.setStyleSheet(
            "QPushButton {background:#a62222; color:white; font-weight:bold; "
            "font-size:13px; padding:4px 14px; border:1px solid #e06b6b; "
            "border-radius:4px;}"
            "QPushButton:hover {background:#c52a2a;}"
        )
        self.estop_btn.clicked.connect(self._on_emergency_stop)
        bar_lay.addWidget(self.estop_btn)

        central = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.addWidget(sys_bar)
        v.addWidget(split, stretch=1)
        # Small docked grasp-probing pane: auto-appears while the executor
        # searches, one line per candidate. Scrollable, text-selectable
        # (Ctrl+C). Re-applied cleanly after the 2026-08-20 rollback.
        self.probe_hist = QtWidgets.QPlainTextEdit()
        self.probe_hist.setReadOnly(True)
        self.probe_hist.setFixedHeight(96)
        self.probe_hist.setMaximumBlockCount(300)
        self.probe_hist.setStyleSheet(
            "QPlainTextEdit{background:#101010;color:#ccc;"
            "font-family:monospace;font-size:11px;}"
        )
        self.probe_hist.setVisible(False)
        v.addWidget(self.probe_hist)
        v.addWidget(bar)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Morty GUI up — waiting for pipeline topics.")
        # Voice readout: what Morty heard / chat state, always visible.
        self.voice_status = QtWidgets.QLabel("🎤 —")
        self.voice_status.setStyleSheet("color:#888; padding-right:8px;")
        self.statusBar().addPermanentWidget(self.voice_status)

        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(100)

    # ── EXECUTE / STOP ───────────────────────────────────────────────
    def _on_camera_click(self, x_px: float, y_px: float) -> None:
        """Route a GUI click with an optional manual execution hold."""
        if self.view_combo.currentIndex() == 1:
            self.statusBar().showMessage(
                "The panel shows the Astra — switch 'view' back to RealSense to click an object.")
            return
        if self._software_stop_latched:
            self.statusBar().showMessage(
                "Software stop is latched — re-arm before selecting a grasp."
            )
            return
        if self.manual_auto_execute.isChecked():
            self._manual_execution_pending = False
            self._manual_execution_released = False
            self.bridge.set_grasp_execution_enabled(True)
            self.bridge.publish_click(x_px, y_px)
            return

        self._manual_execution_pending = True
        self._manual_execution_released = False
        self._grasp_ready = False
        self.bridge.llm_cycle_active = False
        self.bridge.set_grasp_execution_enabled(False)
        self.bridge.publish_click(x_px, y_px)
        self.statusBar().showMessage(
            "Manual object selected — preparing a grasp; execution is paused."
        )

    def _on_manual_auto_execute_toggled(self, enabled: bool) -> None:
        if enabled and self._manual_execution_pending:
            self._manual_execution_pending = False
            self._manual_execution_released = True
            self.bridge.set_grasp_execution_enabled(True)
            self.statusBar().showMessage(
                "Manual auto-execute enabled — prepared grasp released."
            )

    def _on_execute_stop(self) -> None:
        if self._software_stop_latched:
            self.statusBar().showMessage(
                "Software stop is latched — re-arm before executing."
            )
            return
        if self._program_running is False:
            # 2026-09-08: the 14:23 run failed twice with CONTROL_FAILED —
            # the trajectory controller was inactive because the pendant
            # program was stopped. Say so BEFORE the executor is launched.
            self.statusBar().showMessage(
                "Pendant program is STOPPED — Remote mode: the GUI restarts it within seconds; Local mode: press Play on the pendant.")
            return
        if self._exec_proc is None:
            # The persistent executor is running a grasp cycle: the button acts
            # as STOP for that too — freeze in place, don't launch a
            # second executor on top of it (Majd, 2026-08-20).
            if getattr(self.bridge, "llm_cycle_active", False):
                self.bridge.publish_llm_abort()
                # Belt over the abort topic: also stop MoveIt + cancel
                # controller goals directly — 2026-08-24 the abort alone
                # failed to halt a running approach and Majd had to slam
                # the E-stop.
                try:
                    self.bridge.stop_motion()
                except Exception:
                    pass
                self.statusBar().showMessage(
                    "STOP — grasp aborted, motion halted."
                )
                return
            if self._manual_execution_pending:
                if not self._grasp_ready:
                    self.statusBar().showMessage(
                        "Manual grasp is still being prepared — EXECUTE will "
                        "enable when candidates are ready."
                    )
                    return
                self._manual_execution_pending = False
                self._manual_execution_released = True
                # This path runs the grasp inside the PERSISTENT executor,
                # which never pushes hold times of its own (its
                # executor_pulse_hold_s_* default to -1 = "don't touch the
                # adapter"). Without this the adapter keeps its launch
                # defaults and the Advanced spinboxes are decoration
                # (Majd 2026-08-26: "does the gripper work the same as in
                # advanced, 0.40, when I press execute manual button?").
                self._push_gripper_holds()
                self.bridge.set_grasp_execution_enabled(True)
                self.exec_btn.setEnabled(False)
                self.exec_btn.setText("starting manual grasp…")
                self.statusBar().showMessage(
                    "Manual grasp approved — executor starting."
                )
                return
            mode = self.motion_win.executor_mode()
            cmd = list(EXECUTOR_CMDS[mode])
            if mode == "LIN":
                # Manual execution honours the Advanced hold spinboxes
                # (2026-08-26) — appended per press, always current.
                cmd += [
                    f"executor_pulse_hold_s_open:={self.motion_win.grip_open_spin.value():.2f}",
                    f"executor_pulse_hold_s_close:={self.motion_win.grip_close_spin.value():.2f}",
                ]
            self._exec_proc = subprocess.Popen(cmd, start_new_session=True)
            self.bridge.publish_executor_active(True)
            self.exec_btn.setText(f"■ STOP ({mode})")
            self.exec_btn.setStyleSheet(
                f"QPushButton {{background:{RED}; color:white; font-weight:bold; font-size:14px;}}"
            )
            self.statusBar().showMessage(f"Executor launched ({mode}).")
        else:
            try:
                os.killpg(os.getpgid(self._exec_proc.pid), signal.SIGINT)
            except Exception:
                pass
            self.statusBar().showMessage("STOP sent to executor (SIGINT).")

    def _executor_finished(self) -> None:
        rc = self._exec_proc.poll()
        self._exec_proc = None
        self.exp_win.record_event(f"executor_exit={rc}")
        self.mmhri_win.feed_exec_done(rc)
        self.bridge.publish_executor_active(False)
        self.exec_btn.setText("▶ EXECUTE")
        self.exec_btn.setStyleSheet(
            "QPushButton {background:#1d7a2c; color:white; font-weight:bold; font-size:14px;}"
        )
        self.statusBar().showMessage(f"Executor finished (exit {rc}).")

    # ── gripper ──────────────────────────────────────────────────────
    def _push_gripper_holds(self) -> None:
        """Live-apply the Advanced hold spinboxes the moment they change
        (Majd 2026-08-25) — no button press needed."""
        ok = self.bridge.set_gripper_holds(
            self.motion_win.grip_open_spin.value(),
            self.motion_win.grip_close_spin.value())
        self.statusBar().showMessage(
            f"Gripper holds → open {self.motion_win.grip_open_spin.value():.2f}s / "
            f"close {self.motion_win.grip_close_spin.value():.2f}s"
            + ("" if ok else "  ⚠ NOT applied (adapter unreachable)"))

    # ── collision avoidance (pipeline node) ────────────────────
    def _push_neighbour_obstacles(self) -> None:
        """Live-apply the Advanced "Collision avoidance" widgets onto
        /yolo_live_select_grasp (2026-08-28). Reuses the same
        set_pipeline_params helper the Filters tab and the multiview
        toggle use, so a missing pipeline node is a quiet False (the
        GUI is usually started before the pipeline) — reported in the
        status bar the way the gripper holds are."""
        on = self.motion_win.neighbour_cb.isChecked()
        pad = self.motion_win.neighbour_pad_spin.value()
        ok = self.bridge.set_pipeline_params({
            "neighbour_obstacles_enabled": bool(on),
            "neighbour_obstacle_padding_m": float(pad),
        })
        self.statusBar().showMessage(
            f"Neighbour collision boxes → {'ON' if on else 'OFF'}, "
            f"padding {pad:.3f} m"
            + ("" if ok else "  ⚠ NOT applied (pipeline node unreachable)"))

    # ── explicit Apply + verification (2026-08-28) ───────────────────
    @staticmethod
    def _readback_matches(got: Dict[str, object], expected: Dict[str, object]) -> bool:
        """True when every value read off the node equals what was sent."""
        for key, want in expected.items():
            have = got.get(key)
            if have is None:
                return False
            if isinstance(want, bool):
                if bool(have) != bool(want):
                    return False
            elif abs(float(have) - float(want)) > 1e-6:
                return False
        return True

    def _apply_neighbour_obstacles(self) -> None:
        """Apply button for "Collision avoidance": set, then READ BACK off
        /yolo_live_select_grasp and report the node's own values. Bounded
        (~1.1 s worst case) so the Qt thread never hangs."""
        want = {
            "neighbour_obstacles_enabled": bool(self.motion_win.neighbour_cb.isChecked()),
            "neighbour_obstacle_padding_m": float(self.motion_win.neighbour_pad_spin.value()),
        }
        if not self.bridge.set_pipeline_params(want):
            self.motion_win.show_neigh_apply("fail", "pipeline node unreachable")
            self.statusBar().showMessage(
                "⚠ collision settings NOT applied — pipeline node unreachable")
            return
        # Let the set request land before asking for the values back; the
        # two services are independent, so an immediate read could answer
        # from the pre-set state.
        time.sleep(0.1)
        got = self.bridge.get_pipeline_params(list(want))
        if not got or any(got.get(k) is None for k in want):
            self.motion_win.show_neigh_apply("sent", "pipeline node did not answer")
            self.statusBar().showMessage(
                "Collision settings sent, but read-back failed.")
            return
        detail = (f"enabled={bool(got['neighbour_obstacles_enabled'])}, "
                  f"padding={float(got['neighbour_obstacle_padding_m']):.3f}")
        if self._readback_matches(got, want):
            self.motion_win.show_neigh_apply("ok", detail)
            self.statusBar().showMessage(f"Collision settings verified on node — {detail}")
        else:
            self.motion_win.show_neigh_apply("mismatch", detail)
            self.statusBar().showMessage(
                f"⚠ collision settings sent, but node reports {detail}")

    def _mark_vf_stale(self, *_a) -> None:
        MotionWindow._set_status(self.vf_status, MotionWindow.STALE_TEXT, GREY)

    def _show_vf_apply(self, kind: str, detail: str) -> None:
        MotionWindow._set_status(
            self.vf_status, *MotionWindow._apply_status_text(kind, detail))

    def _set_voice_widgets(self, values: Dict[str, object]) -> List[str]:
        """Push loaded/reset voice-filter values onto the four spinboxes
        WITHOUT re-triggering their valueChanged hookup (which would flip
        the group to "stale" and, on every edit, invite a live re-apply).
        Same blockSignals idiom as MotionWindow.set_motion_widgets.

        Returns one warning string per value the spinbox range had to
        clamp — otherwise the widget would quietly disagree with the value
        the node actually holds.
        """
        problems: List[str] = []
        for name, spin in (("whisper_min_rms", self.vf_min_rms),
                           ("whisper_silence_ms", self.vf_silence),
                           ("whisper_max_utterance_s", self.vf_max_utt),
                           ("min_chars", self.vf_min_chars)):
            val = values.get(name)
            if val is None:
                continue
            try:
                num = float(val)
            except (TypeError, ValueError):
                problems.append(f"{name}: {val!r} is not a number — spinbox "
                                f"left as it was")
                continue
            lo, hi = spin.minimum(), spin.maximum()
            if not lo <= num <= hi:
                problems.append(
                    f"{name}: {num:g} is outside the spinbox range "
                    f"{lo:g}..{hi:g} — the box shows the clamped value, the "
                    f"node keeps what it reported")
            blocked = spin.blockSignals(True)
            try:
                spin.setValue(int(round(num))
                              if isinstance(spin, QtWidgets.QSpinBox) else num)
            finally:
                spin.blockSignals(blocked)
        return problems

    def _apply_voice_filters(self) -> None:
        """Apply button for "Voice filters": set on the LIVE speech node,
        then READ BACK off /speech_input_node and report the node's own
        values. Bounded (~1.1 s worst case) so the Qt thread never hangs.

        The node's set-parameters callback retunes the running Whisper
        listener, so this takes effect on the next audio chunk — no voice
        restart. A rejected value comes back as a read-back MISMATCH
        (orange), never as a green "applied"."""
        want = {
            "whisper_min_rms": float(self.vf_min_rms.value()),
            "whisper_silence_ms": int(self.vf_silence.value()),
            "whisper_max_utterance_s": float(self.vf_max_utt.value()),
            "min_chars": int(self.vf_min_chars.value()),
        }
        if not self.bridge.set_voice_filters(want):
            self._show_vf_apply("fail", "speech node unreachable")
            self.statusBar().showMessage(
                "⚠ voice filters NOT applied — speech_input_node param "
                "service unreachable (is the voice stack running?). The "
                "values will still be used at the next Start voice.")
            return
        # Let the set request land before asking for the values back; the
        # two services are independent, so an immediate read could answer
        # from the pre-set state.
        time.sleep(0.1)
        got = self.bridge.get_voice_filters()
        if not got or any(got.get(k) is None for k in want):
            self._show_vf_apply("sent", "speech node did not answer")
            self.statusBar().showMessage(
                "Voice filters sent, but read-back failed.")
            return
        detail = (f"rms={float(got['whisper_min_rms']):.3f}, "
                  f"silence={int(got['whisper_silence_ms'])}ms, "
                  f"max_utt={float(got['whisper_max_utterance_s']):.1f}s, "
                  f"min_chars={int(got['min_chars'])}")
        if self._readback_matches(got, want):
            self._show_vf_apply("ok", detail)
            self.statusBar().showMessage(
                f"Voice filters verified on speech_input_node — {detail}")
        else:
            self._show_vf_apply("mismatch", detail)
            self.statusBar().showMessage(
                f"⚠ voice filters sent, but node reports {detail}")

    def _apply_gripper_holds(self) -> None:
        """Apply button for "Gripper pulse holds": set, then READ BACK off
        /gripper_controller/gripper_command_adapter."""
        want = {
            "pulse_hold_s_open": float(self.motion_win.grip_open_spin.value()),
            "pulse_hold_s_close": float(self.motion_win.grip_close_spin.value()),
        }
        if not self.bridge.set_gripper_holds(
                want["pulse_hold_s_open"], want["pulse_hold_s_close"]):
            self.motion_win.show_grip_apply("fail", "gripper adapter unreachable")
            self.statusBar().showMessage(
                "⚠ hold times NOT applied — gripper adapter param service "
                "unreachable (is the driver running?)")
            return
        time.sleep(0.1)
        got = self.bridge.get_gripper_holds()
        if not got or any(got.get(k) is None for k in want):
            self.motion_win.show_grip_apply("sent", "adapter did not answer")
            self.statusBar().showMessage("Gripper holds sent, but read-back failed.")
            return
        detail = (f"open={float(got['pulse_hold_s_open']):.2f}s, "
                  f"close={float(got['pulse_hold_s_close']):.2f}s")
        if self._readback_matches(got, want):
            self.motion_win.show_grip_apply("ok", detail)
            self.statusBar().showMessage(f"Gripper holds verified on adapter — {detail}")
        else:
            self.motion_win.show_grip_apply("mismatch", detail)
            self.statusBar().showMessage(
                f"⚠ gripper holds sent, but adapter reports {detail}")

    # ── settings profiles (2026-08-28 round 2) ───────────────────────
    # Majd: "make a Save button for the whole program, or Load if I have
    # saved old stuff... and a Reset button but for all the parameters".
    # Node parameters are runtime-only — they die with the pipeline
    # process — so a tuned setup was previously unreproducible.
    #
    # Every action is BOUNDED: the only blocking calls are
    # get_pipeline_params / get_gripper_holds / get_voice_filters, each
    # capped at ~1.0 s by ros_bridge._get_params_sync (0.5 s service wait
    # + 0.5 s reply), and at most six of them run per action (~6.3 s worst
    # case, under a wait cursor — three nodes x before/after read).
    # set_* are fire-and-forget. Nothing spins the executor.
    PROFILE_FILTER = "Morty GUI profiles (*.yaml *.yml);;All files (*)"
    NO_PROFILE_TEXT = "unsaved"

    def _set_profile_name(self, name: Optional[str]) -> None:
        """ONE source of truth for the active-profile string, shown in the
        File menu's disabled read-out (was the Motion tab's prof_label)."""
        self.prof_name_action.setText(f"Profile: {name or self.NO_PROFILE_TEXT}")
        self.prof_name_action.setToolTip(
            name or "No profile loaded or saved yet.")

    def _profile_status(self, kind: str, detail: str) -> None:
        """Profile feedback on the MAIN window's status bar (the Motion
        tab's prof_status label went away with the group box). Same
        honesty vocabulary as the Apply buttons: ✓ only when the values
        came back off the nodes, ⚠ sent-but-unverified, ✗ not applied."""
        mark = {"ok": "\u2713", "warn": "\u26a0"}.get(kind, "\u2717")
        self.statusBar().showMessage(
            f"{mark} {time.strftime('%H:%M:%S')} \u2014 {detail}")

    @staticmethod
    def _value_equal(a, b) -> bool:
        """Read-back comparison that also survives string parameters
        (up_vector_source), which _readback_matches was never asked to
        handle — it stays as it is for the two Apply buttons."""
        if isinstance(a, bool) or isinstance(b, bool):
            return bool(a) == bool(b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return abs(float(a) - float(b)) <= 1e-6
        return a == b

    @staticmethod
    def _coerce_like(value, reference):
        """Match the type the node currently reports.

        ROS SetParameters rejects an INTEGER for a DOUBLE parameter, and a
        hand-edited profile writing `padding: 0` instead of `0.0` is the
        easy way to hit that. Returns (value, problem-or-None).
        """
        if reference is None:
            return value, None
        try:
            if isinstance(reference, bool):
                if isinstance(value, bool):
                    return value, None
                if isinstance(value, str):
                    return value.strip().lower() in ("true", "1", "yes", "on"), None
                return bool(value), None
            if isinstance(value, bool) and not isinstance(reference, bool):
                return value, None
            if isinstance(reference, int):
                return int(value), None
            if isinstance(reference, float):
                return float(value), None
            if isinstance(reference, str):
                return str(value), None
        except (TypeError, ValueError):
            return value, (f"could not read {value!r} as "
                           f"{type(reference).__name__} — sent as is")
        return value, None

    def _profile_snapshot(self):
        """Everything a profile carries, read from the LIVE nodes where a
        node owns the value. Unreadable values come back as None (written
        as YAML null) with a note saying so — never a silent widget echo.
        Bounded: three ~1 s reads (pipeline, gripper adapter, speech
        node)."""
        notes: List[str] = []
        pipe = self.bridge.get_pipeline_params(list(ALL_TUNABLE_PARAMS)) or {}
        if not pipe:
            notes.append(
                "pipeline node did not answer — every filter/perception "
                "value was written as null")
        else:
            unread = [n for n in ALL_TUNABLE_PARAMS if pipe.get(n) is None]
            if unread:
                notes.append(
                    f"{len(unread)} pipeline parameter(s) unreadable, written "
                    f"as null: {', '.join(unread)}")
        filters = {n: pipe.get(n) for n in FILTER_PARAMS}
        perception = {n: pipe.get(n) for n in PERCEPTION_PARAMS}
        collision = {n: pipe.get(n) for n in profiles.COLLISION_KEYS}
        if any(v is None for v in collision.values()):
            collision = {
                "neighbour_obstacles_enabled":
                    bool(self.motion_win.neighbour_cb.isChecked()),
                "neighbour_obstacle_padding_m":
                    float(self.motion_win.neighbour_pad_spin.value()),
            }
            notes.append(
                "collision values taken from the GUI widgets (the pipeline "
                "node did not report them)")
        holds = self.bridge.get_gripper_holds() or {}
        if all(holds.get(k) is not None for k in profiles.GRIPPER_KEYS):
            gripper = {k: float(holds[k]) for k in profiles.GRIPPER_KEYS}
        else:
            gripper = {
                "pulse_hold_s_open": float(self.motion_win.grip_open_spin.value()),
                "pulse_hold_s_close": float(self.motion_win.grip_close_spin.value()),
            }
            notes.append(
                "gripper adapter did not answer — hold times taken from the "
                "GUI widgets")
        # Voice filters: the SAME honesty rule as the pipeline read above —
        # the live speech node is asked, and anything it does not report is
        # written as null with a note, never echoed from the spinboxes (the
        # widgets can hold an un-applied edit, or the voice stack may be
        # down entirely).
        spoke = self.bridge.get_voice_filters() or {}
        if not spoke:
            notes.append(
                "speech node did not answer — every voice filter value was "
                "written as null")
        else:
            unread_v = [n for n in profiles.VOICE_KEYS if spoke.get(n) is None]
            if unread_v:
                notes.append(
                    f"{len(unread_v)} voice filter(s) unreadable, written as "
                    f"null: {', '.join(unread_v)}")
        voice = {n: spoke.get(n) for n in profiles.VOICE_KEYS}
        return (gripper, collision, filters, perception,
                self.motion_win.gui_state(), voice, notes)

    def _profile_save(self) -> None:
        directory = profiles.ensure_profile_dir()
        start = os.path.join(directory, profiles.default_filename())
        path, _sel = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save settings profile", start, self.PROFILE_FILTER)
        if not path:
            return
        if not os.path.splitext(path)[1]:
            path += ".yaml"
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            gripper, collision, filters, perception, gui, voice, notes = \
                self._profile_snapshot()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        doc = profiles.build_profile(gripper, collision, filters, perception,
                                     gui, voice=voice, notes=notes)
        try:
            text = profiles.dump_profile(doc)
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        except (OSError, ValueError) as exc:
            self._profile_status(
                "fail", f"profile NOT saved — could not write "
                        f"{os.path.basename(path)}: {exc}")
            return
        name = os.path.basename(path)
        self._set_profile_name(name)
        n_null = sum(1 for v in list(filters.values())
                     + list(perception.values()) + list(voice.values())
                     if v is None)
        if notes:
            self._profile_status(
                "warn", f"saved {path}, but {n_null} value(s) could not be "
                        f"read off a node (written as null)")
        else:
            self._profile_status(
                "ok", f"saved {path} ({len(filters) + len(perception)} "
                      f"pipeline params + gripper + motion GUI + "
                      f"{len(voice)} voice filters)")
        self.exp_win.record_event(f"profile_saved={name}")

    def _profile_load(self) -> None:
        directory = profiles.ensure_profile_dir()
        path, _sel = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load settings profile", directory,
            self.PROFILE_FILTER)
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                doc = profiles.parse_profile(fh.read())
        except (OSError, ValueError, UnicodeDecodeError) as exc:
            self._profile_status(
                "fail", f"profile NOT loaded — {os.path.basename(path)}: {exc}")
            return
        sections, warnings = profiles.split_profile(doc)
        pipe_want, merge_warnings = profiles.pipeline_values(sections)
        warnings = list(warnings) + list(merge_warnings)
        name = os.path.basename(path)
        # A profile written before the "voice" section existed simply has
        # an empty voice dict here — split_profile does not warn about a
        # missing section, and _profile_apply skips the speech node
        # entirely when the dict is empty, leaving the voice settings
        # exactly as they were.
        self._profile_apply(
            f"Loaded profile — {name}", pipe_want,
            sections[profiles.SECTION_GRIPPER],
            sections[profiles.SECTION_MOTION],
            sections[profiles.SECTION_VOICE], warnings, name)
        self.exp_win.record_event(f"profile_loaded={name}")

    def _profile_reset(self) -> None:
        """Restore the CODE defaults. They are parsed out of
        grasp_pipeline/selection/params.py at runtime (see
        profiles.pipeline_defaults) and — for the four voice filters —
        out of communication_llm/speech_input_node.py (see
        profiles.voice_defaults), so this button follows the nodes' own
        declarations instead of a second copy that could drift."""
        defaults, missing, note = profiles.tunable_defaults()
        if not defaults:
            self._profile_status("fail", f"reset unavailable — {note}")
            QtWidgets.QMessageBox.warning(
                self, "Reset to defaults",
                "Could not read the code defaults:\n\n" + note)
            return
        warnings = [f"defaults source: {note}"]
        if missing:
            warnings.append(
                f"no declare_parameter() default found for "
                f"{len(missing)} parameter(s), left untouched: "
                f"{', '.join(missing)}")
        grip_want = self.motion_win.gripper_defaults()
        gui_want = self.motion_win.gui_defaults()
        # Voice filters: their defaults come from speech_input_node.py's own
        # declare_parameter() calls, parsed the same way (profiles.
        # voice_defaults). The GUI's vf_* spinbox startup values are only a
        # mirror of those declarations, so the node file — not the widgets —
        # is the source of truth and there is no second hardcoded copy. If
        # that file cannot be read we warn and leave the voice filters
        # ALONE, rather than push a guess.
        voice_want, voice_missing, voice_note = profiles.voice_defaults()
        if voice_want:
            warnings.append(f"voice defaults source: {voice_note}")
        else:
            warnings.append(
                f"voice filters left untouched — {voice_note}")
        if voice_missing and voice_want:
            warnings.append(
                f"no declare_parameter() default found for "
                f"{len(voice_missing)} voice filter(s), left untouched: "
                f"{', '.join(voice_missing)}")
        voice_line = (
            f"reset the {len(voice_want)} voice filter(s) to the values "
            f"declared in\ncommunication_llm/speech_input_node.py, "
            if voice_want else "")
        answer = QtWidgets.QMessageBox.question(
            self, "Reset to defaults",
            f"Put ALL {len(defaults)} pipeline parameters back to the values "
            f"declared in\ngrasp_pipeline/selection/params.py, reset the "
            f"gripper pulse holds\nto {grip_want['pulse_hold_s_open']:.2f} s / "
            f"{grip_want['pulse_hold_s_close']:.2f} s, {voice_line}and reset "
            f"the motion GUI settings?\n\nThis changes the RUNNING pipeline "
            f"(and, for the voice filters, the running speech node) "
            f"immediately.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            self._profile_status("warn", "reset cancelled")
            return
        self._profile_apply("Reset to code defaults", dict(defaults),
                            dict(grip_want), dict(gui_want), dict(voice_want),
                            warnings, "code defaults")
        self.exp_win.record_event("profile_reset_to_defaults")

    def _profile_apply(self, title: str, pipe_want: Dict, grip_want: Dict,
                       gui_want: Dict, voice_want: Dict,
                       warnings: List[str], profile_name: str) -> None:
        """Apply, read back, and show an honest "old -> new" summary.

        Never trusts the request: every pipeline/gripper/voice line in the
        summary shows the value the NODE reported afterwards, and anything
        that did not take becomes a FAILED line.

        ``voice_want`` empty (an old profile with no "voice" section, or a
        Reset that could not read the speech node's declarations) means the
        speech node is not touched at all — no request, no warning.
        """
        lines: List[str] = []
        failures: List[str] = []
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            # GUI-only state: no node involved, so it is always verifiable.
            gui_before = self.motion_win.gui_state()
            gui_problems = (self.motion_win.set_gui_state(gui_want)
                            if gui_want else [])
            gui_after = self.motion_win.gui_state()
            # Pipeline parameters.
            pipe_before = (self.bridge.get_pipeline_params(list(pipe_want))
                           if pipe_want else {})
            if pipe_before:
                for name in list(pipe_want):
                    fixed, problem = self._coerce_like(pipe_want[name],
                                                       pipe_before.get(name))
                    pipe_want[name] = fixed
                    if problem:
                        warnings.append(f"{name}: {problem}")
            pipe_after = None
            pipe_sent = bool(pipe_want) and self.bridge.set_pipeline_params(pipe_want)
            if pipe_sent:
                time.sleep(0.1)
                pipe_after = self.bridge.get_pipeline_params(list(pipe_want))
            # Gripper holds.
            grip_before = (self.bridge.get_gripper_holds() if grip_want else {})
            grip_after = None
            grip_sent = False
            if grip_want:
                grip_sent = self.bridge.set_gripper_holds(
                    float(grip_want.get(
                        "pulse_hold_s_open",
                        self.motion_win.grip_open_spin.value())),
                    float(grip_want.get(
                        "pulse_hold_s_close",
                        self.motion_win.grip_close_spin.value())))
                if grip_sent:
                    time.sleep(0.1)
                    grip_after = self.bridge.get_gripper_holds()
            # Voice filters on /speech_input_node (a different node from
            # the pipeline — it can be down on its own).
            voice_before = (self.bridge.get_voice_filters()
                            if voice_want else {})
            if voice_before:
                for name in list(voice_want):
                    fixed, problem = self._coerce_like(voice_want[name],
                                                       voice_before.get(name))
                    voice_want[name] = fixed
                    if problem:
                        warnings.append(f"{name}: {problem}")
            voice_after = None
            voice_sent = bool(voice_want) and self.bridge.set_voice_filters(
                voice_want)
            if voice_sent:
                time.sleep(0.1)
                voice_after = self.bridge.get_voice_filters()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        # Mirror the voice values onto the four spinboxes BEFORE the summary
        # is built, so a value the widget has to clamp is reported as a
        # warning instead of silently disagreeing with the node.
        if voice_want:
            warnings += self._set_voice_widgets(voice_after or voice_want)
        if gui_want:
            lines.append("── motion GUI (no node holds these) ──")
            for key in sorted(gui_want):
                old, new = gui_before.get(key, "?"), gui_after.get(key, "?")
                lines.append(f"  {key}: {old} -> {new}")
            for problem in gui_problems:
                failures.append(f"motion_gui.{problem}")
        if grip_want:
            lines.append("── gripper adapter ──")
            l, f = self._profile_diff(grip_want, grip_before, grip_after,
                                      grip_sent, "gripper adapter")
            lines += l
            failures += f
        if pipe_want:
            lines.append("── pipeline node ──")
            l, f = self._profile_diff(pipe_want, pipe_before, pipe_after,
                                      pipe_sent, "pipeline node")
            lines += l
            failures += f
        voice_failures: List[str] = []
        if voice_want:
            lines.append("── speech node (voice filters) ──")
            l, f = self._profile_diff(voice_want, voice_before, voice_after,
                                      voice_sent, "speech node")
            lines += l
            failures += f
            voice_failures = list(f)
        if warnings:
            lines.append("── warnings ──")
            lines += [f"  {w}" for w in warnings]
        if failures:
            lines.append("── FAILED ──")
            lines += [f"  {f}" for f in failures]

        # Mirror what the node reported onto the Motion widgets and the two
        # Apply status labels, so nothing on screen contradicts the summary.
        self.motion_win.set_motion_widgets(
            grip_after or grip_want,
            {k: (pipe_after or pipe_want).get(k)
             for k in profiles.COLLISION_KEYS})
        if pipe_want and any(k in pipe_want for k in profiles.COLLISION_KEYS):
            if pipe_after:
                self.motion_win.show_neigh_apply(
                    "ok" if not failures else "mismatch",
                    f"enabled={pipe_after.get('neighbour_obstacles_enabled')}, "
                    f"padding={pipe_after.get('neighbour_obstacle_padding_m')}")
            else:
                self.motion_win.show_neigh_apply(
                    "fail" if not pipe_sent else "sent",
                    "pipeline node unreachable" if not pipe_sent
                    else "pipeline node did not answer")
        if grip_want:
            if grip_after:
                self.motion_win.show_grip_apply(
                    "ok", f"open={float(grip_after['pulse_hold_s_open']):.2f}s, "
                          f"close={float(grip_after['pulse_hold_s_close']):.2f}s")
            else:
                self.motion_win.show_grip_apply(
                    "fail" if not grip_sent else "sent",
                    "gripper adapter unreachable" if not grip_sent
                    else "adapter did not answer")
        if voice_want:
            # (the four spinboxes were mirrored above, before the summary
            # lines were built, so a clamp shows up as a warning line)
            if voice_after:
                self._show_vf_apply(
                    "ok" if not voice_failures else "mismatch",
                    ", ".join(f"{k}={voice_after.get(k)}"
                              for k in profiles.VOICE_KEYS
                              if k in voice_want))
            else:
                # Unreachable is not "lost": the spinboxes now hold the
                # profile's values, and the voice launch builds its
                # whisper_* arguments from those boxes — so the next
                # "Start voice" comes up with them (same promise the
                # Apply button's own failure message makes).
                self._show_vf_apply(
                    "fail" if not voice_sent else "sent",
                    "speech node unreachable — values staged for the next "
                    "Start voice" if not voice_sent
                    else "speech node did not answer")
        # Keep the Filters tab honest too (async read, no extra blocking).
        try:
            self.exp_win._refresh_params()
        except Exception as exc:                # pragma: no cover - defensive
            warnings.append(f"could not refresh the Filters tab: {exc}")

        self._set_profile_name(profile_name)
        total = (len(gui_want) + len(grip_want) + len(pipe_want)
                 + len(voice_want))
        # Severity follows the Apply buttons' vocabulary exactly: RED only
        # when nothing could be sent at all, ORANGE for "sent but not
        # verified" (a dead read-back does NOT mean the value was refused),
        # GREEN only when the nodes reported the values back.
        attempted, unreachable = 0, []
        if pipe_want:
            attempted += 1
            if not pipe_sent:
                unreachable.append("pipeline node")
        if grip_want:
            attempted += 1
            if not grip_sent:
                unreachable.append("gripper adapter")
        if voice_want:
            attempted += 1
            if not voice_sent:
                unreachable.append("speech node")
        if attempted and len(unreachable) == attempted:
            kind = "fail"
            detail = (f"{title.lower()}: {' and '.join(unreachable)} "
                      f"unreachable — no node value applied")
        elif failures:
            kind = "warn"
            detail = (f"{title.lower()}: {len(failures)} of {total} value(s) "
                      f"not verified — see the summary")
        elif warnings:
            kind = "warn"
            detail = (f"{title.lower()}: {total} value(s) applied, "
                      f"{len(warnings)} warning(s) — see the summary")
        else:
            kind = "ok"
            detail = f"{title.lower()}: {total} value(s) applied and verified"
        # (detail already carries the counts — one line, no double count)
        self._profile_status(kind, detail)
        self._show_profile_summary(title, lines)

    def _profile_diff(self, want: Dict, before, after, sent: bool,
                      label: str):
        """(summary lines, failure lines) for one node's worth of values."""
        lines: List[str] = []
        failures: List[str] = []
        for name in sorted(want):
            new = want[name]
            old = "unreadable" if not before or before.get(name) is None \
                else before.get(name)
            if not sent:
                lines.append(f"  {name}: {old} -> FAILED ({label} unreachable)")
                failures.append(f"{name}: {label} unreachable")
                continue
            if after is None:
                lines.append(f"  {name}: {old} -> {new}  (UNVERIFIED)")
                failures.append(
                    f"{name}: sent, but {label} did not answer the read-back")
                continue
            got = after.get(name)
            if got is None:
                lines.append(f"  {name}: {old} -> FAILED (unknown to {label})")
                failures.append(f"{name}: {label} does not know this parameter")
                continue
            if self._value_equal(got, new):
                lines.append(f"  {name}: {old} -> {got}")
            else:
                lines.append(f"  {name}: {old} -> {got}  (asked for {new!r})")
                failures.append(
                    f"{name}: asked {new!r}, {label} reports {got!r}")
        return lines, failures

    def _show_profile_summary(self, title: str, lines: List[str]) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(760, 560)
        lay = QtWidgets.QVBoxLayout(dlg)
        head = QtWidgets.QLabel(
            "Every line shows what the NODE reported after the change — not "
            "what was asked for.")
        head.setWordWrap(True)
        head.setStyleSheet("color:#888; font-size:11px;")
        lay.addWidget(head)
        view = QtWidgets.QPlainTextEdit("\n".join(lines))
        view.setReadOnly(True)
        view.setStyleSheet("font-family: monospace; font-size: 11px;")
        lay.addWidget(view, stretch=1)
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        lay.addWidget(buttons)
        dlg.exec_()

    def _on_gripper(self, open_: bool) -> None:
        if self._software_stop_latched:
            self.statusBar().showMessage(
                "Software stop is latched — re-arm before commanding the gripper."
            )
            return
        # Push the (possibly edited) hold durations first, then command.
        pushed = self.bridge.set_gripper_holds(
            self.motion_win.grip_open_spin.value(), self.motion_win.grip_close_spin.value()
        )
        if not pushed:
            self.statusBar().showMessage(
                "⚠ hold times NOT applied — gripper adapter param service "
                "unreachable (is the driver running?)")
        ok = self.bridge.send_gripper(open_)
        self.statusBar().showMessage(
            f"Gripper {'OPEN' if open_ else 'CLOSE'} sent."
            if ok else "Gripper action server not available — is the driver running?"
        )

    # ── System bar handlers ──────────────────────────────────────────
    def _on_reset_cycle(self) -> None:
        """Soft reset of the current pick/detection cycle (Majd,
        2026-08-25). Everything stays running; only the cycle state is
        wiped so detection starts over."""
        self.bridge.reset_cycle()
        self._manual_execution_pending = False
        self._manual_execution_released = False
        self._grasp_ready = False
        if not self._software_stop_latched:
            self.exec_btn.setEnabled(True)
            self.exec_btn.setText("▶ EXECUTE")
        self.statusBar().showMessage(
            "Cycle reset: pick aborted, selection cleared — detection is "
            "waiting for a fresh selection."
        )

    def _on_emergency_stop(self) -> None:
        """Latch/unlatch the GUI's broad software motion stop."""
        if self._software_stop_latched:
            self._software_stop_latched = False
            self.bridge.llm_cycle_active = False
            self.bridge.set_grasp_execution_enabled(True)
            # Re-arm must also REACTIVATE the trajectory controller: the
            # stop deactivates scaled_joint_trajectory_controller, and
            # without this every post-E-Stop execute dies with "Can't
            # accept new action goals. Controller is not running."
            # (found live 2026-08-25 after Majd's E-Stop test).
            try:
                subprocess.Popen(
                    ["ros2", "control", "switch_controllers",
                     "--activate", "scaled_joint_trajectory_controller"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            except Exception:
                pass
            self.go_btn.setEnabled(True)
            self.go_btn.setText("GO")
            self.exec_btn.setEnabled(True)
            self.exec_btn.setText("▶ EXECUTE")
            self.grip_open_btn.setEnabled(True)
            self.grip_close_btn.setEnabled(True)
            self.estop_btn.setText("🛑 E-Stop")
            self.estop_btn.setToolTip(
                "Immediately stops MoveIt, aborts the grasp executor and "
                "fast replay, cancels controller goals, and requests UR "
                "dashboard program stop. This is NOT safety-rated; use "
                "the physical pendant E-stop in an emergency."
            )
            self.estop_btn.setStyleSheet(
                "QPushButton {background:#a62222; color:white; font-weight:600; "
                "font-size:12px; border:1px solid #e06b6b; border-radius:5px;}"
                "QPushButton:hover {background:#c52a2a;}"
            )
            self.statusBar().showMessage(
                "Re-armed: motion controls enabled, trajectory controller "
                "reactivating. If the pendant program stopped, press Play "
                "there too."
            )
            return

        self._software_stop_latched = True
        # Flip the button FIRST (Majd 2026-08-25: "the E-Stop remains
        # E-Stop... I might forget that I pressed it") — the label must
        # change instantly, before any stop call that could take time
        # or fail mid-way.
        self.estop_btn.setText("⛔ Release E-Stop")
        self.estop_btn.setStyleSheet(
            "QPushButton {background:#4f1111; color:white; font-weight:600; "
            "font-size:12px; border:2px solid white; border-radius:5px;}"
        )
        self.estop_btn.setToolTip(
            "SOFTWARE STOP LATCHED. Click to re-arm GUI motion controls; "
            "the real robot may also require Play on the pendant."
        )
        QtWidgets.QApplication.processEvents()
        dashboard_sent = self.bridge.emergency_stop()
        self._fast_pending = None
        self._manual_execution_pending = False
        self._manual_execution_released = False
        self._grasp_ready = False
        # "E-Stop should kill EVERYTHING" (Majd, 2026-08-21): also stop
        # the GUI-managed pipeline launch — its selection/auto-pick loop
        # is what kept restarting cycles after every stop. The driver
        # stays up (it holds the robot link and the dashboard channel);
        # voice stays up (so Morty can still hear "start again").
        try:
            if self._pipeline.running():
                self._pipeline.stop()
                self.statusBar().showMessage(
                    "E-STOP: pipeline stopped too — Start pipeline to resume."
                )
        except Exception:
            pass

        # Also terminate GUI-owned command processes. ROS cancellation above
        # stops the actual controller goal; killing these prevents a blocked
        # commander from submitting another command afterward.
        for proc in (self._move_proc, self._exec_proc):
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except Exception:
                    pass
                QtCore.QTimer.singleShot(
                    1000, lambda p=proc: self._terminate_estopped_process(p)
                )

        self.exec_btn.setEnabled(False)
        self.exec_btn.setText("STOPPED")
        self.go_btn.setEnabled(False)
        self.go_btn.setText("STOPPED")
        self.grip_open_btn.setEnabled(False)
        self.grip_close_btn.setEnabled(False)
        # (Label/style/tooltip already flipped at the TOP of this branch.)
        detail = (
            "UR program stop requested; pendant Play will be required."
            if dashboard_sent else
            "Dashboard service unavailable; controller and MoveIt stops were sent."
        )
        self.statusBar().showMessage(f"SOFTWARE E-STOP LATCHED — {detail}")

    def _terminate_estopped_process(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            pass

    def _on_driver_btn(self) -> None:
        if self._driver.running():
            self._driver.stop()
            self._replay_extra.stop()
            self.driver_btn.setText("stopping…")
            self.driver_btn.setEnabled(False)
            self.statusBar().showMessage("Stopping driver stack (escalating if it hangs)…")
            return
        offline = self.mode_combo.currentIndex() == 1
        if self.cb_camera.isChecked():
            # The driver launch will bring its own RealSense node — a
            # leftover standalone instance would make the device busy.
            self._camera_extra.stop()
            subprocess.Popen(["pkill", "-f", "realsense2_camera_node"])
        if offline:
            cmd = list(OFFLINE_LAUNCH)
            cmd.append(f"use_real_camera:={'true' if self.cb_camera.isChecked() else 'false'}")
        else:
            cmd = list(DRIVER_LAUNCH[self.cb_camera.isChecked()])
        cmd.append(f"start_rviz:={'true' if self.cb_rviz.isChecked() else 'false'}")
        self._driver.start(
            cmd, log_path=os.path.expanduser("~/morty_ws/logs/driver_gui.log")
        )
        # New adapter instance → the hold times must be pushed again.
        self._holds_pushed = False
        # Fast-replay node rides with the driver (works in fake mode too);
        # skip if one already exists (e.g. a terminal-started pipeline).
        if not self.bridge.fast_replay_available():
            self._replay_extra.start(list(REPLAY_CMD))
        self.driver_btn.setText("Stop driver")
        self.mode_combo.setEnabled(False)
        self.statusBar().showMessage(
            f"Driver starting ({'OFFLINE fake hardware' if offline else 'real robot'}, "
            f"camera={'on' if self.cb_camera.isChecked() else 'off'}, "
            f"RViz={'on' if self.cb_rviz.isChecked() else 'off'}) — give it ~20 s."
        )

    def _on_pipeline_btn(self) -> None:
        if self._pipeline.running():
            self._manual_execution_pending = False
            self._manual_execution_released = False
            self.bridge.set_grasp_execution_enabled(True)
            self._pipeline.stop()
            if self._graspgen.running():
                self._graspgen.stop()
            if self._executor.running():
                self._executor.stop()
            self.pipeline_btn.setText("stopping…")
            self.pipeline_btn.setEnabled(False)
            self.statusBar().showMessage("Stopping pipeline (escalating if it hangs)…")
            return
        cmd = list(PIPELINE_CMD)
        if self.cb_multiview.isChecked():
            cmd.append("registration_enabled:=true")
        # GraspGen service first (it preloads the model in ~6 s; ZMQ queues
        # the pipeline's first request until the service is bound).
        if not self._graspgen.running():
            pid = self._pgrep_oldest(GRASPGEN_PGREP)
            if pid is not None:
                self._graspgen.adopt(pid)
            else:
                self._graspgen.start(list(GRASPGEN_SERVICE_CMD), env=os.environ.copy(),
                                     log_path=GRASPGEN_LOG)
        pipeline_env = os.environ.copy()
        pipeline_env.update(PIPELINE_ENV_EXTRA)
        # Timestamped log for post-trial timing analysis (Majd, 2026-08-12:
        # "save the logs so we can understand timing for everything").
        self._pipeline.start(
            cmd, env=pipeline_env, log_path=os.path.expanduser("~/morty_ws/logs/pipeline_gui.log")
        )
        if not self._executor.running():
            exec_cmd = list(EXECUTOR_CMD)
            if self.mode_combo.currentIndex() == 1:
                exec_cmd.append("use_fake_hardware:=true")
            self._executor.start(
                exec_cmd,
                log_path=os.path.expanduser("~/morty_ws/logs/executor_gui.log"),
            )
        self.pipeline_btn.setText("Stop pipeline")
        self._mv_synced = True  # we set the launch arg ourselves — in sync
        self.statusBar().showMessage(
            f"Pipeline starting ({'multiview' if self.cb_multiview.isChecked() else 'single view'})."
        )

    # ── Start ALL sequencer ──────────────────────────────────────────
    def _system_all_running(self) -> bool:
        """True when driver, pipeline AND voice are all up — the state
        in which the Start ALL button flips to Stop ALL."""
        return (self._driver.running() and self._pipeline.running()
                and self._voice.running())

    def _refresh_start_all_btn(self) -> None:
        """Keep the big button honest (Majd 2026-08-31: "when everything
        is on then the start all buttom should go stop all"). Called from
        the poll so it also flips when a subsystem is started/stopped by
        its own button or dies on its own."""
        want = "⏹ Stop ALL" if self._system_all_running() else "▶ Start ALL"
        if self.start_all_btn.text() != want:
            self.start_all_btn.setText(want)

    def _on_start_all(self) -> None:
        if self._system_all_running():
            # STOP everything, reverse order of the start sequence:
            # pipeline first (it talks to the driver), then driver,
            # then voice. Unlike the voice "stop_all" command (which
            # keeps voice up because voice IS the channel it arrived
            # on), a GUI click may take everything down.
            self._on_pipeline_btn()
            if self._driver.running():
                self._on_driver_btn()
            if self._voice.running():
                self._on_voice_btn()
            self._auto_pipeline = False
            self._refresh_start_all_btn()
            self.statusBar().showMessage(
                "Stop ALL: pipeline + driver + voice shutting down.")
            return
        if self._driver.running() and self._pipeline.running():
            self.statusBar().showMessage("Start ALL: everything is already running.")
            return
        # Voice is independent of the robot — start it right away.
        if not self._voice.running():
            self._on_voice_btn()
        if not self._driver.running():
            self._on_driver_btn()
            self._auto_pipeline = True
            self._auto_pipeline_deadline = time.monotonic() + 90.0
            self.statusBar().showMessage(
                "Start ALL: driver + voice starting — pipeline follows "
                "automatically once the robot is up."
            )
        elif not self._pipeline.running():
            self._on_pipeline_btn()

    def _tick_start_all(self, hw_alive: bool) -> None:
        if not self._auto_pipeline:
            return
        if self._pipeline.running():
            self._auto_pipeline = False
            return
        if time.monotonic() > self._auto_pipeline_deadline:
            self._auto_pipeline = False
            self.statusBar().showMessage(
                "Start ALL: robot did not come up in 90 s — start the "
                "pipeline manually when ready (check pendant / ● driver LED)."
            )
            return
        if hw_alive and self._driver.running():
            self._auto_pipeline = False
            self._on_pipeline_btn()
            self.statusBar().showMessage("Start ALL: robot is up — pipeline starting.")

    # ── audio devices ────────────────────────────────────────────────
    def _refresh_devices(self) -> None:
        cur_in, cur_out = _pactl_default("source"), _pactl_default("sink")
        self.in_combo.clear()
        self.in_combo.addItems(_pactl_devices("sources"))
        if cur_in:
            self.in_combo.setCurrentText(cur_in)
        self.out_combo.clear()
        self.out_combo.addItems(_pactl_devices("sinks"))
        if cur_out:
            self.out_combo.setCurrentText(cur_out)

    def _update_model_label(self, *_a) -> None:
        provider = {"claude (debug)": "☁ AWS Bedrock", "openai": "☁ OpenAI API"}.get(
            self.voice_combo.currentText(), "☁ ?"
        )
        model = self._model_names.get(self.voice_combo.currentText()) or "(default model)"
        self.model_label.setText(f"{provider} — {model}")

    # ── mic strip ────────────────────────────────────────────────────
    def _on_mute_toggle(self, muted: bool) -> None:
        if self._mute_suppress:
            return
        self.bridge.set_muted(muted)
        self.mute_btn.setText("🔇 MUTED" if muted else "🎙 live")

    def _update_mic(self, level: float, state: str, muted: bool) -> None:
        self.mic_bar.setValue(int(max(0.0, min(1.0, level * 3.0)) * 100))
        if level > 1e-5:
            db = 20.0 * math.log10(min(1.0, level))
            self.db_label.setText(f"{db:5.1f} dB")
            thr_db = 20.0 * math.log10(max(1e-4, self.vf_min_rms.value()))
            self.db_label.setStyleSheet(
                "font-family: monospace; min-width:64px; color:"
                + (GREEN if db >= thr_db else "#c66") + ";"
            )
        else:
            self.db_label.setText("  −∞ dB")
            self.db_label.setStyleSheet(
                "font-family: monospace; color:#888; min-width:64px;"
            )
        st = (state or "idle").upper()
        color = {"SPEECH": GREEN, "IDLE": "#aaa"}.get(st, ORANGE)
        self.speech_chip.setText(st)
        self.speech_chip.setStyleSheet(
            f"background:#333; color:{color}; padding:3px 10px; "
            "border-radius:8px; font-weight:bold;"
        )
        if muted != self.mute_btn.isChecked():
            self._mute_suppress = True
            self.mute_btn.setChecked(muted)
            self.mute_btn.setText("🔇 MUTED" if muted else "🎙 live")
            self._mute_suppress = False

    def _check_voice_health(self) -> None:
        """If the voice stack runs but its chat node is dead, surface the
        EXACT crash reason from the captured log (Majd: 'would be good to
        see that exact issue')."""
        if (not self._voice.running() or self.bridge.chat_alive()
                or getattr(self, "_voice_err_shown", True)):
            return
        if time.monotonic() - getattr(self, "_voice_start_t", 0) < 20.0:
            return  # still starting up
        try:
            with open(self._voice_log, "r", errors="replace") as f:
                lines = f.readlines()[-120:]
        except OSError:
            return
        picked: List[str] = []
        for ln in lines:
            if re.search(r"RuntimeError|No credentials|Error:|error:|died|Exception",
                         ln) and not re.search(r"sk-[A-Za-z0-9]|Bearer|token", ln):
                picked.append(ln.strip()[:160])
        self._voice_err_shown = True
        if picked:
            self.chat_log.appendPlainText("⚠ VOICE ERROR (from voice_gui.log):")
            for ln in picked[-4:]:
                self.chat_log.appendPlainText(f"   {ln}")
            self.exp_win.record_event(f"voice_error: {picked[-1][:120]}")
        else:
            self.chat_log.appendPlainText(
                "⚠ chat node is down but no error line found — see "
                "~/morty_ws/logs/voice_gui.log"
            )

    def _update_pipeline_status(self, upd) -> None:
        """The two debug lines under the chat: hearing state + model state."""
        level, sstate, muted = upd["mic"]
        self._level_hist.append(level)
        now = time.monotonic()
        voice_up = bool(upd.get("voice_alive")) or self._voice.running()

        def set_line(lbl, text, color):
            lbl.setText(text)
            lbl.setStyleSheet(f"color:{color}; font-weight:bold;")

        # Hearing line. LISTENING requires the level to actually clear the
        # STT gate (Majd's fix: the engine reports "speech" on voice-LIKE
        # patterns even below threshold — those never get transcribed, so
        # showing "I hear you" was a lie).
        thr = self.vf_min_rms.value()
        loud_now = max(list(self._level_hist)[-5:], default=0.0) >= thr
        if not voice_up:
            set_line(self.hear_status, "👂 voice not running", "#888")
        elif muted:
            set_line(self.hear_status, "🔇 microphone MUTED", ORANGE)
        elif sstate == "speech" and loud_now:
            self._was_speech = True
            set_line(self.hear_status, "🎙 LISTENING — I hear you", "#e6c229")
        elif sstate == "speech":
            # Voice-LIKE pattern detected but below the STT gate — THIS
            # is the real "too quiet" case (someone IS speaking).
            # Majd 2026-08-25: plain silence must never show this.
            set_line(self.hear_status,
                     "🔴 you're speaking but TOO QUIET — raise mic / dB",
                     RED)
        else:
            if self._was_speech:
                self._was_speech = False
                self._processing_until = now + 4.0
            if now < self._processing_until:
                set_line(self.hear_status,
                         "⏳ processing what you said…", "#e6c229")
            else:
                # Quiet room, nobody speaking: normal, not an error
                # (Majd 2026-08-25: gated/USB mics deliver digital zero
                # in silence — never call that "wrong input device").
                # Show the live level so a truly dead mic is visible as
                # a frozen -inf dB.
                lvl = max(list(self._level_hist)[-5:], default=0.0)
                db = (f"{20.0 * math.log10(lvl):.0f} dB" if lvl > 1e-9
                      else "-inf dB")
                set_line(self.hear_status,
                         f"👂 quiet — waiting for you to speak ({db})",
                         GREEN)

        # Model line
        cstate = (upd.get("chat_state_now") or "").lower()
        if not voice_up:
            set_line(self.model_status, "🤖 model offline", "#888")
        elif cstate == "thinking":
            set_line(self.model_status, "🧠 Morty is THINKING…", "#e6c229")
        elif cstate == "responding":
            set_line(self.model_status, "🗣 Morty is SPEAKING…", "#3ab0ff")
        else:
            set_line(self.model_status, "🤖 model ready — waiting", GREEN)

    # ── chat (typed input == speech input) ───────────────────────────
    def _on_chat_send(self) -> None:
        text = self.chat_input.text().strip()
        if not text:
            return
        self.chat_input.clear()
        self.chat_log.appendPlainText(f"⌨ You (typed): {text}")
        self.bridge.send_chat_text(text)
        if not self.bridge.chat_alive():
            if self._voice.running():
                self.chat_log.appendPlainText(
                    "  ⚠ voice stack is up but the CHAT NODE is not — it "
                    "probably crashed (API key / model error). Details follow "
                    "below when found."
                )
                self._voice_err_shown = False  # re-scan the log now
            else:
                self.chat_log.appendPlainText(
                    "  (voice not started — press Start voice first)"
                )

    # ── voice (communication_llm) ────────────────────────────────────
    def _on_voice_btn(self) -> None:
        if self._voice.running():
            self._voice.stop()
            self.voice_btn.setText("stopping…")
            self.voice_btn.setEnabled(False)
            self.statusBar().showMessage("Stopping voice stack…")
            return
        provider = self.voice_combo.currentText()
        # No knobs (Majd): AUTO language, default model from .morty_env.
        lang = "AUTO"
        model = ""
        cmd = list(VOICE_BASE[provider]) + VOICE_COMMON_ARGS + VOICE_LANG_ARGS[lang]
        cmd += [
            f"whisper_min_rms:={self.vf_min_rms.value():.3f}",
            f"whisper_silence_ms:={self.vf_silence.value()}",
            f"whisper_max_utterance_s:={self.vf_max_utt.value():.1f}",
            f"min_chars:={self.vf_min_chars.value()}",
        ]
        env = os.environ.copy()
        # Capture the voice stack's output so crash reasons (e.g. missing
        # API key) can be SHOWN instead of guessed.
        os.makedirs(os.path.expanduser("~/morty_ws/logs"), exist_ok=True)
        self._voice_log = os.path.expanduser("~/morty_ws/logs/voice_gui.log")
        self._voice_start_t = time.monotonic()
        self._voice_err_shown = False
        self._voice.start(cmd, env=env, log_path=self._voice_log)
        self.voice_btn.setText("Stop voice")
        self.voice_combo.setEnabled(False)
        self.statusBar().showMessage(
            f"Voice starting ({provider}, {model or 'default model'}, {lang}) — "
            "say something once ● voice is green. Mic gain must stay ~40%."
        )

    # ── live multiview toggle ────────────────────────────────────────
    def _on_multiview_toggle(self, on: bool) -> None:
        if self._mv_suppress:
            return
        if self._pipeline.running():
            ok = self.bridge.set_pipeline_params({"registration_enabled": bool(on)})
            self.statusBar().showMessage(
                f"Multiview {'ON' if on else 'OFF'} — applies from the NEXT "
                "object selection."
                if ok else "Pipeline node not reachable — toggle will apply at next start."
            )
        else:
            self.statusBar().showMessage(
                f"Multiview {'ON' if on else 'OFF'} for the next pipeline start."
            )

    # ── live camera / RViz toggles ───────────────────────────────────
    def _on_rviz_toggle(self, on: bool) -> None:
        # Before the driver runs, the checkbox only chooses launch options.
        if not self._driver.running():
            return
        if on:
            # 2026-09-02 (SYSTEM_STATE_LOG #310): pass the SAME camera flag
            # the driver got. moveit_rviz.launch.py regenerates the shared
            # /tmp URDF with camera_enabled (default false) and starts its
            # own description publisher — with the driver launched with
            # camera=true, RViz received two DIFFERENT robots on one topic
            # and flipped between them (red RobotModel, arm not drawn).
            cmd = list(RVIZ_STANDALONE) + [
                f"camera_enabled:={'true' if self.cb_camera.isChecked() else 'false'}"
            ]
            self._rviz_extra.start(cmd)
            self.statusBar().showMessage("RViz starting (standalone).")
        else:
            self._rviz_extra.stop()
            # Also take down an RViz spawned inside the driver launch tree.
            subprocess.Popen(["pkill", "-x", "rviz2"])
            self.statusBar().showMessage("RViz stopped.")

    def _on_camera_toggle(self, on: bool) -> None:
        if not self._driver.running():
            return
        if on:
            # The D435 releases its V4L2/USB handles a couple of seconds
            # AFTER the old node dies; starting a new driver too early hits
            # "Device or resource busy" (seen live 2026-08-07). Kill any
            # leftover instance, then start after a settle delay.
            subprocess.Popen(["pkill", "-f", "realsense2_camera_node"])
            self.statusBar().showMessage("Camera starting in 3 s (waiting for USB release)…")

            def _start_cam():
                if self.cb_camera.isChecked() and not self._camera_extra.running():
                    self._camera_extra.start(list(CAMERA_STANDALONE))
                    self.statusBar().showMessage("RealSense camera starting (standalone).")

            QtCore.QTimer.singleShot(3000, _start_cam)
        else:
            self._camera_extra.stop()
            subprocess.Popen(["pkill", "-f", "realsense2_camera_node"])
            self.statusBar().showMessage(
                "Camera stopped (USB releases in ~3 s). NOTE: the pipeline needs it — restart before grasping."
            )

    # ── send-to-place / joint goals ──────────────────────────────────
    def _spawn_move(self, extra_params: List[str], label: str) -> None:
        cmd = list(MOVE_CMD_BASE) + extra_params + self.motion_win.motion_args()
        self._move_proc = subprocess.Popen(
            cmd, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        self.go_btn.setText("■ STOP")
        self.statusBar().showMessage(label)

    def _stop_move(self) -> None:
        # Halt the arm first (MoveIt honors this immediately), then end the
        # commander process.
        self.bridge.stop_motion()
        if self._move_proc is not None:
            proc = self._move_proc
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except Exception:
                pass
            self.go_btn.setEnabled(False)
            self.go_btn.setText("stopping…")
            QtCore.QTimer.singleShot(
                1200, lambda p=proc: self._escalate_move_stop(p, signal.SIGTERM)
            )
        if self._fast_pending is not None:
            self._fast_pending = None
            self.go_btn.setText("GO")
        self.statusBar().showMessage("STOP sent — arm halting.")

    def _escalate_move_stop(self, proc: subprocess.Popen, sig) -> None:
        """Ensure a stopped named move cannot leave the GO button stuck."""
        if self._move_proc is not proc or proc.poll() is not None:
            return
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except Exception:
            pass
        if sig == signal.SIGTERM:
            self.statusBar().showMessage(
                "Move process did not stop after SIGINT; sent SIGTERM."
            )
            QtCore.QTimer.singleShot(
                1200, lambda p=proc: self._escalate_move_stop(p, signal.SIGKILL)
            )
        else:
            self.statusBar().showMessage(
                "Move process was force-stopped; GO will reset now."
            )

    def _on_go(self) -> None:
        if self._program_running is False:
            self.statusBar().showMessage(
                "Pendant program is STOPPED — Remote mode: the GUI restarts it within seconds; Local mode: press Play on the pendant.")
            return
        if self._software_stop_latched:
            self.statusBar().showMessage(
                "Software stop is latched — click RE-ARM before moving."
            )
            return
        if self._move_proc is not None or self._fast_pending is not None:
            self._stop_move()
            return
        target = self.place_combo.currentText()
        if self.motion_win.fast_cb.isChecked() and not self.bridge.fast_replay_available():
            self.statusBar().showMessage(
                "⚡ replay node not running (start the pipeline) — using planner instead."
            )
        elif self.motion_win.fast_cb.isChecked():
            dur = float(self.motion_win.fast_dur_spin.value())
            self.bridge.request_fast_replay(target, duration_s=dur)
            self._fast_pending = (target, time.monotonic() + max(8.0, dur + 4.0))
            self.go_btn.setText("■ STOP ⚡")
            self.statusBar().showMessage(
                f"Fast replay to '{target}' over {dur:.1f}s requested…"
            )
            return
        self._spawn_move(
            ["-p", f"target_name:={target}"],
            f"Moving robot to '{target}' ({self.motion_win.planner_combo.currentText()}, "
            f"vel {self.motion_win.vel_spin.value():.2f}).",
        )

    def _on_move_joints(self, radians: List[float]) -> None:
        if self._software_stop_latched:
            self.statusBar().showMessage(
                "Software stop is latched — re-arm before moving."
            )
            return
        if self._program_running is False:
            self.statusBar().showMessage(
                "Pendant program is STOPPED — Remote mode: the GUI restarts it within seconds; Local mode: press Play on the pendant.")
            return
        if self._move_proc is not None:
            return
        params = ["-p", "use_joint_target:=true"]
        for i, val in enumerate(radians):
            params += ["-p", f"joint_target_{i}:={val:.5f}"]
        self._spawn_move(params, "Moving to joint goal from Motion window.")

    def _on_read_joints(self) -> None:
        vals = [math.degrees(self._last_joints.get(j, 0.0)) for j in UR_JOINTS]
        self.motion_win.set_joints_deg(vals)
        self.statusBar().showMessage("Motion window joints set from current robot state.")

    def _check_fast_pending(self, fast_status: tuple) -> None:
        if self._fast_pending is None:
            return
        target, deadline = self._fast_pending
        status, stamp = fast_status
        if status.startswith("done"):
            self._fast_pending = None
            self.go_btn.setEnabled(True)
            self.go_btn.setText("GO")
            self.statusBar().showMessage(f"⚡ Fast replay to '{target}' done.")
            return
        failed = status and not status.startswith("done") and stamp > 0
        if failed or time.monotonic() > deadline:
            self._fast_pending = None
            self.go_btn.setEnabled(True)
            self.go_btn.setText("GO")
            reason = f"status '{status}'" if failed else "timeout (pipeline running?)"
            self.statusBar().showMessage(
                f"Fast replay unavailable ({reason}) — falling back to planner."
            )
            self._spawn_move(
                ["-p", f"target_name:={target}"],
                f"Moving robot to '{target}' (planner fallback).",
            )

    # ── UI refresh ───────────────────────────────────────────────────
    def _reap_system(self) -> None:
        rc = self._driver.reap()
        if rc is not None:
            self.driver_btn.setText("Start driver")
            self.driver_btn.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.statusBar().showMessage(f"Driver stack stopped (exit {rc}).")
        rc = self._pipeline.reap()
        if rc is not None:
            self.pipeline_btn.setText("Start pipeline")
            self.pipeline_btn.setEnabled(True)
            self._mv_synced = False
            self.statusBar().showMessage(f"Pipeline stopped (exit {rc}).")
        self._rviz_extra.reap()
        self._camera_extra.reap()
        self._replay_extra.reap()
        rc = self._voice.reap()
        if rc is not None:
            self.voice_btn.setText("Start voice")
            self.voice_btn.setEnabled(True)
            self.voice_combo.setEnabled(True)
            self.statusBar().showMessage(f"Voice stack stopped (exit {rc}).")

    # ── CPU priority tuner (2026-09-02, SYSTEM_STATE_LOG #300) ─────────
    # RViz's single Ogre render thread needs a full core and gets starved
    # by throughput hogs during grasps (YOLO ~150%, downsampler ~20-80%),
    # which tips it into the never-recovering backlog. We cannot raise
    # RViz above nice 0 without root, but we CAN lower the hogs: both are
    # throughput jobs where +8 costs nothing perceptible. Latency-critical
    # nodes (controller, move_group, executor, motion server, voice) are
    # deliberately NOT touched.
    _NICE_TARGETS = (
        ("grasp_pipeline.run_grasp_pipeline", 8),
        ("pointcloud_downsampler", 8),
    )
    # ── CPU ISOLATION (2026-09-02, SYSTEM_STATE_LOG #302) ────────────
    # The Alienware is 6 physical cores x 2 hyperthreads: sibling pairs
    # (0,6) (1,7) (2,8) (3,9) (4,10) (5,11). Physical core 5 = CPUs
    # {5,11} is RESERVED for the interactive processes — RViz's single
    # render thread, this GUI, and the desktop shell — and every Morty
    # throughput/compute node is confined to the other five cores. Then
    # a grasp, a YOLO burst or a stray build can saturate the work
    # cores without ever starving the screen: the "RViz and the whole
    # PC get stuck" spiral needs contention on the render core, and
    # there is none. The UR controller is deliberately left unpinned.
    _RESERVED_CPUS = {5, 11}
    _WORK_CPUS = set(range(os.cpu_count() or 12)) - _RESERVED_CPUS
    # 2026-09-08 (Majd: "the whole PC is very sticky"): pinning gnome-shell
    # and this GUI to the reserved core was the sluggishness — affinity is
    # INHERITED, so every app launched from the desktop afterwards (VS Code,
    # Firefox, terminals, the updater…) landed on that one physical core
    # too; 46 processes were confined to {5,11} with Morty not even running.
    # Reservation now works the right way round: the WORKERS are kept off
    # the reserved core; the interactive side is free to use every core
    # and simply finds the reserved one idle when the workers are busy.
    # Only RViz's render thread still gets pinned there (#314).
    _INTERACTIVE = ("rviz2",)
    _WORKERS = (
        "grasp_pipeline.run_grasp_pipeline", "pointcloud_downsampler",
        "realsense2_camera_node", "move_group", "speech_input_node",
        "speak_action_server", "chat_node", "intent_router_node",
        "grasp_executor_lin_llm", "move_to_named_pose_server",
        "replay_node", "yolo",
    )

    @staticmethod
    def _pin_all_threads(pid: int, cpus: set) -> bool:
        """Affinity is per-thread on Linux; pin every task of the pid.
        Returns True if anything changed."""
        changed = False
        try:
            tids = os.listdir(f"/proc/{pid}/task")
        except Exception:
            return False
        for t in tids:
            try:
                tid = int(t)
                if os.sched_getaffinity(tid) != cpus:
                    os.sched_setaffinity(tid, cpus)
                    changed = True
            except Exception:
                pass
        return changed

    def _tune_priorities(self) -> None:
        try:
            out = subprocess.run(
                ["ps", "-eo", "pid,ni,args"], capture_output=True,
                text=True, timeout=1.0).stdout
        except Exception:
            return
        for line in out.splitlines()[1:]:
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid_s, ni_s, args = parts
            if "grep" in args or "ps -eo" in args:
                continue
            try:
                pid = int(pid_s)
            except ValueError:
                continue
            for pat, want in self._NICE_TARGETS:
                if pat in args:
                    try:
                        if int(ni_s) < want:
                            os.setpriority(os.PRIO_PROCESS, pid, want)
                            self.statusBar().showMessage(
                                f"CPU priority: {pat.split('.')[-1]} -> nice {want}")
                    except Exception:
                        pass
            head = args.split()[0].rsplit("/", 1)[-1] if args.split() else ""
            if any(k in args for k in self._INTERACTIVE) and \
                    ("claude" not in args):
                if head == "rviz2":
                    # 2026-09-02 (SYSTEM_STATE_LOG #314): RViz's MAIN thread
                    # is the Ogre render loop — that one owns the reserved
                    # core. Its helper threads (ROS spinners; with the
                    # MotionPlanning plugin a ~15% joint_states listener)
                    # go to the work cores so they never steal render time.
                    changed = False
                    try:
                        if os.sched_getaffinity(pid) != self._RESERVED_CPUS:
                            os.sched_setaffinity(pid, self._RESERVED_CPUS); changed = True
                        for t in os.listdir(f"/proc/{pid}/task"):
                            tid = int(t)
                            if tid != pid and os.sched_getaffinity(tid) != self._WORK_CPUS:
                                os.sched_setaffinity(tid, self._WORK_CPUS); changed = True
                    except Exception:
                        pass
                    if changed:
                        self.statusBar().showMessage(
                            "CPU isolation: rviz2 render thread -> reserved core, helpers -> work cores")
                elif self._pin_all_threads(pid, self._RESERVED_CPUS):
                    self.statusBar().showMessage(
                        f"CPU isolation: {head} -> reserved core {sorted(self._RESERVED_CPUS)}")
            elif any(k in args for k in self._WORKERS):
                self._pin_all_threads(pid, self._WORK_CPUS)

    def _poll(self) -> None:
        self._poll_count += 1
        self._reap_system()
        self._refresh_start_all_btn()
        if self._poll_count % 50 == 0:
            self._tune_priorities()
        # One-shot: apply the Advanced hold times as soon as the adapter
        # exists, so VOICE picks (which never press EXECUTE) also grip
        # with what the GUI shows instead of the adapter's launch
        # defaults — they were 0.5/1.2 while the GUI read 0.40/0.40.
        if not getattr(self, "_holds_pushed", False) and self._poll_count % 10 == 0:
            if self.bridge.set_gripper_holds(
                self.motion_win.grip_open_spin.value(),
                self.motion_win.grip_close_spin.value(),
                quiet=True):
                self._holds_pushed = True
        elif getattr(self, "_holds_pushed", False) and self._poll_count % 50 == 0:
            # SELF-HEAL every 5 s (Majd 2026-08-28: "all the grasps should
            # be with 0.30s opening the gripper but isnt like that").
            # The push above was ONE-SHOT: if the driver (and with it the
            # adapter) was restarted afterwards, the adapter came back on
            # its LAUNCH defaults while these spinboxes still displayed the
            # operator's numbers — the GUI then showed a value the robot
            # was not using, with nothing on screen to say so. Re-pushing
            # is idempotent and costs one async param set at 0.2 Hz, so
            # what he reads here is always what the gripper runs.
            self.bridge.set_gripper_holds(
                self.motion_win.grip_open_spin.value(),
                self.motion_win.grip_close_spin.value(),
                quiet=True)
        if self._exec_proc is not None and self._exec_proc.poll() is not None:
            self._executor_finished()
        if self._move_proc is not None and self._move_proc.poll() is not None:
            rc = self._move_proc.poll()
            try:
                out, _ = self._move_proc.communicate(timeout=1.0)
            except Exception:
                out = ""
            self._move_proc = None
            self.go_btn.setEnabled(not self._software_stop_latched)
            self.go_btn.setText(
                "STOPPED" if self._software_stop_latched else "GO"
            )
            verdict = MOVE_EXIT_TEXT.get(rc, f"exit {rc}")
            self.statusBar().showMessage(f"Move finished: {verdict}")
            if rc != 0:
                # Failure debug (Majd's request): dump the commander's own
                # words into the Experiments log so the WHY is visible.
                lines = [ln for ln in (out or "").splitlines() if ln.strip()]
                self.exp_win.append_log(
                    [f"── move FAILED ({verdict}) — commander output ──"]
                    + lines[-15:]
                    + ["── end move debug ──"]
                )
                self.exp_win.record_event(f"move_failed rc={rc} ({verdict})")
                self.adv_action.setChecked(True)
                self.adv_tabs.setCurrentIndex(1)  # Experiments tab
        # Process checks are subprocess calls — do them every ~2 s only.
        if self._poll_count % 20 == 0:
            self._rviz_seen = (
                subprocess.run(["pgrep", "-x", "rviz2"], capture_output=True).returncode == 0
            )
            # Robot reachability: non-blocking ping, result read next tick.
            if self._ping_proc is None:
                self._ping_proc = subprocess.Popen(
                    ["ping", "-c", "1", "-W", "1", ROBOT_IP],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif self._ping_proc.poll() is not None:
                self._robot_pingable = (self._ping_proc.returncode == 0)
                self._ping_proc = None
            # Shuttle PC (Astra host) — same non-blocking ping (2026-09-03).
            if self._ping_shuttle_proc is None:
                self._ping_shuttle_proc = subprocess.Popen(
                    ["ping", "-c", "1", "-W", "1", SHUTTLE_IP],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            elif self._ping_shuttle_proc.poll() is not None:
                self._shuttle_pingable = (self._ping_shuttle_proc.returncode == 0)
                self._ping_shuttle_proc = None
            # Pendant state: the driver only reports CHANGES, so a GUI that
            # starts after the driver may never hear it — ask the dashboard.
            self.bridge.query_program_running()
            self.bridge.query_remote_mode()
            # Remote mode: a stopped program is restarted from here, nobody
            # walks to the pendant (Majd 2026-09-08). Local mode: the
            # pendant's Play is the only way and the LED tooltip says so.
            if self.bridge.auto_play_tick(self._software_stop_latched):
                self.statusBar().showMessage(
                    "Pendant program was stopped — Play sent automatically (Remote mode).")
            if self.bridge.ensure_trajectory_controller_active():
                self.statusBar().showMessage(
                    "Trajectory controller was inactive while the program runs — re-activated it.")
            self._adopt_orphans()
            self._check_voice_health()
            # Sync the multiview checkbox with an adopted/foreign pipeline's
            # actual registration_enabled value.
            if self._pipeline.running() and not self._mv_synced:
                self.bridge.fetch_pipeline_params(["registration_enabled"])
            # Self-heal ⚡: whenever a driver is up (own or adopted) and no
            # replay node exists anywhere, start ours.
            if (self._driver.running() and not self._replay_extra.running()
                    and not self.bridge.fast_replay_available()):
                self._replay_extra.start(list(REPLAY_CMD))
            # Same self-heal for the LLM grasp executor: a pipeline without
            # it selects grasps and then waits on cycle_done forever.
            if self._pipeline.running() and not self._executor.running():
                exec_cmd = list(EXECUTOR_CMD)
                if self.mode_combo.currentIndex() == 1:
                    exec_cmd.append("use_fake_hardware:=true")
                self._executor.start(
                    exec_cmd,
                    log_path=os.path.expanduser("~/morty_ws/logs/executor_gui.log"),
                )
                self.statusBar().showMessage(
                    "Started grasp executor (was missing — picks would never execute)."
                )
            # GO label advertises whether ⚡ fast replay will actually be
            # used, so a planner move is never a surprise.
            if (not self._software_stop_latched and
                    self._move_proc is None and self._fast_pending is None):
                # Majd 2026-08-25: say the MODE, not just ⚡/no-⚡ — a
                # planner move must be readable on the button itself.
                if self.motion_win.fast_cb.isChecked():
                    self.go_btn.setText(
                        "GO ⚡" if self.bridge.fast_replay_available()
                        else "GO (PTP–no ⚡)"
                    )
                else:
                    self.go_btn.setText("GO (PTP)")

        upd = self.bridge.take_updates()
        if upd.get("voice_stop") and self._exec_proc is not None:
            # Voice "stop" also halts a button-launched executor (sync
            # fix, 2026-08-20 — voice and button paths were separate).
            self._on_execute_stop()
            self.statusBar().showMessage(
                "Voice STOP — halting the button-launched executor."
            )
        astra_view = self.view_combo.currentIndex() == 1
        if "image" in upd and upd["image"] is not None:
            if not astra_view:
                self.camera.show_bgr(upd["image"])
            self.mmhri_win.feed_camera(upd["image"])   # MMHRI always records the RealSense
        if astra_view and upd.get("astra_image") is not None:
            self.camera.show_bgr(upd["astra_image"])
        if upd.get("campose") is not None:
            self.view3d.set_campose(np.asarray(upd["campose"]))
        if "frame_cloud" in upd:
            # Live per-frame captures (sweep/single view) show in the same
            # panel; the final result below overwrites them and re-frames.
            # The bbox crop drags in BACKGROUND depth (metres behind the
            # object) — keep only the near blob so the view frames the
            # object, not the room (Majd, 2026-08-11).
            fpts, fcols = upd["frame_cloud"]
            if fpts is not None and len(fpts) > 100:
                z = fpts[:, 2]
                z0 = float(np.percentile(z, 10))
                keep = z < (z0 + 0.35)
                if int(keep.sum()) > 50:
                    fpts = fpts[keep]
                    if fcols is not None and len(fcols) == len(keep):
                        fcols = fcols[keep]
            self.view3d.set_cloud(fpts, fcols)
        if "cloud" in upd:
            pts, cols = upd["cloud"]
            self.view3d.reset_view()  # final result → re-frame from camera side
            self.view3d.set_cloud(pts, cols)
            self.mmhri_win.feed_cloud(pts, cols, dict(self.bridge.poses))
            self.mmhri_win.feed_view3d_snapshot(self.view3d.grabFramebuffer)
            # Standing browser viewer for EVERY cycle, trial or not
            # (Majd, 2026-08-12: "make the HTML directly after it's done").
            self._latest_cloud_html = (pts, cols)
            self._write_latest_html()
        if "poses" in upd:
            for kind, poses in upd["poses"].items():
                self.view3d.set_poses(kind, poses)
            # Trial recorder: grasps arrive AFTER the cloud; snapshot on
            # every update so grasps.json/.ply hold the final state.
            self.mmhri_win.feed_grasps(dict(self.bridge.poses))
            self._write_latest_html()
        # EXECUTE readiness: armed by a best-grasp arrival, disarmed by a
        # new selection cycle.
        cyc = upd.get("cycle_id")
        if cyc is not None and cyc != self._seen_cycle:
            if self._seen_cycle is not None:
                self._grasp_ready = False
            self._seen_cycle = cyc
            self.mmhri_win.feed_selection()
        if (not self._software_stop_latched and
                "poses" in upd and upd["poses"].get("best")):
            self._grasp_ready = True
        if self._software_stop_latched:
            self.exec_btn.setEnabled(False)
            self.exec_btn.setText("STOPPED")
        elif self._exec_proc is None:
            if getattr(self.bridge, "llm_cycle_active", False):
                self._manual_execution_released = False
                # Persistent grasp executor running — the button IS the stop.
                self.exec_btn.setEnabled(True)
                self.exec_btn.setText("■ STOP (active grasp)")
                self.exec_btn.setStyleSheet(
                    "QPushButton {background:#a03030; color:white; "
                    "font-weight:bold; font-size:14px;}"
                )
            elif self._manual_execution_released:
                self.exec_btn.setEnabled(False)
                self.exec_btn.setText("starting manual grasp…")
                self.exec_btn.setStyleSheet(
                    "QPushButton {background:#8a6d1d; color:white; "
                    "font-weight:bold; font-size:14px;}"
                )
            else:
                self.exec_btn.setEnabled(self._grasp_ready)
                self.exec_btn.setText(
                    "▶ EXECUTE" if self._grasp_ready else "▶ EXECUTE (no grasp)"
                )
                self.exec_btn.setStyleSheet(
                    "QPushButton {background:#1d7a2c; color:white; "
                    "font-weight:bold; font-size:14px;}"
                )
        if "filter" in upd:
            self.exp_win.update_filter(upd["filter"])
        if "utility" in upd:
            self.exp_win.update_utility(upd["utility"])
        if "log" in upd:
            self.exp_win.append_log(upd["log"])
        if upd.get("status") and not self._software_stop_latched:
            self.statusBar().showMessage(upd["status"])
        if "cycle_done" in upd:
            self.exp_win.record_cycle(upd["cycle_done"])
        if upd.get("executor_done"):
            self._manual_execution_pending = False
            self._manual_execution_released = False
            self.bridge.set_grasp_execution_enabled(True)
        if "joints" in upd:
            names, positions = upd["joints"]
            self._last_joints.update(
                {n: p for n, p in zip(names, positions) if n in UR_JOINTS}
            )
        if "fast_status" in upd:
            self._check_fast_pending(upd["fast_status"])
        self._tick_start_all(bool(upd.get("hw_alive")))
        if "mic" in upd:
            self._update_mic(*upd["mic"])
            self._update_pipeline_status(upd)
        if "waveform" in upd and upd["waveform"] is not None:
            # Mirrored ±envelope (Majd's reference look), ×4 gain, clipped.
            aw = np.clip(np.abs(upd["waveform"]) * 4.0, 0.0, 1.0)
            self.wave_upper.setData(aw)
            self.wave_lower.setData(-aw)
        if "chat_events" in upd:
            self._processing_until = 0.0  # speech got through — stop "processing"
            for who, text in upd["chat_events"]:
                if who == "Rejected":
                    # "<reason> :: <text>" — reason-only events (e.g. "too
                    # quiet") stay off the chat; the hearing line covers
                    # them and they'd spam on every background rumble.
                    reason, _, heard = text.partition(" :: ")
                    if heard.strip():
                        self.chat_log.appendPlainText(
                            f"⚠ ignored ({reason}): {heard}"
                        )
                        self.mmhri_win.feed_chat(
                            "⚠ ignored", f"({reason}) {heard}"
                        )
                    continue
                if who == "Spoken":
                    # Narrations from router/executor via the speak server.
                    # Chat replies also pass through /speech/ai_said —
                    # dedupe against Morty's recent messages.
                    if text in self._recent_morty_texts:
                        continue
                    self.chat_log.appendPlainText(f"🔊 Morty (status): {text}")
                    self.exp_win.record_event(f"voice status: {text}")
                    # Status narrations are experiment evidence ("Got the
                    # vitamin tube…") — trials 1-3 conversations lacked them.
                    self.mmhri_win.feed_chat("Morty (status)", text)
                    continue
                icon = "🎤" if who == "You" else "🤖"
                if who == "Morty":
                    self._recent_morty_texts.append(text)
                self.chat_log.appendPlainText(f"{icon} {who}: {text}")
                self.voice_status.setText(f"{icon} {text[:70]}")
                self.voice_status.setStyleSheet("color:#ddd; padding-right:8px;")
                # Voice interactions belong in the experiment record too.
                self.exp_win.record_event(f"voice {who}: {text}")
                self.mmhri_win.feed_chat(who, text)
            state = upd.get("chat_state", "")
            self.chat_state_label.setText(f"state: {state or '—'}")
        for action in upd.get("system_intents", []):
            self._handle_system_intent(action)
        if "cand_status" in upd:
            for line in upd["cand_status"]:
                self.exp_win.cand_log.appendPlainText(line)
                # Main-window pane: clear per probing round so it reads
                # as THIS pick's story.
                if line.startswith("grasp 1/"):
                    self.probe_hist.clear()
                self.probe_hist.appendPlainText(line)
            self.probe_hist.setVisible(True)
            self.probe_hist.verticalScrollBar().setValue(
                self.probe_hist.verticalScrollBar().maximum()
            )
            # Latest one also in the status bar so it's visible without
            # opening the Advanced window.
            self.statusBar().showMessage(upd["cand_status"][-1])
        if "pipeline_params" in upd:
            mp = upd["pipeline_params"]
            if mp.get("registration_enabled") is not None:
                self._mv_synced = True
                self._mv_suppress = True
                self.cb_multiview.setChecked(bool(mp["registration_enabled"]))
                self._mv_suppress = False
            if any(k in mp for k in ALL_TUNABLE_PARAMS):
                self.exp_win.update_params(mp)
        self._update_leds(upd)

    @staticmethod
    def _pgrep_oldest(pattern: str) -> Optional[int]:
        r = subprocess.run(["pgrep", "-o", "-f", pattern], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            try:
                return int(r.stdout.strip().splitlines()[0])
            except ValueError:
                return None
        return None

    @staticmethod
    def _pgrep_all(pattern: str) -> List[int]:
        """All matching pids, oldest first (pgrep lists in pid order, which
        is start order for our short-lived session)."""
        r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
        if r.returncode != 0:
            return []
        out: List[int] = []
        for tok in r.stdout.split():
            try:
                out.append(int(tok))
            except ValueError:
                pass
        return out

    def _adopt_newest_executor(self) -> None:
        """The executor is a singleton: two of them race for the same
        candidate topics and the loser wedges in a blocking execute.  A
        stale one survives when a previous GUI died, or when a manual
        relaunch left the old instance running.  Adopt the NEWEST wrapper
        (the one that carries the current binary and parameters) and
        SIGINT every older duplicate so ros2 launch tears it down cleanly."""
        pids = self._pgrep_all(r"ros2 launch .*grasp_executor_lin_ll[m]")
        if not pids:
            return
        newest = max(pids)
        for pid in pids:
            if pid == newest:
                continue
            try:
                os.kill(pid, signal.SIGINT)
                print(f"[gui] retired duplicate executor launch pid {pid}", flush=True)
            except ProcessLookupError:
                pass
        self._executor.adopt(newest)

    def _write_latest_html(self) -> None:
        """~/morty_ws/experiments/latest/view3d.html — refreshed on every
        cycle's cloud + each grasp update, independent of trial recording."""
        cloud = getattr(self, "_latest_cloud_html", None)
        if cloud is None:
            return
        out_dir = os.path.expanduser("~/morty_ws/experiments/latest")
        try:
            os.makedirs(out_dir, exist_ok=True)
            write_view3d_html(out_dir, cloud[0], cloud[1],
                              dict(self.bridge.poses))
        except Exception:
            pass

    def _handle_system_intent(self, action: str) -> None:
        """Execute a voice system-control intent (chat_node published it on
        /robot/intent; the GUI owns the launches). Idempotent: a start on a
        running component (or stop on a stopped one) is only a status note —
        the buttons toggle, so blindly clicking would do the opposite."""
        if action == "start_all":
            self._on_start_all()
            self.statusBar().showMessage("🎙 voice command: Start ALL")
            return
        if action == "stop_all":
            # Voice stays up — it IS the channel the command came from.
            if self._pipeline.running():
                self._on_pipeline_btn()
            if self._driver.running():
                self._on_driver_btn()
            self.statusBar().showMessage(
                "🎙 voice command: stop_all (pipeline + driver; voice stays up)"
            )
            return
        target, want_running, btn = {
            "start_driver": (self._driver, True, self._on_driver_btn),
            "stop_driver": (self._driver, False, self._on_driver_btn),
            "start_pipeline": (self._pipeline, True, self._on_pipeline_btn),
            "stop_pipeline": (self._pipeline, False, self._on_pipeline_btn),
        }.get(action, (None, None, None))
        if target is None:
            return
        if target.running() == want_running:
            self.statusBar().showMessage(
                f"🎙 voice command {action}: already "
                f"{'running' if want_running else 'stopped'}"
            )
            return
        btn()
        self.statusBar().showMessage(f"🎙 voice command: {action}")

    def _adopt_orphans(self) -> None:
        """A previous GUI session's driver/pipeline must show as running —
        otherwise Start would double-launch the stack (the 3-realsense
        incident, 2026-08-07)."""
        if not self._driver.running():
            pid = self._pgrep_oldest(r"ros2 launch ur3e_gripper_3d_moveit_confi[g]")
            if pid is not None:
                self._driver.adopt(pid)
                self.driver_btn.setText("Stop driver")
                self.mode_combo.setEnabled(False)
                self.statusBar().showMessage(
                    "Adopted a driver stack from a previous session — Stop works normally."
                )
        if not self._voice.running():
            pid = self._pgrep_oldest(r"morty\.sh --.*voice-onl[y]")
            if pid is not None:
                self._voice.adopt(pid)
                self.voice_btn.setText("Stop voice")
                self.voice_combo.setEnabled(False)
        if not self._graspgen.running():
            pid = self._pgrep_oldest(GRASPGEN_PGREP)
            if pid is not None:
                self._graspgen.adopt(pid)
        if not self._pipeline.running():
            pid = self._pgrep_oldest(r"ros2 launch grasp_pipeline run_grasp_pipelin[e]")
            if pid is not None:
                self._pipeline.adopt(pid)
                self.pipeline_btn.setText("Stop pipeline")
                self.cb_multiview.setEnabled(False)
                self.statusBar().showMessage(
                    "Adopted a pipeline from a previous session — Stop works normally."
                )
        if not self._executor.running():
            self._adopt_newest_executor()

    def _update_leds(self, upd) -> None:
        def paint(led, color, tip):
            led.setStyleSheet(f"color:{color}; font-weight:bold;")
            led.setToolTip(tip)

        if upd.get("hw_alive"):
            paint(self.led_driver, GREEN, "hardware interface alive (/joint_states flowing)")
        elif self._driver.running():
            paint(self.led_driver, RED, "driver launched but /joint_states not flowing yet")
        else:
            paint(self.led_driver, GREY, "driver not running")
        if upd.get("camera_alive"):
            paint(self.led_camera, GREEN, "RealSense (wrist) frames flowing")
        elif self.cb_camera.isChecked() and self._driver.running():
            paint(self.led_camera, RED, "RealSense expected but no frames — starting up or unplugged?")
        else:
            paint(self.led_camera, GREY, "RealSense off")
        # Astra + VLA recorder (2026-09-03). Astra is only RED when the
        # recorder is armed — otherwise a dead fixed cam is just "not
        # streaming", not a fault.
        vla_up = bool(upd.get("vla_node_up"))
        if upd.get("astra_alive"):
            paint(self.led_astra, GREEN, "Astra (fixed, on the shuttle) frames flowing")
        elif vla_up:   # recorder up (GUI-launched) but its fixed camera is gone
            paint(self.led_astra, RED,
                  "Astra DEAD while VLA recorder is armed — episodes will drop frames. "
                  "Restart it on the shuttle.")
        else:
            paint(self.led_astra, GREY,
                  "Astra not streaming. With ● shuttle green this usually means the driver "
                  "FROZE on the shuttle (2026-09-08) — restart it there: "
                  "ssh demo@168.192.1.102 'bash ~/astra_ws/start_astra_on_shuttle.sh')")
        if self._rviz_seen:
            paint(self.led_rviz, GREEN, "rviz2 process running")
        else:
            paint(self.led_rviz, GREY, "rviz2 not running")
        if upd.get("voice_alive"):
            paint(self.led_voice, GREEN, "voice stack up (/chat/state publishing)")
        elif self._voice.running():
            paint(self.led_voice, RED, "voice launched but chat node not up yet")
        else:
            paint(self.led_voice, GREY, "voice not running")
        # robot = the physical robot answers ping at its IP (on = powered
        # and on the network, driver or not — Majd's definition).
        if self._robot_pingable:
            paint(self.led_robot, GREEN, f"robot answering ping at {ROBOT_IP}")
        else:
            paint(self.led_robot, GREY, f"robot not reachable at {ROBOT_IP} (powered? cable?)")
        if self._shuttle_pingable:
            paint(self.led_shuttle, GREEN, f"shuttle PC answering ping at {SHUTTLE_IP} (Astra host)")
        else:
            paint(self.led_shuttle, GREY, f"shuttle PC not reachable at {SHUTTLE_IP} (off? cable? dock NIC?)")
        program = upd.get("program")
        self._program_running = program
        if program is None:
            if upd.get("hw_alive"):
                # Driver up but no Play state heard yet: the driver reports
                # only changes; the bridge is asking the dashboard meanwhile.
                paint(self.led_play, ORANGE,
                      "pendant program not reported yet — press ▶ Play on the pendant "
                      "(asking the dashboard…)")
            else:
                paint(self.led_play, GREY, "pendant state unknown (driver not connected)")
        elif program:
            paint(self.led_play, GREEN, "pendant program RUNNING — motions will execute")
        else:
            if getattr(self.bridge, "_remote_mode", None) is True:
                paint(self.led_play, ORANGE, "pendant program STOPPED — Remote mode: Play is sent automatically")
            else:
                paint(self.led_play, ORANGE, "pendant program STOPPED — Local mode: press Play ON THE PENDANT "
                                             "(or switch the pendant to Remote and the GUI starts it)")

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        # Default policy: infrastructure survives the GUI (a closed debug
        # window must never take the robot down). But ask, so a deliberate
        # shutdown is one click (Majd, 2026-08-07).
        if self._driver.running() or self._pipeline.running():
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Morty — closing GUI")
            box.setText("Driver/pipeline are still running.")
            box.setInformativeText(
                "Keep them running (robot stays up, GUI re-adopts them next "
                "start) or stop everything?"
            )
            keep = box.addButton("Keep running", QtWidgets.QMessageBox.AcceptRole)
            box.addButton("Stop everything", QtWidgets.QMessageBox.DestructiveRole)
            cancel = box.addButton(QtWidgets.QMessageBox.Cancel)
            box.setDefaultButton(keep)
            box.exec_()
            if box.clickedButton() is cancel:
                ev.ignore()
                return
            if box.clickedButton() is not keep:
                self.mmhri_win.vla_shutdown()   # the recorder we launched, if any
                self._pipeline.stop()
                self._graspgen.stop()
                self._driver.stop()
                self._rviz_extra.stop()
                self._camera_extra.stop()
                self._replay_extra.stop()
                # The escalation timers die with the GUI; SIGINT is usually
                # enough, and a reopened GUI re-adopts any survivor.
        # Never leave an adopted executor latched in manual-wait mode after
        # this GUI (the only approval control) has gone away.
        self.bridge.set_grasp_execution_enabled(True)
        for proc in (self._exec_proc, self._move_proc):
            if proc is not None and proc.poll() is None:
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGINT)
                except Exception:
                    pass
        self.adv_win.close()
        self.mmhri_win.close()
        super().closeEvent(ev)


def main() -> None:
    bridge, spin_thread = start_bridge()
    app = QtWidgets.QApplication(sys.argv)
    win = OperatorWindow(bridge)
    win.show()
    code = app.exec_()
    import rclpy
    rclpy.try_shutdown()
    spin_thread.join(timeout=2.0)
    sys.exit(code)


if __name__ == "__main__":
    main()
