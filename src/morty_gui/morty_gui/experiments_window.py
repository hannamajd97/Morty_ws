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

import pyqtgraph as pg
from .config import *  # noqa: F401,F403

class ExperimentsWindow(QtWidgets.QMainWindow):
    def __init__(self, bridge) -> None:
        super().__init__()
        self.bridge = bridge
        self.setWindowTitle("Morty Experiments — recorder & debug")
        self.resize(960, 780)
        central = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(central)

        rec_bar = QtWidgets.QWidget()
        rec_lay = QtWidgets.QHBoxLayout(rec_bar)
        rec_lay.setContentsMargins(0, 0, 0, 0)
        self.rec_btn = QtWidgets.QPushButton("● Record pipeline stats")
        self.rec_btn.setCheckable(True)
        self.rec_btn.setToolTip(
            "TECHNICAL recorder (thesis/multiview metrics): one CSV row per "
            "grasp cycle — cycle id, best score, utility U(v), full filter-chain "
            "counts — plus executor exits, move failures and voice lines.\n"
            "The MMHRI trial recorder (video/audio/table) is its own window."
        )
        self.rec_btn.toggled.connect(self._toggle_record)
        rec_lay.addWidget(self.rec_btn)
        self.rec_label = QtWidgets.QLabel("not recording")
        self.rec_label.setStyleSheet("color:#888;")
        rec_lay.addWidget(self.rec_label, stretch=1)
        layout.addWidget(rec_bar)
        self._rec_file = None
        self._rec_writer = None
        self._rec_rows = 0

        tabs = QtWidgets.QTabWidget()

        live_tab = QtWidgets.QWidget()
        live_lay = QtWidgets.QVBoxLayout(live_tab)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.filter_table = QtWidgets.QTableWidget(0, 2)
        self.filter_table.setHorizontalHeaderLabels(["filter stage", "grasps left"])
        self.filter_table.horizontalHeader().setStretchLastSection(True)
        self.filter_table.verticalHeader().setVisible(False)
        self.filter_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        split.addWidget(self.filter_table)
        # Per-candidate execution probing ("grasp 7/50: rejected — no IK";
        # Majd 2026-08-12: "show for every grasp it sends what its status is").
        self.cand_log = QtWidgets.QPlainTextEdit()
        self.cand_log.setReadOnly(True)
        self.cand_log.setPlaceholderText(
            "grasp probing — one line per tried candidate"
        )
        self.cand_log.setMaximumBlockCount(300)
        split.addWidget(self.cand_log)
        self.u_plot = pg.PlotWidget(title="grasp utility U(v) per cycle")
        self.u_plot.showGrid(x=True, y=True, alpha=0.3)
        self.u_curve = self.u_plot.plot(pen=pg.mkPen((255, 170, 40), width=2),
                                        symbol="o", symbolSize=5)
        split.addWidget(self.u_plot)
        live_lay.addWidget(split, stretch=1)
        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(800)
        self.log_view.setStyleSheet("font-family: monospace; font-size: 11px;")
        live_lay.addWidget(self.log_view, stretch=1)
        tabs.addTab(live_tab, "Live debug")

        params_tab = QtWidgets.QWidget()
        params_lay = QtWidgets.QVBoxLayout(params_tab)
        btn_row = QtWidgets.QHBoxLayout()
        self.params_refresh_btn = QtWidgets.QPushButton("⟳ Read from pipeline")
        self.params_refresh_btn.clicked.connect(self._refresh_params)
        btn_row.addWidget(self.params_refresh_btn)
        self.params_apply_btn = QtWidgets.QPushButton("Apply changes")
        self.params_apply_btn.clicked.connect(self._apply_params)
        self.params_apply_btn.setEnabled(False)
        btn_row.addWidget(self.params_apply_btn)
        self.params_hint = QtWidgets.QLabel("pipeline must be running")
        self.params_hint.setStyleSheet("color:#888;")
        btn_row.addWidget(self.params_hint, stretch=1)
        params_lay.addLayout(btn_row)
        self.params_table = QtWidgets.QTableWidget(0, 2)
        self.params_table.setHorizontalHeaderLabels(["filter parameter", "value"])
        self.params_table.horizontalHeader().setStretchLastSection(True)
        self.params_table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.params_table.verticalHeader().setVisible(False)
        self.params_table.setStyleSheet("font-size: 13px;")
        params_lay.addWidget(self.params_table, stretch=1)
        note = QtWidgets.QLabel(
            "Values apply to the RUNNING pipeline only (next cycle uses them); "
            "they reset on pipeline restart. Booleans: type true/false."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#888; font-size:10px;")
        params_lay.addWidget(note)
        tabs.addTab(params_tab, "Filter parameters")

        layout.addWidget(tabs, stretch=1)
        self.setCentralWidget(central)
        self._param_types: Dict[str, type] = {}

    # ── filter parameters tab ────────────────────────────────────────
    def _refresh_params(self) -> None:
        ok = self.bridge.fetch_pipeline_params(ALL_TUNABLE_PARAMS)
        self.params_hint.setText("reading…" if ok else "pipeline node not reachable")

    def update_params(self, mapping: Dict) -> None:
        # Rows = section header rows + parameter rows, in PARAM_SECTIONS
        # order. Header rows carry "header" in UserRole so apply skips them.
        rows: list = []
        for title, names in PARAM_SECTIONS:
            rows.append(("header", title))
            rows.extend(("param", n) for n in names)
        self.params_table.setRowCount(len(rows))
        self._param_types = {}
        for i, (kind, name) in enumerate(rows):
            if kind == "header":
                item = QtWidgets.QTableWidgetItem(f"— {name} —")
                item.setFlags(QtCore.Qt.ItemIsEnabled)
                item.setData(QtCore.Qt.UserRole, "header")
                font = item.font()
                font.setBold(True)
                item.setFont(font)
                item.setForeground(QtGui.QColor("#3ab0ff"))
                self.params_table.setItem(i, 0, item)
                self.params_table.removeCellWidget(i, 1)
                blank = QtWidgets.QTableWidgetItem("")
                blank.setFlags(QtCore.Qt.ItemIsEnabled)
                self.params_table.setItem(i, 1, blank)
                continue
            val = mapping.get(name)
            self._param_types[name] = type(val) if val is not None else str
            doc = PARAM_DOCS.get(name, "")
            name_item = QtWidgets.QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
            name_item.setToolTip(doc)
            self.params_table.setItem(i, 0, name_item)
            if isinstance(val, bool):
                combo = QtWidgets.QComboBox()
                combo.addItems(["True", "False"])
                combo.setCurrentText(str(val))
                combo.setToolTip(doc)
                self.params_table.setCellWidget(i, 1, combo)
            else:
                self.params_table.removeCellWidget(i, 1)
                item = QtWidgets.QTableWidgetItem("" if val is None else str(val))
                item.setToolTip(doc)
                self.params_table.setItem(i, 1, item)
        self.params_table.resizeRowsToContents()
        self.params_apply_btn.setEnabled(True)
        self.params_hint.setText(f"read {len(mapping)} parameters from the pipeline")

    def _apply_params(self) -> None:
        mapping: Dict = {}
        for i in range(self.params_table.rowCount()):
            name_item = self.params_table.item(i, 0)
            if name_item is None or name_item.data(QtCore.Qt.UserRole) == "header":
                continue
            name = name_item.text()
            widget = self.params_table.cellWidget(i, 1)
            if isinstance(widget, QtWidgets.QComboBox):
                mapping[name] = widget.currentText() == "True"
                continue
            text = (self.params_table.item(i, 1).text() or "").strip()
            if not text:
                continue
            ty = self._param_types.get(name, str)
            try:
                if ty is bool:
                    mapping[name] = text.lower() in ("true", "1", "yes", "on")
                elif ty is int:
                    mapping[name] = int(text)
                elif ty is float:
                    mapping[name] = float(text)
                else:
                    mapping[name] = text
            except ValueError:
                self.params_hint.setText(f"bad value for {name}: '{text}'")
                return
        ok = self.bridge.set_pipeline_params(mapping)
        self.params_hint.setText(
            f"applied {len(mapping)} parameters" if ok else "pipeline node not reachable"
        )

    def _toggle_record(self, on: bool) -> None:
        import csv
        import datetime
        if on:
            rec_dir = os.path.expanduser("~/morty_ws/experiments")
            os.makedirs(rec_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(rec_dir, f"exp_{stamp}.csv")
            self._rec_file = open(path, "w", newline="")
            self._rec_writer = csv.writer(self._rec_file)
            self._rec_writer.writerow(
                ["time", "kind", "cycle", "best_score", "utility", "filter_chain", "event"]
            )
            self._rec_rows = 0
            self.rec_btn.setText("■ Stop recording")
            self.rec_label.setText(f"recording → {path} (0 rows)")
            self.rec_label.setStyleSheet(f"color:{GREEN};")
        else:
            if self._rec_file is not None:
                self._rec_file.close()
            self._rec_file = None
            self._rec_writer = None
            self.rec_btn.setText("● Record experiments")
            self.rec_label.setText(f"stopped ({self._rec_rows} rows saved)")
            self.rec_label.setStyleSheet("color:#888;")

    def _rec_write(self, row) -> None:
        import datetime
        if self._rec_writer is None:
            return
        self._rec_writer.writerow([datetime.datetime.now().isoformat(timespec="seconds")] + row)
        self._rec_file.flush()
        self._rec_rows += 1
        self.rec_label.setText(self.rec_label.text().rsplit("(", 1)[0] + f"({self._rec_rows} rows)")

    def record_cycle(self, meta) -> None:
        chain = ";".join(f"{s}:{c}" for s, c in meta.get("filter") or [])
        self._rec_write(["cycle", meta.get("cycle"), meta.get("best_score"),
                         meta.get("utility"), chain, ""])

    def record_event(self, text: str) -> None:
        self._rec_write(["event", "", "", "", "", text])

    def update_filter(self, rows) -> None:
        self.filter_table.setRowCount(len(rows))
        for i, (stage, count) in enumerate(rows):
            self.filter_table.setItem(i, 0, QtWidgets.QTableWidgetItem(stage))
            self.filter_table.setItem(i, 1, QtWidgets.QTableWidgetItem(str(count)))

    def update_utility(self, values) -> None:
        self.u_curve.setData(list(range(len(values))), values)

    def append_log(self, lines) -> None:
        for ln in lines:
            self.log_view.appendPlainText(ln)


