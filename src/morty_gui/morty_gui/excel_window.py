"""The MMHRI window's Excel export (2026-09-28, Majd: "a button inside the
experiments window, a selection of the experiments to send, and the data
goes directly into the sheet; the selection updates after every
experiment").

One list of every saved trial, newest first. A ticked trial is (or will be)
in the workbook: the ticks start as what the workbook already holds, a new
trial arrives unticked, and "📊 Send ticked to Excel" rebuilds the workbook
with exactly the ticked trials (excel_export.build + recalc, on a worker
thread). Unticking a trial that is in the workbook and sending removes it —
the window asks first. The list refreshes after every saved or deleted
trial (the MMHRI window calls refresh()) and every 10 s while it is open, so
the "Times" column turns from "preliminary" to "final" when the trial's
post-processing has finished.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from . import excel_export as xl


class _Done(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)


class ExcelExportWindow(QtWidgets.QWidget):
    COLS = ["", "Trial", "Date", "Scenario", "Command", "Full success", "Times", "In Excel"]

    def __init__(self, parent=None, is_busy=lambda: False) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("MMHRI — Excel export")
        self.resize(980, 640)
        self._is_busy = is_busy
        self._pending: dict = {}          # folder -> ticked, changed by the user since the last send
        self._in_excel: set = set()
        self._running = False
        self._done = _Done()
        self._done.finished.connect(self._on_done)
        v = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Excel file:"))
        self.path_lbl = QtWidgets.QLabel(xl.DEFAULT_XLSX)
        self.path_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top.addWidget(self.path_lbl, 1)
        self.open_btn = QtWidgets.QPushButton("Open Excel")
        self.open_btn.clicked.connect(lambda: self._open(xl.DEFAULT_XLSX))
        top.addWidget(self.open_btn)
        folder_btn = QtWidgets.QPushButton("Open folder")
        folder_btn.clicked.connect(lambda: self._open(os.path.dirname(xl.DEFAULT_XLSX)))
        top.addWidget(folder_btn)
        v.addLayout(top)

        flt = QtWidgets.QHBoxLayout()
        flt.addWidget(QtWidgets.QLabel("Show:"))
        self.day = QtWidgets.QComboBox()
        self.day.currentIndexChanged.connect(self._apply_filter)
        flt.addWidget(self.day)
        self.only_new = QtWidgets.QCheckBox("only trials not in Excel yet")
        self.only_new.toggled.connect(self._apply_filter)
        flt.addWidget(self.only_new)
        flt.addStretch(1)
        refresh = QtWidgets.QPushButton("↻ Refresh")
        refresh.clicked.connect(self.refresh)
        flt.addWidget(refresh)
        v.addLayout(flt)

        self.tree = QtWidgets.QTreeWidget()
        self.tree.setColumnCount(len(self.COLS))
        self.tree.setHeaderLabels(self.COLS)
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setUniformRowHeights(True)
        for c, w in enumerate([30, 55, 100, 80, 360, 100, 110, 70]):
            self.tree.setColumnWidth(c, w)
        self.tree.itemChanged.connect(self._on_item_changed)
        v.addWidget(self.tree, 1)

        btns = QtWidgets.QHBoxLayout()
        for text, fn in (("Tick all shown", lambda: self._tick_shown(True)),
                         ("Untick all shown", lambda: self._tick_shown(False)),
                         ("Tick the new ones", self._tick_new)):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        self.send_btn = QtWidgets.QPushButton("📊 Send ticked to Excel")
        self.send_btn.setStyleSheet("QPushButton {background:#1f6f43; color:white; font-weight:bold; padding:4px 14px;}")
        self.send_btn.clicked.connect(self._send)
        btns.addWidget(self.send_btn)
        v.addLayout(btns)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#888;")
        v.addWidget(self.status)

        self.tree.setToolTip("Ticked = in the Excel file after the next send. New trials arrive unticked.")
        self.send_btn.setToolTip("Rebuilds the Excel file with exactly the ticked trials, from their folders\n"
                                 "(trial.json, timeline.json, meta.json) — every sheet up to date.\n"
                                 "Not while a trial is recording.")
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick_timer)
        self._timer.start(10000)
        self.refresh()

    # ── data ─────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Re-read the trial folders and the workbook's trial list."""
        self._in_excel = set(xl.exported_folders(xl.DEFAULT_XLSX))
        rows = []
        for d in xl.all_trial_dirs():
            try:
                tj = json.loads((d / "trial.json").read_text())
            except Exception:
                continue
            st = tj.get("Stage Times") if isinstance(tj.get("Stage Times"), dict) else {}
            rows.append({"folder": str(d), "id": str(tj.get("Trial ID") or ""), "date": str(tj.get("Date") or ""),
                         "scen": str(tj.get("Scenario") or "").split(":")[0], "cmd": str(tj.get("Command Used") or ""),
                         "full": str(tj.get("Full Task Success?") or ""),
                         "times": "final" if st.get("event_times") else ("⏳ preliminary" if st else "—")})
        rows.sort(key=lambda r: (r["date"], r["id"], r["folder"]), reverse=True)
        days = ["All days"] + sorted({r["date"] for r in rows if r["date"]}, reverse=True)
        cur = self.day.currentText()
        self.day.blockSignals(True)
        self.day.clear()
        self.day.addItems(days)
        if cur in days:
            self.day.setCurrentText(cur)
        self.day.blockSignals(False)
        self.tree.blockSignals(True)
        self.tree.clear()
        for r in rows:
            it = QtWidgets.QTreeWidgetItem(["", r["id"], r["date"], r["scen"], r["cmd"], r["full"], r["times"],
                                            "✓" if r["folder"] in self._in_excel else ""])
            it.setData(0, QtCore.Qt.UserRole, r["folder"])
            it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
            ticked = self._pending.get(r["folder"], r["folder"] in self._in_excel)
            it.setCheckState(0, QtCore.Qt.Checked if ticked else QtCore.Qt.Unchecked)
            it.setToolTip(4, r["folder"])
            if r["times"].startswith("⏳"):
                it.setToolTip(6, "The trial's post-processing has not finished yet (about 10-30 s after ⏹).\n"
                                 "Sending now uses the window's own times; send again later for the exact ones.")
            self.tree.addTopLevelItem(it)
        self.tree.blockSignals(False)
        self._apply_filter()
        self._update_status()

    def _items(self):
        return [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]

    def _ticked(self) -> list:
        return [it.data(0, QtCore.Qt.UserRole) for it in self._items() if it.checkState(0) == QtCore.Qt.Checked]

    # ── UI helpers ───────────────────────────────────────────────────────
    def _apply_filter(self) -> None:
        day = self.day.currentText()
        only_new = self.only_new.isChecked()
        for it in self._items():
            hide = (day not in ("", "All days") and it.text(2) != day) or \
                   (only_new and it.data(0, QtCore.Qt.UserRole) in self._in_excel)
            it.setHidden(hide)

    def _on_item_changed(self, it, col) -> None:
        if col == 0:
            self._pending[it.data(0, QtCore.Qt.UserRole)] = it.checkState(0) == QtCore.Qt.Checked
            self._update_status()

    def _tick_shown(self, on: bool) -> None:
        for it in self._items():
            if not it.isHidden():
                it.setCheckState(0, QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)

    def _tick_new(self) -> None:
        for it in self._items():
            if it.data(0, QtCore.Qt.UserRole) not in self._in_excel:
                it.setCheckState(0, QtCore.Qt.Checked)

    def _update_status(self, extra: str = "") -> None:
        if self._running:
            return
        ticked = set(self._ticked())
        add = len(ticked - self._in_excel)
        rem = len(self._in_excel - ticked)
        msg = f"{len(self._in_excel)} trial(s) in the Excel file · {len(ticked)} ticked"
        if add or rem:
            msg += f" — the next send adds {add} and removes {rem}"
        self.status.setText(msg + (("\n" + extra) if extra else ""))
        busy = bool(self._is_busy())
        self.send_btn.setEnabled(not busy and not self._running)
        self.send_btn.setToolTip("Not while a trial is recording." if busy else self.send_btn.toolTip())

    def _tick_timer(self) -> None:
        if self.isVisible() and not self._running:
            self.refresh()

    @staticmethod
    def _open(path: str) -> None:
        if os.path.exists(path):
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)

    # ── send ─────────────────────────────────────────────────────────────
    def _send(self) -> None:
        if self._running or self._is_busy():
            return
        ticked = self._ticked()
        removed = sorted(self._in_excel - set(ticked))
        if removed:
            names = ", ".join(os.path.basename(p) for p in removed[:6]) + (" …" if len(removed) > 6 else "")
            if QtWidgets.QMessageBox.question(
                    self, "Remove trials from the Excel file?",
                    f"{len(removed)} trial(s) in the Excel file are unticked and will be removed from it:\n{names}\n\n"
                    "Their folders are not touched. Continue?") != QtWidgets.QMessageBox.Yes:
                return
        lock = Path(xl.DEFAULT_XLSX).with_name(".~lock." + Path(xl.DEFAULT_XLSX).name + "#")
        self._running = True
        self.send_btn.setEnabled(False)
        self.status.setText(f"⏳ writing {len(ticked)} trial(s) to Excel and computing the summary …")

        def work():
            try:
                res = xl.build(ticked, xl.DEFAULT_XLSX)
                res.update(xl.recalc(xl.DEFAULT_XLSX))
                res["open_elsewhere"] = lock.exists()
            except Exception as exc:  # noqa: BLE001 - report, never crash the GUI
                res = {"error": f"{type(exc).__name__}: {exc}"}
            self._done.finished.emit(res)
        threading.Thread(target=work, daemon=True).start()

    def _on_done(self, res: dict) -> None:
        self._running = False
        if res.get("error"):
            self.refresh()
            self._update_status(f"✖ Excel export failed: {res['error']}")
            return
        self._pending.clear()
        self.refresh()
        extra = f"✔ {res.get('trials')} trial(s) written to {os.path.basename(res.get('path', ''))}"
        if res.get("recalculated"):
            extra += f" · summary computed, {res.get('formula_errors', 0)} formula error(s)"
        else:
            extra += f" · summary not pre-computed ({res.get('why')}); Excel computes it on opening"
        if res.get("open_elsewhere"):
            extra += " · the file is open in LibreOffice — use File → Reload to see the update"
        self._update_status(extra)
