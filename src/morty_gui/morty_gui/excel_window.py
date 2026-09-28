"""The MMHRI window's Excel export (2026-09-28, Majd: "a button inside the
experiments window, a selection of the experiments to send, and the data
goes directly into the sheet; the selection updates after every
experiment" — and later: "don't update the Excel when I send it, what if I
changed stuff inside; and a button just for fast sending the last record").

Sending only ADDS: excel_export.append writes rows for the ticked trials that
are not in the workbook yet and never touches a row, column, sheet or format
that is already there. What is "in the workbook" is read from its
Experiments sheet, so a row deleted there can be sent again. A trial's times
are final ~10-30 s after ⏹ (the post-processing); a row is never updated
later, so the sender WAITS for the final times (at most 90 s) before adding.
While LibreOffice has the file open nothing is written — its next save would
overwrite the rows, or the rows would overwrite unsaved edits.

ExcelSender is the one background worker, used by this window and by the
trial window's "📊 Send last trial" button.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

from PyQt5 import QtCore, QtWidgets

from . import excel_export as xl

FINAL_WAIT_S = 90.0


def _times_final(folder: str) -> bool:
    try:
        st = json.loads((Path(folder) / "trial.json").read_text()).get("Stage Times") or {}
        return bool(st.get("event_times"))
    except Exception:
        return False


class ExcelSender(QtCore.QObject):
    """Adds trials to the workbook on a worker thread; `progress` and
    `finished` arrive in the GUI thread (Qt queued signals)."""
    progress = QtCore.pyqtSignal(str)
    finished = QtCore.pyqtSignal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.running = False

    def send(self, folders: list) -> bool:
        if self.running or not folders:
            return False
        self.running = True
        threading.Thread(target=self._work, args=(list(folders),), daemon=True).start()
        return True

    def _work(self, folders: list) -> None:
        try:
            if xl._lock_path(xl.DEFAULT_XLSX).exists():
                self._finish({"error": "open", "why": "The Excel file is open in LibreOffice — save and close it, then send again."})
                return
            t_end = time.time() + FINAL_WAIT_S
            pending = [f for f in folders if not _times_final(f)]
            while pending and time.time() < t_end:
                self.progress.emit(f"⏳ waiting for the final times of {', '.join(os.path.basename(p) for p in pending)} "
                                   f"(post-processing, about 10-30 s after ⏹) …")
                time.sleep(2.0)
                pending = [f for f in folders if not _times_final(f)]
            self.progress.emit(f"⏳ adding {len(folders)} trial(s) to Excel …")
            res = xl.append(folders, xl.DEFAULT_XLSX)
            if not res.get("error") and res.get("added"):
                res.update(xl.recalc(xl.DEFAULT_XLSX))
            res["still_preliminary"] = [os.path.basename(p) for p in pending]
        except Exception as exc:  # noqa: BLE001 - report, never crash the GUI
            res = {"error": "failed", "why": f"{type(exc).__name__}: {exc}"}
        self._finish(res)

    def _finish(self, res: dict) -> None:
        self.running = False          # before the signal: its handler may send again
        self.finished.emit(res)

    @staticmethod
    def describe(res: dict) -> str:
        if res.get("error"):
            return "✖ " + str(res.get("why") or res["error"])
        if not res.get("added"):
            return "✔ nothing to add — " + str(res.get("note") or "already in the Excel file")
        msg = f"✔ added {res['added']} trial(s) to {os.path.basename(res.get('path', xl.DEFAULT_XLSX))}"
        if res.get("created"):
            msg += " (new file)"
        if res.get("recalculated"):
            msg += f" · summary updated, {res.get('formula_errors', 0)} formula error(s)"
        if res.get("still_preliminary"):
            msg += f" · NOTE: {', '.join(res['still_preliminary'])} added with preliminary times (post-processing did not finish)"
        return msg


class ExcelExportWindow(QtWidgets.QWidget):
    COLS = ["", "Trial", "Date", "Scenario", "Command", "Full success", "Times", "In Excel"]

    def __init__(self, parent=None, is_busy=lambda: False) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("MMHRI — Excel export")
        self.resize(980, 640)
        self._is_busy = is_busy
        self._ticked: set = set()          # folders the user ticked (not yet in Excel)
        self._in_excel: set = set()
        self.sender_ = ExcelSender(self)
        self.sender_.progress.connect(lambda t: self.status.setText(t))
        self.sender_.finished.connect(self._on_done)
        v = QtWidgets.QVBoxLayout(self)

        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Excel file:"))
        self.path_lbl = QtWidgets.QLabel(xl.DEFAULT_XLSX)
        self.path_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        top.addWidget(self.path_lbl, 1)
        open_btn = QtWidgets.QPushButton("Open Excel")
        open_btn.clicked.connect(lambda: self._open(xl.DEFAULT_XLSX))
        top.addWidget(open_btn)
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
                         ("Untick all shown", lambda: self._tick_shown(False))):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            btns.addWidget(b)
        btns.addStretch(1)
        self.send_btn = QtWidgets.QPushButton("📊 Add ticked to Excel")
        self.send_btn.setStyleSheet("QPushButton {background:#1f6f43; color:white; font-weight:bold; padding:4px 14px;}")
        self.send_btn.clicked.connect(self._send)
        btns.addWidget(self.send_btn)
        v.addLayout(btns)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#888;")
        v.addWidget(self.status)

        self.tree.setToolTip("Tick trials to ADD to the Excel file. Trials already in it are marked ✓ and never\n"
                             "changed by a send — edit them in Excel. Delete a row there to be able to send it again.")
        self.send_btn.setToolTip("Adds the ticked trials that are not in the Excel file yet — nothing already in the\n"
                                 "file is changed. Waits for final times first. Not while a trial is recording,\n"
                                 "and not while the file is open in LibreOffice.")
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._tick_timer)
        self._timer.start(10000)
        self.refresh()

    # ── data ─────────────────────────────────────────────────────────────
    def refresh(self) -> None:
        """Re-read the trial folders and what the workbook holds."""
        keys = xl.workbook_keys(xl.DEFAULT_XLSX)
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
                         "times": "final" if st.get("event_times") else ("⏳ preliminary" if st else "—"),
                         "in": xl.in_workbook(str(d), keys)})
        self._in_excel = {r["folder"] for r in rows if r["in"]}
        self._ticked -= self._in_excel
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
                                            "✓" if r["in"] else ""])
            it.setData(0, QtCore.Qt.UserRole, r["folder"])
            if r["in"]:
                it.setFlags(it.flags() & ~QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(0, QtCore.Qt.Checked)
                for c in range(1, len(self.COLS)):
                    it.setForeground(c, QtCore.Qt.gray)
                it.setToolTip(0, "Already in the Excel file — a send never changes it.")
            else:
                it.setFlags(it.flags() | QtCore.Qt.ItemIsUserCheckable)
                it.setCheckState(0, QtCore.Qt.Checked if r["folder"] in self._ticked else QtCore.Qt.Unchecked)
            it.setToolTip(4, r["folder"])
            self.tree.addTopLevelItem(it)
        self.tree.blockSignals(False)
        self._apply_filter()
        self._update_status()

    def _items(self):
        return [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())]

    # ── UI helpers ───────────────────────────────────────────────────────
    def _apply_filter(self) -> None:
        day = self.day.currentText()
        only_new = self.only_new.isChecked()
        for it in self._items():
            hide = (day not in ("", "All days") and it.text(2) != day) or \
                   (only_new and it.data(0, QtCore.Qt.UserRole) in self._in_excel)
            it.setHidden(hide)

    def _on_item_changed(self, it, col) -> None:
        f = it.data(0, QtCore.Qt.UserRole)
        if col == 0 and f not in self._in_excel:
            (self._ticked.add if it.checkState(0) == QtCore.Qt.Checked else self._ticked.discard)(f)
            self._update_status()

    def _tick_shown(self, on: bool) -> None:
        for it in self._items():
            if not it.isHidden() and it.data(0, QtCore.Qt.UserRole) not in self._in_excel:
                it.setCheckState(0, QtCore.Qt.Checked if on else QtCore.Qt.Unchecked)

    def _update_status(self, extra: str = "") -> None:
        if self.sender_.running:
            return
        msg = f"{len(self._in_excel)} trial(s) in the Excel file · {len(self._ticked)} ticked to add"
        self.status.setText(msg + (("\n" + extra) if extra else ""))
        busy = bool(self._is_busy())
        self.send_btn.setEnabled(not busy and bool(self._ticked))

    def _tick_timer(self) -> None:
        if self.isVisible() and not self.sender_.running:
            self.refresh()

    @staticmethod
    def _open(path: str) -> None:
        if os.path.exists(path):
            subprocess.Popen(["xdg-open", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             start_new_session=True)

    # ── send ─────────────────────────────────────────────────────────────
    def _send(self) -> None:
        if self._is_busy() or not self._ticked:
            return
        folders = sorted(self._ticked)
        if self.sender_.send(folders):
            self.send_btn.setEnabled(False)

    def _on_done(self, res: dict) -> None:
        if not res.get("error"):
            self._ticked.clear()
        self.refresh()
        self._update_status(ExcelSender.describe(res))
