"""✎ fix trial — correct a saved trial's form afterwards (2026-10-01).

Majd: "when I press Stop & save I forget to change what I wrote, like S1 or
S2 …". The dialog shows the trial saved last (or any other of the list)
with the SAME fields as the trial window, filled from its trial.json; the
fields you change are marked, "Apply" lists old -> new once more and then
fix_trial.fix_trial rewrites the folder name, trial.json, the VLA outcome,
both CSVs and the workbook row. The work runs off the GUI thread (the
workbook's formula recalculation takes a few seconds)."""
from __future__ import annotations

import os
import threading
from typing import Dict, List

from PyQt5 import QtCore, QtWidgets

from .fix_trial import EDITABLE, YESNO, changes_for, fix_trial, read_form
from .mmhri_window import LAYOUTS, MODIFIERS, OBJECTS, SCENARIOS

CHANGED_STYLE = "background:#fff3b0;"


class FixTrialDialog(QtWidgets.QDialog):
    fixed = QtCore.pyqtSignal(str)            # the trial folder after the fix
    _done = QtCore.pyqtSignal(object)         # worker -> GUI thread

    def __init__(self, folders: List[str], current: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("✎ Fix a saved trial")
        self.setMinimumWidth(620)
        self._folders = folders
        self._loaded: Dict[str, str] = {}
        lay = QtWidgets.QVBoxLayout(self)

        pick = QtWidgets.QHBoxLayout()
        pick.addWidget(QtWidgets.QLabel("Trial:"))
        self.trial_box = QtWidgets.QComboBox()
        for f in folders:
            self.trial_box.addItem(f"{os.path.basename(f)}   ({os.path.basename(os.path.dirname(f))})", f)
        pick.addWidget(self.trial_box, 1)
        lay.addLayout(pick)

        form = QtWidgets.QFormLayout()
        self.w: Dict[str, QtWidgets.QWidget] = {}
        self.w["Scene Layout"] = QtWidgets.QComboBox()
        self.w["Scene Layout"].addItems(LAYOUTS)
        self.w["Scenario"] = QtWidgets.QComboBox()
        self.w["Scenario"].addItems(SCENARIOS)
        mods = QtWidgets.QWidget()
        ml = QtWidgets.QHBoxLayout(mods)
        ml.setContentsMargins(0, 0, 0, 0)
        self.mod_boxes: Dict[str, QtWidgets.QCheckBox] = {}
        for mid, mtxt, _old, _integrated in MODIFIERS:
            cb = QtWidgets.QCheckBox(f"{mid} {mtxt.split(' — ')[0]}")
            self.mod_boxes[mid] = cb
            ml.addWidget(cb)
        ml.addStretch(1)
        self.w["Modifiers"] = mods
        self.w["Command Used"] = QtWidgets.QLineEdit()
        self.w["Intended Target"] = QtWidgets.QComboBox()
        self.w["Intended Target"].setEditable(True)
        self.w["Intended Target"].addItems(["—"] + OBJECTS)
        self.w["Intended Destination"] = QtWidgets.QComboBox()
        self.w["Intended Destination"].setEditable(True)
        self.w["Intended Destination"].addItems(["—", "drop_left", "drop_right", "home"])
        for k in YESNO:
            cb = QtWidgets.QComboBox()
            cb.addItems(["", "yes", "no"])
            self.w[k] = cb
        self.w["Failure Category / Notes"] = QtWidgets.QLineEdit()
        for k in EDITABLE:
            form.addRow(k.replace("Failure Category / Notes", "Failure / Notes") + ":", self.w[k])
        lay.addLayout(form)

        self.diff = QtWidgets.QLabel("")
        self.diff.setWordWrap(True)
        self.diff.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        lay.addWidget(self.diff)
        btns = QtWidgets.QHBoxLayout()
        btns.addStretch(1)
        self.revert_btn = QtWidgets.QPushButton("↺ revert")
        self.apply_btn = QtWidgets.QPushButton("✔ Apply corrections")
        close_btn = QtWidgets.QPushButton("Close")
        for b in (self.revert_btn, self.apply_btn, close_btn):
            btns.addWidget(b)
        lay.addLayout(btns)

        self.trial_box.currentIndexChanged.connect(lambda _i: self._load())
        self.revert_btn.clicked.connect(self._load)
        self.apply_btn.clicked.connect(self._apply)
        close_btn.clicked.connect(self.close)
        self._done.connect(self._on_done)
        for k, wd in self.w.items():
            if isinstance(wd, QtWidgets.QComboBox):
                wd.currentTextChanged.connect(lambda _t: self._show_changes())
            elif isinstance(wd, QtWidgets.QLineEdit):
                wd.textChanged.connect(lambda _t: self._show_changes())
        for cb in self.mod_boxes.values():
            cb.toggled.connect(lambda _on: self._show_changes())
        if current in folders:
            self.trial_box.setCurrentIndex(folders.index(current))
        self._load()

    # ── the form <-> values ──────────────────────────────────────────
    def _folder(self) -> str:
        return self.trial_box.currentData() or ""

    def _set(self, k: str, v: str) -> None:
        wd = self.w[k]
        if k == "Modifiers":
            ids = {x for x in v.split(";") if x}
            for mid, cb in self.mod_boxes.items():
                cb.setChecked(mid in ids)
        elif k == "Scenario":
            sid = v.split(":")[0].strip()
            idx = next((i for i, s in enumerate(SCENARIOS) if s.split(":")[0] == sid), -1)
            wd.setCurrentIndex(max(idx, 0))
        elif isinstance(wd, QtWidgets.QComboBox):
            if wd.isEditable():
                wd.setCurrentText(v)
            else:
                i = wd.findText(v)
                wd.setCurrentIndex(i if i >= 0 else 0)
        else:
            wd.setText(v)

    def _get(self, k: str) -> str:
        wd = self.w[k]
        if k == "Modifiers":
            return ";".join(mid for mid, cb in self.mod_boxes.items() if cb.isChecked())
        if isinstance(wd, QtWidgets.QComboBox):
            return wd.currentText().strip()
        return wd.text().strip()

    def _load(self) -> None:
        f = self._folder()
        if not f:
            self.diff.setText("no saved trial")
            self.apply_btn.setEnabled(False)
            return
        try:
            self._loaded = read_form(f)
        except Exception as exc:  # noqa: BLE001 — shown, not raised in the GUI
            self.diff.setText(f"✖ cannot read {f}: {exc}")
            self.apply_btn.setEnabled(False)
            return
        for k in EDITABLE:
            self._set(k, self._loaded.get(k, ""))
        self._show_changes()

    def _wanted(self) -> Dict[str, str]:
        return {k: self._get(k) for k in EDITABLE}

    def _show_changes(self) -> None:
        if not self._loaded:
            return
        ch = {}
        try:
            ch = changes_for(self._folder(), self._wanted())
        except Exception as exc:  # noqa: BLE001
            self.diff.setText(f"✖ {exc}")
        for k, wd in self.w.items():
            wd.setStyleSheet(CHANGED_STYLE if k in ch else "")
        self.apply_btn.setEnabled(bool(ch))
        self.revert_btn.setEnabled(bool(ch))
        if ch:
            self.diff.setText("Will change:\n" + "\n".join(
                f"  • {k}: {o or '(empty)'}  →  {n or '(empty)'}" for k, (o, n) in ch.items()))
        else:
            self.diff.setText("No changes yet — edit a field (changed fields turn yellow).")

    # ── apply ─────────────────────────────────────────────────────────
    def _apply(self) -> None:
        f, want = self._folder(), self._wanted()
        try:
            preview = fix_trial(f, want, dry_run=True)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.warning(self, "Fix trial", str(exc))
            return
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Fix trial — confirm")
        box.setText("\n".join(preview[:-1]))
        box.setInformativeText("The trial's folder, trial.json, CSV rows and its Excel row (if sent) are "
                               "corrected; backups go to ~/morty_ws/backups/.")
        box.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(QtWidgets.QMessageBox.Yes)
        if box.exec_() != QtWidgets.QMessageBox.Yes:
            return
        self.apply_btn.setEnabled(False)
        self.diff.setText("⏳ correcting … (the workbook's recalculation takes a few seconds)")

        def work():
            try:
                self._done.emit(("ok", f, fix_trial(f, want)))
            except Exception as exc:  # noqa: BLE001
                self._done.emit(("error", f, [f"{type(exc).__name__}: {exc}"]))
        threading.Thread(target=work, daemon=True).start()

    def _on_done(self, res) -> None:
        state, f, lines = res
        if state != "ok":
            self.diff.setText("✖ " + "\n".join(lines) + "\n(nothing or only part changed — see the backups folder)")
            self._show_changes()
            return
        new = next((ln.split(" -> ")[1] for ln in lines if ln.startswith("folder ") and " -> " in ln), "")
        new_f = os.path.join(os.path.dirname(f), new) if new else f
        i = self.trial_box.currentIndex()
        self._folders[i] = new_f
        self.trial_box.blockSignals(True)
        self.trial_box.setItemText(i, f"{os.path.basename(new_f)}   ({os.path.basename(os.path.dirname(new_f))})")
        self.trial_box.setItemData(i, new_f)
        self.trial_box.blockSignals(False)
        self._load()
        self.diff.setText("✔ corrected:\n" + "\n".join("  " + ln for ln in lines))
        self.fixed.emit(new_f)
