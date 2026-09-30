"""Sound popup (2026-09-30, Majd: "slides that correspond with the slides of
Ubuntu"): the speaker and microphone sliders of the Ubuntu top bar, in a
small window under a button of the Morty panel.

Both act on the SAME thing Ubuntu's sliders do — the volume and mute of
PulseAudio's default sink and default source (@DEFAULT_SINK@ here is
echocancel_spk, @DEFAULT_SOURCE@ the wireless mic, the canceller's master).
While the popup is open it re-reads them every 0.5 s, so moving a slider
in Ubuntu moves it here and the other way round. The range is 0–100 %
like Ubuntu's (a level above 100 % set elsewhere shows as 100 % and is
only changed when the slider is moved).

This is the SYSTEM level. Morty's own voice mute (/speech/voice_muted) and
the speech node's mic mute (/speech/muted) are the panel's buttons."""
from __future__ import annotations

import re
import subprocess
from typing import Optional, Tuple

from PyQt5 import QtCore, QtWidgets

_PCT = re.compile(r"(\d+)%")


def _pactl(*args: str) -> str:
    try:
        return subprocess.run(["pactl", *args], capture_output=True, text=True, timeout=1.5).stdout
    except Exception:
        return ""


def read_level(kind: str) -> Optional[Tuple[int, bool]]:
    """(volume %, muted) of the default sink / source, None if pactl fails.
    The volume is the channel average, as Ubuntu shows it."""
    dev = f"@DEFAULT_{kind.upper()}@"
    vols = [int(v) for v in _PCT.findall(_pactl(f"get-{kind}-volume", dev))]
    mute = _pactl(f"get-{kind}-mute", dev)
    if not vols or "Mute:" not in mute:
        return None
    return round(sum(vols) / len(vols)), "yes" in mute


def set_volume(kind: str, pct: int) -> None:
    _pactl(f"set-{kind}-volume", f"@DEFAULT_{kind.upper()}@", f"{int(pct)}%")


def set_mute(kind: str, muted: bool) -> None:
    _pactl(f"set-{kind}-mute", f"@DEFAULT_{kind.upper()}@", "1" if muted else "0")


class _Row(QtWidgets.QWidget):
    """One Ubuntu-style row: icon button (click = mute), slider, percent."""

    ICONS = {"sink": ("🔊", "🔇"), "source": ("🎤", "🚫")}
    TIPS = {"sink": "Speaker (system output — Ubuntu's speaker slider)",
            "source": "Microphone (system input — Ubuntu's microphone slider)"}

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self._muted = False
        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.icon = QtWidgets.QToolButton()
        self.icon.setAutoRaise(True)
        self.icon.setStyleSheet("font-size:18px;")
        self.icon.setToolTip(self.TIPS[kind] + " — click to mute / unmute")
        self.icon.clicked.connect(self._toggle_mute)
        lay.addWidget(self.icon)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setMinimumWidth(220)
        self.slider.setToolTip(self.TIPS[kind])
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal{height:6px;background:#555;border-radius:3px;}"
            "QSlider::sub-page:horizontal{background:#4e9a06;border-radius:3px;}"
            "QSlider::handle:horizontal{background:#eee;width:16px;margin:-6px 0;border-radius:8px;}")
        self.slider.valueChanged.connect(self._moved)
        lay.addWidget(self.slider, stretch=1)
        self.pct = QtWidgets.QLabel("—")
        self.pct.setMinimumWidth(40)
        self.pct.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lay.addWidget(self.pct)
        self._set_icon()

    def _set_icon(self) -> None:
        self.icon.setText(self.ICONS[self.kind][1 if self._muted else 0])
        self.pct.setStyleSheet("color:#888;" if self._muted else "")

    def _moved(self, v: int) -> None:
        self.pct.setText(f"{v} %")
        set_volume(self.kind, v)          # refresh() blocks signals, so this is always the operator

    def _toggle_mute(self) -> None:
        self._muted = not self._muted
        set_mute(self.kind, self._muted)
        self._set_icon()

    def refresh(self) -> None:
        """Take the system's value unless the operator is dragging."""
        lvl = read_level(self.kind)
        if lvl is None:
            self.pct.setText("n/a")
            return
        vol, muted = lvl
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setValue(min(vol, 100))
            self.slider.blockSignals(False)
            self.pct.setText(f"{vol} %")
        if muted != self._muted:
            self._muted = muted
            self._set_icon()


class AudioPopup(QtWidgets.QFrame):
    """The small window: speaker row, microphone row. Opens under a button
    (show_under), closes on a click outside like Ubuntu's menu."""

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Popup)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setStyleSheet("AudioPopup{background:#2b2b2b;border:1px solid #555;border-radius:10px;}"
                           "QLabel{color:#ddd;}")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        title = QtWidgets.QLabel("System sound (same as Ubuntu's sliders)")
        title.setStyleSheet("color:#aaa; font-size:11px;")
        lay.addWidget(title)
        self.speaker = _Row("sink")
        self.mic = _Row("source")
        lay.addWidget(self.speaker)
        lay.addWidget(self.mic)
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self.refresh)

    def refresh(self) -> None:
        self.speaker.refresh()
        self.mic.refresh()

    def show_under(self, w: QtWidgets.QWidget) -> None:
        self.refresh()
        self.adjustSize()
        pos = w.mapToGlobal(QtCore.QPoint(w.width() - self.width(), w.height() + 2))
        self.move(pos)
        self.show()
        self._timer.start()

    def hideEvent(self, e) -> None:
        self._timer.stop()
        super().hideEvent(e)
