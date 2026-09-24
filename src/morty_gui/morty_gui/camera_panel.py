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

class CameraPanel(QtWidgets.QLabel):
    """Aspect-fit video panel; clicks are mapped back to image pixels."""

    clicked_px = QtCore.pyqtSignal(float, float)
    hovered_px = QtCore.pyqtSignal(float, float)   # mouse over the image, throttled (2026-09-18)

    def __init__(self) -> None:
        super().__init__("waiting for the camera (start the driver with realsense) ...")
        self.setMouseTracking(True)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background:#141418; color:#888;")
        self._img_w = 0
        self._img_h = 0
        self._draw_rect = QtCore.QRect()

    def show_bgr(self, bgr: np.ndarray) -> None:
        h, w = bgr.shape[:2]
        self._img_w, self._img_h = w, h
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        pm = QtGui.QPixmap.fromImage(qimg).scaled(
            self.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self._draw_rect = QtCore.QRect(
            (self.width() - pm.width()) // 2, (self.height() - pm.height()) // 2,
            pm.width(), pm.height(),
        )
        self.setPixmap(pm)

    def mouseMoveEvent(self, ev: QtGui.QMouseEvent) -> None:
        # The object under the mouse is the one about to be clicked: the
        # pipeline's precompute puts it first in its queue (hover priority).
        if self._img_w and self._draw_rect.width() and self._draw_rect.contains(ev.pos()):
            import time as _t
            now = _t.monotonic()
            if now - getattr(self, "_last_hover_t", 0.0) >= 0.15:
                self._last_hover_t = now
                fx = (ev.x() - self._draw_rect.x()) / self._draw_rect.width()
                fy = (ev.y() - self._draw_rect.y()) / self._draw_rect.height()
                self.hovered_px.emit(fx * self._img_w, fy * self._img_h)
        super().mouseMoveEvent(ev)

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton and self._img_w and self._draw_rect.width():
            if self._draw_rect.contains(ev.pos()):
                fx = (ev.x() - self._draw_rect.x()) / self._draw_rect.width()
                fy = (ev.y() - self._draw_rect.y()) / self._draw_rect.height()
                self.clicked_px.emit(fx * self._img_w, fy * self._img_h)
        super().mousePressEvent(ev)



CAMERAS = ("RealSense", "Astra")


class CameraWindow(QtWidgets.QWidget):
    """One camera in its own top-level window, with its own RealSense /
    Astra selector (the main panel keeps its own choice)."""

    closed = QtCore.pyqtSignal(object)          # the window
    camera_changed = QtCore.pyqtSignal()

    def __init__(self, camera: str, parent=None) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        top = QtWidgets.QHBoxLayout()
        self.combo = QtWidgets.QComboBox()
        self.combo.addItems([f"view: {c}" for c in CAMERAS])
        self.combo.setCurrentIndex(CAMERAS.index(camera) if camera in CAMERAS else 0)
        self.combo.setToolTip("This window's camera. The main window keeps its own choice.")
        top.addWidget(self.combo)
        top.addStretch(1)
        lay.addLayout(top)
        self.panel = CameraPanel()
        self.panel.setMinimumSize(320, 240)
        lay.addWidget(self.panel, 1)
        self.combo.currentIndexChanged.connect(self._on_combo)
        self._show_camera_name()
        self.resize(720, 560)

    @property
    def camera(self) -> str:
        return CAMERAS[self.combo.currentIndex()]

    def _show_camera_name(self) -> None:
        self.setWindowTitle(f"Morty camera — {self.camera}")
        self.panel.setText(f"waiting for the {self.camera} ...")

    def _on_combo(self, _i: int) -> None:
        self._show_camera_name()
        self.camera_changed.emit()

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        self.closed.emit(self)
        super().closeEvent(ev)


class CameraWindows(QtCore.QObject):
    """Owner of the pop-out camera windows (2026-09-24, Majd: "sometimes I
    want to see both cameras at the same time").

    Frames reach the windows through feed() from the main window's poll —
    the frames the main panel already gets, so no extra ROS subscription.
    The Astra JPEG is decoded only while some view shows it, so this object
    tells the bridge whether a window wants it (want_astra "windows"; the
    main panel is the "main" viewer). A click in a RealSense window is an
    object selection, exactly like a click on the main panel; an Astra
    window only shows. Which windows were open, and where, is remembered:
    they come back at the next GUI start."""

    object_clicked = QtCore.pyqtSignal(float, float)   # RealSense pixel
    hovered = QtCore.pyqtSignal(float, float)          # RealSense pixel
    message = QtCore.pyqtSignal(str)                   # for the status bar

    def __init__(self, bridge, settings: QtCore.QSettings, parent=None) -> None:
        super().__init__(parent)
        self._bridge = bridge
        self._settings = settings
        self._windows: List[CameraWindow] = []

    def open(self, camera: str) -> CameraWindow:
        # Top-level like the Advanced and MMHRI windows: its own taskbar
        # entry, free to sit on another screen or behind the main window.
        w = CameraWindow(camera)
        w.panel.clicked_px.connect(lambda x, y, w=w: self._on_click(w, x, y))
        w.panel.hovered_px.connect(
            lambda x, y, w=w: self.hovered.emit(x, y) if w.camera == "RealSense" else None)
        w.camera_changed.connect(self._sync_astra)
        w.closed.connect(self._on_closed)
        geo = self._settings.value(f"camera_window/{camera}/geometry")
        if geo is not None:
            w.restoreGeometry(geo)
        self._windows.append(w)
        self._sync_astra()
        w.show()
        return w

    def feed(self, camera: str, bgr: np.ndarray) -> None:
        for w in self._windows:
            if w.camera == camera:
                w.panel.show_bgr(bgr)

    def restore(self) -> None:
        """Reopen the windows that were open when the GUI last closed."""
        for camera in self._settings.value("camera_window/open", [], type=list) or []:
            if camera in CAMERAS:
                self.open(camera)

    def close_all(self) -> None:
        """At GUI exit: remember which windows are open, then close them."""
        self._settings.setValue("camera_window/open", [w.camera for w in self._windows])
        for w in list(self._windows):
            w.close()

    def _on_click(self, w: CameraWindow, x_px: float, y_px: float) -> None:
        if w.camera == "RealSense":
            self.object_clicked.emit(x_px, y_px)
        else:
            self.message.emit("That window shows the Astra — clicking an object works in a RealSense view.")

    def _on_closed(self, w: CameraWindow) -> None:
        self._settings.setValue(f"camera_window/{w.camera}/geometry", w.saveGeometry())
        if w in self._windows:
            self._windows.remove(w)
        self._sync_astra()

    def _sync_astra(self) -> None:
        self._bridge.want_astra("windows", any(w.camera == "Astra" for w in self._windows))
