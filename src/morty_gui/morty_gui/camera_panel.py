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

    def __init__(self) -> None:
        super().__init__("waiting for /yolo/debug_image ...")
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

    def mousePressEvent(self, ev: QtGui.QMouseEvent) -> None:
        if ev.button() == QtCore.Qt.LeftButton and self._img_w and self._draw_rect.width():
            if self._draw_rect.contains(ev.pos()):
                fx = (ev.x() - self._draw_rect.x()) / self._draw_rect.width()
                fy = (ev.y() - self._draw_rect.y()) / self._draw_rect.height()
                self.clicked_px.emit(fx * self._img_w, fy * self._img_h)
        super().mousePressEvent(ev)


