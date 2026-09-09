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

class ManagedLaunch(QtCore.QObject):
    """A ros2 launch subprocess with escalating stop.

    Ctrl-C alone is not enough for this stack: move_group segfaults in its
    destructor on Humble shutdown and `ros2 launch` then ignores further
    SIGINTs forever (observed live 2026-08-07). Stop here escalates
    SIGINT → 6 s → SIGTERM (group) → 4 s → SIGKILL (group).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.proc: Optional[subprocess.Popen] = None
        self._adopted_pid: Optional[int] = None
        self._log_f = None
        self._stop_stage = 0
        self._stop_timer = QtCore.QTimer(self)
        self._stop_timer.setSingleShot(True)
        self._stop_timer.timeout.connect(self._escalate)

    def running(self) -> bool:
        if self.proc is not None and self.proc.poll() is None:
            return True
        if self._adopted_pid is not None:
            try:
                os.kill(self._adopted_pid, 0)
                return True
            except OSError:
                self._adopted_pid = None
        return False

    def adopt(self, pid: int) -> None:
        """Take ownership of a launch started by a previous GUI session,
        so Stop works and Start can't double-launch the stack."""
        if not self.running():
            self._adopted_pid = int(pid)
            self._stop_stage = 0

    def start(self, cmd, env=None, log_path=None) -> None:
        if not self.running():
            # A stale escalation timer from the PREVIOUS stop would SIGKILL
            # the process we are about to start (killed the fresh executor
            # 1 s into startup, observed live 2026-08-26 15:30).
            self._stop_timer.stop()
            if log_path:
                # Child writes DIRECTLY to the file (a piped tee would
                # SIGPIPE-kill adopted children if the GUI ever dies); the
                # GUI tails the file into its own terminal so Majd keeps
                # his live console view (2026-08-12).
                self._log_f = open(log_path, "wb")
                self.proc = subprocess.Popen(
                    cmd, start_new_session=True, env=env,
                    stdout=self._log_f, stderr=subprocess.STDOUT,
                )
                threading.Thread(
                    target=self._tail_to_console,
                    args=(log_path, self.proc),
                    daemon=True,
                ).start()
            else:
                self.proc = subprocess.Popen(
                    cmd, start_new_session=True, env=env,
                )
            self._adopted_pid = None
            self._stop_stage = 0

    @staticmethod
    def _tail_to_console(log_path, proc) -> None:
        """Echo the growing log file to the GUI's stdout (like tail -f)."""
        try:
            with open(log_path, "rb") as f:
                while True:
                    line = f.readline()
                    if line:
                        try:
                            sys.stdout.buffer.write(line)
                            sys.stdout.buffer.flush()
                        except Exception:
                            return
                    else:
                        if proc.poll() is not None:
                            return
                        time.sleep(0.2)
        except Exception:
            pass

    def stop(self) -> None:
        if not self.running():
            return
        self._signal(signal.SIGINT)
        self._stop_stage = 1
        self._stop_timer.start(6000)

    def _escalate(self) -> None:
        if not self.running():
            return
        if self._stop_stage == 0:
            # A start() happened since the stop that armed this timer —
            # the running process is the NEW one, never kill it.
            return
        if self._stop_stage == 1:
            self._signal(signal.SIGTERM)
            self._stop_stage = 2
            self._stop_timer.start(4000)
        else:
            self._signal(signal.SIGKILL)

    def _signal(self, sig) -> None:
        pid = self.proc.pid if self.proc is not None else self._adopted_pid
        if pid is None:
            return
        try:
            os.killpg(os.getpgid(pid), sig)
        except Exception:
            pass

    def reap(self) -> Optional[int]:
        """Return the exit code once, when the process has ended."""
        if self.proc is not None and self.proc.poll() is not None:
            rc = self.proc.poll()
            self.proc = None
            self._stop_timer.stop()
            if self._log_f is not None:
                try:
                    self._log_f.close()
                except Exception:
                    pass
                self._log_f = None
            return rc
        if self.proc is None and self._adopted_pid is not None:
            try:
                os.kill(self._adopted_pid, 0)
            except OSError:
                self._adopted_pid = None
                self._stop_timer.stop()
                return -1  # adopted process ended; exit code unknown
        return None


