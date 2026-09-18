"""The MMHRI trial recorder's handle on the VLA episode recorder.

One trial, one episode. The trial window says begin / end / abort; this
object owns the recorder process and speaks its ``/vla_recorder/cmd``
protocol, so the window never touches ROS topics or subprocesses.

Why the episode is not simply "record while the trial runs": the two
records answer different questions. The MMHRI folder documents the
INTERACTION — including trials where the robot must not move (S1 "what do
you see", S2 an absent object). A VLA episode is TRAINING DATA and is only
worth keeping if the arm moved. So the episode opens PROVISIONAL and the
recorder confirms it from the robot itself — an arm joint moving past a
threshold (or a motion intent, whichever is first). A trial that ends
without motion is discarded on the recorder side; nothing here has to
know which scenario it was. (Majd, 2026-09-03: "we don't need to record
when we ask for objects, we need when we move the robot.")

The recorder is launched WITH the experiments window so it is warm by the
first ⏺ — no terminal command to remember. A recorder started elsewhere
(a terminal, a previous GUI) is adopted, never duplicated.
"""
from __future__ import annotations

import os
import subprocess
from typing import Optional

from PyQt5 import QtCore

from .managed_launch import ManagedLaunch

LAUNCH = ["ros2", "launch", "vla_experiments", "vla_record.launch.py",
          "auto_mode:=false"]          # trial-driven, never transcript-driven
NODE = "vla_episode_recorder"
LOG = os.path.expanduser("~/morty_ws/logs/vla_gui.log")
# `[y]` keeps pgrep from matching its own command line (see the
# pkill self-match note in the project memory).
PGREP = r"vla_record\.launch\.p[y]"


def _pgrep_oldest(pattern: str) -> Optional[int]:
    try:
        out = subprocess.run(["pgrep", "-o", "-f", pattern],
                             capture_output=True, text=True, timeout=2.0).stdout.strip()
        return int(out.splitlines()[0]) if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


class VlaLink(QtCore.QObject):
    def __init__(self, bridge, parent=None) -> None:
        self.last_episode: Optional[int] = None   # index of the episode end() closed
        super().__init__(parent)
        self._bridge = bridge          # may be None (window opened standalone)
        self._proc = ManagedLaunch(self)

    # ── recorder process ─────────────────────────────────────────────
    def node_up(self) -> bool:
        """Is a recorder node in the ROS graph — ours or anyone's."""
        return bool(self._bridge is not None and self._bridge.vla_node_up())

    def set_enabled(self, on: bool) -> None:
        """Bring the recorder up (or take ours down). Idempotent."""
        if not on:
            if self._proc.running():
                self._proc.stop()
            return
        if self._proc.running():
            return
        pid = _pgrep_oldest(PGREP)
        if pid is not None:            # a previous GUI's recorder: own it
            self._proc.adopt(pid)
            return
        if self.node_up():             # someone's terminal: use, don't own
            return
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        self._proc.start(LAUNCH, env=os.environ.copy(), log_path=LOG)

    def shutdown(self) -> None:
        """Stop the recorder only if this GUI is the one running it."""
        if self._proc.running():
            self._proc.stop()

    # ── episode lifecycle (mirrors the trial's ⏺ / ⏹ / ✖) ────────────
    def begin(self, fallback_task: str, trial_hint: str = "") -> bool:
        """Open a provisional episode. False = no recorder to talk to.
        ``trial_hint`` is the trial folder (relative to the experiments
        root) the episode's meta.json will name."""
        if self._bridge is None or not self.node_up():
            return False
        self.last_episode = None
        if trial_hint:
            self._bridge.vla_cmd(f"hint {trial_hint}")
        self._bridge.vla_cmd(f"trial {fallback_task}")
        return True

    def end(self, success: Optional[bool]) -> None:
        """Close the episode. None = no human verdict, let the executor's
        cycle_done decide (the recorder discards it anyway if the arm
        never moved)."""
        if self._bridge is not None:
            # Read the index BEFORE the stop lands: the recorder's status
            # says "episode: None" once it is no longer recording. The
            # trial's trial.json names this episode (2026-09-18).
            st = self._bridge.vla_status() or {}
            self.last_episode = st.get("episode") if st.get("recording") else None
            self._bridge.vla_cmd(
                "stop " + {True: "ok", False: "fail", None: "auto"}[success])

    def abort(self) -> None:
        if self._bridge is not None:
            self._bridge.vla_cmd("discard")
