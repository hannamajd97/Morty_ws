"""The MMHRI trial recorder's handle on a ROS bag.

One trial, one bag (Hamid, 2026-09-09: "record ROS bags during the
experiments — both camera streams, joint states, end-effector pose, gripper
state, robot commands, proper timestamps"). The trial window says begin /
end / abort; this class runs ``ros2 bag record`` for exactly that span.

What goes in (config.BAG_TOPICS): both colour streams compressed with their
camera infos, /joint_states, the TCP pose, the tool I/O states (gripper
holding / slip inputs), the trajectory controller's state (commanded vs
actual), speed scaling, the pendant program state, the voice transcript
and intents, the pipeline's detections and selection cycle, the executor's
selected pose / status / cycle_done, /tf and /tf_static. Every message
carries the ROS time it was published, so the bag replays with
``ros2 bag play`` and slices cleanly by time. Measured 2026-09-09: 18
topics, about 2.5 MB/s, i.e. under 100 MB for a 30 s demonstration.

Why a separate process: rosbag2's recorder is C++ and writes from its own
threads; keeping it out of the GUI process means the GUI cannot slow it
down and a GUI crash still leaves a closed, readable bag (the recorder
finalises on SIGINT). ManagedLaunch already does SIGINT → SIGTERM → SIGKILL
escalation with a log file, so it is reused as is.
"""
from __future__ import annotations

import os
import shutil
import time
from typing import Optional

from PyQt5 import QtCore

from .config import BAG_TOPICS
from .managed_launch import ManagedLaunch


class BagLink(QtCore.QObject):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._proc = ManagedLaunch(self)
        self.last_dir: str = ""        # bag folder of the current / last trial

    def running(self) -> bool:
        return self._proc.running()

    # ── lifecycle (mirrors the trial's ⏺ / ⏹ / ✖) ──────────────────────
    def begin(self, parent_dir: str, name: str = "bag") -> Optional[str]:
        """Start recording into ``parent_dir/name``. Returns the bag folder,
        or None when a recorder is already running."""
        if self._proc.running():
            return None
        os.makedirs(parent_dir, exist_ok=True)
        bag_dir = os.path.join(parent_dir, name)
        if os.path.exists(bag_dir):            # rosbag2 refuses an existing folder
            bag_dir = os.path.join(parent_dir, f"{name}_{time.strftime('%H%M%S')}")
        cmd = ["ros2", "bag", "record", "-o", bag_dir] + list(BAG_TOPICS)
        self._proc.start(cmd, env=os.environ.copy(),
                         log_path=os.path.join(parent_dir, "bag_record.log"),
                         die_with_parent=True)   # 2026-09-28: a dead GUI must not leave it recording (trial 081)
        self.last_dir = bag_dir
        return bag_dir

    def end(self) -> str:
        """Stop and keep. SIGINT makes rosbag2 flush its cache and write
        metadata.yaml; ManagedLaunch escalates if it ever hangs."""
        if self._proc.running():
            self._proc.stop()
        return self.last_dir

    def abort(self) -> None:
        """Stop and delete (trial discarded). Waits briefly for the recorder
        to exit so the folder is not removed underneath it."""
        if self._proc.running():
            self._proc.stop()
            for _ in range(50):
                if not self._proc.running():
                    break
                time.sleep(0.1)
        if self.last_dir and os.path.isdir(self.last_dir):
            shutil.rmtree(self.last_dir, ignore_errors=True)
        self.last_dir = ""

    def shutdown(self) -> None:
        if self._proc.running():
            self._proc.stop()

    @staticmethod
    def size_mb(bag_dir: str) -> float:
        total = 0
        for root, _dirs, files in os.walk(bag_dir):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total / 1e6
