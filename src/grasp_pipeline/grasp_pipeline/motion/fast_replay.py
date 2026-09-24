"""Popen-compatible adapter for fast-replay trajectory goals.

The legacy multi-view-sweep code in ``grasp_selection_node`` was written
against ``subprocess.Popen``: it polls, terminates, kills, waits, and
reads stdout/stderr. When we route the same observation moves through
``ur3e_fast_observation`` instead of a subprocess we still want that
calling code to compile unchanged. This handle satisfies the Popen
surface while internally tracking a single fast-replay goal by label.

The owning ``MotionDispatcher`` calls ``_finish(state, reason)`` from
its status callback to flip the handle into a terminal state and unblock
``wait()``. ``terminate()`` and ``kill()`` are routed back to the
dispatcher so the trajectory controller can be cancelled cleanly — the
trajectory controller has no SIGKILL-style escalation, hence both
methods do the same thing.
"""

from __future__ import annotations

import io
import subprocess
import threading
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:  # avoid circular import at runtime
    from .motion_dispatcher import MotionDispatcher


class FastReplayHandle:
    """subprocess.Popen-compatible adapter around a fast-replay goal.

    Public surface (matches subprocess.Popen): ``poll()``, ``wait()``,
    ``terminate()``, ``kill()``, ``returncode``, ``stdout``, ``stderr``.

    Internal: ``_finish(state, reason)`` is called from the dispatcher's
    fast-replay status callback to transition this handle to its terminal
    state and unblock any thread waiting on ``wait()``.
    """

    def __init__(
        self,
        owner: "MotionDispatcher",
        label: str,
        timeout_s: float,
    ) -> None:
        self._owner = owner
        self._label = label
        self._timeout_s = float(timeout_s)
        self._lock = threading.Lock()
        self._done_event = threading.Event()
        self._final_state: Optional[str] = None  # "done" | "failed" | "cancelled"
        # True once the replay node answered "starting:<label>" — it took
        # the request. A slow sweep that never gets this gives up early
        # instead of waiting its whole timeout (2026-09-24).
        self.started = False
        self._final_reason: str = ""
        self.returncode: Optional[int] = None
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    # ------------------------------------------------------------------
    # subprocess.Popen-like interface
    # ------------------------------------------------------------------

    def poll(self) -> Optional[int]:
        return self.returncode

    def wait(self, timeout: Optional[float] = None) -> int:
        if not self._done_event.wait(timeout=timeout):
            raise subprocess.TimeoutExpired("fast-replay", timeout or 0.0)
        return int(self.returncode if self.returncode is not None else 0)

    def terminate(self) -> None:
        # A terminate is a soft cancel for the trajectory action.
        self._owner.cancel_fast_replay(self._label)

    def kill(self) -> None:
        # Mirror terminate; trajectory controller has no SIGKILL semantics.
        self._owner.cancel_fast_replay(self._label)

    # ------------------------------------------------------------------
    # Internal: status pump notifies us via this.
    # ------------------------------------------------------------------

    def _finish(self, state: str, reason: str = "") -> None:
        with self._lock:
            if self._final_state is not None:
                return
            self._final_state = state
            self._final_reason = reason
            if state == "done":
                self.returncode = 0
            elif state == "cancelled":
                self.returncode = -15  # SIGTERM-like
            else:
                self.returncode = 1
            self.stdout.write(f"[fast-replay] {state}:{self._label}\n")
            if reason:
                self.stderr.write(f"[fast-replay] reason={reason}\n")
        self._done_event.set()
