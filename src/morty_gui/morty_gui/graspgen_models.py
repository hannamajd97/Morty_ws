"""GraspGen checkpoints and the switch between them (2026-09-24).

Majd: the Robotiq model "was faster at the load" — the default again, to
check it — and a switcher in the Advanced window: Apply loads the other
checkpoint, only when it is not the one loaded, and says when it is ready.

The service (GraspGen/scripts/graspgen_service.py) loads ONE checkpoint at
start, runs a warm-up inference and only then binds its ZMQ port and prints
"Service ready". A switch is therefore a restart: stop the service, start it
on the other --config, wait for "Service ready" in this session's part of
its log (the ready LED reads the same line). A pick that asks GraspGen
meanwhile fails and can be repeated; the pipeline opens a fresh socket for
every request, so it needs no restart.

The pipeline learns the model through two parameters, passed at its start
and set live after a switch: graspgen_model (part of the grasp precompute's
parameter hash, so grasps stored from the old model are dropped) and
graspgen_origin_to_fingertip_m, the distance from the checkpoint's grasp
origin (its gripper BASE) to the fingertip centre. The last one is only used
with graspgen_matrix_normalized false; the normalised mode that runs today
places the grasp from the object's box and ignores it.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from PyQt5 import QtCore

from .config import GRASPGEN_LOG, GREEN, GREY, ORANGE, RED

_MODELS_DIR = os.path.expanduser("~/morty_ws/models/graspgen")
_SERVICE = os.path.expanduser("~/morty_ws/GraspGen/scripts/graspgen_service.py")
_BIND = "tcp://127.0.0.1:5555"


@dataclass(frozen=True)
class GraspGenModel:
    key: str            # the gripper name in the checkpoint's config
    label: str
    config: str
    fingertip_m: float  # grasp origin (gripper base) -> fingertip centre, along the approach
    note: str


MODELS: Tuple[GraspGenModel, ...] = (
    # Ran until 2026-09-11. PointNet++ backbone, no kernel compile: ready 28 s
    # after start with the pipeline running (2026-09-24; ~5 s on an idle
    # machine, 09-11), 0.12-0.19 s per request. Fingertips 0.195 m ahead of
    # the base (contact points in GraspGen/config/grippers/robotiq_2f_140.yaml);
    # measured on the service 2026-09-24: on a synthetic 4.7 cm tube the
    # grasps reach its axis a median 19.7 cm along the approach.
    GraspGenModel(
        "robotiq_2f_140", "Robotiq 2F-140",
        os.path.join(_MODELS_DIR, "graspgen_robotiq_2f_140.yml"), 0.195,
        "~30 s to ready, 0.12-0.19 s per request. Trained for a 136 mm stroke (the printed "
        "gripper opens 85 mm); on 2026-09-11, 9 of 12 closes with it found nothing."),
    # Since 2026-09-11 (Majd): 105 mm opening and depth, the closest of the
    # shipped grippers to the printed one (85 mm / 80 mm fingers). Its PTv3
    # backbone needs the CUDA build of spconv and compiles its kernels on
    # the first inference, which the service pays in its warm-up: ~78 s from
    # start to ready, then 0.30 s per request. Fingertip distance measured
    # on the service (2026-09-15, synthetic tube).
    GraspGenModel(
        "franka_panda", "Franka Panda",
        os.path.join(_MODELS_DIR, "graspgen_franka_panda.yml"), 0.1034,
        "~80 s to ready (its kernels compile first), 0.30 s per request. 105 mm stroke, "
        "the closest to the printed gripper; the default from 2026-09-11 to 09-24."),
)
DEFAULT_MODEL = "robotiq_2f_140"   # Majd 2026-09-24: "as default, to check"
_SETTINGS_KEY = "graspgen_model"


def model(key: str) -> Optional[GraspGenModel]:
    return next((m for m in MODELS if m.key == key), None)


def service_cmd(m: GraspGenModel) -> List[str]:
    return ["/usr/bin/python3", "-u", _SERVICE, "--config", m.config, "--bind", _BIND]


def model_of_pid(pid: Optional[int]) -> Optional[GraspGenModel]:
    """Which checkpoint a running service loaded: its --config argument.
    Works for a service this GUI started and for an adopted one."""
    if pid is None:
        return None
    try:
        with open(f"/proc/{int(pid)}/cmdline", "rb") as f:
            argv = [a.decode(errors="replace") for a in f.read().split(b"\0")]
        cfg = os.path.basename(argv[argv.index("--config") + 1])
    except (OSError, ValueError, IndexError):
        return None
    return next((m for m in MODELS if os.path.basename(m.config) == cfg), None)


def log_ready(log_path: Optional[str] = None) -> bool:
    """True once the service of the CURRENT log session printed "Service
    ready". The log is appended across restarts (2026-09-17), so only the
    part after the last session header counts — else a previous run's line
    lit the LED while the model was still loading (2026-09-18)."""
    try:
        with open(log_path or GRASPGEN_LOG, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4 * 1024 * 1024))
            data = f.read()
    except OSError:
        return False
    cut = data.rfind(b"===== session started")
    if cut >= 0:
        data = data[cut:]
    return b"Service ready" in data


class GraspGenModels(QtCore.QObject):
    """The chosen checkpoint (remembered), the running one, and the switch."""

    status = QtCore.pyqtSignal(str, str)     # text, colour — for the Advanced tab
    switched = QtCore.pyqtSignal(object)     # GraspGenModel, once the new service answers

    def __init__(self, launch, settings: QtCore.QSettings, parent=None) -> None:
        super().__init__(parent)
        self._launch = launch                # the GUI's ManagedLaunch for the service
        self._settings = settings
        self._target: Optional[GraspGenModel] = None
        self._old_pid: Optional[int] = None
        self._state = "idle"                 # idle | stopping | loading
        self._t0 = 0.0
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._tick)

    # ── what is chosen, what runs ────────────────────────────────────
    def chosen(self) -> GraspGenModel:
        return model(str(self._settings.value(_SETTINGS_KEY, DEFAULT_MODEL))) or model(DEFAULT_MODEL)

    def running(self) -> Optional[GraspGenModel]:
        return model_of_pid(self._launch.pid())

    def serving(self) -> GraspGenModel:
        """The model the pipeline should be told about: the running one,
        else the one that will start."""
        return self.running() or self.chosen()

    def busy(self) -> bool:
        return self._state != "idle"

    def start(self) -> None:
        """Start the service on the chosen checkpoint (with the pipeline)."""
        if not self._launch.running():
            self._launch.start(service_cmd(self.chosen()), env=os.environ.copy(), log_path=GRASPGEN_LOG)

    def describe(self) -> Tuple[str, str]:
        """One status line while no switch runs."""
        run, pick = self.running(), self.chosen()
        if not self._launch.running():
            return f"not running — {pick.label} loads when the pipeline starts", GREY
        if run is None:
            return "running a checkpoint this GUI does not know (see logs/graspgen_gui.log)", ORANGE
        text = f"running: {run.label}, " + ("ready" if log_ready() else "loading")
        if run.key != pick.key:
            return text + f" — chosen: {pick.label}, press Apply to switch", ORANGE
        return text, GREEN if log_ready() else ORANGE

    # ── the switch ───────────────────────────────────────────────────
    def apply(self, key: str) -> None:
        m = model(key)
        if m is None:
            return
        self._settings.setValue(_SETTINGS_KEY, m.key)
        if self.busy():
            self.status.emit(f"a switch to {self._target.label} is still running", ORANGE)
            return
        if not self._launch.running():
            self.status.emit(f"{m.label} saved — it loads when the pipeline starts", GREY)
            return
        run = self.running()
        if run is not None and run.key == m.key:
            self.status.emit(f"{m.label} is already loaded" + ("" if log_ready() else " (still warming up)"),
                             GREEN if log_ready() else ORANGE)
            return
        self._target, self._state, self._t0 = m, "stopping", time.monotonic()
        self._old_pid = self._launch.pid()
        self._launch.stop()
        self.status.emit(f"stopping {run.label if run else 'the running model'} …", ORANGE)
        self._timer.start()

    def _tick(self) -> None:
        el = time.monotonic() - self._t0
        m = self._target
        if self._state == "stopping":
            if self._launch.running():
                if self._launch.pid() != self._old_pid:
                    # A pipeline start in the gap launched the chosen (= target) model.
                    self._state, self._t0 = "loading", time.monotonic()
                return                       # the launch escalates SIGINT → SIGTERM → SIGKILL
            self._launch.reap()
            self._launch.start(service_cmd(m), env=os.environ.copy(), log_path=GRASPGEN_LOG)
            self._state, self._t0 = "loading", time.monotonic()
            self.status.emit(f"loading {m.label} …", ORANGE)
            return
        if log_ready():
            self._finish()
            self.status.emit(f"ready: {m.label} (loaded in {el:.0f} s)", GREEN)
            self.switched.emit(m)
        elif not self._launch.running():
            self._finish()
            self.status.emit(f"{m.label} stopped before it was ready — see logs/graspgen_gui.log", RED)
        else:
            slow = " (its kernels compile first, ~80 s)" if m.key == "franka_panda" else ""
            self.status.emit(f"loading {m.label} … {el:.0f} s{slow}", ORANGE)

    def _finish(self) -> None:
        self._timer.stop()
        self._state = "idle"
