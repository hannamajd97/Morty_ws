"""OpenCV mic visualiser: level bar, mirrored waveform, device switcher.

Subscribes to:
  - /speech/level    (Float32, 0..1)         live RMS
  - /speech/state    (String)                idle / speech / muted
  - /speech/waveform (Float32MultiArray)     signed peak envelope

Renders an OpenCV window with three blocks stacked vertically:

  1. Level bar with traffic-light colour ramp (green → yellow → red).
  2. Mirrored vertical-bar waveform — the recent peak envelope drawn
     up and down from the centre line, scrolling left-to-right. This
     is what people expect when they say "audio wave".
  3. Device row with the current PulseAudio input source + output sink
     and a pair of "<" ">" arrow buttons next to each. Click the
     arrows to cycle through available devices; the visualiser calls
     ``pactl set-default-source/sink`` and moves already-running
     streams so the switch is live (no node restart needed).

Disable with ``mic_visualizer:=false`` on voice_robot.launch.py.
"""

from __future__ import annotations

import collections
import os
import re
import shutil
import subprocess
import threading
import time
from typing import List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, Float32MultiArray, String


# ── Window geometry ─────────────────────────────────────────────────────
_W = 700
_H = 640                            # +40 px so the longer chat panel fits
_PAD = 20

# Vertical layout: stacked sections with consistent gaps.
_GAP = 14
_TITLE_H = 70                       # tall title block (brand + subtitle)
_BAR_Y = _TITLE_H + _GAP
_BAR_H = 10                         # slim, modern progress bar
_WAVE_Y = _BAR_Y + _BAR_H + _GAP + 8
_WAVE_H = 130
_TEXT_Y = _WAVE_Y + _WAVE_H + _GAP
_TEXT_BLOCK_H = 232                 # +36 px to fit 4-line reply with breathing room
_DEV_Y = _TEXT_Y + _TEXT_BLOCK_H + _GAP
_DEV_ROW_H = 36
_ARROW_SIZE = 28
_HISTORY_BARS = 260

# ── Palette (BGR for OpenCV) ────────────────────────────────────────────
# Modern dark theme. Single accent (teal). State colors are reused
# everywhere so the eye learns: green=ok, amber=working, red=blocked.
_COLOR_BG          = (25, 20, 15)      # near-black, slight blue tint
_COLOR_CARD        = (41, 35, 28)      # card surface
_COLOR_CARD_HI     = (64, 56, 46)      # card border / hover
_COLOR_CARD_DEEP   = (32, 27, 21)      # inner panel (slightly darker than card)
_COLOR_TEXT        = (250, 244, 240)   # primary text — almost white
_COLOR_TEXT_DIM    = (164, 147, 137)   # secondary text
_COLOR_TEXT_MUTED  = (120, 105,  95)   # tertiary (placeholders)
_COLOR_ACCENT      = (198, 194,  34)   # teal #22c2c6 — single brand accent
_COLOR_USER        = (220, 220, 220)   # user-said label
_COLOR_OK          = (140, 220, 140)   # green
_COLOR_WORKING     = ( 80, 200, 240)   # amber
_COLOR_BLOCKED     = ( 80,  80, 230)   # red

_STATE_COLORS = {
    "idle":   _COLOR_TEXT_DIM,
    "speech": _COLOR_OK,
    "muted":  _COLOR_BLOCKED,
}
_CHAT_STATE_COLORS = {
    "idle":       _COLOR_TEXT_DIM,
    "thinking":   _COLOR_WORKING,
    "responding": _COLOR_OK,
}

_FONT = cv2.FONT_HERSHEY_DUPLEX
_FONT_LABEL = cv2.FONT_HERSHEY_SIMPLEX

# ── Backwards-compat aliases ────────────────────────────────────────────
# Older code in this file still uses the old names; keep them pointing at
# the new palette so we don't have to rewrite every reference at once.
_COLOR_PANEL       = _COLOR_CARD
_COLOR_PANEL_HI    = _COLOR_CARD_HI
_COLOR_DIM         = _COLOR_TEXT_DIM
_COLOR_TITLE       = _COLOR_TEXT
_COLOR_LEVEL_LOW   = _COLOR_OK
_COLOR_LEVEL_MID   = _COLOR_WORKING
_COLOR_LEVEL_HIGH  = _COLOR_BLOCKED
_COLOR_WAVE_QUIET  = _COLOR_TEXT_DIM
_COLOR_WAVE_SPEECH = _COLOR_OK
_COLOR_WAVE_MUTED  = _COLOR_BLOCKED
_COLOR_ARROW       = _COLOR_TEXT


def _pactl(*args: str, capture: bool = True) -> str:
    """Run a pactl command; return stdout (or '' on failure)."""
    if shutil.which("pactl") is None:
        return ""
    try:
        result = subprocess.run(
            ["pactl", *args],
            capture_output=capture,
            text=True,
            check=False,
            timeout=2.0,
        )
        return result.stdout or ""
    except Exception:
        return ""


def _list_sources() -> List[Tuple[str, str]]:
    """Return [(name, short_label), ...] for usable mic sources.

    Skips ``.monitor`` sources (those are loopbacks of output sinks).
    """
    out = _pactl("list", "short", "sources")
    rows: List[Tuple[str, str]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1]
        if name.endswith(".monitor"):
            continue
        rows.append((name, _short_device_label(name)))
    return rows


def _list_sinks() -> List[Tuple[str, str]]:
    out = _pactl("list", "short", "sinks")
    rows: List[Tuple[str, str]] = []
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        name = parts[1]
        rows.append((name, _short_device_label(name)))
    return rows


def _short_device_label(pa_name: str) -> str:
    """Trim noisy PulseAudio device names so they fit in a 200px label."""
    s = pa_name
    for prefix in ("alsa_input.", "alsa_output.", "bluez_input.", "bluez_output."):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    # Drop very long IDs after the first separator pair.
    parts = s.replace("__", "_").split(".")
    if len(parts) > 1:
        s = parts[0]
    # Friendlier substring picks
    for hint in ("USB_Composite_Device", "USB_Camera", "Built-in", "HDMI", "Bluetooth"):
        if hint.lower() in s.lower():
            return hint.replace("_", " ")
    # Generic trim: keep last 40 chars
    return s[-40:]


def _get_default_source() -> str:
    return _pactl("get-default-source").strip()


def _get_default_sink() -> str:
    return _pactl("get-default-sink").strip()


def _set_default_source(name: str) -> None:
    _pactl("set-default-source", name, capture=False)
    # Move all currently-running source-outputs to the new source so any
    # already-open mic stream (e.g. speech_input_node's PyAudio stream)
    # picks up the new device live, without a node restart.
    out = _pactl("list", "short", "source-outputs")
    for line in out.splitlines():
        sid = line.split("\t", 1)[0].strip()
        if sid:
            _pactl("move-source-output", sid, name, capture=False)


def _set_default_sink(name: str) -> None:
    _pactl("set-default-sink", name, capture=False)
    out = _pactl("list", "short", "sink-inputs")
    for line in out.splitlines():
        sid = line.split("\t", 1)[0].strip()
        if sid:
            _pactl("move-sink-input", sid, name, capture=False)


class MicVisualizerNode(Node):
    def __init__(self) -> None:
        super().__init__("mic_visualizer_node")

        # Display-only gain. Multiplies level + waveform values BEFORE
        # drawing so bars look more responsive without changing the
        # actual mic / VAD decisions upstream. Clamped to [0..1] after.
        # 1.0 = unchanged, 3.0 = soft speech still hits the green zone.
        self.declare_parameter("display_gain", 3.0)
        self.declare_parameter("level_topic", "/speech/level")
        self.declare_parameter("state_topic", "/speech/state")
        self.declare_parameter("waveform_topic", "/speech/waveform")
        # Use the ACCEPTED transcript stream by default — only commands
        # that survived chat_node's echo/junk/RMS filters. The raw stream
        # is /speech/transcript; switch via param if you want the firehose
        # for debugging.
        self.declare_parameter(
            "transcript_topic", "/speech/accepted_transcript"
        )
        self.declare_parameter("assistant_topic", "/robot/assistant_text")
        self.declare_parameter("chat_state_topic", "/chat/state")
        # speech_input_node SUBSCRIBES to /speech/muted (Bool, true=mute) and
        # acts on it directly — there's no separate request topic. So both the
        # state we listen for and the toggle we publish use the same name.
        self.declare_parameter("mute_state_topic", "/speech/muted")
        self.declare_parameter("mute_request_topic", "/speech/muted")
        # Model name shown in the title bar. Falls back to env if param empty.
        self.declare_parameter(
            "model_label", os.environ.get("MORTY_LLM_MODEL", "")
        )
        self.declare_parameter("window_name", "Morty mic")
        self.declare_parameter("draw_period_s", 0.05)  # 20 Hz

        self._window_name = str(self.get_parameter("window_name").value)
        self._model_label = str(self.get_parameter("model_label").value)
        self._muted: bool = False
        # Last mute-click time, used to debounce double-fires from the OpenCV
        # mouse callback (a single physical click can register 2-4 times).
        self._mute_click_t: float = 0.0
        self._lock = threading.Lock()
        self._level: float = 0.0
        self._state: str = "idle"
        self._chat_state: str = "idle"
        self._last_heard: str = ""
        self._last_reply: str = ""
        # Ring of signed peak amplitudes (each in [-1, 1]).
        self._wave: collections.deque[float] = collections.deque(
            [0.0] * _HISTORY_BARS, maxlen=_HISTORY_BARS
        )
        # Wall-clock time of the last waveform message received. Used by
        # _draw() to keep the buffer scrolling during silence so the trace
        # doesn't freeze on the last spoken sample.
        self._last_wave_recv: float = time.monotonic()
        # Vertical scroll offset (in WRAPPED LINES) inside Morty's reply
        # panel. 0 = start at the top; mouse wheel adjusts it. Reset to 0
        # whenever a fresh assistant reply arrives.
        self._reply_scroll: int = 0
        # Hover region for the reply panel — set during _draw, read by the
        # mouse wheel handler so wheel only scrolls when over the panel.
        self._reply_panel_rect: tuple[int, int, int, int] = (0, 0, 0, 0)
        # Total wrapped line count of the current reply (set in _draw so
        # the wheel handler can clamp the scroll offset correctly).
        self._reply_total_lines: int = 0
        # How many wrapped lines fit in the visible reply window (set in
        # _draw based on the panel size; used for scroll clamping).
        self._reply_visible_lines: int = 4

        # Cached device lists + current defaults, refreshed periodically
        # so newly-plugged USB mics show up without a node restart.
        self._sources: List[Tuple[str, str]] = []
        self._sinks: List[Tuple[str, str]] = []
        self._cur_source: str = ""
        self._cur_sink: str = ""
        self._refresh_devices()

        # Hit-boxes for arrow buttons: filled in each draw, read by the
        # mouse callback. Using a snapshot dict avoids needing the lock.
        self._hitboxes: dict = {}

        self.create_subscription(
            Float32,
            str(self.get_parameter("level_topic").value),
            self._on_level,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("state_topic").value),
            self._on_state,
            10,
        )
        self.create_subscription(
            Float32MultiArray,
            str(self.get_parameter("waveform_topic").value),
            self._on_waveform,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("transcript_topic").value),
            self._on_transcript,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("assistant_topic").value),
            self._on_assistant,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("chat_state_topic").value),
            self._on_chat_state,
            10,
        )
        self.create_subscription(
            Bool,
            str(self.get_parameter("mute_state_topic").value),
            self._on_muted,
            10,
        )
        self._mute_pub = self.create_publisher(
            Bool,
            str(self.get_parameter("mute_request_topic").value),
            10,
        )

        period = max(0.01, float(self.get_parameter("draw_period_s").value))
        self._timer = self.create_timer(period, self._draw)

        # Refresh device list on a BACKGROUND THREAD, not a ROS timer.
        # `pactl` blocks for 50-200 ms on a VM, and a ROS timer on the
        # single-threaded executor would freeze the draw callback for the
        # same duration every cycle (visible as a stuck waveform). A daemon
        # thread shells out without touching the executor; the draw callback
        # only ever reads the cached results via self._lock.
        self._refresh_stop = threading.Event()
        self._refresh_thread = threading.Thread(
            target=self._refresh_devices_loop,
            name="mic_viz_pactl_refresh",
            daemon=True,
        )
        self._refresh_thread.start()

        try:
            # WINDOW_AUTOSIZE locks the window at the canvas size so the
            # user can't shrink it smaller than the layout was designed for.
            cv2.namedWindow(self._window_name, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self._window_name, self._on_mouse)
        except Exception as exc:
            self.get_logger().warn(f"OpenCV window init failed: {exc}")

    # ── ROS callbacks ────────────────────────────────────────────────────

    def _on_level(self, msg: Float32) -> None:
        with self._lock:
            self._level = max(0.0, min(1.0, float(msg.data)))

    def _on_state(self, msg: String) -> None:
        s = (msg.data or "").strip().lower()
        if s not in _STATE_COLORS:
            s = "idle"
        with self._lock:
            self._state = s

    def _on_waveform(self, msg: Float32MultiArray) -> None:
        with self._lock:
            for v in msg.data:
                # Clamp to ±1 in case of overflow from int16.
                self._wave.append(max(-1.0, min(1.0, float(v))))
            # Mark that real audio just arrived. The draw loop uses this
            # to scroll silence-pixels into the ring buffer when no
            # waveform messages have arrived for a while — otherwise the
            # last spoken sample stays frozen on screen during silence.
            self._last_wave_recv = time.monotonic()

    @staticmethod
    def _safe_text(s: str) -> str:
        """Replace common Unicode punctuation OpenCV can't render with
        ASCII equivalents, and strip anything else non-ASCII. Prevents
        '???' boxes in the GUI when the LLM uses smart quotes / em-dashes.
        """
        if not s:
            return ""
        replacements = {
            "’": "'", "‘": "'",  # curly single quotes
            "“": '"', "”": '"',  # curly double quotes
            "–": "-", "—": "-",  # en-dash, em-dash
            "…": "...",               # ellipsis
            "·": "-",                 # middle dot
            " ": " ",                 # non-breaking space
        }
        for src, dst in replacements.items():
            s = s.replace(src, dst)
        # Anything still non-ASCII → drop it (better than '?' boxes).
        return s.encode("ascii", "ignore").decode("ascii")

    def _on_transcript(self, msg: String) -> None:
        with self._lock:
            self._last_heard = self._safe_text((msg.data or "").strip())

    def _on_assistant(self, msg: String) -> None:
        with self._lock:
            self._last_reply = self._safe_text((msg.data or "").strip())
            # Fresh reply -> jump back to top so the user sees the start.
            self._reply_scroll = 0

    def _on_muted(self, msg: Bool) -> None:
        with self._lock:
            self._muted = bool(msg.data)

    def _on_chat_state(self, msg: String) -> None:
        s = (msg.data or "").strip().lower()
        if s not in ("idle", "thinking", "responding"):
            s = "idle"
        with self._lock:
            self._chat_state = s

    # ── device handling ──────────────────────────────────────────────────

    def _refresh_devices(self) -> None:
        # Shell out to pactl OUTSIDE the lock — these calls can block for
        # 50-200 ms on a VM. Only hold the lock for the tiny assignment.
        new_sources = _list_sources()
        new_sinks = _list_sinks()
        new_cur_source = _get_default_source()
        new_cur_sink = _get_default_sink()
        with self._lock:
            self._sources = new_sources
            self._sinks = new_sinks
            self._cur_source = new_cur_source
            self._cur_sink = new_cur_sink

    def _refresh_devices_loop(self) -> None:
        """Background-thread loop: refresh the pactl device list every 2 s
        without ever blocking the ROS executor."""
        while not self._refresh_stop.is_set():
            try:
                self._refresh_devices()
            except Exception as exc:
                # Don't kill the thread on a transient pactl hiccup.
                self.get_logger().warn(f"device refresh failed: {exc}")
            # Wait 2 s OR exit promptly when shutdown asks us to stop.
            if self._refresh_stop.wait(2.0):
                break

    def _cycle(self, devices: List[Tuple[str, str]], current: str, direction: int) -> Optional[str]:
        if not devices:
            return None
        names = [d[0] for d in devices]
        try:
            idx = names.index(current)
        except ValueError:
            idx = 0
        new_idx = (idx + direction) % len(names)
        return names[new_idx]

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        # Mouse wheel scrolls Morty's reply panel when the cursor is over it.
        if event == cv2.EVENT_MOUSEWHEEL:
            with self._lock:
                rx1, ry1, rx2, ry2 = self._reply_panel_rect
                total = self._reply_total_lines
            inside = (rx1 <= x <= rx2) and (ry1 <= y <= ry2)
            if not inside or total <= 0:
                return
            # On Linux OpenCV reports wheel delta in flags; sign tells dir.
            # Use a fixed step of 1 line per detent for predictability.
            direction = -1 if flags > 0 else 1
            with self._lock:
                visible = self._reply_visible_lines
                max_off = max(0, total - visible)
                self._reply_scroll = max(
                    0, min(max_off, self._reply_scroll + direction)
                )
            return

        if event != cv2.EVENT_LBUTTONDOWN:
            return
        with self._lock:
            boxes = dict(self._hitboxes)
        for tag, (x1, y1, x2, y2) in boxes.items():
            if x1 <= x <= x2 and y1 <= y <= y2:
                self._handle_click(tag)
                return

    def _handle_click(self, tag: str) -> None:
        with self._lock:
            sources = list(self._sources)
            sinks = list(self._sinks)
            cur_source = self._cur_source
            cur_sink = self._cur_sink
            is_muted = self._muted
        if tag == "mute_toggle":
            # Debounce: ignore clicks <500ms apart. OpenCV can fire the
            # mouse callback 2-4 times per physical click, and the
            # /speech/muted subscriber may also re-trigger our own state
            # change if anyone else republishes.
            import time as _time
            now = _time.time()
            if now - self._mute_click_t < 0.5:
                return
            self._mute_click_t = now
            new_state = not is_muted
            self.get_logger().info(
                f"Mute requested: {'MUTE' if new_state else 'UNMUTE'}"
            )
            self._mute_pub.publish(Bool(data=new_state))
            # Update our local state immediately so the pill flips without
            # waiting for a confirmation publish from the speech node.
            with self._lock:
                self._muted = new_state
            return
        if tag == "in_prev":
            new = self._cycle(sources, cur_source, -1)
            if new and new != cur_source:
                self.get_logger().info(f"Switching input -> {new}")
                _set_default_source(new)
                self._refresh_devices()
        elif tag == "in_next":
            new = self._cycle(sources, cur_source, +1)
            if new and new != cur_source:
                self.get_logger().info(f"Switching input -> {new}")
                _set_default_source(new)
                self._refresh_devices()
        elif tag == "out_prev":
            new = self._cycle(sinks, cur_sink, -1)
            if new and new != cur_sink:
                self.get_logger().info(f"Switching output -> {new}")
                _set_default_sink(new)
                self._refresh_devices()
        elif tag == "out_next":
            new = self._cycle(sinks, cur_sink, +1)
            if new and new != cur_sink:
                self.get_logger().info(f"Switching output -> {new}")
                _set_default_sink(new)
                self._refresh_devices()

    # ── render ───────────────────────────────────────────────────────────

    @staticmethod
    def _rounded_rect(
        img: np.ndarray,
        p1: Tuple[int, int],
        p2: Tuple[int, int],
        color,
        radius: int = 8,
        thickness: int = -1,
    ) -> None:
        """Filled or outlined rounded rectangle, anti-aliased."""
        x1, y1 = p1
        x2, y2 = p2
        if x2 - x1 < 2 * radius or y2 - y1 < 2 * radius:
            cv2.rectangle(img, p1, p2, color, thickness, cv2.LINE_AA)
            return
        if thickness < 0:
            # Filled: draw a centre rect + 4 corner circles + 2 side rects
            cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2),
                          color, -1, cv2.LINE_AA)
            cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius),
                          color, -1, cv2.LINE_AA)
            cv2.circle(img, (x1 + radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x2 - radius, y1 + radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x1 + radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
            cv2.circle(img, (x2 - radius, y2 - radius), radius, color, -1, cv2.LINE_AA)
        else:
            # Outline
            cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness, cv2.LINE_AA)
            cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius),
                        180, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius),
                        270, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius),
                        90, 0, 90, color, thickness, cv2.LINE_AA)
            cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius),
                        0, 0, 90, color, thickness, cv2.LINE_AA)

    def _draw_arrow_button(
        self,
        img: np.ndarray,
        cx: int,
        cy: int,
        direction: str,
        tag: str,
        boxes: dict,
    ) -> None:
        r = _ARROW_SIZE // 2
        x1, y1, x2, y2 = cx - r, cy - r, cx + r, cy + r
        # Rounded button background
        self._rounded_rect(img, (x1, y1), (x2, y2),
                           _COLOR_PANEL_HI, radius=6, thickness=-1)
        # Triangle pointing left or right (anti-aliased)
        tri_r = r - 6
        if direction == "<":
            pts = np.array([
                [cx + tri_r // 2, cy - tri_r],
                [cx + tri_r // 2, cy + tri_r],
                [cx - tri_r, cy],
            ], dtype=np.int32)
        else:
            pts = np.array([
                [cx - tri_r // 2, cy - tri_r],
                [cx - tri_r // 2, cy + tri_r],
                [cx + tri_r, cy],
            ], dtype=np.int32)
        cv2.fillPoly(img, [pts], _COLOR_ARROW, cv2.LINE_AA)
        boxes[tag] = (x1, y1, x2, y2)

    def _draw_device_row(
        self,
        img: np.ndarray,
        y: int,
        label: str,
        device_label: str,
        tag_prev: str,
        tag_next: str,
        boxes: dict,
    ) -> None:
        # Heading
        cv2.putText(
            img, label, (_PAD + 4, y + 21),
            _FONT_LABEL, 0.55, _COLOR_DIM, 1, cv2.LINE_AA,
        )
        # Arrows
        arrow_y = y + _DEV_ROW_H // 2
        ax_prev = _PAD + 60
        ax_next = _W - _PAD - _ARROW_SIZE // 2 - 6
        self._draw_arrow_button(img, ax_prev, arrow_y, "<", tag_prev, boxes)
        self._draw_arrow_button(img, ax_next, arrow_y, ">", tag_next, boxes)
        # Device name
        name_x = ax_prev + _ARROW_SIZE // 2 + 10
        max_w = ax_next - _ARROW_SIZE // 2 - 10 - name_x
        text = device_label or "(none)"
        # Crude truncation by width
        if len(text) * 9 > max_w:
            text = text[: max(3, max_w // 9 - 1)] + "…"
        cv2.putText(
            img, text, (name_x, y + 21),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, _COLOR_TEXT, 1, cv2.LINE_AA,
        )

    @staticmethod
    def _wrap_for_width(text: str, max_chars: int) -> List[str]:
        """Greedy word-wrap that returns a list of lines no longer than
        ``max_chars`` characters. Used to fit the transcript / reply
        text inside the window."""
        if not text:
            return []
        words = text.split()
        lines: List[str] = []
        cur = ""
        for w in words:
            if not cur:
                cur = w
                continue
            if len(cur) + 1 + len(w) <= max_chars:
                cur = cur + " " + w
            else:
                lines.append(cur)
                cur = w
        if cur:
            lines.append(cur)
        return lines

    def _draw(self) -> None:
        # Before snapshotting the buffer for rendering: if no waveform
        # message has arrived in the last 100 ms, push silence samples
        # into the ring so the trace keeps scrolling left instead of
        # freezing on the last spoken chunk. ~3 silence pixels per draw
        # frame at 20 Hz = noticeable scroll without flooding the deque.
        now = time.monotonic()
        with self._lock:
            if (now - self._last_wave_recv) > 0.1:
                for _ in range(3):
                    self._wave.append(0.0)
            level = self._level
            state = self._state
            wave = list(self._wave)
            cur_source = self._cur_source
            cur_sink = self._cur_sink
            heard = self._last_heard
            reply = self._last_reply
            chat_state = self._chat_state
            muted = self._muted

        img = np.full((_H, _W, 3), _COLOR_BG, dtype=np.uint8)
        boxes: dict = {}

        # ── Title block ────────────────────────────────────────────────
        # "Morty" wordmark (large, accent), with the model name as a
        # smaller subtitle underneath. Pills sit to the right, vertically
        # centered on the title row, with safe right-aligned placement.
        cv2.putText(
            img, "Morty",
            (_PAD, 38),
            _FONT, 1.15, _COLOR_ACCENT, 2, cv2.LINE_AA,
        )

        # Subtitle: cleaned model name. Strip provider/region prefixes and
        # the date+version tail so the user sees just the model family.
        if self._model_label:
            model_short = self._model_label
            for prefix in ("us.anthropic.", "eu.anthropic.", "anthropic.",
                           "openai.", "meta.", "mistral.", "cohere.", "amazon."):
                if model_short.startswith(prefix):
                    model_short = model_short[len(prefix):]
                    break
            model_short = re.sub(r"-\d{6,8}-v\d+:\d+$", "", model_short)
            model_short = re.sub(r"-v\d+:\d+$", "", model_short)
            subtitle = f"model: {model_short}"
        else:
            subtitle = "voice + intent"
        cv2.putText(
            img, subtitle,
            (_PAD + 2, 60),
            _FONT_LABEL, 0.46, _COLOR_TEXT_DIM, 1, cv2.LINE_AA,
        )

        # ── Pills (right side of title): MUTE + state ─────────────────
        # Both pills are the same height, vertically centred on the title.
        PILL_FONT = _FONT
        PILL_SCALE = 0.55
        PILL_THICK = 1
        PILL_H = 30
        PILL_PAD_X = 14
        PILL_DOT_R = 5
        PILL_CY = 24
        pill_y1 = PILL_CY - PILL_H // 2
        pill_y2 = PILL_CY + PILL_H // 2

        # Mic-state pill (rightmost).
        pill_text = state.upper() if not muted else "MUTED"
        pill_col = _STATE_COLORS.get(state, _COLOR_TEXT)
        if muted:
            pill_col = _COLOR_BLOCKED
        (tw, _), _ = cv2.getTextSize(pill_text, PILL_FONT, PILL_SCALE, PILL_THICK)
        pill_x2 = _W - _PAD
        pill_x1 = pill_x2 - tw - 2 * PILL_PAD_X - 14
        self._rounded_rect(img, (pill_x1, pill_y1), (pill_x2, pill_y2),
                           _COLOR_CARD, radius=PILL_H // 2, thickness=-1)
        cv2.circle(img, (pill_x1 + PILL_PAD_X, PILL_CY),
                   PILL_DOT_R, pill_col, -1, cv2.LINE_AA)
        cv2.putText(
            img, pill_text,
            (pill_x1 + PILL_PAD_X + PILL_DOT_R + 8, PILL_CY + 5),
            PILL_FONT, PILL_SCALE, _COLOR_TEXT, PILL_THICK, cv2.LINE_AA,
        )

        # Mute toggle pill (just left of the state pill).
        mute_label = "UNMUTE" if muted else "MUTE"
        mute_col = _COLOR_BLOCKED if muted else _COLOR_OK
        (mtw, _), _ = cv2.getTextSize(mute_label, PILL_FONT, PILL_SCALE, PILL_THICK)
        mute_x2 = pill_x1 - 10
        mute_x1 = mute_x2 - mtw - 2 * PILL_PAD_X - 14
        self._rounded_rect(img, (mute_x1, pill_y1), (mute_x2, pill_y2),
                           _COLOR_CARD, radius=PILL_H // 2, thickness=-1)
        # Hover-style outline (subtle border) so it reads as clickable.
        cv2.rectangle(img, (mute_x1, pill_y1), (mute_x2, pill_y2),
                      _COLOR_CARD_HI, 1, cv2.LINE_AA)
        cv2.circle(img, (mute_x1 + PILL_PAD_X, PILL_CY),
                   PILL_DOT_R, mute_col, -1, cv2.LINE_AA)
        cv2.putText(
            img, mute_label,
            (mute_x1 + PILL_PAD_X + PILL_DOT_R + 8, PILL_CY + 5),
            PILL_FONT, PILL_SCALE, _COLOR_TEXT, PILL_THICK, cv2.LINE_AA,
        )
        boxes["mute_toggle"] = (mute_x1, pill_y1, mute_x2, pill_y2)

        # Soft divider under title block
        cv2.line(img, (_PAD, _TITLE_H), (_W - _PAD, _TITLE_H),
                 _COLOR_CARD, 1, cv2.LINE_AA)

        # ── Level bar (rounded) ──────────────────────────────────────
        # Background track
        self._rounded_rect(
            img, (_PAD, _BAR_Y), (_W - _PAD, _BAR_Y + _BAR_H),
            _COLOR_PANEL, radius=_BAR_H // 2, thickness=-1,
        )
        # Filled portion
        fill_w = int((_W - 2 * _PAD - 4) * level)
        if level < 0.4:
            col = _COLOR_LEVEL_LOW
        elif level < 0.75:
            col = _COLOR_LEVEL_MID
        else:
            col = _COLOR_LEVEL_HIGH
        if fill_w > 2:
            self._rounded_rect(
                img,
                (_PAD + 2, _BAR_Y + 2),
                (_PAD + 2 + fill_w, _BAR_Y + _BAR_H - 2),
                col,
                radius=(_BAR_H - 4) // 2,
                thickness=-1,
            )
        # Inline label + value just above the bar so they never overlap it.
        label_y = _BAR_Y - 6
        cv2.putText(
            img, "MIC LEVEL",
            (_PAD, label_y),
            _FONT_LABEL, 0.42, _COLOR_TEXT_DIM, 1, cv2.LINE_AA,
        )
        level_str = f"{level:.2f}"
        (lvw, _), _ = cv2.getTextSize(level_str, _FONT_LABEL, 0.42, 1)
        cv2.putText(
            img, level_str,
            (_W - _PAD - lvw, label_y),
            _FONT_LABEL, 0.42, _COLOR_TEXT_DIM, 1, cv2.LINE_AA,
        )

        # ── Waveform panel ───────────────────────────────────────────
        wave_x0 = _PAD
        wave_x1 = _W - _PAD
        wave_y0 = _WAVE_Y
        wave_y1 = _WAVE_Y + _WAVE_H
        self._rounded_rect(
            img, (wave_x0, wave_y0), (wave_x1, wave_y1),
            _COLOR_PANEL, radius=10, thickness=-1,
        )
        center_y = (wave_y0 + wave_y1) // 2
        usable_h = (wave_y1 - wave_y0) // 2 - 6
        cv2.line(img, (wave_x0 + 8, center_y), (wave_x1 - 8, center_y),
                 _COLOR_PANEL_HI, 1, cv2.LINE_AA)
        if state == "muted":
            bar_col = _COLOR_WAVE_MUTED
        elif state == "speech":
            bar_col = _COLOR_WAVE_SPEECH
        else:
            bar_col = _COLOR_WAVE_QUIET
        if wave:
            n = len(wave)
            avail = wave_x1 - wave_x0 - 16
            bar_w = max(1, avail // n)
            gap = 1 if bar_w >= 3 else 0
            xs = np.linspace(
                wave_x0 + 8, wave_x1 - 8 - bar_w, num=n, dtype=np.int32
            )
            for x, v in zip(xs, wave):
                h = int(abs(v) * usable_h)
                if h <= 1:
                    h = 1  # always show a tiny tick even on silence
                cv2.rectangle(
                    img,
                    (int(x), center_y - h),
                    (int(x) + bar_w - 1 - gap, center_y + h),
                    bar_col,
                    thickness=-1,
                )

        # ── Conversation panel ───────────────────────────────────────
        text_x0 = _PAD
        text_y0 = _TEXT_Y
        text_x1 = _W - _PAD
        text_y1 = _TEXT_Y + _TEXT_BLOCK_H
        self._rounded_rect(
            img, (text_x0, text_y0), (text_x1, text_y1),
            _COLOR_PANEL, radius=10, thickness=-1,
        )
        # Chat-state pill in top-right corner of this panel.
        cs_text = chat_state.upper()
        cs_col = _CHAT_STATE_COLORS.get(chat_state, _COLOR_TEXT)
        (ctw, _), _ = cv2.getTextSize(cs_text, _FONT, 0.5, 1)
        cs_x2 = text_x1 - 14
        cs_x1 = cs_x2 - ctw - 36
        cs_y1 = text_y0 + 10
        cs_y2 = text_y0 + 36
        self._rounded_rect(img, (cs_x1, cs_y1), (cs_x2, cs_y2),
                           _COLOR_CARD_DEEP, radius=13, thickness=-1)
        cv2.circle(img, (cs_x1 + 12, (cs_y1 + cs_y2) // 2),
                   5, cs_col, -1, cv2.LINE_AA)
        cv2.putText(
            img, cs_text,
            (cs_x1 + 26, cs_y2 - 9),
            _FONT, 0.5, _COLOR_TEXT, 1, cv2.LINE_AA,
        )

        # ── User turn ─────────────────────────────────────────────────
        max_chars = 60
        you_label_y = text_y0 + 30
        cv2.putText(
            img, "YOU",
            (text_x0 + 16, you_label_y),
            _FONT_LABEL, 0.46, _COLOR_USER, 1, cv2.LINE_AA,
        )
        you_text_y = text_y0 + 56
        if heard:
            heard_lines = self._wrap_for_width(heard, max_chars)[:2]
            for i, line in enumerate(heard_lines):
                cv2.putText(
                    img, line,
                    (text_x0 + 16, you_text_y + 18 * i),
                    _FONT, 0.52, _COLOR_TEXT, 1, cv2.LINE_AA,
                )
        else:
            cv2.putText(
                img, "Waiting for you to speak…",
                (text_x0 + 16, you_text_y),
                _FONT, 0.46, _COLOR_TEXT_MUTED, 1, cv2.LINE_AA,
            )

        # Divider between You and Morty
        sep_y = text_y0 + 96
        cv2.line(img, (text_x0 + 16, sep_y),
                 (text_x1 - 16, sep_y), _COLOR_CARD_HI, 1, cv2.LINE_AA)

        # ── Morty turn ────────────────────────────────────────────────
        morty_label_y = text_y0 + 120
        cv2.putText(
            img, "MORTY",
            (text_x0 + 16, morty_label_y),
            _FONT_LABEL, 0.46, _COLOR_ACCENT, 1, cv2.LINE_AA,
        )
        morty_text_y = text_y0 + 146
        # Reply panel: SCROLLABLE. Wrap the full reply, show as many lines
        # as fit, expose a wheel hit-area so the user can scroll up/down,
        # and draw a slim scrollbar on the right edge when overflow exists.
        MAX_VISIBLE_LINES = 4
        line_h = 18
        # Reserve a few px on the right for the scrollbar so text doesn't
        # touch it.
        SCROLLBAR_W = 4
        text_inner_x1 = text_x1 - 16 - (SCROLLBAR_W + 6)
        if reply:
            all_lines = self._wrap_for_width(reply, max_chars)
            total = len(all_lines)
            visible = MAX_VISIBLE_LINES
            # Clamp scroll offset against the freshly-computed total in
            # case the wheel handler raced ahead of a shorter reply.
            with self._lock:
                self._reply_total_lines = total
                self._reply_visible_lines = visible
                max_off = max(0, total - visible)
                if self._reply_scroll > max_off:
                    self._reply_scroll = max_off
                offset = self._reply_scroll

            window = all_lines[offset : offset + visible]
            for i, line in enumerate(window):
                cv2.putText(
                    img, line,
                    (text_x0 + 16, morty_text_y + line_h * i),
                    _FONT, 0.52, _COLOR_TEXT, 1, cv2.LINE_AA,
                )

            # Scrollbar (only when there's something to scroll). The track
            # sits along the right edge of the reply window; the thumb
            # height + y is proportional to the visible window.
            if total > visible:
                track_x = text_x1 - 16 - SCROLLBAR_W
                track_y0 = morty_text_y - line_h + 4
                track_y1 = morty_text_y + line_h * visible - 4
                cv2.rectangle(
                    img, (track_x, track_y0), (track_x + SCROLLBAR_W, track_y1),
                    _COLOR_CARD_HI, -1, cv2.LINE_AA,
                )
                track_h = track_y1 - track_y0
                thumb_h = max(20, int(track_h * visible / total))
                thumb_y0 = track_y0 + int(track_h * offset / total)
                thumb_y1 = min(track_y1, thumb_y0 + thumb_h)
                cv2.rectangle(
                    img, (track_x, thumb_y0), (track_x + SCROLLBAR_W, thumb_y1),
                    _COLOR_ACCENT, -1, cv2.LINE_AA,
                )
                # Hint when more content is hidden — small caret arrows
                # above/below the window so the user knows scrolling exists.
                if offset > 0:
                    cv2.putText(
                        img, "^",
                        (text_x1 - 30, morty_text_y - 22),
                        _FONT_LABEL, 0.55, _COLOR_TEXT_DIM, 1, cv2.LINE_AA,
                    )
                if offset + visible < total:
                    cv2.putText(
                        img, "v",
                        (text_x1 - 30, morty_text_y + line_h * visible + 14),
                        _FONT_LABEL, 0.55, _COLOR_TEXT_DIM, 1, cv2.LINE_AA,
                    )
        else:
            with self._lock:
                self._reply_total_lines = 0
            cv2.putText(
                img, "(Morty hasn't replied yet)",
                (text_x0 + 16, morty_text_y),
                _FONT, 0.46, _COLOR_TEXT_MUTED, 1, cv2.LINE_AA,
            )

        # Register the reply panel as a wheel-scroll hit area.
        with self._lock:
            self._reply_panel_rect = (
                text_x0 + 16,
                morty_text_y - line_h,
                text_x1 - 16,
                morty_text_y + line_h * MAX_VISIBLE_LINES,
            )

        # ── Device rows + arrow buttons ──────────────────────────────
        # Background panel
        dev_block_y0 = _DEV_Y - 6
        dev_block_y1 = _DEV_Y + 2 * (_DEV_ROW_H + 6) + 4
        self._rounded_rect(
            img, (_PAD, dev_block_y0), (_W - _PAD, dev_block_y1),
            _COLOR_PANEL, radius=10, thickness=-1,
        )
        self._draw_device_row(
            img,
            _DEV_Y,
            "IN ",
            _short_device_label(cur_source) if cur_source else "(none)",
            "in_prev",
            "in_next",
            boxes,
        )
        self._draw_device_row(
            img,
            _DEV_Y + _DEV_ROW_H + 6,
            "OUT",
            _short_device_label(cur_sink) if cur_sink else "(none)",
            "out_prev",
            "out_next",
            boxes,
        )

        # Publish hit-boxes so the mouse callback can read them.
        with self._lock:
            self._hitboxes = boxes

        try:
            cv2.imshow(self._window_name, img)
            cv2.waitKey(1)
        except Exception:
            # Window may not be available headless; silently degrade.
            pass

    def destroy_node(self) -> bool:
        try:
            cv2.destroyWindow(self._window_name)
        except Exception:
            pass
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MicVisualizerNode()
    # NOTE: must stay on the main thread — OpenCV's window event loop
    # (cv2.imshow / waitKey) only repaints when called from the same
    # thread that created the window, and on Linux only the main thread
    # has the X11 display connection. A MultiThreadedExecutor here
    # produces a frozen black window.
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
