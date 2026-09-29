"""The newest message of a fast topic, on demand — no callbacks, no churn.

Why this exists (2026-09-29)
----------------------------
intent_router and chat_node checked "is the driver alive?" by SAMPLING
/joint_states (500 Hz) — and chat_node also /tf (~830 Hz) — through
rclpy's ``wait_for_message``: a subscription created, one message taken,
the subscription destroyed, every 0.5 s. With SHM off (2026-09-02) every
create/destroy is a discovery round to all ~30 participants, and the
driver starts streaming 500 Hz to a reader that may already be gone
(the unmatch travels over the same congested UDP). Measured live: the two
nodes at 84 % and 59 % CPU with inputs of <= 1 Hz, the busy thread a Fast
DDS receive thread, and the ONLY UDP drops in the graph besides the
image consumers (chat_node 114/s, intent_router 140/s).

What it does
------------
The pattern of grasp_pipeline.tf_listener_lite, for one topic: a private
node that NO executor spins holds ONE standing best-effort, keep-last-1
reader. DDS keeps the newest sample in the reader on its own (native code,
no Python wake-ups, no discovery after the first match); ``take()`` drains
it with ``take_message`` — the call rclpy's executor uses — and returns
the newest message received since the previous ``take()``, or None.

It lives here, not next to DecimatedTransformListener, because importing
anything from grasp_pipeline runs its package __init__, which loads YOLO
and torch (13 s, ~1 GB) — unacceptable in the voice nodes.
"""
from __future__ import annotations

import struct
import threading
from typing import Any, Callable, List, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message


def joint_state_stream(data: bytes) -> int:
    """Stream key of a serialized JointState: the number of joint names.
    /joint_states carries two streams here — the driver's arm (6 names,
    500 Hz) and the gripper adapter's fingers — and a reader that kept only
    the newest message would almost never see the fingers. CDR: 4-byte
    encapsulation header, then (body offsets) stamp 0..8, frame_id length
    at 8, its bytes from 12, the names length at the next multiple of 4.
    Falls back to the payload size."""
    try:
        fmt = "<I" if data[1] == 0x01 else ">I"
        n = struct.unpack_from(fmt, data, 4 + 8)[0]
        off = (12 + n + 3) & ~3
        return struct.unpack_from(fmt, data, 4 + off)[0]
    except (IndexError, struct.error):
        return len(data)


class LatestMessage:
    """``take()`` → the newest ``msg_type`` message on ``topic`` since the
    last call (deserialized), or None when nothing new arrived.

    With ``stream_key`` (a function of the serialized bytes) and a
    ``depth`` that holds everything arriving between two reads,
    ``take_streams()`` returns the newest message of EACH stream."""

    def __init__(self, node: Node, msg_type: Any, topic: str, depth: int = 1,
                 stream_key: Optional[Callable[[bytes], int]] = None) -> None:
        self._msg_type = msg_type
        self._stream_key = stream_key
        self._lock = threading.Lock()
        suffix = topic.strip("/").replace("/", "_") or "topic"
        self._node = rclpy.create_node(
            f"{node.get_name()}_{suffix}_latest",
            context=node.context,
            use_global_arguments=False,      # never inherit the owner's __node:= remap
            start_parameter_services=False,
            enable_rosout=False,
        )
        # Best effort: the reliable publisher sends without keeping resend
        # state for us; keep-last ``depth``: only the newest samples are held.
        qos = QoSProfile(depth=max(1, int(depth)), history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.BEST_EFFORT,
                         durability=DurabilityPolicy.VOLATILE)
        # The callback is never invoked: nothing spins this node.
        self._sub = self._node.create_subscription(
            msg_type, topic, lambda _m: None, qos, raw=True)

    def _drain(self) -> dict:
        """Everything queued since the last read: newest bytes per stream,
        in arrival order of each stream's newest message."""
        newest: dict = {}
        with self._lock, self._sub.handle:
            while True:
                got = self._sub.handle.take_message(self._msg_type, True)
                if got is None:
                    break
                key = self._stream_key(got[0]) if self._stream_key else 0
                newest.pop(key, None)
                newest[key] = got[0]
        return newest

    def take(self) -> Optional[Any]:
        newest = self._drain()
        if not newest:
            return None
        return deserialize_message(list(newest.values())[-1], self._msg_type)

    def take_streams(self) -> List[Any]:
        return [deserialize_message(b, self._msg_type) for b in self._drain().values()]

    def destroy(self) -> None:
        with self._lock:
            self._node.destroy_subscription(self._sub)
            self._node.destroy_node()
