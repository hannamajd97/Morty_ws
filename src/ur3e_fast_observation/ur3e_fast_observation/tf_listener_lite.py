"""Decimating /tf listener for Python nodes on a fast robot.

Why this exists (2026-09-08)
----------------------------
The UR driver publishes joint states at 800 Hz; robot_state_publisher and
the TCP-pose broadcaster turn every one into a ``/tf`` message.  The stock
``tf2_ros.TransformListener`` deserializes each of those ~830 messages per
second into Python objects on the node's executor.  In a process that also
runs YOLO, the segmenter, MediaPipe and GraspGen, that single subscription
holds the interpreter lock for most of every second: GraspGen inference
measured 0.18 s in a clean process and 5 s inside the node, YOLO 46 ms vs
1.2 s, depth frames arrived at 5-9 Hz instead of 30.  A probe that mirrored
the node went from 0.11 s to 3.0 s per inference by adding one such
listener, and back to 0.22 s with this one.

What it does
------------
* Owns a private node whose ``/tf`` subscription is **raw** (serialized
  bytes) and is spun by **no executor at all**.  DDS fills the reader's
  history on its own; a small thread drains it ``max_hz`` times per second
  with ``take_message`` — the same call rclpy's executor uses — so the
  process pays 50 wake-ups per second instead of 830, and never builds a
  Python wait set for them.
* Per drain, only the newest message of each publisher stream is
  deserialized and inserted into the shared ``tf2_ros.Buffer``.  Streams
  are told apart by the transform count in the CDR header
  (robot_state_publisher sends 6 transforms, the TCP broadcaster 1) so
  decimation can never starve one of them.  Interpolation in the buffer
  makes a 20 ms sample spacing invisible to the "latest transform" lookups
  this pipeline uses.
* ``/tf_static`` is drained the same way but taken in full (latched, a
  handful in total).
* The private node ignores the process's global ROS arguments: the launch
  remaps ``__node:=``, and with global arguments the private node would
  inherit the SAME name as its owner (seen 2026-09-08: two
  ``/yolo_live_select_grasp`` in the graph).

Drop-in for ``tf2_ros.TransformListener(buffer, node)``: same buffer, same
QoS, same authority string; nothing else in the node changes.
"""
from __future__ import annotations

import struct
import threading
import time
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile
from rclpy.serialization import deserialize_message
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer

_AUTHORITY = "default_authority"


class DecimatedTransformListener:
    """Feed ``buffer`` from /tf at ≤ ``max_hz`` per stream and /tf_static in full."""

    def __init__(self, buffer: Buffer, node: Node, max_hz: float = 50.0,
                 tf_topic: str = "/tf", tf_static_topic: str = "/tf_static") -> None:
        self.buffer = buffer
        self.max_hz = max(1.0, float(max_hz))
        self._period = 1.0 / self.max_hz
        self._taken = 0
        self._inserted = 0
        self._stop = threading.Event()

        # Same QoS the stock listener uses (tf2_ros/transform_listener.py).
        tf_qos = QoSProfile(depth=100, durability=DurabilityPolicy.VOLATILE,
                            history=HistoryPolicy.KEEP_LAST)
        static_qos = QoSProfile(depth=100, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                history=HistoryPolicy.KEEP_LAST)

        self._node = rclpy.create_node(
            f"{node.get_name()}_tf_listener",
            context=node.context,
            use_global_arguments=False,      # do NOT inherit the owner's __node:= remap
            start_parameter_services=False,
            enable_rosout=False,
        )
        # Callbacks are never invoked: nothing spins this node. The reader
        # queues arrive regardless and are drained below.
        self._sub_tf = self._node.create_subscription(
            TFMessage, tf_topic, self._never, tf_qos, raw=True)
        self._sub_static = self._node.create_subscription(
            TFMessage, tf_static_topic, self._never, static_qos, raw=False)
        self._thread = threading.Thread(target=self._drain_loop, name="tf-listener-lite", daemon=True)
        self._thread.start()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _never(_msg) -> None:  # pragma: no cover - no executor spins this node
        pass

    @staticmethod
    def _stream_key(data: bytes) -> int:
        """Transform count from the CDR payload: 4-byte encapsulation header,
        then the uint32 length of ``transforms``.  Byte 1 of the header gives
        the endianness.  Falls back to the payload size."""
        try:
            fmt = "<I" if data[1] == 0x01 else ">I"
            return struct.unpack_from(fmt, data, 4)[0]
        except (IndexError, struct.error):
            return len(data)

    def _drain_once(self) -> None:
        newest: Dict[int, bytes] = {}
        with self._sub_tf.handle:
            while True:
                got = self._sub_tf.handle.take_message(TFMessage, True)
                if got is None:
                    break
                newest[self._stream_key(got[0])] = got[0]
                self._taken += 1
        for data in newest.values():
            msg = deserialize_message(data, TFMessage)
            for t in msg.transforms:
                self.buffer.set_transform(t, _AUTHORITY)
            self._inserted += 1
        with self._sub_static.handle:
            while True:
                got = self._sub_static.handle.take_message(TFMessage, False)
                if got is None:
                    break
                for t in got[0].transforms:
                    self.buffer.set_transform_static(t, _AUTHORITY)

    def _drain_loop(self) -> None:
        next_t = time.monotonic()
        while not self._stop.is_set():
            try:
                self._drain_once()
            except Exception:
                # Context shutdown or a destroyed handle: leave quietly.
                if not rclpy.ok(context=self._node.context):
                    return
            next_t += self._period
            delay = next_t - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            else:
                next_t = time.monotonic()   # fell behind (GIL stall): resync, don't burst

    # ------------------------------------------------------------------ lifecycle
    def stats(self) -> Dict[str, int]:
        return {"inserted": self._inserted, "dropped": self._taken - self._inserted}

    def unregister(self) -> None:
        """Mirror of tf2_ros.TransformListener.unregister()."""
        self._stop.set()
        self._thread.join(timeout=1.0)
        self._node.destroy_subscription(self._sub_tf)
        self._node.destroy_subscription(self._sub_static)
        self._node.destroy_node()
