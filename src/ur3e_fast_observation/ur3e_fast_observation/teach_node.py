#!/usr/bin/env python3
"""Teach (record) UR3e observation joint poses by snapshotting /joint_states.

This is a strictly read-only node: it never publishes a command to the arm.
The operator moves the robot to a desired observation viewpoint by hand
(freedrive) or with the existing PTP/LIN commander, then captures the
current joint angles into a YAML file. The resulting file is the input to
``replay_node``.

YAML schema
-----------
.. code-block:: yaml

    joint_names:
      - shoulder_pan_joint
      - shoulder_lift_joint
      - elbow_joint
      - wrist_1_joint
      - wrist_2_joint
      - wrist_3_joint
    poses:
      view_01:
        positions: [0.0, -1.57, 0.0, -1.57, 0.0, 0.0]
        captured_at: 1777308000.0
      seed_view:
        positions: [0.1, -1.4, 0.2, -1.5, 0.0, 0.1]
        captured_at: 1777308300.0

Parameters
----------
output_file : str
    Absolute path to the YAML file. Created if it does not exist; new
    captures append to the existing entries (overwrite by re-using a name).
joint_states_topic : str
    Topic providing ``sensor_msgs/JointState``. Defaults to ``/joint_states``.
arm_joint_names : str[]
    The arm joints to capture. Other joints in the message (e.g. gripper)
    are ignored.

Interactive commands
--------------------
- ``[Enter]``       capture the current joint state with an auto label
- ``name=LABEL``    capture with the given label (overwrites if exists)
- ``list``          show stored entries
- ``drop=LABEL``    remove a labelled entry
- ``quit``          exit
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import Dict, List, Optional

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState


ARM_JOINT_NAMES_DEFAULT: List[str] = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


class TeachNode(Node):
    """Read-only node that snapshots joint states into a YAML registry."""

    def __init__(self) -> None:
        super().__init__('ur3e_fast_observation_teach')

        self.declare_parameter('output_file', 'observation_poses.yaml')
        self.declare_parameter('joint_states_topic', '/joint_states')
        self.declare_parameter('arm_joint_names', ARM_JOINT_NAMES_DEFAULT)

        self._output: str = str(self.get_parameter('output_file').value)
        self._topic: str = str(self.get_parameter('joint_states_topic').value)
        self._arm_joints: List[str] = list(
            self.get_parameter('arm_joint_names').value
        )

        self._latest: Optional[JointState] = None
        self._lock = threading.Lock()
        self._poses: Dict[str, Dict] = {}
        self._auto_idx: int = 0

        self.create_subscription(
            JointState, self._topic, self._on_joint_state, 50
        )

        self._load_existing()
        self.get_logger().info(
            f"Teach tool ready. Topic='{self._topic}'. "
            f"Output='{self._output}'. Existing entries: {len(self._poses)}."
        )

    # ------------------------------------------------------------------ I/O

    def _load_existing(self) -> None:
        if not os.path.exists(self._output):
            return
        try:
            with open(self._output, 'r') as fh:
                data = yaml.safe_load(fh) or {}
            poses = data.get('poses') or {}
            if isinstance(poses, dict):
                self._poses = {str(k): dict(v) for k, v in poses.items()}
        except Exception as exc:
            self.get_logger().warn(
                f"Failed to read existing file ({exc}); starting fresh."
            )
            self._poses = {}
        # Initialise auto-counter past any view_NN entries already present.
        max_auto = 0
        for label in self._poses:
            if label.startswith('view_'):
                tail = label[len('view_'):]
                if tail.isdigit():
                    max_auto = max(max_auto, int(tail))
        self._auto_idx = max_auto

    def _save(self) -> None:
        out_dir = os.path.dirname(os.path.abspath(self._output))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        payload = {
            'joint_names': list(self._arm_joints),
            'poses': dict(self._poses),
        }
        tmp_path = f'{self._output}.tmp'
        with open(tmp_path, 'w') as fh:
            yaml.safe_dump(payload, fh, sort_keys=False)
        os.replace(tmp_path, self._output)

    # -------------------------------------------------------------- Helpers

    def _on_joint_state(self, msg: JointState) -> None:
        with self._lock:
            self._latest = msg

    def _snapshot_arm(self) -> Optional[List[float]]:
        with self._lock:
            msg = self._latest
        if msg is None:
            return None
        name_to_pos = dict(zip(msg.name, msg.position))
        positions: List[float] = []
        for joint in self._arm_joints:
            if joint not in name_to_pos:
                self.get_logger().warn(
                    f"Joint '{joint}' missing in /joint_states. "
                    f"Available: {list(name_to_pos.keys())}"
                )
                return None
            positions.append(float(name_to_pos[joint]))
        return positions

    @staticmethod
    def _format_deg(positions: List[float]) -> str:
        return ', '.join(f'{p * 180.0 / math.pi:+7.2f}' for p in positions)

    # ---------------------------------------------------------- Commands

    def capture(self, label: Optional[str]) -> None:
        positions = self._snapshot_arm()
        if positions is None:
            print('[teach] no /joint_states received yet; try again.')
            return
        if not label:
            self._auto_idx += 1
            label = f'view_{self._auto_idx:02d}'
        self._poses[label] = {
            'positions': positions,
            'captured_at': time.time(),
        }
        self._save()
        print(f"[teach] saved '{label}' = [{self._format_deg(positions)}] deg")

    def list_entries(self) -> None:
        if not self._poses:
            print('[teach] no entries yet.')
            return
        print(f'[teach] {len(self._poses)} entries in {self._output}:')
        for label, entry in self._poses.items():
            positions = entry.get('positions', [])
            print(f'  {label:24s} [{self._format_deg(positions)}] deg')

    def drop(self, label: str) -> None:
        if label in self._poses:
            del self._poses[label]
            self._save()
            print(f"[teach] removed '{label}'")
        else:
            print(f"[teach] no such label: {label}")


def _input_loop(node: TeachNode) -> None:
    print(
        'Commands: [Enter]=capture | name=LABEL | list | drop=LABEL | quit'
    )
    while rclpy.ok():
        try:
            line = input('teach> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ('quit', 'q', 'exit'):
            break
        if line == '':
            node.capture(None)
        elif line == 'list':
            node.list_entries()
        elif line.startswith('name='):
            node.capture(line[len('name='):].strip())
        elif line.startswith('drop='):
            node.drop(line[len('drop='):].strip())
        else:
            print(
                'Unknown command. '
                'Use: [Enter] | name=LABEL | list | drop=LABEL | quit'
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeachNode()
    spin_thread = threading.Thread(
        target=rclpy.spin, args=(node,), daemon=True
    )
    spin_thread.start()
    # Allow a moment for the first joint_state message.
    time.sleep(0.5)
    try:
        _input_loop(node)
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
