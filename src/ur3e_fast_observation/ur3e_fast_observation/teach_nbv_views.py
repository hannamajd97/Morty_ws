#!/usr/bin/env python3
"""Teach the four NBV (Next-Best-View) observation poses on the real arm.

Phase 2 of the redesigned active-perception viewpoints. The four targets
are computed mathematically (camera lookAt → TCP transform) and shipped
in this file as Cartesian goals. This tool drives the arm via the
existing PTP commander to each goal, reads back the achieved joint
state, and emits an SRDF-ready snippet.

Usage:

    # 1. Make sure the robot is up and at initgrasp.
    # 2. Make sure the PTP commander launch is available.
    # 3. Run:
    ros2 run ur3e_fast_observation teach_nbv_views \\
        --ros-args -p ptp_commander_executable:=/path/to/observation_ptp_commander

After confirming each move visually, the script saves the achieved joints
to ~/nbv_views.yaml. Paste the entries into
ur3e_with_gripper_3d.srdf as <group_state name="nbv_*">.

The 4 viewpoints (workspace center 0.12, -0.29, 0.60; sphere radius 0.28 m):
  - nbv_front  — azimuth   0°, elevation 50° (parallel to initgrasp)
  - nbv_right  — azimuth +30°, elevation 50°
  - nbv_left   — azimuth -30°, elevation 50°
  - nbv_top    — azimuth   0°, elevation 70° (nearly straight down)
"""

from __future__ import annotations

import math
import os
import shlex
import subprocess
import threading
import time
from typing import Dict, List, Optional

import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import JointState


ARM_JOINTS = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]


# Designed TCP targets (computed from camera-lookAt math).
# Each entry: (name, position_xyz, orientation_xyzw)
NBV_TARGETS = [
    ("nbv_front",
     (0.135, -0.307, 0.784),
     (-0.6645, 0.6645, 0.2418, -0.2418)),
    ("nbv_right",
     (0.142, -0.297, 0.784),
     (0.8138, -0.4698, -0.1710, 0.2962)),
    ("nbv_left",
     (0.125, -0.313, 0.784),
     (-0.4698, 0.8138, 0.2962, -0.1710)),
    ("nbv_top",
     (0.071, -0.307, 0.778),
     (-0.6964, 0.6964, 0.1228, -0.1228)),
]


class TeachNbvViewsNode(Node):
    def __init__(self) -> None:
        super().__init__("teach_nbv_views")
        self.declare_parameter(
            "ptp_commander_executable",
            "/home/deep/morty_ws/install/lib/"
            "ur3e_gripper_3d_moveit_commander/observation_ptp_commander",
        )
        self.declare_parameter(
            "output_yaml",
            os.path.join(os.path.expanduser("~"), "nbv_views.yaml"),
        )
        self.declare_parameter("velocity_scaling", 0.1)
        self.declare_parameter("acceleration_scaling", 0.1)
        self.declare_parameter("planning_time", 5.0)
        self.declare_parameter("settle_seconds", 1.0)

        self._ptp_exe = str(self.get_parameter("ptp_commander_executable").value)
        self._output_yaml = str(self.get_parameter("output_yaml").value)
        self._vel = float(self.get_parameter("velocity_scaling").value)
        self._acc = float(self.get_parameter("acceleration_scaling").value)
        self._plan_t = float(self.get_parameter("planning_time").value)
        self._settle_s = float(self.get_parameter("settle_seconds").value)

        self._latest_joint_state: Optional[JointState] = None
        self._lock = threading.Lock()
        self.create_subscription(JointState, "/joint_states", self._on_js, 50)

    def _on_js(self, msg: JointState) -> None:
        with self._lock:
            self._latest_joint_state = msg

    def _read_arm_joints(self, timeout_s: float = 2.0) -> Optional[Dict[str, float]]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with self._lock:
                msg = self._latest_joint_state
            if msg is not None:
                name_to_pos = dict(zip(msg.name, msg.position))
                if all(j in name_to_pos for j in ARM_JOINTS):
                    return {j: float(name_to_pos[j]) for j in ARM_JOINTS}
            time.sleep(0.05)
        return None

    def drive_to_pose(
        self, name: str, xyz, quat_xyzw, timeout_s: float = 60.0
    ) -> bool:
        """Spawn the PTP commander to drive the arm to a Cartesian pose."""
        if not os.path.exists(self._ptp_exe):
            self.get_logger().error(
                f"PTP commander executable not found: {self._ptp_exe}"
            )
            return False
        cmd = [
            self._ptp_exe,
            "--ros-args",
            "-p", "use_pose_target:=true",
            "-p", "execute:=true",
            "-p", "one_shot:=true",
            "-p", f"velocity_scaling:={self._vel}",
            "-p", f"acceleration_scaling:={self._acc}",
            "-p", f"planning_time:={self._plan_t}",
            "-p", f"reference_frame:=base_link",
            "-p", "end_effector_link:=gripper_tcp",
            "-p", "planning_pipeline:=pilz_industrial_motion_planner",
            "-p", "planner_id:=PTP",
            "-p", "allow_lin_fallback:=true",
            "-p", "fallback_planning_pipeline:=pilz_industrial_motion_planner",
            "-p", "fallback_planner_id:=LIN",
            "-p", f"target_name:={name}",
            "-p", f"pose_x:={xyz[0]}",
            "-p", f"pose_y:={xyz[1]}",
            "-p", f"pose_z:={xyz[2]}",
            "-p", f"pose_qx:={quat_xyzw[0]}",
            "-p", f"pose_qy:={quat_xyzw[1]}",
            "-p", f"pose_qz:={quat_xyzw[2]}",
            "-p", f"pose_qw:={quat_xyzw[3]}",
        ]
        self.get_logger().info(
            f"Driving to '{name}': xyz={xyz}, quat={quat_xyzw}"
        )
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_s,
                env=os.environ.copy(),
            )
        except subprocess.TimeoutExpired:
            self.get_logger().warn(f"Move to '{name}' timed out.")
            return False
        if completed.returncode != 0:
            self.get_logger().warn(
                f"Move to '{name}' failed (rc={completed.returncode}). "
                f"stderr=\n{completed.stderr.strip()[-1500:]}"
            )
            return False
        time.sleep(self._settle_s)
        return True

    @staticmethod
    def _format_joint_block(name: str, joints: Dict[str, float]) -> str:
        """Emit an SRDF group_state snippet."""
        lines = [f'  <group_state name="{name}" group="ur3e_manipulator">']
        for j in ARM_JOINTS:
            lines.append(f'    <joint name="{j}" value="{joints[j]:.10f}"/>')
        lines.append("  </group_state>")
        return "\n".join(lines)

    def run(self) -> int:
        captured: Dict[str, Dict[str, float]] = {}
        for name, xyz, quat in NBV_TARGETS:
            ok = self.drive_to_pose(name, xyz, quat)
            if not ok:
                self.get_logger().warn(
                    f"Skipping joint capture for '{name}' (move failed)."
                )
                continue
            joints = self._read_arm_joints()
            if joints is None:
                self.get_logger().warn(
                    f"No joint state available for '{name}'; skipping."
                )
                continue
            captured[name] = joints
            self.get_logger().info(
                f"Captured '{name}': "
                + ", ".join(f"{j}={v:+.4f}" for j, v in joints.items())
            )

        if not captured:
            self.get_logger().error("No poses captured.")
            return 1

        # Save YAML
        out_dir = os.path.dirname(os.path.abspath(self._output_yaml))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self._output_yaml, "w") as fh:
            yaml.safe_dump(
                {"poses": {name: dict(j) for name, j in captured.items()}},
                fh,
                sort_keys=False,
            )
        self.get_logger().info(f"Wrote {self._output_yaml}")

        print("\n" + "=" * 80)
        print("SRDF group_state entries (paste into ur3e_with_gripper_3d.srdf):")
        print("=" * 80)
        for name, j in captured.items():
            print(self._format_joint_block(name, j))
            print()
        return 0


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeachNbvViewsNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()
    time.sleep(0.5)
    rc = 0
    try:
        rc = node.run()
    finally:
        if rclpy.ok():
            rclpy.shutdown()
        node.destroy_node()
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
