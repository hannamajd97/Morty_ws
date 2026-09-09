#!/usr/bin/env python3
"""Publish a single static table collision box into the MoveIt planning scene.

Rationale: the previous perception-driven obstacle publisher republished a
large PlanningScene diff at 2 Hz (convex hull meshes for clustered non-target
points). That caused ~40 s planning latency because move_group had to
re-integrate a large scene every time a PTP request arrived. For our setup,
observation viewpoints are hand-taught (already collision-free) and the grasp
executor clears world obstacles before each LIN, so the only collision check
that actually matters is "don't drive the arm below the table".

This node publishes one CollisionObject (the table) in base_link once, then
exits its timer. No republishing, no churn, no lock contention.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import CollisionObject, PlanningScene


class StaticTablePublisher(Node):
    def __init__(self) -> None:
        super().__init__("scene_obstacle_publisher")

        self.declare_parameter("planning_scene_topic", "/planning_scene")
        self.declare_parameter("table_frame", "base_link")
        self.declare_parameter("table_id", "static_table")
        # Default: a generous slab under the robot's workspace. Adjust per rig.
        self.declare_parameter("table_center_x", 0.0)
        self.declare_parameter("table_center_y", -0.30)
        self.declare_parameter("table_center_z", 0.48)  # top at ~0.53 with 0.10 thickness
        self.declare_parameter("table_size_x", 1.20)
        self.declare_parameter("table_size_y", 1.00)
        self.declare_parameter("table_size_z", 0.10)
        self.declare_parameter("publish_attempts", 5)
        self.declare_parameter("publish_interval_s", 0.5)

        topic = str(self.get_parameter("planning_scene_topic").value)
        self._table_frame = str(self.get_parameter("table_frame").value)
        self._table_id = str(self.get_parameter("table_id").value)
        self._cx = float(self.get_parameter("table_center_x").value)
        self._cy = float(self.get_parameter("table_center_y").value)
        self._cz = float(self.get_parameter("table_center_z").value)
        self._sx = float(self.get_parameter("table_size_x").value)
        self._sy = float(self.get_parameter("table_size_y").value)
        self._sz = float(self.get_parameter("table_size_z").value)
        self._attempts_left = int(self.get_parameter("publish_attempts").value)

        self.pub_scene = self.create_publisher(PlanningScene, topic, 10)

        interval = float(self.get_parameter("publish_interval_s").value)
        self._timer = self.create_timer(interval, self._publish_once)

        self.get_logger().info(
            "Static table publisher ready. frame='%s' center=(%.2f, %.2f, %.2f) "
            "size=(%.2f, %.2f, %.2f), will publish %d times on '%s'."
            % (self._table_frame, self._cx, self._cy, self._cz,
               self._sx, self._sy, self._sz, self._attempts_left, topic)
        )

    def _build_scene(self) -> PlanningScene:
        co = CollisionObject()
        co.id = self._table_id
        co.header.frame_id = self._table_frame
        co.operation = CollisionObject.ADD

        prim = SolidPrimitive()
        prim.type = SolidPrimitive.BOX
        prim.dimensions = [self._sx, self._sy, self._sz]
        co.primitives.append(prim)

        pose = Pose()
        pose.position.x = self._cx
        pose.position.y = self._cy
        pose.position.z = self._cz
        pose.orientation.w = 1.0
        co.primitive_poses.append(pose)

        scene = PlanningScene()
        scene.is_diff = True
        scene.world.collision_objects = [co]
        return scene

    def _publish_once(self) -> None:
        if self._attempts_left <= 0:
            self._timer.cancel()
            self.get_logger().info("Static table published; no more updates.")
            return
        self.pub_scene.publish(self._build_scene())
        self._attempts_left -= 1


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StaticTablePublisher()
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
