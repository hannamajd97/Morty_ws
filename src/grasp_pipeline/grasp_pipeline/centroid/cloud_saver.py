#!/usr/bin/env python3
"""
cloud_saver.py
==============

Standalone debug utility: grabs one selected-object PointCloud2 off the topic
and writes it to disk (via Open3D) so you can inspect a captured cloud offline
(e.g. open it in a viewer, or feed it to GraspGen by hand). One-shot — saves
the first cloud it sees, then stops. Run with
``ros2 run grasp_pipeline cloud_saver``.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
import open3d as o3d
import numpy as np

class CloudSaver(Node):
    def __init__(self):
        super().__init__('cloud_saver')
        self.create_subscription(PointCloud2, '/owlvit/selected_cloud', self.callback, 10)
        self.once = False

    def callback(self, msg):
        if self.once:
            return
        self.once = True

        # Convert PointCloud2 → list of (x,y,z) tuples
        pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        points = np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float64)

        if points.size == 0:
            self.get_logger().warn("No points received in selected_cloud")
            rclpy.shutdown()
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud("selected_cloud.pcd", pcd)
        self.get_logger().info(f"Saved selected_cloud.pcd with {len(points)} points")
        rclpy.shutdown()


def main():
    rclpy.init()
    rclpy.spin(CloudSaver())

if __name__ == '__main__':
    main()
