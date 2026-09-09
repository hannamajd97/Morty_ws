#!/usr/bin/env python3
"""
cloud_metrics.py
================

Standalone debug node. Subscribes to a selected-object PointCloud2, transforms
it into a target frame (default base_link), and logs simple metrics (point
count, centroid). Handy for sanity-checking that the perception stage is
producing a sensible cloud in the robot frame before grasping. Not part of the
live pick path — run it manually with ``ros2 run grasp_pipeline cloud_metrics``.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from rclpy.duration import Duration
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
import tf2_ros
from tf2_geometry_msgs import do_transform_pose

class CloudMetrics(Node):
    def __init__(self):
        super().__init__("cloud_metrics")
        self.declare_parameter("cloud_topic", "/owlvit/selected_cloud")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("print_every_n", 1)
        self.declare_parameter("min_points", 30)

        self.cloud_topic   = self.get_parameter("cloud_topic").get_parameter_value().string_value
        self.target_frame  = self.get_parameter("target_frame").get_parameter_value().string_value
        self.print_every_n = int(self.get_parameter("print_every_n").get_parameter_value().integer_value or 1)
        self.min_points    = int(self.get_parameter("min_points").get_parameter_value().integer_value or 30)
        self.count = 0

        self.tf_buf = tf2_ros.Buffer(cache_time=Duration(seconds=10.0))
        self.tf_listener = tf2_ros.TransformListener(self.tf_buf, self, spin_thread=True)

        self.sub = self.create_subscription(PointCloud2, self.cloud_topic, self.on_cloud, 10)
        self.get_logger().info(f"Listening on {self.cloud_topic}; reporting in {self.target_frame}")

    def on_cloud(self, msg: PointCloud2):
        pts_list = []
        for rec in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            try:
                x, y, z = rec
            except Exception:
                continue
            pts_list.append((float(x), float(y), float(z)))

        if not pts_list:
            self.get_logger().warn("Selected cloud has 0 valid points.")
            return

        pts = np.asarray(pts_list, dtype=float)  # (N,3)
        n = pts.shape[0]
        if n < self.min_points:
            self.get_logger().warn(f"Very sparse cloud: only {n} points (min_points={self.min_points}).")

        cen  = pts.mean(axis=0)
        mins = pts.min(axis=0); maxs = pts.max(axis=0)
        size = maxs - mins
        cam_dist = float(np.linalg.norm(cen))

        cen_ps = PoseStamped()
        cen_ps.header = msg.header
        cen_ps.pose.position.x, cen_ps.pose.position.y, cen_ps.pose.position.z = map(float, cen.tolist())
        cen_ps.pose.orientation.w = 1.0

        try:
            tf = self.tf_buf.lookup_transform(self.target_frame, msg.header.frame_id, Time(), Duration(seconds=0.25))
            cen_in_base_ps = do_transform_pose(cen_ps, tf)
            b = cen_in_base_ps.pose.position
            base_dist = float(np.linalg.norm([b.x, b.y, b.z]))
            cen_base_str = f"base=({b.x:.3f},{b.y:.3f},{b.z:.3f}) m"
        except Exception as e:
            base_dist = float("nan")
            cen_base_str = f"base=N/A ({e})"

        self.count += 1
        if self.count % self.print_every_n == 0:
            self.get_logger().info(
                f"N={n} | Centroid@{msg.header.frame_id}={cen} m | size≈{size} m | "
                f"dist(camera)={cam_dist:.3f} m | dist({self.target_frame})={base_dist:.3f} m | {cen_base_str}"
            )

def main():
    rclpy.init()
    rclpy.spin(CloudMetrics())
    rclpy.shutdown()

if __name__ == "__main__":
    main()
