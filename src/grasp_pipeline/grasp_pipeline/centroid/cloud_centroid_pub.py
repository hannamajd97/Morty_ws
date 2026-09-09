#!/usr/bin/env python3
"""
cloud_centroid_pub.py
=====================

Standalone helper node: subscribes to a selected-object PointCloud2, computes
its centroid, and republishes it as a PoseStamped. Useful as a cheap "where is
the object" signal (e.g. for a coarse pre-grasp approach or for debugging)
without running full grasp generation. Run with
``ros2 run grasp_pipeline cloud_centroid_pub``.
"""
import rclpy
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped

# we will try both APIs
from sensor_msgs_py import point_cloud2 as pc2

def to_xyz_array(msg: PointCloud2) -> np.ndarray:
    """
    Return an (N,3) float32 array of [x,y,z] from a PointCloud2.
    Tries the NumPy API first, then falls back to the generator.
    Removes NaNs automatically.
    """
    # --- Fast path: NumPy API (if available in your install)
    try:
        # read_points_numpy returns a structured array with fields 'x','y','z'
        read_points_numpy = pc2.read_points_numpy  # may not exist on some distros
        arr = read_points_numpy(msg, field_names=("x", "y", "z"), reshape_organized_cloud=False)
        # Some distros give a structured array; column_stack is the safest
        xyz = np.column_stack((arr['x'].astype(np.float32),
                               arr['y'].astype(np.float32),
                               arr['z'].astype(np.float32)))
        # drop NaNs
        return xyz[~np.isnan(xyz).any(axis=1)]
    except Exception:
        pass

    # --- Fallback: generator API (robust to record/tuple/scalar variants)
    rows = []
    for p in pc2.read_points(msg, field_names=('x', 'y', 'z'), skip_nans=True):
        # p can be tuple/list/numpy.void/record/scalar; handle all cases
        try:
            # tuple/list path
            if isinstance(p, (tuple, list)):
                if len(p) >= 3:
                    rows.append([float(p[0]), float(p[1]), float(p[2])])
                continue
            # numpy record with named fields
            x = float(p['x']); y = float(p['y']); z = float(p['z'])
            rows.append([x, y, z])
        except Exception:
            # last resort: ignore malformed point
            continue
    if not rows:
        return np.empty((0, 3), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)

class CloudCentroidPub(Node):
    def __init__(self):
        super().__init__('cloud_centroid_pub')
        self.declare_parameter('input_cloud_topic', '/owlvit/selected_cloud')
        topic = self.get_parameter('input_cloud_topic').get_parameter_value().string_value

        self.sub = self.create_subscription(PointCloud2, topic, self.cb, 10)
        self.pub = self.create_publisher(PoseStamped, '/graspgen/centroid', 10)
        self.get_logger().info(f"Listening to {topic} and publishing centroid → /graspgen/centroid")

    def cb(self, msg: PointCloud2):
        pts = to_xyz_array(msg)
        n = pts.shape[0]
        if n == 0:
            self.get_logger().warn('selected_cloud has 0 valid XYZ points')
            return

        c = np.nanmean(pts, axis=0)  # (x,y,z)
        ps = PoseStamped()
        ps.header = msg.header           # keep the cloud’s frame_id + stamp
        ps.pose.position.x = float(c[0])
        ps.pose.position.y = float(c[1])
        ps.pose.position.z = float(c[2])
        ps.pose.orientation.w = 1.0      # identity orientation for the test
        self.pub.publish(ps)
        # occasional debug print
        if (self.get_clock().now().nanoseconds // 1_000_000_000) % 2 == 0:
            self.get_logger().info(f"Centroid from {n} pts → ({c[0]:.3f}, {c[1]:.3f}, {c[2]:.3f}) in frame {ps.header.frame_id}")

def main():
    rclpy.init()
    rclpy.spin(CloudCentroidPub())
    rclpy.shutdown()

if __name__ == '__main__':
    main()
