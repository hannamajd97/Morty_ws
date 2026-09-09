#!/usr/bin/env python3
"""
pose_to_base.py
===============

Small relay node: subscribes to a grasp pose published in one TF frame
(typically the camera frame), looks up the transform to the robot base, and
republishes the same pose expressed in the base frame.

Why it exists: some downstream consumers want grasp poses in ``base_link`` for
planning/visualisation, but the perception side publishes them in the camera
frame. Rather than letting every consumer chase TF, this node does the lookup
once and republishes the rotated pose on a separate topic.

The small ``quaternion_from_euler`` helper below is a no-dependency
implementation so this script does not require ``tf_transformations`` to be
installed just to compose a constant orientation offset.

Run standalone:
    ros2 run grasp_pipeline pose_to_base
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import (
    QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy
)
from geometry_msgs.msg import PoseStamped
import tf2_ros
import tf2_geometry_msgs  # register PoseStamped with TF2


# --- tiny quaternion helpers (no external deps) ---
def quaternion_from_euler(roll, pitch, yaw):
    cr = math.cos(roll * 0.5); sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5); sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5);   sy = math.sin(yaw * 0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return x, y, z, w

def euler_from_quaternion(q):
    x, y, z, w = q
    # roll
    sinr_cosp = 2*(w*x + y*z); cosr_cosp = 1 - 2*(x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch
    sinp = 2*(w*y - z*x); sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)
    # yaw
    siny_cosp = 2*(w*z + x*y); cosy_cosp = 1 - 2*(y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw
# --------------------------------------------------

class PoseToBase(Node):
    def __init__(self):
        super().__init__("pose_to_base")

        # Parameters
        self.declare_parameter("in_topic", "/graspgen/best_pose")
        self.declare_parameter("out_topic", "/graspgen/best_pose_base")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("dx", 0.0)
        self.declare_parameter("dy", 0.0)
        self.declare_parameter("dz", 0.0)
        self.declare_parameter("roll_deg", 0.0)
        self.declare_parameter("pitch_deg", 0.0)
        self.declare_parameter("yaw_deg", 0.0)

        in_topic       = self.get_parameter("in_topic").value
        out_topic      = self.get_parameter("out_topic").value
        self.tgt_frame = self.get_parameter("target_frame").value

        qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.pub = self.create_publisher(PoseStamped, out_topic, qos)

        self.tfbuf = tf2_ros.Buffer()
        self.tfl   = tf2_ros.TransformListener(self.tfbuf, self)

        # keep the latest RAW (camera-frame) pose; re-transform it periodically
        self.last_pose_in: PoseStamped | None = None
        self.create_subscription(PoseStamped, in_topic, self.cb, 10)
        self.create_timer(1.0, self.timer_cb)  # re-transform every 1 s

        self.get_logger().info(f"Relaying {in_topic} -> {out_topic} in {self.tgt_frame}")

    def cb(self, msg: PoseStamped):
        self.last_pose_in = msg
        self.publish_transformed(msg)

    def timer_cb(self):
        if self.last_pose_in is not None:
            self.publish_transformed(self.last_pose_in)

    def publish_transformed(self, msg: PoseStamped):
        # stamp slightly in the past so TF always has data
        msg.header.stamp = (self.get_clock().now() - Duration(seconds=0.1)).to_msg()
        try:
            base_pose: PoseStamped = self.tfbuf.transform(
                msg, self.tgt_frame, timeout=Duration(seconds=0.5)
            )
        except Exception as e:
            self.get_logger().warn(f"TF transform failed {msg.header.frame_id}->{self.tgt_frame}: {e}")
            return

        # optional correction
        self._apply_correction(base_pose)
        base_pose.header.frame_id = self.tgt_frame
        self.pub.publish(base_pose)

    def _apply_correction(self, ps: PoseStamped):
        dx = float(self.get_parameter("dx").value)
        dy = float(self.get_parameter("dy").value)
        dz = float(self.get_parameter("dz").value)
        r  = math.radians(float(self.get_parameter("roll_deg").value))
        p  = math.radians(float(self.get_parameter("pitch_deg").value))
        y  = math.radians(float(self.get_parameter("yaw_deg").value))

        ps.pose.position.x += dx
        ps.pose.position.y += dy
        ps.pose.position.z += dz

        if abs(r) > 1e-6 or abs(p) > 1e-6 or abs(y) > 1e-6:
            q = ps.pose.orientation
            r0, p0, y0 = euler_from_quaternion((q.x, q.y, q.z, q.w))
            qx, qy, qz, qw = quaternion_from_euler(r0 + r, p0 + p, y0 + y)
            ps.pose.orientation.x = qx
            ps.pose.orientation.y = qy
            ps.pose.orientation.z = qz
            ps.pose.orientation.w = qw

def main():
    rclpy.init()
    node = PoseToBase()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
