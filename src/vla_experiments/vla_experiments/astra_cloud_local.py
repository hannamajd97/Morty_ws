"""Build an XYZRGB point cloud on the laptop from Astra streams relayed across the dock link.

Preferred inputs (full resolution, compressed by the shuttle relay):
    /astra/depth/compressed_slow   sensor_msgs/CompressedImage, compressedDepth PNG 16UC1 (mm)
    /astra/color/compressed_slow   sensor_msgs/CompressedImage, JPEG
Fallback inputs (subsampled raw): /astra/depth/image_slow, /astra/color/image_slow
Always:      /astra/depth/camera_info  (intrinsics; rescaled to the incoming depth size)
Output:      /astra/depth/points_local  PointCloud2 xyz(+rgb), RELIABLE, in the depth frame.
With the driver's depth_registration on, the depth frame is astra_color_optical_frame and
colour pixel = depth pixel * (colour size / depth size).
"""
import struct, time
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, PointCloud2, PointField


def decode_compressed_depth(m: CompressedImage):
    """image_transport compressedDepth: 12-byte ConfigHeader (int32 format, float depthQuantA, depthQuantB) + PNG."""
    fmt, qa, qb = struct.unpack("<iff", bytes(m.data[:12]))
    img = cv2.imdecode(np.frombuffer(bytes(m.data[12:]), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.dtype == np.uint16:
        return img.astype(np.float32) / 1000.0
    if img.dtype == np.uint8 and qa != 0.0:  # inverse-depth quantised 32FC1
        d = np.zeros(img.shape, np.float32); nz = img > 0
        d[nz] = qa / (img[nz].astype(np.float32) - qb)
        return d
    return img.astype(np.float32)


class AstraCloudLocal(Node):
    def __init__(self):
        super().__init__("astra_cloud_local")
        self.declare_parameter("cloud_topic", "/astra/depth/points_local")
        self.declare_parameter("info_topic", "/astra/depth/camera_info")
        self.declare_parameter("max_range_m", 4.0)
        self.declare_parameter("color_max_age_s", 1.0)
        self.declare_parameter("use_compressed", True)
        self.info = None
        self.color = None  # (wall_time, rgb HxWx3)
        self.count = 0
        self.pub = self.create_publisher(PointCloud2, self.get_parameter("cloud_topic").value,
                                         QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE, history=HistoryPolicy.KEEP_LAST))
        self.create_subscription(CameraInfo, self.get_parameter("info_topic").value, self._on_info, qos_profile_sensor_data)
        if self.get_parameter("use_compressed").value:
            self.create_subscription(CompressedImage, "/astra/depth/compressed_slow", self._on_cdepth, qos_profile_sensor_data)
            self.create_subscription(CompressedImage, "/astra/color/compressed_slow", self._on_ccolor, qos_profile_sensor_data)
        else:
            self.create_subscription(Image, "/astra/depth/image_slow", self._on_depth, qos_profile_sensor_data)
            self.create_subscription(Image, "/astra/color/image_slow", self._on_color, qos_profile_sensor_data)

    def _on_info(self, m):
        self.info = (m.width, m.height, m.k[0], m.k[4], m.k[2], m.k[5])

    def _on_color(self, m):
        if m.encoding not in ("rgb8", "bgr8"):
            return
        a = np.frombuffer(bytes(m.data), np.uint8).reshape(m.height, m.width, 3)
        self.color = (time.time(), a[:, :, ::-1] if m.encoding == "bgr8" else a)

    def _on_ccolor(self, m):
        bgr = cv2.imdecode(np.frombuffer(bytes(m.data), np.uint8), cv2.IMREAD_COLOR)
        if bgr is not None:
            self.color = (time.time(), bgr[:, :, ::-1])

    def _on_depth(self, m):
        if m.encoding == "16UC1":
            d = np.frombuffer(bytes(m.data), np.uint16).reshape(m.height, m.width).astype(np.float32) / 1000.0
        elif m.encoding == "32FC1":
            d = np.frombuffer(bytes(m.data), np.float32).reshape(m.height, m.width)
        else:
            return
        self._publish(d, m.header)

    def _on_cdepth(self, m):
        d = decode_compressed_depth(m)
        if d is not None:
            self._publish(d, m.header)

    def _publish(self, d, header):
        if self.info is None:
            return
        h, w = d.shape
        w0, h0, fx, fy, cx, cy = self.info
        s = w0 / w
        fx, fy, cx, cy = fx / s, fy / s, cx / s, cy / s
        v, u = np.mgrid[0:h, 0:w].astype(np.float32)
        valid = (d > 0.05) & (d < float(self.get_parameter("max_range_m").value))
        z = d[valid]; x = (u[valid] - cx) / fx * z; y = (v[valid] - cy) / fy * z
        pts = np.stack([x, y, z], axis=1).astype(np.float32)
        out = PointCloud2(); out.header = header
        out.header.stamp = self.get_clock().now().to_msg()  # restamped with laptop time: the Astra frames are static, so no clock dependency
        out.height = 1; out.width = pts.shape[0]; out.is_dense = True; out.is_bigendian = False
        col = self.color
        fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate("xyz")]
        if col is not None and time.time() - col[0] < float(self.get_parameter("color_max_age_s").value):
            rgb = col[1]; sy = rgb.shape[0] / h; sx = rgb.shape[1] / w
            vi = np.clip((v[valid] * sy).astype(np.int32), 0, rgb.shape[0] - 1)
            ui = np.clip((u[valid] * sx).astype(np.int32), 0, rgb.shape[1] - 1)
            c = rgb[vi, ui].astype(np.uint32)
            packed = (c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]
            buf = np.zeros((pts.shape[0], 4), dtype=np.float32); buf[:, :3] = pts; buf[:, 3] = packed.view(np.float32)
            fields.append(PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1))
            out.fields = fields; out.point_step = 16; out.row_step = 16 * out.width; out.data = buf.tobytes()
        else:
            out.fields = fields; out.point_step = 12; out.row_step = 12 * out.width; out.data = pts.tobytes()
        self.pub.publish(out)
        self.count += 1
        if self.count == 1:
            self.get_logger().info(f"first cloud: {out.width} points from {w}x{h} depth (scale {s:.1f}), colour={'yes' if out.point_step == 16 else 'no'}, frame {header.frame_id}")


def main(args=None):
    rclpy.init(args=args)
    n = AstraCloudLocal()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node(); rclpy.try_shutdown()


if __name__ == "__main__":
    main()
