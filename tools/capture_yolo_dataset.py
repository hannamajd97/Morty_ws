#!/usr/bin/env python3
"""Capture ROS camera frames for YOLO fine-tuning datasets.

The tool subscribes to a ROS2 image topic, saves BGR frames as JPG/PNG, and
writes a manifest CSV with the scene metadata you enter on the command line.
It is intentionally simple: collect images first, label them later in CVAT,
Roboflow, Label Studio, or any YOLO segmentation labeling tool.
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import os
import threading
import time
from pathlib import Path
from typing import Optional

import cv2

try:
    import rclpy
    from cv_bridge import CvBridge
    from rclpy.node import Node
    from sensor_msgs.msg import Image
except Exception as exc:  # pragma: no cover - import-time environment guard
    raise SystemExit(
        "Failed to import ROS2/cv_bridge. Run:\n"
        "  cd /home/deep/morty_ws\n"
        "  source install/setup.bash\n"
        f"Original error: {exc}"
    )


DEFAULT_TOPIC = "/camera/camera/color/image_raw"
DEFAULT_OUT_DIR = "/home/deep/morty_ws/src/dataset_capture"


class ImageCaptureNode(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("yolo_dataset_capture")
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._frame = None
        self._stamp_sec: Optional[float] = None
        self._frame_id = ""
        self.create_subscription(Image, topic, self._on_image, 10)

    def _on_image(self, msg: Image) -> None:
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"image conversion failed: {exc}")
            return
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        with self._lock:
            self._frame = frame.copy()
            self._stamp_sec = stamp
            self._frame_id = str(msg.header.frame_id)

    def latest(self):
        with self._lock:
            if self._frame is None:
                return None, None, ""
            return self._frame.copy(), self._stamp_sec, self._frame_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture ROS image frames for YOLO segmentation fine-tuning."
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="ROS image topic to subscribe to.")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Root output directory.")
    parser.add_argument(
        "--session",
        default="",
        help="Session folder name. Default: clutter_YYYYMMDD_HHMMSS.",
    )
    parser.add_argument(
        "--mode",
        choices=["manual", "timer"],
        default="manual",
        help="manual: press Enter per image. timer: save every --interval-s.",
    )
    parser.add_argument("--count", type=int, default=100, help="Number of images to save. 0 means unlimited.")
    parser.add_argument("--interval-s", type=float, default=2.0, help="Timer mode save interval.")
    parser.add_argument("--format", choices=["jpg", "png"], default="jpg", help="Output image format.")
    parser.add_argument("--jpg-quality", type=int, default=95, help="JPEG quality 1-100.")
    parser.add_argument(
        "--object-count",
        default="",
        help="Approximate number of objects in the scene, e.g. 3, 4, 5, 6.",
    )
    parser.add_argument(
        "--clutter-level",
        default="",
        help="Free text, e.g. easy, medium, hard, occluded.",
    )
    parser.add_argument(
        "--notes",
        default="",
        help="Free text notes stored in the manifest.",
    )
    parser.add_argument("--prefix", default="img", help="Filename prefix.")
    parser.add_argument(
        "--wait-timeout-s",
        type=float,
        default=10.0,
        help="Seconds to wait for the first image before failing.",
    )
    return parser.parse_args()


def _make_session_dir(args: argparse.Namespace) -> Path:
    session = args.session.strip()
    if not session:
        session = "clutter_" + _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(args.out_dir).expanduser().resolve() / session
    (out / "images").mkdir(parents=True, exist_ok=True)
    return out


def _open_manifest(path: Path):
    is_new = not path.exists()
    fh = path.open("a", newline="")
    writer = csv.DictWriter(
        fh,
        fieldnames=[
            "filename",
            "saved_at",
            "ros_stamp_sec",
            "frame_id",
            "topic",
            "object_count",
            "clutter_level",
            "notes",
        ],
    )
    if is_new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def _save_frame(
    *,
    frame,
    stamp_sec,
    frame_id: str,
    seq: int,
    args: argparse.Namespace,
    session_dir: Path,
    writer,
) -> Path:
    ext = args.format.lower()
    filename = f"{args.prefix}_{seq:04d}.{ext}"
    image_path = session_dir / "images" / filename
    params = []
    if ext == "jpg":
        params = [cv2.IMWRITE_JPEG_QUALITY, int(max(1, min(100, args.jpg_quality)))]
    ok = cv2.imwrite(str(image_path), frame, params)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {image_path}")
    writer.writerow(
        {
            "filename": str(Path("images") / filename),
            "saved_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "ros_stamp_sec": "" if stamp_sec is None else f"{float(stamp_sec):.9f}",
            "frame_id": frame_id,
            "topic": args.topic,
            "object_count": args.object_count,
            "clutter_level": args.clutter_level,
            "notes": args.notes,
        }
    )
    return image_path


def _wait_for_first_frame(node: ImageCaptureNode, timeout_s: float) -> None:
    t0 = time.monotonic()
    while rclpy.ok():
        frame, _, _ = node.latest()
        if frame is not None:
            return
        if time.monotonic() - t0 > timeout_s:
            raise TimeoutError("No image received. Check camera/topic and source install/setup.bash.")
        time.sleep(0.05)


def main() -> int:
    args = _parse_args()
    session_dir = _make_session_dir(args)
    manifest_path = session_dir / "manifest.csv"

    rclpy.init()
    node = ImageCaptureNode(args.topic)
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    manifest_fh = None
    saved = 0
    try:
        _wait_for_first_frame(node, args.wait_timeout_s)
        manifest_fh, writer = _open_manifest(manifest_path)
        print(f"output_dir={session_dir}")
        print(f"images_dir={session_dir / 'images'}")
        print(f"manifest={manifest_path}")
        print(f"topic={args.topic}")
        print("Only save frames you are willing to label later.")

        while rclpy.ok() and (args.count <= 0 or saved < args.count):
            if args.mode == "manual":
                remaining = "unlimited" if args.count <= 0 else str(args.count - saved)
                answer = input(f"[saved={saved}, remaining={remaining}] Enter=save, q=quit > ").strip().lower()
                if answer in {"q", "quit", "exit"}:
                    break
            else:
                time.sleep(max(0.05, float(args.interval_s)))

            frame, stamp_sec, frame_id = node.latest()
            if frame is None:
                print("No frame available yet; skipping.")
                continue
            saved += 1
            path = _save_frame(
                frame=frame,
                stamp_sec=stamp_sec,
                frame_id=frame_id,
                seq=saved,
                args=args,
                session_dir=session_dir,
                writer=writer,
            )
            manifest_fh.flush()
            print(f"saved {saved}: {path}")
    finally:
        if manifest_fh is not None:
            manifest_fh.close()
        node.destroy_node()
        rclpy.shutdown()

    print(f"done saved={saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
