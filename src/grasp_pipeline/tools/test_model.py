#!/usr/bin/env python3
"""Quick YOLO segmentation model test for ROS images, live camera, or image files.

Examples:
  cd /home/deep/morty_ws/src/grasp_pipeline
  /home/deep/morty_ws/.venv_grasp/bin/python test_model.py --list-classes
  /home/deep/morty_ws/.venv_grasp/bin/python test_model.py --conf 0.01
  /home/deep/morty_ws/.venv_grasp/bin/python test_model.py --source ros --conf 0.01
  /home/deep/morty_ws/.venv_grasp/bin/python test_model.py --source image.jpg --conf 0.01
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import cv2
import numpy as np
from ultralytics import YOLO


OLD_MODEL = "/home/deep/morty_ws/src/grasp_pipeline/new_y26s_seg.pt"
NEW_9_CLASS_MODEL = "/home/deep/morty_ws/yolo26s-seg_24_04.pt"
DEFAULT_MODEL = OLD_MODEL
DEFAULT_CAMERA_INDICES = [6, 2, 4, 0]
CLASS_COLORS_BGR = {
    0: (0, 255, 255),      # tape_dispenser: yellow
    1: (255, 0, 255),      # cd_case: magenta
    2: (0, 255, 0),        # toy_faucet: green
    3: (255, 128, 0),      # kitchen_towel: blue-orange
    4: (0, 128, 255),      # nutmeg_jar: orange
    5: (0, 0, 255),        # retail_box: red
    6: (255, 255, 0),      # small_bottle: cyan
    7: (128, 255, 0),      # wooden_maze_puzzle: spring green
    8: (255, 0, 0),        # vitamin_c: blue
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test YOLO detections and scores.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path to YOLO .pt model.")
    parser.add_argument(
        "--preset",
        choices=["old", "new9"],
        default="old",
        help="Model preset. 'old' is the experiment model; 'new9' is the 9-class test model.",
    )
    parser.add_argument(
        "--source",
        default="camera",
        help="Use 'ros', 'camera', a camera index like '0', or an image/video path.",
    )
    parser.add_argument(
        "--ros-topic",
        default="/camera/camera/color/image_raw",
        help="ROS sensor_msgs/Image topic used when --source ros.",
    )
    parser.add_argument("--conf", type=float, default=0.01, help="Detection confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO image size.")
    parser.add_argument("--device", default="cuda", help="YOLO device, e.g. cuda or cpu.")
    parser.add_argument("--camera-index", type=int, default=None, help="Force a camera index.")
    parser.add_argument("--print-every", type=float, default=1.0, help="Seconds between console prints.")
    parser.add_argument("--save", default="", help="Optional path to save annotated image output.")
    parser.add_argument("--save-dir", default="", help="Optional directory to save sampled raw/annotated frames.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many processed frames; 0 means run forever.")
    parser.add_argument("--save-every", type=int, default=10, help="Save every N processed frames when --save-dir is set.")
    parser.add_argument("--no-window", action="store_true", help="Do not open an OpenCV display window.")
    parser.add_argument("--list-classes", action="store_true", help="Print classes and exit.")
    return parser.parse_args()


def detections_from_result(result) -> list[tuple[str, float, list[float]]]:
    names = result.names
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    detections = []
    for box in boxes:
        cls_id = int(box.cls.item())
        label = names.get(cls_id, str(cls_id))
        score = float(box.conf.item())
        bbox = [float(v) for v in box.xyxy[0].tolist()]
        detections.append((label, score, bbox))
    detections.sort(key=lambda item: item[1], reverse=True)
    return detections


def print_detections(detections: list[tuple[str, float, list[float]]]) -> None:
    if not detections:
        print("detections: none")
        return
    print("detections:")
    for label, score, bbox in detections:
        x1, y1, x2, y2 = bbox
        print(f"  {label:24s} score={score:.3f} bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")


def maybe_save_frame(
    save_dir: str,
    frame_idx: int,
    raw_bgr: np.ndarray,
    annotated_bgr: np.ndarray,
    detections: list[tuple[str, float, list[float]]],
) -> None:
    if not save_dir:
        return
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"frame_{frame_idx:04d}_raw.jpg"), raw_bgr)
    cv2.imwrite(str(out_dir / f"frame_{frame_idx:04d}_annotated.jpg"), annotated_bgr)
    txt = out_dir / f"frame_{frame_idx:04d}_detections.txt"
    with txt.open("w", encoding="utf-8") as f:
        if not detections:
            f.write("detections: none\n")
        else:
            for label, score, bbox in detections:
                x1, y1, x2, y2 = bbox
                f.write(f"{label} score={score:.6f} bbox=({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})\n")


def class_color_bgr(class_id: int) -> tuple[int, int, int]:
    if class_id in CLASS_COLORS_BGR:
        return CLASS_COLORS_BGR[class_id]
    # Deterministic fallback for future classes.
    return (
        int((37 * class_id + 60) % 256),
        int((97 * class_id + 130) % 256),
        int((173 * class_id + 200) % 256),
    )


def draw_result_bgr(frame_bgr: np.ndarray, result) -> np.ndarray:
    """Draw readable masks, boxes, and labels using a fixed high-contrast palette."""
    annotated = frame_bgr.copy()
    h, w = annotated.shape[:2]
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)
    if boxes is None:
        return annotated

    mask_data = getattr(masks, "data", None) if masks is not None else None
    if mask_data is not None:
        try:
            if hasattr(mask_data, "detach"):
                mask_data = mask_data.detach()
            if hasattr(mask_data, "cpu"):
                mask_data = mask_data.cpu()
            mask_data = np.asarray(mask_data)
        except Exception:
            mask_data = None

    overlay = annotated.copy()
    names = result.names
    for i, box in enumerate(boxes):
        cls_id = int(box.cls.item())
        color = class_color_bgr(cls_id)

        if mask_data is not None and i < len(mask_data):
            mask = mask_data[i]
            if mask.shape[:2] != (h, w):
                mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
            overlay[mask > 0.5] = color

    annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0.0)

    for box in boxes:
        cls_id = int(box.cls.item())
        label = names.get(cls_id, str(cls_id))
        score = float(box.conf.item())
        color = class_color_bgr(cls_id)
        x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        text = f"{label} {score:.2f}"
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_y1 = max(0, y1 - th - baseline - 8)
        label_y2 = min(h - 1, label_y1 + th + baseline + 8)
        label_x2 = min(w - 1, x1 + tw + 12)
        cv2.rectangle(annotated, (x1, label_y1), (label_x2, label_y2), (0, 0, 0), -1)
        cv2.putText(
            annotated,
            text,
            (x1 + 6, label_y2 - baseline - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def open_camera(forced_index: int | None):
    indices = [forced_index] if forced_index is not None else DEFAULT_CAMERA_INDICES
    for idx in indices:
        cap = cv2.VideoCapture(int(idx))
        if cap.isOpened():
            print(f"opened camera index {idx}")
            return cap
        cap.release()
    raise RuntimeError(f"could not open camera indices: {indices}")


def run_image(model: YOLO, source: str, args: argparse.Namespace) -> None:
    image = cv2.imread(source)
    if image is None:
        raise RuntimeError(f"could not read image: {source}")

    result = model.predict(
        image,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )[0]
    print_detections(detections_from_result(result))

    annotated = draw_result_bgr(image, result)
    save_path = args.save or str(Path(source).with_name(Path(source).stem + "_detections.jpg"))
    cv2.imwrite(save_path, annotated)
    print(f"saved annotated image: {save_path}")


def run_camera(model: YOLO, args: argparse.Namespace) -> None:
    cap = open_camera(args.camera_index)
    print("press 'q' to quit")
    last_print = 0.0
    processed = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("camera frame read failed")
            break

        result = model.predict(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]
        detections = detections_from_result(result)
        processed += 1

        now = time.time()
        if now - last_print >= args.print_every:
            print(f"frame={processed}")
            print_detections(detections)
            last_print = now

        annotated = draw_result_bgr(frame, result)
        if args.save_dir and (processed == 1 or processed % max(1, args.save_every) == 0):
            maybe_save_frame(args.save_dir, processed, frame, annotated, detections)
        if args.max_frames > 0 and processed >= args.max_frames:
            break
        if not args.no_window:
            cv2.imshow("YOLO detection test", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


def run_ros_topic(model: YOLO, args: argparse.Namespace) -> None:
    try:
        import rclpy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image
    except Exception as exc:
        raise RuntimeError(
            "ROS mode needs the ROS/venv environment. Run:\n"
            "  cd /home/deep/morty_ws\n"
            "  source install/setup.bash\n"
            "  source .venv_grasp/bin/activate\n"
            "  source src/grasp_pipeline/venv_env.sh\n"
            "  python src/grasp_pipeline/test_model.py --source ros --preset old --conf 0.01"
        ) from exc

    bridge = CvBridge()
    last_print = 0.0
    latest_frame = {"bgr": None}
    processed = 0

    def image_cb(msg: Image) -> None:
        try:
            latest_frame["bgr"] = bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            print(f"failed to convert ROS image: {exc}")

    rclpy.init(args=None)
    node = rclpy.create_node("yolo_model_ros_topic_test")
    node.create_subscription(Image, args.ros_topic, image_cb, 10)
    print(f"subscribed to ROS image topic: {args.ros_topic}")
    print("press 'q' to quit")

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            frame = latest_frame["bgr"]
            if frame is None:
                continue

            result = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            detections = detections_from_result(result)
            processed += 1

            now = time.time()
            if now - last_print >= args.print_every:
                print(f"frame={processed}")
                print_detections(detections)
                last_print = now

            annotated = draw_result_bgr(frame, result)
            if args.save_dir and (processed == 1 or processed % max(1, args.save_every) == 0):
                maybe_save_frame(args.save_dir, processed, frame, annotated, detections)
            if args.max_frames > 0 and processed >= args.max_frames:
                break
            if not args.no_window:
                cv2.imshow("YOLO ROS topic detection test", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


def main() -> None:
    args = parse_args()
    if args.model == DEFAULT_MODEL:
        args.model = OLD_MODEL if args.preset == "old" else NEW_9_CLASS_MODEL
    model = YOLO(args.model)

    print(f"model: {args.model}")
    print(f"task: {model.task}")
    print("classes:")
    for idx, name in model.names.items():
        print(f"  {idx}: {name}")
    print(f"conf threshold: {args.conf:.3f}")

    if args.list_classes:
        return

    if args.source == "ros":
        run_ros_topic(model, args)
    elif args.source == "camera" or args.source.isdigit():
        if args.source.isdigit() and args.camera_index is None:
            args.camera_index = int(args.source)
        run_camera(model, args)
    else:
        run_image(model, args.source, args)


if __name__ == "__main__":
    main()
