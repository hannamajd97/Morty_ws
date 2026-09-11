#!/usr/bin/env python3
import argparse, json, zmq, numpy as np

def send_points_to_graspgen(points_xyz, endpoint="tcp://127.0.0.1:5555",
                            threshold=0.4, colors=None,
                            num_grasps=200, topk_num_grasps=-1,
                            min_grasps=40, max_tries=6,
                            remove_outliers=True, save_debug=False,
                            timeout_ms=60000,
                            fx=None, fy=None, cx=None, cy=None, depth_scale=None):
    """Send a point cloud to a GraspGen service and return the response."""

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.connect(endpoint)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)

    payload: dict = {
        "points": np.asarray(points_xyz, dtype=np.float32).tolist(),
        "threshold": float(threshold),
        "num_grasps": int(num_grasps),
        "topk_num_grasps": int(topk_num_grasps),
        "min_grasps": int(min_grasps),
        "max_tries": int(max_tries),
        "remove_outliers": bool(remove_outliers),
        "save_debug": bool(save_debug),
    }

    if colors is not None:
        payload["colors"] = np.asarray(colors).tolist()

    # ✅ add intrinsics if available
    if fx is not None: payload["fx"] = float(fx)
    if fy is not None: payload["fy"] = float(fy)
    if cx is not None: payload["cx"] = float(cx)
    if cy is not None: payload["cy"] = float(cy)
    if depth_scale is not None: payload["depth_scale"] = float(depth_scale)

    sock.send_json(payload)
    try:
        reply = sock.recv_json()
    except Exception:
        reply = {"ok": False, "error": "timeout or recv error"}
    return reply


def main():
    ap = argparse.ArgumentParser(description="Send Nx3 points (and optional colors) to GraspGen service.")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--in", dest="in_path", required=True, help=".npy Nx3 float32 points")
    ap.add_argument("--colors", dest="colors_path", help=".npy Nx3 uint8 or float colors (optional)")

    # ✅ optional camera intrinsics
    ap.add_argument("--fx", type=float)
    ap.add_argument("--fy", type=float)
    ap.add_argument("--cx", type=float)
    ap.add_argument("--cy", type=float)
    ap.add_argument("--depth_scale", type=float)

    args = ap.parse_args()

    pts = np.load(args.in_path).astype("float32")
    cols = np.load(args.colors_path) if args.colors_path else None

    resp = send_points_to_graspgen(
        pts,
        endpoint=args.endpoint,
        threshold=args.threshold,
        colors=cols,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_scale=args.depth_scale
    )

    if not resp.get("ok", False):
        print("[ERR]", resp.get("error"))
        raise SystemExit(1)

    print(json.dumps(resp["grasps"], indent=2))


if __name__ == "__main__":
    main()
