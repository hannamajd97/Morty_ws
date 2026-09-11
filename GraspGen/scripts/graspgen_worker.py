#!/usr/bin/env python3
import argparse, sys, os, json, time
import numpy as np

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

def load_points_from_path(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext == ".npz":
        data = np.load(path)
        if "xyz" in data:
            arr = data["xyz"]
        elif "points" in data:
            arr = data["points"]
        else:
            raise ValueError("NPZ missing 'xyz' or 'points' key.")
    elif ext == ".json":
        with open(path, "r") as f:
            obj = json.load(f)
        if "points" in obj:
            arr = np.array(obj["points"], dtype=np.float32)
        else:
            raise ValueError("JSON must contain a 'points' key: [[x,y,z],...]")
    else:
        raise ValueError(f"Unsupported input extension: {ext} (use .npy, .npz, .json)")
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        raise ValueError(f"Point array must be Nx3+, got shape {arr.shape}")
    return arr[:, :3]  # keep xyz

def load_points_from_stdin() -> np.ndarray:
    try:
        obj = json.load(sys.stdin)
    except Exception as e:
        raise ValueError(f"Failed to parse JSON from stdin: {e}")
    if isinstance(obj, dict) and "file" in obj:
        return load_points_from_path(obj["file"])
    if isinstance(obj, dict) and "points" in obj:
        pts = np.asarray(obj["points"], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"'points' must be Nx3+, got {pts.shape}")
        return pts[:, :3]
    raise ValueError("STDIN JSON must contain either {'file': '/path/to.npy'} or {'points': [[x,y,z],...] }")

def grasps_to_payload(grasps: np.ndarray, scores: np.ndarray):
    g = np.asarray(grasps)
    s = np.asarray(scores)
    if g.ndim == 3 and g.shape[1:] == (4, 4):
        g_flat = g.reshape(g.shape[0], 16)
    elif g.ndim == 2 and g.shape[1] == 16:
        g_flat = g
    else:
        # Some builds return 7D pose; we won’t guess. Emit raw array to help debugging.
        return [{"raw": np.asarray(x, dtype=np.float32).tolist(),
                 "score": float(sc)} for x, sc in zip(g, s)]
    return [{"matrix": gf.astype(np.float32).tolist(), "score": float(sc)}
            for gf, sc in zip(g_flat, s)]

def run_once(grasp_sampler, threshold: float, points: np.ndarray):
    grasps, confs = GraspGenSampler.run_inference(
        obj_pc=points.astype(np.float32),
        grasp_sampler=grasp_sampler,
        grasp_threshold=threshold
    )
    return grasps, confs

def watch_loop(in_dir: str, out_dir: str, grasp_sampler, threshold: float, poll=0.25):
    os.makedirs(out_dir, exist_ok=True)
    seen = set()
    print(f"[INFO] Watching '{in_dir}' → '{out_dir}' ... (Ctrl+C to stop)")
    while True:
        try:
            for name in os.listdir(in_dir):
                path = os.path.join(in_dir, name)
                if not os.path.isfile(path):
                    continue
                if not (name.lower().endswith(".npy") or name.lower().endswith(".npz") or name.lower().endswith(".json")):
                    continue
                if path in seen:
                    continue
                # naive "stable file" check: size doesn’t change for one poll cycle
                size1 = os.path.getsize(path)
                time.sleep(poll)
                size2 = os.path.getsize(path)
                if size1 != size2:
                    continue  # still being written
                try:
                    pts = load_points_from_path(path)
                    grasps, confs = run_once(grasp_sampler, threshold, pts)
                    payload = grasps_to_payload(grasps, confs)
                    out_path = os.path.join(out_dir, os.path.basename(path) + ".grasps.json")
                    with open(out_path, "w") as f:
                        json.dump(payload, f)
                    print(f"[OK] {name} → {os.path.basename(out_path)}  (grasps={len(payload)})")
                    seen.add(path)
                except Exception as e:
                    print(f"[ERR] Failed on {name}: {e}")
        except KeyboardInterrupt:
            print("\n[INFO] Stopped.")
            break
        except Exception as e:
            print(f"[WARN] Loop error: {e}")
            time.sleep(poll)

def main():
    ap = argparse.ArgumentParser(description="GraspGen file/pipe worker (no ROS).")
    ap.add_argument("--config", default="/models/checkpoints/graspgen_robotiq_2f_140.yml",
                    help="Gripper config .yml for GraspGen")
    ap.add_argument("--threshold", type=float, default=0.4, help="Grasp confidence threshold")
    # Single-run modes
    ap.add_argument("--in", dest="in_path", help="Input point cloud (.npy|.npz|.json)")
    ap.add_argument("--stdin", action="store_true", help="Read JSON from stdin")
    ap.add_argument("--out", dest="out_path", help="Output grasps JSON path (or stdout if omitted)")
    # Watch mode
    ap.add_argument("--watch", dest="watch_dir", help="Watch this directory for new inputs")
    ap.add_argument("--outbox", dest="out_dir", help="Directory to write outputs in watch mode")
    args = ap.parse_args()

    # Load model
    print(f"[INFO] Loading GraspGen config: {args.config}")
    grasp_cfg = load_grasp_cfg(args.config)
    grasp_sampler = GraspGenSampler(grasp_cfg)
    print("[INFO] GraspGen ready.")

    # Watch mode
    if args.watch_dir:
        out_dir = args.out_dir or args.watch_dir
        return watch_loop(args.watch_dir, out_dir, grasp_sampler, args.threshold)

    # Single-run: read points
    if args.stdin:
        points = load_points_from_stdin()
    elif args.in_path:
        points = load_points_from_path(args.in_path)
    else:
        print("ERROR: provide --in /path/to.npy (or .npz/.json) or use --stdin or --watch DIR.", file=sys.stderr)
        sys.exit(2)

    # Inference
    grasps, confs = run_once(grasp_sampler, args.threshold, points)
    payload = grasps_to_payload(grasps, confs)

    # Output
    if args.out_path:
        with open(args.out_path, "w") as f:
            json.dump(payload, f)
        print(f"[OK] Wrote {args.out_path} (grasps={len(payload)})")
    else:
        json.dump(payload, sys.stdout)
        sys.stdout.write("\n")

if __name__ == "__main__":
    main()

