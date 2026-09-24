#!/usr/bin/env python3
import argparse
import os
import time
import traceback
import numpy as np
import zmq

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

# --- MeshCat (optional) -------------------------------------------------------
USE_HELPERS = True
try:
    import meshcat
    import meshcat.geometry as mg
    from grasp_gen.utils.meshcat_utils import create_visualizer, get_color_from_score, visualize_grasp
except Exception:
    USE_HELPERS = False
    try:
        import meshcat
        import meshcat.geometry as mg
        def create_visualizer():
            return meshcat.Visualizer()
        def get_color_from_score(s, use_255_scale=False):
            s = max(0.0, min(1.0, float(s)))
            c = np.array([s, 0.2, 1.0 - s], dtype=np.float32)
            return (c*255).astype(np.uint8) if use_255_scale else c
        def visualize_grasp(vis, name, T, color=None, gripper_name=None):
            import meshcat.transformations as mt
            vis[name].set_transform(T.astype(np.float32))
            # draw tiny axis triad
            axis_len = 0.03
            colors = [(255,0,0), (0,255,0), (0,0,255)]
            for i, axis in enumerate(["x","y","z"]):
                pts = np.array([[0,0,0], np.eye(3)[i]*axis_len], dtype=np.float32)
                vis[name][f"axis_{axis}"].set_object(
                    mg.Line(
                        mg.PointsGeometry(pts.T),
                        mg.MeshLambertMaterial(color=(colors[i][0]<<16)|(colors[i][1]<<8)|colors[i][2])
                    )
                )
    except Exception:
        pass
# -----------------------------------------------------------------------------


def to_np(x):
    try:
        import torch
        if "Tensor" in type(x).__name__:
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


def grasps_to_payload(grasps, scores):
    g = to_np(grasps); s = to_np(scores)
    if g.ndim == 3 and g.shape[1:] == (4,4):
        g = g.reshape(g.shape[0], 16)
    elif g.ndim == 2 and g.shape[1] == 16:
        pass
    else:
        return [{"raw": to_np(row).astype("float32").tolist(),
                 "score": float(sc)} for row, sc in zip(g, s)]
    return [{"matrix": row.astype("float32").tolist(), "score": float(sc)}
            for row, sc in zip(g, s)]


def main():
    ap = argparse.ArgumentParser(description="GraspGen ZMQ service + MeshCat viz (with colors + telemetry)")
    ap.add_argument("--bind", default="tcp://127.0.0.1:5555", help="e.g., tcp://0.0.0.0:5555")
    ap.add_argument("--config", default="/models/graspgen_robotiq_2f_140.yml")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--verbose", action="store_true")
    # Viz options
    ap.add_argument("--viz", action="store_true", help="Render clouds+grasps in MeshCat")
    ap.add_argument("--max-viz-grasps", type=int, default=100)
    ap.add_argument("--max-viz-points", type=int, default=20000)
    args = ap.parse_args()

    print(f"[INFO] Loading GraspGen config: {args.config}")
    sampler = GraspGenSampler(load_grasp_cfg(args.config))
    # Named in every reply (2026-09-24): the pipeline checks it against the
    # model the GUI said it started — the grasp origin's distance to the
    # fingertips differs per gripper (Franka 0.103 m, Robotiq 0.195 m).
    try:
        gripper = str(sampler.cfg.data.gripper_name)
    except Exception:
        gripper = ""
    print(f"[INFO] Gripper: {gripper or 'unknown'}")
    # Warm-up (2026-09-11): the PTv3 backbone (Franka checkpoint) compiles
    # its sparse-convolution kernels on the first inference — measured 50 s,
    # then 0.3 s per request. Pay it here, before the socket exists, so the
    # first real selection never waits for it; the PointNet++ models take
    # about a second. A synthetic 3 cm cylinder 40 cm from the camera.
    try:
        t_warm = time.time()
        th = np.random.rand(1500) * 2 * np.pi
        z = np.random.rand(1500) * 0.08
        warm = np.stack([0.015 * np.cos(th), z - 0.04, 0.4 + 0.015 * np.sin(th)], 1).astype(np.float32)
        GraspGenSampler.run_inference(warm, sampler, args.threshold, 50, -1, 10, 1, False)
        print(f"[INFO] Warm-up inference done in {time.time() - t_warm:.1f} s")
    except Exception as e:
        print(f"[WARN] Warm-up inference failed: {e}")
    print(f"[INFO] Service ready on {args.bind} (default thr={args.threshold})")

    vis = None
    if args.viz:
        try:
            vis = create_visualizer()
            try: vis.delete()
            except Exception: pass
            print("[INFO] MeshCat visualizer ready.")
        except Exception as e:
            print(f"[WARN] Could not init MeshCat: {e}")
            vis = None

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(args.bind)

    try:
        while True:
            req = sock.recv_json()
            # Defaults (in case client didn’t send intrinsics)
            default_fx = 615.0
            default_fy = 615.0
            default_cx = 320.0
            default_cy = 240.0
            default_depth_scale = 1000.0

            # ✅ Override from request if provided
            fx = req.get("fx", default_fx)
            fy = req.get("fy", default_fy)
            cx = req.get("cx", default_cx)
            cy = req.get("cy", default_cy)
            depth_scale = req.get("depth_scale", default_depth_scale)
            
            try:
                pts = np.asarray(req["points"], dtype=np.float32)
                if pts.ndim != 2 or pts.shape[1] < 3:
                    raise ValueError(f"'points' must be Nx3, got {pts.shape}")

                # optional per-point colors
                cols = req.get("colors")
                cols_np = None
                if cols is not None:
                    cols_np = np.asarray(cols)
                    if cols_np.shape[0] != pts.shape[0] or cols_np.shape[1] < 3:
                        cols_np = None
                    else:
                        cols_np = cols_np[:, :3].astype(np.float32)
                        if cols_np.max() > 1.0:
                            cols_np = cols_np / 255.0

                # knobs
                thr = float(req.get("threshold", args.threshold))
                num_grasps       = int(req.get("num_grasps", 200))
                topk_num_grasps  = int(req.get("topk_num_grasps", -1))

                # Use client-requested values (keep outlier removal off — it's too slow).
                min_grasps = int(req.get("min_grasps", 1))
                max_tries = max(1, min(int(req.get("max_tries", 1)), 3))
                remove_outliers = False

                if args.verbose:
                    print(f"[REQ] {len(pts)} pts | thr={thr} "
                          f"num={num_grasps} topk={topk_num_grasps} "
                          f"min={min_grasps} tries={max_tries} outliers={remove_outliers}", flush=True)

                # optional debug dump of exactly what the service received
                if req.get("save_debug", False):
                    ts = int(time.time())
                    os.makedirs("/code/inbox/debug", exist_ok=True)
                    np.save(f"/code/inbox/debug/pts_{ts}.npy", pts)
                    if cols_np is not None:
                        np.save(f"/code/inbox/debug/cols_{ts}.npy", (cols_np*255).astype(np.uint8))

                # --- draw point cloud (with color if provided) ---
                if vis is not None:
                    try:
                        pts_vis = pts
                        idx = None
                        if pts_vis.shape[0] > args.max_viz_points:
                            idx = np.random.choice(pts_vis.shape[0], args.max_viz_points, replace=False)
                            pts_vis = pts_vis[idx]

                        # quick stats for sanity
                        if args.verbose:
                            mn = pts_vis.min(axis=0); mx = pts_vis.max(axis=0); ct = pts_vis.mean(axis=0)
                            print(f"[CLOUD] min={mn}, max={mx}, mean={ct}", flush=True)

                        try: vis["scene"].delete()
                        except Exception: pass

                        color_arg = None
                        if cols_np is not None:
                            color_arg = cols_np if idx is None else cols_np[idx]
                            color_arg = color_arg.T.astype("float32")

                        pc = mg.PointCloud(
                            position=pts_vis.T.astype("float32"),
                            color=color_arg
                        )
                        vis["scene"]["pointcloud"].set_object(pc)
                    except Exception as e:
                        print(f"[WARN] MeshCat cloud draw failed: {e}")

                # --- inference (positional args to match signature) ---
                grasps, scores = GraspGenSampler.run_inference(
                    pts[:, :3], sampler, thr, num_grasps, topk_num_grasps, min_grasps, max_tries, remove_outliers
                )

                # --- draw grasps ---
                if vis is not None:
                    try:
                        G = to_np(grasps); S = to_np(scores)
                        keep = min(args.max_viz_grasps, G.shape[0])
                        for i in range(keep):
                            color = get_color_from_score(float(S[i]), use_255_scale=False) if 'get_color_from_score' in globals() else None
                            visualize_grasp(
                                vis["scene"], f"grasp_{i}", G[i],
                                color=color,
                                gripper_name=getattr(sampler.cfg.data, "gripper_name", None) if USE_HELPERS else None
                            )
                    except Exception as e:
                        print(f"[WARN] MeshCat grasp draw failed: {e}")

                # --- telemetry + reply ---
                payload = grasps_to_payload(grasps, scores)
                scores_np = to_np(scores)
                pts_stats = {
                    "n_points": int(pts.shape[0]),
                    "xyz_min": np.min(pts[:, :3], axis=0).tolist(),
                    "xyz_max": np.max(pts[:, :3], axis=0).tolist(),
                    "xyz_mean": np.mean(pts[:, :3], axis=0).tolist(),
                }
                if cols_np is not None:
                    pts_stats["rgb_min"]  = np.min(cols_np, axis=0).tolist()
                    pts_stats["rgb_max"]  = np.max(cols_np, axis=0).tolist()
                    pts_stats["rgb_mean"] = np.mean(cols_np, axis=0).tolist()

                reply = {
                    "ok": True,
                    "gripper": gripper,
                    "grasps": payload,
                    "telemetry": {
                        "points": pts_stats,
                        "inference": {
                            "num_grasps_raw": int(scores_np.shape[0]),
                            "num_grasps_after_threshold": int(len(payload)),
                            "best_score": float(scores_np.max()) if scores_np.size else None,
                            "threshold": float(thr),
                        }
                    }
                }
                sock.send_json(reply)
            except Exception as e:
                if args.verbose:
                    traceback.print_exc()
                sock.send_json({"ok": False, "error": str(e)})
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down service.")
    finally:
        try: sock.close(0)
        except Exception: pass
        try: ctx.term()
        except Exception: pass


if __name__ == "__main__":
    main()

