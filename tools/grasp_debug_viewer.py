#!/usr/bin/env python3
"""
grasp_debug_viewer.py — standalone Open3D viewer for GraspGen results.

Loads a point cloud (a real sample object, or any Nx3/Nx6 cloud), runs GraspGen
IN-PROCESS via grasp_pipeline.grasp_client, and renders:
  * the colored point cloud
  * each returned grasp as a gripper wireframe at its 4x4 pose
    (green = high score, red = low score)
  * a world coordinate frame

Usage:
  # newest sample object, top 20 grasps, interactive window:
  python3 grasp_debug_viewer.py

  # a specific cloud + save a screenshot instead of opening a window (offscreen):
  python3 grasp_debug_viewer.py --cloud /path/to/obj.json --save out.png

  --topk N        only draw the N highest-scoring grasps (default 20)
  --gripper NAME  gripper for the wireframe (default robotiq_2f_140)
  --save PATH     render to an image file instead of an interactive window
                  (use this if the WSLg GUI window won't open)
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import open3d as o3d

# Make the grasp_client importable; assumes workspace is sourced or on path.
from grasp_pipeline.grasp_client import send_points_to_graspgen
from grasp_gen.utils.meshcat_utils import load_visualization_gripper_points


SAMPLE_DIR = os.path.expanduser(
    "~/morty_ws/GraspGen/sample_data/real_object_pc"
)


def load_cloud(path):
    """Load a cloud from a GraspGen sample JSON (keys: pc, pc_color)."""
    d = json.load(open(path))
    pc = np.asarray(d["pc"], dtype=np.float32)
    col = None
    if "pc_color" in d:
        col = np.asarray(d["pc_color"], dtype=np.float32)
        if col.max() > 1.0:          # 0..255 -> 0..1 for Open3D
            col = col / 255.0
    return pc, col


def score_to_rgb(s):
    """Red (low) -> green (high) ramp for a 0..1 score."""
    s = max(0.0, min(1.0, float(s)))
    return [1.0 - s, s, 0.0]


def grasp_lineset(matrix_16, gripper_pts, color):
    """Build an Open3D LineSet for one grasp: gripper wireframe at the 4x4 pose."""
    T = np.asarray(matrix_16, dtype=np.float32).reshape(4, 4)
    geoms = []
    for seg in gripper_pts:                  # each seg is [4, N] homogeneous
        pts = (T @ seg)[:3].T                 # transform -> Nx3
        n = pts.shape[0]
        lines = [[i, i + 1] for i in range(n - 1)]
        ls = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(pts),
            lines=o3d.utility.Vector2iVector(lines),
        )
        ls.colors = o3d.utility.Vector3dVector([color] * len(lines))
        geoms.append(ls)
    return geoms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cloud", default=None, help="cloud JSON (default: newest sample)")
    ap.add_argument("--from-json", default=None,
                    help="JSON with precomputed {pc, grasps} — skip running GraspGen "
                         "(used by the live pipeline auto-viz)")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--gripper", default="robotiq_2f_140")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--save", default=None, help="render to image instead of window")
    ap.add_argument("--also", default=None,
                    help="write a second copy of the render here (e.g. latest.png)")
    args = ap.parse_args()

    col = None
    if args.from_json:
        # Pipeline path: cloud + grasps already computed, just render them.
        d = json.load(open(args.from_json))
        pc = np.asarray(d["pc"], dtype=np.float32)
        grasps = d.get("grasps", [])
        print(f"[viewer] from-json: {pc.shape[0]} points, {len(grasps)} grasps")
    else:
        # Standalone path: load a cloud and run GraspGen on it now.
        cloud_path = args.cloud
        if cloud_path is None:
            files = sorted(glob.glob(os.path.join(SAMPLE_DIR, "*.json")))
            if not files:
                print(f"No sample clouds in {SAMPLE_DIR}", file=sys.stderr)
                sys.exit(1)
            cloud_path = files[0]
        print(f"[viewer] cloud: {cloud_path}")
        pc, col = load_cloud(cloud_path)
        print(f"[viewer] {pc.shape[0]} points; running in-process GraspGen…")
        os.environ.setdefault("GRASPGEN_MODE", "inprocess")
        resp = send_points_to_graspgen(pc, threshold=args.threshold,
                                       num_grasps=200, min_grasps=10, max_tries=2,
                                       remove_outliers=False)
        if not resp.get("ok"):
            print("[viewer] GraspGen failed:", resp.get("error"), file=sys.stderr)
            sys.exit(1)
        grasps = resp.get("grasps", [])
        print(f"[viewer] {len(grasps)} grasps returned")

    # keep the top-k by score
    grasps = sorted(grasps, key=lambda g: g.get("score", 0.0), reverse=True)[: args.topk]

    # --- build geometry ---
    geoms = []
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pc[:, :3])
    if col is not None:
        cloud.colors = o3d.utility.Vector3dVector(col[:, :3])
    geoms.append(cloud)
    geoms.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05))

    gripper_pts = load_visualization_gripper_points(args.gripper)
    for g in grasps:
        m = g.get("matrix")
        if m is None:
            continue
        geoms += grasp_lineset(m, gripper_pts, score_to_rgb(g.get("score", 0.0)))

    print(f"[viewer] drawing cloud + {len(grasps)} grasps "
          f"(green=high score, red=low)")

    if args.save:
        # Headless render via EGL OffscreenRenderer — no GL window needed, so it
        # works on WSL where create_window() can't get a GL context.
        W, H = 1280, 960
        rnd = o3d.visualization.rendering.OffscreenRenderer(W, H)
        rnd.scene.set_background([1, 1, 1, 1])
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultUnlit"
        line_mat = o3d.visualization.rendering.MaterialRecord()
        line_mat.shader = "unlitLine"
        line_mat.line_width = 2.0
        for i, gobj in enumerate(geoms):
            m = line_mat if isinstance(gobj, o3d.geometry.LineSet) else mat
            rnd.scene.add_geometry(f"g{i}", gobj, m)
        # Frame the object: look at the cloud centroid from a diagonal vantage.
        c = pc[:, :3].mean(axis=0)
        eye = c + np.array([0.3, -0.3, 0.3], dtype=np.float32)
        rnd.scene.camera.look_at(c.tolist(), eye.tolist(), [0, 0, 1])
        img = rnd.render_to_image()
        o3d.io.write_image(args.save, img)
        print(f"[viewer] saved render to {args.save}")
        if args.also:
            o3d.io.write_image(args.also, img)
            print(f"[viewer] also wrote {args.also}")
    else:
        o3d.visualization.draw_geometries(
            geoms, window_name="GraspGen debug — cloud + grasps",
            width=1280, height=960,
        )


if __name__ == "__main__":
    main()
