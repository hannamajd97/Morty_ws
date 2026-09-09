"""Estimate base_link -> astra_link by registering the Astra cloud onto the wrist RealSense cloud.

Steps
  1. capture N frames: RealSense aligned depth (compressedDepth) + camera_info -> cloud in base_link via TF
     Astra cloud (/astra/depth/points_local) in astra_color_optical_frame
  2. dominant plane in the RealSense cloud = tabletop (in base_link)
     planes in the Astra cloud: keep the horizontal one closest to the camera (table, not floor)
  3. rotate/translate so the Astra plane coincides with the RealSense plane (fixes tilt + height)
  4. yaw + in-plane shift from a 2-D occupancy cross-correlation of the objects above the plane
  5. point-to-plane ICP refinement (Open3D) of the Astra cloud onto the RealSense cloud
  6. base_link<-astra_link = T_base<-astra_opt * inv(T_astra_link<-astra_opt);  print launch args,
     optionally write them into astra_tf.launch.py and config/recorder.yaml (--apply)
"""
import argparse, math, re, struct, sys, time
from pathlib import Path

import numpy as np
import cv2
import open3d as o3d
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import CameraInfo, CompressedImage, PointCloud2
from tf2_ros import Buffer, TransformListener


def tf_to_matrix(tr):
    t, q = tr.transform.translation, tr.transform.rotation
    M = np.eye(4); M[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix(); M[:3, 3] = [t.x, t.y, t.z]
    return M


def decode_cdepth(m):
    img = cv2.imdecode(np.frombuffer(bytes(m.data[12:]), np.uint8), cv2.IMREAD_UNCHANGED)
    return img.astype(np.float32) / 1000.0 if img is not None and img.dtype == np.uint16 else None


def depth_to_points(d, K, max_range):
    fx, fy, cx, cy = K
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]].astype(np.float32)
    ok = (d > 0.05) & (d < max_range)
    z = d[ok]; return np.stack([(u[ok] - cx) / fx * z, (v[ok] - cy) / fy * z, z], 1)


def cloud_msg_to_xyz(m):
    a = np.frombuffer(bytes(m.data), np.uint8).reshape(-1, m.point_step)
    return np.stack([a[:, i:i + 4].copy().view(np.float32).ravel() for i in (0, 4, 8)], 1)


def sample_robot_model(urdf_xml, tf_buffer, pts_per_link=2500):
    """Sample surface points of every collision mesh of the robot, in base_link, at the current joint state."""
    import os, xml.etree.ElementTree as ET
    from ament_index_python.packages import get_package_share_directory
    root = ET.fromstring(urdf_xml); out = []
    for link in root.iter("link"):
        name = link.get("name")
        for col in link.findall("collision"):
            g = col.find("geometry"); m = g.find("mesh") if g is not None else None
            if m is None:
                continue
            fn = m.get("filename"); pkg = fn.split("//")[1].split("/")[0]; rel = "/".join(fn.split("//")[1].split("/")[1:])
            try:
                path = os.path.join(get_package_share_directory(pkg), rel)
                mesh = o3d.io.read_triangle_mesh(path)
            except Exception:
                continue
            if not mesh.has_triangles():
                continue
            sc = m.get("scale"); 
            if sc:
                mesh.scale(float(sc.split()[0]), center=np.zeros(3))
            o = col.find("origin"); xyz = np.array([float(v) for v in (o.get("xyz") if o is not None and o.get("xyz") else "0 0 0").split()])
            rpy = np.array([float(v) for v in (o.get("rpy") if o is not None and o.get("rpy") else "0 0 0").split()])
            M0 = np.eye(4); M0[:3, :3] = R.from_euler("xyz", rpy).as_matrix(); M0[:3, 3] = xyz
            try:
                Ml = tf_to_matrix(tf_buffer.lookup_transform("base_link", name, Time()))
            except Exception:
                continue
            mesh.transform(Ml @ M0)
            n_pts = max(400, int(pts_per_link * mesh.get_surface_area() / 0.05))
            out.append(np.asarray(mesh.sample_points_uniformly(min(n_pts, 20000)).points))
    return np.concatenate(out) if out else np.zeros((0, 3))


class Grab(Node):
    def __init__(self, n_frames, rs_max, astra_max):
        super().__init__("astra_calibrate")
        self.n_frames, self.rs_max, self.astra_max = n_frames, rs_max, astra_max
        self.rs_info = None; self.rs = []; self.astra = []; self.rs_frame = None; self.astra_frame = None
        self.tf = Buffer(); TransformListener(self.tf, self, spin_thread=False)
        self.create_subscription(CameraInfo, "/camera/camera/aligned_depth_to_color/camera_info", self._info, qos_profile_sensor_data)
        self.create_subscription(CompressedImage, "/camera/camera/aligned_depth_to_color/image_raw/compressedDepth", self._rs, qos_profile_sensor_data)
        self.create_subscription(PointCloud2, "/astra/depth/points_local", self._astra, 10)
        from std_msgs.msg import String
        from rclpy.qos import QoSProfile, DurabilityPolicy
        self.urdf = None
        self.create_subscription(String, "/robot_description", lambda m: setattr(self, "urdf", m.data), QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def _info(self, m):
        self.rs_info = (m.k[0], m.k[4], m.k[2], m.k[5]); self.rs_frame = m.header.frame_id

    def _rs(self, m):
        if self.rs_info is None or len(self.rs) >= self.n_frames:
            return
        d = decode_cdepth(m)
        if d is None:
            return
        try:
            M = tf_to_matrix(self.tf.lookup_transform("base_link", m.header.frame_id, Time()))
        except Exception:
            return
        p = depth_to_points(d, self.rs_info, self.rs_max)
        self.rs.append(p @ M[:3, :3].T + M[:3, 3])

    def _astra(self, m):
        if len(self.astra) >= self.n_frames:
            return
        p = cloud_msg_to_xyz(m); p = p[np.linalg.norm(p, axis=1) < self.astra_max]
        self.astra.append(p); self.astra_frame = m.header.frame_id

    def done(self):
        return len(self.rs) >= self.n_frames and len(self.astra) >= self.n_frames and self.urdf is not None


def to_o3d(p, voxel):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p.astype(np.float64)))
    return pc.voxel_down_sample(voxel)


def planes(pc, n_planes=3, dist=0.012):
    """sequential RANSAC; returns list of (normal, d, inlier_points) with n·x + d = 0."""
    out = []; rest = pc
    for _ in range(n_planes):
        if len(rest.points) < 500:
            break
        model, idx = rest.segment_plane(dist, 3, 800)
        n = np.array(model[:3]); d = model[3]
        pts = np.asarray(rest.points)[idx]
        out.append((n / np.linalg.norm(n), d / np.linalg.norm(n), pts))
        rest = rest.select_by_index(idx, invert=True)
    return out


def orient_towards(n, d, origin):
    """flip plane so the normal points towards `origin` (the camera side / up)."""
    if np.dot(n, origin) + d < 0:
        return -n, -d
    return n, d


def rot_a_to_b(a, b):
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b); v = np.cross(a, b); c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else R.from_rotvec(np.pi * np.array([1, 0, 0])).as_matrix()
    return R.from_rotvec(v / np.linalg.norm(v) * math.acos(np.clip(c, -1, 1))).as_matrix()


def occupancy(xy, origin, size, cell):
    g = np.zeros((size, size), np.float32)
    ij = np.floor((xy - origin) / cell).astype(int)
    ok = (ij[:, 0] >= 0) & (ij[:, 0] < size) & (ij[:, 1] >= 0) & (ij[:, 1] < size)
    g[ij[ok, 1], ij[ok, 0]] = 1.0
    return g


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=4)
    ap.add_argument("--rs-max", type=float, default=1.2, help="RealSense max depth [m]")
    ap.add_argument("--astra-max", type=float, default=3.0)
    ap.add_argument("--voxel", type=float, default=0.006)
    ap.add_argument("--apply", action="store_true", help="write result into astra_tf.launch.py and recorder.yaml")
    ap.add_argument("--save", default="", help="save clouds (ply) and result (json) to this dir")
    a = ap.parse_args(argv)

    rclpy.init(); node = Grab(a.frames, a.rs_max, a.astra_max)
    t0 = time.time()
    while rclpy.ok() and not node.done() and time.time() - t0 < 60:
        rclpy.spin_once(node, timeout_sec=0.1)
    if not node.done():
        print(f"timeout: rs frames={len(node.rs)} astra frames={len(node.astra)}", file=sys.stderr); return 1
    T_base_astralink = tf_to_matrix(node.tf.lookup_transform("base_link", "astra_link", Time()))
    T_base_opt0 = tf_to_matrix(node.tf.lookup_transform("base_link", node.astra_frame, Time()))
    T_link_opt = np.linalg.inv(T_base_astralink) @ T_base_opt0
    rs_cam_in_base = tf_to_matrix(node.tf.lookup_transform("base_link", node.rs_frame, Time()))[:3, 3]
    model = sample_robot_model(node.urdf, node.tf)
    print(f"robot model: {len(model)} sampled points from URDF collision meshes")
    node.destroy_node(); rclpy.try_shutdown()

    rs = to_o3d(np.concatenate(node.rs), a.voxel)
    astra_raw = np.concatenate(node.astra)
    astra = to_o3d(astra_raw, a.voxel)
    print(f"clouds: realsense {len(rs.points)} pts (base_link), astra {len(astra.points)} pts (optical)")

    # ---- 2. planes
    rs_planes = planes(rs, 3)
    rs_c = []
    for n, d, pts in rs_planes:
        n, d = orient_towards(n, d, rs_cam_in_base); dist = float(np.mean(np.linalg.norm(pts - rs_cam_in_base, axis=1)))
        par = abs(np.dot(n, rs_planes[0][0])) > 0.9; rs_c.append((n, d, pts, par, dist))
        print(f"  realsense plane: normal {np.round(n, 3)} inliers {len(pts)} mean dist to camera {dist:.2f} m parallel={par}")
    n_b, d_b, rs_in, _, _ = rs_c[0]  # largest plane = floor
    print(f"chosen realsense FLOOR plane: {len(rs_in)} inliers, normal {np.round(n_b, 3)} (towards camera = up)")
    ap_ = planes(astra, 3)
    cand = []
    for n, d, pts in ap_:
        n, d = orient_towards(n, d, np.zeros(3))  # towards the astra camera at the origin
        parallel = abs(np.dot(n, ap_[0][0])) > 0.9
        depth = float(np.mean(pts[:, 2]))
        cand.append((n, d, pts, parallel, depth))
        print(f"  astra plane: normal {np.round(n, 3)} inliers {len(pts)} mean depth {depth:.2f} m parallel_to_largest={parallel}")
    par = [c for c in cand if c[3]]
    n_a, d_a, astra_table, _, _ = cand[0]  # largest plane = floor
    print(f"chosen astra FLOOR plane: {len(astra_table)} inliers")

    # ---- 3. plane alignment: rotate n_a -> n_b, then shift along n_b so offsets match
    R1 = rot_a_to_b(n_a, n_b)
    p_a = astra_table.mean(0); p_a1 = R1 @ p_a
    # RS plane point: project centroid of rs inliers
    p_b = rs_in.mean(0)
    t1 = p_b - p_a1  # brings the astra table centroid onto the rs table centroid (in-plane part fixed later)
    t1 = t1 - n_b * (np.dot(t1, n_b)) + n_b * np.dot(p_b - p_a1, n_b)  # keep only along-normal + centroid guess
    T1 = np.eye(4); T1[:3, :3] = R1; T1[:3, 3] = p_b - p_a1  # start with centroids coincident (yaw/xy solved next)

    # ---- 4. yaw + in-plane shift from objects above the plane
    def above(pts, n, d, lo=0.015, hi=0.30):
        h = pts @ n + d
        return pts[(h > lo) & (h < hi)]
    rs_obj = np.concatenate([above(np.asarray(rs.points), n_b, d_b, 0.05, 1.35), above(model, n_b, d_b, 0.05, 1.35)])
    astra_all1 = (np.asarray(astra.points) @ R1.T) + T1[:3, 3]
    d_b1 = -np.dot(n_b, (astra_table @ R1.T + T1[:3, 3]).mean(0))
    astra_obj = above(astra_all1, n_b, d_b1, 0.05, 1.35)
    # 2-D basis in the plane
    e1 = np.cross(n_b, [1, 0, 0]); e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 0.1 else np.cross(n_b, [0, 1, 0]) / np.linalg.norm(np.cross(n_b, [0, 1, 0]))
    e2 = np.cross(n_b, e1); B = np.stack([e1, e2], 1)
    rs_xy = rs_obj @ B; as_xy = astra_obj @ B
    cell = 0.01; size = 512; origin = rs_xy.mean(0) - size * cell / 2
    g_ref = occupancy(rs_xy, origin, size, cell); Fr = np.fft.rfft2(g_ref)
    best = (-1, 0, None); c0 = as_xy.mean(0)
    for yaw_deg in np.arange(-180, 180, 1.0):
        th = math.radians(yaw_deg); c, s_ = math.cos(th), math.sin(th)
        rot = np.array([[c, -s_], [s_, c]]); xy = (as_xy - c0) @ rot.T + rs_xy.mean(0)
        g_s = occupancy(xy, origin, size, cell)
        corr = np.fft.irfft2(Fr * np.conj(np.fft.rfft2(g_s)), s=g_ref.shape)
        k = np.unravel_index(np.argmax(corr), corr.shape); score = corr[k]
        if score > best[0]:
            shift = np.array([(k[1] + size // 2) % size - size // 2, (k[0] + size // 2) % size - size // 2]) * cell
            best = (score, yaw_deg, shift)
    score, yaw_deg, shift = best
    print(f"objects: reference {len(rs_obj)} pts (rs objects + robot model), astra {len(astra_obj)} pts; best astra yaw = {yaw_deg:.0f} deg, shift {np.round(shift, 3)} m, overlap {score:.0f} cells")
    # astra_obj rotated by yaw about n_b around its centroid, then moved so its centroid lands on rs centroid + shift
    Ryaw = R.from_rotvec(math.radians(yaw_deg) * n_b).as_matrix()
    pivot = B @ c0 + n_b * 0.0
    target = B @ (rs_xy.mean(0) + shift)
    T2 = np.eye(4); T2[:3, :3] = Ryaw
    # translation: keep the along-normal component unchanged, move in-plane centroid to target
    p_src = astra_obj.mean(0)
    T2[:3, 3] = (p_src - Ryaw @ p_src) + B @ (B.T @ (target - p_src))
    T_coarse = T2 @ T1

    # ---- 5. ICP refine (astra -> rs), restricted to the RS neighbourhood
    from scipy.spatial import cKDTree
    target = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate([np.asarray(rs.points), model]))).voxel_down_sample(a.voxel)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.03, 30))
    rs_pts = np.concatenate(node.rs); rng = np.random.default_rng(0)
    rs_sub = rs_pts[rng.choice(len(rs_pts), min(20000, len(rs_pts)), replace=False)]

    def refine(T0, label):
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(astra.points))).transform(T0)
        bb = target.get_axis_aligned_bounding_box(); bb = o3d.geometry.AxisAlignedBoundingBox(bb.min_bound - 0.15, bb.max_bound + 0.15)
        src = src.crop(bb)
        if len(src.points) < 500:
            print(f"  [{label}] too few points after crop"); return None
        src.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(0.03, 30))
        T = np.eye(4)
        for thr in (0.05, 0.025, 0.012):
            res = o3d.pipelines.registration.registration_icp(src, target, thr, T, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                                                              o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80))
            T = res.transformation
        A = np.asarray(src.transform(T).points); tree = cKDTree(A)
        d_rs, _ = tree.query(rs_sub); d_mo, _ = tree.query(model)
        score = float(np.median(d_rs)) + float(np.median(d_mo))
        print(f"  [{label}] icp fitness {res.fitness:.3f} rmse {res.inlier_rmse*1000:.1f} mm | validation: realsense median {np.median(d_rs)*1000:.0f} mm ({np.mean(d_rs<0.02)*100:.0f}% <2cm), robot model median {np.median(d_mo)*1000:.0f} mm ({np.mean(d_mo<0.02)*100:.0f}% <2cm)")
        return score, T @ T0, src

    cands = []
    r = refine(T_coarse, "A plane+corr")
    if r: cands.append(r)
    # candidate B: FPFH RANSAC global registration (no initial guess)
    vg = 0.02
    src_g = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(astra.points))).voxel_down_sample(vg)
    tgt_g = target.voxel_down_sample(vg)
    for pc in (src_g, tgt_g):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(vg * 2, 30))
    f_s = o3d.pipelines.registration.compute_fpfh_feature(src_g, o3d.geometry.KDTreeSearchParamHybrid(vg * 5, 100))
    f_t = o3d.pipelines.registration.compute_fpfh_feature(tgt_g, o3d.geometry.KDTreeSearchParamHybrid(vg * 5, 100))
    best_b = None
    for trial in range(3):
        rg = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_g, tgt_g, f_s, f_t, True, vg * 1.5, o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9), o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(vg * 1.5)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999))
        r = refine(rg.transformation, f"B fpfh-ransac #{trial} (fit {rg.fitness:.2f})")
        if r and (best_b is None or r[0] < best_b[0]):
            best_b = r
    if best_b: cands.append(best_b)
    score, T_base_opt, astra_c = min(cands, key=lambda c: c[0])
    print(f"chosen candidate score {score*1000:.0f} mm (sum of medians)")
    T_icp = np.eye(4)
    T_base_link = T_base_opt @ np.linalg.inv(T_link_opt)
    xyz = T_base_link[:3, 3]; rpy = R.from_matrix(T_base_link[:3, :3]).as_euler("xyz")
    delta = np.linalg.inv(T_base_astralink) @ T_base_link
    print("\n==== RESULT base_link -> astra_link ====")
    print(f"xyz  = {xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}   (guess was {np.round(T_base_astralink[:3,3],3)})")
    print(f"rpy  = {rpy[0]:.4f} {rpy[1]:.4f} {rpy[2]:.4f}   (deg {np.round(np.degrees(rpy),1)})")
    print(f"change vs guess: {np.linalg.norm(delta[:3,3])*100:.1f} cm, {np.degrees(np.linalg.norm(R.from_matrix(delta[:3,:3]).as_rotvec())):.1f} deg")
    print(f"launch: ros2 launch vla_experiments astra_tf.launch.py x:={xyz[0]:.4f} y:={xyz[1]:.4f} z:={xyz[2]:.4f} roll:={rpy[0]:.4f} pitch:={rpy[1]:.4f} yaw:={rpy[2]:.4f}")
    from scipy.spatial import cKDTree
    A = np.asarray(astra_c.points)
    for nm, ref in (("realsense", np.concatenate(node.rs)), ("robot model", model)):
        dd, _ = cKDTree(A).query(ref[np.random.default_rng(0).choice(len(ref), min(20000, len(ref)), replace=False)])
        print(f"validation {nm:12s} -> nearest astra point: median {np.median(dd)*1000:.1f} mm, 90% {np.percentile(dd,90)*1000:.1f} mm, within 2 cm {np.mean(dd<0.02)*100:.0f}%")
    if a.save:
        Path(a.save).mkdir(parents=True, exist_ok=True)
        o3d.io.write_point_cloud(f"{a.save}/rs_base.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(node.rs)))); o3d.io.write_point_cloud(f"{a.save}/model_base.ply", o3d.geometry.PointCloud(o3d.utility.Vector3dVector(model))); o3d.io.write_point_cloud(f"{a.save}/astra_base_aligned.ply", astra_c)
        import json; json.dump({"xyz": xyz.tolist(), "rpy": rpy.tolist(), "T_base_astra_link": T_base_link.tolist(), "score_sum_medians_m": score}, open(f"{a.save}/result.json", "w"), indent=2)
    if a.apply:
        share = Path(__file__).resolve().parents[1]
        lp = share / "launch" / "astra_tf.launch.py"; s = lp.read_text()
        for k, v in zip(("x", "y", "z", "roll", "pitch", "yaw"), list(xyz) + list(rpy)):
            s = re.sub(rf'DeclareLaunchArgument\("{k}", default_value="[^"]*"\)', f'DeclareLaunchArgument("{k}", default_value="{v:.4f}")', s)
        lp.write_text(s)
        cp = share / "config" / "recorder.yaml"; c = cp.read_text()
        c = re.sub(r"astra_tf_xyz: \[[^\]]*\]", f"astra_tf_xyz: [{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}]", c)
        c = re.sub(r"astra_tf_rpy: \[[^\]]*\]", f"astra_tf_rpy: [{rpy[0]:.4f}, {rpy[1]:.4f}, {rpy[2]:.4f}]", c)
        cp.write_text(c); print(f"applied to {lp.name} and {cp.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
