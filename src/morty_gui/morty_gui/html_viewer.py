"""Self-contained interactive 3D viewer for trial recordings.

Writes ``view3d.html`` next to ``cloud.ply``: the colored object cloud plus
the grasp glyphs (kept green / rejected red / best blue / executed orange),
rendered with three.js in any browser — no MeshLab required (its 2020.09
build wedges on point-cloud loads, Majd 2026-08-12).

The point data is embedded in the file; only the three.js library comes
from the CDN (lab PC is online).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional

import numpy as np

_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Morty trial 3D</title>
<style>html,body{margin:0;height:100%;background:#111;color:#ddd;font-family:sans-serif}
#hud{position:fixed;top:8px;left:10px;font-size:13px;opacity:.8}</style>
</head>
<body>
<div id="hud">drag = rotate &nbsp; wheel = zoom &nbsp; right-drag = pan<br>
<span style="color:#2ecc40">kept</span> ·
<span style="color:#d33a3a">rejected</span> ·
<span style="color:#3ab0ff">best</span> ·
<span style="color:#f39c12">executed</span></div>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
<script>
const DATA = __DATA__;
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x111111);
const renderer = new THREE.WebGLRenderer({antialias:true});
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);
const camera = new THREE.PerspectiveCamera(50,
  window.innerWidth/window.innerHeight, 0.001, 100);

const pos = new Float32Array(DATA.points.flat());
const col = new Float32Array(DATA.colors.flat());
const geo = new THREE.BufferGeometry();
geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
scene.add(new THREE.Points(geo,
  new THREE.PointsMaterial({size: DATA.point_size, vertexColors: true})));

const KINDCOL = {kept:0x2ecc40, rejected:0xd33a3a, best:0x3ab0ff, exec:0xf39c12};
for (const [kind, segs] of Object.entries(DATA.grasps)) {
  if (!segs.length) continue;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position',
    new THREE.BufferAttribute(new Float32Array(segs.flat()), 3));
  scene.add(new THREE.LineSegments(g,
    new THREE.LineBasicMaterial({color: KINDCOL[kind] || 0xcccccc})));
}

geo.computeBoundingSphere();
const c = geo.boundingSphere.center, r = Math.max(geo.boundingSphere.radius, 0.05);
// The data is in the camera OPTICAL frame (+z forward, up ~ -y), while
// three.js defaults to +y up -- that mismatch is what tilted the scene.
// Take the up from the payload and place the camera consistently with it:
// back off along the optical view axis (-z), orthogonalised against up,
// then lift along up.  Display only; the point data below is untouched.
const UP = new THREE.Vector3(
  DATA.scene_up[0], DATA.scene_up[1], DATA.scene_up[2]).normalize();
let back = new THREE.Vector3(0, 0, -1);
if (Math.abs(back.dot(UP)) > 0.9) back.set(0, 1, 0);
back.addScaledVector(UP, -back.dot(UP)).normalize();
camera.up.copy(UP);
camera.position.copy(c).addScaledVector(back, 2.5*r).addScaledVector(UP, 1.2*r);
const controls = new THREE.OrbitControls(camera, renderer.domElement);
controls.object.up.copy(UP);   // OrbitControls orbits about ITS object's up
controls.target.copy(c);
controls.update();
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth/window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});
(function animate(){requestAnimationFrame(animate);controls.update();
 renderer.render(scene,camera);})();
</script>
</body>
</html>
"""


def _quat_to_mat(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = (x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n**0.5, y / n**0.5, z / n**0.5, w / n**0.5
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _glyph_segments(poses) -> List[List[float]]:
    """Each pose → two line segments (approach along +z, width bar on x),
    flattened as [x1,y1,z1, x2,y2,z2] pairs for three.js LineSegments."""
    segs: List[List[float]] = []
    for p in poses:
        if isinstance(p, dict):  # grasps.json form
            o = np.array(p["position"], dtype=float)
            q = p["orientation"]
            r = _quat_to_mat(q[0], q[1], q[2], q[3])
        else:  # geometry_msgs Pose
            o = np.array([p.position.x, p.position.y, p.position.z])
            r = _quat_to_mat(p.orientation.x, p.orientation.y,
                             p.orientation.z, p.orientation.w)
        zax, xax = r[:, 2], r[:, 0]
        a, b = o, o + 0.05 * zax
        c1, c2 = o - 0.03 * xax, o + 0.03 * xax
        segs.append([float(v) for v in (*a, *b)])
        segs.append([float(v) for v in (*c1, *c2)])
    return segs


#: Physical "up" of the data frame.  The clouds/poses handed to this module
#: are in camera_color_optical_frame, whose up is ~ -y.  SEAM (not wired in
#: this task): the pipeline measures the real table-plane normal in that same
#: frame every cycle, and passing it as ``scene_up`` would replace this
#: nominal value.  Rendering only -- the embedded points never change.
SCENE_UP_OPTICAL = (0.0, -1.0, 0.0)


def write_view3d_html(trial_dir: str, pts: np.ndarray,
                      cols01: Optional[np.ndarray],
                      poses_by_kind: Dict,
                      scene_up=SCENE_UP_OPTICAL) -> None:
    """Write/overwrite <trial_dir>/view3d.html."""
    pts = np.asarray(pts, dtype=np.float32)
    if pts.size == 0:
        return
    # Subsample for a snappy page.
    if len(pts) > 25000:
        idx = np.random.choice(len(pts), 25000, replace=False)
        pts = pts[idx]
        cols01 = cols01[idx] if cols01 is not None else None
    if cols01 is None:
        cols01 = np.full_like(pts, 0.8)
    data = {
        "points": np.round(pts, 5).tolist(),
        "colors": np.round(np.clip(cols01, 0, 1), 3).tolist(),
        "point_size": 0.002,
        "scene_up": [float(v) for v in np.asarray(scene_up,
                                                  dtype=float).reshape(3)],
        "grasps": {k: _glyph_segments(v) for k, v in poses_by_kind.items()},
    }
    html = _TEMPLATE.replace("__DATA__", json.dumps(data))
    with open(os.path.join(trial_dir, "view3d.html"), "w") as f:
        f.write(html)


def read_ply_ascii(path: str):
    """Read our own ASCII PLY (x y z r g b) → (pts, cols01)."""
    with open(path) as f:
        line = ""
        n = 0
        while line.strip() != "end_header":
            line = f.readline()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
        rows = [f.readline().split() for _ in range(n)]
    arr = np.array(rows, dtype=float)
    return arr[:, :3].astype(np.float32), (arr[:, 3:6] / 255.0
                                           if arr.shape[1] >= 6 else None)
