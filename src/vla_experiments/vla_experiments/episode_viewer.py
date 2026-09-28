"""Step-by-step viewer for one recorded episode (2026-09-28, Majd: "the
pictures and the corresponding time and joints — where can I see them, and
are we sure they are right?").

    python3 -m vla_experiments.episode_viewer <trial or episode dir> [...]
    (or `vla_view <dir> ...` after a colcon build of vla_experiments)

writes ``viewer.html`` into each folder. Open it in a browser: both camera
frames of a step side by side, and next to them what the dataset stores for
that same step — the time, the six joint angles, the gripper, the driver's
TCP, the instruction and segment, and how much older each frame is than the
joint sample it is paired with (from the cameras' own header stamps).

Below the frames a timeline plots the arm's joint speed against how much the
wrist and exterior images change from one step to the next (mean absolute
difference of consecutive frames, computed here from the JPEGs). When frames
and joints belong together the curves rise and fall together; a frame filed
under the wrong step, a frozen camera or a clock offset shows up as a shift
between them. The page also reports the time shift that best aligns the two
curves (about 0 s when the pairing is right) — the same check that verified
the 2026-09-28 trials (image motion vs joint speed, corr 0.55-0.89, best
shift within about one 10 Hz step).

The page only references the JPEGs in images/ (relative paths); nothing is
copied, and the folder stays self-contained.
"""
from __future__ import annotations

import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np

from .raw_dataset import Episode


def _motion(ep: Episode, cam: str) -> np.ndarray:
    """Mean absolute grey-level change between consecutive frames (1/8 size)."""
    import cv2
    m = np.zeros(ep.n)
    prev = None
    for i in range(ep.n):
        img = cv2.imread(str(ep.image_path(cam, i)), cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if img is None:
            m[i] = np.nan
            prev = None
            continue
        img = img.astype(np.float32)
        if prev is not None and prev.shape == img.shape:
            m[i] = float(np.abs(img - prev).mean())
        prev = img
    return m


def _best_shift(ep: Episode, cam: str, motion: np.ndarray) -> tuple:
    """(shift_s, corr_at_best, corr_at_0): the joint-speed shift that best
    matches the image motion, both placed on their own capture stamps."""
    s = ep.steps
    tj = s["joint_stamp"]
    dq = np.r_[0.0, np.linalg.norm(np.diff(s["joint_pos"], axis=0), axis=1) / np.maximum(np.diff(tj), 1e-3)]
    order = np.argsort(tj)
    tj, dq = tj[order], dq[order]
    st = s["img_stamp_" + cam]
    tm = 0.5 * (st + np.r_[st[0], st[:-1]])
    ok = np.r_[False, np.diff(st) > 1e-4] & np.isfinite(motion)
    if ok.sum() < 10 or np.nanstd(motion[ok]) == 0:
        return (float("nan"), float("nan"), float("nan"))
    best = (0.0, -2.0)
    for off in np.arange(-0.5, 0.5001, 0.01):
        c = np.corrcoef(motion[ok], np.interp(tm[ok] + off, tj, dq))[0, 1]
        if c > best[1]:
            best = (float(off), float(c))
    c0 = float(np.corrcoef(motion[ok], np.interp(tm[ok], tj, dq))[0, 1])
    return (best[0], best[1], c0)


def _r(a, nd):
    return [None if (v is None or not np.isfinite(v)) else round(float(v), nd) for v in np.asarray(a, dtype=float)]


def build(ep_dir: Path) -> Path:
    ep = Episode(ep_dir)
    s = ep.steps
    n = ep.n
    motion = {cam: _motion(ep, cam) for cam in ("exterior", "wrist")}
    shift = {cam: _best_shift(ep, cam, motion[cam]) for cam in ("exterior", "wrist")}
    speed = np.r_[0.0, np.linalg.norm(np.diff(s["joint_pos"], axis=0), axis=1) / np.maximum(np.diff(s["t"]), 1e-3)]
    ins = ep.instructions
    names = ep.meta.get("phase_names") or []
    data = {
        "id": ep.episode_id,
        "task": ep.task,
        "outcome": ep.outcome,
        "n": n,
        "fps": ep.meta.get("fps"),
        "t": _r(s["t"], 3),
        "wall": _r(s["stamp_ros"], 3),
        "q": [_r(np.degrees(row), 1) for row in s["joint_pos"]],
        "grip": _r(s["gripper"], 2),
        "gcmd": _r(s["gripper_cmd"], 0) if "gripper_cmd" in s else [None] * n,
        "tcp": [_r(row * 1000.0, 1) for row in s.get("ur_tcp_pos", s["tcp_pos"])],
        "lag_ext": _r((s["joint_stamp"] - s["img_stamp_exterior"]) * 1000.0, 0),
        "lag_wri": _r((s["joint_stamp"] - s["img_stamp_wrist"]) * 1000.0, 0),
        "instr": _r(s["instruction_idx"], 0) if "instruction_idx" in s else [-1] * n,
        "seg": _r(s["segment"], 0) if "segment" in s else [0] * n,
        "phase": _r(s["phase"], 0) if "phase" in s else [-1] * n,
        "speed": _r(speed, 3),
        "mot_ext": _r(motion["exterior"], 2),
        "mot_wri": _r(motion["wrist"], 2),
        "instructions": [{"text": i.get("instruction", ""), "said": i.get("transcript"),
                          "superseded": bool(i.get("superseded"))} for i in ins],
        "phase_names": names,
        "shift": {cam: {"best_ms": None if not np.isfinite(v[0]) else round(v[0] * 1000),
                        "corr": None if not np.isfinite(v[1]) else round(v[1], 2),
                        "corr0": None if not np.isfinite(v[2]) else round(v[2], 2)} for cam, v in shift.items()},
        "started": _dt.datetime.fromtimestamp(float(s["stamp_ros"][0])).strftime("%Y-%m-%d %H:%M:%S"),
    }
    html = _PAGE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    out = Path(ep_dir) / "viewer.html"
    out.write_text(html, encoding="utf-8")
    return out


_PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Episode viewer</title>
<style>
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff;--acc:#2f6fbd;--acc2:#c2552d;--acc3:#3f8f5a;--task:#eef3fb;}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#a3a29c;--line:#34332f;--card:#1f1f1d;--acc:#6da3e8;--acc2:#e58a63;--acc3:#6cc08a;--task:#1e2633;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mut);margin:0 0 12px}
.frames{display:grid;grid-template-columns:1fr 1fr;gap:12px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
figure img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#000}
figcaption{padding:6px 10px;color:var(--mut);font-size:13px;font-variant-numeric:tabular-nums}
.bar{display:flex;gap:8px;align-items:center;margin:12px 0}
.bar input[type=range]{flex:1}
button{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card h2{font-size:13px;margin:0 0 6px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td{padding:2px 4px}td:last-child{text-align:right}
canvas{width:100%;height:170px;display:block;background:var(--card);border:1px solid var(--line);border-radius:8px;cursor:crosshair}
.legend{display:flex;gap:14px;color:var(--mut);font-size:13px;margin:6px 0 0;flex-wrap:wrap}
.sw{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px}
.ok{color:var(--acc3)}.warn{color:var(--acc2)}
@media (max-width:700px){.frames{grid-template-columns:1fr}}
</style></head><body><main>
<h1 id="title"></h1><p class="sub" id="sub"></p>
<div class="frames">
 <figure><img id="ext" alt="exterior camera frame"><figcaption id="capExt"></figcaption></figure>
 <figure><img id="wri" alt="wrist camera frame"><figcaption id="capWri"></figcaption></figure>
</div>
<div class="bar"><button id="prev" title="previous step (←)">◀</button><button id="play" title="play / pause (space)">▶</button><button id="next" title="next step (→)">▶|</button>
<input id="slider" type="range" min="0" value="0"><span id="stepno"></span></div>
<canvas id="plot"></canvas>
<div class="legend"><span><span class="sw" style="background:var(--fg)"></span>joint speed</span><span><span class="sw" style="background:var(--acc)"></span>wrist image change</span><span><span class="sw" style="background:var(--acc2)"></span>exterior image change</span><span><span class="sw" style="background:var(--task)"></span>task segment</span><span id="shift"></span></div>
<div class="grid" style="margin-top:12px">
 <div class="card"><h2>Step</h2><table id="tStep"></table></div>
 <div class="card"><h2>Joints (deg)</h2><table id="tJoints"></table></div>
 <div class="card"><h2>Gripper and TCP</h2><table id="tGrip"></table></div>
</div>
</main>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const JN=["shoulder pan","shoulder lift","elbow","wrist 1","wrist 2","wrist 3"],SEG=["idle","task","reset"];
$("title").textContent=D.id+" — "+D.task;
$("sub").textContent=D.n+" steps at "+D.fps+" Hz, started "+D.started+", outcome "+D.outcome;
const sh=D.shift;function shTxt(c,n){const v=sh[c];if(!v||v.best_ms==null)return n+": n/a";return n+": best shift "+(v.best_ms>0?"+":"")+v.best_ms+" ms, corr "+v.corr+" (at 0 ms "+v.corr0+")";}
$("shift").textContent="alignment — "+shTxt("wrist","wrist")+"; "+shTxt("exterior","exterior");
const pad=i=>String(i).padStart(6,"0");
const fmt=(v,d=0,u="")=>v==null?"—":(typeof v==="number"?v.toFixed(d):v)+u;
function row(k,v){return"<tr><td>"+k+"</td><td>"+v+"</td></tr>";}
function lagCls(v){return v==null?"":(Math.abs(v)>150?"warn":"ok");}
let i=0,timer=null;const sl=$("slider");sl.max=D.n-1;
function wall(ts){if(ts==null)return"—";const d=new Date(ts*1000);return d.toLocaleTimeString([], {hour12:false})+"."+String(d.getMilliseconds()).padStart(3,"0");}
function show(k){i=Math.max(0,Math.min(D.n-1,k));sl.value=i;
 $("ext").src="images/exterior/"+pad(i)+".jpg";$("wri").src="images/wrist/"+pad(i)+".jpg";
 $("capExt").innerHTML="exterior (Astra) · frame "+pad(i)+".jpg · <span class='"+lagCls(D.lag_ext[i])+"'>"+fmt(D.lag_ext[i],0," ms")+"</span> older than the joints";
 $("capWri").innerHTML="wrist (RealSense) · frame "+pad(i)+".jpg · <span class='"+lagCls(D.lag_wri[i])+"'>"+fmt(D.lag_wri[i],0," ms")+"</span> older than the joints";
 $("stepno").textContent=i+" / "+(D.n-1);
 const ii=D.instr[i],ins=(ii!=null&&ii>=0)?D.instructions[ii]:null,ph=D.phase[i];
 $("tStep").innerHTML=row("t",fmt(D.t[i],2," s"))+row("wall clock",wall(D.wall[i]))+row("segment",SEG[D.seg[i]]||"—")+
  row("instruction",ins?ins.text:"—")+row("said",ins&&ins.said?"“"+ins.said+"”":"—")+row("executor phase",(ph!=null&&ph>=0&&D.phase_names[ph])?D.phase_names[ph].replace(/_/g," "):"—");
 $("tJoints").innerHTML=D.q[i].map((v,j)=>row(JN[j],fmt(v,1,"°"))).join("");
 const p=D.tcp[i]||[];
 $("tGrip").innerHTML=row("gripper joint",fmt(D.grip[i],2))+row("last gripper command",D.gcmd[i]==null?"—":(D.gcmd[i]>=0.5?"close":"open"))+
  row("TCP x",fmt(p[0],1," mm"))+row("TCP y",fmt(p[1],1," mm"))+row("TCP z",fmt(p[2],1," mm"))+row("joint speed",fmt(D.speed[i],3," rad/s"));
 draw();}
const cv=$("plot"),cx=cv.getContext("2d");
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function series(a){const m=Math.max(...a.map(v=>v==null?0:v))||1;return a.map(v=>v==null?null:v/m);}
const S=series(D.speed),W=series(D.mot_wri),E=series(D.mot_ext);
function draw(){const r=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*r;cv.height=h*r;cx.setTransform(r,0,0,r,0,0);
 cx.clearRect(0,0,w,h);const X=k=>6+(w-12)*k/(D.n-1),Y=v=>h-8-(h-16)*v;
 cx.fillStyle=css("--task");for(let k=0;k<D.n;k++){if(D.seg[k]===1)cx.fillRect(X(k)-0.5,0,(w-12)/(D.n-1)+1,h);}
 [[E,"--acc2"],[W,"--acc"],[S,"--fg"]].forEach(([a,c])=>{cx.strokeStyle=css(c);cx.lineWidth=1.4;cx.beginPath();let st=false;
  a.forEach((v,k)=>{if(v==null){st=false;return;}if(!st){cx.moveTo(X(k),Y(v));st=true;}else cx.lineTo(X(k),Y(v));});cx.stroke();});
 cx.strokeStyle=css("--acc3");cx.lineWidth=2;cx.beginPath();cx.moveTo(X(i),0);cx.lineTo(X(i),h);cx.stroke();}
cv.addEventListener("click",e=>{const b=cv.getBoundingClientRect();show(Math.round((e.clientX-b.left-6)/(b.width-12)*(D.n-1)));});
sl.addEventListener("input",()=>show(+sl.value));
$("prev").onclick=()=>show(i-1);$("next").onclick=()=>show(i+1);
function toggle(){if(timer){clearInterval(timer);timer=null;$("play").textContent="▶";}else{timer=setInterval(()=>{if(i>=D.n-1)toggle();else show(i+1);},1000/(D.fps||10));$("play").textContent="⏸";}}
$("play").onclick=toggle;
document.addEventListener("keydown",e=>{if(e.key==="ArrowRight")show(i+1);else if(e.key==="ArrowLeft")show(i-1);else if(e.key===" "){e.preventDefault();toggle();}});
window.addEventListener("resize",draw);
for(let k=1;k<=3;k++){new Image().src="images/wrist/"+pad(k)+".jpg";new Image().src="images/exterior/"+pad(k)+".jpg";}
show(0);
</script></body></html>
"""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: vla_view <trial or episode dir> [...]   (writes viewer.html into each)")
        return 2
    for a in argv:
        p = Path(a).expanduser()
        if not (p / "steps.npz").exists():
            print(f"skip {p}: no steps.npz")
            continue
        out = build(p)
        print(f"{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
