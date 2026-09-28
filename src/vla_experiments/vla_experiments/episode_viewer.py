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


def _num(v, nd):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _intent_text(e: dict) -> str:
    a = str(e.get("action_intent") or "")
    attrs = e.get("object_attributes") if isinstance(e.get("object_attributes"), dict) else {}
    bits = [a]
    if e.get("object_intent"):
        bits.append(str(e["object_intent"]).replace("_", " "))
    for k in ("position", "colour", "ordinal", "destination", "then"):
        if attrs.get(k):
            bits.append(f"{k}={attrs[k]}")
    return " ".join(bits)


def _event_log(ep: Episode, ep_dir: Path, t0: float) -> list:
    """Everything the episode recorded, in time order: [t, kind, text,
    approx]. t is on the episode clock (the step times); Morty's lines come
    from the trial's conversation.txt, which the GUI stamps on receipt, so
    they are marked approximate (the GUI was seen up to 3 s late)."""
    import re
    ev = ep.meta.get("events") or {}
    out = []

    def add(stamp, kind, text, approx=False):
        v = _num(float(stamp) - t0, 2)
        if v is not None:
            out.append([v, kind, text, approx])

    for e in ev.get("transcript", []):
        add(e["stamp"], "you", "“" + str(e.get("text", "")) + "”")
    for e in ev.get("intent", []):
        add(e["stamp"], "command", _intent_text(e))
    for e in ev.get("selection", []):
        add(e["stamp"], "select", f"{str(e.get('label') or '').replace('_', ' ')} selected ({e.get('source') or '?'})")
    for e in ev.get("interactive_status", []):
        add(e["stamp"], "robot", str(e.get("status", "")).replace("_", " "))
    for e in ev.get("gripper_cmd", []):
        add(e["stamp"], "gripper", "open" if float(e.get("value", 1.0 if e.get("cmd") == "open" else 0.0)) > 0.5 else "close")
    for e in ev.get("executor_phase", []):
        add(e["stamp"], "phase", str(e.get("name", "")).replace("_", " "))
    for e in ev.get("cycle_done", []):
        add(e["stamp"], "cycle", "done" if e.get("ok") else ("failed: " + str(e.get("reason") or "")))
    conv, tl = ep_dir / "conversation.txt", ep_dir / "timeline.json"
    if conv.exists() and tl.exists():
        try:
            start_wall = float(json.loads(tl.read_text())["start_wall"])
            for line in conv.read_text(errors="ignore").splitlines():
                m = re.match(r"^\[\s*([\d.]+)s\]\s+(Morty(?: \(status\))?):\s*(.*)$", line)
                if m:
                    add(start_wall + float(m.group(1)), "morty", m.group(3), True)
        except Exception:
            pass
    out.sort(key=lambda r: r[0])
    return out


def build(ep_dir: Path) -> Path:
    ep = Episode(ep_dir)
    s = ep.steps
    n = ep.n
    motion = {cam: _motion(ep, cam) for cam in ("exterior", "wrist")}
    shift = {cam: _best_shift(ep, cam, motion[cam]) for cam in ("exterior", "wrist")}
    speed = np.r_[0.0, np.linalg.norm(np.diff(s["joint_pos"], axis=0), axis=1) / np.maximum(np.diff(s["t"]), 1e-3)]
    ins = ep.instructions
    names = ep.meta.get("phase_names") or []
    t0 = float(ep.meta.get("started_ros") or (float(s["stamp_ros"][0]) - float(s["t"][0])))
    instr_idx = s["instruction_idx"] if "instruction_idx" in s else np.full(n, -1.0)
    instructions = []
    for k, i in enumerate(ins):
        on = np.where(instr_idx == k)[0]
        attrs = i.get("attributes") if isinstance(i.get("attributes"), dict) else {}
        instructions.append({
            "text": i.get("instruction", ""), "said": i.get("transcript"),
            "source": attrs.get("source") if not i.get("transcript") else None,
            "action": i.get("action_intent"), "superseded": bool(i.get("superseded")),
            "t": _num(i.get("t"), 2) if i.get("t") is not None else _num(float(i.get("stamp", t0)) - t0, 2),
            "said_t": _num(float(i["transcript_stamp"]) - t0, 2) if i.get("transcript_stamp") else None,
            "first": int(on[0]) if len(on) else None, "last": int(on[-1]) if len(on) else None})
    data = {
        "id": ep.episode_id,
        "task": ep.task,
        "outcome": ep.outcome,
        "n": n,
        "fps": ep.meta.get("fps"),
        "t": _r(s["t"], 3),
        "wall": _r(s["stamp_ros"], 3),
        "q": [_r(np.degrees(row), 1) for row in s["joint_pos"]],
        "qd": [_r(np.degrees(row), 1) for row in s["joint_vel"]],
        # 2026-09-28: joint_acc / joint_current come from the bag (vla_joints);
        # absent for a trial without a bag.
        "qdd": [_r(np.degrees(row), 0) for row in s["joint_acc"]] if "joint_acc" in s else None,
        "cur": [_r(row, 2) for row in s["joint_current"]] if "joint_current" in s else None,
        "grip": _r(s["gripper"], 2),
        "gcmd": _r(s["gripper_cmd"], 0) if "gripper_cmd" in s else [None] * n,
        "gcmd_age": _r(s["gripper_cmd_age"], 1) if "gripper_cmd_age" in s else [None] * n,
        "tcp": [_r(row * 1000.0, 1) for row in s.get("ur_tcp_pos", s["tcp_pos"])],
        "lag_ext": _r((s["joint_stamp"] - s["img_stamp_exterior"]) * 1000.0, 0),
        "lag_wri": _r((s["joint_stamp"] - s["img_stamp_wrist"]) * 1000.0, 0),
        "instr": _r(instr_idx, 0),
        "seg": _r(s["segment"], 0) if "segment" in s else [0] * n,
        "phase": _r(s["phase"], 0) if "phase" in s else [-1] * n,
        "speed": _r(speed, 3),
        "mot_ext": _r(motion["exterior"], 2),
        "mot_wri": _r(motion["wrist"], 2),
        "instructions": instructions,
        "log": _event_log(ep, Path(ep_dir), t0),
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
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mut:#6b6b66;--line:#e2e1dc;--card:#fff;--acc:#2f6fbd;--acc2:#c2552d;--acc3:#3f8f5a;--task:#eef3fb;--hl:#fff4d6;--mark:#8a5cc7;}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#a3a29c;--line:#34332f;--card:#1f1f1d;--acc:#6da3e8;--acc2:#e58a63;--acc3:#6cc08a;--task:#1e2633;--hl:#3a3320;--mark:#b48ef0;}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1320px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 2px}.sub{color:var(--mut);margin:0 0 12px}
.frames{display:grid;grid-template-columns:1fr 1fr;gap:12px}
figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:hidden}
figure img{display:block;width:100%;aspect-ratio:4/3;object-fit:contain;background:#000}
figcaption{padding:6px 10px;color:var(--mut);font-size:13px;font-variant-numeric:tabular-nums}
.now{margin:10px 0 0;padding:8px 12px;background:var(--card);border:1px solid var(--line);border-radius:8px;font-size:15px}
.now b{font-weight:600}.now .mut{color:var(--mut);font-size:13px}
.bar{display:flex;gap:8px;align-items:center;margin:12px 0}
.bar input[type=range]{flex:1}
button{font:inherit;padding:4px 10px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);cursor:pointer}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:12px}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.card h2{font-size:13px;margin:0 0 6px;color:var(--mut);font-weight:600;text-transform:uppercase;letter-spacing:.04em}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td{padding:2px 4px;vertical-align:top}.kv td:last-child{text-align:right}
canvas{width:100%;height:190px;display:block;background:var(--card);border:1px solid var(--line);border-radius:8px;cursor:crosshair}
.legend{display:flex;gap:14px;color:var(--mut);font-size:13px;margin:6px 0 0;flex-wrap:wrap}
.sw{display:inline-block;width:12px;height:3px;vertical-align:middle;margin-right:5px}
.ok{color:var(--acc3)}.warn{color:var(--acc2)}
.jt td{text-align:right}.jt td:first-child{text-align:left}.jt tr:first-child td{color:var(--mut);font-size:12px}
.list tr{cursor:pointer}.list tr:hover td{background:var(--task)}
.list tr.cur td{background:var(--hl)}.list tr.future td{color:var(--mut)}
.list td.t{white-space:nowrap;color:var(--mut);width:1%}.list td.k{white-space:nowrap;width:1%;font-size:12px;color:var(--mut)}
.num{display:inline-block;min-width:18px;text-align:center;border-radius:9px;background:var(--mark);color:#fff;font-size:12px;margin-right:6px}
.sup{text-decoration:line-through;color:var(--mut)}
.scroll{max-height:320px;overflow:auto}
@media (max-width:800px){.frames,.grid2{grid-template-columns:1fr}}
</style></head><body><main>
<h1 id="title"></h1><p class="sub" id="sub"></p>
<div class="frames">
 <figure><img id="ext" alt="exterior camera frame"><figcaption id="capExt"></figcaption></figure>
 <figure><img id="wri" alt="wrist camera frame"><figcaption id="capWri"></figcaption></figure>
</div>
<div class="now" id="now"></div>
<div class="bar"><button id="prev" title="previous step (←)">◀</button><button id="play" title="play / pause (space)">▶</button><button id="next" title="next step (→)">▶|</button>
<input id="slider" type="range" min="0" value="0"><span id="stepno"></span></div>
<canvas id="plot"></canvas>
<div class="legend"><span><span class="sw" style="background:var(--fg)"></span>joint speed</span><span><span class="sw" style="background:var(--acc)"></span>wrist image change</span><span><span class="sw" style="background:var(--acc2)"></span>exterior image change</span><span><span class="sw" style="background:var(--mark)"></span>instruction starts (numbered)</span><span><span class="sw" style="background:var(--task)"></span>task segment</span><span id="shift"></span></div>
<div class="grid2">
 <div class="card"><h2>Instructions — click to jump</h2><table class="list" id="tIns"></table></div>
 <div class="card"><h2>Everything recorded, in time order — click to jump</h2><div class="scroll" id="logBox"><table class="list" id="tLog"></table></div></div>
</div>
<div class="grid" style="margin-top:12px">
 <div class="card"><h2>Step</h2><table class="kv" id="tStep"></table></div>
 <div class="card" style="grid-column:span 2"><h2>Joints — velocity from the robot, acceleration and current from the full-rate bag</h2><table class="kv jt" id="tJoints"></table></div>
 <div class="card"><h2>Gripper and TCP</h2><table class="kv" id="tGrip"></table></div>
</div>
</main>
<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const JN=["shoulder pan","shoulder lift","elbow","wrist 1","wrist 2","wrist 3"],SEG=["idle","task","reset"];
const KIND={you:"you said",command:"command",select:"selection",robot:"robot",gripper:"gripper cmd",phase:"executor",cycle:"cycle",morty:"Morty said"};
$("title").textContent=D.id+" — "+D.task;
$("sub").textContent=D.n+" steps at "+D.fps+" Hz, started "+D.started+", outcome "+D.outcome+" · times are seconds on the episode clock (step t)";
const sh=D.shift;function shTxt(c,n){const v=sh[c];if(!v||v.best_ms==null)return n+": n/a";return n+": best shift "+(v.best_ms>0?"+":"")+v.best_ms+" ms, corr "+v.corr+" (at 0 ms "+v.corr0+")";}
$("shift").textContent="alignment — "+shTxt("wrist","wrist")+"; "+shTxt("exterior","exterior");
const pad=i=>String(i).padStart(6,"0");
const fmt=(v,d=0,u="")=>v==null?"—":(typeof v==="number"?v.toFixed(d):v)+u;
const esc=s=>String(s==null?"":s).replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function row(k,v){return"<tr><td>"+k+"</td><td>"+v+"</td></tr>";}
function lagCls(v){return v==null?"":(Math.abs(v)>150?"warn":"ok");}
function stepAt(tt){let lo=0,hi=D.n-1;while(lo<hi){const m=(lo+hi+1)>>1;if(D.t[m]<=tt)lo=m;else hi=m-1;}return lo;}
let i=0,timer=null;const sl=$("slider");sl.max=D.n-1;
function wall(ts){if(ts==null)return"—";const d=new Date(ts*1000);return d.toLocaleTimeString([], {hour12:false})+"."+String(d.getMilliseconds()).padStart(3,"0");}
// instruction table
$("tIns").innerHTML=D.instructions.length?D.instructions.map((s,k)=>{
 const who=s.said?"said at "+fmt(s.said_t,2," s")+": “"+esc(s.said)+"”":(s.source?"from "+esc(s.source):"");
 const span=s.first!=null?"steps "+s.first+"–"+s.last+" (t "+fmt(D.t[s.first],1)+"–"+fmt(D.t[s.last],1)+" s)":(s.superseded?"superseded — replaced before the arm moved, steps stay idle":"no steps");
 return"<tr data-k='"+k+"'><td class='t'><span class='num'>"+(k+1)+"</span>"+fmt(s.t,2," s")+"</td><td><span class='"+(s.superseded?"sup":"")+"'>"+esc(s.text)+"</span><br><span style='color:var(--mut);font-size:12px'>"+who+(who?" · ":"")+span+"</span></td></tr>";}).join(""):"<tr><td>no instructions recorded</td></tr>";
[...$("tIns").querySelectorAll("tr[data-k]")].forEach(r=>r.onclick=()=>{const s=D.instructions[+r.dataset.k];show(s.first!=null?s.first:stepAt(s.t));});
// event log
$("tLog").innerHTML=D.log.map((e,k)=>"<tr data-e='"+k+"'><td class='t'>"+(e[3]?"≈":"")+fmt(e[0],2," s")+"</td><td class='k'>"+KIND[e[1]]+"</td><td>"+esc(e[2])+"</td></tr>").join("");
const logRows=[...$("tLog").querySelectorAll("tr")];logRows.forEach(r=>r.onclick=()=>show(stepAt(D.log[+r.dataset.e][0])));
function show(k){i=Math.max(0,Math.min(D.n-1,k));sl.value=i;const tt=D.t[i];
 $("ext").src="images/exterior/"+pad(i)+".jpg";$("wri").src="images/wrist/"+pad(i)+".jpg";
 $("capExt").innerHTML="exterior (Astra) · frame "+pad(i)+".jpg · <span class='"+lagCls(D.lag_ext[i])+"'>"+fmt(D.lag_ext[i],0," ms")+"</span> older than the joints";
 $("capWri").innerHTML="wrist (RealSense) · frame "+pad(i)+".jpg · <span class='"+lagCls(D.lag_wri[i])+"'>"+fmt(D.lag_wri[i],0," ms")+"</span> older than the joints";
 $("stepno").textContent=i+" / "+(D.n-1);
 const ii=D.instr[i],ins=(ii!=null&&ii>=0)?D.instructions[ii]:null,ph=D.phase[i];
 const nxt=D.instructions.find(s=>s.t>tt+1e-6);
 $("now").innerHTML="t = <b>"+fmt(tt,2," s")+"</b> · "+(SEG[D.seg[i]]||"—")+" · instruction: <b>"+(ins?"<span class='num'>"+(ii+1)+"</span>"+esc(ins.text):"none (idle)")+"</b>"+
  (nxt?" <span class='mut'>· next: “"+esc(nxt.text)+"” at "+fmt(nxt.t,2," s")+" (in "+fmt(nxt.t-tt,1," s")+")</span>":" <span class='mut'>· no later instruction</span>");
 $("tStep").innerHTML=row("t",fmt(tt,2," s"))+row("wall clock",wall(D.wall[i]))+row("segment",SEG[D.seg[i]]||"—")+
  row("instruction",ins?(ii+1)+" — "+esc(ins.text):"—")+row("said",ins&&ins.said?"“"+esc(ins.said)+"”":(ins&&ins.source?"("+esc(ins.source)+")":"—"))+row("executor phase",(ph!=null&&ph>=0&&D.phase_names[ph])?D.phase_names[ph].replace(/_/g," "):"—");
 $("tJoints").innerHTML="<tr><td></td><td>angle</td><td>velocity</td><td>accel.</td><td>current</td></tr>"+D.q[i].map((v,j)=>"<tr><td>"+JN[j]+"</td><td>"+fmt(v,1,"°")+"</td><td>"+fmt(D.qd[i][j],1,"°/s")+"</td><td>"+(D.qdd?fmt(D.qdd[i][j],0,"°/s²"):"—")+"</td><td>"+(D.cur?fmt(D.cur[i][j],2," A"):"—")+"</td></tr>").join("");
 const p=D.tcp[i]||[],g=D.grip[i],gc=D.gcmd[i];
 $("tGrip").innerHTML=row("gripper state (adapter report)",g==null?"—":(g>=0.5?"open":"closed")+" ("+fmt(g,2)+")")+
  row("last gripper command",gc==null?"—":(gc>=0.5?"open":"close")+(D.gcmd_age[i]!=null?" · "+fmt(D.gcmd_age[i],1," s ago"):""))+
  row("TCP x",fmt(p[0],1," mm"))+row("TCP y",fmt(p[1],1," mm"))+row("TCP z",fmt(p[2],1," mm"))+row("joint speed",fmt(D.speed[i],3," rad/s"));
 [...$("tIns").querySelectorAll("tr[data-k]")].forEach(r=>{const k=+r.dataset.k,s=D.instructions[k];r.className=(k===ii)?"cur":(s.t>tt?"future":"");});
 let last=-1;logRows.forEach((r,k)=>{const f=D.log[k][0]>tt+1e-6;r.className=f?"future":"";if(!f)last=k;});
 if(last>=0){logRows[last].className="cur";const box=$("logBox"),r=logRows[last];if(r.offsetTop<box.scrollTop||r.offsetTop>box.scrollTop+box.clientHeight-30)box.scrollTop=r.offsetTop-box.clientHeight/2;}
 try{history.replaceState(null,"","#"+i);}catch(e){}
 draw();}
const cv=$("plot"),cx=cv.getContext("2d");
function css(v){return getComputedStyle(document.documentElement).getPropertyValue(v).trim();}
function series(a){const m=Math.max(...a.map(v=>v==null?0:v))||1;return a.map(v=>v==null?null:v/m);}
const S=series(D.speed),W=series(D.mot_wri),E=series(D.mot_ext);
function draw(){const r=window.devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;cv.width=w*r;cv.height=h*r;cx.setTransform(r,0,0,r,0,0);
 cx.clearRect(0,0,w,h);const X=k=>6+(w-12)*k/(D.n-1),Y=v=>h-8-(h-34)*v;
 cx.fillStyle=css("--task");for(let k=0;k<D.n;k++){if(D.seg[k]===1)cx.fillRect(X(k)-0.5,0,(w-12)/(D.n-1)+1,h);}
 [[E,"--acc2"],[W,"--acc"],[S,"--fg"]].forEach(([a,c])=>{cx.strokeStyle=css(c);cx.lineWidth=1.4;cx.beginPath();let st=false;
  a.forEach((v,k)=>{if(v==null){st=false;return;}if(!st){cx.moveTo(X(k),Y(v));st=true;}else cx.lineTo(X(k),Y(v));});cx.stroke();});
 cx.font="12px system-ui,sans-serif";cx.textAlign="center";
 D.instructions.forEach((s,k)=>{const x=X(stepAt(s.t));cx.strokeStyle=css("--mark");cx.lineWidth=1.5;cx.setLineDash(s.superseded?[3,3]:[]);cx.beginPath();cx.moveTo(x,18);cx.lineTo(x,h);cx.stroke();cx.setLineDash([]);
  cx.fillStyle=css("--mark");cx.beginPath();cx.arc(x,10,8,0,6.3);cx.fill();cx.fillStyle="#fff";cx.fillText(String(k+1),x,14);});
 cx.strokeStyle=css("--acc3");cx.lineWidth=2;cx.beginPath();cx.moveTo(X(i),0);cx.lineTo(X(i),h);cx.stroke();}
cv.addEventListener("click",e=>{const b=cv.getBoundingClientRect();show(Math.round((e.clientX-b.left-6)/(b.width-12)*(D.n-1)));});
sl.addEventListener("input",()=>show(+sl.value));
$("prev").onclick=()=>show(i-1);$("next").onclick=()=>show(i+1);
function toggle(){if(timer){clearInterval(timer);timer=null;$("play").textContent="▶";}else{timer=setInterval(()=>{if(i>=D.n-1)toggle();else show(i+1);},1000/(D.fps||10));$("play").textContent="⏸";}}
$("play").onclick=toggle;
document.addEventListener("keydown",e=>{if(e.key==="ArrowRight")show(i+1);else if(e.key==="ArrowLeft")show(i-1);else if(e.key===" "){e.preventDefault();toggle();}});
window.addEventListener("resize",draw);
for(let k=1;k<=3;k++){new Image().src="images/wrist/"+pad(k)+".jpg";new Image().src="images/exterior/"+pad(k)+".jpg";}
show(parseInt((location.hash||"").replace("#",""))||0);
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
