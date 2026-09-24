"""Split from app.py 2026-08-25 (paper-artifact stage 1): mechanical
move, zero behavior change. Backup: ~/morty_ws/backups/app.py.2026-08-25.pre_split
"""
import os

# ── Fast-DDS transport: UDP only, shared memory OFF (2026-09-02) ─────────
# Three native segfaults in libfastrtps in one day (a Python node, the
# grasp executor, a move_group core dump) — the Fast-DDS 2.6 shared-memory
# transport faulting when a peer dies abruptly. Set here, in the GUI's own
# process environment at import time, so EVERY child it spawns (driver,
# pipeline, executor, voice via os.environ.copy()) inherits it. morty.sh and
# ~/.bashrc carry the same export for launches outside the GUI. Only takes
# effect for processes started after it is set → full-stack restart.
os.environ.setdefault(
    "FASTRTPS_DEFAULT_PROFILES_FILE",
    os.path.expanduser("~/morty_ws/config/fastdds_udp_only.xml"),
)
# RViz under Wayland/Xwayland + NVIDIA (2026-09-02, SYSTEM_STATE_LOG #300):
# vsync frame-pacing stalls compound with CPU starvation into the
# "fps drops until stuck" spiral. Inherited by rviz2 spawned from here.
os.environ.setdefault("__GL_SYNC_TO_VBLANK", "0")

import subprocess
from typing import Dict, List

__all__ = [
    "ALL_TUNABLE_PARAMS",
    "CAMERA_STANDALONE",
    "DRIVER_LAUNCH",
    "EXECUTOR_CMD",
    "EXECUTOR_CMDS",
    "FILTER_PARAMS",
    "GREEN",
    "GREY",
    "GRIP_HELP",
    "MOVE_CMD_BASE",
    "MOVE_EXIT_TEXT",
    "NAMED_PLACES", "named_places", "yaml_poses", "POSES_FILE",
    "NEIGH_HELP",
    "OFFLINE_LAUNCH",
    "ORANGE",
    "PARAM_DOCS",
    "PARAM_SECTIONS",
    "PERCEPTION_PARAMS",
    "PIPELINE_CMD",
    "GRASPGEN_SERVICE_CMD", "GRASPGEN_LOG", "GRASPGEN_PGREP", "PIPELINE_ENV_EXTRA",
    "NAMING_SERVICE_CMD", "NAMING_LOG", "NAMING_PGREP", "LLAMA_SERVER_BIN", "NAMED_POSE_LOG",
    "PLANNER_LIBS",
    "PROFILE_HELP",
    "RED",
    "REPLAY_CMD", "REPLAY_LOG", "NAMED_POSE_SERVER_CMD",
    "ROBOT_IP",
    "SHUTTLE_IP",
    "RVIZ_STANDALONE",
    "UR_JOINTS",
    "VOICE_BASE",
    "GWDG_BRAINS",
    "VOICE_COMMON_ARGS",
    "VOICE_FILTER_PARAMS",
    "VOICE_LANG_ARGS",
    "STT_ENGINE",
    "WHISPER_TURBO",
    "_pactl_default",
    "_pactl_devices",
    "_pactl_set_default",
    "read_model_names",
]

ROBOT_IP = "192.168.1.102"
# The shuttle PC (Astra camera host, demo@tory). Its address really is
# 168.192.x — the NIC was configured with the octets swapped and every
# script/DDS profile follows it. Pinged by the GUI for the ● shuttle LED.
SHUTTLE_IP = "168.192.1.102"

# Voice layer: morty.sh sources the provider env itself (.morty_env /
# bedrock script) and --voice-only starts ONLY the chat stack (STT → LLM →
# TTS + intent router) — driver and pipeline stay ours.
# mic_visualizer:=false — its level bar / mute / states now live INSIDE the
# Morty 🎤 tab. with_replay:=false — the GUI already runs the replay node
# with the driver (the voice launch bringing its own caused DUPLICATE replay
# nodes, observed live 2026-08-11).
# GWDG brains (2026-09-24, Majd: Gemma 4 default): the Bedrock launch path
# with MORTY_MODEL_OVERRIDE set to the gwdg/ model id; llm_service streams
# from GWDG and lets Claude (BEDROCK_MODEL) answer any turn GWDG cannot.
# Measured on Morty's prompt: Gemma 0.80 s, DeepSeek 0.86 s, Haiku 1.16 s
# per full answer, commands 4/4 each; DeepSeek replied in English to German.
GWDG_BRAINS = {
    "gemma 4 (gwdg)": "gwdg/gemma-4-31b-it",
    "deepseek v4 flash (gwdg)": "gwdg/deepseek-v4-flash-0731",
}
VOICE_BASE = {
    **{name: ["bash", os.path.expanduser("~/morty_ws/morty.sh"), "--debug", "--voice-only"]
       for name in GWDG_BRAINS},          # first entry = the selector's default
    "claude (debug)": ["bash", os.path.expanduser("~/morty_ws/morty.sh"),
                       "--debug", "--voice-only"],
    "openai": ["bash", os.path.expanduser("~/morty_ws/morty.sh"), "--voice-only"],
}
VOICE_COMMON_ARGS = ["mic_visualizer:=false", "with_replay:=false", "with_named_pose_server:=false"]
# Whisper large-v3-turbo (CT2 conversion, loadable by faster-whisper 1.0.1)
# on the GPU: benchmarked 2026-08-12 on this box — 0.93 s per utterance,
# word-perfect DE, ~1.5 GB VRAM next to YOLO's 0.9 GB. The old `base` model
# produced "STALE TO ALL" for "start all". First voice start downloads
# ~1.6 GB once; later loads come from the HF cache.
WHISPER_TURBO = "deepdml/faster-whisper-large-v3-turbo-ct2"
# 2026-09-23: the live engine is NVIDIA Parakeet TDT 0.6B v3 (stt_engine:=
# parakeet): 52 ms per utterance vs 641 ms for Whisper turbo on this GPU,
# equal texts on the trial recordings, German included, no hallucinated
# "Thank you." on silence. It detects the language per utterance itself,
# so the EN/DE pins below only matter when stt_engine:=whisper is chosen.
# 2026-09-24: "omni" = Qwen3-Omni at GWDG (grasp 3/3, gripper 2/2, 0.3 s on
# Majd's voice) with that Parakeet as the per-sentence fallback on timeout,
# error, rate limit or budget cap (communication_llm/gwdg_omni.py).
# "parakeet" = local only, nothing leaves the machine.
STT_ENGINE = "omni"
VOICE_LANG_ARGS = {
    # AUTO: multilingual model + per-utterance language detection — EN and
    # DE (and others) both just work. EN/DE remain as explicit pins.
    "AUTO": [f"stt_engine:={STT_ENGINE}", f"whisper_model:={WHISPER_TURBO}", "whisper_language:=auto",
             "whisper_device:=cuda", "whisper_compute_type:=int8"],
    "EN": [f"stt_engine:={STT_ENGINE}", f"whisper_model:={WHISPER_TURBO}", "whisper_language:=en",
           "whisper_device:=cuda", "whisper_compute_type:=int8"],
    "DE": [f"stt_engine:={STT_ENGINE}", f"whisper_model:={WHISPER_TURBO}", "whisper_language:=de",
           "whisper_device:=cuda", "whisper_compute_type:=int8"],
}


def read_model_names() -> Dict[str, str]:
    """Read ONLY the model-name lines from .morty_env (model IDs are not
    secrets; keys are never touched)."""
    out = {"claude (debug)": "", "openai": ""}
    try:
        with open(os.path.expanduser("~/morty_ws/.morty_env")) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export BEDROCK_MODEL="):
                    out["claude (debug)"] = line.split("=", 1)[1].strip('"')
                elif line.startswith("export OPENAI_MODEL="):
                    out["openai"] = line.split("=", 1)[1].strip('"')
    except OSError:
        pass
    return out


def _pactl_devices(kind: str) -> List[str]:
    """kind: 'sources' or 'sinks' → device names (monitors excluded for mic)."""
    try:
        r = subprocess.run(["pactl", "list", "short", kind],
                           capture_output=True, text=True, timeout=2)
        names = [ln.split("\t")[1] for ln in r.stdout.splitlines() if "\t" in ln]
        if kind == "sources":
            names = [n for n in names if not n.endswith(".monitor")]
        return names
    except Exception:
        return []


def _pactl_default(kind: str) -> str:
    try:
        r = subprocess.run(["pactl", f"get-default-{kind}"],
                           capture_output=True, text=True, timeout=2)
        return r.stdout.strip()
    except Exception:
        return ""


def _pactl_filters_on(kind: str, master: str) -> Dict[str, str]:
    """{index: name} of the filter devices (echocancel_mic / echocancel_spk)
    whose master is `master` — PulseAudio tags them device.master_device."""
    r = subprocess.run(["pactl", "list", kind + "s"], capture_output=True, text=True, timeout=2)
    out: Dict[str, str] = {}
    idx = name = ""
    for ln in r.stdout.splitlines():
        s = ln.strip()
        if s.startswith(("Source #", "Sink #")):
            idx, name = s.split("#", 1)[1].strip(), ""
        elif s.startswith("Name:"):
            name = s.split(":", 1)[1].strip()
        elif s.startswith("device.master_device") and s.split("=", 1)[1].strip().strip('"') == master:
            out[idx] = name
    return out


def _pactl_set_default(kind: str, name: str) -> None:
    """Set default and move running streams, like the old mic window did.

    Streams on the echo canceller are left where they are when `name` is
    that canceller's own master (2026-09-24). Picking the wireless mic in
    the IN list used to move the speech node from echocancel_mic onto the
    raw mic — the canceller's input — and from then on Morty heard his own
    voice mixed into the operator's (found 15:26: node on the raw mic although
    its launch pins echocancel_mic). The same holds for the speaker and
    echocancel_spk: TTS moved off it leaves the canceller without its
    reference. Choosing a different device still moves everything."""
    try:
        subprocess.run(["pactl", f"set-default-{kind}", name], timeout=2)
        list_kind = "source-outputs" if kind == "source" else "sink-inputs"
        move_cmd = "move-source-output" if kind == "source" else "move-sink-input"
        keep = _pactl_filters_on(kind, name)          # device index -> filter name
        r = subprocess.run(["pactl", "list", "short", list_kind],
                           capture_output=True, text=True, timeout=2)
        for ln in r.stdout.splitlines():
            cols = ln.split("\t")
            if len(cols) > 1 and cols[1] in keep:
                continue
            subprocess.run(["pactl", move_cmd, cols[0], name], timeout=2)
    except Exception:
        pass

# SRDF group states of ur3e_manipulator (open/close are the gripper group).
NAMED_PLACES = [
    "home", "init", "drop_left_side", "drop_right_side",
    "detect_left", "detect_right", "detect_higher", "old_initgrasp", "sleep_home",
]
POSES_FILE = os.path.expanduser("~/morty_ws/observation_poses.yaml")


def yaml_poses() -> Dict[str, List[float]]:
    """Joint poses from observation_poses.yaml (the replay node's and the
    named-pose server's source), name -> six radians. Empty if unreadable."""
    try:
        import yaml
        with open(POSES_FILE) as fh:
            d = yaml.safe_load(fh) or {}
        return {str(k): [float(x) for x in v.get("positions", [])]
                for k, v in (d.get("poses") or {}).items() if isinstance(v, dict)}
    except Exception:
        return {}


def named_places() -> List[str]:
    """Dropdown entries: the SRDF states, then poses that exist only in the
    yaml (2026-09-23, Majd: vla_drop_right "isn't included in the GUI").
    The planner path reaches a yaml-only pose as a joint target."""
    extra = [n for n in yaml_poses() if n not in NAMED_PLACES]
    return list(NAMED_PLACES) + extra

# UR kinematic chain order — matches MoveIt's group joint order, which is
# what observation_ptp_commander's joint_target_<i> params index into.
UR_JOINTS = [
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
]

# Planner libraries, mirroring the RViz MotionPlanning context tab.
PLANNER_LIBS: Dict[str, List[str]] = {
    "pilz_industrial_motion_planner": ["PTP", "LIN", "CIRC"],
    "ompl": [
        "RRTConnectkConfigDefault", "RRTkConfigDefault", "RRTstarkConfigDefault",
        "ESTkConfigDefault", "KPIECEkConfigDefault", "BKPIECEkConfigDefault",
        "LBKPIECEkConfigDefault", "PRMkConfigDefault", "PRMstarkConfigDefault",
        "SBLkConfigDefault", "TRRTkConfigDefault",
    ],
}

# LIN is Majd's verified pick executor (first real pick 2026-08-07).
# PTP / PILZ launches take fewer args — offered as experiments only.
EXECUTOR_CMDS: Dict[str, List[str]] = {
    "LIN": [
        "ros2", "launch", "ur3e_gripper_3d_moveit_commander",
        "grasp_executor_lin.launch.py",
        "best_pose:=true",
        "use_candidate_pose_array:=true",
        "max_candidate_tries:=10",
        # First batch only (Majd 2026-09-11): in every successful pick of
        # that day the accepted grasp was among the first 6; when the first
        # 10 fail the rest fail too and only cost 5-10 s more.
        "candidate_try_all_batches:=false",
        # Pulse holds are appended DYNAMICALLY at EXECUTE press from the
        # Advanced spinboxes (Majd 2026-08-26: "does the manual execution
        # know?" — it didn't; these were hardcoded 0.5/1.5).
        "override_gripper_pulse_from_executor:=true",
        "velocity_scaling:=0.1",
        "acceleration_scaling:=0.1",
        "one_shot:=true",
        "execute:=true",
        # Manual button = grasp and HOLD (Majd 2026-08-26); the voice
        # executor (EXECUTOR_CMD) keeps the full interactive sequence.
        "stop_after_grasp:=true",
    ],
    "PTP": [
        "ros2", "launch", "ur3e_gripper_3d_moveit_commander",
        "grasp_executor.launch.py", "execute:=true",
    ],
    "PILZ": [
        "ros2", "launch", "ur3e_gripper_3d_moveit_commander",
        "grasp_executor_pilz.launch.py", "execute:=true",
    ],
}

DRIVER_LAUNCH = {
    # with camera / without camera map to the two real-hardware launch files.
    True: ["ros2", "launch", "ur3e_gripper_3d_moveit_config",
           "ur3e_gripper_3d_moveit_real_camera.launch.py"],
    False: ["ros2", "launch", "ur3e_gripper_3d_moveit_config",
            "ur3e_gripper_3d_moveit_real.launch.py"],
}

# Offline: the base launch with mock hardware — no robot, no UR driver.
# The real_* wrappers pin use_fake_hardware:=false, so offline must call
# the base launch directly.
OFFLINE_LAUNCH = [
    "ros2", "launch", "ur3e_gripper_3d_moveit_config",
    "ur3e_gripper_3d_moveit.launch.py",
    "use_fake_hardware:=true", "use_ur_driver:=false",
]

# The GUI owns the fast-replay node (started with the driver — it only
# needs the trajectory controller, not the pipeline; historically it was
# bundled in the pipeline launch). with_replay:=false stops the pipeline
# from starting a duplicate instance.
# GraspGen as its own process (2026-09-09, stage one of the speed work):
# inside the pipeline node one inference cost 0.3-0.9 s because YOLO, the
# hand tracker and the camera callbacks share that process's interpreter
# lock; in a clean process it is 0.10 s. The pipeline talks to it over ZMQ
# (GRASPGEN_MODE=zmq, grasp_client.py); the GUI starts it with the pipeline.
GRASPGEN_SERVICE_CMD = [
    "/usr/bin/python3", "-u", os.path.expanduser("~/morty_ws/GraspGen/scripts/graspgen_service.py"),
    # Franka Panda checkpoint since 2026-09-11 (Majd): 105 mm opening and
    # 105 mm depth are far closer to the printed gripper (85 mm / 80 mm)
    # than the Robotiq 2F-140 (136 mm / 195 mm) the system ran on before;
    # GraspGen's own FAQ suggests re-targeting to the closest stroke.
    # Franka Panda checkpoint (Majd, 2026-09-11 17:05): 105 mm opening and
    # 105 mm depth are far closer to the printed gripper (85 mm / 80 mm
    # fingers) than the Robotiq 2F-140 (136 mm / 195 mm). Its PTv3 backbone
    # needs the CUDA build of spconv (spconv-cu121, installed that day; the
    # CPU wheel failed every request) and compiles its kernels on the first
    # inference (~50 s) — the service warms up before it reports ready.
    # Inference 0.30 s against 0.12 s for the Robotiq model.
    "--config", os.path.expanduser("~/morty_ws/models/graspgen/graspgen_franka_panda.yml"),
    "--bind", "tcp://127.0.0.1:5555",
]
GRASPGEN_LOG = os.path.expanduser("~/morty_ws/logs/graspgen_gui.log")
# Naming service for open-set detection (2026-09-15): Qwen3-VL-8B behind
# llama.cpp's server, started with the pipeline like the GraspGen service.
# The pipeline asks it for a name for every object the trained YOLO does
# not know (perception/open_set.py; 0.15 s per object). 6 GB on the card;
# models and the build of llama-server are described in models/vlm/README.md.
LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN", os.path.expanduser("~/vlm/llama.cpp/build/bin/llama-server"))
NAMING_SERVICE_CMD = [
    LLAMA_SERVER_BIN,
    # Q3_K_M since the evening of 2026-09-15: 4.0 GB on the card with the
    # vision encoder on the CPU (Q4: 4.8 GB), same names on the test crops,
    # 0.35 s per name. The Q4 file stays next to it for comparison.
    "-m", os.path.expanduser("~/morty_ws/models/vlm/qwen3-vl-8b-q3_k_m.gguf"),
    "--mmproj", os.path.expanduser("~/morty_ws/models/vlm/qwen3-vl-8b-mmproj.gguf"),
    # One slot, 1024 context, 256 batch: the crops are tiny and the answer is
    # 6 tokens; the defaults (4 slots x 2048) cost 240 MB more on a card
    # that has 550 MB spare with everything running (7.6 of 8.2 GB).
    "--port", "8090", "-ngl", "99", "-c", "1024", "-np", "1", "-ub", "256", "-b", "512", "-fa", "on",
    # Vision encoder on the CPU (measured 2026-09-15): 4.8 GB on the card
    # instead of 5.9, same names, 0.37 s instead of 0.22 s per object. The
    # evening the full stack hit 8.88 MiB free and froze the desktop.
    "--no-mmproj-offload",
]
NAMING_LOG = os.path.expanduser("~/morty_ws/logs/naming_gui.log")
NAMED_POSE_LOG = os.path.expanduser("~/morty_ws/logs/named_pose_gui.log")
NAMING_PGREP = r"llama-server.*qwen3-vl-8b-q4_k_[m]"
GRASPGEN_PGREP = r"graspgen_service\.p[y]"
# MORTY_GRASP_PRECOMPUTE: the grasp precompute cache is on for every
# GUI-started pipeline since 2026-09-11 (tested that afternoon: hits publish
# in 0.3 s; a GUI restarted without the variable silently lost it).
PIPELINE_ENV_EXTRA = {"GRASPGEN_MODE": "zmq", "MORTY_GRASP_PRECOMPUTE": "true"}

PIPELINE_CMD = ["ros2", "launch", "grasp_pipeline", "run_grasp_pipeline.launch.py",
                "with_replay:=false",
                # Voice-pick plumbing (mirrors morty_full.launch.py): the GUI
                # also starts grasp_executor_lin_llm, which reports on
                # /grasp_executor_lin_llm/cycle_done — without this remap the
                # selection thread waits on the wrong topic forever. The 1 s
                # manual-wait re-entry is what lets a voice-set label land
                # instead of hanging in the wait loop.
                "executor_cycle_done_topic:=/grasp_executor_lin_llm/cycle_done",
                "selection_manual_wait_s:=1.0",
                "suspend_inference_while_selecting:=false",
                # 2026-08-24 thesis teardown: the retired orientation knobs
                # (horizontal_score_weight, tilt band, center_band_m) are
                # gone — the thesis §4.6.14 cascade defaults in params.py
                # rule now; do NOT re-add launch overrides here (that
                # silent-override trap bit us three times).
                "max_graspgen_points:=1000"]

# The interactive LIN executor — the node that actually drives the arm to the
# chosen grasp, closes the gripper, then holds and waits for a drop
# destination on /grasp_executor_lin_llm/resume ("drop right" / "drop left" /
# "home"). Without it the pipeline selects, publishes grasps, and then waits
# on cycle_done forever (2026-08-11 "why doesn't it execute" bug). Args
# mirror morty_full.launch.py, incl. the 20 s candidate wait (default 0.35 s
# times out before the pipeline finishes filtering).
EXECUTOR_CMD = ["ros2", "launch", "ur3e_gripper_3d_moveit_commander",
                "grasp_executor_lin_llm.launch.py",
                "best_pose:=true", "use_candidate_pose_array:=true",
                "execute:=true", "candidate_wait_timeout_s:=20.0",
                "enable_wrist_twin_rewrite:=false",
                # Probe only the first batch of 10 candidates (Majd
                # 2026-09-11): the accepted grasp was among the first 6 in
                # every successful pick that day; the rest only cost time.
                "candidate_try_all_batches:=false",
                # Staged approach, 7 cm (Majd 2026-09-11): PTP to a stand-off
                # in front of the object, then a 7 cm straight line. One long
                # line from home crossed the elbow singularity on far objects
                # and Pilz refused every candidate (cycle 18 that day).
                # Off again (Majd 2026-09-11 17:40): the stand-off costs ~2 s
                # per pick (joint move 3.3 s + 10 cm line 1.4 s + two planning
                # stops against one 2.2-4.1 s straight line). Price: objects
                # near 0.7 m whose straight line from home crosses the elbow
                # limit are refused again ("straight line not drivable" in the
                # log). Candidate for later: straight line first, stand-off
                # only as the fallback.
                "use_pregrasp_staging:=true", "staging_mode:=cartesian", "pregrasp_blend_radius_m:=0.035",
                "pregrasp_approach_distance:=0.05",
                # 50 % (Majd 2026-09-11 18:30; the launch defaults gave
                # 0.2 / 0.1): arm travel ~3 s -> ~1.8 s per pick. Live-tunable
                # afterwards: ros2 param set /grasp_executor_lin_llm velocity_scaling X
                # 0.4 since 2026-09-16 (Majd: "0.4 would be enough, no need to be that aggressive").
                "velocity_scaling:=0.4",
                # 1.0 since 2026-09-16: braking distance = v^2 / 2a; at half the
                # allowance the arm crept the last centimetres into the object.
                "acceleration_scaling:=1.0",
                # Run the pick to the end without questions (Majd 2026-09-11
                # 19:05: with auto-execute every pick ended in Reset because
                # the executor waited for a drop destination nobody could
                # give from the GUI): grasp, lift, home, drop_right_side,
                # open, home. Voice picks still get their pauses from the
                # router's per-pick parameter.
                # wait_for_destination back to true (Majd 2026-09-15: "I
                # want the drop executed one time when I press it"): after
                # grasp + lift the executor holds above the table and waits
                # for Left / Right from the GUI, then goes home and drops
                # there. Nothing is asked by voice; Release stays automatic.
                "wait_for_destination:=true",
                "wait_for_release:=true"]   # voice "drop right" waits at the drop for "release"; the GUI buttons send the +release_home compound (2026-09-18)

# The named-pose action server rides with the driver too (2026-09-16): the
# grasp executor moves home / to the drop through /move_to_named_pose with a
# collision pre-check; without the server it fell back to the bare replay
# topic, whose status came late — 8 s timeouts and a second trip to the
# drop side. It used to exist only while the voice stack ran.
NAMED_POSE_SERVER_CMD = [
    "ros2", "run", "ur3e_fast_observation", "move_to_named_pose_server", "--ros-args",
    "-p", "collision_check:=true",
]
REPLAY_LOG = os.path.expanduser("~/morty_ws/logs/replay_gui.log")   # 2026-09-18: was /dev/pts — two hangs left no trace
REPLAY_CMD = [
    "ros2", "run", "ur3e_fast_observation", "replay_node", "--ros-args",
    "-p", "poses_file:=" + os.path.expanduser("~/morty_ws/observation_poses.yaml"),
    "-p", "controller_action_ns:=/scaled_joint_trajectory_controller",
    "-p", "move_duration_s:=2.0",
]

# Standalone components for the LIVE camera/RViz toggles.
RVIZ_STANDALONE = ["ros2", "launch", "ur3e_gripper_3d_moveit_config", "moveit_rviz.launch.py"]
CAMERA_STANDALONE = ["ros2", "launch", "gripper_3d", "realsense_d345_hardware.launch.py"]

# observation_ptp_commander exit codes (from its source).
MOVE_EXIT_TEXT = {
    0: "OK",
    1: "setup failed / unknown target",
    2: "planning failed",
    3: "execution failed — is the pendant program PLAYING?",
}

MOVE_CMD_BASE = [
    "ros2", "run", "ur3e_gripper_3d_moveit_commander", "observation_ptp_commander",
    "--ros-args",
    "-p", "execute:=true",
    "-p", "one_shot:=true",
]

# Filter parameters shown/edited live on the pipeline node, grouped as the
# filter-chain log names them.
# The Filter-parameters tab shows two titled sections (Majd, 2026-08-11):
# grasp filters on top, perception/capture knobs below.
PERCEPTION_PARAMS = [
    "max_graspgen_points",
    "graspgen_num_grasps",
    "graspgen_topk_num_grasps",
    "pregrasp_filter_voxel_pre_downsample_m",
    "min_points_for_graspgen",
    "apply_cloud_cleaning",
    "conf",
    "registration_voxel_size_m",
    "registration_continuous_observation_max_frames",
]

# 2026-08-24 thesis teardown: the tab now shows ONLY the thesis §4.6.14
# cascade knobs (plus the executor floor and cloud-cleanup toggles). The
# retired debugging-era knobs (tilt band, horizontal weight, camera view
# cone, from-behind, reject-side, …) still exist as node params but are
# no longer read by the cascade — showing them would only mislead.
FILTER_PARAMS = [
    # thesis stage 2 — support-plane (TCP above table; no ascending approach)
    "support_min_height_m",
    "support_from_below_dot",
    # coffee-grasp level band + thesis stage 3 mid-height band
    "max_level_tilt_deg",
    "side_horizontal_dot",
    "side_height_min_frac",
    "side_height_max_frac",
    # thesis stage 1 — from-behind (camera-frame rule)
    "from_behind_min_dot",
    # front-of-camera preference (score reweight, 0 = off)
    "front_approach_weight",
    # up-vector source: "tf" (fixed table via TF) or "measured" (ring ladder)
    "up_vector_source",
    # colour-based table measurement (2026-08-27): primary; ring = fallback
    "table_from_colour",
    "table_colour_h_lo",
    "table_colour_h_hi",
    "table_colour_s_min",
    "table_colour_v_min",
    # neighbour obstacle boxes (2026-08-27): one padded box per NON-target object
    "neighbour_obstacles_enabled",
    "neighbour_obstacle_padding_m",
    # thesis stage 4 — roll-upright (fingers level) + palm twin fix
    "max_finger_vertical_dot",
    "palm_canonicalize",
    # thesis stage 6 — centroid Gaussian reweight (TCP vs object middle)
    "centroid_sigma_factor",
    # executor-side safety floor + cloud cleanup
    "candidate_min_world_z",
    "apply_largest_cluster_filter",
    "apply_radius_outlier_filter",
    "radius_filter_m",
    "apply_plane_above_filter",
]

# (section title, parameter list) in display order; ALL_TUNABLE_PARAMS is
# the flat list used for fetch/apply.
PARAM_SECTIONS = [
    ("GRASP FILTERS", FILTER_PARAMS),
    ("PERCEPTION & CAPTURE", PERCEPTION_PARAMS),
]
ALL_TUNABLE_PARAMS = FILTER_PARAMS + PERCEPTION_PARAMS

# Voice-filter parameters of the STT node (/speech_input_node), edited by
# the "Voice filters" group and carried in the profile's "voice" section.
# They are NOT pipeline parameters (different node), so they are
# deliberately kept out of ALL_TUNABLE_PARAMS. One list, three readers:
# ros_bridge.MortyGuiBridge.VOICE_FILTER_PARAMS, profiles.VOICE_KEYS and
# app's Voice-filters widgets.
VOICE_FILTER_PARAMS = [
    "whisper_min_rms",
    "whisper_silence_ms",
    "whisper_max_utterance_s",
    "min_chars",
]

# Hover documentation for the filter parameters (Majd's request):
# what each one actually does, in operator language.
PARAM_DOCS = {
    "open_set_enabled": "Open-set detection (2026-09-15): YOLOE draws a box around every object on the "
                        "table, the trained YOLO keeps its own ten, the rest are named once by the naming "
                        "service (q8b in the overlay; ys = trained YOLO). Off = detection exactly as before.",
    "identity_check_enabled": "A trained-YOLO detection must look like the reference photos of its class "
                              "(DINOv2 similarity, 10 ms) or it is treated as an unknown object and renamed. "
                              "Needs models/identity_refs (tools/build_identity_refs.py); inactive without them.",
    # ── thesis §4.6.14 cascade (2026-08-24) ──
    "support_min_height_m": "Thesis stage 2: minimum gripper-TCP height above the table plane (m). Grasps below are rejected.",
    "support_from_below_dot": "Thesis stage 2: reject approaches that ASCEND more than asin(this) — 0.2 ≈ 11.5°. Stops 'up through the object' grasps; raise to ~0.4 if too strict for the low camera pose.",
    "side_horizontal_dot": "Thesis stage 3: |approach·up| below this = a SIDE grasp (0.5 = within 30° of horizontal). Only side grasps get the height band.",
    "max_level_tilt_deg": "Coffee-grasp rule: approach must be within this many degrees of level (both directions), measured against the MEASURED table plane. 0 disables.",
    "side_height_min_frac": "Mid-height band: grasp point at least this fraction of the object's height above the table (0.10 = not scraping the table).",
    "side_height_max_frac": "Mid-height band: grasp point at most this fraction of the object's height (0.90 = below the cap).",
    "from_behind_min_dot": "Camera rule: reject approaches opposing the camera→object line beyond this dot. -0.15 keeps side grasps, kills the unseen back. -1 disables.",
    "front_approach_weight": "Prefer grasps approaching from the CAMERA'S FRONT: score multiplied by exp(w*(approach.view-1)). Higher = stronger preference, 0 = off. Also speeds probing (side grasps fail IK).",
    "up_vector_source": "Where UP comes from: 'tf' = fixed table direction through the calibrated camera TF every cycle (Majd's method); 'measured' = measure the table in the photo, with memory + TF + camera-prior fallbacks.",
    "table_from_colour": "Measure the table plane from the yellow tabletop COLOUR first (primary source); the depth ring around the bbox stays the fallback.",
    "table_colour_h_lo": "Yellow hue lower bound (OpenCV H, 0-179). Default 20.",
    "table_colour_h_hi": "Yellow hue upper bound (OpenCV H, 0-179). Default 45.",
    "table_colour_s_min": "Minimum saturation (0-255) for a pixel to count as table yellow — rejects grey/washed-out pixels.",
    "table_colour_v_min": "Minimum brightness (0-255) for table yellow — rejects deep shadow.",
    "neighbour_obstacles_enabled": "Publish a padded collision BOX for every detected object EXCEPT the grasp target, so MoveIt routes around the other items on the table instead of through them. Off = planner sees only the static furniture.",
    "neighbour_obstacle_padding_m": "Safety cushion added around each neighbour box, per side (m). 0.01 = 1 cm. Bigger = safer but harder to find a plan in a crowded scene.",
    "max_finger_vertical_dot": "Thesis stage 4 (fingers): finger-opening axis must stay within asin(this) of level — 0.5 = 30°. A vertical finger dives into the table. Strict: no fallback.",
    "palm_canonicalize": "Rotate any wrong-side-up candidate 180° about its approach axis — the twin grips identically and the gripper arrives right side up. Lossless; replaces the old (sign-inverted) flipped-grasp filter.",
    "centroid_sigma_factor": "Thesis stage 6: Gaussian score reweight around the object middle, judged at the gripper TCP. σ = factor × object size; 0.5 = thesis default ('half the bounding-box size'). 0 disables.",
    "apply_finger_penetration_filter": "Reject grasps whose closing finger volumes would crash through object points. The most expensive filter (~10 s CPU).",
    "finger_penetration_finger_length_m": "Length of the simulated finger boxes used by the penetration check.",
    "finger_penetration_finger_thickness_m": "Thickness of the simulated finger boxes.",
    "finger_penetration_gripper_opening_m": "Maximum opening between the two simulated fingers.",
    "ignore_behind_grasps": "Reject grasps that approach from BEHIND the object (away from the camera).",
    "behind_grasp_threshold": "How far behind the object centroid (m) an approach must start to count as 'behind'.",
    "reject_from_behind_min_forward_dot": "Minimum dot(approach, camera view). 0.0 allows sideways approaches; raise toward 1.0 to force front-only grasps (fixes flat-object edge pinches).",
    "reject_side_grasps": "Reject grasps approaching from the object's sides.",
    "prefer_side_grasps": "Opposite bias: prefer side approaches over top ones.",
    "max_down_from_centroid": "How far BELOW the object centroid (m) a grasp point may sit.",
    "max_up_from_centroid": "How far ABOVE the object centroid (m) a grasp point may sit.",
    "max_grasp_centroid_dist_m": "Maximum distance (m) between grasp position and the object centroid.",
    "orientation_consistency_filter": "Drop grasps whose orientation deviates from the dominant orientation cluster.",
    "max_orientation_delta_deg": "Allowed deviation (deg) for the orientation-consistency filter.",
    "orientation_score_reweight": "Reweight grasp scores by how much wrist rotation the robot would need.",
    "orientation_score_weight": "Strength of the orientation reweight (0 = off, 1 = full).",
    "orientation_score_ref_deg": "Wrist rotation (deg) treated as the reference cost of 1.0 (thesis: 45°).",
    "apply_largest_cluster_filter": "Keep only the largest connected point cluster (removes background fragments).",
    "apply_radius_outlier_filter": "Remove lonely points with too few neighbours.",
    "radius_filter_m": "Neighbourhood radius (m) for the outlier removal.",
    "apply_plane_above_filter": "Keep only points above the fitted table plane.",
    "candidate_min_world_z": "Minimum TCP height in world/base coordinates. The modeled tabletop ends at 0.600 m; default 0.615 m adds 15 mm clearance. 0 disables.",
    "horizontal_score_weight": "How strongly to prefer horizontal approaches in candidate scoring. 0 disables the preference.",
    "require_horizontal_approach": "When enabled, reject approaches tilted beyond max_approach_tilt_deg.",
    "max_approach_tilt_deg": "Maximum approach tilt away from horizontal in either direction. Default 20 degrees.",
    "max_downward_approach_tilt_deg": "Stricter downward approach limit. Default 5 degrees prevents the 10 cm LIN approach from dropping toward the table.",
    "center_band_m": "Optional grasp-height limit around the object centroid. 0 disables this hard height restriction.",
    "max_graspgen_points": "Downsample the object cloud to at most this many points before sending to GraspGen. Default 1000; -1 sends everything and is slower.",
    "graspgen_num_grasps": "How many grasps GraspGen samples per request (was 1500). Fewer = faster inference AND faster filter chain.",
    "graspgen_topk_num_grasps": "Keep only the best N grasps by confidence before filtering. -1 keeps all; 200 cuts downstream work ~6x.",
    "pregrasp_filter_voxel_pre_downsample_m": "Voxel size (m) for the cheap pre-downsample before the filter chain runs.",
    "min_points_for_graspgen": "Skip GraspGen entirely when the segmented cloud has fewer points than this (too little geometry for sane grasps).",
    "apply_cloud_cleaning": "Extra statistical cloud cleanup before grasp generation.",
    "conf": "YOLO detection confidence threshold (0-1). Lower finds more objects but more ghosts.",
    "registration_voxel_size_m": "Voxel size (m) for fusing multi-view clouds during registration.",
    "registration_continuous_observation_max_frames": "How many frames the continuous-observation sweep captures at most.",
}

# Tooltip help for the Motion tab's compact groups (Majd 2026-08-28 round
# 2: "why this too big box when it can be smaller"). The long grey notes
# that used to eat a layout row each live here instead — same words, zero
# vertical space. Built from PARAM_DOCS where a doc already exists, so the
# operator-language description of a parameter stays written down once.
NEIGH_HELP = (
    "Avoid other objects on the table (neighbour collision boxes).\n\n"
    + PARAM_DOCS["neighbour_obstacles_enabled"]
    + "\n\nPadding: " + PARAM_DOCS["neighbour_obstacle_padding_m"]
    + "\n\nOne box per detected object EXCEPT the target — watch them on\n"
    "/grasp_pipeline/neighbour_obstacles_cloud in RViz."
)

GRIP_HELP = (
    "How long the adapter holds the gripper's open/close pulse.\n"
    "Applied on every Open/Close press, and live as you change them.\n"
    "Adapter maxima: open 1.0 s, close 1.5 s."
)

PROFILE_HELP = (
    "Save / Load / Reset every tunable this GUI owns.\n\n"
    "Pipeline, gripper and voice-filter parameters are RUNTIME-ONLY — they\n"
    "die with the node process. A profile is a .yaml snapshot of them (plus\n"
    "the motion GUI state) under ~/morty_ws/config/gui_profiles/.\n\n"
    "Reset to defaults restores the values declared in\n"
    "grasp_pipeline/selection/params.py, and for the four voice filters in\n"
    "communication_llm/speech_input_node.py — read from those files at\n"
    "runtime, so they never disagree with the code.\n\n"
    "A profile saved before the voice filters were included simply has no\n"
    "'voice' section; loading it leaves them untouched."
)

GREEN = "#2ecc40"
RED = "#d33a3a"
ORANGE = "#f39c12"
GREY = "#666"


# ── ROS bag per trial (Hamid 2026-09-09) ────────────────────────────────────
# Recorded by bag_link.BagLink for the span of every MMHRI trial / VLA
# episode. Compressed colour only (depth is not used by the VLA models and
# would be 18 MB/s raw); everything else is small. ~2.5 MB/s in total.
BAG_TOPICS = [
    "/camera/camera/color/image_raw/compressed",          # wrist RealSense
    "/camera/camera/color/camera_info",
    "/astra/color/compressed_slow",                       # fixed Astra (shuttle PC relay, 10 Hz)
    "/astra/color/camera_info",
    "/joint_states",
    "/tcp_pose_broadcaster/pose",                         # end-effector pose
    "/scaled_joint_trajectory_controller/controller_state",  # commanded vs actual joints
    "/io_and_status_controller/io_states",                # gripper: TI0 holding, TI1 slip, tool outputs
    "/gripper_controller/joint_states",                   # finger joint mirror
    "/speed_scaling_state_broadcaster/speed_scaling",
    "/io_and_status_controller/robot_program_running",
    "/speech/accepted_transcript",                        # what the human said
    "/robot/intent",                                      # what the LLM decided
    "/yolo/all_detections",
    "/graspgen/selection_cycle_id",
    "/grasp_executor_lin_llm/selected_pose",
    "/grasp_executor_lin_llm/interactive_status",
    "/grasp_executor_lin_llm/cycle_done",
    "/tf",
    "/tf_static",
]
# Trials with MMHRI evidence keep their bag inside the trial folder; a
# VLA-only trial (no trial folder) puts it here, one folder per trial.
BAG_DIR_VLA_ONLY = os.path.expanduser("~/morty_ws/vla_experiments/bags")
