# tools/

Experiment, debug, and dataset utilities for the grasping pipeline. None of
these are required to *run* the system — they automate experiments, parse logs,
generate plots/reports, and help capture training data.

Most scripts honor `WS_DIR` (default `$HOME/new_ur_gripper_ws`). All hardcoded
old-machine paths have been replaced with `$HOME`-relative ones.

## Multi-view grasping experiments

Run repeated multi-view picks and capture timestamped logs + structured CSV/JSON
for later analysis. This is the harness you launch when collecting experiment
data for the thesis.

```bash
# default: new 9-class YOLO model, multi-view + dynamic NBV + TSDF
bash ~/new_ur_gripper_ws/tools/run_grasp_experiment.sh --tag vitamin_tube_run1

# use the older 3-class model
bash ~/new_ur_gripper_ws/tools/run_grasp_experiment.sh --tag retail_box_old1 --model-preset old

# pass extra ros2 launch overrides
bash ~/new_ur_gripper_ws/tools/run_grasp_experiment.sh --tag run42 registration_motion_timeout_s:=150
```

What it writes under `$WS_DIR/src/experiment_logs/`:

| Suffix              | Contents                                                                 |
|---------------------|--------------------------------------------------------------------------|
| `…_<tag>.log`        | Full raw stdout/stderr of the run                                        |
| `…_<tag>_summary.log`| Greps the key NBV / sweep / grasp-utility lines for quick scanning       |
| `…_<tag>_meta.txt`   | Run metadata (model, classes, launch args)                               |
| `…_<tag>_structured.json` | Parsed runs/views (built by `parse_grasp_experiment.py`)           |
| `…_<tag>_run.csv`    | One row per run                                                          |
| `…_<tag>_views.csv`  | One row per accepted view                                                |

A `trap` makes sure the summary + CSV/JSON are written **even if you Ctrl-C** the run.

### Combine and analyse many runs

```bash
# 1. Parse a single run's raw log (run_grasp_experiment.sh already does this on exit):
python3 ~/new_ur_gripper_ws/tools/parse_grasp_experiment.py \
    --raw-log <ts>_<tag>.log \
    --json-out <ts>_<tag>_structured.json \
    --run-csv-out <ts>_<tag>_run.csv \
    --views-csv-out <ts>_<tag>_views.csv

# 2. Aggregate every run in the experiment_logs/ directory into one CSV:
python3 ~/new_ur_gripper_ws/tools/aggregate_grasp_experiments.py

# 3. Generate plots from the aggregate CSV:
python3 ~/new_ur_gripper_ws/tools/generate_experiment_plots.py

# 4. Build an HTML report from the aggregate CSV:
python3 ~/new_ur_gripper_ws/tools/build_experiment_report.py
```

All four python scripts use `argparse` and accept `--help` to show their full
flag list and defaults.

## Debug viewer (cloud + grasps)

Render a point cloud + the grasps GraspGen returns as a PNG, headless (no GUI
window — works on WSL). Reads either a sample-data JSON or a precomputed
`{pc, grasps}` JSON.

```bash
# render the newest GraspGen sample object + its grasps:
python3 ~/new_ur_gripper_ws/tools/grasp_debug_viewer.py --topk 20 --save /tmp/grasps.png

# from a JSON the pipeline produced (used by GRASPGEN_SAVE_VIZ):
python3 ~/new_ur_gripper_ws/tools/grasp_debug_viewer.py --from-json input.json --save out.png
```

The live pipeline can call this automatically every grasp request — set
`GRASPGEN_SAVE_VIZ=/path/to/dir` before launching; each result writes
`grasp_<timestamp>.png` plus `latest.png` in that dir.

## YOLO dataset capture

Capture ROS2 camera frames to JPG/PNG + a manifest CSV, so you can later label
them (CVAT / Roboflow / Label Studio) and fine-tune the YOLO segmentation model.

```bash
python3 ~/new_ur_gripper_ws/tools/capture_yolo_dataset.py --help
```

## ROS graph export

Snapshot the LIVE ROS node/topic graph to DOT/SVG/PNG (useful for thesis
figures or sanity-checking what's actually running).

```bash
# run while the system is live (morty.sh / launch is up in another terminal):
python3 ~/new_ur_gripper_ws/tools/export_ros_graph.py --output-prefix ~/ros_graph
```
