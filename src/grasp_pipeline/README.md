# grasp_pipeline

Perception and grasp generation for Morty. One ROS 2 node,
`yolo_live_select_grasp`, takes the wrist RealSense stream and turns a
selected object into ranked grasp candidates for the executor.

## What the node does, in order

1. **Detection** — YOLO on every colour frame (`yolo_detector.py`), detections
   published on `/yolo/all_detections` with a base-frame distance and a
   reach verdict per object (`perception/reach_envelope.py`: green, `edge N%`,
   `too far`).
2. **Selection** — finger pointing with MediaPipe (`point_selector.py`,
   3D ray from the index finger through the depth image, dwell to commit),
   a click in the debug window, or a voice/LLM choice on `/yolo/select_index`
   and `/yolo/select_detection`.
3. **Capture** — segmentation of the selected object (`segmentation.py`),
   median fusion of the last five depth frames kept in `frame_buffer.py`'s
   ring (no waiting at selection), object cloud extraction and
   table-plane estimate (`perception/cloud_extraction.py`), one padded
   collision box per neighbouring object for the planner
   (`perception/neighbour_obstacles.py`).
4. **GraspGen** — NVIDIA GraspGen on the object cloud, in its own
   process since 2026-09-09 (`GraspGen/scripts/graspgen_service.py`, ZMQ on
   port 5555, started by the GUI with the pipeline; `GRASPGEN_MODE=zmq` in
   `grasp_client.py`). One inference is 0.10 s there against 0.3–0.9 s
   inside this node, whose YOLO, hand tracker and camera callbacks compete
   for the interpreter lock. `GRASPGEN_MODE=inprocess` restores the old path.
   With `graspgen_precompute:=true` (off by default; `MORTY_GRASP_PRECOMPUTE=true`
   for a GUI-started pipeline) a worker runs steps 3–5 for every detection
   that has been stable for 0.2 s while nobody has selected
   (`grasping/precompute.py`). A selection whose stored result is still
   valid — object box, depth and arm pose unchanged, parameters unchanged —
   replays the captured outputs and publishes at once (`[precompute] HIT`
   in the log; otherwise `MISS: <why>` and the normal path). Precomputed
   objects carry a `*` in the overlay. Nothing is published or moved by the
   worker itself. An object whose grasps failed (none survived, or the
   executor could not carry out the pick) is recomputed up to
   `precompute_max_retries` (5) times, always behind objects not yet
   computed; a moved object starts again from zero.
5. **Filter chain** — `grasping/filter_pipeline.py`: upright/level band,
   from-behind rejection, height window, roll canonicalisation, front
   preference, centroid and end-effector orientation re-weighting,
   orientation consistency. Survivors are published on
   `/graspgen/candidate_poses_base_link` together with
   `/graspgen/selection_cycle_id`; the node then waits for the executor's
   `/grasp_executor_lin_llm/cycle_done`.

Debug outputs: `/yolo/debug_image` (boxes, labels, pointer, hand
landmarks, selected mask, reach colours), `/graspgen/viz/*` pose arrays for
RViz and the GUI, `/yolo/selected_cloud_rgb`.

## Running

Normally started by the GUI. By hand:

```bash
ros2 launch grasp_pipeline run_grasp_pipeline.launch.py
```

The launch declares every node parameter as a launch argument (about 230);
the ones that matter day to day:

| argument | meaning |
| --- | --- |
| `yolo_world_model`, `conf`, `imgsz`, `device`, `yolo_half` | detector model and thresholds |
| `pointing_mode` (`3d`/`2d`), `mediapipe_cone_deg`, `mediapipe_hysteresis`, `mediapipe_min_hold_ms` | pointing behaviour |
| `graspgen_num_grasps`, `grasp_threshold`, `max_graspgen_points` | GraspGen request |
| `neighbour_obstacle_padding_m` | padding of the neighbour boxes (5 mm) |
| `single_view_multiframe`, `single_view_frames` | depth fusion |
| `publish_debug_image`, `debug_publish_fps` | overlay |
| `tf_listener_max_hz` | decimated `/tf` listener rate (see below) |
| `graspgen_precompute`, `precompute_stable_s` | grasp precompute cache (see step 4) |

## Files

| path | role |
| --- | --- |
| `grasp_selection_node.py` | the node: timers, selection thread, state machine |
| `grasping/pick_pipeline.py` | one pick cycle from selection to published candidates (`_compute` = perception phases, `_publish` = candidates and viewer poses) |
| `grasping/precompute.py` | the precompute worker: object tracks, validity checks, replay |
| `grasping/filter_pipeline.py`, `grasp_filters.py`, `grasping/orientation.py`, `grasping/grasp_scorer.py` | the filter chain and scoring |
| `perception/` | cloud extraction, neighbour obstacles, reach envelope |
| `point_selector.py`, `selection/params.py` | pointing/voice selection and all parameter declarations |
| `debug/debug_renderer.py`, `debug_image_publisher.py` | the overlay |
| `tf_listener_lite.py` | raw `/tf` drained at 50 Hz on its own thread; the stock tf2 listener at 830 Hz starved the whole process |
| `frame_buffer.py` | latest colour/depth/intrinsics |
| `motion/`, `registration_pipeline/`, `registration.py`, `nbv_candidates.py`, `tsdf_fusion.py` | multi-view capture (observation sweeps) |
| `centroid/`, `open3d_cloud_viewer.py`, `grasp_viz_viewer.py`, `click_select_viewer.py`, `scene_obstacle_publisher.py`, `pose_to_base.py` | helper nodes and viewers |
| `models/hand_landmarker.task` | MediaPipe hand model (shipped) |
| `scripts/reach_table.py` | regenerates the reach table from the arm kinematics |

The YOLO weights `yolo26s-seg_24_04.pt` live in this package folder and are
in the repository (23 MB).

## Performance notes

GraspGen inference is 0.2–0.9 s; segmentation 0.1–0.4 s; a full selection-
to-candidates cycle about 3 s. The `[perf] infer_ms_avg` line in the log is
the health gauge: YOLO under 300 ms means the process is healthy, above
500 ms something is starving it (machine load, or another Python thread
holding the interpreter lock).
