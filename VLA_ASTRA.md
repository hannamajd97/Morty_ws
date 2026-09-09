# VLA_ASTRA.md — Complete handover: Astra camera, shuttle link, and VLA data recording for Morty

Written 2026-09-03 by Claude (Claude Code session on the Alienware laptop), for whoever — human or AI — continues this work.
It covers everything done in this session (2026-09-02 and 2026-09-03), why it was done, what the
system looks like right now, how to operate it, what broke and why, and what is still open.

If you only read one section, read **§10 Current state** and **§11 Runbooks**.

---

## Table of contents

1. Goal of the work
2. The system before this session (what already existed)
3. Key facts you must not get wrong
4. Day 1 (2026-09-02) — timeline of what was done and why
5. Day 2 (2026-09-03) — timeline of what was done and why
6. Hardware and network: laptop, dock, shuttle PC, robot, cameras
7. The Astra camera pipeline in detail
8. The Fast DDS / ROS 2 networking lessons (read before touching discovery)
9. The `vla_experiments` package in detail (recorder, formats, converters, tools)
10. Current state of the system (end of session)
11. Runbooks (start, restart after failures, record, inspect, convert)
12. What the VLA models need (openpi and OpenVLA) and the training plan
13. Known issues, open items, next steps
14. Pitfalls and traps encountered (so you don't repeat them)
15. File index (everything created or modified, with paths)
16. Command cheat-sheet

---

## 1. Goal of the work

Majd (the user, `deep@Alienware15R3`, email reddevil96mh@gmail.com) wants to fine-tune a
vision-language-action model (VLA) — **OpenVLA** and/or **openpi (π0 / π0-FAST / π0.5)** — on
demonstrations recorded with the Morty robot system (a UR3e arm with a custom 3D-printed pneumatic
gripper, a wrist-mounted Intel RealSense D435, and a voice/LLM pipeline).

For that, a second, fixed "exterior" camera was needed in addition to the wrist camera. The camera
chosen is an **Orbbec Astra Pro** (structured-light RGB-D, USB 2.0). Its cable is too short to reach
the laptop, so it is plugged into a second PC (the "shuttle PC", hostname `tory`), which is connected
to the laptop through a network cable. The camera data has to reach the laptop over that cable.

Concretely, the session had to deliver:

1. Get the Astra recognised, streaming, and visible as a point cloud in RViz on the laptop.
2. Put the Astra at the correct pose relative to the robot (`base_link -> astra_link`).
3. Find out exactly what OpenVLA and openpi need as training data, and build a recorder that captures
   it (both cameras, joint states, end-effector pose, gripper state, actions, language instruction),
   synchronised at a fixed rate, in a folder separate from the existing MMHRI experiments.
4. Make it possible to record the normal MMHRI experiments and the VLA data **at the same time**,
   without mixing the two, and only for trials that actually move the robot (not for "what do you
   see" questions).
5. Provide converters to the LeRobot format (openpi) and the RLDS format (OpenVLA).

All five are done to the degree described in §10. The main unresolved point is the exact Astra pose
(§7.7) — Majd set it by eye and confirmed "very near"; an automatic calibration was tried and rejected.

---

## 2. The system before this session (what already existed)

Workspace: `/home/deep/morty_ws` (ROS 2 Humble, Ubuntu 22.04, Fast DDS, **not a git repo**, merged
colcon install layout: installed files are under `install/share/<pkg>/...` and `install/lib/...`,
not `install/<pkg>/...`).

Packages under `src/` (16 colcon packages; 9 project packages + UR vendor driver/description):

| package | purpose |
|---|---|
| `ur3e_gripper_3d_moveit_config` | robot bring-up: UR driver + MoveIt + RealSense + static TFs; launch `ur3e_gripper_3d_moveit_real_camera.launch.py` is what the GUI starts |
| `gripper_3d` | URDF/xacro of the gripper and camera mount, RealSense launch (`realsense_d345_hardware.launch.py`), gripper IO adapter |
| `grasp_pipeline` | YOLO + grasp generation (Python, runs in a venv via `run_grasp_pipeline_venv`) |
| `ur3e_gripper_3d_moveit_commander` | C++ executors (`grasp_executor_lin_llm` etc.), gripper action server |
| `communication_llm` | voice: `chat_node`, `speech_input_node`, `speak_action_server`; publishes `/speech/accepted_transcript`, `/robot/intent` |
| `intent_router` | turns intents into robot actions |
| `morty_gui` | the Tk/Qt GUI (`ros2 run morty_gui morty_gui`) that starts/stops driver, pipeline, voice, and records MMHRI trials |
| `ur3e_fast_observation` | named poses (`observation_poses.yaml` at workspace root), replay node |
| `morty_interfaces` | msgs/srvs/actions |
| `vla_experiments` | **NEW in this session** (see §9) |

Existing experiment recording: the GUI's MMHRI window writes `experiments/mmhri/trial_NNN/`
(camera.mp4 of the YOLO debug image, audio.wav, conversation.txt, events.json, trial.json, PLY
clouds, per-trial log slices) plus two master CSVs. **It records derived data, not raw sensor
streams**: no depth, no joint states, no TF. Nothing in the workspace used rosbag, LeRobot, RLDS,
HDF5 or parquet before this session.

Robot motion is done by MoveIt (Pilz LIN/PTP, OMPL fallback) and by direct
`FollowJointTrajectory` goals to `/scaled_joint_trajectory_controller`. There is no teleoperation.

Key topics that already existed and are used by the new recorder:

| topic | type | notes |
|---|---|---|
| `/joint_states` | sensor_msgs/JointState | on real hardware the 6 arm joints from the UR driver's `joint_state_broadcaster` **plus** the gripper joint merged in by `gripper_command_adapter`; when the arm driver is down only the 2 gripper joints remain (this is how you detect a dead driver) |
| `/scaled_joint_trajectory_controller/controller_state` | control_msgs/JointTrajectoryControllerState | commanded reference positions; exists but published no data during the tests (controller possibly inactive) |
| `/tcp_pose_broadcaster/pose` | geometry_msgs/PoseStamped | TCP pose in frame `base`; recorder prefers TF `base_link -> gripper_tcp` |
| `/io_and_status_controller/io_states` | ur_msgs/IOStates | tool digital input pin 16 = grasp detected |
| `/camera/camera/color/image_raw[/compressed]` | wrist RealSense colour | recorder uses the `/compressed` (JPEG) variant |
| `/camera/camera/aligned_depth_to_color/image_raw/compressedDepth` | wrist depth PNG | used by the calibration tool |
| `/speech/accepted_transcript` | std_msgs/String | the spoken command after acceptance |
| `/robot/intent` | std_msgs/String (JSON) | `{"action_intent": "...", "object_intent": ..., "object_attributes": ...}` |
| `/grasp_executor_lin_llm/cycle_done` | std_msgs/String | `"done|<cycle_id>[|note]"` or `"failed|<cycle_id>|<reason>"` |

Frames: root `world` ≡ `base_link` (identity). Wrist camera chain on real hardware:
`tool0 -> camera_mount` (xyz 0.0001 -0.09405 -0.01335, rpy 0 -1.3963 1.5708) `-> camera_camera_link`
(xyz 0.0042 0.0175 0.0125) `-> camera_color_optical_frame` etc. (published by the RealSense driver).
The Morty RViz config (`ur3e_gripper_3d_moveit_config/config/moveit.rviz`) uses **`camera_camera_link`
as Fixed Frame** — this matters (§14).

---

## 3. Key facts you must not get wrong

- **The laptop** is `Alienware15R3`, user `deep`, Ubuntu 22.04, ROS 2 Humble, GTX 1080 8 GB (cannot
  fine-tune a VLA; fine for recording and conversion). Its wired NIC `enp70s0` is on the robot network
  as `192.168.1.101`; the UR3e controller is `192.168.1.102`. NetworkManager profile "Profile 2" holds
  that static IP. Wi-Fi `wlp71s0` is on "freeCampus" (192.168.16/20).
- **The shuttle PC** is hostname `tory`, user `demo`, Ubuntu 22.04, ROS 2 Humble already installed,
  8 cores, 15 GB RAM, **no GPU**, Docker installed. Its wired port `eth1` has a static IP that was
  **mistyped by whoever set it up: `168.192.1.102`** (they meant 192.168.1.102, the robot's IP).
  Its Wi-Fi is `192.168.1.120` on router RCR4-104 — the same numeric range as the robot LAN, so the
  shuttle's wired port must **never** be put on 192.168.1.x. The laptop's SSH key is installed on the
  shuttle: `ssh demo@168.192.1.102` works without a password. `sudo` on the shuttle needs demo's
  password (Majd knows it; it was typed once in this session).
- **The link** laptop⇄shuttle is a plain Ethernet cable from the Ethernet port of a USB-C dock
  (Realtek RTS5411 hub + ASIX AX88179A NIC, Linux interface `enxc8a362a67c99`) to the shuttle's eth1.
  NetworkManager profile `shuttle-link` on the laptop: static `192.168.2.1/24` **plus**
  `168.192.1.101/24` (the second address is a workaround for the shuttle's typo), `never-default`.
  Plan (not done): change the shuttle's eth1 to `192.168.2.2/24` and drop the 168.192 address.
- **Base frame orientation**: in this robot's `base_link`, **true vertical (up) is the direction
  (-0.707, 0.707, 0)**; base +z is horizontal. Consequence: "70.9 cm straight above the base" is the
  translation (-0.501, 0.501, 0), which is what Majd converged to for the Astra. The tabletop is at
  about 0.622 m above the base along that vertical; the table's yaw is 45° (0.7854 rad) about base z
  (`ur3e_gripper_3d_moveit_config/config/environment.yaml`). Earlier memory notes saying the same
  thing were confirmed by the data in this session.
- **Astra pose in use** (set by Majd visually, "very near", saved as defaults):
  `base_link -> astra_link` xyz **(-0.51, 0.51, 0.065)**, rpy **(0.0, -0.576, -0.77)**.
- The VLA data folder is `~/morty_ws/vla_experiments/datasets/<dataset_name>/` — deliberately outside
  `experiments/` (MMHRI) so the two never mix.
- Every camera stream from the shuttle is **compressed by a relay on the shuttle** and decoded on the
  laptop. Raw 640×480 images and raw clouds (4.9 MB each) **cannot** cross the link with the default
  kernel socket buffers (`net.core.rmem_max = 212992` on both machines, never raised).

---

## 4. Day 1 (2026-09-02) — timeline

### 4.1 "I connected a separate camera, can you check it"
- `lsusb`/kernel log showed only the built-in webcam. The kernel log revealed an Intel RealSense D435
  (serial 816513022440) had been on USB bus 2 and disconnected at 14:35. Later the Astra appeared
  behind a chain of two Realtek hubs on the dock's USB-2 path and **reconnected 27 times in one
  minute** with `-71` errors ("Maybe the USB cable is bad?"). Conclusion: a depth camera on a chained,
  bus-powered USB-2 hub is unusable; plug directly or use an active extension.
- The Astra Pro enumerates as **two USB devices**: `2bc5:0403` "Astra Pro" (depth, OpenNI2, vendor
  class) and `2bc5:0501` "Astra Pro HD Camera" (RGB, standard UVC webcam, `/dev/videoN`).
- Majd could not plug the Astra into the laptop (cable too short) and moved it to the shuttle PC.

### 4.2 Getting a network path to the shuttle
- A USB-C dock with a Genesys hub + RTL8153 Ethernet was tried first; later a different dock (RTS5411
  + AX88179A) was used. First attempt to configure the dock NIC with `nmcli device connect` **stole the
  robot's "Profile 2"** from `enp70s0` and cut the UR3e link for a minute. Recovered with
  `nmcli con up "Profile 2" ifname enp70s0`. Memory note `nmcli-steals-robot-profile` records this:
  never bare `nmcli device connect` on a new NIC; create a dedicated profile.
- Created profile `shuttle-link` (192.168.2.1/24). Majd photographed the shuttle's network settings:
  eth1 = 168.192.1.102/24 (typo), Wi-Fi 192.168.1.120, docker bridges 172.17.0.1 and 192.168.56.1
  (irrelevant). Added 168.192.1.101/24 to the laptop's dock profile; the shuttle then answered ping,
  MAC 80:ee:73:dc:64:f6, SSH open.
- Getting a login: Majd did not know the shuttle credentials at first. Eventually he logged in as
  `demo` (password known to him; hostname `tory`). Attempts by Claude to script the password into SSH
  were blocked by the tool sandbox (correctly). Majd ran `ssh-copy-id demo@168.192.1.102` from the
  laptop; from then on key login works.

### 4.3 ROS 2 discovery across the cable
- Both machines: ROS 2 Humble, Fast DDS, domain 0, `ROS_LOCALHOST_ONLY=0`. The shuttle's `.bashrc`
  sets `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`; all Astra processes on the shuttle are started with
  `rmw_fastrtps_cpp` explicitly. The laptop stack runs with
  `FASTRTPS_DEFAULT_PROFILES_FILE=~/morty_ws/config/fastdds_udp_only.xml` (SHM disabled because of
  earlier segfaults; see `SYSTEM_STATE_LOG.md` #291).
- Discovery worked in tests within ~1 s once the laptop's `ros2` daemon cache was refreshed
  (`ros2 daemon stop; ros2 daemon start`, or `--no-daemon`). Many later "it doesn't work" moments were
  the stale daemon.

### 4.4 Astra driver on the shuttle
- Driver: `orbbec/ros2_astra_camera` (GitHub), cloned into `~/astra_ws/src/ros2_astra_camera` on both
  the laptop (built there too, as a fallback) and the shuttle. The shuttle's apt lists were from 2024
  and its ROS GPG key had expired (2025-06-01); fixed with
  `sudo curl ... ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && sudo apt update`, then
  `apt install libgoogle-glog-dev libuvc-dev ros-humble-camera-info-manager ros-humble-image-publisher
  ros-humble-compressed-image-transport`. The driver bundles its own OpenNI2 (`openni2_redist`).
  udev rule `99-orbbec-usb.rules` (MODE 0666, GROUP users) already existed on the shuttle.
- **Driver bug #1 (segfault at start):** `PointCloudXyzNode`/`PointCloudXyzrgbNode` call
  `connectCb()` (subscribes to depth) **before** creating `pub_point_cloud_`; the first depth frame
  dereferences a null publisher (`segfault at 70 in libastra_camera.so`, backtrace obtained with
  gdb). Patched in `astra_camera/src/point_cloud_proc/point_cloud_xyz.cpp` and
  `point_cloud_xyzrgb.cpp`: create the publisher first (inside the mutex scope), call `connectCb()`
  after the scope (it locks the same non-recursive mutex), and guard `if (!pub_point_cloud_) return;`.
  Patched sources are on both machines.
- **Driver bug #2 (colour):** the libuvc colour path fails with "attempt to claim already-claimed
  interface 1" while the kernel `uvcvideo` driver holds the RGB interface, and the node crashes.
  After that first crash `uvcvideo` is detached and the next start works (colour 640×480 MJPEG 30 Hz).
  Hence the start script starts the driver **twice if the first attempt dies**. After a shuttle reboot
  expect exactly one crash. `/dev/video*` disappears on the shuttle once libuvc owns the interface.
  A V4L2/OpenCV fallback publisher `~/astra_ws/astra_color_v4l2.py` exists but is not used.
- Driver TF: the node names its base frame from parameter `camera_link_frame_id` (default
  `camera_link`, **not** `<camera_name>_link`). The launch file `astra_pro.launch.xml` was patched to
  pass `camera_link_frame_id = $(var camera_name)_link`, so with `camera_name:=astra` the frames are
  `astra_link`, `astra_depth_frame`, `astra_color_frame`, `astra_depth_optical_frame`,
  `astra_color_optical_frame`, topics `/astra/...`. (On day 2 the driver's TF publishing was turned
  off entirely, see §5.)

### 4.5 Bandwidth and discovery problems, and their solutions
- The full XYZ cloud from the driver is 640×480×16 B ≈ 4.9 MB at 30 Hz (≈1.2 Gbit/s) — impossible on
  the link; even a 300 KB message overflowed the receiver's UDP socket buffer (`rmem_max 212992`).
- **Solution: a relay on the shuttle** (`~/astra_ws/astra_slow_relay.py`) that throttles and
  compresses: depth 320×240 16UC1 as PNG (`/astra/depth/compressed_slow`, ~29 KB) and colour 640×480
  as JPEG (`/astra/color/compressed_slow`, ~43 KB), 10 Hz each (was 5 Hz on day 1), plus raw
  subsampled fallbacks `/astra/depth/image_slow` (160×120, 4 Hz) and `/astra/color/image_slow`
  (320×240, 5 Hz). The driver has **no image_transport plugins**, so it does not publish compressed
  topics itself; the relay does the encoding in Python.
- **On the laptop, `astra_cloud_local`** (in `vla_experiments`) decodes the compressed pair and
  builds an XYZRGB cloud `/astra/depth/points_local` (~66 k points, RELIABLE QoS so RViz's default
  subscription matches). Colour is sampled per depth pixel because the driver runs with
  `depth_registration:=true` (depth in `astra_color_optical_frame`, camera_info of depth = colour
  intrinsics scaled to 320×240).
- **Discovery pitfall A:** the first laptop profile set `initialPeersList` with only the shuttle IP.
  In Fast DDS, a non-empty `initialPeersList` **replaces the default multicast peer**, so same-host
  discovery silently dies (nothing on the laptop could see anything else, nothing on the shuttle could
  see the driver). Fix: always include `239.255.0.1:7400` in the list. Both profiles now do.
- **Discovery pitfall B:** without any interface restriction, the shuttle sent every sample to all
  five laptop addresses (three of them routed over the shuttle's Wi-Fi), causing ~50 % loss even of
  tiny messages. An `interfaceWhiteList` on the laptop fixed that, **but** (day 2) a whitelist on the
  dock NIC makes every process using it deaf whenever the dock link drops (the interface loses its
  address). The whitelist was removed again; with the shuttle-side profile fixed (multicast + peer)
  the loss did not come back.
- The laptop's driver-side CLI checks were often misleading because `ros2 topic hz/list` use the
  daemon; use `--no-daemon` or a tiny rclpy probe.

### 4.6 RViz on the laptop
- RViz started from the VS Code terminal crashes with
  `symbol lookup error: /snap/core20/.../libpthread.so.0: undefined symbol: __libc_pthread_init`
  because VS Code (snap) pollutes the environment. Launch RViz with `env -i ...` (see
  `scripts/astra_view.sh`) or from a normal terminal. Same for `gnome-screenshot`.
- An RViz config with both clouds was written:
  `src/vla_experiments/rviz/astra_vs_realsense.rviz` (fixed frame `base_link`, Astra cloud on
  `/astra/depth/points_local`, RealSense raw cloud, Astra colour image, TF).
- The Astra cloud was first shown white (RViz's Color Transformer defaulted to Intensity); set RGB8.
  It looked coarse at 160×120; moving to the compressed 320×240 depth fixed that.

### 4.7 Astra pose (calibration)
- Majd's initial description: "above base_link in y about 70.9 cm and rotated like camera_camera_link".
  First guess (0, 0.709, 0) rpy (0, -1.3963, 1.5708) was far off.
- An automatic registration tool was written (`astra_calibrate`, §9.6): RealSense depth → cloud in
  base_link, Astra cloud in its optical frame, plane alignment + 2-D yaw/shift correlation + FPFH
  RANSAC + point-to-plane ICP against RealSense cloud **plus robot model sampled from the URDF
  collision meshes** at the current joint state. Three runs gave inconsistent results (the RealSense
  mostly sees the floor beyond the table from its pose; plane identification was the weak point).
  Majd rejected the results and asked for manual steps instead.
- Manual procedure Majd followed with the `astra_tf.launch.py` arguments: yaw -45° (`-0.7854`) so the
  camera's red axis aligns with the table edges (table yaw is +45°); translation 0.709 m along true
  vertical = `x:=-0.5013 y:=0.5013`; pitch -30…-33° (`-0.5236…-0.576`) so the camera looks ~57–60°
  down; z (sideways along base z) 0.05–0.15. He settled on
  **`x:=-0.51 y:=0.51 z:=0.065 roll:=0.0 pitch:=-0.5760 yaw:=-0.77`** and asked to save it as the
  current pose. It is saved in the launch defaults and `config/recorder.yaml`; Majd says it is "very
  near", not exact. A later data check showed the Astra's tabletop plane and the RealSense's tabletop
  did **not** coincide in height with the earlier automatic candidate; with the manual pose no such
  numeric check was done — **verify visually with both clouds before relying on it**.

### 4.8 The VLA package (created day 1, extended day 2) — see §9.

---

## 5. Day 2 (2026-09-03) — timeline

### 5.1 Morning check
- The shuttle had rebooted overnight; driver and relay were gone. Restarted with the start script
  (two attempts as expected). Laptop helpers restarted. Streams verified (camera_info 29 Hz,
  compressed depth/colour 4.5 Hz → later 10 Hz, cloud ~5 Hz).
- The relay was raised to 10 Hz for the compressed streams so a 10 Hz recording gets a fresh
  exterior frame every step.

### 5.2 Recording both experiments at once (auto mode)
- Requirement: record MMHRI trials as before **and** VLA episodes, in separate folders, VLA only for
  motion commands. Implemented as **auto mode** in the VLA recorder, driven purely by ROS events the
  trial already emits (no GUI changes): accepted transcript → provisional episode (task = command);
  `/robot/intent` `action_intent` in the motion set → confirmed, otherwise discarded; `cycle_done`
  `done|…` → success, `failed|…` → failure; safety timers (25 s without intent → discard, 240 s →
  stop failed). `meta.json` stores intent, cycle_done payload and `mmhri_trial_hint` (newest
  `experiments/mmhri/trial_NNN` at that moment) to match the two records.
- Motion intents (from `intent_router_node.py`): `pick_and_place, place, drop_held, drop_left,
  drop_right, go_home, carry_release, release_home, release, open_gripper, close_gripper`.
- The recorder now reads **compressed** images for both cameras (`/astra/color/compressed_slow`,
  `/camera/camera/color/image_raw/compressed`); raw 900 KB frames were being dropped by the local
  UDP path.
- `vla_record.launch.py` sets `FASTRTPS_DEFAULT_PROFILES_FILE` to the packaged
  `config/fastdds_shuttle_peer.xml`, `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`, `ROS_LOCALHOST_ONLY=0`,
  and has `auto_mode:=true|false`.
- **End-to-end test passed** with the real robot, the Astra and the wrist camera: a simulated trial
  (transcript → pick_and_place intent → `done|42`) produced an episode with 87 steps in 8.6 s at
  10 Hz, 0 dropped ticks, real joint positions, TCP pose, gripper state, 640×480 JPEGs from both
  cameras, `mmhri_trial_hint = trial_007`; a "what do you see" trial was discarded; a
  `failed|43|…` trial was marked failed. (Test dataset deleted afterwards.)

### 5.3 The recurring "no Astra" and its real causes (in order of discovery)
1. **Arm driver dead** (10:38:57): `ur_ros2_control_node` exited with `-13` (SIGPIPE) right after a
   motion — the pendant's External Control program had stopped. Then `/joint_states` had only the two
   gripper joints and no arm TF existed. Consequences: the recorder refused to write (correct), and
   the Morty RViz could not place anything hanging off `base_link` because its Fixed Frame is the
   wrist camera. Majd restarted the driver from the GUI.
2. **Dock link drops at 10:56 and 11:30** (and again ~11:45): the **shuttle suspends after 20 min**
   of no keyboard/mouse because it sits at the lightdm **login screen** (no graphical user logged in;
   the `demo` sessions are only SSH). Each suspend kills the link (laptop sees carrier 0) and the
   driver. GNOME "AC" sleep setting was already `nothing`; the greeter has its own rule. **Fix applied
   (Majd, with sudo): `systemctl mask sleep.target suspend.target hibernate.target hybrid-sleep.target`
   on the shuttle.** Also set GNOME battery/ac sleep to `nothing` for user demo.
3. **Laptop helpers deaf after a link drop**: the whitelist on the dock NIC (see §4.5 pitfall B) —
   removed from `fastdds_shuttle_peer.xml`.
4. **Cloud builder crashed once** ("publisher's context is invalid") and once produced only 1 Hz at
   100 % CPU when three RViz windows subscribed with RELIABLE QoS over UDP loopback (retransmission
   churn with 1 MB messages). Restarting it with `scripts/astra_restart_laptop.sh` fixes it; keep
   the number of subscribers to the cloud small, or raise the kernel socket buffers (§13).
5. **The user's RViz display pointed at the wrong topic** (`/camera/camera/depth/color/points`, the
   RealSense) — found by screenshot. Changed to `/astra/depth/points_local`.
6. **The Morty RViz uses `camera_camera_link` (wrist) as Fixed Frame.** With the arm driver down
   there is no path from the Astra frames to that frame and RViz drops the clouds silently
   (`Message Filter dropping message … queue is full` in its log). A separate RViz with fixed frame
   `base_link` showed the cloud fine. Also, the Astra's internal frames came from the shuttle driver
   as *dynamic* transforms with the shuttle's timestamps (+80…130 ms clock offset), which made the
   tf2 message filter wait; **fix: the driver now runs with `publish_tf:=false` and the laptop
   publishes the Astra's internal frames as static transforms** (values read once from the driver),
   and `astra_cloud_local` restamps clouds with laptop time. Nothing Astra-related depends on the
   shuttle clock any more.
7. A permanent **"Astra pointcloud"** display was added to the Morty RViz config
   (`src/ur3e_gripper_3d_moveit_config/config/moveit.rviz`, backup
   `moveit.rviz.bak_before_astra`): topic `/astra/depth/points_local`, Reliable, Points, 3 px,
   RGB8. It appears the next time the GUI starts RViz. **It still only shows the cloud while the arm
   driver runs (because of the wrist-camera fixed frame), or if Fixed Frame is set to `base_link`.**
8. At the end of the session the Morty GUI and robot launch were **closed** (launch ended 11:51,
   `move_group` segfaulted on shutdown, which is a pre-existing shutdown crash, not caused here).

### 5.4 Other things done on day 2
- `astra_restart_laptop.sh` written (restarts static TF + cloud builder, prints cloud rate).
- Documentation: `src/vla_experiments/README.md` updated with the auto-mode workflow;
  memory notes updated (`~/.claude/projects/-home-deep-morty-ws/memory/`).
- Advice given on extending the Astra's USB cable: an **active USB 2.0 extension (repeater) cable,
  5–10 m, USB-A female→male, ideally with external power**, plugged directly into the laptop (not the
  dock). With that, the shuttle and the network link become unnecessary: the driver is already built
  on the laptop (`~/astra_ws` there too) and only two topic names would change.

---

## 6. Hardware and network

### 6.1 Devices

| device | details |
|---|---|
| Laptop | Alienware 15 R3, Ubuntu 22.04, kernel 6.8, ROS 2 Humble, GTX 1080 8 GB, Wayland GNOME session (`DISPLAY=:1` for X apps), 116 GB free |
| UR3e | controller at 192.168.1.102 (built-in NIC `enp70s0`, laptop 192.168.1.101). Driver from `Universal_Robots_ROS2_Driver` in `src/`. `tcp_pose_broadcaster` active on real hardware. Pendant: External Control program must be playing |
| Wrist camera | Intel RealSense D435 (this week's unit serial 819612072994; another unit 816513022440 exists), USB 3 port bus 2, 640×480 colour, 848×480 depth, aligned depth on |
| Exterior camera | Orbbec Astra Pro, serial 18042630147, USB 2, plugged into the shuttle; depth 320×240@30 registered to colour 640×480@30 |
| Shuttle PC | `tory`, user `demo`, home `/localhome/demo`, Ubuntu 22.04, ROS 2 Humble, Docker 24, lightdm greeter, 8 cores/15 GB, **no GPU**; suspend now masked |
| Dock | USB-C dock: Realtek RTS5411 hub + ASIX AX88179A gigabit Ethernet (`enxc8a362a67c99`, cdc_ncm) |

### 6.2 Network addresses

| host / interface | address | purpose |
|---|---|---|
| laptop `enp70s0` | 192.168.1.101/24 (NM "Profile 2") | robot LAN |
| laptop `enxc8a362a67c99` | 192.168.2.1/24 + 168.192.1.101/24 (NM "shuttle-link", never-default) | link to the shuttle |
| laptop `wlp71s0` | DHCP on freeCampus | internet |
| shuttle `eth1` | 168.192.1.102/24 (profile "UR", typo) | link to the laptop |
| shuttle `wlan0` | 192.168.1.120/24 via RCR4-104 | shuttle's internet (works: apt/git OK) |
| shuttle `enp1s0` | 10.0.0.1/8, DOWN | unused |
| UR3e | 192.168.1.102 | robot |

Both clocks are NTP-synchronised (offset measured +0.08…+0.13 s shuttle ahead). ROS discovery
crosses the cable in both directions (multicast works both ways on the raw sockets; verified).

### 6.3 Fast DDS profiles

- `~/morty_ws/config/fastdds_udp_only.xml` — the stack's profile (UDP only, no SHM). Unchanged.
- `~/morty_ws/config/fastdds_shuttle_peer.xml` (a copy is packaged at
  `src/vla_experiments/config/fastdds_shuttle_peer.xml` and used by `vla_record.launch.py`):
  UDPv4 + SHM user transports, 16 MB send/receive buffer request (capped by the kernel until sysctl is
  raised), **no interfaceWhiteList**, `initialPeersList` = 168.192.1.102, 192.168.2.2 **and**
  239.255.0.1:7400 (multicast kept). Use it for anything on the laptop that must talk to the shuttle.
- `~/astra_ws/fastdds_link_peer.xml` on the shuttle (also on the laptop at the same path): built-in
  style UDP + SHM, initial peers 168.192.1.101 + multicast. Used by the driver and the relay.
  (`fastdds_link_only.xml` and `fastdds_link_peer_nobuf.xml` exist there from experiments; unused.)

---

## 7. The Astra camera pipeline in detail

```
[shuttle tory]                                              [laptop Alienware15R3]
Astra Pro (USB2)
  │ OpenNI2 depth 320x240@30 (registered) + libuvc colour 640x480@30
  ▼
astra_camera_node  (ros2_astra_camera, patched; camera_name=astra; publish_tf=false)
  topics /astra/depth/image_raw (16UC1), /astra/color/image_raw (rgb8), /astra/depth/camera_info, /astra/color/camera_info
  │  (local, SHM/UDP)
  ▼
astra_slow_relay.py  ──── /astra/depth/compressed_slow (PNG, 10 Hz, ~29 KB) ────────▶  astra_cloud_local
                     ──── /astra/color/compressed_slow (JPEG, 10 Hz, ~43 KB) ───────▶  (decode, XYZRGB)
                     ──── /astra/depth/image_slow, /astra/color/image_slow (raw fallbacks) ──▶ /astra/depth/points_local (~5 Hz, 66k pts, RELIABLE, laptop-stamped)
  camera_info (30 Hz, small) crosses directly ───────────────────────────────────────▶  recorder (episode_recorder) reads /astra/color/compressed_slow
                                                                                        RViz shows /astra/depth/points_local
TF (all static, published on the laptop by astra_tf.launch.py):
  base_link -> astra_link (Majd's pose)
  astra_link -> astra_depth_frame (identity), astra_link -> astra_color_frame (0.00018 0.02498 -0.00041, q 0.002784 0.001013 0.001103 0.999995)
  astra_depth_frame -> astra_depth_optical_frame, astra_color_frame -> astra_color_optical_frame (q -0.5 0.5 -0.5 0.5)
```

### 7.1 Shuttle side files (`/localhome/demo/astra_ws/`)
- `src/ros2_astra_camera/` — driver source with the two patches (§4.4) and the launch patch.
  Built with `colcon build --symlink-install` (install under `~/astra_ws/install`).
- `start_astra_on_shuttle.sh` — kills old driver/relay/colour publisher, removes stale
  `/dev/shm/fastrtps_*`, starts the driver:
  `ros2 launch astra_camera astra_pro.launch.xml camera_name:=astra enable_color:=true
  use_uvc_camera:=true color_width:=640 color_height:=480 color_fps:=30 enable_ir:=false
  enable_colored_point_cloud:=false depth_registration:=true enable_point_cloud:=false
  publish_tf:=false depth_width:=320 depth_height:=240 depth_fps:=30`, then the relay
  `python3 ~/astra_ws/astra_slow_relay.py --compressed-hz 10 --image-stride 2 --image-hz 4 --color-hz 5`,
  waits for "depth is started". Env: `rmw_fastrtps_cpp`, domain 0, `fastdds_link_peer.xml`.
  Logs: `~/astra_driver.log`, `~/astra_relay.log`. Run it from the laptop as
  `ssh demo@168.192.1.102 'bash ~/astra_ws/start_astra_on_shuttle.sh'` (**the quotes matter**,
  otherwise the laptop expands `~`).
- `astra_slow_relay.py` — the throttling/compressing relay (also kept at `~/astra_ws/` on the laptop).
- `astra_color_v4l2.py` — unused V4L2 colour fallback.
- `shuttle_setup.sh` — one-shot installer (ROS deps, clone, udev, build, bashrc env); it refuses to
  run on the laptop (hostname guard). It was accidentally run on the laptop twice before the guard
  existed; the laptop's `.bashrc` was restored from `~/.bashrc.bak_*`.
- `local_probe.py` — small test script.

### 7.2 Laptop side files
- `~/astra_ws/` — a second checkout/build of the driver (works; used to validate the build), the
  relay copy, the setup script, the DDS profiles.
- `src/vla_experiments/launch/astra_tf.launch.py` — static TF `base_link->astra_link` (defaults =
  Majd's pose, overridable with `x: y: z: roll: pitch: yaw:`) + the four internal static frames.
- `src/vla_experiments/vla_experiments/astra_cloud_local.py` — cloud builder (entry point
  `astra_cloud_local`; parameters `use_compressed` (default true), `max_range_m` 4.0,
  `color_max_age_s` 1.0; publishes xyz-only if no fresh colour).
- `src/vla_experiments/scripts/astra_restart_laptop.sh` — restart static TF + cloud builder.
- `src/vla_experiments/scripts/astra_view.sh` — the above plus RViz with `rviz/astra_vs_realsense.rviz`
  launched in a clean environment.

### 7.3 Data rates observed (healthy state)
camera_info 30 Hz; compressed depth ≈8–10 Hz; compressed colour ≈8–10 Hz; `points_local` 4–6 Hz with
one or two subscribers (drops to ~1 Hz at 100 % CPU with three RELIABLE subscribers).

### 7.4 What the Astra sees
From its mount it sees the whole yellow table with the objects, the floor around it and the robot
base/arm at the bottom of the image; depth valid on ~87 % of pixels, 0.63–2.4 m, table at ≈1.4 m.
Registration (depth↔colour) was verified visually: depth edges coincide with the table edges.

### 7.5 Astra specs vs RealSense (for decisions)
Astra Pro: structured light, depth 640×480@30 (we use 320×240), colour 1280×720 max (we use 640×480),
range 0.6–8 m, FOV 60°×49.5°, USB 2, indoor only. D435: active stereo, 0.28–3 m, 87°×58°, USB 3.
For VLA only RGB matters; the Astra's 0.6 m minimum range is irrelevant.

### 7.6 Calibration tooling
`ros2 run vla_experiments astra_calibrate [--frames N] [--save DIR] [--apply]` (§9.6). Status:
experimental, results rejected; candidates are evaluated by median nearest-neighbour distances to the
RealSense cloud and to the robot model. If revisited: (1) fix plane identification (the RealSense
often sees the floor, not the table), (2) prefer robot-model registration, (3) validate with the
tabletop height (0.622 m along vertical) and by projecting the robot model into the Astra image.

### 7.7 Current Astra pose and how to adjust it
`ros2 launch vla_experiments astra_tf.launch.py` (defaults) or with overrides, e.g.
`... z:=0.10 pitch:=-0.55`. Rules Majd used: yaw turns about blue (z), pitch about the *already
yawed* green axis, roll about red; a distance d along the turned red axis is `x:=d*0.7071,
y:=-d*0.7071`; along the turned green axis `x:=d*0.7071, y:=d*0.7071`; along blue plain `z:=d`.
When changed permanently: update the `default_value`s in the launch file and `astra_tf_xyz/rpy` in
`config/recorder.yaml` (the recorder writes them into every episode's `meta.json`).

---

## 8. Fast DDS / ROS 2 networking lessons

1. `initialPeersList` replaces the default multicast peer — always add `239.255.0.1:7400`.
2. An `interfaceWhiteList` on a hot-pluggable interface makes the process deaf when the interface
   disappears. Don't whitelist the dock NIC.
3. Kernel UDP buffers are 212992 B on both machines. Any DDS message above ~150 KB is unreliable
   over UDP (loopback included) until `net.core.rmem_max/wmem_max` (and `rmem_default`) are raised
   (needs sudo on each machine; recommended 64 MB). Until then: compress, subsample, throttle.
4. RELIABLE large messages to several UDP-only subscribers cause retransmission storms that pin the
   publisher at 100 % CPU (seen with 1 MB clouds and three RViz windows).
5. `ros2 topic list/hz/echo` use a daemon whose cache goes stale; `ros2 daemon stop && ros2 daemon
   start`, or `--no-daemon`. `tf2_echo` was unreliable for static frames in this environment;
   use a small rclpy `Buffer` probe with a longer window.
6. Mixed timestamps across machines break tf2 message filters in RViz even with sub-second offsets
   when the fixed frame needs both machines' transforms. Keep cross-machine transforms static and
   restamp data on the receiving side.
7. The laptop's stack has ~40–50 participants; discovery of a new participant normally completes in
   1–2 s, but static (transient-local) samples occasionally took much longer or never arrived for
   some probes. If something "does not exist" in tf2, check with a raw `/tf_static` subscriber
   before assuming the publisher is gone.
8. Local (same-host) delivery between a SHM-capable participant and the UDP-only stack works over UDP
   loopback, subject to lesson 3.

---

## 9. The `vla_experiments` package

Location: `~/morty_ws/src/vla_experiments/` (ament_python, maintainer Majd Hanna, MIT).
Built with `colcon build --symlink-install --packages-select vla_experiments` (merged install).

### 9.1 Layout
```
vla_experiments/
  package.xml, setup.py, setup.cfg, resource/vla_experiments, README.md
  vla_experiments/
    __init__.py
    raw_dataset.py        raw episode format, Episode reader, action derivation, `vla_dataset_info`
    episode_recorder.py   the recorder node (manual + auto mode)
    episode_cli.py        `vla_episode` CLI
    convert_to_lerobot.py LeRobot v2.1 export (openpi)
    convert_to_rlds.py    copies the TFDS builder next to the raw data (OpenVLA)
    astra_cloud_local.py  laptop cloud builder (§7)
    astra_calibrate.py    experimental Astra<->robot registration (§7.6)
  rlds_builder/morty_ur3e/__init__.py, morty_ur3e_dataset_builder.py
  launch/astra_tf.launch.py, launch/vla_record.launch.py
  config/recorder.yaml, config/fastdds_shuttle_peer.xml
  rviz/astra_vs_realsense.rviz
  scripts/astra_restart_laptop.sh, scripts/astra_view.sh
```
Entry points: `episode_recorder`, `vla_episode`, `vla_dataset_info`, `convert_to_lerobot`,
`convert_to_rlds`, `astra_cloud_local`, `astra_calibrate`.

### 9.2 Raw episode format (`morty_vla_raw_v1`)
```
~/morty_ws/vla_experiments/datasets/<dataset_name>/episode_000012/
  meta.json      task, success, intent, cycle_done, note, mmhri_trial_hint, fps, num_steps,
                 duration_s, dropped_ticks, drop_reasons, started_unix, robot_type, arm_joints,
                 gripper_joint, gripper_open_pos, frames{base,ee,tool}, cameras{exterior,wrist}
                 (CameraInfo: width,height,K,D,distortion_model,frame_id), image_sources,
                 astra_static_tf_base_to_astra_link{xyz,rpy}, step_keys
  steps.npz      arrays, one row per step (10 Hz):
                 t, stamp_ros, joint_pos(6), joint_vel(6), joint_cmd(6, NaN if no controller_state),
                 tcp_pos(3), tcp_quat(4 xyzw), tool0_pos(3), tool0_quat(4), gripper(0..1, 1=open),
                 gripper_cmd(0/1 or NaN), grasp_detected(-1/0/1), img_age_exterior, img_age_wrist, joint_age
  images/exterior/000000.jpg ...   Astra colour 640x480, JPEG q95
  images/wrist/000000.jpg ...      RealSense colour 640x480
```
Arm joint order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3. Gripper joint
`gripper_close_open_joint`, normalised by `gripper_open_pos` 0.23 rad. EE frame `gripper_tcp`,
tool frame `tool0`, base `base_link`, all from TF at the tick.

Actions are **not stored**; `raw_dataset.derive_actions()` computes at conversion time:
`joint_abs_next` (next-step absolute joints, or the controller reference if recorded, + gripper
target), `joint_delta`, `eef_delta` (Δxyz, Δrpy in base frame, + gripper). Gripper target is binary
from the next-step state or the explicit `gripper_cmd`. State encodings from
`raw_dataset.state_vectors()`: `joint7` (openpi UR5 style), `pos_euler8` (OpenVLA POS_EULER),
`joint8` (OpenVLA JOINT).

### 9.3 Recorder behaviour (`episode_recorder`)
- Timer at `fps` (10 Hz). A tick is **dropped, not written**, if joints are older than
  `max_joint_age_s` (0.2), a required camera image older than `max_image_age_s` (0.5), or the EE TF is
  unavailable; drop counts/reasons go into `meta.json`. This is why a dead arm driver yields empty
  episodes ("no frames captured; discarding").
- Sources (all parameters, `config/recorder.yaml`): exterior compressed
  `/astra/color/compressed_slow` (raw fallback `/astra/color/image_raw` if the compressed topic
  parameter is empty), wrist compressed `/camera/camera/color/image_raw/compressed`, infos
  `/astra/color/camera_info`, `/camera/camera/color/camera_info`, `/joint_states`,
  `/scaled_joint_trajectory_controller/controller_state`, `/tcp_pose_broadcaster/pose`,
  `/io_and_status_controller/io_states`, `/vla_recorder/gripper_cmd`, `/speech/accepted_transcript`,
  `/robot/intent`, `/grasp_executor_lin_llm/cycle_done`, commands on `/vla_recorder/cmd`.
- Status latched on `/vla_recorder/status` (JSON: recording, provisional, auto_mode, episode, task,
  intent, frames, dropped, sources{exterior,wrist,joints,controller_ref,io,tf_ee}, dataset_dir,
  episodes_on_disk). `ros2 run vla_experiments vla_episode status` prints it.
- Manual commands: `start <task>`, `stop`, `stop fail`, `discard`, `task <text>`, `gripper open|close`.
- Auto mode: described in §5.2. Parameters `auto_mode`, `intent_topic`, `motion_intents`,
  `cycle_done_topic`, `intent_wait_s` 25, `max_episode_s` 240, `mmhri_dir`.
- On Ctrl-C while recording the episode is saved with success=false, note "recorder shut down".

### 9.4 Launch
`ros2 launch vla_experiments vla_record.launch.py [auto_mode:=true] [dataset_name:=morty_ur3e]
[with_astra_tf:=true] [require_exterior:=true] [require_wrist:=true] [config:=...]`.
It also starts `astra_tf.launch.py` (with_astra_tf) and sets the DDS env for its children.

### 9.5 Converters
- `convert_to_lerobot --raw <dir> --repo-id <user>/morty_ur3e [--image-size 256] [--action-space
  joint_abs|joint_delta|eef_delta] [--state-space joint7|pos_euler8] [--fps 10] [--push]` — needs the
  `lerobot` package at the commit openpi pins (`0cf8648`, `CODEBASE_VERSION v2.1`,
  `LeRobotDataset.create/add_frame/save_episode`, no `consolidate()`), Python ≥ 3.11 → run it inside
  openpi's environment on the training machine. Features: `image`, `wrist_image` (dtype image →
  video), `state`, `actions`, `task`.
- `convert_to_rlds --raw <dir> --out ~/rlds/morty_ur3e` copies `rlds_builder/morty_ur3e/` (a
  `tfds` `GeneratorBasedBuilder` reading the raw folders directly; image size via
  `MORTY_VLA_IMG`, raw path via `MORTY_VLA_RAW`) then `cd ~/rlds/morty_ur3e && MORTY_VLA_RAW=<dir>
  tfds build --overwrite` in the `rlds_dataset_builder` conda env. Fields: `observation/image`,
  `observation/wrist_image`, `observation/state` (POS_EULER 8), `observation/joint_state` (8),
  `action` (EEF_POS 7), `language_instruction`, `language_embedding` (USE if available, else zeros).
  The builder docstring contains the exact `configs.py` entry for OpenVLA.

### 9.6 Tools
- `vla_dataset_info <dataset_dir> [--all]` — per-episode summary (steps, duration, success, drops,
  motion, task).
- `astra_calibrate` — §7.6.

### 9.7 Tests performed
- Day 1 dry run with a synthetic robot + live RealSense: 53 steps at 10.0 Hz, jitter 0.2 ms.
- Day 2 full run with the real robot + Astra + RealSense in auto mode: §5.2. Rotation math checked
  against scipy (2e-15).

---

## 10. Current state of the system (end of session, 2026-09-03 ~12:15)

- **Shuttle (`tory`)**: awake, suspend masked, link up. Astra driver and relay running (started by
  `start_astra_on_shuttle.sh`, `publish_tf:=false`). Streams reach the laptop.
- **Laptop**: `astra_tf.launch.py` (static base→astra + internal frames) and `astra_cloud_local`
  running as background processes started by `astra_restart_laptop.sh` (logs in the session
  scratchpad `/tmp/claude-1000/-home-deep-morty-ws/<session>/scratchpad/`). They do not survive a
  reboot; rerun the script.
- **Robot stack**: **stopped**. The Morty GUI is closed; its driver launch ended at 11:51 (move_group
  segfaulted on shutdown). Two orphaned `rviz2` windows may still be open. No `/joint_states` from the
  arm, no arm TF. To record, start the GUI and the driver (pendant External Control → Play first).
- **Morty RViz config** now contains the "Astra pointcloud" display; it shows the cloud only when
  the arm driver runs (fixed frame `camera_camera_link`) or if the fixed frame is set to `base_link`.
- **VLA recorder**: not running. Package built and tested. Dataset folder
  `~/morty_ws/vla_experiments/datasets/` is empty (test sets deleted).
- **Astra pose**: saved defaults (-0.51, 0.51, 0.065 / 0, -0.576, -0.77), approximate.
- **Kernel socket buffers**: default (small) on both machines.
- **Shuttle eth1 typo**: still 168.192.1.102; laptop carries a matching second address.
- **Memory notes** for the AI assistant in `~/.claude/projects/-home-deep-morty-ws/memory/`:
  `shuttle-pc-and-astra-setup.md`, `astra-over-shuttle-link.md`, `nmcli-steals-robot-profile.md`,
  `pkill-self-match-trap.md` (+ older ones).

---

## 11. Runbooks

### 11.1 Full start (cold)
1. Robot: power on, pendant External Control → Play. Laptop: `ros2 run morty_gui morty_gui` (from a
   normal terminal with `source ~/morty_ws/install/setup.bash`), start the driver from the GUI. Check
   `ros2 node list | grep controller_manager` and that `/joint_states` has 6 arm joints.
2. Shuttle camera: `ssh demo@168.192.1.102 'bash ~/astra_ws/start_astra_on_shuttle.sh'`
   (if it prints "driver failed" once after a reboot, run it again).
3. Laptop Astra helpers: `bash ~/morty_ws/src/vla_experiments/scripts/astra_restart_laptop.sh`
   (prints the cloud rate; "none yet" means the shuttle side is not streaming).
4. Optional view: `bash ~/morty_ws/src/vla_experiments/scripts/astra_view.sh` (RViz, fixed frame
   base_link, both clouds) — or in the Morty RViz enable "Astra pointcloud".
5. VLA recorder: `source ~/morty_ws/install/setup.bash && ros2 launch vla_experiments
   vla_record.launch.py auto_mode:=true`, then `ros2 run vla_experiments vla_episode status` — all
   `sources` should be true (controller_ref may stay false; that only removes the commanded-joint
   field).

### 11.2 After the link dropped / shuttle rebooted / "no Astra cloud"
```
cat /sys/class/net/enxc8a362a67c99/carrier          # 1 = cable link ok
ping -c1 168.192.1.102                              # shuttle answers?
ssh demo@168.192.1.102 'bash ~/astra_ws/start_astra_on_shuttle.sh'
bash ~/morty_ws/src/vla_experiments/scripts/astra_restart_laptop.sh
```
Then check, in this order: the RViz display topic is `/astra/depth/points_local`; the arm driver runs
(or fixed frame = base_link); the RViz view is zoomed out (Distance ≈ 4, Pitch ≈ 1).

### 11.3 Diagnose quickly (laptop, any terminal with ROS sourced)
```
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/morty_ws/config/fastdds_shuttle_peer.xml
ros2 daemon stop; ros2 daemon start
ros2 topic hz /astra/depth/camera_info           # 30 Hz  -> driver alive, link ok
ros2 topic hz /astra/color/compressed_slow       # ~10 Hz -> relay alive
ros2 topic hz /astra/depth/points_local          # ~5 Hz  -> cloud builder alive
ros2 topic echo --once --no-daemon /joint_states --field name   # 6 arm joints -> arm driver alive
```

### 11.4 Record a session (auto mode)
Keep the GUI as usual; run the recorder (11.1 step 5) in a second terminal; do voice trials. Motion
trials become `episode_NNNNNN` under `~/morty_ws/vla_experiments/datasets/morty_ur3e/`; questions
are discarded; each MMHRI trial keeps its own folder. Inspect with
`ros2 run vla_experiments vla_dataset_info ~/morty_ws/vla_experiments/datasets/morty_ur3e`.
Manual mode: launch without `auto_mode`, then `vla_episode start "..."` / `stop`.

### 11.5 Convert
See §9.5. Copy the raw dataset folder to the GPU machine (rsync); LeRobot conversion needs openpi's
environment; RLDS needs `rlds_dataset_builder`'s environment.

### 11.6 Change the Astra pose
`ros2 launch vla_experiments astra_tf.launch.py x:=… y:=… z:=… roll:=… pitch:=… yaw:=…` (stop the
running static TF first: `pkill -f astra_static_t[f]`), judge in RViz with both clouds, then write
the numbers into the launch defaults and `config/recorder.yaml`.

### 11.7 Automatic start at boot (added 2026-09-03)
- Shuttle: user crontab of `demo` has `@reboot sleep 25 && bash ~/astra_ws/start_astra_on_shuttle.sh`
  (log `~/astra_ws/logs/autostart.log`). The start script retries once by itself when the first
  driver start dies (the libuvc/uvcvideo hand-over after a reboot). Because suspend is masked, a
  reboot is the only event that stops the camera; after a reboot it comes back on its own ~1 min
  after power-on.
- Laptop: user crontab of `deep` has `@reboot sleep 40 && bash
  ~/morty_ws/src/vla_experiments/scripts/astra_restart_laptop.sh` (log
  `~/morty_ws/vla_experiments/logs/autostart.log`): static Astra TF + cloud builder.
- `crontab -l` on either machine shows/edits the jobs. Manual restarts still work exactly as before.

---

## 12. What the VLA models need, and the training plan

| | openpi (LeRobot v2.1) | OpenVLA (RLDS) |
|---|---|---|
| Per-frame keys | `image`, `wrist_image`, `state`, `actions`, `task` | `observation/image`, `observation/wrist_image`, `observation/state`, `action`, `language_instruction` |
| State | UR5 example: 6 joints + gripper (7) | POS_EULER: xyz rpy pad gripper (8) or JOINT (8) |
| Action | 7-D joint targets (absolute; `DeltaActions` in the data config) | EEF delta xyz + delta rpy + gripper (7), `ActionEncoding.EEF_POS` |
| Rate | 10 Hz in the LIBERO/UR5 examples | any, 5–10 Hz typical |
| Compute | LoRA > 22.5 GB GPU, full > 70 GB; Python ≥ 3.11 | LoRA ≈ 72 GB at batch 16 on A100 (smaller batch works) |
| Register | new `DataConfig`/`TrainConfig` in `src/openpi/training/config.py`, `compute_norm_stats.py` | entries in `configs.py`, `transforms.py`, `mixtures.py` |

Depth is not used by either. The policy bakes the camera pose into its weights: **do not move the
Astra between recording and deployment**; keep resolution/exposure/lighting constant. Collect 50–150
successful demonstrations per instruction, vary object positions and add distractors.

Remaining steps to a running policy: (1) real recordings (hours of robot time), (2) conversion,
(3) fine-tuning on a large GPU (neither laptop nor shuttle can do it — university cluster or cloud),
(4) deployment: a policy server on the GPU machine and a client on the laptop that turns 10 Hz actions
into `scaled_joint_trajectory_controller` goals (not written yet; about half a day).

---

## 13. Known issues, open items, next steps (prioritised)

1. **Verify the Astra pose** with both clouds overlaid (fixed frame base_link, arm running). If off,
   adjust (§11.6). Consider a proper hand-eye calibration (checkerboard/ArUco seen by both cameras,
   or robot-model registration with the RealSense pointed at the table).
2. **Raise UDP socket buffers on both machines** (needs sudo):
   `sudo sysctl -w net.core.rmem_max=67108864 net.core.wmem_max=67108864 net.core.rmem_default=8388608
   net.core.wmem_default=8388608` and persist in `/etc/sysctl.d/90-ros-dds.conf`. This removes most
   of the "big message" fragility (RealSense raw cloud, Astra cloud to several RViz windows).
3. **Fix the shuttle's eth1 address** to 192.168.2.2/24 (profile "UR" in its NetworkManager) and
   remove 168.192.1.101 from the laptop's `shuttle-link` profile and from the DDS profiles.
4. **Better: move the Astra to the laptop** with an active USB 2.0 extension cable (5–10 m, powered).
   Then the shuttle, the cable, the relay and the DDS peer profiles are unnecessary: run the driver on
   the laptop (`~/astra_ws` is built there) with the same arguments, and point `astra_cloud_local` /
   the recorder at the raw topics (`use_compressed:=false`, `exterior_compressed_topic:=""`).
5. `controller_state` published nothing during tests → `joint_cmd` is NaN and actions fall back to
   next-step joints. Check which trajectory controller is active during real trials
   (`ros2 control list_controllers`); if `scaled_joint_trajectory_controller` is active it should
   publish `~/controller_state` at 100 Hz.
6. The `morty_gui` driver launch's `move_group` segfaults on shutdown and `ur_ros2_control_node` dies
   with SIGPIPE when the pendant program stops — pre-existing, not addressed.
7. Auto mode assumes one command per trial; a second accepted transcript while an episode is
   confirmed is ignored (only a still-provisional episode takes the newer text).
8. The cloud builder and static TF are plain background processes; consider a systemd user service
   or adding them to the GUI's camera launch.
9. Colour on the Astra depends on the "crash once, then works" libuvc behaviour after each shuttle
   reboot; the start script handles it, but a cleaner fix would be building Orbbec's libuvc fork or
   using the V4L2 fallback publisher.
10. Deployment client for the trained policy (§12) does not exist yet.

---

## 14. Pitfalls and traps encountered

- **`pkill -f <pattern>` in the same shell command as the launch string kills the shell itself**
  (exit 144 locally, 255 over ssh) and leaves orphan processes. The `[a]bc` bracket trick only
  protects the pkill argument; any other literal occurrence (a log file name, the launch line) still
  matches. Put kill and launch in separate commands, or hide the literal (`EXE=$(printf
  'astra_cloud_loca%s' l)`). Memory note `pkill-self-match-trap`.
- Do not `pkill -x rviz2` — it would kill the GUI's RViz too; identify by parent PID/age.
- `ssh host bash ~/x.sh` expands `~` on the laptop; quote the remote command.
- RViz/gnome-screenshot from the VS Code terminal crash on snap libraries → `env -i`.
- `nmcli device connect <new NIC>` steals "Profile 2" from the robot NIC.
- The shuttle at its login screen suspends after 20 min (fixed by masking sleep targets).
- `ros2` CLI daemon cache; `--no-daemon`.
- `initialPeersList` without the multicast entry; interface whitelist on a hot-plug NIC.
- Message sizes vs 212 KB kernel buffers.
- Another Claude session ran in parallel in VS Code on this laptop ("Starting grasp on the tape
  dispenser…") and spawned throwaway RViz windows; unexplained RViz windows may come from it.
- Two Astra frames named `camera_link` lingered in RViz's TF list from the driver's first launch —
  stale display memory, harmless.
- `tf2_echo` and short-lived tf2 probes sometimes reported "frame does not exist" for frames that a
  raw `/tf_static` subscriber received within a second; use longer windows or raw subscribers.

---

## 15. File index

### Laptop, workspace `~/morty_ws`
- `VLA_ASTRA.md` — this document.
- `config/fastdds_shuttle_peer.xml` — DDS profile for laptop processes that talk to the shuttle.
- `src/vla_experiments/**` — the new package (§9), including `README.md`, `launch/astra_tf.launch.py`,
  `launch/vla_record.launch.py`, `config/recorder.yaml`, `config/fastdds_shuttle_peer.xml`,
  `rviz/astra_vs_realsense.rviz`, `scripts/astra_restart_laptop.sh`, `scripts/astra_view.sh`,
  `rlds_builder/morty_ur3e/`.
- `src/ur3e_gripper_3d_moveit_config/config/moveit.rviz` — Astra pointcloud display added;
  backup `moveit.rviz.bak_before_astra`.
- `vla_experiments/datasets/` — VLA data folder (empty now).
- `~/astra_ws/` — driver checkout + build on the laptop, `shuttle_setup.sh`, `start_astra_on_shuttle.sh`
  (master copy), `astra_slow_relay.py` (master copy), `astra_color_v4l2.py`, `fastdds_link_peer.xml`,
  `fastdds_link_only.xml`, `fastdds_link_peer_nobuf.xml`.
- `~/.ssh/id_ed25519(.pub)` — key generated this session, installed on the shuttle.
- `~/.bashrc` — unchanged in the end (a block accidentally appended by `shuttle_setup.sh` was removed;
  backup `~/.bashrc.bak_<time>`).
- `~/.claude/projects/-home-deep-morty-ws/memory/*.md` — assistant memory notes.

### Shuttle, `/localhome/demo`
- `astra_ws/src/ros2_astra_camera/` (patched), `astra_ws/install/`, `astra_ws/start_astra_on_shuttle.sh`,
  `astra_ws/astra_slow_relay.py`, `astra_ws/astra_color_v4l2.py`, `astra_ws/fastdds_link_peer.xml` (+
  the two experimental profiles), `astra_ws/local_probe.py`, `shuttle_setup.sh`, logs
  `astra_driver.log`, `astra_relay.log`, `astra_variantA.log`, `astra_gdb.log`.
- `/etc/systemd/system/{sleep,suspend,hibernate,hybrid-sleep}.target -> /dev/null` (masked).
- `/usr/share/keyrings/ros-archive-keyring.gpg` refreshed; apt packages installed (§4.4).
- GNOME power settings for user demo set to never sleep.

---

## 16. Command cheat-sheet

```bash
# shuttle
ssh demo@168.192.1.102                                            # key login
ssh demo@168.192.1.102 'bash ~/astra_ws/start_astra_on_shuttle.sh' # start camera (run twice if it dies once)
ssh demo@168.192.1.102 'pgrep -fc "[a]stra_camera_node"; tail -3 ~/astra_driver.log'

# laptop helpers
bash ~/morty_ws/src/vla_experiments/scripts/astra_restart_laptop.sh
bash ~/morty_ws/src/vla_experiments/scripts/astra_view.sh          # + RViz (clean env)
ros2 launch vla_experiments astra_tf.launch.py x:=-0.51 y:=0.51 z:=0.065 roll:=0 pitch:=-0.576 yaw:=-0.77

# recorder
source ~/morty_ws/install/setup.bash
ros2 launch vla_experiments vla_record.launch.py auto_mode:=true
ros2 run vla_experiments vla_episode status
ros2 run vla_experiments vla_episode start "pick up the vitamin c"   # manual mode
ros2 run vla_experiments vla_episode stop
ros2 run vla_experiments vla_dataset_info ~/morty_ws/vla_experiments/datasets/morty_ur3e

# export (on the GPU machine / proper envs)
python -m vla_experiments.convert_to_lerobot --raw <raw> --repo-id <user>/morty_ur3e --image-size 256
ros2 run vla_experiments convert_to_rlds --raw <raw> --out ~/rlds/morty_ur3e
cd ~/rlds/morty_ur3e && MORTY_VLA_RAW=<raw> tfds build --overwrite

# network
cat /sys/class/net/enxc8a362a67c99/carrier; ping -c1 168.192.1.102; ping -c1 192.168.1.102
nmcli con show shuttle-link | grep ipv4.addresses
nmcli con up "Profile 2" ifname enp70s0                            # if the robot link ever gets stolen

# build
cd ~/morty_ws && colcon build --symlink-install --packages-select vla_experiments
```

End of document.
