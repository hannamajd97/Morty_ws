# gripper_3d

Description and I/O of Morty's 3D-printed two-finger gripper, with the
wrist-mounted RealSense.

## What is in here

| path | content |
| --- | --- |
| `urdf/gripper_3d.xacro` | the gripper macro: base, camera mount, two revolute fingers, `gripper_tcp` 12 cm in front of the flange; collision geometry uses the convex hulls, visuals the full meshes |
| `meshes/` | printed parts (`zusammen_greifer_3_c*.stl`, `finger_*.stl`), RealSense model, `old_*` earlier revisions |
| `meshes/collision/` | convex hulls (`gripper_base_hull`, `camera_mount_hull`, `finger_*_hull`) — a few thousand triangles instead of 157k, which halves MoveIt's collision-check time |
| `scripts/gripper_command_adapter` | exposes the `GripperCommand` action MoveIt expects (`/gripper_controller/gripper_cmd`) and drives the UR tool digital outputs with timed pulses (`pulse_hold_s_open` / `pulse_hold_s_close`) |
| `scripts/gripper_io_bridge` | open/close `Trigger` services on the tool outputs and mirroring of the finger joints for RViz |
| `scripts/robot_description_publisher` | publishes the combined arm + gripper description |
| `launch/` | `ur3e_gripper_3d.launch.py` (RViz view), `run_gripper_u3re_camera.launch.py` (view + optional real camera), `realsense_d345_hardware.launch.py` (camera driver only), `gripper_io_hardware.launch.py` (I/O bridge only) |
| `config/`, `rviz/` | camera and RViz settings |

In the running system the `ur3e_gripper_3d_moveit_config` launch starts the
command adapter and the camera; the launch files here are for viewing the
model on its own.

## The gripper's own intelligence

The gripper carries force-sensing strips on the fingers and its own logic.
It reports on two tool digital inputs: TI0 high = holding an object, TI1
high = slip detected, and on slip it squeezes harder by itself. The executor
only reacts (slows the arm, narrates); see the commander package.

## View the model

```bash
ros2 launch gripper_3d ur3e_gripper_3d.launch.py
```

Hinge positions, axis and joint limits are the `*_hinge_xyz`,
`hinge_axis_xyz` and limit parameters at the top of the xacro.
