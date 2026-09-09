# gripper_3d

UR3e description with a mesh-based gripper:
- 1 mesh for the gripper body/base
- 1 mesh for the left finger
- 1 mesh for the right finger

The fingers are connected with **revolute joints** so you can visualize open/close in RViz using `joint_state_publisher_gui`.

## Put your meshes here

Place your files in:
- `gripper_3d/meshes/zusammen_greifer_3_neu_without_camera_fingers.stl`
- `gripper_3d/meshes/finger_left.stl`
- `gripper_3d/meshes/finger_right.stl`

If your STL units are **mm**, keep `mesh_scale="0.001 0.001 0.001"`. If they are meters, use `1 1 1`.

## View in RViz

```bash
cd ~/ur_ros2_ws
colcon build --packages-select gripper_3d
source install/setup.bash
ros2 launch gripper_3d ur3e_gripper_3d.launch.py
```

## One-command demo: UR3e + gripper (+ optional real camera)

Starts the RViz display (with `camera_enabled:=true`). Optionally starts the RealSense driver if real hardware is present.

```bash
ros2 launch gripper_3d run_gripper_u3re_camera.launch.py
```

Enable the real camera:

```bash
ros2 launch gripper_3d run_gripper_u3re_camera.launch.py use_real_camera:=true
```

## Real hardware: RealSense driver only

`gripper_3d` models the camera as link `camera_camera_link` rigidly attached to `tool0` (see `gripper_3d/urdf/gripper_3d.xacro`).

To run only the RealSense driver (no robot/gripper display):

```bash
ros2 launch gripper_3d realsense_d345_hardware.launch.py
```

Notes:
- `realsense_d345_hardware.launch.py` does not publish any extra static TF; the `tool0 -> camera_camera_link` transform should come from your robot's `robot_state_publisher` using the `gripper_3d` xacro.
- The RealSense driver usually applies a TF prefix (e.g. `camera_`), so `base_frame_id` is set to `camera_link` and the resulting published base frame becomes `camera_camera_link` (don’t set `base_frame_id` to `camera_camera_link` or you’ll get `camera_camera_camera_link`).

## Real hardware: gripper I/O -> RViz finger joints

`ur_pose_go` toggles UR tool digital outputs to open/close the gripper. `gripper_3d` can mirror that to the finger joints so RViz shows open/close on the model.

Start the bridge node:

```bash
ros2 launch gripper_3d gripper_io_hardware.launch.py
```

Then command the gripper:

```bash
ros2 service call /gripper_io_bridge/open std_srvs/srv/Trigger {}
ros2 service call /gripper_io_bridge/close std_srvs/srv/Trigger {}
```

## Tune hinge

Edit these params in `gripper_3d/urdf/gripper_3d.xacro`:
- `left_hinge_xyz`, `right_hinge_xyz` (hinge locations)
- `hinge_axis_xyz` (hinge axis)
- joint limits (`lower`, `upper`)
