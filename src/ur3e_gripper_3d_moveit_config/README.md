# ur3e_gripper_3d_moveit_config

MoveIt 2 configuration for the UR3e with the `gripper_3d` gripper, and the
launch files that bring up the real robot.

## Launch files

| launch | use |
| --- | --- |
| `ur3e_gripper_3d_moveit_real_camera.launch.py` | **the one the GUI starts**: UR driver (robot at 192.168.1.102), controllers, MoveIt `move_group`, RealSense, gripper command adapter, point-cloud downsampler, planning-scene environment |
| `ur3e_gripper_3d_moveit_real.launch.py` | same without the camera |
| `ur3e_gripper_3d_moveit.launch.py` | generic entry with `use_fake_hardware` / `use_ur_driver` / `robot_ip` arguments; the GUI's offline mode uses `use_fake_hardware:=true use_ur_driver:=false` |
| `move_group.launch.py`, `moveit_rviz.launch.py`, `setup_assistant.launch.py` | the usual MoveIt pieces |

## Configuration

| file | content |
| --- | --- |
| `ur3e_with_gripper_3d.srdf` | planning groups `ur3e_manipulator` (base_link → gripper_tcp) and `gripper`, named poses, disabled collision pairs |
| `kinematics.yaml` | KDL solver (pick_ik was tried and reverted; see the comments there) |
| `ompl_planning.yaml`, `pilz_cartesian_limits.yaml` | planners; Pilz LIN is what the executor probes with |
| `joint_limits.yaml`, `ur3e_joint_limits.yaml` | velocity/acceleration limits |
| `ros2_controllers.yaml`, `moveit_controllers.yaml`, `ur_driver_moveit_controllers.yaml` | `scaled_joint_trajectory_controller`, gripper controller mapping |
| `ur3e_calibration.yaml` | the robot's factory calibration (link lengths used by the driver) |
| `environment.yaml` | the static scene: table, mount surface, wall, flower box |
| `home_positions.yaml` | joint values of the named poses |
| `moveit.rviz` | RViz layout incl. the executed-pose display |

## Helper scripts (`scripts/`)

| script | role |
| --- | --- |
| `planning_scene_environment_publisher` | publishes the static collision objects from `environment.yaml` into the planning scene |
| `environment_markers_publisher` | the same objects as RViz markers |
| `pointcloud_downsampler` | downsamples the RealSense cloud for the viewers |

## Notes

- The robot base is mounted sideways: base `+z` is horizontal. Anything that
  needs "up" must use the table-frame up vector, never base z.
- Stopping the pendant program deactivates `scaled_joint_trajectory_controller`;
  the GUI re-activates it.
