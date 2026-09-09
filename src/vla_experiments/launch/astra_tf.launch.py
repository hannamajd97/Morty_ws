"""Static transform for the fixed Orbbec Astra: base_link -> astra_link.

Pose set manually by Majd in RViz on 2026-09-02 (Astra cloud overlaid on the RealSense cloud):
  xyz  -0.51  0.51  0.065     (0.72 m straight up: in this base frame vertical is (-0.707, 0.707, 0);
                               0.065 m sideways along base z)
  rpy   0.0  -0.5760  -0.77   (yaw ~ -44 deg like the table, pitch -33 deg => looking ~57 deg down)
Override from the command line, e.g.
  ros2 launch vla_experiments astra_tf.launch.py z:=0.10 pitch:=-0.55
The Astra driver runs with camera_name:=astra and camera_link_frame_id:=astra_link (see
~/astra_ws/start_astra_on_shuttle.sh on the shuttle), so its frames hang below `astra_link`.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("x", default_value="-0.51"),
        DeclareLaunchArgument("y", default_value="0.51"),
        DeclareLaunchArgument("z", default_value="0.065"),
        DeclareLaunchArgument("roll", default_value="0.0"),
        DeclareLaunchArgument("pitch", default_value="-0.576"),
        DeclareLaunchArgument("yaw", default_value="-0.77"),
        DeclareLaunchArgument("parent_frame", default_value="base_link"),
        DeclareLaunchArgument("child_frame", default_value="astra_link"),
    ]
    tf = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="astra_static_tf",
        arguments=[
            "--x", LaunchConfiguration("x"),
            "--y", LaunchConfiguration("y"),
            "--z", LaunchConfiguration("z"),
            "--roll", LaunchConfiguration("roll"),
            "--pitch", LaunchConfiguration("pitch"),
            "--yaw", LaunchConfiguration("yaw"),
            "--frame-id", LaunchConfiguration("parent_frame"),
            "--child-frame-id", LaunchConfiguration("child_frame"),
        ],
        output="screen",
    )
    # The Astra driver runs with publish_tf:=false; its internal frames are published here as
    # STATIC transforms (values read from the driver once, 2026-09-03) so nothing depends on the
    # shuttle's clock. Optical frames use the usual REP-103 camera->optical rotation.
    def static(name, parent, child, xyz, quat):
        return Node(package="tf2_ros", executable="static_transform_publisher", name=name, output="log",
                    arguments=["--x", str(xyz[0]), "--y", str(xyz[1]), "--z", str(xyz[2]),
                               "--qx", str(quat[0]), "--qy", str(quat[1]), "--qz", str(quat[2]), "--qw", str(quat[3]),
                               "--frame-id", parent, "--child-frame-id", child])
    optical = (-0.5, 0.5, -0.5, 0.5)
    internal = [
        static("astra_depth_frame_tf", "astra_link", "astra_depth_frame", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)),
        static("astra_color_frame_tf", "astra_link", "astra_color_frame", (0.00018, 0.02498, -0.00041), (0.002784, 0.001013, 0.001103, 0.999995)),
        static("astra_depth_optical_tf", "astra_depth_frame", "astra_depth_optical_frame", (0.0, 0.0, 0.0), optical),
        static("astra_color_optical_tf", "astra_color_frame", "astra_color_optical_frame", (0.0, 0.0, 0.0), optical),
    ]
    return LaunchDescription(args + [tf] + internal)
