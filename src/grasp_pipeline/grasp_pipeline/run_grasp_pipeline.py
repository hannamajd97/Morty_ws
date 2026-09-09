#!/usr/bin/env python3
"""
main.py
=======

Entry point for running the grasp selection node. This script
initialises the ROS2 runtime, creates an instance of
``grasp_pipeline.grasp_selection_node.YoloLiveSelectGrasp`` and
spins until shutdown. It is installed as a console script by the
provided setup.py in your ROS2 package (if applicable), but can also
be executed directly.

Usage::

    python3 main.py

or if installed in a ROS2 package::

    ros2 run your_package_name main

"""

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.exceptions import ROSInterruptException
from grasp_pipeline.grasp_selection_node import YoloLiveSelectGrasp


def main(args=None) -> None:
    """Initialise ROS2 and run the grasp selection node."""
    rclpy.init(args=args)
    node = YoloLiveSelectGrasp()

    # MultiThreadedExecutor (2026-08-06): YOLO inference takes 0.9-2.7 s per
    # frame on this machine, and under the previous `rclpy.spin()` —
    # a SingleThreadedExecutor — that one callback blocked *everything* else on
    # the same thread: the debug-image timer never ran (so /yolo/debug_image
    # published nothing and the debug window never appeared), and even
    # `ros2 param get` timed out. The node looked hung at ~100% CPU.
    #
    # The detection timer and the debug timer are in separate callback groups
    # (see grasp_selection_node), so the debug image now renders while
    # inference is still running.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ROSInterruptException):
        pass
    finally:
        node.get_logger().info("Shutting down node.")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()