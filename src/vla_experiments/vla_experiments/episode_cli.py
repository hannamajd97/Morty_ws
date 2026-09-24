"""Tiny CLI for the recorder.

    vla_episode start "pick up the vitamin c and put it in the box"
    vla_episode stop            # outcome success
    vla_episode stop fail       # keep, outcome failure
    vla_episode stop unknown    # keep, outcome unknown
    vla_episode stop auto       # the executor's cycle_done verdict, else unknown
    vla_episode discard
    vla_episode task "new instruction"
    vla_episode hint <trial folder>  # the next trial episode is written INTO that folder
    vla_episode status
    vla_episode gripper open|close   # manual note of a gripper command (the adapter publishes the real ones)
"""
import json
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy
from std_msgs.msg import String

LATCHED = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.RELIABLE,
                     durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in ("start", "stop", "discard", "task", "hint", "status", "gripper"):
        print(__doc__)
        return 2
    rclpy.init()
    node = Node("vla_episode_cli")
    try:
        if argv[0] == "status":
            got = []
            node.create_subscription(String, "/vla_recorder/status", lambda m: got.append(m.data), LATCHED)
            t0 = time.time()
            while not got and time.time() - t0 < 3.0:
                rclpy.spin_once(node, timeout_sec=0.1)
            if not got:
                print("no recorder status received (is episode_recorder running?)")
                return 1
            print(json.dumps(json.loads(got[0]), indent=2))
            return 0
        if argv[0] == "gripper":
            pub = node.create_publisher(String, "/vla_recorder/gripper_cmd", 10)
            text = argv[1] if len(argv) > 1 else "open"
        else:
            pub = node.create_publisher(String, "/vla_recorder/cmd", 10)
            text = argv[0] + (" " + " ".join(argv[1:]) if len(argv) > 1 else "")
        t0 = time.time()
        while pub.get_subscription_count() == 0 and time.time() - t0 < 3.0:
            rclpy.spin_once(node, timeout_sec=0.1)
        if pub.get_subscription_count() == 0:
            print("recorder not listening (is episode_recorder running?)")
            return 1
        # The count says WE discovered the recorder's reader; the reader
        # may not have matched this brand-new writer yet, and a message
        # sent in that window is silently lost (seen 2026-09-24: the
        # first "hint" after start-up never arrived). Give discovery a
        # moment to complete on both sides.
        t0 = time.time()
        while time.time() - t0 < 0.5:
            rclpy.spin_once(node, timeout_sec=0.1)
        pub.publish(String(data=text))
        rclpy.spin_once(node, timeout_sec=0.2)
        print(f"sent: {text}")
        return 0
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    sys.exit(main())
