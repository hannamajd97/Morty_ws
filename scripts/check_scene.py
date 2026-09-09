#!/usr/bin/env python3
"""Print MoveIt's collision objects with poses/sizes — the no-RViz way to
verify the table is modeled where the real one stands (2026-08-20).

Usage (driver must be running):  python3 scripts/check_scene.py
"""
import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPlanningScene
from moveit_msgs.msg import PlanningSceneComponents

REAL_TABLE_TOP_Z = 0.60  # measured: object-table surface in world(=base)


def main() -> None:
    rclpy.init()
    node = Node("scene_check")
    cli = node.create_client(GetPlanningScene, "/get_planning_scene")
    if not cli.wait_for_service(timeout_sec=5.0):
        print("get_planning_scene service not available — is the driver "
              "(move_group) running?")
        return
    req = GetPlanningScene.Request()
    req.components.components = (
        PlanningSceneComponents.WORLD_OBJECT_GEOMETRY
    )
    fut = cli.call_async(req)
    rclpy.spin_until_future_complete(node, fut, timeout_sec=5.0)
    if fut.result() is None:
        print("service call timed out")
        return
    objs = fut.result().scene.world.collision_objects
    if not objs:
        print("planning scene has NO collision objects (environment "
              "publisher not up yet?)")
        return
    for co in objs:
        p = co.pose.position
        for prim in co.primitives:
            dims = list(prim.dimensions)
            line = (f"{co.id:>16}: center=({p.x:+.3f},{p.y:+.3f},{p.z:+.3f})"
                    f"  dims={[round(d, 3) for d in dims]}")
            if len(dims) == 3:
                top = p.z + dims[2] / 2.0
                bot = p.z - dims[2] / 2.0
                line += f"  z-span [{bot:.3f} .. {top:.3f}]"
                if co.id == "table":
                    diff = top - REAL_TABLE_TOP_Z
                    verdict = ("OK (top just under the real surface)"
                               if -0.03 <= diff <= 0.005 else
                               f"WRONG by {diff:+.3f} m vs real "
                               f"{REAL_TABLE_TOP_Z}")
                    line += f"   <== TABLE: {verdict}"
            print(line)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
