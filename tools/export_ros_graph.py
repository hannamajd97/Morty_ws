#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import rclpy
from rclpy.node import Node

from qt_dotgraph.pydotfactory import PydotFactory
from rqt_graph.dotcode import NODE_TOPIC_ALL_GRAPH, RosGraphDotcodeGenerator
from rqt_graph.rosgraph2_impl import Graph


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export the live ROS graph to DOT/SVG/PNG.")
    parser.add_argument(
        "--output-prefix",
        default="/home/deep/morty_ws/src/ros_graph_full",
        help="Output path prefix without extension.",
    )
    parser.add_argument("--namespace-filter", default=".*")
    parser.add_argument("--topic-filter", default=".*")
    parser.add_argument("--cluster-level", type=int, default=2)
    parser.add_argument("--no-group-tf", action="store_true")
    parser.add_argument("--no-group-image", action="store_true")
    parser.add_argument("--hide-tf-nodes", action="store_true")
    parser.add_argument("--hide-single-connection-topics", action="store_true")
    parser.add_argument("--hide-dead-end-topics", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--unreachable", action="store_true")
    parser.add_argument(
        "--discovery-wait-s",
        type=float,
        default=3.0,
        help="Time to wait for DDS discovery before exporting.",
    )
    parser.add_argument(
        "--refresh-passes",
        type=int,
        default=3,
        help="How many graph refresh passes to run before exporting.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = Node("ros_graph_exporter")
    try:
        discovery_wait_s = max(0.0, float(args.discovery_wait_s))
        refresh_passes = max(1, int(args.refresh_passes))

        start = time.monotonic()
        last_count = -1
        while time.monotonic() - start < discovery_wait_s:
            rclpy.spin_once(node, timeout_sec=0.1)
            current_count = len(node.get_node_names_and_namespaces())
            if current_count != last_count:
                print(f"discovery_nodes={current_count}")
                last_count = current_count

        graph = Graph(node)
        graph.set_node_stale(5.0)
        graph.graph_stale = 0.0
        for idx in range(refresh_passes):
            updated = graph._graph_refresh()
            print(f"graph_refresh_pass={idx + 1}/{refresh_passes} updated={updated}")
            rclpy.spin_once(node, timeout_sec=0.1)

        generator = RosGraphDotcodeGenerator(node)
        dotcode = generator.generate_dotcode(
            rosgraphinst=graph,
            ns_filter=args.namespace_filter,
            topic_filter=args.topic_filter,
            graph_mode=NODE_TOPIC_ALL_GRAPH,
            hide_single_connection_topics=args.hide_single_connection_topics,
            hide_dead_end_topics=args.hide_dead_end_topics,
            cluster_namespaces_level=max(0, args.cluster_level),
            accumulate_actions=True,
            dotcode_factory=PydotFactory(),
            orientation="LR",
            quiet=args.quiet,
            unreachable=args.unreachable,
            group_tf_nodes=not args.no_group_tf,
            hide_tf_nodes=args.hide_tf_nodes,
            group_image_nodes=not args.no_group_image,
            hide_dynamic_reconfigure=False,
        )

        dot_path = out_prefix.with_suffix(".dot")
        svg_path = out_prefix.with_suffix(".svg")
        png_path = out_prefix.with_suffix(".png")
        dot_path.write_text(dotcode, encoding="utf-8")

        subprocess.run(["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)], check=True)
        subprocess.run(["dot", "-Tpng", str(dot_path), "-o", str(png_path)], check=True)

        discovered_nodes = node.get_node_names_and_namespaces()
        discovered_topics = node.get_topic_names_and_types()
        discovered_services = node.get_service_names_and_types()
        print(f"saved_dot={dot_path}")
        print(f"saved_svg={svg_path}")
        print(f"saved_png={png_path}")
        print(f"discovered_node_count={len(discovered_nodes)}")
        print(f"discovered_topic_count={len(discovered_topics)}")
        print(f"discovered_service_count={len(discovered_services)}")
        print(f"node_count={len(graph.nn_nodes)}")
        print(f"topic_count={len(graph.nt_nodes)}")
        print(f"service_count={len(graph.srvs)}")
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
