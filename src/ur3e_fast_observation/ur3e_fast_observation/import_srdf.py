#!/usr/bin/env python3
"""Import SRDF group_states into the ur3e_fast_observation YAML registry.

The MoveIt SRDF (e.g. ``ur3e_with_gripper_3d.srdf``) already defines named
joint configurations like ``initgrasp``, ``detect_left``, ``detect_right``,
``detect_higher`` for the ``ur3e_manipulator`` group. Re-teaching them from
``/joint_states`` is unnecessary; this tool reads the SRDF directly and
emits the YAML schema consumed by ``replay_node``.

Usage
-----

::

    ros2 run ur3e_fast_observation import_srdf \\
        --srdf $HOME/morty_ws/src/ur3e_gripper_3d_moveit_config/config/ur3e_with_gripper_3d.srdf \\
        --group ur3e_manipulator \\
        --output $HOME/morty_ws/observation_poses.yaml

By default it imports every ``group_state`` belonging to the requested
group. Use ``--names`` to restrict the set, or ``--exclude`` to drop a few.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Set
from xml.etree import ElementTree as ET

import yaml


ARM_JOINT_ORDER: List[str] = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]


def parse_srdf_group_states(
    srdf_path: str,
    group: str,
    joint_order: List[str],
) -> Dict[str, List[float]]:
    """Return a dict ``{state_name: positions_in_joint_order}`` from an SRDF."""
    if not os.path.isfile(srdf_path):
        raise FileNotFoundError(f"SRDF not found: {srdf_path}")
    tree = ET.parse(srdf_path)
    root = tree.getroot()

    states: Dict[str, List[float]] = {}
    for gs in root.findall('group_state'):
        if gs.attrib.get('group') != group:
            continue
        name = gs.attrib.get('name')
        if not name:
            continue
        joint_values: Dict[str, float] = {}
        for joint in gs.findall('joint'):
            jn = joint.attrib.get('name')
            jv = joint.attrib.get('value')
            if jn is None or jv is None:
                continue
            try:
                joint_values[jn] = float(jv)
            except ValueError:
                continue
        # Only emit states that cover the full requested joint order.
        if all(j in joint_values for j in joint_order):
            states[name] = [joint_values[j] for j in joint_order]
    return states


def write_yaml(
    output_path: str,
    joint_order: List[str],
    states: Dict[str, List[float]],
    source_srdf: str,
) -> None:
    out_dir = os.path.dirname(os.path.abspath(output_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    poses: Dict[str, Dict] = {}
    captured_at = time.time()
    for name, positions in states.items():
        poses[name] = {
            'positions': [float(v) for v in positions],
            'captured_at': captured_at,
            'source': f'srdf:{source_srdf}',
        }
    payload = {
        'joint_names': list(joint_order),
        'poses': poses,
    }
    tmp = f'{output_path}.tmp'
    with open(tmp, 'w') as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)
    os.replace(tmp, output_path)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            'Import named group_states from a MoveIt SRDF into the '
            'observation_poses.yaml format consumed by replay_node.'
        )
    )
    parser.add_argument(
        '--srdf', required=True,
        help='Path to the .srdf (or .srdf.xacro that has been pre-rendered).',
    )
    parser.add_argument(
        '--group', default='ur3e_manipulator',
        help='SRDF group name. Default: ur3e_manipulator.',
    )
    parser.add_argument(
        '--output', required=True,
        help='Destination YAML file.',
    )
    parser.add_argument(
        '--names', nargs='*', default=None,
        help='Optional list of state names to include. Default: all.',
    )
    parser.add_argument(
        '--exclude', nargs='*', default=None,
        help='Optional list of state names to exclude.',
    )
    parser.add_argument(
        '--joint-order', nargs='+', default=ARM_JOINT_ORDER,
        help='Joint names in trajectory order. Default: UR3e arm joints.',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Parse and print what would be written, without touching disk.',
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    states = parse_srdf_group_states(args.srdf, args.group, args.joint_order)
    if not states:
        print(
            f"[import-srdf] no group_states found in '{args.srdf}' "
            f"for group '{args.group}' covering joints {args.joint_order}.",
            file=sys.stderr,
        )
        return 1

    keep: Set[str] = set(states.keys())
    if args.names:
        keep &= set(args.names)
    if args.exclude:
        keep -= set(args.exclude)
    filtered = {name: states[name] for name in keep}

    if not filtered:
        print('[import-srdf] filter left no states to write.', file=sys.stderr)
        return 1

    if args.dry_run:
        print(f"[import-srdf] (dry run) would write {len(filtered)} pose(s) "
              f"to {args.output}:")
        for name, positions in filtered.items():
            print(f'  {name}: {positions}')
        return 0

    write_yaml(args.output, args.joint_order, filtered, args.srdf)
    print(
        f"[import-srdf] wrote {len(filtered)} pose(s) to {args.output}: "
        f"{sorted(filtered.keys())}"
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
