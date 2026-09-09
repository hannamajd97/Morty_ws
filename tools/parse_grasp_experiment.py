#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


RE_DIRECTION_DETAILS = re.compile(
    r"\[continuous\] direction score details: .*?"
    r"height_ratio=(?P<height_ratio>-?\d+(?:\.\d+)?), "
    r"lateral_score=(?P<lateral_score>-?\d+(?:\.\d+)?), "
    r"depth_slope=(?P<depth_slope>-?\d+(?:\.\d+)?), "
    r"depth_slope_bonus=(?P<depth_slope_bonus>-?\d+(?:\.\d+)?), .*?"
    r"img_offset_px=(?P<img_offset_px>-?\d+(?:\.\d+)?), "
    r"center_bias_bonus=(?P<center_bias_bonus>-?\d+(?:\.\d+)?), "
    r"higher_low_penalty=(?P<higher_low_penalty>-?\d+(?:\.\d+)?), "
    r"higher_edge_penalty=(?P<higher_edge_penalty>-?\d+(?:\.\d+)?)"
)
RE_DIRECTION_SELECTED = re.compile(
    r"\[continuous\] one-direction score selected '(?P<direction>[^']+)' "
    r"\(left=(?P<left>-?\d+(?:\.\d+)?), "
    r"right=(?P<right>-?\d+(?:\.\d+)?), "
    r"higher=(?P<higher>-?\d+(?:\.\d+)?)\)"
)
RE_ACCEPTED_FRAME = re.compile(
    r"\[continuous\] accepted frame (?P<frame>\d+)/(?P<max_frames>\d+) "
    r"\(dt=(?P<dt>-?\d+(?:\.\d+)?)m, dr=(?P<dr>-?\d+(?:\.\d+)?)deg, pts=(?P<pts>\d+)\)"
)
RE_ACCEPTED_FINAL = re.compile(
    r"\[continuous\] accepted final frame "
    r"\(dt=(?P<dt>-?\d+(?:\.\d+)?)m, dr=(?P<dr>-?\d+(?:\.\d+)?)deg, pts=(?P<pts>\d+)\)"
)
RE_SWEEP_COMPLETE = re.compile(
    r"\[continuous\] sweep leg '(?P<direction>[^']+)' complete: "
    r"captured (?P<captured>\d+) frames \(prior=(?P<prior>\d+)\)"
)
RE_STOPPING = re.compile(
    r"\[continuous\] stopping sweep '(?P<direction>[^']+)' (?P<reason>.*)"
)
RE_BORDER_REJECT = re.compile(
    r"\[continuous\] frame (?P<frame>\d+) rejected for border clipping "
    r"\(area_ratio=(?P<area_ratio>-?\d+(?:\.\d+)?), "
    r"height_ratio=(?P<height_ratio>-?\d+(?:\.\d+)?), "
    r"edges=L(?P<left>[01])/T(?P<top>[01])/R(?P<right>[01])/B(?P<bottom>[01])\)"
)
RE_UTILITY_U = re.compile(
    r"\[grasp-utility\] (?P<label>view_[^:]+|final_cloud): U=(?P<U>-?\d+(?:\.\d+)?) "
    r"\(count=(?P<count>\d+), diversity_deg=(?P<diversity>-?\d+(?:\.\d+)?), "
    r"mean_score=(?P<mean_score>-?\d+(?:\.\d+)?), instability=(?P<instability>-?\d+(?:\.\d+)?)\)"
)
RE_UTILITY_DELTA = re.compile(
    r"\[grasp-utility\] (?P<label>view_[^:]+): delta_U=(?P<delta>-?\d+(?:\.\d+)?) "
    r"\(prev=(?P<prev>-?\d+(?:\.\d+)?) -> now=(?P<now>-?\d+(?:\.\d+)?)\)"
)
RE_UTILITY_VOXEL = re.compile(
    r"\[grasp-utility\] (?P<label>view_[^:]+): voxel_gain "
    r"\(new_voxels=(?P<new_voxels>\d+), gain_ratio=(?P<gain_ratio>-?\d+(?:\.\d+)?), "
    r"total_voxels=(?P<total_voxels>\d+)\)"
)
RE_BEST_GRASP_SCORE = re.compile(
    r"Best grasp score=(?P<score>-?\d+(?:\.\d+)?) @ "
    r"t=\((?P<x>-?\d+(?:\.\d+)?),(?P<y>-?\d+(?:\.\d+)?),(?P<z>-?\d+(?:\.\d+)?)\) "
    r"frame=(?P<frame>\S+)"
)
RE_EXECUTOR_STATUS = re.compile(
    r"Received executor cycle status: (?P<status>done|failed)(?: \(cycle_id=(?P<cycle_id>\d+)\))?"
)
RE_PUBLISHED_CANDIDATES = re.compile(
    r"Published (?P<count>\d+) grasp candidates on '/graspgen/candidate_poses_base_link'"
)


def _as_float(text: str | None) -> float | None:
    if text is None:
        return None
    return float(text)


def _as_int(text: str | None) -> int | None:
    if text is None:
        return None
    return int(text)


@dataclass
class UtilityEntry:
    label: str
    kind: str
    U: float | None = None
    delta_U: float | None = None
    prev_U: float | None = None
    now_U: float | None = None
    count: int | None = None
    diversity_deg: float | None = None
    mean_score: float | None = None
    instability: float | None = None
    new_voxels: int | None = None
    gain_ratio: float | None = None
    total_voxels: int | None = None


@dataclass
class ParsedRun:
    raw_log: str
    direction_details: dict[str, Any] = field(default_factory=dict)
    direction_selected: str | None = None
    direction_scores: dict[str, float] = field(default_factory=dict)
    accepted_frames: list[dict[str, Any]] = field(default_factory=list)
    accepted_final: dict[str, Any] | None = None
    sweep_complete: dict[str, Any] | None = None
    stop_events: list[dict[str, Any]] = field(default_factory=list)
    border_rejections: list[dict[str, Any]] = field(default_factory=list)
    utility_by_label: dict[str, UtilityEntry] = field(default_factory=dict)
    best_grasp: dict[str, Any] | None = None
    executor_status: str | None = None
    executor_cycle_id: int | None = None
    published_candidate_count: int | None = None

    def get_or_make_utility(self, label: str) -> UtilityEntry:
        entry = self.utility_by_label.get(label)
        if entry is None:
            kind = "final" if label == "final_cloud" else (
                "seed" if label == "view_1_seed" else "view"
            )
            entry = UtilityEntry(label=label, kind=kind)
            self.utility_by_label[label] = entry
        return entry

    def ordered_utility_labels(self) -> list[str]:
        def sort_key(label: str) -> tuple[int, int]:
            if label == "view_1_seed":
                return (0, 1)
            if label == "final_cloud":
                return (2, 9999)
            m = re.match(r"view_(\d+)", label)
            if m:
                return (1, int(m.group(1)))
            return (3, 9999)

        return sorted(self.utility_by_label.keys(), key=sort_key)

    def to_structured_dict(self) -> dict[str, Any]:
        utility_sequence = []
        for label in self.ordered_utility_labels():
            entry = self.utility_by_label[label]
            utility_sequence.append(
                {
                    "label": entry.label,
                    "kind": entry.kind,
                    "U": entry.U,
                    "delta_U": entry.delta_U,
                    "prev_U": entry.prev_U,
                    "now_U": entry.now_U,
                    "count": entry.count,
                    "diversity_deg": entry.diversity_deg,
                    "mean_score": entry.mean_score,
                    "instability": entry.instability,
                    "new_voxels": entry.new_voxels,
                    "gain_ratio": entry.gain_ratio,
                    "total_voxels": entry.total_voxels,
                }
            )

        seed_u = self.utility_by_label.get("view_1_seed").U if "view_1_seed" in self.utility_by_label else None
        final_u = self.utility_by_label.get("final_cloud").U if "final_cloud" in self.utility_by_label else None
        view_entries = [
            e for e in (self.utility_by_label[label] for label in self.ordered_utility_labels())
            if e.U is not None and e.kind != "final"
        ]
        best_entry = max(view_entries, key=lambda e: e.U) if view_entries else None
        nonseed_entries = [e for e in view_entries if e.kind == "view"]
        best_nonseed_entry = max(nonseed_entries, key=lambda e: e.U) if nonseed_entries else None
        delta_values = [e.delta_U for e in self.utility_by_label.values() if e.delta_U is not None]
        total_new_voxels = sum(e.new_voxels or 0 for e in self.utility_by_label.values() if e.new_voxels is not None)
        final_count = self.utility_by_label.get("final_cloud").count if "final_cloud" in self.utility_by_label else None
        best_score = self.best_grasp["score"] if self.best_grasp is not None else None
        offline_proxy = self._offline_executability_proxy(final_u, final_count, best_score)

        summary = {
            "direction_selected": self.direction_selected,
            "direction_scores": self.direction_scores,
            "seed_U": seed_u,
            "best_U_during_sweep": best_entry.U if best_entry is not None else None,
            "best_label_during_sweep": best_entry.label if best_entry is not None else None,
            "best_nonseed_U": best_nonseed_entry.U if best_nonseed_entry is not None else None,
            "best_nonseed_label": best_nonseed_entry.label if best_nonseed_entry is not None else None,
            "final_U": final_u,
            "final_minus_seed_U": (final_u - seed_u) if final_u is not None and seed_u is not None else None,
            "best_minus_seed_U": (best_entry.U - seed_u) if best_entry is not None and seed_u is not None else None,
            "best_nonseed_minus_seed_U": (
                best_nonseed_entry.U - seed_u
            ) if best_nonseed_entry is not None and seed_u is not None else None,
            "max_delta_U": max(delta_values) if delta_values else None,
            "min_delta_U": min(delta_values) if delta_values else None,
            "accepted_followup_frames": len(self.accepted_frames) + (1 if self.accepted_final is not None else 0),
            "sweep_captured_frames_reported": self.sweep_complete["captured"] if self.sweep_complete else None,
            "stop_reason": self.stop_events[-1]["reason"] if self.stop_events else None,
            "border_rejection_count": len(self.border_rejections),
            "total_new_voxels": total_new_voxels,
            "published_candidate_count": self.published_candidate_count,
            "best_grasp_score": best_score,
            "executor_status": self.executor_status,
            "executor_cycle_id": self.executor_cycle_id,
            "offline_proxy_executable": offline_proxy["executable"],
            "offline_proxy_reason": offline_proxy["reason"],
        }

        return {
            "raw_log": self.raw_log,
            "direction_details": self.direction_details,
            "direction_selected": self.direction_selected,
            "direction_scores": self.direction_scores,
            "accepted_frames": self.accepted_frames,
            "accepted_final": self.accepted_final,
            "sweep_complete": self.sweep_complete,
            "stop_events": self.stop_events,
            "border_rejections": self.border_rejections,
            "utility_sequence": utility_sequence,
            "best_grasp": self.best_grasp,
            "summary": summary,
        }

    @staticmethod
    def _offline_executability_proxy(
        final_u: float | None,
        final_count: int | None,
        best_score: float | None,
    ) -> dict[str, Any]:
        """Conservative offline proxy, not a physical execution result."""
        missing = []
        if final_u is None:
            missing.append("final_U")
        if final_count is None:
            missing.append("final_candidate_count")
        if best_score is None:
            missing.append("best_grasp_score")
        if missing:
            return {
                "executable": None,
                "reason": "missing " + ",".join(missing),
            }

        checks = [
            (final_u >= 4.5, "final_U", final_u, 4.5),
            (final_count >= 50, "final_count", final_count, 50),
            (best_score >= 0.40, "best_grasp_score", best_score, 0.40),
        ]
        executable = all(ok for ok, _, _, _ in checks)
        return {
            "executable": executable,
            "reason": "; ".join(
                f"{'PASS' if ok else 'FAIL'} {name}={value:.3f}>={threshold:.3f}"
                for ok, name, value, threshold in checks
            ),
        }


def parse_log(raw_log: Path) -> ParsedRun:
    parsed = ParsedRun(raw_log=str(raw_log))
    for line in raw_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := RE_DIRECTION_DETAILS.search(line):
            parsed.direction_details = {
                "height_ratio": _as_float(m.group("height_ratio")),
                "lateral_score": _as_float(m.group("lateral_score")),
                "depth_slope": _as_float(m.group("depth_slope")),
                "depth_slope_bonus": _as_float(m.group("depth_slope_bonus")),
                "img_offset_px": _as_float(m.group("img_offset_px")),
                "center_bias_bonus": _as_float(m.group("center_bias_bonus")),
                "higher_low_penalty": _as_float(m.group("higher_low_penalty")),
                "higher_edge_penalty": _as_float(m.group("higher_edge_penalty")),
            }
            continue
        if m := RE_DIRECTION_SELECTED.search(line):
            parsed.direction_selected = m.group("direction")
            parsed.direction_scores = {
                "detect_left": float(m.group("left")),
                "detect_right": float(m.group("right")),
                "detect_higher": float(m.group("higher")),
            }
            continue
        if m := RE_ACCEPTED_FRAME.search(line):
            parsed.accepted_frames.append(
                {
                    "frame_index": int(m.group("frame")),
                    "max_frames": int(m.group("max_frames")),
                    "dt_m": float(m.group("dt")),
                    "dr_deg": float(m.group("dr")),
                    "points": int(m.group("pts")),
                }
            )
            continue
        if m := RE_ACCEPTED_FINAL.search(line):
            parsed.accepted_final = {
                "dt_m": float(m.group("dt")),
                "dr_deg": float(m.group("dr")),
                "points": int(m.group("pts")),
            }
            continue
        if m := RE_SWEEP_COMPLETE.search(line):
            parsed.sweep_complete = {
                "direction": m.group("direction"),
                "captured": int(m.group("captured")),
                "prior": int(m.group("prior")),
            }
            continue
        if m := RE_STOPPING.search(line):
            parsed.stop_events.append(
                {
                    "direction": m.group("direction"),
                    "reason": m.group("reason").strip(),
                }
            )
            continue
        if m := RE_BORDER_REJECT.search(line):
            parsed.border_rejections.append(
                {
                    "frame_index": int(m.group("frame")),
                    "area_ratio": float(m.group("area_ratio")),
                    "height_ratio": float(m.group("height_ratio")),
                    "touches_left": bool(int(m.group("left"))),
                    "touches_top": bool(int(m.group("top"))),
                    "touches_right": bool(int(m.group("right"))),
                    "touches_bottom": bool(int(m.group("bottom"))),
                }
            )
            continue
        if m := RE_UTILITY_U.search(line):
            entry = parsed.get_or_make_utility(m.group("label"))
            entry.U = float(m.group("U"))
            entry.count = int(m.group("count"))
            entry.diversity_deg = float(m.group("diversity"))
            entry.mean_score = float(m.group("mean_score"))
            entry.instability = float(m.group("instability"))
            continue
        if m := RE_UTILITY_DELTA.search(line):
            entry = parsed.get_or_make_utility(m.group("label"))
            entry.delta_U = float(m.group("delta"))
            entry.prev_U = float(m.group("prev"))
            entry.now_U = float(m.group("now"))
            continue
        if m := RE_UTILITY_VOXEL.search(line):
            entry = parsed.get_or_make_utility(m.group("label"))
            entry.new_voxels = int(m.group("new_voxels"))
            entry.gain_ratio = float(m.group("gain_ratio"))
            entry.total_voxels = int(m.group("total_voxels"))
            continue
        if m := RE_PUBLISHED_CANDIDATES.search(line):
            parsed.published_candidate_count = int(m.group("count"))
            continue
        if m := RE_BEST_GRASP_SCORE.search(line):
            parsed.best_grasp = {
                "score": float(m.group("score")),
                "position": {
                    "x": float(m.group("x")),
                    "y": float(m.group("y")),
                    "z": float(m.group("z")),
                },
                "frame": m.group("frame"),
            }
            continue
        if m := RE_EXECUTOR_STATUS.search(line):
            parsed.executor_status = m.group("status")
            parsed.executor_cycle_id = _as_int(m.group("cycle_id"))
            continue
    return parsed


def write_run_csv(csv_path: Path, structured: dict[str, Any]) -> None:
    summary = structured["summary"]
    row = {
        "raw_log": structured["raw_log"],
        "direction_selected": summary["direction_selected"],
        "seed_U": summary["seed_U"],
        "best_U_during_sweep": summary["best_U_during_sweep"],
        "best_label_during_sweep": summary["best_label_during_sweep"],
        "best_nonseed_U": summary["best_nonseed_U"],
        "best_nonseed_label": summary["best_nonseed_label"],
        "final_U": summary["final_U"],
        "final_minus_seed_U": summary["final_minus_seed_U"],
        "best_minus_seed_U": summary["best_minus_seed_U"],
        "best_nonseed_minus_seed_U": summary["best_nonseed_minus_seed_U"],
        "max_delta_U": summary["max_delta_U"],
        "min_delta_U": summary["min_delta_U"],
        "accepted_followup_frames": summary["accepted_followup_frames"],
        "sweep_captured_frames_reported": summary["sweep_captured_frames_reported"],
        "border_rejection_count": summary["border_rejection_count"],
        "total_new_voxels": summary["total_new_voxels"],
        "published_candidate_count": summary["published_candidate_count"],
        "best_grasp_score": summary["best_grasp_score"],
        "executor_status": summary["executor_status"],
        "executor_cycle_id": summary["executor_cycle_id"],
        "offline_proxy_executable": summary["offline_proxy_executable"],
        "offline_proxy_reason": summary["offline_proxy_reason"],
        "stop_reason": summary["stop_reason"],
        "score_detect_left": structured["direction_scores"].get("detect_left"),
        "score_detect_right": structured["direction_scores"].get("detect_right"),
        "score_detect_higher": structured["direction_scores"].get("detect_higher"),
        "higher_low_penalty": structured["direction_details"].get("higher_low_penalty"),
        "higher_edge_penalty": structured["direction_details"].get("higher_edge_penalty"),
    }
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def write_views_csv(csv_path: Path, structured: dict[str, Any]) -> None:
    rows = []
    for entry in structured["utility_sequence"]:
        rows.append(
            {
                "raw_log": structured["raw_log"],
                "label": entry["label"],
                "kind": entry["kind"],
                "U": entry["U"],
                "delta_U": entry["delta_U"],
                "prev_U": entry["prev_U"],
                "now_U": entry["now_U"],
                "count": entry["count"],
                "diversity_deg": entry["diversity_deg"],
                "mean_score": entry["mean_score"],
                "instability": entry["instability"],
                "new_voxels": entry["new_voxels"],
                "gain_ratio": entry["gain_ratio"],
                "total_voxels": entry["total_voxels"],
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "raw_log",
                    "label",
                    "kind",
                    "U",
                    "delta_U",
                    "prev_U",
                    "now_U",
                    "count",
                    "diversity_deg",
                    "mean_score",
                    "instability",
                    "new_voxels",
                    "gain_ratio",
                    "total_voxels",
                ]
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Parse a grasp experiment raw log into structured files.")
    parser.add_argument("--raw-log", required=True, help="Path to the full raw log file.")
    parser.add_argument("--json-out", required=True, help="Path to the structured JSON output.")
    parser.add_argument("--run-csv-out", required=True, help="Path to the one-row run summary CSV.")
    parser.add_argument("--views-csv-out", required=True, help="Path to the per-view CSV.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    raw_log = Path(args.raw_log)
    parsed = parse_log(raw_log)
    structured = parsed.to_structured_dict()

    json_out = Path(args.json_out)
    run_csv_out = Path(args.run_csv_out)
    views_csv_out = Path(args.views_csv_out)
    json_out.write_text(json.dumps(structured, indent=2), encoding="utf-8")
    write_run_csv(run_csv_out, structured)
    write_views_csv(views_csv_out, structured)

    print(f"structured_json={json_out}")
    print(f"run_csv={run_csv_out}")
    print(f"views_csv={views_csv_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
