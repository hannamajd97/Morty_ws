#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FILENAME_RE = re.compile(r"^(?P<timestamp>\d{8}_\d{6})_(?P<tag>.+)_run$")
RUN_TAG_RE = re.compile(r"^(?P<object>.+)_run(?P<index>\d+)$")

NUMERIC_INT_FIELDS = {
    "accepted_followup_frames",
    "sweep_captured_frames_reported",
    "border_rejection_count",
    "total_new_voxels",
}

NUMERIC_FLOAT_FIELDS = {
    "seed_U",
    "best_U_during_sweep",
    "best_nonseed_U",
    "final_U",
    "final_minus_seed_U",
    "best_minus_seed_U",
    "best_nonseed_minus_seed_U",
    "max_delta_U",
    "min_delta_U",
    "score_detect_left",
    "score_detect_right",
    "score_detect_higher",
    "higher_low_penalty",
    "higher_edge_penalty",
}

OBJECT_NORMALIZATION = {
    "viretail_box": "retail_box",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate structured grasp experiment CSVs.")
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("/home/deep/morty_ws/src/experiment_logs"),
        help="Directory containing per-run *_run.csv files.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("/home/deep/morty_ws/src/experiment_logs/grasp_experiments"),
        help="Prefix for generated CSV files.",
    )
    return parser.parse_args()


def _coerce_value(key: str, value: str) -> Any:
    value = value.strip()
    if value == "":
        return None
    if key in NUMERIC_INT_FIELDS:
        try:
            return int(float(value))
        except Exception:
            return value
    if key in NUMERIC_FLOAT_FIELDS:
        try:
            return float(value)
        except Exception:
            return value
    return value


def _object_name_from_tag(run_tag: str) -> tuple[str, int | None]:
    m = RUN_TAG_RE.match(run_tag)
    if not m:
        return OBJECT_NORMALIZATION.get(run_tag, run_tag), None
    obj = OBJECT_NORMALIZATION.get(m.group("object"), m.group("object"))
    return obj, int(m.group("index"))


def _read_run_row(path: Path) -> dict[str, Any]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        row = next(reader)

    stem_match = FILENAME_RE.match(path.stem)
    timestamp = stem_match.group("timestamp") if stem_match else ""
    run_tag = stem_match.group("tag") if stem_match else path.stem
    object_name, run_index = _object_name_from_tag(run_tag)

    parsed = {k: _coerce_value(k, v or "") for k, v in row.items()}
    parsed["csv_file"] = str(path)
    parsed["timestamp"] = timestamp
    parsed["run_tag"] = run_tag
    parsed["object_name"] = object_name
    parsed["run_index"] = run_index
    parsed["is_complete"] = bool(
        parsed.get("direction_selected")
        and parsed.get("seed_U") is not None
        and parsed.get("final_U") is not None
    )
    return parsed


def _sort_key(row: dict[str, Any]) -> tuple[str, int, str]:
    run_index = row.get("run_index")
    return (
        str(row.get("object_name") or ""),
        int(run_index) if run_index is not None else 10**9,
        str(row.get("timestamp") or ""),
    )


def _pick_latest_per_tag(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tag[str(row["run_tag"])].append(row)

    chosen: list[dict[str, Any]] = []
    for tag, group in by_tag.items():
        group_sorted = sorted(
            group,
            key=lambda r: (bool(r["is_complete"]), str(r.get("timestamp") or "")),
        )
        chosen.append(group_sorted[-1])
    return sorted(chosen, key=_sort_key)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_object_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("is_complete"):
            by_object[str(row["object_name"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for object_name, group in sorted(by_object.items()):
        final_changes = [float(r["final_minus_seed_U"]) for r in group if r.get("final_minus_seed_U") is not None]
        voxels = [float(r["total_new_voxels"]) for r in group if r.get("total_new_voxels") is not None]
        improved = sum(1 for r in group if (r.get("final_minus_seed_U") or 0.0) > 0.0)
        worsened = sum(1 for r in group if (r.get("final_minus_seed_U") or 0.0) < 0.0)
        flat = len(group) - improved - worsened
        summary_rows.append(
            {
                "object_name": object_name,
                "complete_runs": len(group),
                "improved_runs": improved,
                "worsened_runs": worsened,
                "flat_runs": flat,
                "mean_final_minus_seed_U": mean(final_changes) if final_changes else None,
                "mean_total_new_voxels": mean(voxels) if voxels else None,
                "max_final_minus_seed_U": max(final_changes) if final_changes else None,
                "min_final_minus_seed_U": min(final_changes) if final_changes else None,
            }
        )
    return summary_rows


def main() -> int:
    args = parse_args()
    run_csvs = sorted(args.logs_dir.glob("*_run.csv"))
    if not run_csvs:
        raise SystemExit(f"No *_run.csv files found in {args.logs_dir}")

    all_rows = sorted((_read_run_row(path) for path in run_csvs), key=_sort_key)
    latest_per_tag = _pick_latest_per_tag(all_rows)
    complete_rows = [row for row in latest_per_tag if row["is_complete"]]
    object_summary = _build_object_summary(latest_per_tag)

    base_fields = [
        "timestamp",
        "run_tag",
        "object_name",
        "run_index",
        "is_complete",
        "csv_file",
    ]
    metric_fields = [
        "raw_log",
        "direction_selected",
        "seed_U",
        "best_U_during_sweep",
        "best_label_during_sweep",
        "best_nonseed_U",
        "best_nonseed_label",
        "final_U",
        "final_minus_seed_U",
        "best_minus_seed_U",
        "best_nonseed_minus_seed_U",
        "max_delta_U",
        "min_delta_U",
        "accepted_followup_frames",
        "sweep_captured_frames_reported",
        "border_rejection_count",
        "total_new_voxels",
        "stop_reason",
        "score_detect_left",
        "score_detect_right",
        "score_detect_higher",
        "higher_low_penalty",
        "higher_edge_penalty",
    ]
    row_fields = base_fields + metric_fields
    summary_fields = [
        "object_name",
        "complete_runs",
        "improved_runs",
        "worsened_runs",
        "flat_runs",
        "mean_final_minus_seed_U",
        "mean_total_new_voxels",
        "max_final_minus_seed_U",
        "min_final_minus_seed_U",
    ]

    all_path = Path(f"{args.output_prefix}_all.csv")
    latest_path = Path(f"{args.output_prefix}_latest_per_tag.csv")
    complete_path = Path(f"{args.output_prefix}_complete.csv")
    summary_path = Path(f"{args.output_prefix}_object_summary.csv")

    _write_csv(all_path, all_rows, row_fields)
    _write_csv(latest_path, latest_per_tag, row_fields)
    _write_csv(complete_path, complete_rows, row_fields)
    _write_csv(summary_path, object_summary, summary_fields)

    print(f"wrote_all={all_path}")
    print(f"wrote_latest_per_tag={latest_path}")
    print(f"wrote_complete={complete_path}")
    print(f"wrote_object_summary={summary_path}")
    print(f"all_rows={len(all_rows)}")
    print(f"latest_per_tag_rows={len(latest_per_tag)}")
    print(f"complete_rows={len(complete_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
