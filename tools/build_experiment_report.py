#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build revised experiment results report from aggregated CSVs.")
    parser.add_argument(
        "--complete-csv",
        type=Path,
        default=Path("/home/deep/morty_ws/src/experiment_logs/grasp_experiments_complete.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/home/deep/morty_ws/src/experiment_logs/grasp_experiments_object_summary.csv"),
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("/home/deep/morty_ws/src/experiment_logs/experiment_results_report_revised.html"),
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _i(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _fmt(value: float | int | None, digits: int = 3, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if signed:
        return f"{value:+.{digits}f}"
    return f"{value:.{digits}f}"


def _escape(text: str | None) -> str:
    return html.escape("" if text is None else str(text))


def _table(headers: list[str], rows: list[list[str]]) -> str:
    parts = ["<table>", "<thead><tr>"]
    for h in headers:
        parts.append(f"<th>{_escape(h)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(f"<td>{cell}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "".join(parts)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs)
    den_y = sum((y - my) ** 2 for y in ys)
    den = (den_x * den_y) ** 0.5
    return None if den == 0.0 else num / den


def build_html(complete_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> str:
    by_object: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in complete_rows:
        by_object[row["object_name"]].append(row)

    for rows in by_object.values():
        rows.sort(key=lambda r: int(r["run_index"]))

    total_runs = len(complete_rows)
    total_improved = sum(1 for r in complete_rows if _f(r["final_minus_seed_U"]) and _f(r["final_minus_seed_U"]) > 0.0)
    total_worsened = sum(1 for r in complete_rows if _f(r["final_minus_seed_U"]) and _f(r["final_minus_seed_U"]) < 0.0)
    direction_counts = Counter(r["direction_selected"] for r in complete_rows)
    stop_counts = Counter((r["stop_reason"] or "sweep complete") for r in complete_rows)

    voxel_vals = [_f(r["total_new_voxels"]) for r in complete_rows]
    delta_vals = [_f(r["final_minus_seed_U"]) for r in complete_rows]
    voxel_vals_f = [v for v in voxel_vals if v is not None]
    delta_vals_f = [d for d in delta_vals if d is not None]
    corr = _pearson([float(r["total_new_voxels"]) for r in complete_rows], [float(r["final_minus_seed_U"]) for r in complete_rows])

    per_run_tables = []
    for object_name in sorted(by_object):
        rows = []
        for r in by_object[object_name]:
            rows.append(
                [
                    _escape(r["run_index"]),
                    _escape(r["direction_selected"]),
                    _fmt(_f(r["seed_U"])),
                    _fmt(_f(r["final_U"])),
                    _fmt(_f(r["final_minus_seed_U"]), signed=True),
                    _escape(r["accepted_followup_frames"]),
                    _escape(r["total_new_voxels"]),
                    _escape((r["stop_reason"] or "sweep complete").replace("because ", "")),
                ]
            )
        per_run_tables.append(
            f"<h3>{_escape(object_name.replace('_', ' ').title())}</h3>"
            + _table(
                ["Run", "Direction", "Seed U", "Final U", "Final - Seed U", "Frames", "New Voxels", "Stop Reason"],
                rows,
            )
        )

    summary_table_rows = []
    for r in summary_rows:
        summary_table_rows.append(
            [
                _escape(r["object_name"]),
                _escape(r["complete_runs"]),
                _escape(r["improved_runs"]),
                _escape(r["worsened_runs"]),
                _fmt(_f(r["mean_final_minus_seed_U"]), signed=True),
                _fmt(_f(r["mean_total_new_voxels"]), digits=1),
            ]
        )

    direction_table_rows = []
    deltas_by_direction: dict[str, list[float]] = defaultdict(list)
    for r in complete_rows:
        if _f(r["final_minus_seed_U"]) is not None:
            deltas_by_direction[r["direction_selected"]].append(float(r["final_minus_seed_U"]))
    for direction in sorted(direction_counts):
        direction_table_rows.append(
            [
                _escape(direction),
                str(direction_counts[direction]),
                _fmt(mean(deltas_by_direction[direction]) if deltas_by_direction[direction] else None, signed=True),
            ]
        )

    retail_box_rows = by_object.get("retail_box", [])
    vitamin_rows = by_object.get("vitamin_tube", [])

    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Multi-View Grasp Utility Experiment Results (Revised)</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      max-width: 980px;
      margin: 32px auto;
      line-height: 1.45;
      color: #111;
    }}
    h1, h2, h3 {{ color: #1b3556; }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 12px 0 22px 0;
      font-size: 0.95rem;
    }}
    th, td {{
      border: 1px solid #bfc7d1;
      padding: 6px 8px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #e9eff6;
    }}
    code {{
      background: #f4f4f4;
      padding: 1px 4px;
      border-radius: 3px;
    }}
    .note {{
      background: #f7f9fb;
      border-left: 4px solid #8aa5c2;
      padding: 10px 12px;
      margin: 14px 0;
    }}
  </style>
</head>
<body>
  <h1>Multi-View Grasp Utility Experiment Results</h1>

  <h2>1. Research Question</h2>
  <p>This report evaluates whether additional viewpoints in the active grasping pipeline improve downstream grasp utility. The key question is not whether more geometry can be acquired, but whether that additional geometry actually improves the quality of the grasp set used by the final planner.</p>
  <p>The current thesis direction is: <strong>the value of an additional view is state-dependent</strong>. It depends on object geometry, current viewpoint quality, and the current grasp-utility state. As a result, geometric gain alone is not a reliable continuation criterion.</p>

  <h2>2. System Used In These Experiments</h2>
  <p>The evaluated system uses a UR3e arm with a wrist-mounted Intel RealSense D345 camera. The main pipeline stages are:</p>
  <ul>
    <li>initial YOLO-based object detection and user selection from the seed pose <code>initgrasp</code></li>
    <li>instance segmentation and seed point-cloud extraction</li>
    <li>continuous observation sweep toward <code>detect_left</code>, <code>detect_right</code>, or <code>detect_higher</code></li>
    <li>tracked follow-up localization using projected 3D priors fused with fresh segmentation</li>
    <li>multi-view fusion and per-view grasp-utility evaluation</li>
    <li>utility-aware stopping when further views stop helping</li>
  </ul>
  <div class="note">
    <strong>Important:</strong> these experiments were run on the framing-aware and utility-aware version of the pipeline. The report therefore evaluates a stabilized active-perception policy rather than the earlier unstable “always trust detect_higher” behavior.
  </div>

  <h2>3. Metrics Used</h2>
  <p><strong>Seed U</strong> is the grasp utility from the initial single-view observation. <strong>Final U</strong> is the utility from the final fused cloud. <strong>Final - Seed U</strong> is the main outcome metric for each run.</p>
  <p><strong>New Voxels</strong> measures geometric gain from added observations. It is included specifically to test whether geometric gain tracks grasp-utility gain. <strong>Frames</strong> is the number of accepted follow-up views that were actually fused.</p>

  <h2>4. Dataset Used For Analysis</h2>
  <p>The current analysis includes <strong>{total_runs}</strong> complete runs:</p>
  { _table(["Object", "Runs", "Improved", "Worsened", "Mean Final - Seed U", "Mean New Voxels"], summary_table_rows) }
  <p>Incomplete retries and partial logs were excluded from the analysis table.</p>

  <h2>5. Per-Run Results</h2>
  {''.join(per_run_tables)}

  <h2>6. Cross-Run Analysis</h2>

  <h3>6.1 Geometric Gain Is Not A Reliable Positive Proxy For Grasp Gain</h3>
  <p>Across all {total_runs} complete runs, every run added new voxels, but utility did not improve consistently. The overall mean utility change was <strong>{_fmt(mean(delta_vals_f), signed=True)}</strong>. The voxel-gain versus utility-change correlation in this small dataset is <strong>{_fmt(corr)}</strong>, which is not only non-positive but moderately negative. This directly contradicts the assumption that more geometric coverage should reliably improve the downstream grasp set.</p>
  <ul>
    <li>Retail box run 1 added 1316 voxels but lost {_fmt(_f(retail_box_rows[0]["final_minus_seed_U"]), signed=True)} in utility.</li>
    <li>Retail box run 4 added only 514 voxels but improved by {_fmt(_f([r for r in retail_box_rows if r["run_index"] == "4"][0]["final_minus_seed_U"]), signed=True)}.</li>
    <li>Vitamin tube run 6 added 1041 voxels but lost {_fmt(_f(vitamin_rows[-1]["final_minus_seed_U"]), signed=True)}.</li>
  </ul>

  <h3>6.2 Retail Box Shows State-Dependent Benefit, Not Uniform Benefit</h3>
  <p><strong>Retail box</strong> is not a simple “multi-view helps” case. It is a higher-variance object. Two runs degraded strongly, while four improved. The strongest positive cases occurred when the seed view was weak: retail box runs 4 and 7 started from low seed utility and improved by {_fmt(_f([r for r in retail_box_rows if r["run_index"] == "4"][0]["final_minus_seed_U"]), signed=True)} and {_fmt(_f([r for r in retail_box_rows if r["run_index"] == "7"][0]["final_minus_seed_U"]), signed=True)} respectively. This supports a state-dependent interpretation: lateral views are especially useful when the initial box observation hides important depth or side-surface structure.</p>

  <h3>6.3 Vitamin Tube Mostly Shows Flat-To-Negative Return From Extra Views</h3>
  <p><strong>Vitamin tube</strong> shows the opposite trend. It improved in only 2 of 5 runs, with mean utility change <strong>{_fmt(_f([r for r in summary_rows if r["object_name"] == "vitamin_tube"][0]["mean_final_minus_seed_U"]), signed=True)}</strong>. This suggests that the seed overhead view already captures much of the useful graspable geometry for this cylindrical object, and additional views often add coverage without creating a better final grasp set.</p>

  <h3>6.4 High-Seed Runs Tend To Have Lower Marginal Value</h3>
  <p>The strongest improvements generally occurred in weak-seed cases rather than already-strong seed cases. This is clearest for retail box runs 4, 6, and 7, where seed utility was low to moderate and added views produced meaningful gains. In contrast, several high-seed runs degraded after additional views. This should be described as a trend, not an absolute rule, but it supports the claim that the value of another view depends on the current grasp state.</p>

  <h3>6.5 Utility-Aware Stopping Behaved Reasonably</h3>
  <p>Utility-aware stopping triggered in {stop_counts['because grasp utility stopped improving.']} of {total_runs} complete runs. These were cases where utility had already flattened or declined. The stopping mechanism therefore appears to be aligned with the thesis goal: avoid continuing a sweep just because geometry is still increasing.</p>

  <h3>6.6 Direction Selection</h3>
  { _table(["Direction", "Times Chosen", "Mean Final - Seed U"], direction_table_rows) }
  <p><code>detect_higher</code> was never selected in the completed runs, which is consistent with the framing-aware penalties added earlier. The policy is effectively choosing between left and right lateral views for these objects.</p>

  <h2>7. Main Findings</h2>
  <ol>
    <li>Additional views are <strong>not uniformly useful</strong>. Across the 11 complete runs, utility improved in 6 runs and worsened in 5.</li>
    <li>The value of an additional view is <strong>state-dependent</strong>. It depends on object geometry, current viewpoint quality, and current grasp-utility state.</li>
    <li>Geometric gain is a <strong>weak continuation criterion</strong>. New voxels were consistently positive, but utility change was inconsistent and often negative.</li>
    <li>Utility-aware stopping is justified. The data supports stopping based on grasp-state evolution rather than geometry accumulation alone.</li>
  </ol>

  <h2>8. Thesis Framing</h2>
  <p><strong>Main claim:</strong> In this active grasping pipeline, the utility of an additional observation is state-dependent; geometric gain is therefore an unreliable standalone proxy for downstream grasp quality.</p>
  <p><strong>Secondary claim:</strong> Active perception should use utility-aware continuation or stopping rules, rather than geometry-only continuation targets.</p>

  <h2>9. Limitations</h2>
  <ul>
    <li>Only two object categories are currently analysed.</li>
    <li>The dataset is thesis-scale, not large enough for strong statistical claims.</li>
    <li>The experiments use one planner pipeline and one robot-camera configuration.</li>
    <li>The current document is strongest as a systems/experimental contribution, not as a broad theory of active perception.</li>
  </ul>

  <h2>10. What Still Needs To Be Added</h2>
  <ul>
    <li>a pipeline figure</li>
    <li>a plot of <code>Final - Seed U</code> by run and object</li>
    <li>a scatter plot of <code>New Voxels</code> versus <code>Final - Seed U</code></li>
    <li>two qualitative frame sequences: one helpful extra-view case and one non-helpful case</li>
  </ul>
</body>
</html>
"""
    return html_text


def main() -> int:
    args = parse_args()
    complete_rows = _read_csv(args.complete_csv)
    summary_rows = _read_csv(args.summary_csv)
    html_text = build_html(complete_rows, summary_rows)
    args.output_html.write_text(html_text, encoding="utf-8")
    print(f"wrote_html={args.output_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
