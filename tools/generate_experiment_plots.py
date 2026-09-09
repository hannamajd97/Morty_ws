#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OBJECT_ORDER = ["retail_box", "vitamin_tube"]
OBJECT_COLORS = {
    "retail_box": "#1f77b4",
    "vitamin_tube": "#d62728",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate analysis plots from aggregated grasp experiment CSV.")
    parser.add_argument(
        "--complete-csv",
        type=Path,
        default=Path("/home/deep/morty_ws/src/experiment_logs/grasp_experiments_complete.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/home/deep/morty_ws/src/experiment_logs"),
    )
    return parser.parse_args()


def _read_rows(path: Path) -> list[dict[str, str]]:
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


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = sum((x - mx) ** 2 for x in xs)
    deny = sum((y - my) ** 2 for y in ys)
    den = (denx * deny) ** 0.5
    return None if den == 0.0 else num / den


def _save(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def plot_final_minus_seed(rows: list[dict[str, str]], output_dir: Path) -> None:
    ordered = sorted(
        rows,
        key=lambda r: (OBJECT_ORDER.index(r["object_name"]), _i(r["run_index"]) or 9999),
    )
    x_labels = [f"{r['object_name'].replace('_', ' ')} {r['run_index']}" for r in ordered]
    y_vals = [_f(r["final_minus_seed_U"]) for r in ordered]
    colors = [OBJECT_COLORS.get(r["object_name"], "#444444") for r in ordered]

    fig, ax = plt.subplots(figsize=(12, 5.5))
    bars = ax.bar(range(len(ordered)), y_vals, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Final Utility Change by Run")
    ax.set_ylabel("Final U - Seed U")
    ax.set_xlabel("Run")
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(x_labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for idx, bar in enumerate(bars):
        y = y_vals[idx]
        if y is None:
            continue
        va = "bottom" if y >= 0 else "top"
        offset = 0.03 if y >= 0 else -0.03
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y + offset,
            f"{y:+.2f}",
            ha="center",
            va=va,
            fontsize=8,
        )

    handles = [
        plt.Line2D([0], [0], marker="s", color="w", label=obj.replace("_", " "),
                   markerfacecolor=OBJECT_COLORS[obj], markeredgecolor="black", markersize=10)
        for obj in OBJECT_ORDER
    ]
    ax.legend(handles=handles, title="Object", loc="upper right")

    _save(fig, output_dir / "experiment_plot_final_minus_seed")


def plot_voxels_vs_delta(rows: list[dict[str, str]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    xs_all: list[float] = []
    ys_all: list[float] = []

    for obj in OBJECT_ORDER:
        obj_rows = [r for r in rows if r["object_name"] == obj]
        xs = [_f(r["total_new_voxels"]) for r in obj_rows]
        ys = [_f(r["final_minus_seed_U"]) for r in obj_rows]
        xs_all.extend([x for x in xs if x is not None])
        ys_all.extend([y for y in ys if y is not None])
        ax.scatter(
            xs,
            ys,
            s=80,
            color=OBJECT_COLORS[obj],
            edgecolor="black",
            linewidth=0.7,
            alpha=0.9,
            label=obj.replace("_", " "),
        )
        for r, x, y in zip(obj_rows, xs, ys):
            if x is None or y is None:
                continue
            ax.annotate(
                f"{obj[:2]}-{r['run_index']}",
                (x, y),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8,
            )

    corr = _pearson(xs_all, ys_all)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("Voxel Gain vs Final Utility Change")
    ax.set_xlabel("Total New Voxels")
    ax.set_ylabel("Final U - Seed U")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(title="Object", loc="best")
    if corr is not None:
        ax.text(
            0.98,
            0.02,
            f"Pearson r = {corr:.2f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#cccccc", "boxstyle": "round,pad=0.25"},
        )

    _save(fig, output_dir / "experiment_plot_voxels_vs_delta")


def main() -> int:
    args = parse_args()
    rows = _read_rows(args.complete_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_final_minus_seed(rows, args.output_dir)
    plot_voxels_vs_delta(rows, args.output_dir)
    print(f"wrote={args.output_dir / 'experiment_plot_final_minus_seed.png'}")
    print(f"wrote={args.output_dir / 'experiment_plot_final_minus_seed.svg'}")
    print(f"wrote={args.output_dir / 'experiment_plot_voxels_vs_delta.png'}")
    print(f"wrote={args.output_dir / 'experiment_plot_voxels_vs_delta.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
