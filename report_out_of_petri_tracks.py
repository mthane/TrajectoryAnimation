#!/usr/bin/env python3
"""Report and plot larvae positions outside the Petri dish (starvation dataset)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

from create_larvae_animations import (
    VideoGenConfig,
    apply_fixed_axis_xy_limits,
    config_from_module_globals,
    detect_columns,
    display_group_label,
    load_and_preprocess,
    petri_dish_circle,
    stabilized_limits,
)

PAPER_LABELS = {
    "4d_fed": "4d Sated",
    "4d_starved": "4d Starved",
    "5d_fed": "5d Sated",
    "5d_starved": "5d Starved",
}


def arena_limits(df: pd.DataFrame, x_col: str, y_col: str, cfg: VideoGenConfig) -> Tuple[float, float, float, float]:
    xlim = stabilized_limits(float(df[x_col].min()), float(df[x_col].max()))
    ylim = stabilized_limits(float(df[y_col].min()), float(df[y_col].max()))
    xlim, ylim = apply_fixed_axis_xy_limits(xlim, ylim, cfg.fixed_axis_xy_limits)
    return xlim[0], xlim[1], ylim[0], ylim[1]


def build_outside_tables(
    df: pd.DataFrame,
    cols,
    cfg: VideoGenConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[float, float, float]]:
    xlim = (float(df[cols.x].min()), float(df[cols.x].max()))
    ylim = (float(df[cols.y].min()), float(df[cols.y].max()))
    xlim = stabilized_limits(*xlim)
    ylim = stabilized_limits(*ylim)
    xlim, ylim = apply_fixed_axis_xy_limits(xlim, ylim, cfg.fixed_axis_xy_limits)
    cx, cy, r = petri_dish_circle(xlim, ylim, cfg)

    work = df.copy()
    work["radius_to_center"] = np.sqrt((work[cols.x] - cx) ** 2 + (work[cols.y] - cy) ** 2)
    work["outside_petri_dish"] = work["radius_to_center"] > r
    work["distance_outside_mm"] = np.maximum(0.0, work["radius_to_center"] - r)

    meta_cols = [c for c in ("condition", "trial", "name", "group", cols.larva_id, cols.frame) if c in work.columns]
    outside = work.loc[work["outside_petri_dish"]].copy()
    if len(outside) == 0:
        return outside, pd.DataFrame(), (cx, cy, r)

    point_report_cols = meta_cols + [
        cols.x,
        cols.y,
        "radius_to_center",
        "distance_outside_mm",
        "outside_petri_dish",
    ]
    point_report = outside[point_report_cols].rename(
        columns={
            cols.larva_id: "track_id",
            cols.frame: "frame_index",
            cols.x: "x",
            cols.y: "y",
        }
    )
    if "condition" in point_report.columns:
        point_report["condition_label"] = point_report["condition"].astype(str).map(
            lambda c: PAPER_LABELS.get(c, display_group_label(c, cfg))
        )

    track_groups = []
    for track_id, g in outside.groupby("larva_id_norm"):
        row0 = g.iloc[0]
        track_groups.append(
            {
                "track_id": str(track_id),
                "condition": str(row0.get("condition", "unknown")),
                "condition_label": PAPER_LABELS.get(
                    str(row0.get("condition", "")), display_group_label(str(row0.get("condition", "")), cfg)
                ),
                "trial": str(row0.get("trial", "")),
                "name": str(row0.get("name", "")),
                "group": str(row0.get("group", "")),
                "n_outside_points": int(len(g)),
                "max_radius": float(g["radius_to_center"].max()),
                "max_distance_outside": float(g["distance_outside_mm"].max()),
                "frame_min": int(pd.to_numeric(g[cols.frame], errors="coerce").min()),
                "frame_max": int(pd.to_numeric(g[cols.frame], errors="coerce").max()),
                "x_at_max_radius": float(g.loc[g["radius_to_center"].idxmax(), cols.x]),
                "y_at_max_radius": float(g.loc[g["radius_to_center"].idxmax(), cols.y]),
            }
        )
    track_summary = pd.DataFrame(track_groups).sort_values(
        ["condition", "max_distance_outside"], ascending=[True, False]
    )
    return point_report, track_summary, (cx, cy, r)


def _clip_point_to_axes(x: float, y: float, x0: float, x1: float, y0: float, y1: float) -> Tuple[float, float]:
    return float(np.clip(x, x0, x1)), float(np.clip(y, y0, y1))


def _label_position_inside_arena(
    x: float,
    y: float,
    cx: float,
    cy: float,
    dish_r: float,
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    label_index: int,
    n_labels: int,
) -> Tuple[float, float]:
    """Place ID label inside the visible arena (not at off-scale outlier coordinates)."""
    dx, dy = x - cx, y - cy
    dist = float(np.hypot(dx, dy))
    margin = 3.0
    if dist > 1e-6:
        # Along bearing of the outlier, but well inside the green circle.
        frac = 0.72
        lx = cx + frac * dish_r * dx / dist
        ly = cy + frac * dish_r * dy / dist
    else:
        lx, ly = cx, cy
    # Stagger labels on a small arc so they do not overlap when bearings are similar.
    angle = float(np.arctan2(dy, dx)) + (label_index - 0.5 * (n_labels - 1)) * 0.22
    lx = cx + 0.55 * dish_r * np.cos(angle)
    ly = cy + 0.55 * dish_r * np.sin(angle)
    return _clip_point_to_axes(lx, ly, x0 + margin, x1 - margin, y0 + margin, y1 - margin)


def plot_condition_outside_tracks(
    df: pd.DataFrame,
    cols,
    cfg: VideoGenConfig,
    condition: str,
    flagged_track_ids: List[str],
    output_path: Path,
    dish: Tuple[float, float, float],
) -> None:
    cx, cy, r = dish
    x0, x1, y0, y1 = arena_limits(df, cols.x, cols.y, cfg)
    sub = df[df["condition"].astype(str) == str(condition)].copy()
    flagged = sub[sub["larva_id_norm"].astype(str).isin(flagged_track_ids)].copy()
    if len(flagged) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 9), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.add_patch(
        Circle((cx, cy), r, fill=False, linestyle="-", linewidth=1.6, edgecolor="#2ca02c", alpha=0.95)
    )

    sorted_ids = sorted(flagged_track_ids, key=str)
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(sorted_ids))))
    legend_handles: List = []

    for i, track_id in enumerate(sorted_ids):
        tr = flagged[flagged["larva_id_norm"].astype(str) == str(track_id)].sort_values(cols.frame)
        xs = tr[cols.x].astype(float).to_numpy()
        ys = tr[cols.y].astype(float).to_numpy()
        rad = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
        outside_mask = rad > r
        color = colors[i % len(colors)]

        (line_in,) = ax.plot(
            xs[~outside_mask],
            ys[~outside_mask],
            color=color,
            alpha=0.25,
            linewidth=0.8,
            zorder=2,
            clip_on=True,
        )
        ax.plot(
            xs[outside_mask],
            ys[outside_mask],
            color=color,
            alpha=0.95,
            linewidth=2.0,
            zorder=4,
            clip_on=False,
        )
        ax.scatter(
            xs[outside_mask],
            ys[outside_mask],
            s=22,
            color=color,
            zorder=5,
            clip_on=False,
        )

        if outside_mask.any():
            j = int(np.argmax(rad))
            ox, oy = float(xs[j]), float(ys[j])
            ax.scatter([ox], [oy], s=55, facecolors="none", edgecolors=color, linewidths=1.8, zorder=6, clip_on=False)
            tx, ty = _clip_point_to_axes(ox, oy, x0, x1, y0, y1)
            lx, ly = _label_position_inside_arena(ox, oy, cx, cy, r, x0, x1, y0, y1, i, len(sorted_ids))
            ax.annotate(
                f"id {track_id}",
                xy=(tx, ty),
                xytext=(lx, ly),
                fontsize=10,
                fontweight="bold",
                color=color,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor=color, alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.2, shrinkA=2, shrinkB=2),
                zorder=7,
            )
        legend_handles.append(line_in)

    ax.legend(
        legend_handles,
        [f"id {tid}" for tid in sorted_ids],
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        title="Flagged tracks",
    )

    label = PAPER_LABELS.get(str(condition), display_group_label(condition, cfg))
    ax.set_title(
        f"Out-of-Petri tracks — {label}\n"
        f"(green circle r={r:g}; bold = outside dish; labels inside plot area)",
        fontsize=12,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_overview_all_conditions(
    track_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if track_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    labels = track_summary["condition_label"].astype(str)
    y = np.arange(len(track_summary))
    ax.barh(y, track_summary["max_distance_outside"].astype(float), color="#d62728", alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(
        [f"id {r.track_id} ({r.condition_label})" for r in track_summary.itertuples()],
        fontsize=9,
    )
    ax.set_xlabel("Max distance outside Petri dish (same units as x/y)")
    ax.set_title("Tracks with any position outside the Petri dish")
    ax.invert_yaxis()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Report larvae outside the Petri dish boundary.")
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path("starvation2-2026-04-23_analyzed.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("animations/out_of_petri_report"),
    )
    parser.add_argument("--petri-dish-radius", type=float, default=None)
    args = parser.parse_args(argv)

    cfg = config_from_module_globals()
    if args.petri_dish_radius is not None:
        cfg = cfg.__class__(**{**cfg.__dict__, "petri_dish_radius": float(args.petri_dish_radius)})

    cfg = cfg.__class__(**{**cfg.__dict__, "data_path": args.input_csv})
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = detect_columns(args.input_csv, cfg)
    df = load_and_preprocess(args.input_csv, cols, cfg, skip_group_value_filter=True)
    meta_cols = [c for c in ("trial", "name", "group") if c in pd.read_csv(args.input_csv, nrows=0).columns]
    if meta_cols:
        meta = pd.read_csv(
            args.input_csv,
            usecols=[cols.larva_id, *meta_cols],
            low_memory=False,
        ).drop_duplicates(subset=[cols.larva_id])
        df = df.merge(meta, on=cols.larva_id, how="left", suffixes=("", "_meta"))
    point_report, track_summary, dish = build_outside_tables(df, cols, cfg)
    cx, cy, r = dish

    print(f"Petri dish: center=({cx:.3f}, {cy:.3f}), radius={r:.3f}")
    print(f"Total rows analyzed: {len(df):,}")
    print(f"Outside points: {len(point_report):,}")
    print(f"Outside tracks: {len(track_summary):,}")

    point_path = out_dir / "out_of_petri_points.csv"
    track_path = out_dir / "out_of_petri_tracks_summary.csv"
    point_report.to_csv(point_path, index=False)
    track_summary.to_csv(track_path, index=False)
    print(f"Wrote {point_path}")
    print(f"Wrote {track_path}")

    if track_summary.empty:
        print("No out-of-dish tracks found.")
        return

    print("\nTrack summary:")
    print(track_summary.to_string(index=False))

    plot_overview_all_conditions(track_summary, out_dir / "out_of_petri_tracks_overview.png")

    for cond in sorted(track_summary["condition"].astype(str).unique()):
        slug = cond.replace("/", "_")
        cond_ids = track_summary.loc[track_summary["condition"].astype(str) == cond, "track_id"].astype(str).tolist()
        plot_condition_outside_tracks(
            df, cols, cfg, cond, cond_ids, out_dir / f"out_of_petri_tracks_{slug}.png", dish
        )

    if "5d_fed" in track_summary["condition"].astype(str).values:
        cond_ids = track_summary.loc[track_summary["condition"] == "5d_fed", "track_id"].astype(str).tolist()
        plot_condition_outside_tracks(
            df, cols, cfg, "5d_fed", cond_ids, out_dir / "out_of_petri_tracks_5d_Sated_for_Michael.png", dish
        )

    print(f"\nFigures saved under {out_dir.resolve()}")


if __name__ == "__main__":
    main()
