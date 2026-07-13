#!/usr/bin/env python3
"""Publication-ready KDE-only density movies and static summaries for starvation manuscript."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

from create_larvae_animations import (
    _run_compare_subplot_pipeline,
    apply_fixed_axis_xy_limits,
    compute_group_normalization_factors,
    compute_shared_density_vmaxs_for_conditions,
    config_from_module_globals,
    detect_columns,
    load_and_preprocess,
    prepare_frame_cache,
    setup_style,
    stabilized_limits,
)

# --- Adjustable defaults ---
# Publication outputs are always fed-vs-starved side-by-side (fed left, starved right).
COMPARE_PAIRS = (
    ("4d_fed_vs_starved", "4d_fed", "4d_starved"),
    ("5d_fed_vs_starved", "5d_fed", "5d_starved"),
)
ALL_CONDITIONS = tuple(
    cond for pair in COMPARE_PAIRS for cond in pair[1:]
)
GROUP_LABELS = {
    "4d_fed": "4d Fed",
    "4d_starved": "4d Starved",
    "5d_fed": "5d Fed",
    "5d_starved": "5d Starved",
}
ARTIFACT_TRACK_IDS = ("1637", "2937", "2354", "2366")

# KDE / grid: movies use stronger smoothing to avoid dotted single-larva artifacts.
KDE_SIGMA_BINS_MOVIE = 3.8
KDE_SIGMA_BINS_SUMMARY = 1.6
DENSITY_BINS_MOVIE = 64
DENSITY_BINS_SUMMARY = 96
HC_DENSITY_BINS_MOVIE = 48
HC_DENSITY_BINS_SUMMARY = 64

# Temporal windows (frames at 16 fps recording).
WINDOW_SIZE_FRAMES = 180
HC_WINDOW_SIZE_FRAMES = 360

# Color scale: global across conditions; sqrt norm + percentile clip for high-density contrast.
DENSITY_VMAX_PERCENTILE = 99.5
HC_DENSITY_VMAX_PERCENTILE = 99.0
DENSITY_USE_SQRT_NORM = True
DENSITY_CMAP = "cividis"
DENSITY_NORMALIZATION = "max_visible_larvae"

# Playback / movie length
FPS = 16
RECORDING_FPS = 16.0
FRAME_STRIDE = 1
DEFAULT_PLAYBACK_SPEEDS: Tuple[float, ...] = (1.0, 2.0)
# Full recording (~3 min at 16 fps); no test subset unless --test-duration-seconds is passed.
SUMMARY_DURATION_SECONDS = 10.0
SUMMARY_PAUSE_SECONDS = 0.25


@dataclass
class FinalizeKdeConfig:
    input_csv: Path = Path("starvation2-2026-04-23_analyzed.csv")
    output_dir: Path = Path("outputs/starvation_kde_final")
    playback_speeds: Tuple[float, ...] = DEFAULT_PLAYBACK_SPEEDS
    make_hc: bool = True
    test_duration_seconds: Optional[float] = None


def build_video_cfg(f: FinalizeKdeConfig):
    base = config_from_module_globals()
    return replace(
        base,
        data_path=f.input_csv,
        output_dir=f.output_dir,
        density_method="kde",
        kde_sigma_bins_movie=KDE_SIGMA_BINS_MOVIE,
        kde_sigma_bins_summary=KDE_SIGMA_BINS_SUMMARY,
        density_bins_movie=DENSITY_BINS_MOVIE,
        density_bins_summary=DENSITY_BINS_SUMMARY,
        hc_density_bins_movie=HC_DENSITY_BINS_MOVIE,
        hc_density_bins_summary=HC_DENSITY_BINS_SUMMARY,
        density_vmax_percentile=DENSITY_VMAX_PERCENTILE,
        hc_density_vmax_percentile=HC_DENSITY_VMAX_PERCENTILE,
        density_use_sqrt_norm=DENSITY_USE_SQRT_NORM,
        density_cmap=DENSITY_CMAP,
        density_normalization=DENSITY_NORMALIZATION,
        window_size_frames=WINDOW_SIZE_FRAMES,
        hc_window_size_frames=HC_WINDOW_SIZE_FRAMES,
        fps=FPS,
        recording_fps=RECORDING_FPS,
        frame_stride=FRAME_STRIDE,
        playback_speeds=f.playback_speeds,
        exclude_track_ids=list(ARTIFACT_TRACK_IDS),
        make_track_animation=False,
        make_density_animation=True,
        make_all_animations=f.make_hc,
        save_summary_image=True,
        group_condition_column="condition",
        group_display_labels=GROUP_LABELS,
        test_duration_seconds=f.test_duration_seconds,
        append_pooled_summary_at_end=True,
        append_summary_frame=True,
        summary_duration_seconds=SUMMARY_DURATION_SECONDS,
        summary_pause_seconds=SUMMARY_PAUSE_SECONDS,
    )


def compute_global_vmax(cfg, cols, df, conditions):
    xlim = stabilized_limits(float(df[cols.x].min()), float(df[cols.x].max()))
    ylim = stabilized_limits(float(df[cols.y].min()), float(df[cols.y].max()))
    xlim, ylim = apply_fixed_axis_xy_limits(xlim, ylim, cfg.fixed_axis_xy_limits)
    unique_frames, _ = prepare_frame_cache(df, cols.frame, cols.x, cols.y)
    norm_factors = compute_group_normalization_factors(
        df, cols, cfg.group_condition_column, list(conditions), cfg.density_normalization
    )
    return compute_shared_density_vmaxs_for_conditions(
        cfg,
        cols,
        df,
        cfg.group_condition_column,
        list(conditions),
        xlim,
        ylim,
        unique_frames,
        norm_factors=norm_factors,
    )


def write_methods_draft(out_dir: Path, gv: dict, playback_speeds: Tuple[float, ...]) -> None:
    if playback_speeds == (1.0, 2.0):
        speed_desc = "real-time (1x) and 2x"
    else:
        speed_desc = ", ".join(f"{s:g}x" for s in playback_speeds)
    window_sec = WINDOW_SIZE_FRAMES / RECORDING_FPS
    animals_vmax = gv.get("animals", "n/a")
    hc_vmax = gv.get("hc", "n/a")
    excluded = ", ".join(ARTIFACT_TRACK_IDS)
    text = f"""# Methods and legend draft (KDE density outputs)

## Trajectory density maps

Larval centroid trajectories were summarized as time-resolved two-dimensional kernel
density estimate (KDE) maps in the Petri-dish arena (fixed axes +/-47.5 mm). Raw
spatial histograms were not used for final figures; all publication outputs use
Gaussian KDE smoothing of per-frame occupancy.

For movies, KDE bandwidth was sigma = {KDE_SIGMA_BINS_MOVIE} grid bins and the
render grid was {DENSITY_BINS_MOVIE} x {DENSITY_BINS_MOVIE} bins, with a sliding
temporal window of {WINDOW_SIZE_FRAMES} frames (~{window_sec:.1f} s at
{RECORDING_FPS:g} fps recording). Stronger smoothing avoids bright single-animal dots
that can appear with fine bins or weak smoothing.

For static summary panels (full 3-min recording), finer resolution was used:
sigma = {KDE_SIGMA_BINS_SUMMARY} bins on a {DENSITY_BINS_SUMMARY} x {DENSITY_BINS_SUMMARY} grid.

Head-cast (HC) event density (when shown) used window {HC_WINDOW_SIZE_FRAMES} frames and
movie/summary grids {HC_DENSITY_BINS_MOVIE}/{HC_DENSITY_BINS_SUMMARY} bins.

Density was normalized by the maximum number of larvae co-visible in any frame
({DENSITY_NORMALIZATION}) so maps are comparable across conditions.

## Color scale

Maps use the perceptually uniform {DENSITY_CMAP} colormap with global color limits
shared across all conditions (larvae vmax = {animals_vmax}; HC vmax = {hc_vmax}).
Values above the {DENSITY_VMAX_PERCENTILE}th percentile ({HC_DENSITY_VMAX_PERCENTILE}th for HC) were
clipped before setting vmax. Larval maps use sqrt (power 0.5) normalization for better
contrast among high-density regions.

## Movies

Animations play at {FPS} fps output ({speed_desc}). Filenames include `_realtime` or `_2x`.
Each movie shows the full ~3 min recording, then holds the pooled full-recording
summary panel(s) for {SUMMARY_DURATION_SECONDS:.0f} s at the end.

## Legend suggestion

Time-resolved KDE density of larval positions (and optionally head-cast events) in the
arena; warmer colors indicate higher occupancy. Same color scale across panels/conditions.

## Exclusions

Tracks {excluded} (out-of-arena artifacts in 5d Fed) were excluded.

## Comparisons

All publication movies and summary panels are side-by-side fed vs starved
(4d and 5d); fed is always the left panel, starved the right panel.
"""
    (out_dir / "methods_and_legend_draft.md").write_text(text, encoding="utf-8")


def run_finalize(f: FinalizeKdeConfig) -> None:
    f.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    cfg = build_video_cfg(f)
    cols = detect_columns(cfg.data_path, cfg)
    df = load_and_preprocess(cfg.data_path, cols, cfg, skip_group_value_filter=True)
    conds = [c for c in ALL_CONDITIONS if c in set(df[cfg.group_condition_column].astype(str))]
    if not conds:
        raise ValueError("No expected condition values found in input CSV.")

    gv = compute_global_vmax(cfg, cols, df, conds)
    print("Global density vmax (shared color scale):", gv, flush=True)

    for slug, fed, starved in COMPARE_PAIRS:
        if fed not in conds or starved not in conds:
            raise ValueError(
                f"Missing conditions for {slug!r}: need {fed!r} and {starved!r}; have {conds}"
            )
        out = f.output_dir / slug
        out.mkdir(parents=True, exist_ok=True)
        cc = replace(
            cfg,
            output_dir=out,
            compare_conditions_subplots=True,
            selected_groups=[fed, starved],
            compare_output_slug=slug,
            global_density_vmax=gv,
        )
        print(f"--- fed vs starved KDE: {fed} vs {starved} -> {out} ---", flush=True)
        _run_compare_subplot_pipeline(cc, cols)

    manifest = {
        "kde_only": True,
        "histogram_outputs": False,
        "global_vmax": gv,
        "kde_sigma_bins_movie": KDE_SIGMA_BINS_MOVIE,
        "kde_sigma_bins_summary": KDE_SIGMA_BINS_SUMMARY,
        "density_bins_movie": DENSITY_BINS_MOVIE,
        "density_bins_summary": DENSITY_BINS_SUMMARY,
        "density_vmax_percentile": DENSITY_VMAX_PERCENTILE,
        "hc_density_vmax_percentile": HC_DENSITY_VMAX_PERCENTILE,
        "density_use_sqrt_norm": DENSITY_USE_SQRT_NORM,
        "density_cmap": DENSITY_CMAP,
        "density_normalization": DENSITY_NORMALIZATION,
        "window_size_frames": WINDOW_SIZE_FRAMES,
        "summary_duration_seconds": SUMMARY_DURATION_SECONDS,
        "append_pooled_summary_at_end": True,
        "test_duration_seconds": f.test_duration_seconds,
        "playback_speeds": list(f.playback_speeds),
        "excluded_track_ids": list(ARTIFACT_TRACK_IDS),
        "conditions": conds,
        "compare_pairs": [
            {"slug": slug, "fed": fed, "starved": starved}
            for slug, fed, starved in COMPARE_PAIRS
        ],
    }
    (f.output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_methods_draft(f.output_dir, gv, f.playback_speeds)
    print(f"Done. Outputs in {f.output_dir.resolve()}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="KDE-only final density movies and summaries for starvation manuscript."
    )
    p.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=FinalizeKdeConfig.input_csv,
        help="Input tracking CSV",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=FinalizeKdeConfig.output_dir,
        help="Output root directory",
    )
    p.add_argument(
        "--test-duration-seconds",
        type=float,
        default=None,
        help="Limit animation to first N seconds (quick test)",
    )
    p.add_argument("--no-2x", action="store_true", help="Skip 2x speed movies")
    p.add_argument("--no-hc", action="store_true", help="Skip head-cast density outputs")
    args = p.parse_args()
    speeds: Tuple[float, ...] = (1.0,) if args.no_2x else DEFAULT_PLAYBACK_SPEEDS
    cfg = FinalizeKdeConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        playback_speeds=speeds,
        make_hc=not args.no_hc,
        test_duration_seconds=args.test_duration_seconds,
    )
    run_finalize(cfg)


if __name__ == "__main__":
    main()
