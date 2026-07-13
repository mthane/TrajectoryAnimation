#!/usr/bin/env python3
"""Publication-ready KDE density + trajectory movies and static summaries for starvation manuscript."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Tuple

from create_larvae_animations import (
    _run_compare_subplot_pipeline,
    _run_pipeline_core,
    apply_exclude_track_ids,
    apply_fixed_axis_xy_limits,
    compute_group_normalization_factors,
    compute_shared_density_vmaxs_for_conditions,
    compute_summary_vmax_from_conditions,
    config_from_module_globals,
    detect_columns,
    load_and_preprocess,
    maybe_detect_and_filter_artifacts,
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

# KDE / grid: movies use coarser bins + stronger smoothing; summaries use finer bins.
KDE_SIGMA_BINS_MOVIE = 3.8
KDE_SIGMA_BINS_SUMMARY = 1.6
DENSITY_BINS_SUMMARY = 96
DENSITY_BINS_MOVIE = 56
HC_DENSITY_BINS_MOVIE = 40
HC_DENSITY_BINS_SUMMARY = 64

# Temporal windows (frames at 16 fps recording).
WINDOW_SIZE_FRAMES = 180
HC_WINDOW_SIZE_FRAMES = 360
TRACK_HISTORY_SECONDS = 10.0

# Color scale: linear viridis; movies use high percentile clip, summaries use tighter clip.
DENSITY_VMAX_PERCENTILE = 99.5
HC_DENSITY_VMAX_PERCENTILE = 99.0
DENSITY_VMAX_PERCENTILE_SUMMARY = 93.0
HC_DENSITY_VMAX_PERCENTILE_SUMMARY = 90.0
DENSITY_USE_SQRT_NORM = False
HC_DENSITY_USE_LOG_NORM = False
DENSITY_CMAP = "viridis"
DENSITY_NORMALIZATION = "max_visible_larvae"

# Playback / movie length
FPS = 16
RECORDING_FPS = 16.0
FRAME_STRIDE = 1
DEFAULT_PLAYBACK_SPEEDS: Tuple[float, ...] = (1.0, 2.0)
SUMMARY_DURATION_SECONDS = 10.0
SUMMARY_PAUSE_SECONDS = 0.25


@dataclass
class FinalizeKdeConfig:
    input_csv: Path = Path("data/starvation2-2026-04-23_analyzed.csv")
    output_dir: Path = Path("outputs/starvation_kde_final")
    playback_speeds: Tuple[float, ...] = DEFAULT_PLAYBACK_SPEEDS
    make_hc: bool = True
    make_per_condition: bool = True
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
        density_vmax_percentile_summary=DENSITY_VMAX_PERCENTILE_SUMMARY,
        hc_density_vmax_percentile_summary=HC_DENSITY_VMAX_PERCENTILE_SUMMARY,
        density_use_sqrt_norm=DENSITY_USE_SQRT_NORM,
        hc_density_use_log_norm=HC_DENSITY_USE_LOG_NORM,
        density_cmap=DENSITY_CMAP,
        density_normalization=DENSITY_NORMALIZATION,
        window_size_frames=WINDOW_SIZE_FRAMES,
        hc_window_size_frames=HC_WINDOW_SIZE_FRAMES,
        track_history_seconds=TRACK_HISTORY_SECONDS,
        fps=FPS,
        recording_fps=RECORDING_FPS,
        frame_stride=FRAME_STRIDE,
        playback_speeds=f.playback_speeds,
        exclude_track_ids=list(ARTIFACT_TRACK_IDS),
        make_track_animation=True,
        make_track_uniform_color_video=False,
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


def _prepare_df_slice(df, cols, cfg, cond: str):
    sub = df[df[cfg.group_condition_column].astype(str) == str(cond)].copy()
    xlim = stabilized_limits(float(sub[cols.x].min()), float(sub[cols.x].max()))
    ylim = stabilized_limits(float(sub[cols.y].min()), float(sub[cols.y].max()))
    xlim, ylim = apply_fixed_axis_xy_limits(xlim, ylim, cfg.fixed_axis_xy_limits)
    sub = apply_exclude_track_ids(sub, cfg, context_label=cond)
    sub, _ = maybe_detect_and_filter_artifacts(
        sub, cols, cfg, output_dir=cfg.output_dir, context_label=cond, xlim=xlim, ylim=ylim
    )
    return sub


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


def compute_global_summary_vmax(cfg, cols, df, conditions):
    """Tighter color limits for static summary panels (full recording, finer grid)."""
    xlim = stabilized_limits(float(df[cols.x].min()), float(df[cols.x].max()))
    ylim = stabilized_limits(float(df[cols.y].min()), float(df[cols.y].max()))
    xlim, ylim = apply_fixed_axis_xy_limits(xlim, ylim, cfg.fixed_axis_xy_limits)
    norm_factors = compute_group_normalization_factors(
        df, cols, cfg.group_condition_column, list(conditions), cfg.density_normalization
    )
    frame_to_points_by_cond = {}
    for cond in conditions:
        sub = _prepare_df_slice(df, cols, cfg, cond)
        _, fp = prepare_frame_cache(sub, cols.frame, cols.x, cols.y)
        frame_to_points_by_cond[str(cond)] = fp
    out = {
        "animals": compute_summary_vmax_from_conditions(
            frame_to_points_by_cond,
            xlim,
            ylim,
            cfg,
            hc_mode=False,
            norm_factors=norm_factors,
        ),
    }
    if cfg.make_all_animations:
        try:
            from create_larvae_animations import filter_hc_rows, resolve_hc_column

            hc_col = resolve_hc_column(df.columns.tolist(), cfg)
            hc_by_cond = {}
            for cond in conditions:
                sub = _prepare_df_slice(df, cols, cfg, cond)
                df_h = filter_hc_rows(sub, hc_col, cfg)
                _, fp_h = prepare_frame_cache(df_h, cols.frame, cols.x, cols.y)
                hc_by_cond[str(cond)] = fp_h
            out["hc"] = compute_summary_vmax_from_conditions(
                hc_by_cond,
                xlim,
                ylim,
                cfg,
                hc_mode=True,
                norm_factors=norm_factors,
            )
        except ValueError:
            pass
    return out


def write_methods_draft(
    out_dir: Path,
    gv_movie: dict,
    gv_summary: dict,
    playback_speeds: Tuple[float, ...],
) -> None:
    if playback_speeds == (1.0, 2.0):
        speed_desc = "real-time (1x) and 2x"
    else:
        speed_desc = ", ".join(f"{s:g}x" for s in playback_speeds)
    window_sec = WINDOW_SIZE_FRAMES / RECORDING_FPS
    hc_window_sec = HC_WINDOW_SIZE_FRAMES / RECORDING_FPS
    animals_vmax_movie = gv_movie.get("animals", "n/a")
    hc_vmax_movie = gv_movie.get("hc", "n/a")
    animals_vmax_summary = gv_summary.get("animals", "n/a")
    hc_vmax_summary = gv_summary.get("hc", "n/a")
    excluded = ", ".join(ARTIFACT_TRACK_IDS)
    norm_desc = (
        "square-root (power 0.5) normalization"
        if DENSITY_USE_SQRT_NORM
        else "linear normalization"
    )
    hc_norm_desc = (
        "logarithmic normalization"
        if HC_DENSITY_USE_LOG_NORM
        else "linear normalization"
    )
    text = f"""# Methods and legend draft (KDE density + trajectory outputs)

## Trajectory density maps

Larval centroid trajectories were summarized as time-resolved two-dimensional kernel
density estimate (KDE) maps in the Petri-dish arena (fixed axes +/-47.5 mm, dish radius
45 mm). Raw spatial histograms were not used for final figures; all publication outputs use
Gaussian KDE smoothing of per-frame occupancy.

**Movies** used KDE bandwidth sigma = {KDE_SIGMA_BINS_MOVIE} grid bins on a
{DENSITY_BINS_MOVIE} x {DENSITY_BINS_MOVIE} bin grid (coarser cells for stable playback),
with a sliding temporal window of {WINDOW_SIZE_FRAMES} frames (~{window_sec:.1f} s at
{RECORDING_FPS:g} fps recording). Stronger smoothing avoids bright single-animal dots that
can appear with fine bins or weak smoothing.

**Static summary panels** (full ~3 min recording stacked into one map) used finer resolution:
sigma = {KDE_SIGMA_BINS_SUMMARY} bins on a {DENSITY_BINS_SUMMARY} x {DENSITY_BINS_SUMMARY} grid.

Head-cast (HC) event density (when shown) used a sliding window of {HC_WINDOW_SIZE_FRAMES}
frames (~{hc_window_sec:.1f} s) and movie/summary grids of
{HC_DENSITY_BINS_MOVIE} / {HC_DENSITY_BINS_SUMMARY} bins.

Density was normalized by the maximum number of larvae co-visible in any frame
(`{DENSITY_NORMALIZATION}`) so maps are comparable across conditions.

## Color scale

Maps use the perceptually uniform **{DENSITY_CMAP}** colormap with **linear** intensity mapping
(no sqrt or log transform). Larval maps use {norm_desc}; HC maps use {hc_norm_desc}.

**Movies** share a global color scale across all conditions, with vmax set from the
{DENSITY_VMAX_PERCENTILE}th percentile of sampled frame peaks (larvae movie vmax =
{animals_vmax_movie}; HC movie vmax = {hc_vmax_movie}; HC percentile =
{HC_DENSITY_VMAX_PERCENTILE}).

**Static summary PNG/PDF panels** use a **tighter** color scale to highlight spatial
differences: vmax from the {DENSITY_VMAX_PERCENTILE_SUMMARY}th percentile of full-recording
KDE peaks (larvae summary vmax = {animals_vmax_summary}; HC summary vmax =
{hc_vmax_summary}; HC summary percentile = {HC_DENSITY_VMAX_PERCENTILE_SUMMARY}).

## Trajectory (line) animations

Cumulative larval paths were rendered as line/scatter animations with a trailing history of
the last {TRACK_HISTORY_SECONDS:g} s per larva (individual colors per track ID). Compare
movies show fed vs starved side-by-side (fed left, starved right) with the same layout as
density compare movies.

## Movies

Animations play at {FPS} fps output ({speed_desc}). Filenames include `_realtime` or `_2x`.
Each density movie shows the full ~3 min recording, then holds the pooled full-recording
summary panel(s) for {SUMMARY_DURATION_SECONDS:.0f} s at the end (summary panels use the
tighter color limits described above).

## Legend suggestion

Time-resolved KDE density of larval positions (and optionally head-cast events) in the
arena; color intensity indicates normalized occupancy (linear {DENSITY_CMAP} scale).
Trajectory animations show recent movement paths colored by individual larva.
Same movie color scale across panels/conditions; summary figures use a tighter scale.

## Exclusions

Tracks {excluded} (out-of-arena artifacts in 5d Fed) were excluded.

## Comparisons

All publication compare movies and summary panels are side-by-side fed vs starved
(4d and 5d); fed is always the left panel, starved the right panel.

Per-condition outputs (density + trajectories) are saved under `per_condition/<condition>/`.
"""
    (out_dir / "methods_and_legend_draft.md").write_text(text, encoding="utf-8")


def run_per_condition(cfg, cols, df, cond: str, gv: dict) -> None:
    out = cfg.output_dir / "per_condition" / cond
    out.mkdir(parents=True, exist_ok=True)
    sub = _prepare_df_slice(df, cols, cfg, cond)
    if len(sub) == 0:
        print(f"Skipping empty condition {cond!r}", flush=True)
        return
    stem = f"starvation_{cond}_kde"
    cc = replace(
        cfg,
        output_dir=out,
        group_condition_value=cond,
        density_movie_stem=stem,
        global_density_vmax=gv,
        compare_conditions_subplots=False,
        make_track_animation=True,
    )
    print(f"--- per-condition: {cond} -> {out} ---", flush=True)
    _run_pipeline_core(cc, cols, sub, context_label=cond, shared_density_vmax=gv)


def run_finalize(f: FinalizeKdeConfig) -> None:
    f.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    cfg = build_video_cfg(f)
    cols = detect_columns(cfg.data_path, cfg)
    df = load_and_preprocess(cfg.data_path, cols, cfg, skip_group_value_filter=True)
    conds = [c for c in ALL_CONDITIONS if c in set(df[cfg.group_condition_column].astype(str))]
    if not conds:
        raise ValueError("No expected condition values found in input CSV.")

    gv_movie = compute_global_vmax(cfg, cols, df, conds)
    gv_summary = compute_global_summary_vmax(cfg, cols, df, conds)
    print("Global density vmax (movies, shared color scale):", gv_movie, flush=True)
    print("Global density vmax (summaries, tighter scale):", gv_summary, flush=True)

    if f.make_per_condition:
        for cond in conds:
            run_per_condition(cfg, cols, df, cond, gv_movie)

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
            global_density_vmax=gv_movie,
        )
        print(f"--- fed vs starved KDE + tracks: {fed} vs {starved} -> {out} ---", flush=True)
        _run_compare_subplot_pipeline(cc, cols)

    manifest = {
        "kde_only": True,
        "histogram_outputs": False,
        "global_vmax_movie": gv_movie,
        "global_vmax_summary": gv_summary,
        "kde_sigma_bins_movie": KDE_SIGMA_BINS_MOVIE,
        "kde_sigma_bins_summary": KDE_SIGMA_BINS_SUMMARY,
        "density_bins_movie": DENSITY_BINS_MOVIE,
        "density_bins_summary": DENSITY_BINS_SUMMARY,
        "hc_density_bins_movie": HC_DENSITY_BINS_MOVIE,
        "hc_density_bins_summary": HC_DENSITY_BINS_SUMMARY,
        "density_vmax_percentile_movie": DENSITY_VMAX_PERCENTILE,
        "hc_density_vmax_percentile_movie": HC_DENSITY_VMAX_PERCENTILE,
        "density_vmax_percentile_summary": DENSITY_VMAX_PERCENTILE_SUMMARY,
        "hc_density_vmax_percentile_summary": HC_DENSITY_VMAX_PERCENTILE_SUMMARY,
        "density_use_sqrt_norm": DENSITY_USE_SQRT_NORM,
        "hc_density_use_log_norm": HC_DENSITY_USE_LOG_NORM,
        "density_cmap": DENSITY_CMAP,
        "density_normalization": DENSITY_NORMALIZATION,
        "window_size_frames": WINDOW_SIZE_FRAMES,
        "hc_window_size_frames": HC_WINDOW_SIZE_FRAMES,
        "track_history_seconds": TRACK_HISTORY_SECONDS,
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
    write_methods_draft(f.output_dir, gv_movie, gv_summary, f.playback_speeds)
    print(f"Done. Outputs in {f.output_dir.resolve()}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="KDE final density + trajectory movies and summaries for starvation manuscript."
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
    p.add_argument(
        "--no-per-condition",
        action="store_true",
        help="Skip per-condition density + track outputs",
    )
    args = p.parse_args()
    speeds: Tuple[float, ...] = (1.0,) if args.no_2x else DEFAULT_PLAYBACK_SPEEDS
    cfg = FinalizeKdeConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        playback_speeds=speeds,
        make_hc=not args.no_hc,
        make_per_condition=not args.no_per_condition,
        test_duration_seconds=args.test_duration_seconds,
    )
    run_finalize(cfg)


if __name__ == "__main__":
    main()
