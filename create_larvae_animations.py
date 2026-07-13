#!/usr/bin/env python3
"""Create larval trajectory animations from tracking data.

Outputs (defaults):
  - animations/larvae_tracks_window.mp4
  - animations/larvae_tracks_uniform_color.mp4 (all larvae use TRACK_COLOR; omitted if main mode is already "same")
  - animations/larvae_density.mp4
Optional summary PNGs when SAVE_SUMMARY_IMAGE is True (default: on).

New parameters (see VideoGenConfig / argparse --help):
  - Track history window in seconds, track color mode (same vs per-ID).
  - Density mode: all positions vs HC-only frames (with configurable HC column/value).
  - Optional full-video summary frame appended to the MP4 + saved as PNG.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import warnings
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Set, Tuple

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import cv2


# =========================
# Config (edit as needed)
# =========================
# Default CSV when you run `python create_larvae_animations.py` with no path.
# Prefer: `python create_larvae_animations.py path/to/any_tracking_file.csv` or `--data`.
DATA_PATH = Path("embedding_dan.csv")
OUTPUT_DIR = Path("animations")

# Optional fixed column names. Set to None for auto-detection.
X_COLUMN_OVERRIDE = None  # Example: "centroid_x"
Y_COLUMN_OVERRIDE = None  # Example: "centroid_y"
FRAME_COLUMN_OVERRIDE = None  # Example: "frame"
ID_COLUMN_OVERRIDE = None  # Example: "id"

# Time column: if set, used for track-history filtering (seconds). Else derived from frame column.
TIMESTAMP_COLUMN_OVERRIDE = None  # Example: "time_sec"
# If False, frame column values are treated as frame indices and divided by RECORDING_FPS for seconds.
# If True, frame column values are already in seconds (ignored when TIMESTAMP_COLUMN_OVERRIDE is set).
FRAME_VALUES_ARE_SECONDS = False

# Optional filter by group condition/category.
GROUP_CONDITION_COLUMN = "condition"
GROUP_CONDITION_VALUE = None  # Example: "EM_Rewarded"
# If True: load once, then run the full pipeline for each distinct value in GROUP_CONDITION_COLUMN
# (outputs under OUTPUT_DIR/<safe_condition_name>/). Implies the condition column is read from CSV.
RUN_EACH_CONDITION = False
# One video with subplots (one panel per condition) + optional pooled summary at end (all data).
COMPARE_CONDITIONS_SUBPLOTS = False
# After compare animation: hold last multi-panel frame, then show full-recording summary (same subplot layout).
APPEND_POOLED_SUMMARY_AT_END = True

WINDOW_SIZE_FRAMES = 50
# Optional separate sliding window for HC density (frames with data). None = use WINDOW_SIZE_FRAMES.
HC_WINDOW_SIZE_FRAMES: Optional[int] = None
FRAME_STRIDE = 1
# Recording frame rate (frames per second) and output video FPS.
RECORDING_FPS = 16
FPS = 16
DPI = 140
FIGSIZE = (8, 8)
# None = use entire recording; set e.g. 30 to preview only the first N seconds while iterating.
TEST_DURATION_SECONDS: Optional[float] = None

MAKE_TRACK_ANIMATION = True
# If True (and main track animation runs), also write larvae_tracks_uniform_color.mp4 (all larvae TRACK_COLOR).
MAKE_TRACK_UNIFORM_COLOR_VIDEO = True

MAKE_DENSITY_ANIMATION = True
# If True: one run produces track video + all-frame density + HC density (ignores DENSITY_MODE for which densities to build).
MAKE_ALL_ANIMATIONS = False

DENSITY_BINS = 90
# Separate grid resolution for HC density (sparse data); None = max(24, DENSITY_BINS // 2).
HC_DENSITY_BINS: Optional[int] = None
# HC maps are sparse: use log color scale and percentile vmax (not a fixed 1.0 floor).
HC_DENSITY_VMAX_PERCENTILE: float = 98.0
HC_DENSITY_USE_LOG_NORM: bool = True
# Larval density color-scale percentile (<100 clips hot outliers; HC uses HC_DENSITY_VMAX_PERCENTILE).
DENSITY_VMAX_PERCENTILE: float = 100.0
DENSITY_USE_SQRT_NORM: bool = False
DENSITY_CMAP = "magma"
# Optional separate KDE/grid settings for movies vs static summary figures.
KDE_SIGMA_BINS_MOVIE: Optional[float] = None
KDE_SIGMA_BINS_SUMMARY: Optional[float] = None
DENSITY_BINS_MOVIE: Optional[int] = None
DENSITY_BINS_SUMMARY: Optional[int] = None
HC_DENSITY_BINS_MOVIE: Optional[int] = None
HC_DENSITY_BINS_SUMMARY: Optional[int] = None
# Density maps: no-data bins inside the dish; axes/margins stay light
DENSITY_NODATA_COLOR = "black"
DENSITY_AXES_FACE_COLOR = "white"
DENSITY_PETRI_EDGE_COLOR = "#2ca02c"  # green outline on density plots
TRACK_CMAP = "tab20"
# Per-track colors: hash-based random HSV (same larva ID → same color; independent of ID sort order).
TRACK_COLOR_SEED = 42
DENSITY_VMAX_SAMPLE_FRAMES = 1200
# "hist" for raw 2D histogram, "kde" for smoothed KDE-like map.
DENSITY_METHOD = "kde"
# Approximate KDE bandwidth in histogram-bin units (used when DENSITY_METHOD="kde").
KDE_SIGMA_BINS = 1.8

# CLI ``--fast`` applies these overrides (then any explicit flags win). Tuned for quicker previews.
FAST_PRESET_OVERRIDES: Dict[str, object] = {
    "frame_stride": 4,
    "fps": 8,
    "density_bins": 48,
    "hc_density_bins": 32,
    "density_vmax_sample_frames": 400,
    "density_method": "hist",
    "figsize": (6.0, 6.0),
    "dpi": 110,
}
DENSITY_MODE: Literal["animals", "all", "hc"] = "animals"

# HC detection: auto-pick a column from HC_COLUMN_CANDIDATES if HC_FILTER_COLUMN is None.
# "HCs" is tried first (common head-cast column name). Override if your file differs.
HC_FILTER_COLUMN: Optional[str] = None
HC_FILTER_VALUE: Optional[str] = None  # Match as string; None triggers boolean/numeric heuristics (see filter_hc_rows)

INFER_PETRI_DISH = True
# Petri dish outline radius in the same units as x/y (fixed; not inferred from data extent).
PETRI_DISH_RADIUS = 45.0
# Exact x and y axis limits (same units as positions). None = use data min/max (after stabilized_limits).
FIXED_AXIS_XY_LIMITS: Optional[Tuple[float, float]] = (-47.5, 47.5)

# Density normalization: divide raw spatial density by a fixed per-group factor (animals/tracks, not HC events).
# "none" | "num_tracks" | "max_visible_larvae" (max larvae co-visible in any frame in the group).
DENSITY_NORMALIZATION: Literal["none", "num_tracks", "max_visible_larvae"] = "none"
# Optional subset of group IDs for compare mode (None = all distinct values in the condition column).
SELECTED_GROUPS: Optional[List[str]] = None
# Map internal condition IDs to display strings (titles, legends, filenames when slug is set).
GROUP_DISPLAY_LABELS: Dict[str, str] = {}
# Optional suffix for compare output filenames, e.g. "4d_sated_vs_starved".
COMPARE_OUTPUT_SLUG: Optional[str] = None

# Track rendering
# None = cumulative over full video so far; float > 0 = only points with time in [t - window, t].
TRACK_HISTORY_SECONDS: Optional[float] = None
# "same": TRACK_COLOR for every animal; "individual": stable color per larva_id (TRACK_CMAP).
TRACK_COLOR_MODE: Literal["same", "individual"] = "individual"
TRACK_COLOR = "#1f77b4"

# Summary frame / tail of video (defaults preserve legacy one-shot animations).
APPEND_SUMMARY_FRAME = False
SUMMARY_PAUSE_SECONDS = 0.5
SUMMARY_DURATION_SECONDS = 5.0
SAVE_SUMMARY_IMAGE = True

# Artifact diagnostics / optional filtering
DETECT_ARTIFACTS = False
ARTIFACT_VELOCITY_THRESHOLD: Optional[float] = None
ARTIFACT_DISPLACEMENT_THRESHOLD: Optional[float] = None
ARTIFACT_MAD_MULTIPLIER = 10.0
EXCLUDE_ARTIFACT_TRACKS = False
# Explicit larva/track IDs to drop from all rendered outputs (e.g. out-of-Petri artifacts).
EXCLUDE_TRACK_IDS: Optional[List[str]] = None


HC_COLUMN_CANDIDATES: Tuple[str, ...] = (
    "HCs",
    "is_hc",
    "HC",
    "hc",
    "is_HC",
    "head_cast",
    "HeadCast",
    "behavior",
    "motif",
    "cluster_label",
    "label",
    "state",
)


@dataclass
class VideoGenConfig:
    """Runtime configuration; built from module globals and/or argparse."""

    data_path: Path = field(default_factory=lambda: DATA_PATH)
    output_dir: Path = field(default_factory=lambda: OUTPUT_DIR)
    x_column_override: Optional[str] = None
    y_column_override: Optional[str] = None
    frame_column_override: Optional[str] = None
    id_column_override: Optional[str] = None
    # Time: optional explicit timestamp column (seconds); else derived from frame column.
    timestamp_column_override: Optional[str] = None
    frame_values_are_seconds: bool = False
    group_condition_column: str = "condition"
    group_condition_value: Optional[str] = None
    # One full pipeline run per distinct value in group_condition_column (subfolders under output_dir).
    run_each_condition: bool = False
    # Multi-condition comparison: single MP4 with one subplot per condition (mutually exclusive with run_each in practice).
    compare_conditions_subplots: bool = False
    # Append full-recording summary after compare animation (multi-panel, same layout as compare video).
    append_pooled_summary_at_end: bool = True
    window_size_frames: int = 50
    hc_window_size_frames: Optional[int] = None
    frame_stride: int = 1
    recording_fps: float = 16.0
    fps: int = 16
    dpi: int = 140
    figsize: Tuple[float, float] = (8.0, 8.0)
    test_duration_seconds: Optional[float] = None
    make_track_animation: bool = True
    make_density_animation: bool = True
    # Second track MP4 with track_color_mode forced to "same" (skipped if already "same").
    make_track_uniform_color_video: bool = True
    # When True, render track (if enabled) plus both density outputs: animals + HC.
    make_all_animations: bool = False
    density_bins: int = 90
    hc_density_bins: Optional[int] = None
    hc_density_vmax_percentile: float = 98.0
    hc_density_vmax_percentile_summary: Optional[float] = None
    hc_density_use_log_norm: bool = True
    density_vmax_percentile: float = 100.0
    density_vmax_percentile_summary: Optional[float] = None
    density_use_sqrt_norm: bool = False
    density_cmap: str = "magma"
    track_cmap: str = "tab20"
    density_vmax_sample_frames: int = 1200
    density_method: str = "kde"
    kde_sigma_bins: float = 1.8
    kde_sigma_bins_movie: Optional[float] = None
    kde_sigma_bins_summary: Optional[float] = None
    density_bins_movie: Optional[int] = None
    density_bins_summary: Optional[int] = None
    hc_density_bins_movie: Optional[int] = None
    hc_density_bins_summary: Optional[int] = None
    density_mode: str = "animals"
    hc_filter_column: Optional[str] = None
    hc_filter_value: Optional[str] = None
    infer_petri_dish: bool = True
    petri_dish_radius: float = 45.0
    # When not None, force both xlim and ylim to (min, max) for a fixed square arena.
    fixed_axis_xy_limits: Optional[Tuple[float, float]] = (-47.5, 47.5)
    track_history_seconds: Optional[float] = None
    density_normalization: str = "none"
    selected_groups: Optional[List[str]] = None
    group_display_labels: Dict[str, str] = field(default_factory=dict)
    compare_output_slug: Optional[str] = None
    track_color_mode: str = "individual"
    track_color: str = "#1f77b4"
    track_color_seed: int = 42
    track_color_random_hsv: bool = True
    append_summary_frame: bool = False
    summary_pause_seconds: float = 0.5
    summary_duration_seconds: float = 5.0
    save_summary_image: bool = True
    detect_artifacts: bool = False
    artifact_velocity_threshold: Optional[float] = None
    artifact_displacement_threshold: Optional[float] = None
    artifact_mad_multiplier: float = 10.0
    exclude_artifact_tracks: bool = False
    exclude_track_ids: Optional[List[str]] = None
    global_density_vmax: Optional[Dict[str, float]] = None
    playback_speeds: Tuple[float, ...] = (1.0,)
    density_movie_stem: Optional[str] = None


def config_from_module_globals() -> VideoGenConfig:
    return VideoGenConfig(
        data_path=DATA_PATH,
        output_dir=OUTPUT_DIR,
        x_column_override=X_COLUMN_OVERRIDE,
        y_column_override=Y_COLUMN_OVERRIDE,
        frame_column_override=FRAME_COLUMN_OVERRIDE,
        id_column_override=ID_COLUMN_OVERRIDE,
        timestamp_column_override=TIMESTAMP_COLUMN_OVERRIDE,
        frame_values_are_seconds=FRAME_VALUES_ARE_SECONDS,
        group_condition_column=GROUP_CONDITION_COLUMN,
        group_condition_value=GROUP_CONDITION_VALUE,
        run_each_condition=RUN_EACH_CONDITION,
        compare_conditions_subplots=COMPARE_CONDITIONS_SUBPLOTS,
        append_pooled_summary_at_end=APPEND_POOLED_SUMMARY_AT_END,
        window_size_frames=WINDOW_SIZE_FRAMES,
        hc_window_size_frames=HC_WINDOW_SIZE_FRAMES,
        frame_stride=FRAME_STRIDE,
        recording_fps=float(RECORDING_FPS),
        fps=FPS,
        dpi=DPI,
        figsize=FIGSIZE,
        test_duration_seconds=TEST_DURATION_SECONDS,
        make_track_animation=MAKE_TRACK_ANIMATION,
        make_density_animation=MAKE_DENSITY_ANIMATION,
        make_track_uniform_color_video=MAKE_TRACK_UNIFORM_COLOR_VIDEO,
        make_all_animations=MAKE_ALL_ANIMATIONS,
        density_bins=DENSITY_BINS,
        hc_density_bins=HC_DENSITY_BINS,
        hc_density_vmax_percentile=float(HC_DENSITY_VMAX_PERCENTILE),
        hc_density_use_log_norm=HC_DENSITY_USE_LOG_NORM,
        density_vmax_percentile=float(DENSITY_VMAX_PERCENTILE),
        density_use_sqrt_norm=DENSITY_USE_SQRT_NORM,
        density_cmap=DENSITY_CMAP,
        track_cmap=TRACK_CMAP,
        density_vmax_sample_frames=DENSITY_VMAX_SAMPLE_FRAMES,
        density_method=DENSITY_METHOD,
        kde_sigma_bins=KDE_SIGMA_BINS,
        kde_sigma_bins_movie=KDE_SIGMA_BINS_MOVIE,
        kde_sigma_bins_summary=KDE_SIGMA_BINS_SUMMARY,
        density_bins_movie=DENSITY_BINS_MOVIE,
        density_bins_summary=DENSITY_BINS_SUMMARY,
        hc_density_bins_movie=HC_DENSITY_BINS_MOVIE,
        hc_density_bins_summary=HC_DENSITY_BINS_SUMMARY,
        density_mode=DENSITY_MODE,
        hc_filter_column=HC_FILTER_COLUMN,
        hc_filter_value=HC_FILTER_VALUE,
        infer_petri_dish=INFER_PETRI_DISH,
        petri_dish_radius=float(PETRI_DISH_RADIUS),
        fixed_axis_xy_limits=FIXED_AXIS_XY_LIMITS,
        track_history_seconds=TRACK_HISTORY_SECONDS,
        density_normalization=DENSITY_NORMALIZATION,
        selected_groups=SELECTED_GROUPS,
        group_display_labels=dict(GROUP_DISPLAY_LABELS),
        compare_output_slug=COMPARE_OUTPUT_SLUG,
        track_color_mode=TRACK_COLOR_MODE,
        track_color=TRACK_COLOR,
        track_color_seed=TRACK_COLOR_SEED,
        track_color_random_hsv=True,
        append_summary_frame=APPEND_SUMMARY_FRAME,
        summary_pause_seconds=SUMMARY_PAUSE_SECONDS,
        summary_duration_seconds=SUMMARY_DURATION_SECONDS,
        save_summary_image=SAVE_SUMMARY_IMAGE,
        detect_artifacts=DETECT_ARTIFACTS,
        artifact_velocity_threshold=ARTIFACT_VELOCITY_THRESHOLD,
        artifact_displacement_threshold=ARTIFACT_DISPLACEMENT_THRESHOLD,
        artifact_mad_multiplier=ARTIFACT_MAD_MULTIPLIER,
        exclude_artifact_tracks=EXCLUDE_ARTIFACT_TRACKS,
        exclude_track_ids=list(EXCLUDE_TRACK_IDS) if EXCLUDE_TRACK_IDS else None,
        global_density_vmax=None,
        playback_speeds=(1.0,),
    )


@dataclass
class ColumnSelection:
    frame: str
    larva_id: str
    x: str
    y: str


@dataclass
class ArtifactDetectionResult:
    flagged_track_ids: Set[str]
    report_df: pd.DataFrame
    displacement_median: float
    displacement_mad: float
    velocity_median: float
    velocity_mad: float


def _pick_first_existing(candidates: Iterable[str], columns: List[str]) -> str | None:
    present = set(columns)
    for c in candidates:
        if c in present:
            return c
    return None


def detect_columns(csv_path: Path, cfg: VideoGenConfig) -> ColumnSelection:
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

    frame_col = cfg.frame_column_override or _pick_first_existing(
        ["frame", "t", "time", "timestamp", "frame_idx"], columns
    )
    id_col = cfg.id_column_override or _pick_first_existing(
        ["id", "track_id", "larva_id", "track", "animal_id"], columns
    )
    x_col = cfg.x_column_override or _pick_first_existing(
        ["centroid_x", "centoid_x", "spinepoint_x6_conv", "spinepoint_x_6_conv"],
        columns,
    )
    y_col = cfg.y_column_override or _pick_first_existing(
        ["centroid_y", "centoid_y", "spinepoint_y6_conv", "spinepoint_y_6_conv"],
        columns,
    )

    missing = []
    if frame_col is None:
        missing.append("frame/time column (e.g. frame, t, time)")
    if id_col is None:
        missing.append("larva ID column (e.g. id, track_id)")
    if x_col is None:
        missing.append(
            "x position column (tried: centroid_x, centoid_x, spinepoint_x6_conv, spinepoint_x_6_conv)"
        )
    if y_col is None:
        missing.append(
            "y position column (tried: centroid_y, centoid_y, spinepoint_y6_conv, spinepoint_y_6_conv)"
        )

    if missing:
        pretty_cols = ", ".join(columns)
        raise ValueError(
            "Required columns missing:\n"
            + "\n".join(f"- {m}" for m in missing)
            + f"\n\nAvailable columns:\n{pretty_cols}"
        )

    print("Detected columns:")
    print(f"  frame/time: {frame_col}")
    print(f"  larva ID:    {id_col}")
    print(f"  x centroid:  {x_col}")
    print(f"  y centroid:  {y_col}")

    return ColumnSelection(frame=frame_col, larva_id=id_col, x=x_col, y=y_col)


def attach_time_seconds(df: pd.DataFrame, cols: ColumnSelection, cfg: VideoGenConfig) -> pd.DataFrame:
    """Add _time_sec column for track-history and labeling."""
    out = df.copy()
    ts_col = cfg.timestamp_column_override
    if ts_col is not None:
        if ts_col not in out.columns:
            raise ValueError(
                f"TIMESTAMP_COLUMN_OVERRIDE='{ts_col}' not found in CSV. "
                "Unset it or fix the column name."
            )
        out["_time_sec"] = pd.to_numeric(out[ts_col], errors="coerce")
    else:
        fv = pd.to_numeric(out[cols.frame], errors="coerce")
        if cfg.frame_values_are_seconds:
            out["_time_sec"] = fv
        else:
            out["_time_sec"] = fv / float(cfg.recording_fps)
    return out


def resolve_hc_column(columns: List[str], cfg: VideoGenConfig) -> str:
    if cfg.hc_filter_column:
        if cfg.hc_filter_column not in columns:
            raise ValueError(
                f"HC_FILTER_COLUMN='{cfg.hc_filter_column}' not in CSV. "
                "Set HC_FILTER_COLUMN / HC_FILTER_VALUE to match your data."
            )
        return cfg.hc_filter_column
    found = _pick_first_existing(HC_COLUMN_CANDIDATES, columns)
    if found is None:
        raise ValueError(
            "Density mode 'hc' requires a head-cast / HC column. "
            "No known column was found. Set HC_FILTER_COLUMN (and usually HC_FILTER_VALUE) "
            f"to match your CSV. Tried: {', '.join(HC_COLUMN_CANDIDATES)}"
        )
    print(f"HC column auto-selected: {found} (override with HC_FILTER_COLUMN if wrong)")
    return found


def filter_hc_rows(df: pd.DataFrame, hc_col: str, cfg: VideoGenConfig) -> pd.DataFrame:
    """Keep rows classified as HC. Uses HC_FILTER_VALUE when provided."""
    s = df[hc_col]
    if cfg.hc_filter_value is not None:
        target = str(cfg.hc_filter_value).strip().lower()
        mask = s.astype(str).str.strip().str.lower() == target
        return df.loc[mask].copy()

    # Boolean / 0-1
    if s.dtype == bool:
        return df.loc[s].copy()
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().all() and set(np.unique(sn.dropna().astype(int))).issubset({0, 1}):
        return df.loc[sn > 0.5].copy()

    # String heuristics: common tokens
    sl = s.astype(str).str.strip().str.lower()
    mask = sl.isin(["hc", "hcs", "head_cast", "head cast", "true", "1", "yes"])
    if mask.any():
        return df.loc[mask].copy()

    warnings.warn(
        f"HC_FILTER_VALUE is not set and column '{hc_col}' is ambiguous. "
        "Set HC_FILTER_VALUE to the exact label that marks HC frames (e.g. 'HC').",
        UserWarning,
        stacklevel=2,
    )
    return df.iloc[0:0].copy()


def load_and_preprocess(
    csv_path: Path,
    cols: ColumnSelection,
    cfg: VideoGenConfig,
    *,
    skip_group_value_filter: bool = False,
) -> pd.DataFrame:
    """Load CSV. When ``skip_group_value_filter`` is True, keep all rows but still read the group column."""
    load_group_column = (
        cfg.group_condition_value is not None
        or cfg.run_each_condition
        or cfg.compare_conditions_subplots
        or skip_group_value_filter
    )
    apply_group_filter = cfg.group_condition_value is not None and not skip_group_value_filter

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [cols.frame, cols.larva_id, cols.x, cols.y]
    try:
        dm0 = normalize_density_mode(cfg.density_mode)
    except ValueError:
        dm0 = "animals"
    extra_for_hc = dm0 == "hc" or cfg.make_all_animations
    if extra_for_hc:
        hc_col_name = resolve_hc_column(header, cfg)
        if hc_col_name not in usecols:
            usecols.append(hc_col_name)

    if load_group_column:
        if cfg.group_condition_column not in header:
            raise ValueError(
                f"GROUP_CONDITION_COLUMN='{cfg.group_condition_column}' not found in CSV columns."
            )
        usecols.append(cfg.group_condition_column)

    ts_extra = cfg.timestamp_column_override
    if ts_extra and ts_extra not in usecols and ts_extra in header:
        usecols.append(ts_extra)

    df = pd.read_csv(csv_path, usecols=list(dict.fromkeys(usecols)), low_memory=False)

    if apply_group_filter:
        before_group = len(df)
        df = df[df[cfg.group_condition_column].astype(str) == str(cfg.group_condition_value)].copy()
        print(
            f"Group filter: {cfg.group_condition_column} == {cfg.group_condition_value}; "
            f"kept {len(df):,}/{before_group:,} rows."
        )

    df[cols.frame] = pd.to_numeric(df[cols.frame], errors="coerce")
    df[cols.x] = pd.to_numeric(df[cols.x], errors="coerce")
    df[cols.y] = pd.to_numeric(df[cols.y], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[cols.frame, cols.x, cols.y, cols.larva_id]).copy()
    after_drop = len(df)
    print(f"Rows loaded: {before:,}")
    print(f"Rows after dropping missing frame/id/x/y: {after_drop:,}")

    df["larva_id_norm"] = df[cols.larva_id].astype(str)
    df = df.sort_values(by=[cols.frame, "larva_id_norm"]).reset_index(drop=True)

    if cfg.test_duration_seconds is not None:
        if cfg.test_duration_seconds <= 0:
            raise ValueError("test_duration_seconds must be > 0 or None.")
        start_frame = float(df[cols.frame].min())
        end_frame = start_frame + (cfg.test_duration_seconds * cfg.recording_fps)
        before_window = len(df)
        df = df[df[cols.frame] <= end_frame].copy()
        print(
            "Testing subset enabled: "
            f"first {cfg.test_duration_seconds}s at {cfg.recording_fps} fps "
            f"(<= frame {end_frame:.0f}); kept {len(df):,}/{before_window:,} rows."
        )

    df = attach_time_seconds(df, cols, cfg)

    x_min, x_max = float(df[cols.x].min()), float(df[cols.x].max())
    y_min, y_max = float(df[cols.y].min()), float(df[cols.y].max())
    print(f"Global x limits: ({x_min:.3f}, {x_max:.3f})")
    print(f"Global y limits: ({y_min:.3f}, {y_max:.3f})")

    if np.isclose(x_min, x_max) and np.isclose(y_min, y_max):
        print(
            "WARNING: centroid coordinates are constant in this subset. "
            "Animation will appear static."
        )

    return df


def prepare_frame_cache(
    df: pd.DataFrame, frame_col: str, x_col: str, y_col: str
) -> Tuple[np.ndarray, Dict[float, np.ndarray]]:
    """Map frame key -> array columns [larva_id_norm, x, y, time_sec]."""
    frame_to_points: Dict[float, np.ndarray] = {}
    for frame_value, g in df.groupby(frame_col, sort=True):
        arr = np.column_stack(
            [
                g["larva_id_norm"].astype(str),
                g[x_col].astype(float),
                g[y_col].astype(float),
                g["_time_sec"].astype(float),
            ]
        )
        frame_to_points[float(frame_value)] = arr
    unique_frames = np.array(sorted(frame_to_points.keys()), dtype=float)
    return unique_frames, frame_to_points


def _robust_median_mad(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 0.0
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, mad


def detect_artifact_tracks(
    df: pd.DataFrame,
    cols: ColumnSelection,
    cfg: VideoGenConfig,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> ArtifactDetectionResult:
    """Detect suspicious tracks by large jumps, high velocity, and dish-boundary violations."""
    if len(df) == 0:
        return ArtifactDetectionResult(set(), pd.DataFrame(), 0.0, 0.0, 0.0, 0.0)

    dish_cx, dish_cy, dish_r = petri_dish_circle(xlim, ylim, cfg)
    work = df[[cols.frame, cols.x, cols.y, "_time_sec", "larva_id_norm"]].copy()
    work = work.sort_values(["larva_id_norm", cols.frame]).reset_index(drop=True)
    work["prev_x"] = work.groupby("larva_id_norm")[cols.x].shift(1)
    work["prev_y"] = work.groupby("larva_id_norm")[cols.y].shift(1)
    work["prev_t"] = work.groupby("larva_id_norm")["_time_sec"].shift(1)
    work["dx"] = work[cols.x] - work["prev_x"]
    work["dy"] = work[cols.y] - work["prev_y"]
    work["dt"] = work["_time_sec"] - work["prev_t"]
    work["displacement"] = np.sqrt(work["dx"] ** 2 + work["dy"] ** 2)
    work["velocity"] = np.where(work["dt"] > 0, work["displacement"] / work["dt"], np.nan)
    work["radius_to_center"] = np.sqrt((work[cols.x] - dish_cx) ** 2 + (work[cols.y] - dish_cy) ** 2)
    work["outside_dish"] = work["radius_to_center"] > dish_r

    disp = pd.to_numeric(work["displacement"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    vel = pd.to_numeric(work["velocity"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    d_med, d_mad = _robust_median_mad(disp)
    v_med, v_mad = _robust_median_mad(vel)
    d_thr_robust = d_med + max(0.0, float(cfg.artifact_mad_multiplier)) * d_mad
    v_thr_robust = v_med + max(0.0, float(cfg.artifact_mad_multiplier)) * v_mad
    d_thr = float(cfg.artifact_displacement_threshold) if cfg.artifact_displacement_threshold is not None else d_thr_robust
    v_thr = float(cfg.artifact_velocity_threshold) if cfg.artifact_velocity_threshold is not None else v_thr_robust

    jump_mask = pd.to_numeric(work["displacement"], errors="coerce") > d_thr
    vel_mask = pd.to_numeric(work["velocity"], errors="coerce") > v_thr
    out_mask = work["outside_dish"].fillna(False)
    flagged = work.loc[jump_mask | vel_mask | out_mask].copy()
    if len(flagged) == 0:
        return ArtifactDetectionResult(set(), pd.DataFrame(), d_med, d_mad, v_med, v_mad)

    def reason_for_row(row: pd.Series) -> str:
        reasons: List[str] = []
        if bool(row.get("outside_dish", False)):
            reasons.append("outside_dish")
        if pd.notna(row.get("displacement")) and float(row["displacement"]) > d_thr:
            reasons.append("large_displacement")
        if pd.notna(row.get("velocity")) and float(row["velocity"]) > v_thr:
            reasons.append("high_velocity")
        return "|".join(reasons) if reasons else "unknown"

    flagged["reason"] = flagged.apply(reason_for_row, axis=1)
    report = pd.DataFrame(
        {
            "track_id": flagged["larva_id_norm"].astype(str),
            "frame_index": pd.to_numeric(flagged[cols.frame], errors="coerce"),
            "x_before_jump": pd.to_numeric(flagged["prev_x"], errors="coerce"),
            "y_before_jump": pd.to_numeric(flagged["prev_y"], errors="coerce"),
            "x_after_jump": pd.to_numeric(flagged[cols.x], errors="coerce"),
            "y_after_jump": pd.to_numeric(flagged[cols.y], errors="coerce"),
            "displacement": pd.to_numeric(flagged["displacement"], errors="coerce"),
            "velocity": pd.to_numeric(flagged["velocity"], errors="coerce"),
            "outside_petri_dish": flagged["outside_dish"].astype(bool),
            "reason": flagged["reason"].astype(str),
        }
    )
    if cfg.group_condition_column in df.columns:
        group_map = (
            df[["larva_id_norm", cfg.group_condition_column]]
            .drop_duplicates(subset=["larva_id_norm"])
            .set_index("larva_id_norm")[cfg.group_condition_column]
            .astype(str)
            .to_dict()
        )
        report.insert(0, "condition", report["track_id"].map(group_map).fillna("unknown"))
    else:
        report.insert(0, "condition", "all")

    flagged_tracks = {str(x) for x in report["track_id"].dropna().unique().tolist()}
    return ArtifactDetectionResult(flagged_tracks, report, d_med, d_mad, v_med, v_mad)


def petri_dish_circle(
    xlim: Tuple[float, float], ylim: Tuple[float, float], cfg: VideoGenConfig
) -> Tuple[float, float, float]:
    """Center at arena midpoint (or origin if limits are symmetric); radius from config."""
    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])
    if np.isclose(xlim[0], -xlim[1]) and np.isclose(ylim[0], -ylim[1]):
        cx, cy = 0.0, 0.0
    return cx, cy, float(cfg.petri_dish_radius)


def log_track_summary_diagnostics(
    df: pd.DataFrame,
    cols: ColumnSelection,
    cfg: VideoGenConfig,
    *,
    context_label: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    pre_filter_track_count: int,
    post_filter_track_count: int,
) -> None:
    n_points = len(df)
    x_min_raw = float(pd.to_numeric(df[cols.x], errors="coerce").min()) if n_points else float("nan")
    x_max_raw = float(pd.to_numeric(df[cols.x], errors="coerce").max()) if n_points else float("nan")
    y_min_raw = float(pd.to_numeric(df[cols.y], errors="coerce").min()) if n_points else float("nan")
    y_max_raw = float(pd.to_numeric(df[cols.y], errors="coerce").max()) if n_points else float("nan")
    cx, cy, r = petri_dish_circle(xlim, ylim, cfg)
    outside_mask = np.sqrt((pd.to_numeric(df[cols.x], errors="coerce") - cx) ** 2 + (pd.to_numeric(df[cols.y], errors="coerce") - cy) ** 2) > r
    outside_points = int(outside_mask.fillna(False).sum()) if n_points else 0
    print(f"[Track summary diagnostics][{context_label}]")
    print(f"  tracks before filtering: {pre_filter_track_count:,}")
    print(f"  tracks after filtering: {post_filter_track_count:,}")
    print(f"  points plotted (rows): {n_points:,}")
    print(f"  coordinate ranges before/after transform (no transform applied):")
    print(f"    x raw=({x_min_raw:.3f}, {x_max_raw:.3f}) -> plotted=({x_min_raw:.3f}, {x_max_raw:.3f})")
    print(f"    y raw=({y_min_raw:.3f}, {y_max_raw:.3f}) -> plotted=({y_min_raw:.3f}, {y_max_raw:.3f})")
    print(
        f"  petri mask/boundary filter applied to tracks: no "
        f"(overlay only; center=({cx:.3f}, {cy:.3f}), radius={r:.3f}, outside_points={outside_points:,})"
    )


def apply_exclude_track_ids(df: pd.DataFrame, cfg: VideoGenConfig, *, context_label: str) -> pd.DataFrame:
    """Remove explicitly listed track IDs before rendering."""
    if not cfg.exclude_track_ids:
        return df
    want = {str(tid).strip() for tid in cfg.exclude_track_ids if str(tid).strip()}
    if not want:
        return df
    pre_tracks = int(df["larva_id_norm"].nunique()) if "larva_id_norm" in df.columns else 0
    before_rows = len(df)
    out = df[~df["larva_id_norm"].astype(str).isin(want)].copy()
    post_tracks = int(out["larva_id_norm"].nunique()) if "larva_id_norm" in out.columns else 0
    print(
        f"[Track exclusion][{context_label}] removed {pre_tracks - post_tracks:,} track(s) "
        f"({before_rows - len(out):,} rows) for IDs: {', '.join(sorted(want))}"
    )
    return out


def maybe_detect_and_filter_artifacts(
    df: pd.DataFrame,
    cols: ColumnSelection,
    cfg: VideoGenConfig,
    *,
    output_dir: Path,
    context_label: str,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
) -> Tuple[pd.DataFrame, Set[str]]:
    pre_tracks = int(df["larva_id_norm"].nunique()) if "larva_id_norm" in df.columns else 0
    if not cfg.detect_artifacts:
        log_track_summary_diagnostics(
            df,
            cols,
            cfg,
            context_label=context_label,
            xlim=xlim,
            ylim=ylim,
            pre_filter_track_count=pre_tracks,
            post_filter_track_count=pre_tracks,
        )
        return df, set()

    result = detect_artifact_tracks(df, cols, cfg, xlim, ylim)
    report_name = "artifact_track_report.csv" if context_label == "single" else f"artifact_track_report_{safe_condition_subdir(context_label)}.csv"
    report_path = output_dir / report_name
    result.report_df.to_csv(report_path, index=False)
    print(
        f"[Artifact detection][{context_label}] "
        f"flagged_tracks={len(result.flagged_track_ids):,}, flagged_events={len(result.report_df):,} -> {report_path}"
    )
    print(
        f"[Artifact detection][{context_label}] robust thresholds: "
        f"disp_med={result.displacement_median:.4f}, disp_mad={result.displacement_mad:.4f}, "
        f"vel_med={result.velocity_median:.4f}, vel_mad={result.velocity_mad:.4f}, "
        f"mad_multiplier={cfg.artifact_mad_multiplier:g}, "
        f"velocity_threshold={cfg.artifact_velocity_threshold if cfg.artifact_velocity_threshold is not None else 'auto'}, "
        f"displacement_threshold={cfg.artifact_displacement_threshold if cfg.artifact_displacement_threshold is not None else 'auto'}"
    )
    if result.flagged_track_ids:
        show = sorted(result.flagged_track_ids)
        preview = ", ".join(show[:20])
        more = " ..." if len(show) > 20 else ""
        print(f"[Artifact detection][{context_label}] flagged track IDs: {preview}{more}")

    out_df = df
    post_tracks = pre_tracks
    if cfg.exclude_artifact_tracks and result.flagged_track_ids:
        out_df = df[~df["larva_id_norm"].astype(str).isin(result.flagged_track_ids)].copy()
        post_tracks = int(out_df["larva_id_norm"].nunique())
        print(
            f"[Artifact filtering][{context_label}] excluded {pre_tracks - post_tracks:,} tracks; "
            f"remaining tracks={post_tracks:,}, rows={len(out_df):,}"
        )
    elif cfg.exclude_artifact_tracks:
        print(f"[Artifact filtering][{context_label}] enabled, but no tracks were flagged.")

    log_track_summary_diagnostics(
        out_df,
        cols,
        cfg,
        context_label=context_label,
        xlim=xlim,
        ylim=ylim,
        pre_filter_track_count=pre_tracks,
        post_filter_track_count=post_tracks,
    )
    return out_df, result.flagged_track_ids


def draw_petri_dish(
    ax,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    cfg: VideoGenConfig,
    *,
    linestyle: str = "--",
    linewidth: float = 1.2,
    alpha: float = 0.65,
    edgecolor: Optional[str] = None,
    zorder: int = 10,
) -> Optional[Circle]:
    """Draw Petri dish outline in plot coordinates; returns the patch or None if disabled."""
    if not cfg.infer_petri_dish:
        return None
    cx, cy, r = petri_dish_circle(xlim, ylim, cfg)
    ec = edgecolor if edgecolor is not None else "0.45"
    patch = Circle(
        (cx, cy),
        r,
        fill=False,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        edgecolor=ec,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def draw_density_petri_dish(
    ax,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    cfg: VideoGenConfig,
) -> Optional[Circle]:
    """Green Petri outline for density maps."""
    return draw_petri_dish(
        ax,
        xlim,
        ylim,
        cfg,
        linestyle="-",
        linewidth=1.4,
        alpha=0.9,
        edgecolor=DENSITY_PETRI_EDGE_COLOR,
        zorder=10,
    )


def display_group_label(group_id: str, cfg: VideoGenConfig) -> str:
    """Readable label for plots; falls back to internal group_id."""
    return cfg.group_display_labels.get(str(group_id), str(group_id))


def parse_group_label_arg(spec: str) -> Tuple[str, str]:
    """Parse ``internal_id:Display Label`` from CLI."""
    if ":" not in spec:
        raise ValueError(f"Group label must be ID:Label, got {spec!r}")
    internal, label = spec.split(":", 1)
    internal = internal.strip()
    label = label.strip()
    if not internal or not label:
        raise ValueError(f"Group label must be ID:Label, got {spec!r}")
    return internal, label


def filter_selected_groups(
    cond_values: List[str], cfg: VideoGenConfig
) -> List[str]:
    """Restrict compare panels to cfg.selected_groups when set."""
    if not cfg.selected_groups:
        return cond_values
    want = {str(g) for g in cfg.selected_groups}
    # Preserve caller order (e.g. fed left, starved right) instead of sorted cond_values.
    out = [str(g) for g in cfg.selected_groups if str(g) in {str(v) for v in cond_values}]
    missing = want - {str(v) for v in out}
    if missing:
        warnings.warn(
            f"selected_groups not found in data: {sorted(missing)}",
            UserWarning,
            stacklevel=2,
        )
    if not out:
        raise ValueError(
            f"No matching groups for selected_groups={cfg.selected_groups!r} "
            f"(available: {cond_values})."
        )
    return out


def compare_output_stem(cfg: VideoGenConfig, base: str) -> str:
    """Insert compare slug after ``_compare``, e.g. larvae_density_compare_summary -> larvae_density_compare_4d_summary."""
    if not cfg.compare_output_slug:
        return base
    slug = safe_condition_subdir(cfg.compare_output_slug)
    token = "_compare"
    if token in base:
        pos = base.find(token) + len(token)
        return base[:pos] + f"_{slug}" + base[pos:]
    return f"{base}_{slug}"


def normalize_density_method_name(method: str) -> str:
    m = method.lower().strip()
    if m in ("none", "off", "0"):
        return "none"
    if m in ("num_tracks", "tracks", "n_tracks"):
        return "num_tracks"
    if m in ("max_visible_larvae", "max_larvae", "visible", "max_visible"):
        return "max_visible_larvae"
    raise ValueError(
        f"Unknown density_normalization {method!r}. "
        "Use none, num_tracks, or max_visible_larvae."
    )


def compute_group_normalization_factors(
    df_animals: pd.DataFrame,
    cols: ColumnSelection,
    group_col: str,
    group_ids: Sequence[str],
    method: str,
) -> Dict[str, float]:
    """Fixed per-group divisors from animal tracking (same factor for larval and HC density)."""
    norm = normalize_density_method_name(method)
    factors: Dict[str, float] = {}
    for gid in group_ids:
        key = str(gid)
        sub = df_animals[df_animals[group_col].astype(str) == key]
        if norm == "none":
            factors[key] = 1.0
        elif norm == "num_tracks":
            n = int(sub["larva_id_norm"].nunique()) if len(sub) else 0
            factors[key] = float(max(1, n))
        else:
            if len(sub) == 0:
                factors[key] = 1.0
            else:
                per_frame = sub.groupby(cols.frame)["larva_id_norm"].nunique()
                factors[key] = float(max(1.0, per_frame.max()))
    return factors


def apply_density_normalization(
    dens: np.ndarray, factor: float
) -> np.ndarray:
    if factor is None or factor <= 0 or np.isclose(factor, 1.0):
        return dens
    return dens / float(factor)


def density_cbar_label(hc_mode: bool, cfg: VideoGenConfig) -> str:
    norm = normalize_density_method_name(cfg.density_normalization)
    if norm == "none":
        return "HC events / bin" if hc_mode else "Larvae / bin"
    return "HC density per larva" if hc_mode else "Larval density per animal"


def stabilized_limits(vmin: float, vmax: float) -> Tuple[float, float]:
    if np.isclose(vmin, vmax):
        pad = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def apply_fixed_axis_xy_limits(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    fixed: Optional[Tuple[float, float]],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """If ``fixed`` is ``(lo, hi)``, set both xlim and ylim to that range; else keep data-derived limits."""
    if fixed is None:
        return xlim, ylim
    lo, hi = float(fixed[0]), float(fixed[1])
    if not np.isfinite(lo) or not np.isfinite(hi) or lo >= hi:
        raise ValueError("fixed_axis_xy_limits must be (min, max) with min < max and finite values.")
    return (lo, hi), (lo, hi)


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["grid.alpha"] = 0.25
    mpl.rcParams["font.size"] = 11


def append_frame_to_window(recent_frames: deque, frame_key: float, frame_to_points: Dict[float, np.ndarray]) -> None:
    """Only extend the sliding window with frames that have tracking data."""
    if frame_key in frame_to_points and len(frame_to_points[frame_key]) > 0:
        recent_frames.append(frame_key)


def get_window_points(
    target_frame: float,
    recent_frames: deque,
    frame_to_points: Dict[float, np.ndarray],
    window_size: int,
) -> np.ndarray:
    """Last ``window_size`` frames that contain data (not empty frame-index gaps)."""
    del target_frame  # kept for call-site compatibility
    frames_with_data = [f for f in recent_frames if f in frame_to_points]
    if not frames_with_data:
        return np.empty((0, 5), dtype=object)
    if len(frames_with_data) > window_size:
        frames_with_data = frames_with_data[-window_size:]
    arrays = [frame_to_points[f] for f in frames_with_data]
    stacked = np.vstack(arrays)
    frame_col = np.concatenate(
        [np.full(len(frame_to_points[f]), f, dtype=float) for f in frames_with_data]
    )
    return np.column_stack([stacked, frame_col])


def random_track_color_for_id(larva_id: str, seed: int = 42) -> np.ndarray:
    """Stable pseudo-random RGBA per track ID (uniform hue; not tied to sorted index)."""
    digest = hashlib.md5(f"{seed}:{larva_id}".encode("utf-8")).digest()
    hue = int.from_bytes(digest[0:2], "big") / 65535.0
    sat = 0.55 + (digest[2] / 255.0) * 0.45
    val = 0.65 + (digest[3] / 255.0) * 0.35
    rgb = mcolors.hsv_to_rgb((hue, sat, val))
    return np.array([*rgb, 1.0], dtype=float)


def build_color_map(
    ids: Sequence[str],
    cmap_name: str,
    *,
    seed: int = 42,
    random_hsv: bool = True,
) -> Dict[str, np.ndarray]:
    """Map track IDs to RGBA colors (default: hash-random HSV, reproducible per ID)."""
    color_map: Dict[str, np.ndarray] = {}
    sorted_ids = sorted({str(i) for i in ids})
    if random_hsv:
        for larva_id in sorted_ids:
            color_map[larva_id] = random_track_color_for_id(larva_id, seed=seed)
        return color_map
    cmap = plt.get_cmap(cmap_name, max(20, len(sorted_ids)))
    for i, larva_id in enumerate(sorted_ids):
        color_map[larva_id] = np.array(cmap(i % cmap.N))
    return color_map


def same_color_rgba(hex_or_name: str, n: int) -> np.ndarray:
    rgba = np.array(mcolors.to_rgba(hex_or_name))
    return np.tile(rgba, (n, 1))


def make_video_writer(path: Path, frame_size: Tuple[int, int], fps: int) -> cv2.VideoWriter:
    fourcc_candidates = ("mp4v", "avc1", "H264")
    w, h = frame_size
    for code in fourcc_candidates:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), fps, (w, h))
        if writer.isOpened():
            print(f"Using OpenCV codec: {code}")
            return writer
        writer.release()
    raise RuntimeError(
        "Could not open MP4 video writer with OpenCV codecs (tried: mp4v, avc1, H264)."
    )


def fig_to_bgr_frame(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def validate_track_history(cfg: VideoGenConfig) -> None:
    if cfg.track_history_seconds is not None and cfg.track_history_seconds <= 0:
        raise ValueError("track_history_seconds must be > 0 when set (or None for cumulative).")


def normalize_density_mode(mode: str) -> str:
    m = mode.lower().strip()
    if m in ("animals", "all", "animal"):
        return "animals"
    if m == "hc":
        return "hc"
    raise ValueError(f"Unknown density_mode: {mode!r}. Use 'animals', 'all', or 'hc'.")


def effective_hc_density_bins(cfg: VideoGenConfig) -> int:
    if cfg.hc_density_bins is not None:
        return int(cfg.hc_density_bins)
    # Coarser bins by default so sparse HC occupancy stays visible.
    return max(24, cfg.density_bins // 2)


def effective_kde_sigma_bins(cfg: VideoGenConfig, *, for_summary: bool = False) -> float:
    """KDE Gaussian blur sigma (histogram-bin units) for movies or static summaries."""
    if for_summary and cfg.kde_sigma_bins_summary is not None:
        return float(cfg.kde_sigma_bins_summary)
    if not for_summary and cfg.kde_sigma_bins_movie is not None:
        return float(cfg.kde_sigma_bins_movie)
    return float(cfg.kde_sigma_bins)


def effective_density_bins_render(
    cfg: VideoGenConfig, *, for_summary: bool = False, hc_mode: bool = False
) -> int:
    """Grid resolution for density movies (coarser) or static summaries (finer)."""
    if hc_mode:
        if for_summary and cfg.hc_density_bins_summary is not None:
            return max(16, int(cfg.hc_density_bins_summary))
        if not for_summary and cfg.hc_density_bins_movie is not None:
            return max(16, int(cfg.hc_density_bins_movie))
        return effective_hc_density_bins(cfg)
    if for_summary and cfg.density_bins_summary is not None:
        return max(16, int(cfg.density_bins_summary))
    if not for_summary and cfg.density_bins_movie is not None:
        return max(16, int(cfg.density_bins_movie))
    return max(16, int(cfg.density_bins))


def effective_density_window_frames(cfg: VideoGenConfig, hc_mode: bool) -> int:
    """Sliding window length in frames (with data) for larval vs HC density."""
    if hc_mode and cfg.hc_window_size_frames is not None:
        return max(1, int(cfg.hc_window_size_frames))
    return max(1, int(cfg.window_size_frames))


def union_frame_keys(
    frame_to_points: Dict[float, np.ndarray],
    *more: Dict[float, np.ndarray],
) -> np.ndarray:
    """Sorted frame indices present in one or more caches."""
    keys: set = set(float(k) for k in frame_to_points.keys())
    for fp in more:
        keys.update(float(k) for k in fp.keys())
    if not keys:
        return np.array([], dtype=float)
    return np.array(sorted(keys), dtype=float)


def density_vmax_sample_sequence(
    timeline_frames: np.ndarray,
    frame_to_points: Dict[float, np.ndarray],
    *,
    hc_mode: bool,
) -> np.ndarray:
    """Frames used to estimate color scale. HC: only frames that contain HC points."""
    data_frames = union_frame_keys(frame_to_points)
    if hc_mode and len(data_frames) > 0:
        return data_frames
    return timeline_frames


def effective_density_vmax_percentile(
    cfg: VideoGenConfig, *, hc_mode: bool, for_summary: bool = False
) -> float:
    """Percentile used to clip density color scale (movies vs static summaries)."""
    if hc_mode:
        if for_summary and cfg.hc_density_vmax_percentile_summary is not None:
            return float(cfg.hc_density_vmax_percentile_summary)
        return float(cfg.hc_density_vmax_percentile)
    if for_summary and cfg.density_vmax_percentile_summary is not None:
        return float(cfg.density_vmax_percentile_summary)
    return float(cfg.density_vmax_percentile)


def finalize_density_vmax(
    peak_values: List[float],
    *,
    hc_mode: bool,
    cfg: VideoGenConfig,
    for_summary: bool = False,
) -> float:
    """Turn sampled per-frame density peaks into a display vmax."""
    positive = [float(p) for p in peak_values if p > 0 and np.isfinite(p)]
    if not positive:
        return 1e-6
    pct = effective_density_vmax_percentile(cfg, hc_mode=hc_mode, for_summary=for_summary)
    pct = min(100.0, max(50.0, pct))
    if pct < 100.0:
        return max(float(np.percentile(positive, pct)), 1e-9)
    return max(float(np.max(positive)), 1e-6)


def density_color_norm(vmax: float, hc_mode: bool, cfg: VideoGenConfig):
    """Linear, sqrt, or log color normalization for density heatmaps."""
    vmax = max(float(vmax), 1e-9)
    if hc_mode and cfg.hc_density_use_log_norm:
        vmin = max(vmax * 1e-3, 1e-10)
        if vmin >= vmax:
            vmin = vmax / 100.0
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    if not hc_mode and cfg.density_use_sqrt_norm:
        return mcolors.PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    return None


def shared_density_color_norm(
    vmax: float, hc_mode: bool, cfg: VideoGenConfig
) -> mcolors.Normalize:
    """One Normalize/LogNorm instance for all condition panels (same color scale)."""
    norm = density_color_norm(vmax, hc_mode, cfg)
    if norm is None:
        return mcolors.Normalize(vmin=0.0, vmax=max(float(vmax), 1e-9))
    return norm


def density_display_cmap(cfg: VideoGenConfig) -> mcolors.Colormap:
    """Colormap with black for masked / empty (zero) density bins."""
    cmap = plt.get_cmap(cfg.density_cmap).copy()
    cmap.set_bad(DENSITY_NODATA_COLOR)
    cmap.set_under(DENSITY_NODATA_COLOR)
    return cmap


def mask_empty_density(dens: np.ndarray) -> np.ma.MaskedArray:
    """Zero or negative bins are treated as no data (rendered black)."""
    d = np.asarray(dens, dtype=float)
    return np.ma.masked_where(d <= 0, d)


def density_imshow_kwargs(
    vmax: float,
    hc_mode: bool,
    cfg: VideoGenConfig,
    *,
    norm: Optional[mcolors.Normalize] = None,
) -> Dict[str, object]:
    """Common imshow kwargs for density heatmaps."""
    kw: Dict[str, object] = dict(
        origin="lower",
        cmap=density_display_cmap(cfg),
        interpolation="nearest",
        aspect="equal",
    )
    if norm is None:
        norm = shared_density_color_norm(vmax, hc_mode, cfg)
    kw["norm"] = norm
    return kw


def petri_clip_patch(
    ax, xlim: Tuple[float, float], ylim: Tuple[float, float], cfg: VideoGenConfig
) -> Circle:
    """Clip path matching the Petri dish (density only drawn inside)."""
    cx, cy, r = petri_dish_circle(xlim, ylim, cfg)
    return Circle((cx, cy), r, transform=ax.transData)


def apply_density_petri_clip(image, ax, xlim: Tuple[float, float], ylim: Tuple[float, float], cfg: VideoGenConfig) -> None:
    """Restrict heatmap to the dish interior; outside stays axes background (white)."""
    image.set_clip_path(petri_clip_patch(ax, xlim, ylim, cfg))


def style_density_axes(ax, xlim: Tuple[float, float], ylim: Tuple[float, float], cfg: VideoGenConfig) -> None:
    ax.set_facecolor(DENSITY_AXES_FACE_COLOR)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    hide_arena_axes(ax)
    draw_density_petri_dish(ax, xlim, ylim, cfg)


def compute_density_grid(
    x: np.ndarray,
    y: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    density_method: str,
    kde_sigma_bins: float,
) -> np.ndarray:
    h, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    dens = h.T.astype(np.float32)
    if density_method.lower() == "kde":
        dens = cv2.GaussianBlur(dens, ksize=(0, 0), sigmaX=kde_sigma_bins, sigmaY=kde_sigma_bins)
    return dens


def _max_density(
    frame_sequence: np.ndarray,
    frame_to_points: Dict[float, np.ndarray],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    window_size_frames: int,
    density_method: str,
    kde_sigma_bins: float,
    sample_cap: int,
    norm_factor: float = 1.0,
    *,
    hc_mode: bool = False,
    cfg: Optional[VideoGenConfig] = None,
) -> float:
    sample_seq = density_vmax_sample_sequence(
        frame_sequence, frame_to_points, hc_mode=hc_mode
    )
    n = len(sample_seq)
    if n == 0:
        return 1e-6
    sample_n = min(sample_cap, n)
    sample_idx = np.linspace(0, n - 1, sample_n, dtype=int)
    sampled_frames = sample_seq[sample_idx]

    recent_frames: deque = deque()
    peaks: List[float] = []
    for f in sampled_frames:
        append_frame_to_window(recent_frames, float(f), frame_to_points)
        window = get_window_points(f, recent_frames, frame_to_points, window_size_frames)
        if len(window) == 0:
            continue
        x = window[:, 1].astype(float)
        y = window[:, 2].astype(float)
        dens = compute_density_grid(x, y, x_edges, y_edges, density_method, kde_sigma_bins)
        dens = apply_density_normalization(dens, norm_factor)
        peak = float(dens.max())
        if peak > 0:
            peaks.append(peak)

    if cfg is None:
        cfg = VideoGenConfig()
    return finalize_density_vmax(peaks, hc_mode=hc_mode, cfg=cfg)


def setup_track_figure(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    title: str,
    cfg: VideoGenConfig,
):
    fig, ax = plt.subplots(figsize=cfg.figsize)
    scatter = ax.scatter([], [], s=6, linewidths=0)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    hide_arena_axes(ax)
    draw_petri_dish(ax, xlim, ylim, cfg)
    clock = add_digital_clock_to_axes(ax, corner="upper_left")
    return fig, ax, scatter, clock


def draw_track_points(
    scatter,
    xs: np.ndarray,
    ys: np.ndarray,
    ids: List[str],
    color_map: Dict[str, np.ndarray],
    cfg: VideoGenConfig,
):
    n = len(xs)
    if n == 0:
        scatter.set_offsets(np.empty((0, 2)))
        scatter.set_facecolors(np.empty((0, 4)))
        return
    if cfg.track_color_mode == "same":
        rgba = same_color_rgba(cfg.track_color, n)
    else:
        rgba = np.vstack([color_map[k].copy() for k in ids])
    rgba[:, 3] = 0.35
    scatter.set_offsets(np.column_stack([xs, ys]))
    scatter.set_facecolors(rgba)


def frame_end_times_sec(frame_to_points: Dict[float, np.ndarray]) -> Dict[float, float]:
    """Latest _time_sec per frame key (matches animation clock per frame)."""
    return {float(k): float(np.max(arr[:, 3])) for k, arr in frame_to_points.items()}


def hide_arena_axes(ax) -> None:
    """Petri-dish view: no axis labels or tick marks (videos and summaries)."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(
        axis="both",
        which="both",
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )


def count_unique_tracks(frame_to_points: Dict[float, np.ndarray]) -> int:
    return len({str(row[0]) for arr in frame_to_points.values() for row in arr})


def count_hc_events(frame_to_points: Dict[float, np.ndarray]) -> int:
    """Total head-cast events (rows) in a frame cache (typically HC-filtered data)."""
    return int(sum(len(arr) for arr in frame_to_points.values()))


def build_recording_stats(
    animals_frame_to_points: Dict[float, np.ndarray],
    hc_frame_to_points: Optional[Dict[float, np.ndarray]] = None,
) -> Tuple[int, int]:
    n_animals = count_unique_tracks(animals_frame_to_points)
    n_hcs = count_hc_events(hc_frame_to_points) if hc_frame_to_points else 0
    return n_animals, n_hcs


def build_recording_stats_by_condition(
    conditions: Sequence[str],
    animals_by_cond: Dict[str, Dict[float, np.ndarray]],
    hc_by_cond: Dict[str, Dict[float, np.ndarray]],
) -> Dict[str, Tuple[int, int]]:
    out: Dict[str, Tuple[int, int]] = {}
    for cond in conditions:
        key = str(cond)
        a_fp = animals_by_cond.get(key, {})
        h_fp = hc_by_cond.get(key, {})
        out[key] = (count_unique_tracks(a_fp), count_hc_events(h_fp))
    return out


def prepare_hc_frame_cache_pooled(
    df: pd.DataFrame,
    cols: ColumnSelection,
    cfg: VideoGenConfig,
) -> Dict[float, np.ndarray]:
    try:
        hc_col = resolve_hc_column(df.columns.tolist(), cfg)
    except ValueError:
        return {}
    df_h = filter_hc_rows(df, hc_col, cfg)
    if len(df_h) == 0:
        return {}
    _, fp = prepare_frame_cache(df_h, cols.frame, cols.x, cols.y)
    return fp


def prepare_hc_frame_caches_by_condition(
    df: pd.DataFrame,
    cols: ColumnSelection,
    cfg: VideoGenConfig,
    group_col: str,
    condition_values: Sequence[str],
) -> Dict[str, Dict[float, np.ndarray]]:
    out: Dict[str, Dict[float, np.ndarray]] = {}
    try:
        hc_col = resolve_hc_column(df.columns.tolist(), cfg)
    except ValueError:
        for val in condition_values:
            out[str(val)] = {}
        return out
    for val in condition_values:
        key = str(val)
        df_sub = df[df[group_col].astype(str) == key].copy()
        df_h = filter_hc_rows(df_sub, hc_col, cfg)
        if len(df_h) == 0:
            out[key] = {}
            continue
        _, fp = prepare_frame_cache(df_h, cols.frame, cols.x, cols.y)
        out[key] = fp
    return out


def add_summary_recording_stats_box(
    ax,
    n_animals: int,
    n_hcs: int,
    *,
    corner: Literal["lower_right", "lower_left", "upper_right"] = "lower_right",
) -> None:
    """Final summary overlay: total animals (tracks) and HC events for the full recording."""
    positions = {
        "lower_right": (0.98, 0.02, "right", "bottom"),
        "lower_left": (0.02, 0.02, "left", "bottom"),
        "upper_right": (0.98, 0.98, "right", "top"),
    }
    x, y, ha, va = positions[corner]
    text = f"Animals: {n_animals:,}\nHCs: {n_hcs:,}"
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va=va,
        fontsize=11,
        fontfamily="monospace",
        fontweight="bold",
        color="white",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="black",
            edgecolor="0.75",
            alpha=0.88,
            linewidth=1.2,
        ),
        zorder=25,
    )


def timeline_origin_sec(
    times_by_frame: Dict[float, float],
    frame_sequence: np.ndarray,
) -> float:
    """Recording time (seconds) at the first animated frame."""
    for f in frame_sequence:
        t = times_by_frame.get(float(f))
        if t is not None and np.isfinite(t):
            return float(t)
    if times_by_frame:
        return float(min(times_by_frame.values()))
    return 0.0


# Digital clock overlay (videos): monospace digits in a boxed HUD corner readout
DIGITAL_CLOCK_COLOR = "#39ff14"
DIGITAL_CLOCK_BOX_ALPHA = 0.88


def format_digital_clock_seconds(elapsed: float) -> str:
    """MM:SS.s display (HH:MM:SS.s when >= 1 hour)."""
    elapsed = max(0.0, float(elapsed))
    total_sec = elapsed
    hours = int(total_sec // 3600)
    rem = total_sec - hours * 3600
    minutes = int(rem // 60)
    sec = rem - minutes * 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:05.2f}"
    return f"{minutes:02d}:{sec:05.2f}"


def elapsed_seconds_for_frame(
    frame_key: float,
    times_by_frame: Dict[float, float],
    t_origin: float,
) -> Optional[float]:
    t = times_by_frame.get(float(frame_key))
    if t is None or not np.isfinite(t):
        return None
    return max(0.0, float(t) - float(t_origin))


def digital_clock_text_kwargs() -> Dict[str, object]:
    return dict(
        fontsize=14,
        fontfamily="monospace",
        fontweight="bold",
        color=DIGITAL_CLOCK_COLOR,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="black",
            edgecolor=DIGITAL_CLOCK_COLOR,
            alpha=DIGITAL_CLOCK_BOX_ALPHA,
            linewidth=1.5,
        ),
    )


def add_digital_clock_to_axes(
    ax,
    *,
    corner: Literal["upper_left", "lower_left", "upper_right", "lower_right"] = "upper_left",
):
    """Corner digital clock on a single subplot (axes coordinates)."""
    positions = {
        "upper_left": (0.02, 0.98, "left", "top"),
        "upper_right": (0.98, 0.98, "right", "top"),
        "lower_left": (0.02, 0.02, "left", "bottom"),
        "lower_right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = positions[corner]
    kw = digital_clock_text_kwargs()
    return ax.text(
        x,
        y,
        "00:00.0",
        transform=ax.transAxes,
        ha=ha,
        va=va,
        **kw,
    )


def add_digital_clock_to_figure(
    fig: plt.Figure,
    *,
    corner: Literal["upper_left", "lower_left", "upper_right", "lower_right"] = "lower_left",
):
    """One shared digital clock for multi-panel compare videos (figure coordinates)."""
    positions = {
        "upper_left": (0.02, 0.98, "left", "top"),
        "upper_right": (0.98, 0.98, "right", "top"),
        "lower_left": (0.02, 0.02, "left", "bottom"),
        "lower_right": (0.98, 0.02, "right", "bottom"),
    }
    x, y, ha, va = positions[corner]
    kw = digital_clock_text_kwargs()
    return fig.text(
        x,
        y,
        "00:00.0",
        transform=fig.transFigure,
        ha=ha,
        va=va,
        zorder=100,
        **kw,
    )


def update_digital_clock(
    clock_artist,
    frame_key: float,
    times_by_frame: Dict[float, float],
    t_origin: float,
) -> None:
    elapsed = elapsed_seconds_for_frame(frame_key, times_by_frame, t_origin)
    if elapsed is None:
        clock_artist.set_text("")
    else:
        clock_artist.set_text(format_digital_clock_seconds(elapsed))


def create_track_animation(
    unique_frames: np.ndarray,
    frame_to_points: Dict[float, np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    output_path: Path,
    color_map: Dict[str, np.ndarray],
    cfg: VideoGenConfig,
    *,
    hc_frame_to_points: Optional[Dict[float, np.ndarray]] = None,
):
    validate_track_history(cfg)
    title = "Larval trajectories"
    if cfg.track_history_seconds is None:
        title = "Larval cumulative trajectories"
    else:
        title = f"Larval trajectories (last {cfg.track_history_seconds:g}s)"
    if cfg.track_color_mode == "same":
        title = f"{title} (uniform color)"

    fig, ax, scatter, clock = setup_track_figure(xlim, ylim, title, cfg)
    frame_sequence = unique_frames[:: cfg.frame_stride]
    times_by_frame = frame_end_times_sec(frame_to_points)
    t_origin = timeline_origin_sec(times_by_frame, frame_sequence)

    point_window: deque = deque()
    cumulative_ids: List[str] = []
    cumulative_x: List[float] = []
    cumulative_y: List[float] = []

    scatter.set_offsets(np.empty((0, 2)))
    scatter.set_facecolors(np.empty((0, 4)))
    update_digital_clock(clock, float(frame_sequence[0]) if len(frame_sequence) else 0.0, times_by_frame, t_origin)
    fig.canvas.draw()
    width_px, height_px = fig.canvas.get_width_height()
    writer = make_video_writer(output_path, (width_px, height_px), cfg.fps)
    last_bgr: Optional[np.ndarray] = None

    try:
        for i, f in enumerate(frame_sequence):
            if i % 200 == 0:
                print(f"Track animation progress: {i}/{len(frame_sequence)}", flush=True)
            fk = float(f)
            current = frame_to_points.get(fk)
            if cfg.track_history_seconds is not None:
                if current is not None and len(current) > 0:
                    for row in current:
                        point_window.append(
                            (str(row[0]), float(row[1]), float(row[2]), float(row[3]))
                        )
                t_cur = times_by_frame[fk]
                cutoff = t_cur - cfg.track_history_seconds
                while point_window and point_window[0][3] < cutoff:
                    point_window.popleft()
                xs_arr = np.asarray([p[1] for p in point_window], dtype=float)
                ys_arr = np.asarray([p[2] for p in point_window], dtype=float)
                ids_f = [p[0] for p in point_window]
                draw_track_points(scatter, xs_arr, ys_arr, ids_f, color_map, cfg)
            else:
                if current is not None and len(current) > 0:
                    for row in current:
                        cumulative_ids.append(str(row[0]))
                        cumulative_x.append(float(row[1]))
                        cumulative_y.append(float(row[2]))
                if not cumulative_x:
                    draw_track_points(scatter, np.array([]), np.array([]), [], color_map, cfg)
                else:
                    draw_track_points(
                        scatter,
                        np.asarray(cumulative_x, dtype=float),
                        np.asarray(cumulative_y, dtype=float),
                        cumulative_ids,
                        color_map,
                        cfg,
                    )

            update_digital_clock(clock, fk, times_by_frame, t_origin)
            last_bgr = fig_to_bgr_frame(fig)
            writer.write(last_bgr)

        if cfg.append_summary_frame and last_bgr is not None:
            summary_bgr = render_track_summary_bgr(
                xlim, ylim, frame_to_points, color_map, cfg, hc_frame_to_points=hc_frame_to_points
            )
            append_summary_to_writer(
                writer, last_bgr, summary_bgr, cfg.fps, cfg.summary_pause_seconds, cfg.summary_duration_seconds
            )
    finally:
        writer.release()
    plt.close(fig)


def render_track_summary_bgr(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    frame_to_points: Dict[float, np.ndarray],
    color_map: Dict[str, np.ndarray],
    cfg: VideoGenConfig,
    *,
    title: Optional[str] = None,
    recording_stats: Optional[Tuple[int, int]] = None,
    hc_frame_to_points: Optional[Dict[float, np.ndarray]] = None,
) -> np.ndarray:
    """Single static frame: all points in video (full cumulative tracks)."""
    if recording_stats is None:
        recording_stats = build_recording_stats(frame_to_points, hc_frame_to_points)
    n_animals, n_hcs = recording_stats
    plot_title = title or "Larval trajectories (full recording)"
    fig, ax, scatter, _ = setup_track_figure(xlim, ylim, plot_title, cfg)
    xs: List[float] = []
    ys: List[float] = []
    ids: List[str] = []
    for arr in frame_to_points.values():
        for row in arr:
            ids.append(str(row[0]))
            xs.append(float(row[1]))
            ys.append(float(row[2]))
    draw_track_points(scatter, np.asarray(xs), np.asarray(ys), ids, color_map, cfg)
    add_summary_recording_stats_box(ax, n_animals, n_hcs)
    fig.canvas.draw()
    out = fig_to_bgr_frame(fig)
    plt.close(fig)
    return out


def setup_density_figure(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    bins: int,
    vmax: float,
    title: str,
    cbar_label: str,
    cfg: VideoGenConfig,
    *,
    hc_mode: bool = False,
):
    fig, ax = plt.subplots(figsize=cfg.figsize, facecolor=DENSITY_AXES_FACE_COLOR)
    imshow_kw = density_imshow_kwargs(vmax, hc_mode, cfg)
    imshow_kw["extent"] = [xlim[0], xlim[1], ylim[0], ylim[1]]
    image = ax.imshow(mask_empty_density(np.zeros((bins, bins))), **imshow_kw)
    apply_density_petri_clip(image, ax, xlim, ylim, cfg)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    style_density_axes(ax, xlim, ylim, cfg)
    ax.set_title(title)
    clock = add_digital_clock_to_axes(ax, corner="upper_left")
    return fig, ax, image, clock


def create_density_animation(
    unique_frames: np.ndarray,
    frame_to_points: Dict[float, np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    output_path: Path,
    cfg: VideoGenConfig,
    *,
    density_bins: int,
    hc_mode: bool,
    norm_factor: float = 1.0,
    recording_stats: Optional[Tuple[int, int]] = None,
    animals_frame_to_points: Optional[Dict[float, np.ndarray]] = None,
    hc_frame_to_points: Optional[Dict[float, np.ndarray]] = None,
    vmax_override: Optional[float] = None,
    output_fps: Optional[int] = None,
):
    movie_bins = effective_density_bins_render(cfg, for_summary=False, hc_mode=hc_mode)
    kde_sigma = effective_kde_sigma_bins(cfg, for_summary=False)
    frame_sequence = unique_frames[:: cfg.frame_stride]
    x_edges = np.linspace(xlim[0], xlim[1], movie_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], movie_bins + 1)
    window_frames = effective_density_window_frames(cfg, hc_mode)
    if vmax_override is not None:
        vmax = float(vmax_override)
    else:
        vmax = _max_density(
            frame_sequence,
            frame_to_points,
            x_edges,
            y_edges,
            window_frames,
            cfg.density_method,
            kde_sigma,
            cfg.density_vmax_sample_frames,
            norm_factor=norm_factor,
            hc_mode=hc_mode,
            cfg=cfg,
        )
    if hc_mode:
        print(f"HC density color scale vmax={vmax:.6g} (percentile={cfg.hc_density_vmax_percentile})", flush=True)
    else:
        print(f"Larval density color scale vmax={vmax:.6g} (percentile={cfg.density_vmax_percentile})", flush=True)

    density_label = "KDE-smoothed" if cfg.density_method.lower() == "kde" else "Histogram"
    if hc_mode:
        title = f"HC occupancy density ({density_label}, sliding window)"
    else:
        title = f"Larval occupancy density ({density_label}, sliding window)"
    cbar_label = density_cbar_label(hc_mode, cfg)

    fig, ax, image, clock = setup_density_figure(
        xlim, ylim, movie_bins, vmax, title, cbar_label, cfg, hc_mode=hc_mode
    )
    recent_frames: deque = deque()
    times_by_frame = frame_end_times_sec(frame_to_points)
    t_origin = timeline_origin_sec(times_by_frame, frame_sequence)
    if len(frame_sequence) > 0:
        update_digital_clock(clock, float(frame_sequence[0]), times_by_frame, t_origin)

    fig.canvas.draw()
    width_px, height_px = fig.canvas.get_width_height()
    writer_fps = int(output_fps) if output_fps is not None else int(cfg.fps)
    writer = make_video_writer(output_path, (width_px, height_px), writer_fps)
    last_bgr: Optional[np.ndarray] = None

    try:
        for i, f in enumerate(frame_sequence):
            if i % 200 == 0:
                print(f"Density animation progress: {i}/{len(frame_sequence)}", flush=True)
            fk = float(f)
            append_frame_to_window(recent_frames, fk, frame_to_points)
            window = get_window_points(fk, recent_frames, frame_to_points, window_frames)
            if len(window) == 0:
                dens = np.zeros((movie_bins, movie_bins))
            else:
                x = window[:, 1].astype(float)
                y = window[:, 2].astype(float)
                dens = compute_density_grid(x, y, x_edges, y_edges, cfg.density_method, kde_sigma)
                dens = apply_density_normalization(dens, norm_factor)
            image.set_data(mask_empty_density(dens))
            update_digital_clock(clock, fk, times_by_frame, t_origin)
            last_bgr = fig_to_bgr_frame(fig)
            writer.write(last_bgr)

        if cfg.append_summary_frame and last_bgr is not None:
            summary_bgr = render_density_summary_bgr(
                xlim,
                ylim,
                frame_to_points,
                effective_density_bins_render(cfg, for_summary=True, hc_mode=hc_mode),
                vmax,
                hc_mode,
                cfg,
                norm_factor=norm_factor,
                recording_stats=recording_stats,
                animals_frame_to_points=animals_frame_to_points,
                hc_frame_to_points=hc_frame_to_points,
            )
            append_summary_to_writer(
                writer, last_bgr, summary_bgr, writer_fps, cfg.summary_pause_seconds, cfg.summary_duration_seconds
            )
    finally:
        writer.release()
    plt.close(fig)


def save_density_summary_figure(
    fig,
    path_base: Path,
    *,
    dpi: int = 160,
) -> Tuple[Path, Path]:
    """Save static density figure as PNG and PDF."""
    path_base.parent.mkdir(parents=True, exist_ok=True)
    png_path = path_base.with_suffix(".png")
    pdf_path = path_base.with_suffix(".pdf")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    return png_path, pdf_path


def render_density_summary_figure(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    frame_to_points: Dict[float, np.ndarray],
    vmax: float,
    hc_mode: bool,
    cfg: VideoGenConfig,
    *,
    norm_factor: float = 1.0,
    recording_stats: Optional[Tuple[int, int]] = None,
    animals_frame_to_points: Optional[Dict[float, np.ndarray]] = None,
    hc_frame_to_points: Optional[Dict[float, np.ndarray]] = None,
    title_suffix: str = "",
):
    """Matplotlib figure for full-recording KDE summary (finer grid than movies)."""
    bins = effective_density_bins_render(cfg, for_summary=True, hc_mode=hc_mode)
    x_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], bins + 1)
    xs: List[float] = []
    ys: List[float] = []
    for arr in frame_to_points.values():
        xs.extend(arr[:, 1].astype(float).tolist())
        ys.extend(arr[:, 2].astype(float).tolist())
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    kde_sigma = effective_kde_sigma_bins(cfg, for_summary=True)
    dens = compute_density_grid(x, y, x_edges, y_edges, cfg.density_method, kde_sigma)
    dens = apply_density_normalization(dens, norm_factor)

    density_label = "KDE-smoothed" if cfg.density_method.lower() == "kde" else "Histogram"
    if hc_mode:
        title = f"HC occupancy density ({density_label}, full recording){title_suffix}"
    else:
        title = f"Larval occupancy density ({density_label}, full recording){title_suffix}"
    cbar_label = density_cbar_label(hc_mode, cfg)

    fig, ax, image, _ = setup_density_figure(
        xlim, ylim, bins, vmax, title, cbar_label, cfg, hc_mode=hc_mode
    )
    image.set_data(mask_empty_density(dens))
    if recording_stats is None:
        a_fp = animals_frame_to_points if animals_frame_to_points is not None else frame_to_points
        h_fp = hc_frame_to_points if hc_frame_to_points is not None else (
            frame_to_points if hc_mode else None
        )
        recording_stats = build_recording_stats(a_fp, h_fp)
    add_summary_recording_stats_box(ax, recording_stats[0], recording_stats[1])
    return fig


def render_density_summary_bgr(
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    frame_to_points: Dict[float, np.ndarray],
    bins: int,
    vmax: float,
    hc_mode: bool,
    cfg: VideoGenConfig,
    *,
    auto_vmax_from_data: bool = False,
    norm_factor: float = 1.0,
    recording_stats: Optional[Tuple[int, int]] = None,
    animals_frame_to_points: Optional[Dict[float, np.ndarray]] = None,
    hc_frame_to_points: Optional[Dict[float, np.ndarray]] = None,
) -> np.ndarray:
    """Full-video density (all points stacked). If auto_vmax_from_data, scale color range to this map only."""
    x_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], bins + 1)
    xs: List[float] = []
    ys: List[float] = []
    for arr in frame_to_points.values():
        xs.extend(arr[:, 1].astype(float).tolist())
        ys.extend(arr[:, 2].astype(float).tolist())
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    summary_bins = bins
    kde_sigma = effective_kde_sigma_bins(cfg, for_summary=True)
    dens = compute_density_grid(x, y, x_edges, y_edges, cfg.density_method, kde_sigma)
    dens = apply_density_normalization(dens, norm_factor)
    if auto_vmax_from_data:
        v_show = max(float(np.max(dens)), 1e-9)
        v_show = finalize_density_vmax([v_show], hc_mode=hc_mode, cfg=cfg, for_summary=True)
    else:
        v_show = vmax

    density_label = "KDE-smoothed" if cfg.density_method.lower() == "kde" else "Histogram"
    if hc_mode:
        title = f"HC occupancy density ({density_label}, full recording)"
    else:
        title = f"Larval occupancy density ({density_label}, full recording)"
    cbar_label = density_cbar_label(hc_mode, cfg)

    fig, ax, image, _ = setup_density_figure(
        xlim, ylim, summary_bins, v_show, title, cbar_label, cfg, hc_mode=hc_mode
    )
    image.set_data(mask_empty_density(dens))
    if recording_stats is None:
        a_fp = animals_frame_to_points if animals_frame_to_points is not None else frame_to_points
        h_fp = hc_frame_to_points if hc_frame_to_points is not None else (
            frame_to_points if hc_mode else None
        )
        recording_stats = build_recording_stats(a_fp, h_fp)
    add_summary_recording_stats_box(ax, recording_stats[0], recording_stats[1])
    fig.canvas.draw()
    out = fig_to_bgr_frame(fig)
    plt.close(fig)
    return out


def subplot_grid_dims(n_panels: int) -> Tuple[int, int]:
    if n_panels <= 0:
        return 1, 1
    ncols = int(math.ceil(math.sqrt(n_panels)))
    nrows = int(math.ceil(n_panels / ncols))
    return nrows, ncols


def _max_density_across_conditions(
    frame_sequence: np.ndarray,
    frame_to_points_by_cond: Dict[str, Dict[float, np.ndarray]],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    window_size_frames: int,
    density_method: str,
    kde_sigma_bins: float,
    sample_cap: int,
    norm_factors: Optional[Dict[str, float]] = None,
    *,
    hc_mode: bool = False,
    cfg: Optional[VideoGenConfig] = None,
) -> float:
    if cfg is None:
        cfg = VideoGenConfig()
    all_peaks: List[float] = []
    for cond, fp in frame_to_points_by_cond.items():
        nf = 1.0 if norm_factors is None else float(norm_factors.get(str(cond), 1.0))
        sample_seq = density_vmax_sample_sequence(
            frame_sequence, fp, hc_mode=hc_mode
        )
        n = len(sample_seq)
        if n == 0:
            continue
        sample_n = min(sample_cap, n)
        sample_idx = np.linspace(0, n - 1, sample_n, dtype=int)
        recent_frames: deque = deque()
        for f in sample_seq[sample_idx]:
            append_frame_to_window(recent_frames, float(f), fp)
            window = get_window_points(f, recent_frames, fp, window_size_frames)
            if len(window) == 0:
                continue
            dens = compute_density_grid(
                window[:, 1].astype(float),
                window[:, 2].astype(float),
                x_edges,
                y_edges,
                density_method,
                kde_sigma_bins,
            )
            dens = apply_density_normalization(dens, nf)
            peak = float(dens.max())
            if peak > 0:
                all_peaks.append(peak)
    return finalize_density_vmax(all_peaks, hc_mode=hc_mode, cfg=cfg, for_summary=False)


def compute_summary_vmax_from_conditions(
    frame_to_points_by_cond: Dict[str, Dict[float, np.ndarray]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    cfg: VideoGenConfig,
    *,
    hc_mode: bool,
    norm_factors: Optional[Dict[str, float]] = None,
) -> float:
    """Shared vmax for static summary panels (full recording, finer grid, summary percentile)."""
    bins = effective_density_bins_render(cfg, for_summary=True, hc_mode=hc_mode)
    x_edges = np.linspace(xlim[0], xlim[1], bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], bins + 1)
    kde_sigma = effective_kde_sigma_bins(cfg, for_summary=True)
    peaks: List[float] = []
    for cond, fp in frame_to_points_by_cond.items():
        xs: List[float] = []
        ys: List[float] = []
        for arr in fp.values():
            xs.extend(arr[:, 1].astype(float).tolist())
            ys.extend(arr[:, 2].astype(float).tolist())
        if not xs:
            continue
        dens = compute_density_grid(
            np.asarray(xs, dtype=float),
            np.asarray(ys, dtype=float),
            x_edges,
            y_edges,
            cfg.density_method,
            kde_sigma,
        )
        nf = 1.0 if norm_factors is None else float(norm_factors.get(str(cond), 1.0))
        dens = apply_density_normalization(dens, nf)
        peak = float(dens.max())
        if peak > 0:
            peaks.append(peak)
    return finalize_density_vmax(peaks, hc_mode=hc_mode, cfg=cfg, for_summary=True)


def resolve_compare_density_vmax(
    frame_sequence: np.ndarray,
    frame_to_points_by_cond: Dict[str, Dict[float, np.ndarray]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    cfg: VideoGenConfig,
    *,
    density_bins: int,
    hc_mode: bool,
    norm_factors: Optional[Dict[str, float]] = None,
) -> float:
    """Shared color-scale vmax for multi-panel compare density (all conditions)."""
    movie_bins = effective_density_bins_render(cfg, for_summary=False, hc_mode=hc_mode)
    x_edges = np.linspace(xlim[0], xlim[1], movie_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], movie_bins + 1)
    window_frames = effective_density_window_frames(cfg, hc_mode)
    kde_sigma = effective_kde_sigma_bins(cfg, for_summary=False)
    movie_bins = effective_density_bins_render(cfg, for_summary=False, hc_mode=hc_mode)
    x_edges = np.linspace(xlim[0], xlim[1], movie_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], movie_bins + 1)
    return _max_density_across_conditions(
        frame_sequence,
        frame_to_points_by_cond,
        x_edges,
        y_edges,
        window_frames,
        cfg.density_method,
        kde_sigma,
        cfg.density_vmax_sample_frames,
        norm_factors=norm_factors,
        hc_mode=hc_mode,
        cfg=cfg,
    )


def compute_shared_density_vmaxs_for_conditions(
    cfg: VideoGenConfig,
    cols: ColumnSelection,
    df_full: pd.DataFrame,
    gcc: str,
    cond_values: List[str],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    unique_frames: np.ndarray,
    norm_factors: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Per-mode vmax shared across --each-condition subfolders (animals + optional HC)."""
    frame_sequence = unique_frames[:: cfg.frame_stride]
    frame_to_points_by_cond: Dict[str, Dict[float, np.ndarray]] = {}
    for val in cond_values:
        df_sub = df_full[df_full[gcc].astype(str) == str(val)].copy()
        _, fp = prepare_frame_cache(df_sub, cols.frame, cols.x, cols.y)
        frame_to_points_by_cond[str(val)] = fp

    out: Dict[str, float] = {}
    out["animals"] = resolve_compare_density_vmax(
        frame_sequence,
        frame_to_points_by_cond,
        xlim,
        ylim,
        cfg,
        density_bins=effective_density_bins_render(cfg, for_summary=False, hc_mode=False),
        hc_mode=False,
        norm_factors=norm_factors,
    )
    try:
        hc_col = resolve_hc_column(df_full.columns.tolist(), cfg)
    except ValueError:
        return out
    by_cond_hc: Dict[str, Dict[float, np.ndarray]] = {}
    for val in cond_values:
        df_sub = df_full[df_full[gcc].astype(str) == str(val)].copy()
        df_h = filter_hc_rows(df_sub, hc_col, cfg)
        _, fp = prepare_frame_cache(df_h, cols.frame, cols.x, cols.y)
        by_cond_hc[str(val)] = fp
    if not any(len(fp) > 0 for fp in by_cond_hc.values()):
        return out
    bins_hc = effective_density_bins_render(cfg, for_summary=False, hc_mode=True)
    out["hc"] = resolve_compare_density_vmax(
        frame_sequence,
        by_cond_hc,
        xlim,
        ylim,
        cfg,
        density_bins=bins_hc,
        hc_mode=True,
        norm_factors=norm_factors,
    )
    return out


def render_track_compare_summary_subplots_bgr(
    conditions: List[str],
    frame_to_points_by_cond: Dict[str, Dict[float, np.ndarray]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    color_map: Dict[str, np.ndarray],
    cfg: VideoGenConfig,
    *,
    stats_by_cond: Optional[Dict[str, Tuple[int, int]]] = None,
    hc_by_cond: Optional[Dict[str, Dict[float, np.ndarray]]] = None,
) -> np.ndarray:
    """Static multi-panel frame: full trajectories per condition (matches compare video layout)."""
    n = len(conditions)
    nrows, ncols = subplot_grid_dims(n)
    fw, fh = cfg.figsize
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fw * ncols * 0.95, fh * nrows * 0.95),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    for i, cond in enumerate(conditions):
        ax = axes_flat[i]
        sc = ax.scatter([], [], s=5, linewidths=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        fp = frame_to_points_by_cond[cond]
        key = str(cond)
        if stats_by_cond is None:
            h_fp = (hc_by_cond or {}).get(key, {})
            n_a, n_h = build_recording_stats(fp, h_fp)
        else:
            n_a, n_h = stats_by_cond.get(key, (0, 0))
        ax.set_title(display_group_label(cond, cfg), fontsize=10)
        hide_arena_axes(ax)
        draw_petri_dish(ax, xlim, ylim, cfg, linewidth=1.0)
        add_summary_recording_stats_box(ax, n_a, n_h)
        xs: List[float] = []
        ys: List[float] = []
        ids: List[str] = []
        for arr in fp.values():
            for row in arr:
                ids.append(str(row[0]))
                xs.append(float(row[1]))
                ys.append(float(row[2]))
        draw_track_points(sc, np.asarray(xs, dtype=float), np.asarray(ys, dtype=float), ids, color_map, cfg)

    fig.suptitle(
        "Larval trajectories by condition (full recording per panel)"
        + (" (uniform color)" if cfg.track_color_mode == "same" else ""),
        fontsize=12,
    )
    fig.canvas.draw()
    out = fig_to_bgr_frame(fig)
    plt.close(fig)
    return out


def render_density_compare_summary_subplots_bgr(
    conditions: List[str],
    frame_to_points_by_cond: Dict[str, Dict[float, np.ndarray]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    cfg: VideoGenConfig,
    *,
    density_bins: int,
    hc_mode: bool,
    norm_factors: Optional[Dict[str, float]] = None,
    stats_by_cond: Optional[Dict[str, Tuple[int, int]]] = None,
    animals_by_cond: Optional[Dict[str, Dict[float, np.ndarray]]] = None,
    hc_by_cond: Optional[Dict[str, Dict[float, np.ndarray]]] = None,
    vmax: Optional[float] = None,
) -> np.ndarray:
    """Static multi-panel frame: full-recording density per condition, shared color scale."""
    x_edges = np.linspace(xlim[0], xlim[1], density_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], density_bins + 1)
    per_dens: List[np.ndarray] = []
    for cond in conditions:
        fp = frame_to_points_by_cond[cond]
        xs: List[float] = []
        ys: List[float] = []
        for arr in fp.values():
            xs.extend(arr[:, 1].astype(float).tolist())
            ys.extend(arr[:, 2].astype(float).tolist())
        x = np.asarray(xs, dtype=float)
        y = np.asarray(ys, dtype=float)
        if len(x) == 0:
            per_dens.append(np.zeros((density_bins, density_bins), dtype=np.float32))
        else:
            dmap = compute_density_grid(
                x, y, x_edges, y_edges, cfg.density_method, effective_kde_sigma_bins(cfg, for_summary=True)
            )
            nf = 1.0 if norm_factors is None else float(norm_factors.get(str(cond), 1.0))
            per_dens.append(apply_density_normalization(dmap, nf))
    peak_vals = [float(d.max()) for d in per_dens if d.size and float(d.max()) > 0]
    summary_vmax = finalize_density_vmax(peak_vals, hc_mode=hc_mode, cfg=cfg, for_summary=True)
    shared_norm = shared_density_color_norm(summary_vmax, hc_mode, cfg)

    n = len(conditions)
    nrows, ncols = subplot_grid_dims(n)
    fw, fh = cfg.figsize
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fw * ncols * 0.95, fh * nrows * 0.95),
        squeeze=False,
        facecolor=DENSITY_AXES_FACE_COLOR,
    )
    plt.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.07, wspace=0.25, hspace=0.30)
    axes_flat = axes.ravel()
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    density_label = "KDE-smoothed" if cfg.density_method.lower() == "kde" else "Histogram"
    if hc_mode:
        supt = f"HC density by condition ({density_label}, full recording per panel)"
    else:
        supt = f"Larval density by condition ({density_label}, full recording per panel)"
    cbar_label = density_cbar_label(hc_mode, cfg)

    images: List = []
    for i, cond in enumerate(conditions):
        ax = axes_flat[i]
        imshow_kw = density_imshow_kwargs(summary_vmax, hc_mode, cfg, norm=shared_norm)
        imshow_kw["extent"] = [xlim[0], xlim[1], ylim[0], ylim[1]]
        ax.set_facecolor(DENSITY_AXES_FACE_COLOR)
        im = ax.imshow(mask_empty_density(per_dens[i]), **imshow_kw)
        apply_density_petri_clip(im, ax, xlim, ylim, cfg)
        style_density_axes(ax, xlim, ylim, cfg)
        ax.set_title(display_group_label(cond, cfg), fontsize=10)
        key = str(cond)
        if stats_by_cond is None:
            a_fp = (animals_by_cond or frame_to_points_by_cond).get(key, {})
            h_fp = (hc_by_cond or {}).get(key, {})
            n_a, n_h = build_recording_stats(a_fp, h_fp)
        else:
            n_a, n_h = stats_by_cond.get(key, (0, 0))
        add_summary_recording_stats_box(ax, n_a, n_h)
        images.append(im)

    fig.colorbar(images[0], ax=list(axes_flat[:n]), fraction=0.035, pad=0.02).set_label(cbar_label)
    fig.suptitle(supt, fontsize=12)
    fig.canvas.draw()
    out = fig_to_bgr_frame(fig)
    plt.close(fig)
    return out


def create_track_animation_compare(
    conditions: List[str],
    frame_to_points_by_cond: Dict[str, Dict[float, np.ndarray]],
    frame_to_points_pooled: Dict[float, np.ndarray],
    unique_frames: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    output_path: Path,
    color_map: Dict[str, np.ndarray],
    cfg: VideoGenConfig,
    *,
    stats_by_cond: Optional[Dict[str, Tuple[int, int]]] = None,
    hc_by_cond: Optional[Dict[str, Dict[float, np.ndarray]]] = None,
) -> None:
    """Multi-panel track video (one subplot per condition) and optional pooled summary at end."""
    validate_track_history(cfg)
    n = len(conditions)
    nrows, ncols = subplot_grid_dims(n)
    fw, fh = cfg.figsize
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fw * ncols * 0.95, fh * nrows * 0.95),
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    scatters: List = []
    for i, cond in enumerate(conditions):
        ax = axes_flat[i]
        sc = ax.scatter([], [], s=5, linewidths=0)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(display_group_label(cond, cfg), fontsize=10)
        hide_arena_axes(ax)
        draw_petri_dish(ax, xlim, ylim, cfg, linewidth=1.0)
        scatters.append(sc)

    supt = (
        "Larval trajectories by condition (cumulative)"
        if cfg.track_history_seconds is None
        else f"Larval trajectories by condition (last {cfg.track_history_seconds:g}s)"
    )
    if cfg.track_color_mode == "same":
        supt = f"{supt} (uniform color)"
    fig.suptitle(supt, fontsize=12)
    clock = add_digital_clock_to_figure(fig, corner="lower_left")

    times_global = frame_end_times_sec(frame_to_points_pooled)
    frame_sequence = unique_frames[:: cfg.frame_stride]
    t_origin = timeline_origin_sec(times_global, frame_sequence)
    if len(frame_sequence) > 0:
        update_digital_clock(clock, float(frame_sequence[0]), times_global, t_origin)
    point_windows: Dict[str, deque] = {c: deque() for c in conditions}
    cum_x: Dict[str, List[float]] = {c: [] for c in conditions}
    cum_y: Dict[str, List[float]] = {c: [] for c in conditions}
    cum_ids: Dict[str, List[str]] = {c: [] for c in conditions}

    fig.canvas.draw()
    wpx, hpx = fig.canvas.get_width_height()
    writer = make_video_writer(output_path, (wpx, hpx), cfg.fps)
    last_bgr: Optional[np.ndarray] = None

    try:
        for i, f in enumerate(frame_sequence):
            if i % 200 == 0:
                print(f"Compare track animation: {i}/{len(frame_sequence)}", flush=True)
            fk = float(f)
            t_cur = times_global.get(fk, fk / float(cfg.recording_fps))

            for ci, cond in enumerate(conditions):
                fp = frame_to_points_by_cond[cond]
                current = fp.get(fk)
                if cfg.track_history_seconds is not None:
                    if current is not None and len(current) > 0:
                        for row in current:
                            point_windows[cond].append(
                                (str(row[0]), float(row[1]), float(row[2]), float(row[3]))
                            )
                    cutoff = t_cur - cfg.track_history_seconds
                    while point_windows[cond] and point_windows[cond][0][3] < cutoff:
                        point_windows[cond].popleft()
                    xs_arr = np.asarray([p[1] for p in point_windows[cond]], dtype=float)
                    ys_arr = np.asarray([p[2] for p in point_windows[cond]], dtype=float)
                    ids_f = [p[0] for p in point_windows[cond]]
                    draw_track_points(scatters[ci], xs_arr, ys_arr, ids_f, color_map, cfg)
                else:
                    if current is not None and len(current) > 0:
                        for row in current:
                            cum_ids[cond].append(str(row[0]))
                            cum_x[cond].append(float(row[1]))
                            cum_y[cond].append(float(row[2]))
                    if not cum_x[cond]:
                        draw_track_points(scatters[ci], np.array([]), np.array([]), [], color_map, cfg)
                    else:
                        draw_track_points(
                            scatters[ci],
                            np.asarray(cum_x[cond], dtype=float),
                            np.asarray(cum_y[cond], dtype=float),
                            cum_ids[cond],
                            color_map,
                            cfg,
                        )

            update_digital_clock(clock, fk, times_global, t_origin)
            last_bgr = fig_to_bgr_frame(fig)
            writer.write(last_bgr)

        if cfg.append_pooled_summary_at_end and last_bgr is not None:
            summary_bgr = render_track_compare_summary_subplots_bgr(
                conditions,
                frame_to_points_by_cond,
                xlim,
                ylim,
                color_map,
                cfg,
                stats_by_cond=stats_by_cond,
                hc_by_cond=hc_by_cond,
            )
            append_summary_to_writer(
                writer,
                last_bgr,
                summary_bgr,
                cfg.fps,
                cfg.summary_pause_seconds,
                cfg.summary_duration_seconds,
            )
    finally:
        writer.release()
    plt.close(fig)


def create_density_animation_compare(
    conditions: List[str],
    frame_to_points_by_cond: Dict[str, Dict[float, np.ndarray]],
    frame_to_points_pooled: Dict[float, np.ndarray],
    unique_frames: np.ndarray,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    output_path: Path,
    cfg: VideoGenConfig,
    *,
    density_bins: int,
    hc_mode: bool,
    norm_factors: Optional[Dict[str, float]] = None,
    stats_by_cond: Optional[Dict[str, Tuple[int, int]]] = None,
    animals_by_cond: Optional[Dict[str, Dict[float, np.ndarray]]] = None,
    hc_by_cond: Optional[Dict[str, Dict[float, np.ndarray]]] = None,
    vmax: Optional[float] = None,
    output_fps: Optional[int] = None,
) -> None:
    """Multi-panel density video and optional pooled full-dataset density at end."""
    frame_sequence = unique_frames[:: cfg.frame_stride]
    movie_bins = density_bins
    kde_sigma = effective_kde_sigma_bins(cfg, for_summary=False)
    x_edges = np.linspace(xlim[0], xlim[1], movie_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], density_bins + 1)
    if vmax is None:
        vmax = resolve_compare_density_vmax(
            frame_sequence,
            frame_to_points_by_cond,
            xlim,
            ylim,
            cfg,
            density_bins=density_bins,
            hc_mode=hc_mode,
            norm_factors=norm_factors,
        )
    if hc_mode:
        print(f"HC compare density color scale vmax={vmax:.6g} (percentile={cfg.hc_density_vmax_percentile})", flush=True)

    density_label = "KDE-smoothed" if cfg.density_method.lower() == "kde" else "Histogram"
    if hc_mode:
        supt = f"HC density by condition ({density_label}, sliding window)"
    else:
        supt = f"Larval density by condition ({density_label}, sliding window)"
    cbar_label = density_cbar_label(hc_mode, cfg)
    shared_norm = shared_density_color_norm(vmax, hc_mode, cfg)

    n = len(conditions)
    nrows, ncols = subplot_grid_dims(n)
    fw, fh = cfg.figsize
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fw * ncols * 0.95, fh * nrows * 0.95),
        squeeze=False,
        facecolor=DENSITY_AXES_FACE_COLOR,
    )
    plt.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.07, wspace=0.25, hspace=0.30)
    axes_flat = axes.ravel()
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    images: List = []
    recent_by_cond: Dict[str, deque] = {c: deque() for c in conditions}
    window_frames = effective_density_window_frames(cfg, hc_mode)
    for i, cond in enumerate(conditions):
        ax = axes_flat[i]
        imshow_kw = density_imshow_kwargs(vmax, hc_mode, cfg, norm=shared_norm)
        imshow_kw["extent"] = [xlim[0], xlim[1], ylim[0], ylim[1]]
        ax.set_facecolor(DENSITY_AXES_FACE_COLOR)
        im = ax.imshow(
            mask_empty_density(np.zeros((density_bins, density_bins))),
            **imshow_kw,
        )
        apply_density_petri_clip(im, ax, xlim, ylim, cfg)
        style_density_axes(ax, xlim, ylim, cfg)
        ax.set_title(display_group_label(cond, cfg), fontsize=10)
        images.append(im)

    fig.colorbar(images[0], ax=list(axes_flat[:n]), fraction=0.035, pad=0.02).set_label(cbar_label)
    fig.suptitle(supt, fontsize=12)
    clock = add_digital_clock_to_figure(fig, corner="lower_left")

    times_global = frame_end_times_sec(frame_to_points_pooled)
    t_origin = timeline_origin_sec(times_global, frame_sequence)
    if len(frame_sequence) > 0:
        update_digital_clock(clock, float(frame_sequence[0]), times_global, t_origin)

    fig.canvas.draw()
    wpx, hpx = fig.canvas.get_width_height()
    writer_fps = int(output_fps) if output_fps is not None else int(cfg.fps)
    writer = make_video_writer(output_path, (wpx, hpx), writer_fps)
    last_bgr: Optional[np.ndarray] = None

    try:
        for i, f in enumerate(frame_sequence):
            if i % 200 == 0:
                print(f"Compare density animation: {i}/{len(frame_sequence)}", flush=True)
            fk = float(f)
            for ci, cond in enumerate(conditions):
                fp = frame_to_points_by_cond[cond]
                append_frame_to_window(recent_by_cond[cond], fk, fp)
                window = get_window_points(
                    fk,
                    recent_by_cond[cond],
                    fp,
                    window_frames,
                )
                nf = 1.0 if norm_factors is None else float(norm_factors.get(str(cond), 1.0))
                if len(window) == 0:
                    dens = np.zeros((density_bins, density_bins))
                else:
                    xw = window[:, 1].astype(float)
                    yw = window[:, 2].astype(float)
                    dens = compute_density_grid(
                        xw, yw, x_edges, y_edges, cfg.density_method, kde_sigma
                    )
                    dens = apply_density_normalization(dens, nf)
                images[ci].set_data(mask_empty_density(dens))
            update_digital_clock(clock, fk, times_global, t_origin)
            last_bgr = fig_to_bgr_frame(fig)
            writer.write(last_bgr)

        if cfg.append_pooled_summary_at_end and last_bgr is not None:
            summary_bins = effective_density_bins_render(cfg, for_summary=True, hc_mode=hc_mode)
            summary_bgr = render_density_compare_summary_subplots_bgr(
                conditions,
                frame_to_points_by_cond,
                xlim,
                ylim,
                cfg,
                density_bins=summary_bins,
                hc_mode=hc_mode,
                norm_factors=norm_factors,
                stats_by_cond=stats_by_cond,
                animals_by_cond=animals_by_cond,
                hc_by_cond=hc_by_cond,
                vmax=vmax,
            )
            append_summary_to_writer(
                writer,
                last_bgr,
                summary_bgr,
                writer_fps,
                cfg.summary_pause_seconds,
                cfg.summary_duration_seconds,
            )
    finally:
        writer.release()
    plt.close(fig)


def _run_compare_subplot_pipeline(cfg: VideoGenConfig, cols: ColumnSelection) -> None:
    gcc = (cfg.group_condition_column or "").strip()
    if not gcc:
        raise ValueError(
            "compare_conditions_subplots requires group_condition_column "
            "(set GROUP_CONDITION_COLUMN or --group-condition-column)."
        )
    df_full = load_and_preprocess(cfg.data_path, cols, cfg, skip_group_value_filter=True)
    if gcc not in df_full.columns:
        raise ValueError(f"group_condition_column {gcc!r} missing after load.")

    raw_vals = df_full[gcc].dropna()
    cond_values = sorted(raw_vals.astype(str).unique(), key=lambda x: x)
    if cfg.group_condition_value is not None:
        want = str(cfg.group_condition_value)
        cond_values = [v for v in cond_values if v == want]
    cond_values = filter_selected_groups(cond_values, cfg)
    if not cond_values:
        warnings.warn("No condition values to compare; nothing rendered.", UserWarning, stacklevel=2)
        return

    norm_method = normalize_density_method_name(cfg.density_normalization)
    norm_factors = compute_group_normalization_factors(
        df_full, cols, gcc, cond_values, norm_method
    )
    if norm_method != "none":
        print(
            "Density normalization (fixed per group, animal-based): "
            + ", ".join(f"{display_group_label(k, cfg)}={v:g}" for k, v in norm_factors.items())
        )

    print(
        f"Compare subplots: {len(cond_values)} panel(s) for column {gcc!r} -> {cfg.output_dir} "
        f"(subplot full-recording summary at end: {cfg.append_pooled_summary_at_end})"
    )

    xlim = (float(df_full[cols.x].min()), float(df_full[cols.x].max()))
    ylim = (float(df_full[cols.y].min()), float(df_full[cols.y].max()))
    xlim = stabilized_limits(*xlim)
    ylim = stabilized_limits(*ylim)
    xlim, ylim = apply_fixed_axis_xy_limits(xlim, ylim, cfg.fixed_axis_xy_limits)

    df_full_effective = df_full
    filtered_parts: List[pd.DataFrame] = []
    for val in cond_values:
        key = str(val)
        df_sub = df_full[df_full[gcc].astype(str) == key].copy()
        df_sub = apply_exclude_track_ids(df_sub, cfg, context_label=display_group_label(key, cfg))
        df_sub_eff, _ = maybe_detect_and_filter_artifacts(
            df_sub,
            cols,
            cfg,
            output_dir=cfg.output_dir,
            context_label=display_group_label(key, cfg),
            xlim=xlim,
            ylim=ylim,
        )
        filtered_parts.append(df_sub_eff)
    if filtered_parts:
        df_full_effective = pd.concat(filtered_parts, ignore_index=True)
    print(
        f"[Artifact diagnostics][compare] effective rows after optional filtering: "
        f"{len(df_full_effective):,}/{len(df_full):,}"
    )

    unique_pooled, frame_to_points_pooled = prepare_frame_cache(
        df_full_effective, cols.frame, cols.x, cols.y
    )
    frame_to_points_by_cond: Dict[str, Dict[float, np.ndarray]] = {}
    for val in cond_values:
        df_sub = df_full_effective[df_full_effective[gcc].astype(str) == str(val)].copy()
        _, fp = prepare_frame_cache(df_sub, cols.frame, cols.x, cols.y)
        frame_to_points_by_cond[str(val)] = fp

    hc_by_cond = prepare_hc_frame_caches_by_condition(df_full_effective, cols, cfg, gcc, cond_values)
    stats_by_cond = build_recording_stats_by_condition(
        cond_values, frame_to_points_by_cond, hc_by_cond
    )

    all_ids: List[str] = []
    for fp in frame_to_points_by_cond.values():
        all_ids.extend(collect_all_ids(fp))
    color_map = build_color_map(
        sorted(set(all_ids)),
        cfg.track_cmap,
        seed=cfg.track_color_seed,
        random_hsv=cfg.track_color_random_hsv,
    )

    if cfg.save_summary_image and len(unique_pooled) > 0:
        ppath = cfg.output_dir / f"{compare_output_stem(cfg, 'larvae_tracks_compare_summary')}.png"
        save_summary_png(
            render_track_compare_summary_subplots_bgr(
                cond_values,
                frame_to_points_by_cond,
                xlim,
                ylim,
                color_map,
                cfg,
                stats_by_cond=stats_by_cond,
                hc_by_cond=hc_by_cond,
            ),
            ppath,
        )
        print(f"Saved {ppath}")

    if cfg.make_track_animation and len(unique_pooled) > 0:
        out = cfg.output_dir / f"{compare_output_stem(cfg, 'larvae_tracks_compare')}.mp4"
        print(f"Creating compare track animation -> {out}")
        create_track_animation_compare(
            cond_values,
            frame_to_points_by_cond,
            frame_to_points_pooled,
            unique_pooled,
            xlim,
            ylim,
            out,
            color_map,
            cfg,
            stats_by_cond=stats_by_cond,
            hc_by_cond=hc_by_cond,
        )
        print("Compare track animation complete.")
        if (
            cfg.make_track_uniform_color_video
            and cfg.track_color_mode != "same"
        ):
            cfg_uni = replace(cfg, track_color_mode="same")
            out_uni = cfg.output_dir / f"{compare_output_stem(cfg, 'larvae_tracks_compare_uniform_color')}.mp4"
            print(f"Creating compare uniform-color track animation -> {out_uni}")
            create_track_animation_compare(
                cond_values,
                frame_to_points_by_cond,
                frame_to_points_pooled,
                unique_pooled,
                xlim,
                ylim,
                out_uni,
                color_map,
                cfg_uni,
                stats_by_cond=stats_by_cond,
                hc_by_cond=hc_by_cond,
            )
            print("Compare uniform-color track animation complete.")
            if cfg.save_summary_image:
                uni_png = cfg.output_dir / f"{compare_output_stem(cfg, 'larvae_tracks_compare_uniform_color_summary')}.png"
                save_summary_png(
                    render_track_compare_summary_subplots_bgr(
                        cond_values,
                        frame_to_points_by_cond,
                        xlim,
                        ylim,
                        color_map,
                        cfg_uni,
                        stats_by_cond=stats_by_cond,
                        hc_by_cond=hc_by_cond,
                    ),
                    uni_png,
                )
                print(f"Saved {uni_png}")

    if cfg.make_density_animation and len(unique_pooled) > 0:
        if cfg.make_all_animations:
            density_modes = ["animals", "hc"]
        else:
            density_modes = [normalize_density_mode(cfg.density_mode)]

        for dm_run in density_modes:
            hc_mode = dm_run == "hc"
            bins_movie = effective_density_bins_render(cfg, for_summary=False, hc_mode=hc_mode)
            bins_summary = effective_density_bins_render(cfg, for_summary=True, hc_mode=hc_mode)
            by_cond: Dict[str, Dict[float, np.ndarray]] = {}
            fp_pooled_use = frame_to_points_pooled

            if hc_mode:
                try:
                    hc_col = resolve_hc_column(df_full_effective.columns.tolist(), cfg)
                except ValueError as e:
                    warnings.warn(f"Skipping HC compare density: {e}", UserWarning, stacklevel=2)
                    continue
                df_hc_full = filter_hc_rows(df_full_effective, hc_col, cfg)
                if len(df_hc_full) == 0:
                    warnings.warn("HC pooled data empty; skipping HC compare density.", UserWarning, stacklevel=2)
                    continue
                _, fp_pooled_use = prepare_frame_cache(df_hc_full, cols.frame, cols.x, cols.y)
                for val in cond_values:
                    df_sub = df_full_effective[df_full_effective[gcc].astype(str) == str(val)].copy()
                    df_h = filter_hc_rows(df_sub, hc_col, cfg)
                    _, fp_c = prepare_frame_cache(df_h, cols.frame, cols.x, cols.y)
                    by_cond[str(val)] = fp_c
            else:
                by_cond = frame_to_points_by_cond

            base = "larvae_density_hc_compare" if hc_mode else "larvae_density_compare"
            if cfg.global_density_vmax and dm_run in cfg.global_density_vmax:
                compare_vmax = float(cfg.global_density_vmax[dm_run])
            else:
                compare_vmax = resolve_compare_density_vmax(
                    unique_pooled[:: cfg.frame_stride],
                    by_cond,
                    xlim,
                    ylim,
                    cfg,
                    density_bins=bins_movie,
                    hc_mode=hc_mode,
                    norm_factors=norm_factors,
                )
            if hc_mode:
                print(
                    f"Shared HC density color scale vmax={compare_vmax:.6g} "
                    f"(percentile={cfg.hc_density_vmax_percentile})",
                    flush=True,
                )
            else:
                print(
                    f"Shared larval density color scale vmax={compare_vmax:.6g} "
                    f"(percentile={cfg.density_vmax_percentile})",
                    flush=True,
                )
            for speed in cfg.playback_speeds:
                if abs(speed - 1.0) < 1e-6:
                    suffix = "realtime"
                    out_fps = int(cfg.fps)
                else:
                    suffix = f"{speed:g}x".replace(".", "p")
                    out_fps = max(1, int(round(cfg.fps * float(speed))))
                out = cfg.output_dir / f"{compare_output_stem(cfg, base)}_{suffix}.mp4"
                print(f"Creating compare density animation ({dm_run}, {suffix}) -> {out}")
                create_density_animation_compare(
                    cond_values,
                    by_cond,
                    fp_pooled_use,
                    unique_pooled,
                    xlim,
                    ylim,
                    out,
                    cfg,
                    density_bins=bins_movie,
                    hc_mode=hc_mode,
                    norm_factors=norm_factors,
                    stats_by_cond=stats_by_cond,
                    animals_by_cond=frame_to_points_by_cond,
                    hc_by_cond=hc_by_cond,
                    vmax=compare_vmax,
                    output_fps=out_fps,
                )
            print("Compare density animation complete.")
            if cfg.save_summary_image:
                png_base = (
                    "larvae_density_hc_compare_summary"
                    if hc_mode
                    else "larvae_density_compare_summary"
                )
                ppath = cfg.output_dir / f"{compare_output_stem(cfg, png_base)}.png"
                save_summary_png(
                    render_density_compare_summary_subplots_bgr(
                        cond_values,
                        by_cond,
                        xlim,
                        ylim,
                        cfg,
                        density_bins=bins_summary,
                        hc_mode=hc_mode,
                        norm_factors=norm_factors,
                        stats_by_cond=stats_by_cond,
                        animals_by_cond=frame_to_points_by_cond,
                        hc_by_cond=hc_by_cond,
                        vmax=compare_vmax,
                    ),
                    ppath,
                )
                print(f"Saved {ppath}")


def append_summary_to_writer(
    writer: cv2.VideoWriter,
    last_anim_bgr: np.ndarray,
    summary_bgr: np.ndarray,
    fps: int,
    pause_seconds: float,
    summary_seconds: float,
) -> None:
    pause_n = max(0, int(round(pause_seconds * fps)))
    summary_n = max(1, int(round(summary_seconds * fps)))
    h, w = last_anim_bgr.shape[:2]
    if summary_bgr.shape[:2] != (h, w):
        summary_bgr = cv2.resize(summary_bgr, (w, h), interpolation=cv2.INTER_AREA)
    for _ in range(pause_n):
        writer.write(last_anim_bgr)
    for _ in range(summary_n):
        writer.write(summary_bgr)


def save_summary_png(bgr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), bgr)


def collect_all_ids(frame_to_points: Dict[float, np.ndarray]) -> List[str]:
    ids = sorted({str(row[0]) for arr in frame_to_points.values() for row in arr})
    return ids


def safe_condition_subdir(label: str, max_len: int = 80) -> str:
    """Filesystem-safe folder name for a condition value."""
    s = str(label).strip()
    for ch in '\\/:*?"<>|\n\r\t':
        s = s.replace(ch, "_")
    s = "_".join(s.split())
    if not s:
        s = "condition"
    return s[:max_len]


def run_density_animation_pass(
    cfg: VideoGenConfig,
    cols: ColumnSelection,
    df: pd.DataFrame,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    dm_normalized: str,
    *,
    norm_factor: float = 1.0,
    vmax_override: Optional[float] = None,
    movie_stem: Optional[str] = None,
    playback_speeds: Optional[Sequence[float]] = None,
    save_pdf_summary: bool = False,
) -> None:
    """Render one density video: dm_normalized is ``animals`` or ``hc``."""
    _, fp_animals = prepare_frame_cache(df, cols.frame, cols.x, cols.y)
    hc_fp_all = prepare_hc_frame_cache_pooled(df, cols, cfg)
    recording_stats = build_recording_stats(fp_animals, hc_fp_all)

    df_density = df
    if dm_normalized == "hc":
        try:
            hc_col = resolve_hc_column(df.columns.tolist(), cfg)
        except ValueError as e:
            warnings.warn(f"Skipping HC density: {e}", UserWarning, stacklevel=2)
            return
        df_hc = filter_hc_rows(df, hc_col, cfg)
        if len(df_hc) == 0:
            warnings.warn(
                "HC-filtered dataframe is empty; skipping HC density video. "
                "Check HC_FILTER_COLUMN / HC_FILTER_VALUE.",
                UserWarning,
                stacklevel=2,
            )
            return
        df_density = df_hc

    unique_frames_d, frame_to_points_d = prepare_frame_cache(
        df_density, cols.frame, cols.x, cols.y
    )
    if len(unique_frames_d) == 0:
        warnings.warn(
            f"No frames available for density animation ({dm_normalized}); skipping.",
            UserWarning,
            stacklevel=2,
        )
        return

    hc_mode = dm_normalized == "hc"
    bins_movie = effective_density_bins_render(cfg, for_summary=False, hc_mode=hc_mode)
    default_stem = "larvae_density_hc" if hc_mode else "larvae_density"
    stem_root = movie_stem or cfg.density_movie_stem
    if stem_root and hc_mode and not str(stem_root).endswith("_hc"):
        stem_root = f"{stem_root}_hc"
    base_stem = stem_root or default_stem
    speeds = list(playback_speeds) if playback_speeds else [1.0]

    for speed in speeds:
        if abs(speed - 1.0) < 1e-6:
            suffix = "realtime"
            out_fps = int(cfg.fps)
        else:
            suffix = f"{speed:g}x".replace(".", "p")
            out_fps = max(1, int(round(cfg.fps * float(speed))))
        density_out = cfg.output_dir / f"{base_stem}_{suffix}.mp4"
        print(
            f"Creating density animation ({dm_normalized}, {suffix}) -> {density_out} "
            f"(movie_bins={bins_movie}, kde_sigma={effective_kde_sigma_bins(cfg, for_summary=False):g}, fps={out_fps})"
        )
        create_density_animation(
            unique_frames_d,
            frame_to_points_d,
            xlim,
            ylim,
            density_out,
            cfg,
            density_bins=bins_movie,
            hc_mode=hc_mode,
            norm_factor=norm_factor,
            recording_stats=recording_stats,
            animals_frame_to_points=fp_animals,
            hc_frame_to_points=hc_fp_all,
            vmax_override=vmax_override,
            output_fps=out_fps,
        )
    print("Density animation complete.")

    if cfg.save_summary_image:
        hc_mode_summary = hc_mode
        summary_bins = effective_density_bins_render(cfg, for_summary=True, hc_mode=hc_mode_summary)
        if (
            cfg.density_vmax_percentile_summary is not None
            or (hc_mode_summary and cfg.hc_density_vmax_percentile_summary is not None)
        ):
            vmax = compute_summary_vmax_from_conditions(
                {"_single": frame_to_points_d},
                xlim,
                ylim,
                cfg,
                hc_mode=hc_mode_summary,
                norm_factors={"_single": norm_factor},
            )
        elif vmax_override is not None:
            vmax = float(vmax_override)
        else:
            x_edges = np.linspace(xlim[0], xlim[1], summary_bins + 1)
            y_edges = np.linspace(ylim[0], ylim[1], summary_bins + 1)
            vmax = _max_density(
                unique_frames_d[:: cfg.frame_stride],
                frame_to_points_d,
                x_edges,
                y_edges,
                effective_density_window_frames(cfg, hc_mode_summary),
                cfg.density_method,
                effective_kde_sigma_bins(cfg, for_summary=False),
                cfg.density_vmax_sample_frames,
                norm_factor=norm_factor,
                hc_mode=hc_mode_summary,
                cfg=cfg,
            )
        summary_base = cfg.output_dir / f"{base_stem}_summary"
        fig = render_density_summary_figure(
            xlim,
            ylim,
            frame_to_points_d,
            vmax,
            hc_mode_summary,
            cfg,
            norm_factor=norm_factor,
            recording_stats=recording_stats,
            animals_frame_to_points=fp_animals,
            hc_frame_to_points=hc_fp_all,
        )
        png_path, pdf_path = save_density_summary_figure(fig, summary_base, dpi=cfg.dpi)
        plt.close(fig)
        print(f"Saved density summary -> {png_path}")
        if save_pdf_summary or pdf_path.exists():
            print(f"Saved density summary PDF -> {pdf_path}")


def _run_pipeline_core(
    cfg: VideoGenConfig,
    cols: ColumnSelection,
    df: pd.DataFrame,
    *,
    context_label: str = "single",
    shared_density_vmax: Optional[Dict[str, float]] = None,
) -> None:
    """Render animations for an already-loaded, filtered dataframe (``cfg.output_dir`` is this run's folder)."""
    xlim = (float(df[cols.x].min()), float(df[cols.x].max()))
    ylim = (float(df[cols.y].min()), float(df[cols.y].max()))
    xlim = stabilized_limits(*xlim)
    ylim = stabilized_limits(*ylim)
    xlim, ylim = apply_fixed_axis_xy_limits(xlim, ylim, cfg.fixed_axis_xy_limits)
    df = apply_exclude_track_ids(df, cfg, context_label=context_label)
    df, _ = maybe_detect_and_filter_artifacts(
        df,
        cols,
        cfg,
        output_dir=cfg.output_dir,
        context_label=context_label,
        xlim=xlim,
        ylim=ylim,
    )

    unique_frames, frame_to_points = prepare_frame_cache(df, cols.frame, cols.x, cols.y)
    hc_fp_all = prepare_hc_frame_cache_pooled(df, cols, cfg)

    norm_factor = 1.0
    norm_method = normalize_density_method_name(cfg.density_normalization)
    if norm_method != "none":
        gcc = (cfg.group_condition_column or "").strip()
        if gcc and gcc in df.columns:
            gvals = [str(cfg.group_condition_value)] if cfg.group_condition_value else [str(v) for v in df[gcc].astype(str).unique()]
            factors = compute_group_normalization_factors(df, cols, gcc, gvals, norm_method)
            if len(factors) == 1:
                norm_factor = float(next(iter(factors.values())))
            elif len(factors) > 1:
                warnings.warn(
                    "density_normalization with multiple groups in one run uses the mean factor. "
                    "Use --compare-subplots with --groups for per-panel factors.",
                    UserWarning,
                    stacklevel=2,
                )
                norm_factor = float(np.mean(list(factors.values())))
            print(f"Density normalization factor: {norm_factor:g} ({norm_method})")

    all_ids = collect_all_ids(frame_to_points)
    color_map = build_color_map(
        all_ids,
        cfg.track_cmap,
        seed=cfg.track_color_seed,
        random_hsv=cfg.track_color_random_hsv,
    )

    if cfg.make_track_animation:
        if len(unique_frames) == 0:
            warnings.warn(
                "No frames available for track animation; skipping.",
                UserWarning,
                stacklevel=2,
            )
        else:
            track_out = cfg.output_dir / "larvae_tracks_window.mp4"
            print(f"Creating track animation -> {track_out}")
            create_track_animation(
                unique_frames,
                frame_to_points,
                xlim,
                ylim,
                track_out,
                color_map,
                cfg,
                hc_frame_to_points=hc_fp_all,
            )
            print("Track animation complete.")
            if (
                cfg.make_track_uniform_color_video
                and cfg.track_color_mode != "same"
            ):
                cfg_uni = replace(cfg, track_color_mode="same")
                uni_out = cfg.output_dir / "larvae_tracks_uniform_color.mp4"
                print(f"Creating uniform-color track animation -> {uni_out}")
                create_track_animation(
                    unique_frames,
                    frame_to_points,
                    xlim,
                    ylim,
                    uni_out,
                    color_map,
                    cfg_uni,
                    hc_frame_to_points=hc_fp_all,
                )
                print("Uniform-color track animation complete.")
        if cfg.save_summary_image and len(unique_frames) > 0:
            summary_path = cfg.output_dir / "larvae_tracks_window_summary.png"
            sum_bgr = render_track_summary_bgr(
                xlim, ylim, frame_to_points, color_map, cfg, hc_frame_to_points=hc_fp_all
            )
            save_summary_png(sum_bgr, summary_path)
            print(f"Saved track summary image -> {summary_path}")
            if (
                cfg.make_track_uniform_color_video
                and cfg.track_color_mode != "same"
            ):
                cfg_uni = replace(cfg, track_color_mode="same")
                uni_png = cfg.output_dir / "larvae_tracks_uniform_color_summary.png"
                save_summary_png(
                    render_track_summary_bgr(
                        xlim,
                        ylim,
                        frame_to_points,
                        color_map,
                        cfg_uni,
                        hc_frame_to_points=hc_fp_all,
                    ),
                    uni_png,
                )
                print(f"Saved track summary image -> {uni_png}")

    if cfg.make_density_animation:
        if cfg.make_all_animations:
            density_modes = ["animals", "hc"]
            print(
                "All animations: writing larvae_density.mp4 and larvae_density_hc.mp4"
                + (" (and track video above)." if cfg.make_track_animation else " (track skipped).")
            )
        else:
            density_modes = [normalize_density_mode(cfg.density_mode)]
        for dm_run in density_modes:
            run_density_animation_pass(
                cfg,
                cols,
                df,
                xlim,
                ylim,
                dm_run,
                norm_factor=norm_factor,
                vmax_override=(shared_density_vmax or {}).get(dm_run),
                playback_speeds=cfg.playback_speeds,
                save_pdf_summary=True,
            )


def run_pipeline(cfg: VideoGenConfig) -> None:
    if not cfg.data_path.exists():
        raise FileNotFoundError(f"Input file not found: {cfg.data_path.resolve()}")
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    cols = detect_columns(cfg.data_path, cfg)

    if cfg.compare_conditions_subplots:
        if cfg.run_each_condition:
            warnings.warn(
                "compare_conditions_subplots is enabled: skipping run_each_condition (per-folder batch).",
                UserWarning,
                stacklevel=2,
            )
        _run_compare_subplot_pipeline(cfg, cols)
        print("Done (compare subplots).")
        return

    if cfg.run_each_condition:
        gcc = (cfg.group_condition_column or "").strip()
        if not gcc:
            raise ValueError(
                "run_each_condition requires a non-empty group_condition_column "
                "(set GROUP_CONDITION_COLUMN or --group-condition-column)."
            )
        df_full = load_and_preprocess(cfg.data_path, cols, cfg, skip_group_value_filter=True)
        if gcc not in df_full.columns:
            raise ValueError(f"group_condition_column {gcc!r} missing after load (internal error).")
        raw_vals = df_full[gcc].dropna()
        cond_values = sorted(raw_vals.astype(str).unique(), key=lambda x: x)
        if cfg.group_condition_value is not None:
            want = str(cfg.group_condition_value)
            cond_values = [v for v in cond_values if v == want]
            if not cond_values:
                warnings.warn(
                    f"No rows match {gcc} == {want!r}; nothing to render.",
                    UserWarning,
                    stacklevel=2,
                )
                print("Done.")
                return
        print(
            f"Per-condition run: {len(cond_values)} distinct value(s) in column {gcc!r} "
            f"-> subfolders under {cfg.output_dir}"
        )
        shared_vmaxs: Optional[Dict[str, float]] = None
        if cfg.make_density_animation and len(cond_values) > 1:
            xlim_batch = (float(df_full[cols.x].min()), float(df_full[cols.x].max()))
            ylim_batch = (float(df_full[cols.y].min()), float(df_full[cols.y].max()))
            xlim_batch = stabilized_limits(*xlim_batch)
            ylim_batch = stabilized_limits(*ylim_batch)
            xlim_batch, ylim_batch = apply_fixed_axis_xy_limits(
                xlim_batch, ylim_batch, cfg.fixed_axis_xy_limits
            )
            unique_batch, _ = prepare_frame_cache(df_full, cols.frame, cols.x, cols.y)
            norm_factors_batch: Optional[Dict[str, float]] = None
            norm_method = normalize_density_method_name(cfg.density_normalization)
            if norm_method != "none":
                norm_factors_batch = compute_group_normalization_factors(
                    df_full, cols, gcc, cond_values, norm_method
                )
            shared_vmaxs = compute_shared_density_vmaxs_for_conditions(
                cfg,
                cols,
                df_full,
                gcc,
                cond_values,
                xlim_batch,
                ylim_batch,
                unique_batch,
                norm_factors=norm_factors_batch,
            )
            if shared_vmaxs:
                parts = [f"animals={shared_vmaxs['animals']:.6g}"]
                if "hc" in shared_vmaxs:
                    parts.append(f"hc={shared_vmaxs['hc']:.6g}")
                print(
                    "Shared density color scale across conditions: " + ", ".join(parts),
                    flush=True,
                )
        base_out = cfg.output_dir
        for val in cond_values:
            df_sub = df_full[df_full[gcc].astype(str) == str(val)].copy()
            if len(df_sub) == 0:
                continue
            subdir = safe_condition_subdir(val)
            out_dir = base_out / subdir
            out_dir.mkdir(parents=True, exist_ok=True)
            cfg_i = replace(cfg, output_dir=out_dir)
            print(f"--- {gcc} = {val!r} -> {out_dir} ({len(df_sub):,} rows) ---")
            _run_pipeline_core(
                cfg_i,
                cols,
                df_sub,
                context_label=str(val),
                shared_density_vmax=shared_vmaxs,
            )
        print("Done (per-condition batch).")
        return

    df = load_and_preprocess(cfg.data_path, cols, cfg)
    _run_pipeline_core(cfg, cols, df, context_label="single")
    print("Done.")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create larval trajectory / density animations.",
        epilog="Examples:  python create_larvae_animations.py data.csv --fast  |  "
        "python create_larvae_animations.py data.csv --fast --density-method kde",
    )
    p.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=None,
        metavar="CSV",
        help="Input tracking CSV (any path). If omitted, uses DATA_PATH from the script config.",
    )
    p.add_argument(
        "--data",
        "-i",
        type=Path,
        default=None,
        dest="data_flag",
        help="Same as positional CSV (alternative way to pass the input file)",
    )
    p.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    p.add_argument(
        "--each-condition",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="each_condition",
        help="Run once per distinct value in --group-condition-column; videos go to subfolders (RUN_EACH_CONDITION).",
    )
    p.add_argument(
        "--group-condition-column",
        type=str,
        default=None,
        help="CSV column for experimental condition (default: GROUP_CONDITION_COLUMN in script).",
    )
    p.add_argument(
        "--group-condition-value",
        type=str,
        default=None,
        help="Optional: keep only this condition (with --each-condition, only this value is processed).",
    )
    p.add_argument(
        "--compare-subplots",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="compare_subplots",
        help="One video with subplots per condition + pooled all-data summary at end (COMPARE_CONDITIONS_SUBPLOTS).",
    )
    p.add_argument(
        "--append-pooled-summary",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="append_pooled_summary",
        help="Append full pooled density/tracks after compare animation (APPEND_POOLED_SUMMARY_AT_END).",
    )
    p.add_argument("--density-mode", choices=("animals", "all", "hc"), default=None)
    p.add_argument(
        "--normalization",
        choices=("none", "num_tracks", "max_visible_larvae"),
        default=None,
        help="Divide density by a fixed per-group animal count (not per frame). HC uses the same factor.",
    )
    p.add_argument(
        "--groups",
        nargs="+",
        default=None,
        metavar="ID",
        help="Compare mode: only these condition IDs (internal CSV values).",
    )
    p.add_argument(
        "--group-label",
        action="append",
        default=None,
        metavar="ID:Label",
        dest="group_labels",
        help="Display label for a condition, e.g. 4d_fed:4d Sated (repeatable).",
    )
    p.add_argument(
        "--compare-output-slug",
        type=str,
        default=None,
        help="Suffix for compare output filenames, e.g. 4d_sated_vs_starved.",
    )
    p.add_argument(
        "--track-history-seconds",
        type=float,
        default=None,
        help="Trailing track window in seconds; None = cumulative animation.",
    )
    p.add_argument(
        "--track-window-seconds",
        type=float,
        default=None,
        dest="track_window_seconds",
        help="Alias for --track-history-seconds.",
    )
    p.add_argument("--track-color-mode", choices=("same", "individual"), default=None)
    p.add_argument("--track-color", type=str, default=None)
    p.add_argument(
        "--track-color-seed",
        type=int,
        default=None,
        help="Seed for per-track random colors in individual mode (default: 42)",
    )
    p.add_argument(
        "--track-cmap-indexed",
        action="store_true",
        help="Use legacy tab20 index colors (sorted ID order) instead of hash-random HSV",
    )
    p.add_argument(
        "--tracks-summary-only",
        action="store_true",
        help="Compare mode: save track comparison summary PNGs only (no videos)",
    )
    p.add_argument("--hc-filter-column", type=str, default=None)
    p.add_argument("--hc-filter-value", type=str, default=None)
    p.add_argument("--hc-density-bins", type=int, default=None)
    p.add_argument("--append-summary-frame", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--summary-pause-seconds", type=float, default=None)
    p.add_argument("--summary-duration-seconds", type=float, default=None)
    p.add_argument(
        "--summary-hold-seconds",
        "--summary_hold_seconds",
        type=float,
        default=None,
        dest="summary_hold_seconds",
        help="Alias for --summary-duration-seconds (hold final summary frame for N seconds).",
    )
    p.add_argument("--save-summary-image", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--detect-artifacts",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="detect_artifacts",
        help="Detect suspicious track jumps/out-of-dish points and write artifact report CSV.",
    )
    p.add_argument(
        "--artifact-velocity-threshold",
        type=float,
        default=None,
        help="Absolute velocity threshold for artifact flagging; default uses robust median+MAD rule.",
    )
    p.add_argument(
        "--artifact-displacement-threshold",
        type=float,
        default=None,
        help="Absolute frame-to-frame displacement threshold for artifact flagging; default uses robust median+MAD rule.",
    )
    p.add_argument(
        "--artifact-mad-multiplier",
        type=float,
        default=None,
        help="MAD multiplier N for robust thresholds median + N*MAD (default: 10).",
    )
    p.add_argument(
        "--exclude-artifact-tracks",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="exclude_artifact_tracks",
        help="When artifact detection is enabled, remove flagged tracks from rendered outputs.",
    )
    p.add_argument(
        "--exclude-track-ids",
        nargs="+",
        default=None,
        metavar="ID",
        dest="exclude_track_ids",
        help="Explicit track/larva IDs to exclude from all videos and summary figures.",
    )
    p.add_argument(
        "--all-animations",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Generate track + all-frame density + HC density (module: MAKE_ALL_ANIMATIONS). Use --no-all-animations to force single density mode from DENSITY_MODE.",
    )
    p.add_argument("--no-track", action="store_true", help="Skip track animation")
    p.add_argument("--no-density", action="store_true", help="Skip density animation")
    p.add_argument(
        "--uniform-color-track",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="uniform_color_track",
        help="Also write larvae_tracks_uniform_color.mp4 (and compare variant); default on.",
    )
    p.add_argument(
        "--recording-fps",
        type=float,
        default=None,
        help="Frames per second of the recording (for time conversion when using frame indices)",
    )
    p.add_argument(
        "--test-duration-seconds",
        type=float,
        default=None,
        metavar="SEC",
        dest="test_duration_seconds",
        help="If set, only use the first SEC seconds of the recording (subset of frames). None = full CSV.",
    )
    p.add_argument(
        "--timestamp-column",
        type=str,
        default=None,
        help="Optional CSV column with time in seconds (overrides frame-based time)",
    )
    p.add_argument(
        "--frame-values-are-seconds",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Treat frame column as seconds (ignored if --timestamp-column is set)",
    )
    p.add_argument(
        "--petri-dish",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="petri_dish",
        help="Draw Petri dish circle on plots (INFER_PETRI_DISH).",
    )
    p.add_argument(
        "--petri-dish-radius",
        type=float,
        default=None,
        help="Petri dish outline radius in x/y units (default: 45 from config)",
    )
    p.add_argument(
        "--fixed-axis-limits",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=None,
        dest="fixed_axis_limits",
        help="Force x and y axes to [MIN, MAX] (default from script, e.g. -47.5 47.5). Omit to keep module default.",
    )
    p.add_argument(
        "--auto-axis-limits",
        action="store_true",
        help="Use data min/max only (set fixed_axis_xy_limits to None)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="Speed preset: fewer frames, coarser density, histogram mode, smaller figure (see FAST_PRESET_OVERRIDES in script).",
    )
    p.add_argument(
        "--window-size-frames",
        type=int,
        default=None,
        metavar="N",
        dest="window_size_frames",
        help="Larval density sliding window: last N frames with data (default from script, e.g. 50)",
    )
    p.add_argument(
        "--hc-window-size-frames",
        type=int,
        default=None,
        metavar="N",
        dest="hc_window_size_frames",
        help="HC density sliding window (frames with data). None = same as --window-size-frames.",
    )
    p.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        metavar="N",
        help="Render every Nth source frame (default from script; fast preset uses 4)",
    )
    p.add_argument(
        "--fps",
        type=int,
        default=None,
        metavar="N",
        help="Output video FPS (default from script; fast preset uses 8)",
    )
    p.add_argument(
        "--density-bins",
        type=int,
        default=None,
        help="2D histogram grid size per axis (default from script)",
    )
    p.add_argument(
        "--density-method",
        choices=("hist", "kde"),
        default=None,
        help="hist is faster than kde (default from script)",
    )
    p.add_argument(
        "--kde-sigma-bins",
        type=float,
        default=None,
        dest="kde_sigma_bins",
        metavar="SIGMA",
        help="KDE Gaussian blur sigma in histogram-bin units (lower = less smoothing, less yellow wash; default 1.8)",
    )
    p.add_argument(
        "--hc-density-vmax-percentile",
        type=float,
        default=None,
        dest="hc_density_vmax_percentile",
        metavar="PCT",
        help="HC density color-scale percentile (higher = less saturation; default 98)",
    )
    p.add_argument(
        "--density-vmax-percentile",
        type=float,
        default=None,
        dest="density_vmax_percentile",
        metavar="PCT",
        help="Larval density color-scale percentile (<100 clips hot outliers; default 100)",
    )
    p.add_argument(
        "--density-use-sqrt-norm",
        action=argparse.BooleanOptionalAction,
        default=None,
        dest="density_use_sqrt_norm",
        help="Use sqrt color normalization for larval density (better high-end contrast).",
    )
    p.add_argument(
        "--kde-sigma-bins-movie",
        type=float,
        default=None,
        dest="kde_sigma_bins_movie",
        help="KDE sigma for density movies (stronger smoothing avoids dotted appearance).",
    )
    p.add_argument(
        "--kde-sigma-bins-summary",
        type=float,
        default=None,
        dest="kde_sigma_bins_summary",
        help="KDE sigma for static summary density figures.",
    )
    p.add_argument(
        "--density-bins-movie",
        type=int,
        default=None,
        dest="density_bins_movie",
        help="Coarser grid resolution for density movies.",
    )
    p.add_argument(
        "--density-bins-summary",
        type=int,
        default=None,
        dest="density_bins_summary",
        help="Finer grid resolution for static summary density figures.",
    )
    p.add_argument(
        "--playback-speeds",
        nargs="+",
        type=float,
        default=None,
        metavar="SPEED",
        dest="playback_speeds",
        help="Output movie playback speed multipliers (1=real-time, 2=2x).",
    )
    return p


def merge_cli_into_config(cfg: VideoGenConfig, args: argparse.Namespace) -> VideoGenConfig:
    overrides: Dict[str, object] = {}
    if getattr(args, "fast", False):
        overrides.update(dict(FAST_PRESET_OVERRIDES))
        print(
            "Fast preset enabled: "
            + ", ".join(f"{k}={v}" for k, v in FAST_PRESET_OVERRIDES.items())
            + " (override with --frame-stride, --fps, --density-bins, --density-method, …)",
            flush=True,
        )
    input_path = args.input_csv
    if args.data_flag is not None:
        if input_path is not None and input_path != args.data_flag:
            warnings.warn(
                "Both a positional CSV and --data/-i were given; using the positional path.",
                UserWarning,
                stacklevel=2,
            )
        elif input_path is None:
            input_path = args.data_flag
    if input_path is not None:
        overrides["data_path"] = Path(input_path).expanduser()
    if args.output_dir is not None:
        overrides["output_dir"] = args.output_dir
    if args.each_condition is True:
        overrides["run_each_condition"] = True
    elif args.each_condition is False:
        overrides["run_each_condition"] = False
    if args.group_condition_column is not None:
        overrides["group_condition_column"] = args.group_condition_column
    if args.group_condition_value is not None:
        overrides["group_condition_value"] = args.group_condition_value
    if args.compare_subplots is True:
        overrides["compare_conditions_subplots"] = True
    elif args.compare_subplots is False:
        overrides["compare_conditions_subplots"] = False
    if args.append_pooled_summary is not None:
        overrides["append_pooled_summary_at_end"] = args.append_pooled_summary
    if args.density_mode is not None:
        overrides["density_mode"] = args.density_mode
    if getattr(args, "normalization", None) is not None:
        overrides["density_normalization"] = normalize_density_method_name(args.normalization)
    if getattr(args, "groups", None) is not None:
        overrides["selected_groups"] = [str(g) for g in args.groups]
    if getattr(args, "group_labels", None):
        label_map: Dict[str, str] = {}
        for spec in args.group_labels:
            internal, label = parse_group_label_arg(spec)
            label_map[internal] = label
        overrides["group_display_labels"] = label_map
    if getattr(args, "compare_output_slug", None) is not None:
        overrides["compare_output_slug"] = str(args.compare_output_slug).strip() or None
    track_win = getattr(args, "track_window_seconds", None)
    if track_win is not None:
        overrides["track_history_seconds"] = float(track_win)
    elif args.track_history_seconds is not None:
        overrides["track_history_seconds"] = args.track_history_seconds
    if getattr(args, "petri_dish", None) is not None:
        overrides["infer_petri_dish"] = bool(args.petri_dish)
    if args.track_color_mode is not None:
        overrides["track_color_mode"] = args.track_color_mode
    if args.track_color is not None:
        overrides["track_color"] = args.track_color
    if getattr(args, "track_color_seed", None) is not None:
        overrides["track_color_seed"] = int(args.track_color_seed)
    if getattr(args, "track_cmap_indexed", False):
        overrides["track_color_random_hsv"] = False
    if getattr(args, "tracks_summary_only", False):
        overrides["compare_conditions_subplots"] = True
        overrides["make_track_animation"] = False
        overrides["make_density_animation"] = False
        overrides["make_track_uniform_color_video"] = False
        overrides["save_summary_image"] = True
    if args.hc_filter_column is not None:
        overrides["hc_filter_column"] = args.hc_filter_column
    if args.hc_filter_value is not None:
        overrides["hc_filter_value"] = args.hc_filter_value
    if args.hc_density_bins is not None:
        overrides["hc_density_bins"] = args.hc_density_bins
    if args.append_summary_frame is not None:
        overrides["append_summary_frame"] = args.append_summary_frame
    if args.summary_pause_seconds is not None:
        overrides["summary_pause_seconds"] = args.summary_pause_seconds
    if args.summary_duration_seconds is not None:
        overrides["summary_duration_seconds"] = args.summary_duration_seconds
    if getattr(args, "summary_hold_seconds", None) is not None:
        overrides["summary_duration_seconds"] = float(args.summary_hold_seconds)
    if args.save_summary_image is not None:
        overrides["save_summary_image"] = args.save_summary_image
    if getattr(args, "detect_artifacts", None) is not None:
        overrides["detect_artifacts"] = bool(args.detect_artifacts)
    if getattr(args, "artifact_velocity_threshold", None) is not None:
        overrides["artifact_velocity_threshold"] = float(args.artifact_velocity_threshold)
    if getattr(args, "artifact_displacement_threshold", None) is not None:
        overrides["artifact_displacement_threshold"] = float(args.artifact_displacement_threshold)
    if getattr(args, "artifact_mad_multiplier", None) is not None:
        overrides["artifact_mad_multiplier"] = float(args.artifact_mad_multiplier)
    if getattr(args, "exclude_artifact_tracks", None) is not None:
        overrides["exclude_artifact_tracks"] = bool(args.exclude_artifact_tracks)
    if getattr(args, "exclude_track_ids", None) is not None:
        overrides["exclude_track_ids"] = [str(tid) for tid in args.exclude_track_ids]
    if args.all_animations is True:
        overrides["make_all_animations"] = True
        overrides["make_track_animation"] = True
        overrides["make_density_animation"] = True
    elif args.all_animations is False:
        overrides["make_all_animations"] = False
    if args.no_track:
        overrides["make_track_animation"] = False
    if args.no_density:
        overrides["make_density_animation"] = False
    if args.uniform_color_track is True:
        overrides["make_track_uniform_color_video"] = True
    elif args.uniform_color_track is False:
        overrides["make_track_uniform_color_video"] = False
    if args.recording_fps is not None:
        overrides["recording_fps"] = float(args.recording_fps)
    if getattr(args, "test_duration_seconds", None) is not None:
        overrides["test_duration_seconds"] = float(args.test_duration_seconds)
    if args.timestamp_column is not None:
        overrides["timestamp_column_override"] = args.timestamp_column
    if args.frame_values_are_seconds is not None:
        overrides["frame_values_are_seconds"] = args.frame_values_are_seconds
    if args.petri_dish_radius is not None:
        overrides["petri_dish_radius"] = float(args.petri_dish_radius)
    if getattr(args, "auto_axis_limits", False):
        overrides["fixed_axis_xy_limits"] = None
    if getattr(args, "fixed_axis_limits", None) is not None:
        lo, hi = float(args.fixed_axis_limits[0]), float(args.fixed_axis_limits[1])
        if lo >= hi:
            raise ValueError("--fixed-axis-limits requires MIN < MAX.")
        overrides["fixed_axis_xy_limits"] = (lo, hi)
    if getattr(args, "window_size_frames", None) is not None:
        overrides["window_size_frames"] = max(1, int(args.window_size_frames))
    if getattr(args, "hc_window_size_frames", None) is not None:
        overrides["hc_window_size_frames"] = max(1, int(args.hc_window_size_frames))
    if getattr(args, "frame_stride", None) is not None:
        overrides["frame_stride"] = max(1, int(args.frame_stride))
    if getattr(args, "fps", None) is not None:
        overrides["fps"] = max(1, int(args.fps))
    if getattr(args, "density_bins", None) is not None:
        overrides["density_bins"] = max(8, int(args.density_bins))
    if getattr(args, "density_method", None) is not None:
        overrides["density_method"] = str(args.density_method)
    if getattr(args, "kde_sigma_bins", None) is not None:
        overrides["kde_sigma_bins"] = max(0.0, float(args.kde_sigma_bins))
    if getattr(args, "kde_sigma_bins_movie", None) is not None:
        overrides["kde_sigma_bins_movie"] = max(0.0, float(args.kde_sigma_bins_movie))
    if getattr(args, "kde_sigma_bins_summary", None) is not None:
        overrides["kde_sigma_bins_summary"] = max(0.0, float(args.kde_sigma_bins_summary))
    if getattr(args, "density_bins_movie", None) is not None:
        overrides["density_bins_movie"] = max(16, int(args.density_bins_movie))
    if getattr(args, "density_bins_summary", None) is not None:
        overrides["density_bins_summary"] = max(16, int(args.density_bins_summary))
    if getattr(args, "density_vmax_percentile", None) is not None:
        overrides["density_vmax_percentile"] = float(args.density_vmax_percentile)
    if getattr(args, "density_use_sqrt_norm", None) is not None:
        overrides["density_use_sqrt_norm"] = bool(args.density_use_sqrt_norm)
    if getattr(args, "playback_speeds", None) is not None:
        overrides["playback_speeds"] = tuple(float(s) for s in args.playback_speeds)
    if getattr(args, "hc_density_vmax_percentile", None) is not None:
        overrides["hc_density_vmax_percentile"] = float(args.hc_density_vmax_percentile)
    return replace(cfg, **overrides)


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = config_from_module_globals()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = merge_cli_into_config(cfg, args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
