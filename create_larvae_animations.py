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
import math
import warnings
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

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
DENSITY_CMAP = "magma"
TRACK_CMAP = "tab20"
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
SUMMARY_DURATION_SECONDS = 3.0
SAVE_SUMMARY_IMAGE = True


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
    density_cmap: str = "magma"
    track_cmap: str = "tab20"
    density_vmax_sample_frames: int = 1200
    density_method: str = "kde"
    kde_sigma_bins: float = 1.8
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
    append_summary_frame: bool = False
    summary_pause_seconds: float = 0.5
    summary_duration_seconds: float = 3.0
    save_summary_image: bool = True


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
        density_cmap=DENSITY_CMAP,
        track_cmap=TRACK_CMAP,
        density_vmax_sample_frames=DENSITY_VMAX_SAMPLE_FRAMES,
        density_method=DENSITY_METHOD,
        kde_sigma_bins=KDE_SIGMA_BINS,
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
        append_summary_frame=APPEND_SUMMARY_FRAME,
        summary_pause_seconds=SUMMARY_PAUSE_SECONDS,
        summary_duration_seconds=SUMMARY_DURATION_SECONDS,
        save_summary_image=SAVE_SUMMARY_IMAGE,
    )


@dataclass
class ColumnSelection:
    frame: str
    larva_id: str
    x: str
    y: str


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


def petri_dish_circle(
    xlim: Tuple[float, float], ylim: Tuple[float, float], cfg: VideoGenConfig
) -> Tuple[float, float, float]:
    """Center at arena midpoint (or origin if limits are symmetric); radius from config."""
    cx = 0.5 * (xlim[0] + xlim[1])
    cy = 0.5 * (ylim[0] + ylim[1])
    if np.isclose(xlim[0], -xlim[1]) and np.isclose(ylim[0], -ylim[1]):
        cx, cy = 0.0, 0.0
    return cx, cy, float(cfg.petri_dish_radius)


def draw_petri_dish(
    ax,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    cfg: VideoGenConfig,
    *,
    linestyle: str = "--",
    linewidth: float = 1.2,
    alpha: float = 0.65,
) -> Optional[Circle]:
    """Draw Petri dish outline in plot coordinates; returns the patch or None if disabled."""
    if not cfg.infer_petri_dish:
        return None
    cx, cy, r = petri_dish_circle(xlim, ylim, cfg)
    patch = Circle((cx, cy), r, fill=False, linestyle=linestyle, linewidth=linewidth, alpha=alpha)
    ax.add_patch(patch)
    return patch


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
    out = [v for v in cond_values if str(v) in want]
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


def build_color_map(ids: Sequence[str], cmap_name: str) -> Dict[str, np.ndarray]:
    cmap = plt.get_cmap(cmap_name, max(20, len(ids)))
    color_map: Dict[str, np.ndarray] = {}
    sorted_ids = sorted(set(ids), key=lambda v: v)
    for i, larva_id in enumerate(sorted_ids):
        rgba = np.array(cmap(i % cmap.N))
        color_map[larva_id] = rgba
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
) -> float:
    n = len(frame_sequence)
    if n == 0:
        return 1.0
    sample_n = min(sample_cap, n)
    sample_idx = np.linspace(0, n - 1, sample_n, dtype=int)
    sampled_frames = frame_sequence[sample_idx]

    recent_frames: deque = deque()
    vmax = 1.0
    for f in sampled_frames:
        append_frame_to_window(recent_frames, float(f), frame_to_points)
        window = get_window_points(f, recent_frames, frame_to_points, window_size_frames)
        if len(window) == 0:
            continue
        x = window[:, 1].astype(float)
        y = window[:, 2].astype(float)
        dens = compute_density_grid(x, y, x_edges, y_edges, density_method, kde_sigma_bins)
        dens = apply_density_normalization(dens, norm_factor)
        vmax = max(vmax, float(dens.max()))
    return vmax


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
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    draw_petri_dish(ax, xlim, ylim, cfg)
    annotation = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    return fig, ax, scatter, annotation


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


def create_track_animation(
    unique_frames: np.ndarray,
    frame_to_points: Dict[float, np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    output_path: Path,
    color_map: Dict[str, np.ndarray],
    cfg: VideoGenConfig,
):
    validate_track_history(cfg)
    title = "Larval trajectories"
    if cfg.track_history_seconds is None:
        title = "Larval cumulative trajectories"
    else:
        title = f"Larval trajectories (last {cfg.track_history_seconds:g}s)"
    if cfg.track_color_mode == "same":
        title = f"{title} (uniform color)"

    fig, ax, scatter, annotation = setup_track_figure(xlim, ylim, title, cfg)
    frame_sequence = unique_frames[:: cfg.frame_stride]
    times_by_frame = frame_end_times_sec(frame_to_points)

    point_window: deque = deque()
    cumulative_ids: List[str] = []
    cumulative_x: List[float] = []
    cumulative_y: List[float] = []

    scatter.set_offsets(np.empty((0, 2)))
    scatter.set_facecolors(np.empty((0, 4)))
    annotation.set_text("")
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

            annotation.set_text(f"Frame: {int(f)}")
            last_bgr = fig_to_bgr_frame(fig)
            writer.write(last_bgr)

        if cfg.append_summary_frame and last_bgr is not None:
            summary_bgr = render_track_summary_bgr(xlim, ylim, frame_to_points, color_map, cfg)
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
) -> np.ndarray:
    """Single static frame: all points in video (full cumulative tracks)."""
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
):
    fig, ax = plt.subplots(figsize=cfg.figsize)
    image = ax.imshow(
        np.zeros((bins, bins)),
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap=cfg.density_cmap,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(cbar_label)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")
    draw_petri_dish(ax, xlim, ylim, cfg)
    annotation = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", color="white")
    return fig, ax, image, annotation


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
):
    frame_sequence = unique_frames[:: cfg.frame_stride]
    x_edges = np.linspace(xlim[0], xlim[1], density_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], density_bins + 1)
    vmax = _max_density(
        frame_sequence,
        frame_to_points,
        x_edges,
        y_edges,
        cfg.window_size_frames,
        cfg.density_method,
        cfg.kde_sigma_bins,
        cfg.density_vmax_sample_frames,
        norm_factor=norm_factor,
    )

    density_label = "KDE-smoothed" if cfg.density_method.lower() == "kde" else "Histogram"
    if hc_mode:
        title = f"HC occupancy density ({density_label}, sliding window)"
    else:
        title = f"Larval occupancy density ({density_label}, sliding window)"
    cbar_label = density_cbar_label(hc_mode, cfg)

    fig, ax, image, annotation = setup_density_figure(
        xlim, ylim, density_bins, vmax, title, cbar_label, cfg
    )
    recent_frames: deque = deque()

    fig.canvas.draw()
    width_px, height_px = fig.canvas.get_width_height()
    writer = make_video_writer(output_path, (width_px, height_px), cfg.fps)
    last_bgr: Optional[np.ndarray] = None

    try:
        for i, f in enumerate(frame_sequence):
            if i % 200 == 0:
                print(f"Density animation progress: {i}/{len(frame_sequence)}", flush=True)
            fk = float(f)
            append_frame_to_window(recent_frames, fk, frame_to_points)
            window = get_window_points(fk, recent_frames, frame_to_points, cfg.window_size_frames)
            if len(window) == 0:
                dens = np.zeros((density_bins, density_bins))
            else:
                x = window[:, 1].astype(float)
                y = window[:, 2].astype(float)
                dens = compute_density_grid(x, y, x_edges, y_edges, cfg.density_method, cfg.kde_sigma_bins)
                dens = apply_density_normalization(dens, norm_factor)
            image.set_data(dens)
            annotation.set_text(f"Frame: {int(f)}")
            last_bgr = fig_to_bgr_frame(fig)
            writer.write(last_bgr)

        if cfg.append_summary_frame and last_bgr is not None:
            summary_bgr = render_density_summary_bgr(
                xlim, ylim, frame_to_points, density_bins, vmax, hc_mode, cfg, norm_factor=norm_factor
            )
            append_summary_to_writer(
                writer, last_bgr, summary_bgr, cfg.fps, cfg.summary_pause_seconds, cfg.summary_duration_seconds
            )
    finally:
        writer.release()
    plt.close(fig)


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
    dens = compute_density_grid(x, y, x_edges, y_edges, cfg.density_method, cfg.kde_sigma_bins)
    dens = apply_density_normalization(dens, norm_factor)
    v_show = max(float(np.max(dens)), 1e-6) if auto_vmax_from_data else vmax

    density_label = "KDE-smoothed" if cfg.density_method.lower() == "kde" else "Histogram"
    if hc_mode:
        title = f"HC occupancy density ({density_label}, full recording)"
    else:
        title = f"Larval occupancy density ({density_label}, full recording)"
    cbar_label = density_cbar_label(hc_mode, cfg)

    fig, ax, image, _ = setup_density_figure(
        xlim, ylim, bins, v_show, title, cbar_label, cfg
    )
    image.set_data(dens)
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
) -> float:
    vmax = 1.0
    for cond, fp in frame_to_points_by_cond.items():
        nf = 1.0 if norm_factors is None else float(norm_factors.get(str(cond), 1.0))
        vmax = max(
            vmax,
            _max_density(
                frame_sequence,
                fp,
                x_edges,
                y_edges,
                window_size_frames,
                density_method,
                kde_sigma_bins,
                sample_cap,
                norm_factor=nf,
            ),
        )
    return vmax


def render_track_compare_summary_subplots_bgr(
    conditions: List[str],
    frame_to_points_by_cond: Dict[str, Dict[float, np.ndarray]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    color_map: Dict[str, np.ndarray],
    cfg: VideoGenConfig,
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
        ax.set_title(display_group_label(cond, cfg), fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        draw_petri_dish(ax, xlim, ylim, cfg, linewidth=1.0)
        fp = frame_to_points_by_cond[cond]
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
            dmap = compute_density_grid(x, y, x_edges, y_edges, cfg.density_method, cfg.kde_sigma_bins)
            nf = 1.0 if norm_factors is None else float(norm_factors.get(str(cond), 1.0))
            per_dens.append(apply_density_normalization(dmap, nf))
    vmax = max(float(d.max()) for d in per_dens) if per_dens else 1.0
    vmax = max(vmax, 1e-6)

    n = len(conditions)
    nrows, ncols = subplot_grid_dims(n)
    fw, fh = cfg.figsize
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fw * ncols * 0.95, fh * nrows * 0.95),
        squeeze=False,
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
        im = ax.imshow(
            per_dens[i],
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            cmap=cfg.density_cmap,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(display_group_label(cond, cfg), fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        draw_petri_dish(ax, xlim, ylim, cfg, linewidth=1.0)
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
        ax.set_xlabel("x")
        ax.set_ylabel("y")
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
    ann = fig.text(0.02, 0.02, "", fontsize=10)

    times_global = frame_end_times_sec(frame_to_points_pooled)
    frame_sequence = unique_frames[:: cfg.frame_stride]
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

            ann.set_text(f"Frame: {int(fk)}")
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
) -> None:
    """Multi-panel density video and optional pooled full-dataset density at end."""
    frame_sequence = unique_frames[:: cfg.frame_stride]
    x_edges = np.linspace(xlim[0], xlim[1], density_bins + 1)
    y_edges = np.linspace(ylim[0], ylim[1], density_bins + 1)
    vmax = _max_density_across_conditions(
        frame_sequence,
        frame_to_points_by_cond,
        x_edges,
        y_edges,
        cfg.window_size_frames,
        cfg.density_method,
        cfg.kde_sigma_bins,
        cfg.density_vmax_sample_frames,
        norm_factors=norm_factors,
    )

    density_label = "KDE-smoothed" if cfg.density_method.lower() == "kde" else "Histogram"
    if hc_mode:
        supt = f"HC density by condition ({density_label}, sliding window)"
    else:
        supt = f"Larval density by condition ({density_label}, sliding window)"
    cbar_label = density_cbar_label(hc_mode, cfg)

    n = len(conditions)
    nrows, ncols = subplot_grid_dims(n)
    fw, fh = cfg.figsize
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(fw * ncols * 0.95, fh * nrows * 0.95),
        squeeze=False,
    )
    plt.subplots_adjust(left=0.06, right=0.90, top=0.90, bottom=0.07, wspace=0.25, hspace=0.30)
    axes_flat = axes.ravel()
    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    images: List = []
    recent_by_cond: Dict[str, deque] = {c: deque() for c in conditions}
    for i, cond in enumerate(conditions):
        ax = axes_flat[i]
        im = ax.imshow(
            np.zeros((density_bins, density_bins)),
            origin="lower",
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            cmap=cfg.density_cmap,
            vmin=0,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(display_group_label(cond, cfg), fontsize=10)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        draw_petri_dish(ax, xlim, ylim, cfg, linewidth=1.0)
        images.append(im)

    fig.colorbar(images[0], ax=list(axes_flat[:n]), fraction=0.035, pad=0.02).set_label(cbar_label)
    fig.suptitle(supt, fontsize=12)
    ann = fig.text(0.02, 0.02, "", color="white", fontsize=10)

    fig.canvas.draw()
    wpx, hpx = fig.canvas.get_width_height()
    writer = make_video_writer(output_path, (wpx, hpx), cfg.fps)
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
                    cfg.window_size_frames,
                )
                nf = 1.0 if norm_factors is None else float(norm_factors.get(str(cond), 1.0))
                if len(window) == 0:
                    dens = np.zeros((density_bins, density_bins))
                else:
                    xw = window[:, 1].astype(float)
                    yw = window[:, 2].astype(float)
                    dens = compute_density_grid(
                        xw, yw, x_edges, y_edges, cfg.density_method, cfg.kde_sigma_bins
                    )
                    dens = apply_density_normalization(dens, nf)
                images[ci].set_data(dens)
            ann.set_text(f"Frame: {int(fk)}")
            last_bgr = fig_to_bgr_frame(fig)
            writer.write(last_bgr)

        if cfg.append_pooled_summary_at_end and last_bgr is not None:
            summary_bgr = render_density_compare_summary_subplots_bgr(
                conditions,
                frame_to_points_by_cond,
                xlim,
                ylim,
                cfg,
                density_bins=density_bins,
                hc_mode=hc_mode,
                norm_factors=norm_factors,
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

    unique_pooled, frame_to_points_pooled = prepare_frame_cache(
        df_full, cols.frame, cols.x, cols.y
    )
    frame_to_points_by_cond: Dict[str, Dict[float, np.ndarray]] = {}
    for val in cond_values:
        df_sub = df_full[df_full[gcc].astype(str) == str(val)].copy()
        _, fp = prepare_frame_cache(df_sub, cols.frame, cols.x, cols.y)
        frame_to_points_by_cond[str(val)] = fp

    all_ids: List[str] = []
    for fp in frame_to_points_by_cond.values():
        all_ids.extend(collect_all_ids(fp))
    color_map = build_color_map(sorted(set(all_ids)), cfg.track_cmap)

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
        )
        print("Compare track animation complete.")
        if cfg.save_summary_image:
            ppath = cfg.output_dir / f"{compare_output_stem(cfg, 'larvae_tracks_compare_summary')}.png"
            save_summary_png(
                render_track_compare_summary_subplots_bgr(
                    cond_values,
                    frame_to_points_by_cond,
                    xlim,
                    ylim,
                    color_map,
                    cfg,
                ),
                ppath,
            )
            print(f"Saved {ppath}")
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
            bins_use = effective_hc_density_bins(cfg) if hc_mode else cfg.density_bins
            by_cond: Dict[str, Dict[float, np.ndarray]] = {}
            fp_pooled_use = frame_to_points_pooled

            if hc_mode:
                try:
                    hc_col = resolve_hc_column(df_full.columns.tolist(), cfg)
                except ValueError as e:
                    warnings.warn(f"Skipping HC compare density: {e}", UserWarning, stacklevel=2)
                    continue
                df_hc_full = filter_hc_rows(df_full, hc_col, cfg)
                if len(df_hc_full) == 0:
                    warnings.warn("HC pooled data empty; skipping HC compare density.", UserWarning, stacklevel=2)
                    continue
                _, fp_pooled_use = prepare_frame_cache(df_hc_full, cols.frame, cols.x, cols.y)
                for val in cond_values:
                    df_sub = df_full[df_full[gcc].astype(str) == str(val)].copy()
                    df_h = filter_hc_rows(df_sub, hc_col, cfg)
                    _, fp_c = prepare_frame_cache(df_h, cols.frame, cols.x, cols.y)
                    by_cond[str(val)] = fp_c
            else:
                by_cond = frame_to_points_by_cond

            base = "larvae_density_hc_compare" if hc_mode else "larvae_density_compare"
            out = cfg.output_dir / f"{compare_output_stem(cfg, base)}.mp4"
            print(f"Creating compare density animation ({dm_run}) -> {out}")
            create_density_animation_compare(
                cond_values,
                by_cond,
                fp_pooled_use,
                unique_pooled,
                xlim,
                ylim,
                out,
                cfg,
                density_bins=bins_use,
                hc_mode=hc_mode,
                norm_factors=norm_factors,
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
                        density_bins=bins_use,
                        hc_mode=hc_mode,
                        norm_factors=norm_factors,
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
) -> None:
    """Render one density video: dm_normalized is ``animals`` or ``hc``."""
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

    density_out = cfg.output_dir / ("larvae_density_hc.mp4" if dm_normalized == "hc" else "larvae_density.mp4")
    bins_use = effective_hc_density_bins(cfg) if dm_normalized == "hc" else cfg.density_bins
    print(f"Creating density animation ({dm_normalized}) -> {density_out} (bins={bins_use})")
    create_density_animation(
        unique_frames_d,
        frame_to_points_d,
        xlim,
        ylim,
        density_out,
        cfg,
        density_bins=bins_use,
        hc_mode=(dm_normalized == "hc"),
        norm_factor=norm_factor,
    )
    print("Density animation complete.")

    if cfg.save_summary_image:
        x_edges = np.linspace(xlim[0], xlim[1], bins_use + 1)
        y_edges = np.linspace(ylim[0], ylim[1], bins_use + 1)
        vmax = _max_density(
            unique_frames_d[:: cfg.frame_stride],
            frame_to_points_d,
            x_edges,
            y_edges,
            cfg.window_size_frames,
            cfg.density_method,
            cfg.kde_sigma_bins,
            cfg.density_vmax_sample_frames,
            norm_factor=norm_factor,
        )
        summary_path = cfg.output_dir / (
            "larvae_density_hc_summary.png" if dm_normalized == "hc" else "larvae_density_summary.png"
        )
        sum_bgr = render_density_summary_bgr(
            xlim,
            ylim,
            frame_to_points_d,
            bins_use,
            vmax,
            dm_normalized == "hc",
            cfg,
            norm_factor=norm_factor,
        )
        save_summary_png(sum_bgr, summary_path)
        print(f"Saved density summary image -> {summary_path}")


def _run_pipeline_core(cfg: VideoGenConfig, cols: ColumnSelection, df: pd.DataFrame) -> None:
    """Render animations for an already-loaded, filtered dataframe (``cfg.output_dir`` is this run's folder)."""
    xlim = (float(df[cols.x].min()), float(df[cols.x].max()))
    ylim = (float(df[cols.y].min()), float(df[cols.y].max()))
    xlim = stabilized_limits(*xlim)
    ylim = stabilized_limits(*ylim)
    xlim, ylim = apply_fixed_axis_xy_limits(xlim, ylim, cfg.fixed_axis_xy_limits)

    unique_frames, frame_to_points = prepare_frame_cache(df, cols.frame, cols.x, cols.y)

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
    color_map = build_color_map(all_ids, cfg.track_cmap)

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
                unique_frames, frame_to_points, xlim, ylim, track_out, color_map, cfg
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
                    unique_frames, frame_to_points, xlim, ylim, uni_out, color_map, cfg_uni
                )
                print("Uniform-color track animation complete.")
        if cfg.save_summary_image and len(unique_frames) > 0:
            summary_path = cfg.output_dir / "larvae_tracks_window_summary.png"
            sum_bgr = render_track_summary_bgr(xlim, ylim, frame_to_points, color_map, cfg)
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
                        xlim, ylim, frame_to_points, color_map, cfg_uni
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
            run_density_animation_pass(cfg, cols, df, xlim, ylim, dm_run, norm_factor=norm_factor)


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
            _run_pipeline_core(cfg_i, cols, df_sub)
        print("Done (per-condition batch).")
        return

    df = load_and_preprocess(cfg.data_path, cols, cfg)
    _run_pipeline_core(cfg, cols, df)
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
    p.add_argument("--hc-filter-column", type=str, default=None)
    p.add_argument("--hc-filter-value", type=str, default=None)
    p.add_argument("--hc-density-bins", type=int, default=None)
    p.add_argument("--append-summary-frame", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--summary-pause-seconds", type=float, default=None)
    p.add_argument("--summary-duration-seconds", type=float, default=None)
    p.add_argument("--save-summary-image", action=argparse.BooleanOptionalAction, default=None)
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
    if args.save_summary_image is not None:
        overrides["save_summary_image"] = args.save_summary_image
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
    if getattr(args, "frame_stride", None) is not None:
        overrides["frame_stride"] = max(1, int(args.frame_stride))
    if getattr(args, "fps", None) is not None:
        overrides["fps"] = max(1, int(args.fps))
    if getattr(args, "density_bins", None) is not None:
        overrides["density_bins"] = max(8, int(args.density_bins))
    if getattr(args, "density_method", None) is not None:
        overrides["density_method"] = str(args.density_method)
    return replace(cfg, **overrides)


def main(argv: Optional[Sequence[str]] = None) -> None:
    cfg = config_from_module_globals()
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    cfg = merge_cli_into_config(cfg, args)
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
