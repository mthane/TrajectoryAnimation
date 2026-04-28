#!/usr/bin/env python3
"""Create larval trajectory animations from tracking data.

Outputs:
  - animations/larvae_tracks_window.mp4
  - animations/larvae_density.mp4
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import cv2


# =========================
# Config (edit as needed)
# =========================
DATA_PATH = Path("embedding_dan.csv")
OUTPUT_DIR = Path("animations")

# Optional fixed column names. Set to None for auto-detection.
X_COLUMN_OVERRIDE = None  # Example: "centroid_x"
Y_COLUMN_OVERRIDE = None  # Example: "centroid_y"
FRAME_COLUMN_OVERRIDE = None  # Example: "frame"
ID_COLUMN_OVERRIDE = None  # Example: "id"

# Optional filter by group condition/category.
GROUP_CONDITION_COLUMN = "condition"
GROUP_CONDITION_VALUE = None  # Example: "EM_Rewarded"

WINDOW_SIZE_FRAMES = 50
FRAME_STRIDE = 1
# Recording frame rate (frames per second) and output video FPS.
RECORDING_FPS = 16
FPS = 16
DPI = 140
FIGSIZE = (8, 8)
TEST_DURATION_SECONDS = 30

MAKE_TRACK_ANIMATION = True
MAKE_DENSITY_ANIMATION = True

DENSITY_BINS = 90
DENSITY_CMAP = "magma"
TRACK_CMAP = "tab20"
DENSITY_VMAX_SAMPLE_FRAMES = 1200
# "hist" for raw 2D histogram, "kde" for smoothed KDE-like map.
DENSITY_METHOD = "kde"
# Approximate KDE bandwidth in histogram-bin units (used when DENSITY_METHOD="kde").
KDE_SIGMA_BINS = 1.8

INFER_PETRI_DISH = True
PETRI_MARGIN_FRACTION = 0.02


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


def detect_columns(csv_path: Path) -> ColumnSelection:
    columns = pd.read_csv(csv_path, nrows=0).columns.tolist()

    frame_col = FRAME_COLUMN_OVERRIDE or _pick_first_existing(
        ["frame", "t", "time", "timestamp", "frame_idx"], columns
    )
    id_col = ID_COLUMN_OVERRIDE or _pick_first_existing(
        ["id", "track_id", "larva_id", "track", "animal_id"], columns
    )
    # Position columns for this dataset.
    x_col = X_COLUMN_OVERRIDE or _pick_first_existing(
        ["centroid_x", "centoid_x", "spinepoint_x6_conv", "spinepoint_x_6_conv"],
        columns,
    )
    y_col = Y_COLUMN_OVERRIDE or _pick_first_existing(
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


def load_and_preprocess(csv_path: Path, cols: ColumnSelection) -> pd.DataFrame:
    usecols = [cols.frame, cols.larva_id, cols.x, cols.y]
    if GROUP_CONDITION_VALUE is not None:
        if GROUP_CONDITION_COLUMN not in pd.read_csv(csv_path, nrows=0).columns.tolist():
            raise ValueError(
                f"GROUP_CONDITION_COLUMN='{GROUP_CONDITION_COLUMN}' not found in CSV columns."
            )
        usecols.append(GROUP_CONDITION_COLUMN)
    df = pd.read_csv(csv_path, usecols=usecols, low_memory=False)

    if GROUP_CONDITION_VALUE is not None:
        before_group = len(df)
        df = df[df[GROUP_CONDITION_COLUMN].astype(str) == str(GROUP_CONDITION_VALUE)].copy()
        print(
            f"Group filter: {GROUP_CONDITION_COLUMN} == {GROUP_CONDITION_VALUE}; "
            f"kept {len(df):,}/{before_group:,} rows."
        )

    # Ensure numeric coordinates and sortable frame values.
    df[cols.frame] = pd.to_numeric(df[cols.frame], errors="coerce")
    df[cols.x] = pd.to_numeric(df[cols.x], errors="coerce")
    df[cols.y] = pd.to_numeric(df[cols.y], errors="coerce")

    before = len(df)
    df = df.dropna(subset=[cols.frame, cols.x, cols.y, cols.larva_id]).copy()
    after_drop = len(df)
    print(f"Rows loaded: {before:,}")
    print(f"Rows after dropping missing frame/id/x/y: {after_drop:,}")

    # Normalize ID to category/int code for stable coloring.
    df["larva_id_norm"] = df[cols.larva_id].astype(str)
    df = df.sort_values(by=[cols.frame, "larva_id_norm"]).reset_index(drop=True)

    if TEST_DURATION_SECONDS is not None:
        if TEST_DURATION_SECONDS <= 0:
            raise ValueError("TEST_DURATION_SECONDS must be > 0 or set to None.")
        start_frame = float(df[cols.frame].min())
        end_frame = start_frame + (TEST_DURATION_SECONDS * RECORDING_FPS)
        before_window = len(df)
        df = df[df[cols.frame] <= end_frame].copy()
        print(
            "Testing subset enabled: "
            f"first {TEST_DURATION_SECONDS}s at {RECORDING_FPS} fps "
            f"(<= frame {end_frame:.0f}); kept {len(df):,}/{before_window:,} rows."
        )

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
    frame_to_points: Dict[float, np.ndarray] = {}
    for frame_value, g in df.groupby(frame_col, sort=True):
        frame_to_points[float(frame_value)] = g[["larva_id_norm", x_col, y_col]].to_numpy()
    unique_frames = np.array(sorted(frame_to_points.keys()), dtype=float)
    return unique_frames, frame_to_points


def infer_petri_circle(x_min: float, x_max: float, y_min: float, y_max: float):
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    radius = 0.5 * min(x_max - x_min, y_max - y_min) * (1 - PETRI_MARGIN_FRACTION)
    return cx, cy, radius


def stabilized_limits(vmin: float, vmax: float) -> Tuple[float, float]:
    if np.isclose(vmin, vmax):
        pad = 1.0 if np.isclose(vmin, 0.0) else abs(vmin) * 0.05
        return vmin - pad, vmax + pad
    return vmin, vmax


def setup_style():
    plt.style.use("seaborn-v0_8-whitegrid")
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["grid.alpha"] = 0.25
    mpl.rcParams["font.size"] = 11


def get_window_points(
    target_frame: float,
    recent_frames: deque,
    frame_to_points: Dict[float, np.ndarray],
    window_size: int,
) -> np.ndarray:
    while recent_frames and recent_frames[0] < (target_frame - window_size + 1):
        recent_frames.popleft()
    if not recent_frames:
        return np.empty((0, 4), dtype=object)
    arrays = [frame_to_points[f] for f in recent_frames if f in frame_to_points]
    if not arrays:
        return np.empty((0, 4), dtype=object)
    stacked = np.vstack(arrays)
    # Add frame col to compute fade based on point age.
    frame_col = np.concatenate(
        [np.full(len(frame_to_points[f]), f, dtype=float) for f in recent_frames]
    )
    return np.column_stack([stacked, frame_col])


def build_color_map(ids: List[str]) -> Dict[str, np.ndarray]:
    cmap = plt.get_cmap(TRACK_CMAP, max(20, len(ids)))
    color_map: Dict[str, np.ndarray] = {}
    for i, larva_id in enumerate(ids):
        rgba = np.array(cmap(i % cmap.N))
        color_map[larva_id] = rgba
    return color_map


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


def create_track_animation(
    unique_frames: np.ndarray,
    frame_to_points: Dict[float, np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    output_path: Path,
):
    ids = sorted(
        {str(row[0]) for arr in frame_to_points.values() for row in arr},
        key=lambda v: v,
    )
    colors = build_color_map(ids)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    scatter = ax.scatter([], [], s=6, linewidths=0)

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Larval cumulative trajectories")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")

    if INFER_PETRI_DISH:
        cx, cy, r = infer_petri_circle(*xlim, *ylim)
        ax.add_patch(Circle((cx, cy), r, fill=False, linestyle="--", linewidth=1.2, alpha=0.65))

    annotation = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    frame_sequence = unique_frames[::FRAME_STRIDE]
    cumulative_x: List[float] = []
    cumulative_y: List[float] = []
    cumulative_ids: List[str] = []

    # Initialize frame to ensure canvas has expected dimensions.
    scatter.set_offsets(np.empty((0, 2)))
    scatter.set_facecolors(np.empty((0, 4)))
    annotation.set_text("")
    fig.canvas.draw()
    width_px, height_px = fig.canvas.get_width_height()
    writer = make_video_writer(output_path, (width_px, height_px), FPS)

    try:
        for i, f in enumerate(frame_sequence):
            if i % 200 == 0:
                print(f"Track animation progress: {i}/{len(frame_sequence)}", flush=True)
            current = frame_to_points.get(float(f))
            if current is not None and len(current) > 0:
                cumulative_ids.extend(current[:, 0].astype(str).tolist())
                cumulative_x.extend(current[:, 1].astype(float).tolist())
                cumulative_y.extend(current[:, 2].astype(float).tolist())

            if not cumulative_x:
                scatter.set_offsets(np.empty((0, 2)))
                scatter.set_facecolors(np.empty((0, 4)))
            else:
                x = np.asarray(cumulative_x, dtype=float)
                y = np.asarray(cumulative_y, dtype=float)
                rgba = np.vstack([colors[k].copy() for k in cumulative_ids])
                rgba[:, 3] = 0.35
                scatter.set_offsets(np.column_stack([x, y]))
                scatter.set_facecolors(rgba)

            annotation.set_text(f"Frame: {int(f)}")
            frame_bgr = fig_to_bgr_frame(fig)
            writer.write(frame_bgr)
    finally:
        writer.release()
    plt.close(fig)


def _max_density(
    frame_sequence: np.ndarray,
    frame_to_points: Dict[float, np.ndarray],
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> float:
    n = len(frame_sequence)
    if n == 0:
        return 1.0
    sample_n = min(DENSITY_VMAX_SAMPLE_FRAMES, n)
    sample_idx = np.linspace(0, n - 1, sample_n, dtype=int)
    sampled_frames = frame_sequence[sample_idx]

    recent_frames: deque = deque()
    vmax = 1.0
    for f in sampled_frames:
        recent_frames.append(f)
        window = get_window_points(f, recent_frames, frame_to_points, WINDOW_SIZE_FRAMES)
        if len(window) == 0:
            continue
        x = window[:, 1].astype(float)
        y = window[:, 2].astype(float)
        dens = compute_density_grid(x, y, x_edges, y_edges)
        vmax = max(vmax, float(dens.max()))
    return vmax


def compute_density_grid(
    x: np.ndarray, y: np.ndarray, x_edges: np.ndarray, y_edges: np.ndarray
) -> np.ndarray:
    h, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
    dens = h.T.astype(np.float32)
    if DENSITY_METHOD.lower() == "kde":
        # Gaussian smoothing over occupancy bins approximates a KDE surface.
        dens = cv2.GaussianBlur(dens, ksize=(0, 0), sigmaX=KDE_SIGMA_BINS, sigmaY=KDE_SIGMA_BINS)
    return dens


def create_density_animation(
    unique_frames: np.ndarray,
    frame_to_points: Dict[float, np.ndarray],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    output_path: Path,
):
    frame_sequence = unique_frames[::FRAME_STRIDE]
    x_edges = np.linspace(xlim[0], xlim[1], DENSITY_BINS + 1)
    y_edges = np.linspace(ylim[0], ylim[1], DENSITY_BINS + 1)
    vmax = _max_density(frame_sequence, frame_to_points, x_edges, y_edges)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    # (xbins-1, ybins-1), transposed to match imshow orientation.
    image = ax.imshow(
        np.zeros((DENSITY_BINS, DENSITY_BINS)),
        origin="lower",
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        cmap=DENSITY_CMAP,
        vmin=0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Larva count per bin")

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    density_label = "KDE-smoothed" if DENSITY_METHOD.lower() == "kde" else "Histogram"
    ax.set_title(f"Larval occupancy density ({density_label}, sliding window)")
    ax.set_xlabel("x position")
    ax.set_ylabel("y position")

    if INFER_PETRI_DISH:
        cx, cy, r = infer_petri_circle(*xlim, *ylim)
        ax.add_patch(Circle((cx, cy), r, fill=False, linestyle="--", linewidth=1.2, alpha=0.65))

    annotation = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", color="white")
    recent_frames: deque = deque()

    fig.canvas.draw()
    width_px, height_px = fig.canvas.get_width_height()
    writer = make_video_writer(output_path, (width_px, height_px), FPS)

    try:
        for i, f in enumerate(frame_sequence):
            if i % 200 == 0:
                print(f"Density animation progress: {i}/{len(frame_sequence)}", flush=True)
            recent_frames.append(f)
            window = get_window_points(f, recent_frames, frame_to_points, WINDOW_SIZE_FRAMES)
            if len(window) == 0:
                dens = np.zeros((DENSITY_BINS, DENSITY_BINS))
            else:
                x = window[:, 1].astype(float)
                y = window[:, 2].astype(float)
                dens = compute_density_grid(x, y, x_edges, y_edges)
            image.set_data(dens)
            annotation.set_text(f"Frame: {int(f)}")
            frame_bgr = fig_to_bgr_frame(fig)
            writer.write(frame_bgr)
    finally:
        writer.release()
    plt.close(fig)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {DATA_PATH.resolve()}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    cols = detect_columns(DATA_PATH)
    df = load_and_preprocess(DATA_PATH, cols)
    xlim = (float(df[cols.x].min()), float(df[cols.x].max()))
    ylim = (float(df[cols.y].min()), float(df[cols.y].max()))
    xlim = stabilized_limits(*xlim)
    ylim = stabilized_limits(*ylim)
    unique_frames, frame_to_points = prepare_frame_cache(df, cols.frame, cols.x, cols.y)

    if MAKE_TRACK_ANIMATION:
        track_out = OUTPUT_DIR / "larvae_tracks_window.mp4"
        print(f"Creating track animation -> {track_out}")
        create_track_animation(unique_frames, frame_to_points, xlim, ylim, track_out)
        print("Track animation complete.")

    if MAKE_DENSITY_ANIMATION:
        density_out = OUTPUT_DIR / "larvae_density.mp4"
        print(f"Creating density animation -> {density_out}")
        create_density_animation(unique_frames, frame_to_points, xlim, ylim, density_out)
        print("Density animation complete.")

    print("Done.")


if __name__ == "__main__":
    main()
