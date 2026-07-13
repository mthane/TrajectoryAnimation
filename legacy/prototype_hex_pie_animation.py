#!/usr/bin/env python3
"""Prototype: hex-grid categorical glyph animation (pie charts).

Creates one animation per `group_condition` showing local category composition
over time on a hexagonal grid.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon, Wedge
import numpy as np
import pandas as pd


# =========================
# Config (edit as needed)
# =========================
DATA_DIR = Path(".")
DATA_FILE_PREFIX = "Learning_annotated"
DATA_EXTENSIONS = (".csv", ".parquet", ".feather", ".rds")

# Required schema (set for your dataset)
# For Learning_annotated_01_12.csv these are centroid_x / centroid_y.
# If your dataset has spinepoint centroids, set these to spinepoint_x6_conv / spinepoint_y6_conv.
X_COL = "centroid_x"
Y_COL = "centroid_y"
CATEGORY_COL = "cluster_label"
GROUP_COL = "group_condition"
TIME_COL_OVERRIDE: Optional[str] = None
TIME_COL_CANDIDATES = ("frame", "time", "t", "frame_id", "timestamp")
ID_COL_CANDIDATES = ("id", "track_id", "larva_id", "track", "animal_id")

OUTPUT_DIR = Path("animations/hex_pie_prototype")

# Hex/grid and animation controls
HEX_SIZE = 2.0
WINDOW_SIZE = 100
FRAME_STEP = 5
FPS = 16
FIGSIZE = (9, 9)
DPI = 130
MIN_COUNT = 8
MAX_ANIMATION_FRAMES: Optional[int] = 300  # set None for full run

# Pie glyph sizing
PIE_RADIUS_MIN = 0.15
PIE_RADIUS_MAX = 0.9
PIE_RADIUS_COUNT_QUANTILE = 0.95

# Rendering mode: "pie" (prototype target) or "dominant" (faster debug)
RENDER_MODE = "pie"


def sanitize_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return safe.strip("_") or "group"


def pick_first_existing(candidates: Iterable[str], columns: List[str]) -> Optional[str]:
    present = set(columns)
    for c in candidates:
        if c in present:
            return c
    return None


def find_input_file() -> Path:
    files = sorted(
        p
        for p in DATA_DIR.iterdir()
        if p.is_file() and p.name.startswith(DATA_FILE_PREFIX) and p.suffix.lower() in DATA_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No file found starting with '{DATA_FILE_PREFIX}' in {DATA_DIR.resolve()} "
            f"with extensions {DATA_EXTENSIONS}."
        )
    print(f"Using input file: {files[0].name}")
    return files[0]


def load_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path, low_memory=False)
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".feather":
        return pd.read_feather(path)
    if ext == ".rds":
        try:
            import pyreadr  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Reading .rds requires `pyreadr`. Install with: pip install pyreadr"
            ) from exc
        out = pyreadr.read_r(str(path))
        if not out:
            raise ValueError(f"No objects found in RDS file: {path}")
        first_key = next(iter(out.keys()))
        return out[first_key]
    raise ValueError(f"Unsupported file extension: {ext}")


def infer_time_column(columns: List[str]) -> str:
    if TIME_COL_OVERRIDE:
        if TIME_COL_OVERRIDE not in columns:
            raise ValueError(
                f"Configured TIME_COL_OVERRIDE='{TIME_COL_OVERRIDE}' not found in columns."
            )
        return TIME_COL_OVERRIDE
    found = pick_first_existing(TIME_COL_CANDIDATES, columns)
    if found is None:
        raise ValueError(
            f"No time/frame column found. Tried in order: {TIME_COL_CANDIDATES}. "
            "Set TIME_COL_OVERRIDE explicitly."
        )
    return found


def validate_columns(columns: List[str], time_col: str) -> None:
    required = [X_COL, Y_COL, CATEGORY_COL, GROUP_COL, time_col]
    missing = [c for c in required if c not in columns]
    if missing:
        pretty = ", ".join(columns)
        raise ValueError(
            "Required columns missing:\n"
            + "\n".join(f"- {c}" for c in missing)
            + f"\n\nAvailable columns:\n{pretty}"
        )


def hex_index_for_points(x: np.ndarray, y: np.ndarray, x_min: float, y_min: float) -> Tuple[np.ndarray, np.ndarray]:
    dy = (math.sqrt(3) / 2.0) * HEX_SIZE
    row = np.round((y - y_min) / dy).astype(np.int32)
    x_offset = (row % 2) * (HEX_SIZE / 2.0)
    col = np.round((x - x_min - x_offset) / HEX_SIZE).astype(np.int32)
    return col, row


def hex_center(col: int, row: int, x_min: float, y_min: float) -> Tuple[float, float]:
    dy = (math.sqrt(3) / 2.0) * HEX_SIZE
    x_offset = (row % 2) * (HEX_SIZE / 2.0)
    cx = x_min + (col * HEX_SIZE) + x_offset
    cy = y_min + (row * dy)
    return cx, cy


def hex_polygon(cx: float, cy: float, r: float) -> np.ndarray:
    angles = np.deg2rad(np.arange(0, 360, 60)) + np.deg2rad(30)
    return np.column_stack([cx + r * np.cos(angles), cy + r * np.sin(angles)])


def radius_from_count(count: int, count_scale: float) -> float:
    if count_scale <= 0:
        return PIE_RADIUS_MIN
    frac = min(max(count / count_scale, 0.0), 1.0)
    return PIE_RADIUS_MIN + (PIE_RADIUS_MAX - PIE_RADIUS_MIN) * math.sqrt(frac)


def render_group_frame(
    ax: plt.Axes,
    frame_data: pd.DataFrame,
    color_map: Dict[str, Tuple[float, float, float, float]],
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    group_value: str,
    time_value: float,
    count_scale: float,
):
    ax.clear()
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(X_COL)
    ax.set_ylabel(Y_COL)
    ax.set_title(f"group_condition={group_value} | time={int(time_value)} | window={WINDOW_SIZE}")

    if frame_data.empty:
        return

    grouped = (
        frame_data.groupby(["hex_col", "hex_row", CATEGORY_COL], observed=True)
        .size()
        .rename("count")
        .reset_index()
    )
    totals = (
        grouped.groupby(["hex_col", "hex_row"], observed=True)["count"]
        .sum()
        .rename("total_count")
        .reset_index()
    )
    merged = grouped.merge(totals, on=["hex_col", "hex_row"], how="left")
    merged = merged[merged["total_count"] >= MIN_COUNT].copy()
    if merged.empty:
        return
    merged["prop"] = merged["count"] / merged["total_count"]

    for (col, row), cell in merged.groupby(["hex_col", "hex_row"], observed=True):
        cx, cy = hex_center(int(col), int(row), xlim[0], ylim[0])
        total_count = int(cell["total_count"].iloc[0])
        radius = radius_from_count(total_count, count_scale)

        # Debug mode for faster rendering: fill hex with dominant cluster color.
        if RENDER_MODE.lower() == "dominant":
            top = cell.sort_values("prop", ascending=False).iloc[0]
            poly = Polygon(
                hex_polygon(cx, cy, HEX_SIZE / 2.0),
                closed=True,
                facecolor=color_map[str(top[CATEGORY_COL])],
                edgecolor="white",
                linewidth=0.25,
                alpha=0.9,
            )
            ax.add_patch(poly)
            continue

        # Pie glyph prototype. This is where future "Droplet glyphs" can plug in.
        start = 0.0
        for _, row_item in cell.sort_values(CATEGORY_COL).iterrows():
            frac = float(row_item["prop"])
            theta1 = 360.0 * start
            theta2 = 360.0 * (start + frac)
            wedge = Wedge(
                center=(cx, cy),
                r=radius,
                theta1=theta1,
                theta2=theta2,
                facecolor=color_map[str(row_item[CATEGORY_COL])],
                edgecolor="white",
                linewidth=0.2,
            )
            ax.add_patch(wedge)
            start += frac
        ax.add_patch(Circle((cx, cy), radius=radius, fill=False, edgecolor="black", linewidth=0.2, alpha=0.6))


def make_writer(path: Path, width: int, height: int) -> cv2.VideoWriter:
    for codec in ("mp4v", "avc1", "H264"):
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*codec), FPS, (width, height))
        if writer.isOpened():
            print(f"Using codec {codec} for {path.name}")
            return writer
        writer.release()
    raise RuntimeError(f"Could not initialize MP4 writer for {path}")


def fig_to_bgr(fig: plt.Figure) -> np.ndarray:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    rgb = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)[..., :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    input_file = find_input_file()
    df = load_table(input_file)
    print(f"Loaded rows: {len(df):,}")
    print("Available columns:")
    print(", ".join(df.columns.astype(str)))

    time_col = infer_time_column(df.columns.tolist())
    validate_columns(df.columns.tolist(), time_col)

    id_col = pick_first_existing(ID_COL_CANDIDATES, df.columns.tolist())
    if id_col:
        print(f"Inferred ID column: {id_col}")
    print(f"Using time column: {time_col}")

    df[X_COL] = pd.to_numeric(df[X_COL], errors="coerce")
    df[Y_COL] = pd.to_numeric(df[Y_COL], errors="coerce")
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df = df.dropna(subset=[X_COL, Y_COL, CATEGORY_COL, GROUP_COL, time_col]).copy()

    df[CATEGORY_COL] = df[CATEGORY_COL].astype(str)
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    df = df.sort_values([GROUP_COL, time_col]).reset_index(drop=True)
    print(f"Rows after preprocessing: {len(df):,}")

    xlim = (float(df[X_COL].min()), float(df[X_COL].max()))
    ylim = (float(df[Y_COL].min()), float(df[Y_COL].max()))
    print(f"Global x limits: {xlim}")
    print(f"Global y limits: {ylim}")

    hex_col, hex_row = hex_index_for_points(df[X_COL].to_numpy(), df[Y_COL].to_numpy(), xlim[0], ylim[0])
    df["hex_col"] = hex_col
    df["hex_row"] = hex_row

    categories = sorted(df[CATEGORY_COL].unique().tolist())
    cmap = plt.get_cmap("tab20", max(20, len(categories)))
    color_map: Dict[str, Tuple[float, float, float, float]] = {}
    for i, cat in enumerate(categories):
        color_map[str(cat)] = tuple(cmap(i % cmap.N))
    print(f"Detected categories: {len(categories)}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = sorted(df[GROUP_COL].unique().tolist())
    print(f"Found group_condition values: {len(groups)}")

    for group_idx, group_value in enumerate(groups, start=1):
        group_df = df[df[GROUP_COL] == group_value].copy()
        times = np.sort(group_df[time_col].unique())
        if len(times) == 0:
            continue
        times = times[::FRAME_STEP]
        if MAX_ANIMATION_FRAMES is not None:
            times = times[:MAX_ANIMATION_FRAMES]

        safe_group = sanitize_filename(group_value)
        out_mp4 = OUTPUT_DIR / f"hex_pie_{safe_group}.mp4"
        out_png = OUTPUT_DIR / f"hex_pie_{safe_group}_preview.png"

        print(
            f"[{group_idx}/{len(groups)}] Rendering group '{group_value}' "
            f"with {len(times)} animation frames -> {out_mp4.name}"
        )

        fig, ax = plt.subplots(figsize=FIGSIZE)
        fig.canvas.draw()
        width_px, height_px = fig.canvas.get_width_height()
        writer = make_writer(out_mp4, width_px, height_px)

        try:
            rep_idx = len(times) // 2
            for i, t_val in enumerate(times):
                if i % 25 == 0:
                    print(f"  frame {i+1}/{len(times)} (time={int(t_val)})")
                t_min = t_val - WINDOW_SIZE
                window_df = group_df[(group_df[time_col] >= t_min) & (group_df[time_col] <= t_val)]
                if window_df.empty:
                    frame_img = fig_to_bgr(fig)
                    writer.write(frame_img)
                    continue

                count_scale = float(window_df.groupby(["hex_col", "hex_row"], observed=True).size().quantile(PIE_RADIUS_COUNT_QUANTILE))
                render_group_frame(
                    ax=ax,
                    frame_data=window_df,
                    color_map=color_map,
                    xlim=xlim,
                    ylim=ylim,
                    group_value=group_value,
                    time_value=float(t_val),
                    count_scale=count_scale,
                )
                frame_img = fig_to_bgr(fig)
                writer.write(frame_img)

                if i == rep_idx:
                    fig.savefig(out_png, dpi=DPI, bbox_inches="tight")
        finally:
            writer.release()
            plt.close(fig)

    print("All group animations complete.")


if __name__ == "__main__":
    main()

