#!/usr/bin/env python3
"""Generate manuscript-specific larval animation outputs (paired 4d / 5d comparisons).

Calls ``create_larvae_animations.main()`` with fixed paper defaults; the core CLI stays generic.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

from create_larvae_animations import main as run_animations

# Internal condition IDs in the CSV -> manuscript panel titles
PAPER_GROUP_LABELS: Tuple[Tuple[str, str], ...] = (
    ("4d_fed", "4d Sated"),
    ("4d_starved", "4d Starved"),
    ("5d_fed", "5d Sated"),
    ("5d_starved", "5d Starved"),
)

# Paired comparisons: (slug for filenames, group A, group B)
PAPER_COMPARISONS: Tuple[Tuple[str, str, str], ...] = (
    ("4d_sated_vs_starved", "4d_fed", "4d_starved"),
    ("5d_sated_vs_starved", "5d_fed", "5d_starved"),
)

# Both spatial density estimators for manuscript figures
PAPER_DENSITY_METHODS: Tuple[str, ...] = ("hist", "kde")

# Density sliding window: 15 s at 16 fps (matches RECORDING_FPS in create_larvae_animations.py)
PAPER_RECORDING_FPS = 16.0
PAPER_DENSITY_WINDOW_SECONDS = 15.0
PAPER_DENSITY_WINDOW_FRAMES = int(PAPER_RECORDING_FPS * PAPER_DENSITY_WINDOW_SECONDS)
PAPER_HC_DENSITY_WINDOW_SECONDS = 30.0
PAPER_HC_DENSITY_WINDOW_FRAMES = int(PAPER_RECORDING_FPS * PAPER_HC_DENSITY_WINDOW_SECONDS)


def label_args_for_groups(group_ids: Sequence[str]) -> List[str]:
    lookup = dict(PAPER_GROUP_LABELS)
    out: List[str] = []
    for gid in group_ids:
        if gid not in lookup:
            raise KeyError(f"Unknown group id {gid!r}; expected one of {sorted(lookup)}")
        out.append(f"--group-label")
        out.append(f"{gid}:{lookup[gid]}")
    return out


def build_run_argv(
    csv_path: Path,
    output_dir: Path,
    slug: str,
    group_a: str,
    group_b: str,
    *,
    normalization: str,
    track_window_seconds: float,
    summary_duration_seconds: float,
    use_fast: bool = True,
    density_method: str | None = None,
    skip_track: bool = False,
    extra: Sequence[str] = (),
) -> List[str]:
    """CLI argv for one paired compare run (track + larval density + HC density)."""
    base = [
        str(csv_path),
        "--output-dir",
        str(output_dir),
        "--compare-subplots",
        "--group-condition-column",
        "condition",
        "--groups",
        group_a,
        group_b,
        "--compare-output-slug",
        slug,
        "--all-animations",
        "--normalization",
        normalization,
        "--track-window-seconds",
        str(track_window_seconds),
        "--window-size-frames",
        str(PAPER_DENSITY_WINDOW_FRAMES),
        "--hc-window-size-frames",
        str(PAPER_HC_DENSITY_WINDOW_FRAMES),
        "--append-pooled-summary",
        "--summary-duration-seconds",
        str(summary_duration_seconds),
        "--summary-pause-seconds",
        "0.25",
        "--petri-dish",
        "--no-uniform-color-track",
    ]
    if skip_track:
        base.append("--no-track")
    if use_fast:
        base.append("--fast")
    if density_method is not None:
        base.append("--density-method")
        base.append(density_method)
    base.extend(label_args_for_groups([group_a, group_b]))
    base.extend(extra)
    return base


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run paired manuscript comparisons (4d and 5d fed vs starved)."
    )
    p.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=Path("starvation2-2026-04-23_analyzed.csv"),
        help="Tracking CSV (default: starvation2-2026-04-23_analyzed.csv)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("animations/paper"),
        help="Root folder for paired comparison outputs",
    )
    p.add_argument(
        "--normalization",
        choices=("none", "num_tracks", "max_visible_larvae"),
        default="max_visible_larvae",
        help="Per-group density divisor (animal-based; same for HC)",
    )
    p.add_argument(
        "--track-window-seconds",
        type=float,
        default=10.0,
        help="Trailing trajectory window in videos (full tracks in summary still)",
    )
    p.add_argument(
        "--summary-duration-seconds",
        type=float,
        default=2.5,
        help="Hold full-recording summary frame at end of each video",
    )
    p.add_argument(
        "--only",
        choices=("4d", "5d", "both"),
        default="both",
        help="Run one or both age comparisons",
    )
    speed = p.add_mutually_exclusive_group()
    speed.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        help="Use create_larvae_animations --fast preset (default).",
    )
    speed.add_argument(
        "--no-fast",
        dest="fast",
        action="store_false",
        help="Full-quality rendering (no --fast preset).",
    )
    p.set_defaults(fast=True)
    p.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra flags forwarded to create_larvae_animations.py (prefix with --)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    comparisons = list(PAPER_COMPARISONS)
    if args.only == "4d":
        comparisons = [c for c in comparisons if c[0].startswith("4d")]
    elif args.only == "5d":
        comparisons = [c for c in comparisons if c[0].startswith("5d")]

    extra = list(args.extra_args)
    if extra and extra[0] == "--":
        extra = extra[1:]

    use_fast = args.fast
    if use_fast:
        print("Fast preset enabled (use --no-fast for full quality).\n", flush=True)

    for slug, ga, gb in comparisons:
        out = args.output_dir / slug
        out.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Paper comparison: {ga} vs {gb} -> {out} ===\n", flush=True)

        for method in PAPER_DENSITY_METHODS:
            # First pass (hist): tracks + histogram density; KDE pass: density only, distinct slug
            is_kde = method == "kde"
            run_slug = f"{slug}_kde" if is_kde else slug
            skip_track = is_kde
            print(
                f"--- Density method: {method} (slug={run_slug!r}"
                + (", tracks skipped" if skip_track else ", incl. tracks")
                + ") ---\n",
                flush=True,
            )
            run_argv = build_run_argv(
                args.input_csv,
                out,
                run_slug,
                ga,
                gb,
                normalization=args.normalization,
                track_window_seconds=args.track_window_seconds,
                summary_duration_seconds=args.summary_duration_seconds,
                use_fast=use_fast,
                density_method=method,
                skip_track=skip_track,
                extra=extra,
            )
            run_animations(run_argv)

    print("\nPaper outputs complete.", flush=True)


if __name__ == "__main__":
    main()
