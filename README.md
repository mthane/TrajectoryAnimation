# Larval Trajectory Animation

Python toolkit for larval centroid trajectory and kernel-density animations from tracking CSVs. Built for starvation-experiment manuscript figures (fed vs starved, 4d and 5d).

## Repository layout

```
├── create_larvae_animations.py      # Core rendering engine
├── finalize_starvation_kde_movies.py # Publication batch (recommended for manuscript)
├── report_out_of_petri_tracks.py    # QC: out-of-arena track report
├── requirements.txt
├── data/                            # Input CSVs (not in git — see data/README.md)
├── outputs/starvation_kde_final/    # Final publication movies + summaries
└── legacy/                          # Archived scripts and old iterations
```

## Quick start

### Requirements

- Python 3.9+ (3.10+ recommended)
- Packages: `pandas`, `numpy`, `matplotlib`, `opencv-python`

```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

No standalone **ffmpeg** is required; MP4 encoding uses OpenCV.

### Input data

Place the tracking CSV in `data/` (see `data/README.md`). For the starvation manuscript:

```text
data/starvation2-2026-04-23_analyzed.csv
```

### Reproduce publication outputs

```bash
python finalize_starvation_kde_movies.py data/starvation2-2026-04-23_analyzed.csv
```

Writes to `outputs/starvation_kde_final/`:

| Subfolder | Contents |
|-----------|----------|
| `per_condition/<condition>/` | Per-condition track + larval density + HC density (1× and 2×) |
| `4d_fed_vs_starved/`, `5d_fed_vs_starved/` | Side-by-side compare movies + summary PNGs |
| `run_manifest.json` | Machine-readable render parameters |
| `methods_and_legend_draft.md` | Methods / legend text for the paper |
| `RENDERING_PARAMETERS.md` | Parameter table |

Each movie ends with a **10 s** full-recording summary panel. Movies use **viridis** with a **linear** color scale; summary panels use finer bins and tighter color limits than the time-resolved frames.

Quick test (first 30 s only):

```bash
python finalize_starvation_kde_movies.py data/starvation2-2026-04-23_analyzed.csv --test-duration-seconds 30 --output-dir outputs/starvation_kde_final_test
```

### Generic animations (any CSV)

```bash
python create_larvae_animations.py path/to/tracking.csv
python create_larvae_animations.py path/to/tracking.csv --help
```

Faster preview:

```bash
python create_larvae_animations.py path/to/tracking.csv --fast
# or: run_larvae_animations_fast.bat / run_larvae_animations_fast.sh
```

Common options:

| Flag | Purpose |
|------|---------|
| `--compare-subplots` | One video with a subplot per condition |
| `--all-animations` | Track + larval density + HC density |
| `--each-condition` | Separate output folder per condition |
| `--groups A B` | Limit compare panels to listed conditions |
| `--normalization max_visible_larvae` | Normalize density by peak co-visible larvae |
| `--track-window-seconds 10` | Trailing trajectory window in videos |
| `--test-duration-seconds 30` | Render only the first N seconds |

## Column auto-detection

| Role | Tried names (in order) |
|------|------------------------|
| x | `centroid_x`, `centoid_x`, `spinepoint_x6_conv`, … |
| y | `centroid_y`, `centoid_y`, `spinepoint_y6_conv`, … |
| frame | `frame`, `t`, `time`, `timestamp`, … |
| ID | `id`, `track_id`, `larva_id`, … |
| HC | `HCs`, `is_hc`, `behavior`, … |

Override with `--x-column`, `--y-column`, `--frame-column`, `--id-column`, `--hc-filter-column`.

## QC: artifact tracks

Tracks outside the Petri dish (IDs 1637, 2937, 2354, 2366 in 5d Fed) are excluded in the publication pipeline. To inspect out-of-arena points:

```bash
python report_out_of_petri_tracks.py data/starvation2-2026-04-23_analyzed.csv
```

## Submission packaging

- **Code:** this repository (scripts + `requirements.txt` + parameter docs).
- **Media:** MP4/PNG files under `outputs/starvation_kde_final/` (gitignored by size; ship as supplemental zip or external storage).
- **Data:** `data/starvation2-2026-04-23_analyzed.csv` (host separately or document download).
- **Legacy:** older iterations live in `legacy/` and are not needed to reproduce final figures.

## Troubleshooting

- **Center dot / no movement** — Wrong x/y columns; check CSV headers and overrides.
- **Slow rendering** — Use `--fast`, `--test-duration-seconds`, higher `--frame-stride`, or fewer `--density-bins`.
- **Missing columns** — Run with explicit `--x-column` / `--y-column` flags; error lists available columns.
