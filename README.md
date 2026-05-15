# Larval Trajectory Animation

**Use the Python script** (`create_larvae_animations.py`) for trajectory and density videos. It is the maintained implementation: flexible input paths, HC density mode, track history, summary frames, and Petri-dish styling.

The **R script** (`create_larvae_animations.R`) is **legacy** and **not kept in sync** with the Python pipeline. It often fails in practice (e.g. missing or misconfigured `av` / `gifski`, MP4 API differences, frame ordering when encoding). **Prefer Python** unless you are patching the R workflow yourself.

## FFmpeg: Do You Need It?

Short answer: **not for Python**.

- **Python**: OpenCV (`opencv-python`) writes MP4; no standalone `ffmpeg` CLI required.
- **R** (if you still try it): prefers `av` for MP4; may fall back to `gifski` for GIF.

If MP4 encoding fails:

- Python: install/update `opencv-python`
- R: install `av` or `gifski` (success not guaranteed; see note above)

## 1) Python version (recommended)

### Requirements

- Python 3.9+ (3.10+ recommended)
- packages: `pandas`, `numpy`, `matplotlib`, `opencv-python`

### Install

```bash
python -m venv .venv
```

Activate:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib opencv-python
```

### Run with any input CSV

Pass your file as the **first argument**, or use **`--data` / `-i`**:

```bash
python create_larvae_animations.py path/to/your_tracking.csv
python create_larvae_animations.py -i path/to/your_tracking.csv
```

If you run with **no** path, the script uses `DATA_PATH` at the top of `create_larvae_animations.py` (default sample name `embedding_dan.csv`).

```bash
python create_larvae_animations.py --help
```

### Faster preview (`--fast`)

Rendering is slow (Matplotlib redraws every frame). Use **`--fast`** for a coarser, quicker pass. Optional wrappers: **`run_larvae_animations_fast.bat`** (Windows) or **`run_larvae_animations_fast.sh`** (macOS/Linux) — they run `python …/create_larvae_animations.py --fast` and forward extra arguments.

```bash
python create_larvae_animations.py ./data/run.csv --fast
./run_larvae_animations_fast.sh ./data/run.csv --compare-subplots --group-condition-column condition

# Keep the fast stride/FPS but use KDE again (slower than histogram)
python create_larvae_animations.py ./data/run.csv --fast --density-method kde
```

`--fast` applies **`FAST_PRESET_OVERRIDES`** in the script (stride, FPS, bin count, histogram mode, smaller figure). Later flags **override** the preset (e.g. `--frame-stride 2`, `--fps 12`, `--density-bins 72`).

Module-level variables in the script still apply; CLI flags override them.

**Head-cast (HC) column:** many lab exports use a column named **`HCs`**. The Python script tries that name first when auto-detecting HC frames for `--density-mode hc`. If your file uses another column, set `HC_FILTER_COLUMN` / `--hc-filter-column` (and optionally `HC_FILTER_VALUE` / `--hc-filter-value`).

**New Python parameters (see script header and `--help`)**

| Area | Parameters | Notes |
|------|------------|--------|
| Track window | `TRACK_HISTORY_SECONDS` / `--track-history-seconds` | `None` = cumulative (default). If set, only points with `_time_sec` in the last *N* seconds are drawn. Time comes from `TIMESTAMP_COLUMN_OVERRIDE` or `frame / RECORDING_FPS` (or frame column as seconds if `FRAME_VALUES_ARE_SECONDS`). |
| Track color | `TRACK_COLOR_MODE`, `TRACK_COLOR` | `individual` = stable color per larva (`TRACK_CMAP`); `same` = one color for all tracks. |
| Uniform track video | `MAKE_TRACK_UNIFORM_COLOR_VIDEO` / `--uniform-color-track` / `--no-uniform-color-track` | Extra **`larvae_tracks_uniform_color.mp4`** (and summary PNG) using `TRACK_COLOR` for every animal. Compare mode: **`larvae_tracks_compare_uniform_color.mp4`**. Skipped when the main `TRACK_COLOR_MODE` is already `"same"`. |
| Density | `DENSITY_MODE` | `animals` / `all` = all positions; `hc` = only rows identified as head-cast (HC) frames. |
| HC filters | `HC_FILTER_COLUMN`, `HC_FILTER_VALUE` | Auto-detect prefers **`HCs`**. Override if your CSV differs. |
| HC grid | `HC_DENSITY_BINS` / `--hc-density-bins` | Coarser default than `DENSITY_BINS` so sparse HC occupancy stays visible. |
| Summary | `APPEND_SUMMARY_FRAME`, `SUMMARY_PAUSE_SECONDS`, `SUMMARY_DURATION_SECONDS`, `SAVE_SUMMARY_IMAGE` | Append a full-recording summary to single-panel videos; PNG summaries default **on** (`SAVE_SUMMARY_IMAGE = True`). Disable with `--no-save-summary-image`. |
| All outputs | `MAKE_ALL_ANIMATIONS` / `--all-animations` | One run: **track** + **all-frame density** (`larvae_density.mp4`) + **HC density** (`larvae_density_hc.mp4`). Ignores `DENSITY_MODE` for which densities to build. `--no-all-animations` forces a single density pass from `DENSITY_MODE`. Combine with `--no-track` or `--no-density` if you only want the density pair or only the track. |
| Per condition | `RUN_EACH_CONDITION` / `--each-condition` | Runs the pipeline **once per distinct value** in `GROUP_CONDITION_COLUMN` (or `--group-condition-column`). Writes to `OUTPUT_DIR/<safe_condition_name>/` (same filenames in each folder). Optional `--group-condition-value` limits to one condition. |
| Compare in one video | `COMPARE_CONDITIONS_SUBPLOTS` / `--compare-subplots` | **One MP4** with a **subplot per condition** (shared axes / color scale for density). After the time course, appends a **multi-panel** full-recording summary (same layout as the video, one full trajectory or density map per condition). `APPEND_POOLED_SUMMARY_AT_END` / `--append-pooled-summary` controls that tail. PNGs: `larvae_tracks_compare_summary.png`, `larvae_density_compare_summary.png`, `larvae_density_hc_compare_summary.png`. Incompatible with `--each-condition` (compare mode wins). |
| Density normalization | `DENSITY_NORMALIZATION` / `--normalization` | `none`, `num_tracks`, or `max_visible_larvae`. Divides raw spatial density by a **fixed per-group** animal count (computed once from tracking, not per frame). HC density uses the **same** animal-based factor. |
| Selected groups | `SELECTED_GROUPS` / `--groups` | Compare mode: only listed condition IDs (e.g. two panels for a paired figure). |
| Display labels | `GROUP_DISPLAY_LABELS` / `--group-label ID:Label` | Manuscript-readable subplot titles (internal IDs unchanged in the CSV). |
| Compare filenames | `COMPARE_OUTPUT_SLUG` / `--compare-output-slug` | Inserts a slug into compare output names, e.g. `larvae_density_compare_4d_sated_vs_starved.mp4`. |
| Petri outline | `INFER_PETRI_DISH` / `--petri-dish` / `--no-petri-dish` | Circle outline in track and density plots (`draw_petri_dish`). |
| Track window | `TRACK_HISTORY_SECONDS` / `--track-window-seconds` | Trailing trajectory window in **videos**; summary PNG/MP4 tail still uses **full** tracks. |
| Axis limits | `FIXED_AXIS_XY_LIMITS` / `--fixed-axis-limits MIN MAX`, `--auto-axis-limits` | Default **(-47.5, 47.5)** for both **x** and **y** (fixed square arena). `None` or `--auto-axis-limits` uses data min/max after `stabilized_limits`. |
| Fast preview | `--fast`, `FAST_PRESET_OVERRIDES`, `run_larvae_animations_fast.*` | Lower quality / fewer frames for speed; override piecewise with `--frame-stride`, `--fps`, `--density-bins`, `--density-method`. |

**Example commands**

```bash
# Any CSV as input
python create_larvae_animations.py ./data/experiment_12.csv

# Track + both density videos in one run
python create_larvae_animations.py ./data/run.csv --all-animations

# Both densities only (no track video)
python create_larvae_animations.py ./data/run.csv --all-animations --no-track

# One batch per experimental condition (subfolders under animations/)
python create_larvae_animations.py ./data/run.csv --each-condition --group-condition-column condition

# Same, plus track + both density videos for every condition
python create_larvae_animations.py ./data/run.csv --each-condition --all-animations

# Subplots per condition in ONE video, then multi-panel full-recording summary at the end
python create_larvae_animations.py ./data/run.csv --compare-subplots --group-condition-column condition

# Compare + track + both density types (each ends with subplot summary tail)
python create_larvae_animations.py ./data/run.csv --compare-subplots --all-animations --group-condition-column condition

# HC density using default HC column detection (tries "HCs" first)
python create_larvae_animations.py ./data/run.csv --density-mode hc

# Explicit HC column if needed
python create_larvae_animations.py ./data/run.csv --density-mode hc --hc-filter-column HCs

# Cumulative tracks, single shared color
python create_larvae_animations.py ./data/run.csv --track-color-mode same --track-color "#1f77b4"

# Last 10 s trails, per-animal colors
python create_larvae_animations.py ./data/run.csv --track-history-seconds 10 --track-color-mode individual

# Animation + summary tail (PNG summaries are on by default; use --no-save-summary-image to skip)
python create_larvae_animations.py ./data/run.csv --append-summary-frame --summary-pause-seconds 0.5 --summary-duration-seconds 3

# Paired compare with normalization and readable labels
python create_larvae_animations.py ./data/run.csv --compare-subplots --groups 4d_fed 4d_starved \
  --group-label 4d_fed:4d\ Sated --group-label 4d_starved:4d\ Starved \
  --normalization max_visible_larvae --track-window-seconds 10 --all-animations \
  --compare-output-slug 4d_sated_vs_starved --petri-dish

# Manuscript batch (both 4d and 5d pairs; uses --fast by default)
# Each pair: tracks + histogram density (base slug), then KDE density only (slug *_kde)
# Density window: larval 15 s (240 frames), HC 30 s (480 frames) at 16 fps; trajectory trail: 10 s
python generate_paper_outputs.py starvation2-2026-04-23_analyzed.csv --output-dir animations/paper

# Publication-quality manuscript outputs (slower)
python generate_paper_outputs.py starvation2-2026-04-23_analyzed.csv --no-fast

# Fast preview of one pair + first 30s only
python generate_paper_outputs.py data.csv --only 4d -- --test-duration-seconds 30
```

## 2) R version (legacy; may not run)

`create_larvae_animations.R` is **not equivalent** to the current Python tool (no HC density mode, no track history window, no summary append, etc.). It is left in the repo for reference but **is not supported** here.

If you rely on R, expect to debug local package installs (`av`, `gifski`, `MASS`) and encoding yourself, or **port your workflow to Python** using the flags above.

### Requirements (if you experiment with R)

- R 4.1+ recommended
- `ggplot2`, `dplyr`, `readr`, `tidyr`
- `av` and/or `gifski` for video
- `MASS` if `DENSITY_METHOD = "kde"`

### Run

```bash
Rscript create_larvae_animations.R
```

Edit `DATA_PATH` at the top of the R script; there is no CLI for arbitrary files.

## Input data and column detection

Auto-detection priority for **x** / **y**:

- `x`: `centroid_x` → `centoid_x` → `spinepoint_x6_conv` → `spinepoint_x_6_conv`
- `y`: `centroid_y` → `centoid_y` → `spinepoint_y6_conv` → `spinepoint_y_6_conv`

**Frame / time:** `frame` → `t` → `time` → `timestamp` → `frame_idx`

**Larva ID:** `id` → `track_id` → `larva_id` → `track` → `animal_id`

**HC / head-cast (Python, `--density-mode hc`):** tries **`HCs`** first, then other common names (`is_hc`, `behavior`, …). Set overrides if your CSV differs.

If required columns are missing, the Python script fails with a clear error listing available columns.

## Common configuration (Python)

Edit variables at the top of `create_larvae_animations.py`:

- `DATA_PATH` default CSV when no CLI path is given
- `OUTPUT_DIR`
- `X_COLUMN_OVERRIDE`, `Y_COLUMN_OVERRIDE`, `FRAME_COLUMN_OVERRIDE`, `ID_COLUMN_OVERRIDE`
- `GROUP_CONDITION_COLUMN`, `GROUP_CONDITION_VALUE`
- `RECORDING_FPS`, `FPS`, `WINDOW_SIZE_FRAMES`, `FRAME_STRIDE`, `TEST_DURATION_SECONDS` (`None` for full length)
- `MAKE_TRACK_ANIMATION`, `MAKE_DENSITY_ANIMATION`, `MAKE_ALL_ANIMATIONS` (track + both density videos in one run)
- `MAKE_TRACK_UNIFORM_COLOR_VIDEO` (also writes `larvae_tracks_uniform_color.mp4` with one color for all larvae; skipped if `TRACK_COLOR_MODE` is already `"same"`)
- `RUN_EACH_CONDITION` with `GROUP_CONDITION_COLUMN` (and optional `GROUP_CONDITION_VALUE` to restrict)
- `COMPARE_CONDITIONS_SUBPLOTS`, `APPEND_POOLED_SUMMARY_AT_END` (subplot comparison + pooled ending)
- `PETRI_DISH_RADIUS` (dish outline), `FIXED_AXIS_XY_LIMITS` (default `(-47.5, 47.5)` for x and y, or `None` for data-driven limits)
- `DENSITY_METHOD` (`"hist"` or `"kde"`), `KDE_SIGMA_BINS`

## Example configuration snippet (Python)

```python
DATA_PATH = Path("my_default.csv")  # or pass any file on the command line
X_COLUMN_OVERRIDE = "centroid_x"
Y_COLUMN_OVERRIDE = "centroid_y"
GROUP_CONDITION_COLUMN = "condition"
GROUP_CONDITION_VALUE = "EM_Rewarded"
RECORDING_FPS = 16
FPS = 16
TEST_DURATION_SECONDS = None  # full recording; use e.g. 30 for a short preview
DENSITY_MODE = "hc"  # uses HCs column when present
```

## Troubleshooting

- **Everything is centered at one point** — Check x/y columns; they may be constant or wrong.
- **Missing-column error** — Set explicit overrides for frame/id/x/y.
- **Rendering is slow** — Increase `FRAME_STRIDE`, reduce `DENSITY_BINS` / `FPS`, use `TEST_DURATION_SECONDS` while iterating.
- **R script errors** — Use the Python pipeline instead; the R path is not maintained.
