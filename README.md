# Larval Trajectory Animation (Python + R)

This repository includes two equivalent scripts to generate larval trajectory animations:

- `create_larvae_animations.py` (Python)
- `create_larvae_animations.R` (R)

Both scripts:

- read tracking data from a CSV file
- auto-detect frame/id/x/y columns (with optional overrides)
- support optional group filtering
- create:
  - `animations/larvae_tracks_window.mp4` (cumulative tracks over time)
  - `animations/larvae_density.mp4` (sliding-window density map)
- keep fixed axis limits for comparability
- do **not** modify original data

## FFmpeg: Do You Need It?

Short answer: **not usually**.

- **Python script**: uses OpenCV video writer (`opencv-python`), so no standalone `ffmpeg` CLI is required.
- **R script**:
  - prefers `av` package for MP4 output (no manual ffmpeg install needed in most setups)
  - if `av` is unavailable, falls back to GIF output via `gifski`

If MP4 encoding fails in your environment, install/update:

- Python: `opencv-python`
- R: `av` (preferred) or `gifski` fallback

## 1) Python Version

### Requirements

- Python 3.9+ (3.10+ recommended)
- packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `opencv-python`

### Install

```bash
python -m venv .venv
```

Activate:

- Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

Install deps:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib opencv-python
```

### Run

```bash
python create_larvae_animations.py
```

## 2) R Version

### Requirements

- R 4.1+ recommended
- required packages:
  - `ggplot2`
  - `dplyr`
  - `readr`
  - `tidyr`
- video/animation packages:
  - `av` (preferred MP4 output)
  - `gifski` (fallback if MP4 encoder unavailable)
- optional:
  - `MASS` (required when `DENSITY_METHOD = "kde"`)

### Install

```r
install.packages(c("ggplot2", "dplyr", "readr", "tidyr", "av", "gifski", "MASS"))
```

### Run

```bash
Rscript create_larvae_animations.R
```

## Input Data and Column Detection

Default input file for both scripts:

- `embedding_dan.csv`

Auto-detection priority for x/y:

- `x`: `centroid_x` -> `centoid_x` -> `spinepoint_x6_conv` -> `spinepoint_x_6_conv`
- `y`: `centroid_y` -> `centoid_y` -> `spinepoint_y6_conv` -> `spinepoint_y_6_conv`

Auto-detection for frame/time:

- `frame` -> `t` -> `time` -> `timestamp` -> `frame_idx`

Auto-detection for larva ID:

- `id` -> `track_id` -> `larva_id` -> `track` -> `animal_id`

If required columns are missing, scripts fail with a clear error listing available columns.

## Common Configuration (Both Scripts)

Edit config variables at the top of each script:

- `DATA_PATH` / `DATA_PATH <- ...` input CSV path
- `OUTPUT_DIR` output folder
- `X_COLUMN_OVERRIDE`, `Y_COLUMN_OVERRIDE` (or `NULL`/`None`)
- `FRAME_COLUMN_OVERRIDE`, `ID_COLUMN_OVERRIDE` (or `NULL`/`None`)
- `GROUP_CONDITION_COLUMN` column used for optional filtering
- `GROUP_CONDITION_VALUE` set value to filter rows, or disable with `NULL`/`None`
- `RECORDING_FPS` recording frame rate (used for test-window slicing)
- `FPS` output animation framerate
- `WINDOW_SIZE_FRAMES` density sliding window
- `FRAME_STRIDE` frame subsampling
- `TEST_DURATION_SECONDS` set short test run; disable with `NULL`/`None`
- `MAKE_TRACK_ANIMATION` / `MAKE_DENSITY_ANIMATION`

Density controls:

- Python:
  - `DENSITY_METHOD` = `"hist"` or `"kde"`
  - `KDE_SIGMA_BINS` smoothing bandwidth for `"kde"`
- R:
  - `DENSITY_METHOD` = `"hist"` or `"kde"`
  - `KDE_BANDWIDTH` smoothing bandwidth for `"kde"` (uses `MASS::kde2d`)

## Example Configuration Snippets

### Python

```python
X_COLUMN_OVERRIDE = "centroid_x"
Y_COLUMN_OVERRIDE = "centroid_y"
GROUP_CONDITION_COLUMN = "condition"
GROUP_CONDITION_VALUE = "EM_Rewarded"
RECORDING_FPS = 16
FPS = 16
TEST_DURATION_SECONDS = 30
```

### R

```r
X_COLUMN_OVERRIDE <- "centroid_x"
Y_COLUMN_OVERRIDE <- "centroid_y"
GROUP_CONDITION_COLUMN <- "condition"
GROUP_CONDITION_VALUE <- "EM_Rewarded"
RECORDING_FPS <- 16
FPS <- 16
TEST_DURATION_SECONDS <- 30
```

## Troubleshooting

- **Everything is centered at one point**
  - Check whether chosen x/y columns are constant or all zeros.
  - Switch x/y overrides to correct columns.

- **Missing-column error**
  - Verify CSV headers.
  - Set explicit overrides for frame/id/x/y.

- **R KDE fails**
  - Install `MASS` or switch `DENSITY_METHOD` to `"hist"`.

- **MP4 encoding fails in R**
  - Install/update `av`.
  - If still unavailable, script writes GIF using `gifski`.

- **Rendering is slow**
  - Increase `FRAME_STRIDE` (e.g., `2` or `5`)
  - Reduce `DENSITY_BINS`
  - Reduce `FPS`
  - Keep `TEST_DURATION_SECONDS` set during development

