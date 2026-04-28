# Larval Trajectory Animation

This project generates animation videos of larval positions over time from tracking data.

The script auto-detects position coordinates, with preference order:
- `x`: `centroid_x` -> `centoid_x` -> `spinepoint_x6_conv` -> `spinepoint_x_6_conv`
- `y`: `centroid_y` -> `centoid_y` -> `spinepoint_y6_conv` -> `spinepoint_y_6_conv`

## What the Script Produces

Running `create_larvae_animations.py` creates:
- `animations/larvae_tracks_window.mp4`  
  Sliding-window trajectories for all larvae, with fading trails.
- `animations/larvae_density.mp4`  
  Sliding-window 2D occupancy density map with fixed color scale.

## Requirements

- Python 3.9+ (3.10+ recommended)
- Packages:
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `opencv-python`

## Installation

From the project root:

```bash
python -m venv .venv
```

Activate virtual environment:

- Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

- macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib opencv-python
```

## Input Data

By default, the script reads:
- `embedding_dan.csv`

It auto-detects:
- Frame/time column (for example: `frame`, `t`, `time`)
- Larva ID column (for example: `id`, `track_id`)

It requires one detected position pair from the list above.

If required columns are missing, the script exits with a clear error and prints available columns.

## Usage

From the project root:

```bash
python create_larvae_animations.py
```

The script will:
1. Inspect and detect key columns.
2. Convert frame/x/y to numeric.
3. Remove rows with missing frame/id/x/y.
4. Sort rows by frame and ID.
5. Compute global fixed x/y limits.
6. Render track and density videos to `animations/`.

Example: force centroid columns and filter one condition by editing config:

```python
X_COLUMN_OVERRIDE = "centroid_x"
Y_COLUMN_OVERRIDE = "centroid_y"
GROUP_CONDITION_COLUMN = "condition"
GROUP_CONDITION_VALUE = "EM_Rewarded"
RECORDING_FPS = 16
FPS = 16
```

## Configuration

Edit the config section at the top of `create_larvae_animations.py`:

- `X_COLUMN_OVERRIDE`, `Y_COLUMN_OVERRIDE` - manually set x/y position columns (or `None` for auto-detect)
- `FRAME_COLUMN_OVERRIDE`, `ID_COLUMN_OVERRIDE` - manually set frame/ID columns (or `None` for auto-detect)
- `GROUP_CONDITION_COLUMN` - column name used for group filtering (default `condition`)
- `GROUP_CONDITION_VALUE` - set a value (for example, `EM_Rewarded`) to filter rows, or `None` to disable
- `DATA_PATH` - input CSV path
- `OUTPUT_DIR` - output folder
- `WINDOW_SIZE_FRAMES` - sliding window size (default `50`)
- `FRAME_STRIDE` - frame subsampling factor
- `RECORDING_FPS` - recording frame rate used for time-based test slicing
- `FPS` - video frame rate
- `DPI`, `FIGSIZE` - output resolution/figure size
- `MAKE_TRACK_ANIMATION` - enable/disable track video
- `MAKE_DENSITY_ANIMATION` - enable/disable density video
- `DENSITY_BINS` - density grid resolution
- `DENSITY_CMAP`, `TRACK_CMAP` - colormaps
- `DENSITY_VMAX_SAMPLE_FRAMES` - sample count used to estimate fixed density color max
- `DENSITY_METHOD` - `"hist"` or `"kde"` for smooth density
- `KDE_SIGMA_BINS` - smoothing bandwidth when using KDE mode
- `INFER_PETRI_DISH` - enable petri dish boundary overlay
- `PETRI_MARGIN_FRACTION` - margin used for inferred petri circle

## Notes

- Original data files are never modified.
- Axis limits are fixed for publication-friendly comparisons over time.
- The plot uses equal aspect ratio (`x` and `y` on same scale).
- Density video uses a fixed color scale over the full animation.

## Troubleshooting

- **Error: required columns missing**  
  Check CSV headers and ensure centroid columns are present.

- **Video writer / codec error**  
  Install or update OpenCV:
  ```bash
  python -m pip install --upgrade opencv-python
  ```

- **Very slow rendering on large datasets**  
  Reduce runtime by adjusting:
  - `FRAME_STRIDE` (for example, `2` or `5`)
  - `DENSITY_BINS` (for example, `60`)
  - `FPS`

- **All points collapse to one location**  
  This usually means centroid values in the source data are constant (often near zero).  
  Verify the values in `spinepoint_x6_conv` / `spinepoint_y6_conv` (or `_6_` variants).

