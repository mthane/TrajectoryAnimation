# Methods and legend draft (KDE density + trajectory outputs)

## Trajectory density maps

Larval centroid trajectories were summarized as time-resolved two-dimensional kernel
density estimate (KDE) maps in the Petri-dish arena (fixed axes +/-47.5 mm, dish radius
45 mm). Raw spatial histograms were not used for final figures; all publication outputs use
Gaussian KDE smoothing of per-frame occupancy.

**Movies** used KDE bandwidth sigma = 3.8 grid bins on a
56 x 56 bin grid (coarser cells for stable playback),
with a sliding temporal window of 180 frames (~11.2 s at
16 fps recording). Stronger smoothing avoids bright single-animal dots that
can appear with fine bins or weak smoothing.

**Static summary panels** (full ~3 min recording stacked into one map) used finer resolution:
sigma = 1.6 bins on a 96 x 96 grid.

Head-cast (HC) event density (when shown) used a sliding window of 360
frames (~22.5 s) and movie/summary grids of
40 / 64 bins.

Density was normalized by the maximum number of larvae co-visible in any frame
(`max_visible_larvae`) so maps are comparable across conditions.

## Color scale

Maps use the perceptually uniform **viridis** colormap with **linear** intensity mapping
(no sqrt or log transform). Larval maps use linear normalization; HC maps use linear normalization.

**Movies** share a global color scale across all conditions, with vmax set from the
99.5th percentile of sampled frame peaks (larvae movie vmax =
0.5834765031933785; HC movie vmax = 0.021733320020139217; HC percentile =
99.0).

**Static summary PNG/PDF panels** use a **tighter** color scale to highlight spatial
differences: vmax from the 93.0th percentile of full-recording
KDE peaks (larvae summary vmax = 2.1192602503299716; HC summary vmax =
0.05724595524370671; HC summary percentile = 90.0).

## Trajectory (line) animations

Cumulative larval paths were rendered as line/scatter animations with a trailing history of
the last 10 s per larva (individual colors per track ID). Compare
movies show fed vs starved side-by-side (fed left, starved right) with the same layout as
density compare movies.

## Movies

Animations play at 16 fps output (real-time (1x) and 2x). Filenames include `_realtime` or `_2x`.
Each density movie shows the full ~3 min recording, then holds the pooled full-recording
summary panel(s) for 10 s at the end (summary panels use the
tighter color limits described above).

## Legend suggestion

Time-resolved KDE density of larval positions (and optionally head-cast events) in the
arena; color intensity indicates normalized occupancy (linear viridis scale).
Trajectory animations show recent movement paths colored by individual larva.
Same movie color scale across panels/conditions; summary figures use a tighter scale.

## Exclusions

Tracks 1637, 2937, 2354, 2366 (out-of-arena artifacts in 5d Fed) were excluded.

## Comparisons

All publication compare movies and summary panels are side-by-side fed vs starved
(4d and 5d); fed is always the left panel, starved the right panel.

Per-condition outputs (density + trajectories) are saved under `per_condition/<condition>/`.
