# Methods and legend draft (KDE density outputs)

## Trajectory density maps

Larval centroid trajectories were summarized as time-resolved two-dimensional kernel
density estimate (KDE) maps in the Petri-dish arena (fixed axes +/-47.5 mm). Raw
spatial histograms were not used for final figures; all publication outputs use
Gaussian KDE smoothing of per-frame occupancy.

For movies, KDE bandwidth was sigma = 3.8 grid bins and the
render grid was 64 x 64 bins, with a sliding
temporal window of 180 frames (~11.2 s at
16 fps recording). Stronger smoothing avoids bright single-animal dots
that can appear with fine bins or weak smoothing.

For static summary panels (full 3-min recording), finer resolution was used:
sigma = 1.6 bins on a 120 x 120 grid.

Head-cast (HC) event density (when shown) used window 360 frames and
movie/summary grids 48/72 bins.

Density was normalized by the maximum number of larvae co-visible in any frame
(max_visible_larvae) so maps are comparable across conditions.

## Color scale

Maps use the perceptually uniform cividis colormap with global color limits
shared across all conditions (larvae vmax = 0.6224872222542762; HC vmax = 0.01745432533323765).
Values above the 99.5th percentile (99.0th for HC) were
clipped before setting vmax. Larval maps use sqrt (power 0.5) normalization for better
contrast among high-density regions.

## Movies

Animations play at 16 fps output (real-time (1x) and 2x). Filenames include `_realtime` or `_2x`.

## Legend suggestion

Time-resolved KDE density of larval positions (and optionally head-cast events) in the
arena; warmer colors indicate higher occupancy. Same color scale across panels/conditions.

## Exclusions

Tracks 1637, 2937, 2354, 2366 (out-of-arena artifacts in 5d Sated) were excluded.
