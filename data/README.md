# Input data

Large tracking CSVs are **not** committed to git. Place your files here (or pass an absolute path to the scripts).

## Publication dataset

| File | Used by |
|------|---------|
| `starvation2-2026-04-23_analyzed.csv` | `finalize_starvation_kde_movies.py` (starvation manuscript) |

Expected columns (auto-detected): `frame`, `id`, `centroid_x` / `centoid_x`, `centroid_y`, `condition`, and optionally `HCs` for head-cast density.

## Optional demo / development files

| File | Purpose |
|------|---------|
| `embedding_dan.csv` | Default sample for `create_larvae_animations.py` quick tests |
| `Learning_annotated_01_12.csv` | Hex-pie prototype (see `legacy/`) |

## Example

```bash
python finalize_starvation_kde_movies.py data/starvation2-2026-04-23_analyzed.csv
```
