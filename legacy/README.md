# Legacy materials

Archived scripts, intermediate outputs, and experiments kept for reference. **Not required** for reproducing the final publication figures.

## Scripts

| Item | Notes |
|------|-------|
| `create_larvae_animations.R` | Unsupported R port; use Python instead |
| `generate_paper_outputs.py` | Earlier manuscript batch (hist + KDE via `animations/paper/`) |
| `run_paper_hq_clean.ps1` | PowerShell batch for HQ iteration runs |
| `scripts/finalize_starvation_kde_movies.py` | Older KDE-only driver (cividis, no trajectories) |
| `prototype_hex_pie_animation.py` | Experimental hex-grid categorical animation |

## Outputs

| Folder | Notes |
|--------|-------|
| `animations/` | Development runs (`paper/`, `paper_hq_*`, dated reruns) |
| `outputs/starvation_kde_final_test/` | Short test render |
| `outputs/starvation_kde_final_compare/` | Old compare slug (`4d_sated_vs_starved`) |
| `outputs/logs/` | Pipeline run logs |
| `archives/` | Zip bundles of animations/outputs |

## Active pipeline

Use the root-level scripts documented in `README.md`:

- `create_larvae_animations.py` — core engine
- `finalize_starvation_kde_movies.py` — final KDE + trajectory publication batch
- `outputs/starvation_kde_final/` — manuscript deliverables
