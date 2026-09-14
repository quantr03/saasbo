# sagp study: narrative analysis, 2026-09-12-1849

Interpretive account of the sagp high-dimensional BO study, built on the numeric analysis snapshot
`/scratch/work/tranq8/sagp_analysis/2026-09-12-1428` (`latest` at the time of writing; as-of 2026-09-12 14:47 EEST,
280/280 runs). Read-only on the runs and the repo; nothing under `sagp_analysis/latest` was modified and the symlink
was not repointed.

- `NARRATIVE.html`      self-contained HTML build of everything below (figures embedded, [L#] tags linked to the ledger,
                        supporting tables as collapsible appendices); built by `build_html.py` under Slurm (`run3.sbatch`).
- `NARRATIVE.md`        the narrative: answer, design, optimization by family, identification, diagnostics,
                        things that look wrong, claims for the paper, what to run next, 150-word summary.
- `captions.md`         draft captions for the six figures in `latest/figures/`.
- `claims_ledger.csv`   every [L#] tag in the narrative -> file, row/column, kind (observation / inference / hypothesis).
- `tables/`             this narrative's own computations (see below).
- `compute.py`, `compute2.py`, `run.sbatch`, `run2.sbatch`, `logs/`   how the tables were produced (Slurm jobs
                        20228562 and 20228597, analysis venv `latest/venv`).

## tables/

From `compute.py`:
- `eps_constants.json`                 ELL_EPS, RHO_EPS, ALPHA_AMPLITUDE reproduced with numpy from gp.py's definitions.
- `decoupled_S_rank_check.csv`         rank order of S recomputed from families.build's seeding, checked against manifests.
- `decoupled_per_coordinate.csv`, `decoupled_breakdown.csv`   which of the 8 decoupled coordinates each method finds,
                                       by (share, lengthscale) class: detected (p_active>0.5), top-8 by native / sobol.
- `degenerate_scan_all_runs.csv`, `degenerate_scan_flagged.csv`   every readout of every cell run scanned for >=50
                                       active coordinates or AP<0.3 at t>=100.
- `degenerate_trajectory_aligned3_AL.csv`   per-readout trajectory (n active, AP, rho on/off S, gate status) for
                                       aligned3/additive-lengthscale seeds 00, 02, 08.
- `regret_vs_identification_per_run.csv`, `regret_vs_identification_spearman.csv`, `final_regret_wide.csv`.
- `native_on_true_S_per_coordinate.csv`, `native_on_true_S_summary.csv`, `native_off_S_per_run.csv`
                                       the native score and p_active on the true coordinates (and off them) at t=199.
- `lockon_times.csv`                   first t at which the median AP / F1 crosses 0.9 / 0.8, per family and cell.

From `compute2.py`:
- `ten_seed_bounds.csv`                sign-test p and exact 95% CI on the win probability for k of 10 paired seeds.
- `identical_regret_pairs.csv`, `incumbent_boundary_per_run.csv`, `incumbent_boundary_summary.csv`
                                       the product-lengthscale / oracle_S identical-regret seeds; how often the
                                       incumbent sits on the box boundary in the active coordinates.
- `collapse_draws_aligned3_AL.csv`     the retained NUTS draws (outputscale, noise, tausq, rho) in the sparse and the
                                       all-active mode of aligned3/additive-lengthscale.
- `native_off_S_summary.csv`           off-S native maxima and false positives per cell and family.
- `near_duplicate_queries_per_run.csv`, `near_duplicate_queries_summary.csv`   fraction of late queries within 0.01
                                       (max-norm) of an earlier query.

Login-node reads (no compute):
- `y_std_final_per_run.csv`, `y_std_final_summary.csv`   the target standardization std at t=199 per cell run.
- `dsp_map_fallbacks.csv`              dsp_map's NotPSDError exception rows and Adam-fallback rows per run.
