# sagp2 study analysis snapshot

Read-only analysis of `/scratch/work/tranq8/saasbo/runs` for the SECOND family set
(interaction_g0.00, interaction_g0.50, interaction_g0.75, anti_aligned; 7 methods x 10 seeds = 280 runs,
launched 2026-09-21 from commit 2866ed21, finished 2026-09-25).

Same script and layout as `../../2026-09-12-1428` (see its README): `REPORT.md`, `figures/`, `tables/`,
`summary.json`, `analyze.py`, `run_analysis.sbatch`. `venv` is a symlink to the first snapshot's venv.
This study lives under `sagp_analysis/sagp2/` with its own `latest`, so "What changed" diffs sagp2
snapshots against each other and not against the first family set.

Re-run: copy `analyze.py` and `run_analysis.sbatch` into a new `sagp_analysis/sagp2/<YYYY-MM-DD-HHMM>/`, symlink
`venv`, then from anywhere:
`sbatch --chdir=<dir> --export=ALL,ASOF=,FAMILIES=interaction_g0.00+interaction_g0.50+interaction_g0.75+anti_aligned run_analysis.sbatch`
(plus-separated: Slurm splits `--export` values on commas), and repoint `sagp_analysis/sagp2/latest` afterwards.
