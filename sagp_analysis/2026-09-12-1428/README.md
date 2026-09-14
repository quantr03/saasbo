# sagp study analysis snapshot

Read-only analysis of `/scratch/work/tranq8/saasbo/runs` (never modified).

- `REPORT.md`      the findings; starts with the five things to know, then sections 1-8 as requested.
- `figures/*.png`  regret curves, final-regret strip, identification, identification vs t, diagnostics, cost vs t.
- `tables/*.csv|md` every table in the report plus per-run detail (inventory, sanity per run, identification per run,
                    final regret per run, regret curves median/IQR, gate reasons, cost by device).
- `tables/anomalies.md`, `tables/what_changed.md`, `summary.json`  machine-readable side outputs.
- `analyze.py`     the script; `run_analysis.sbatch` builds `venv/` (numpy, scipy, pandas, matplotlib; separate from the
                    study's `saasbo` env) and runs it under Slurm. `venv-freeze.txt` pins what was installed.

Re-run: copy `analyze.py` and `run_analysis.sbatch` into a new `sagp_analysis/<YYYY-MM-DD-HHMM>/`, then
`sbatch --export=ALL,ASOF="<timestamp>" run_analysis.sbatch`, and repoint `sagp_analysis/latest`.
The script diffs against whatever `latest` pointed at when it ran (section "What changed").
