# R2-D2 study analysis snapshot

Read-only analysis of the half-Cauchy (HC) runs under `runs/` and the R2-D2 runs under `runs_r2d2/`
(plan `docs/superpowers/plans/2026-09-27-r2d2-cells.md`, D11 and Task 14; preregistration
`docs/superpowers/specs/2026-09-27-r2d2-prereg.md`). One script serves the pilot's gate G3 (stage 3)
and, unchanged, the full analysis of stage 4.

- `analyze.py`       the script, derived from `../../sagp2/2026-09-25-1242/analyze.py` (the `--families`
                     version). `--runs` is repeatable, `--families` and `--seeds` select the scope, and a
                     method table (11 methods: 4 HC cells, their 4 R2-D2 twins, 3 references) replaces the
                     hard-coded lists. On the stored HC runs of sagp2's four families it reproduces every
                     sagp2 table it shares (paired tests, final regret, identification, diagnostics, cost)
                     to 1e-12.
- `run_analysis.sbatch` builds `venv/` (numpy, scipy, pandas, matplotlib; separate from the study's
                     `saasbo` env) and runs the script under Slurm.
- `test_analyze.py`  tests of the pure functions (effects, G3, bootstrap, readouts, pairing and
                     prior-code checks) and, marked slow, the whole pipeline on a synthetic fixture and a
                     cross-check of the method table against `sagp.gp.CELLS`.
- `fixture.py`       builds a `runs_r2d2/`-shaped tree from stored HC runs with a known injected Delta.

Outputs (written by a run into this directory): `REPORT.md`, `g3.json`, `summary.json`, `figures/`,
`tables/`, and under Slurm `venv/`, `venv-freeze.txt` and the `*.out` log. `.gitignore` ignores all
of them, so a run never dirties the clone (a dirty clone would fail the next launch's prior-code
check); commit the ones worth keeping with `git add -f`. New against sagp2: `effects_per_unit`,
`effects_by_family`, `effects_pooled`, `g3`, `pr1_sign_agreement`, `pr3_twin_medians`,
`pr4_identification_twin_pairs`, `identification_by_prior_family`, `r2_readouts_per_run_t`,
`r2_readouts_vs_t_median`, `cost_twin_by_device`, `sanity_twin_pairing`, `sanity_prior_code`.

G3 is read (GO / STOP) only on the pilot's scope, `--families aligned10,decoupled --seeds 0-4`, with all
40 twin pairs complete and every sanity check passing; otherwise `REPORT.md` says INCONCLUSIVE (with
what is missing) or "not read" (another scope). The prior-code check needs a git repository holding
every run's commits (`--repo`, default the repository this script is in). `--launch-commit` anchors
it: REPORT.md prints the (`sagp/gp.py`, `sagp/r2d2.py`) blob pair at that commit, and
`prior_code_ok` fails if the R2-D2 runs' modal pair is another one. The sbatch takes it as
`LAUNCH_COMMIT`; the pilot's is 97eda925b, which stage 4 keeps, since the pilot's runs are part of
the full replicate.

Run on Triton (from anywhere; the snapshot lives inside the clone `$WRKDIR/saasbo`):

    sbatch --chdir=<this dir> --export=ALL,ASOF=,LAUNCH_COMMIT=97eda925b run_analysis.sbatch   # pilot, G3
    sbatch --chdir=<this dir> --export=ALL,ASOF=,LAUNCH_COMMIT=97eda925b,SEEDS=0-9,FAMILIES=aligned3+aligned10+anti_aligned+decoupled+interaction_g0.00+interaction_g0.25+interaction_g0.50+interaction_g0.75 run_analysis.sbatch

Locally, with an interpreter that has pandas and matplotlib (the `saasbo` env has neither):

    python sagp_analysis/r2d2/<stamp>/analyze.py --runs runs --runs runs_r2d2 \
        --out sagp_analysis/r2d2/<stamp> --families aligned10,decoupled --seeds 0-4 --launch-commit 97eda925b
    python -m pytest -q -m "not slow" sagp_analysis/r2d2/<stamp>/test_analyze.py   # fast tests
    python -m pytest -q -m slow sagp_analysis/r2d2/<stamp>/test_analyze.py         # fixture pipeline

This study lives under `sagp_analysis/r2d2/` with its own `latest`, so "What changed" diffs R2-D2
snapshots against each other. For stage 4, copy `analyze.py`, `run_analysis.sbatch`, `fixture.py` and
`test_analyze.py` into a new `sagp_analysis/r2d2/<YYYY-MM-DD-HHMM>/` and repoint `latest` afterwards.
The stage-4 confirmatory run should use the package versions of the G3 run: before its first
sbatch, build `venv/` in the new directory and install the G3 run's `venv-freeze.txt` into it
(with the `saasbo` env's python, as the sbatch does: `python -m venv venv && venv/bin/python -m pip
install -r <G3 dir>/venv-freeze.txt`); the sbatch then reuses that venv instead of installing the
latest versions.
