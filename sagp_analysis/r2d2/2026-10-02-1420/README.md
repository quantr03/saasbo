# R2-D2 study analysis snapshot (stage 4, confirmatory)

This snapshot is the stage-4 copy of the G3 snapshot `../2026-09-29-0245/`, which stays as the record of
gate G3. Two changes against it, from the Task 14b re-review: REPORT's caveat on G3 criterion (b) names
the run's actual number of (family, seed) clusters (`g3_interval_caveat`) instead of a fixed "10", and
`summary.json` records the interpreter's python, numpy, scipy, pandas and matplotlib versions
(`versions`); `run_analysis.sbatch` writes `venv-freeze.txt` on every run, also when `venv/` was
pre-built. Rerun on the pilot scope with the G3 interpreter, this copy reproduces the G3 directory's
`g3.json` byte for byte and every file under `tables/`.

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
                     `saasbo` env) unless it exists, records it in `venv-freeze.txt`, and runs the
                     script under Slurm.
- `test_analyze.py`  tests of the pure functions (effects, G3, bootstrap, readouts, pairing and
                     prior-code checks) and, marked slow, the whole pipeline on a synthetic fixture and a
                     cross-check of the method table against `sagp.gp.CELLS`.
- `fixture.py`       builds a `runs_r2d2/`-shaped tree from stored HC runs with a known injected Delta.

Outputs (written by a run into this directory): `REPORT.md`, `g3.json`, `summary.json`, `figures/`,
`tables/`, and under Slurm `venv/`, `venv-freeze.txt` and the `*.out` log. `summary.json`'s `versions`
records the interpreter and the four packages a run used, under Slurm or locally. `.gitignore` ignores all
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

## The stage-4 run

Stage 4 is the confirmatory analysis of the full replicate: all eight objective families, ten seeds,
once all 320 R2-D2 runs are complete (the prereg allows no interim analysis). From the repository root,
with the G3 run's interpreter:

    python sagp_analysis/r2d2/2026-10-02-1420/analyze.py --runs runs --runs runs_r2d2 --out sagp_analysis/r2d2/2026-10-02-1420 --families aligned3,aligned10,anti_aligned,decoupled,interaction_g0.00,interaction_g0.25,interaction_g0.50,interaction_g0.75 --seeds 0-9 --launch-commit 97eda925b

That is, `python` is `/opt/anaconda3/bin/python`, the interpreter the G3 run was executed with (locally,
not under the sbatch). Its versions, which this snapshot's pilot-scope rerun recorded in `summary.json`
and which reproduced G3's `g3.json` and `tables/` exactly: python 3.13.9, numpy 1.26.4, scipy 1.16.3,
pandas 2.3.3, matplotlib 3.10.6. Check the new `summary.json`'s `versions` against them. `--families`
takes a comma-separated list (the sbatch's `FAMILIES` is plus-separated only because Slurm splits
`--export` on commas).

G3 is read only on the pilot's scope (`--families aligned10,decoupled --seeds 0-4`), so this run's REPORT
says "G3: not read" and its G3 table is descriptive. The confirmatory results are the predictions PR1-PR4
and the effects tables (`effects_by_family`, `effects_pooled`, `pr1_sign_agreement`, `pr3_twin_medians`,
`pr4_identification_twin_pairs`), read as the signed prereg `docs/superpowers/specs/2026-09-27-r2d2-prereg.md`
states them with its 2026-09-29 clarifications: PR1's sign is that of the median over seeds; PR4 reads
the 90 % (family, seed)-cluster bootstrap interval of the median paired AP difference against
(-0.05, 0.05); Holm runs over four kinds within each objective family ({S(HC), P(HC)},
{S(R2-D2), P(R2-D2)}, {dS, dP}, the four Delta_c). The pooled statistics resample 80 (family, seed)
clusters.

`latest` already points at this directory, so the stage-4 REPORT's "What changed" section finds no
previous analysis; to diff against G3, compare with `../2026-09-29-0245/` directly.

## Other ways to run it

Run on Triton (from anywhere; the snapshot lives inside the clone `$WRKDIR/saasbo`):

    sbatch --chdir=<this dir> --export=ALL,ASOF=,LAUNCH_COMMIT=97eda925b run_analysis.sbatch   # pilot, G3
    sbatch --chdir=<this dir> --export=ALL,ASOF=,LAUNCH_COMMIT=97eda925b,SEEDS=0-9,FAMILIES=aligned3+aligned10+anti_aligned+decoupled+interaction_g0.00+interaction_g0.25+interaction_g0.50+interaction_g0.75 run_analysis.sbatch

Locally, with an interpreter that has pandas and matplotlib (the `saasbo` env has neither):

    python sagp_analysis/r2d2/<stamp>/analyze.py --runs runs --runs runs_r2d2 \
        --out sagp_analysis/r2d2/<stamp> --families aligned10,decoupled --seeds 0-4 --launch-commit 97eda925b
    python -m pytest -q -m "not slow" sagp_analysis/r2d2/<stamp>/test_analyze.py   # fast tests
    python -m pytest -q -m slow sagp_analysis/r2d2/<stamp>/test_analyze.py         # fixture pipeline

This study lives under `sagp_analysis/r2d2/` with its own `latest`, so "What changed" diffs R2-D2
snapshots against each other. A later snapshot copies `analyze.py`, `run_analysis.sbatch`, `fixture.py`
and `test_analyze.py` into a new `sagp_analysis/r2d2/<YYYY-MM-DD-HHMM>/` and repoints `latest`.

If stage 4 runs under the sbatch instead of locally, it should use the G3 run's package versions. The G3
run was local, so there is no G3 `venv-freeze.txt` to copy in; build `venv/` in this directory before the
first sbatch, pinned to the versions above, with the `saasbo` env's python as the sbatch does:
`python -m venv venv && venv/bin/python -m pip install numpy==1.26.4 scipy==1.16.3 pandas==2.3.3
matplotlib==3.10.6`. (Where a run does leave a `venv-freeze.txt`, copy it into the next snapshot and
install it with `pip install -r` instead.) The sbatch then reuses that venv and writes this run's own
`venv-freeze.txt`.
