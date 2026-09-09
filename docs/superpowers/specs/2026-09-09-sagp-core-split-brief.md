# sagp core-split brief (verbatim, 2026-09-09)

You are planning (not yet implementing) a restructuring of the `sagp` module in the repository
`/Users/quantr03/Library/CloudStorage/OneDrive-AaltoUniversity/2026 Summer Internship'/saasbo`
(the path contains an apostrophe; quote it in every command). Read this whole brief, then produce an
implementation plan. Do not write code until I approve the plan.

## Goal

`sagp/gp.py` and `sagp/bo.py` must become minimal core files in the spirit of the vendored SAASBO
reference at the repo root: `saasgp.py` (198 lines: kernels, one NumPyro model, `run_inference`,
`compute_choleskys`/`predict`/`posterior`) and `saasbo.py` (190 lines: `ei`, `optimize_ei`, one loop
function). Everything that is experiment machinery — readouts, the offline identification runner, the
MAP reference fits, logging, checkpoint/resume, provenance manifests, the CLI, the pilot — moves out
into experiment modules that call the core through its public API. The main experiment code (SQ1
identification, SQ2/SQ3 BO runs) must read as a script that imports `sagp.gp` and `sagp.bo` and calls
their functions, the way `saasbo.run_saasbo` calls `SAASGP`.

Today `sagp/gp.py` is 1,674 lines and `sagp/bo.py` 1,278 lines. Their top-level sections are:
- gp.py: `# --- kernels ---` (matern52_1d, v_of_ell, _quad_mean, _kbar_all, _kbar_diag,
  centered_matern52_1d, the four cell kernels, four diagonal helpers, cell_kernel_diag);
  `# --- cells ---` (the four NumPyro models, `Cell`, `CELLS`, the prior constants ACTIVE_EPS,
  ALPHA_LENGTHSCALE, ELL_EPS, RHO_EPS, ALPHA_AMPLITUDE, ELL_PRIOR); `# --- inference ---`
  (NUTSConfig, DiagThresholds, Diagnostics, _chunk_size, FittedGP with posterior/alphas, _run_nuts,
  _group_extremes, _diagnose, fit); `# --- MAP references ---` (_ell_prior_loc, _dsp_start,
  _dsp_neg_log_joint, fit_map); `# --- readouts ---` (shares_from_amplitudes, component_means,
  _sobol_exact_additive, _pair_term, _sobol_exact_product_amplitude, _posterior_mean, _sobol_qmc,
  readouts, manipulation_checks); `# --- identification ---` (standardize, identify).
- bo.py: `# --- seeding ---` (IterRNG, iteration_rngs, initial_design); `# --- acquisition ---`
  (log1mexp, log_h, log_ei, log_ei_sum, optimize_ei — a five-line-diff copy of saasbo.optimize_ei);
  `# --- configuration ---` (RunConfig, config_hash, run_dir_for); `# --- provenance ---`
  (_git_provenance, _versions, _environment, _reference_sha256, _objective_labels);
  `# --- the run directory ---` (RunLogger: checkpoint, CSV rows, coords, samples npz, manifest,
  resume truncation); `# --- one iteration ---` (_fit_for, _fit_with_retry, _propose, _observe,
  _iteration); `# --- the loop ---` (run_bo); `# --- the command line ---` (argparse, resolve_config,
  _load_objective, main).

## What must not change (behavioural contract)

- Reference fidelity, pinned by tests: the product/lengthscale cell's NUTS draws equal
  `saasgp.SAASGP` bit-for-bit (tests/test_sagp_inference.py, `-k bit_for_bit`, three tests), its
  posterior equals `SAASGP.posterior` exactly (tests/test_sagp_cells.py), and with `acq="ei"`,
  `fixed_noise=1e-6`, `noiseless=True` the loop reproduces `saasbo.run_saasbo`'s 25-point trajectory
  bit-for-bit (tests/test_bo_reference.py, slow). These must pass unchanged in substance.
- The four cells differ in nothing but the kernel/prior block: one sampler call, one budget
  (512/256/16, tree depth 6, one chain, fresh chain per fit), one standardization
  (`z = -(y - mean)/std`), one acquisition (LogEI; `acq="ei"` only for the reproduction test), one
  optimizer (the copied `optimize_ei`), one seeded initial design, one per-iteration seeding scheme
  (`iteration_rngs(seed, t)` — every draw from (seed, t) only).
- The diagnostics and their gate (thresholds 1.1 / 16 / 5 on the log of every sampled positive
  site, per-group fields, one refit with doubled warm-up, `ok`/`refit`/`excluded`) stay as they are
  — a separate decision is pending on the gate policy and this restructuring must not entangle it.
- The prediction memory policy (row-chunking when n_test·(n+64)·D > 2^24; chunk size = the largest
  divisor of S ≤ 8) stays; it is what makes D = 100 runs fit in memory.
- The experiment outputs stay reproducible and resumable: `iterations.csv` schema, `coords.csv`,
  `samples/t{t:03d}.npz`, `checkpoint.npz`, `manifest.json`, `environment.lock.txt`, bit-identical
  resume, the CLI contract (`python -m sagp.bo --family … --seed … --cell … --T … --out …`,
  `--dry-run`, exit codes) — the CLI may move to a different module path if the plan says so, but
  the SLURM-facing command line must remain a one-liner.
- Vendored `saasgp.py`, `saasbo.py`, `util.py` remain byte-identical (one earlier compat line in
  saasgp.py is already committed). The 442 non-slow tests and the 10 slow tests must all still
  pass; tests may be moved and renamed to follow the code, never weakened.

## Read before planning (in this order)

1. `docs/superpowers/specs/2026-09-07-sagp-brief.md` — the original brief (binding spec).
2. `docs/superpowers/plans/2026-09-07-sagp-model-and-bo.md` — the approved plan: §2 interface,
   §3 layout, §4 the loop, §5 the eight decisions (D1 reuse-vs-copy is the one this restructuring
   revisits), §6 test map.
3. `.superpowers/sdd/2026-09-07-sagp-model-and-bo/progress.md` — the execution ledger: every ruling
   R0–R48, the deferred minors, the cost measurements (the centered cells are 15–60× the reference
   per gradient in-loop; the full budget was nevertheless kept), and the pilot results.
4. `saasgp.py`, `saasbo.py` — the shape and size you are aiming for.
5. `sagp/gp.py`, `sagp/bo.py`, `sagp/__init__.py`, `scripts/pilot_sagp.py`, the nine `tests/test_sagp_*.py`
   and `tests/test_bo_*.py` files, `README.md`'s sagp section.

## Decisions the plan must make (with a recommendation and reason for each)

1. The target module map: which names stay in `gp.py` and `bo.py` (my expectation: gp.py keeps
   kernels, the four models, `NUTSConfig`, `fit`, `FittedGP.posterior`; bo.py keeps
   `iteration_rngs`/`initial_design`, `log_ei`, `optimize_ei`, and a `run_bo` that is a loop and
   nothing else), and where everything else goes (e.g. `sagp/readouts.py`, `sagp/references.py`
   for `fit_map`/oracle-S/Sobol search, `sagp/identify.py`, `sagp/experiment.py` or
   `experiments/run_bo.py` for RunConfig/RunLogger/provenance/CLI). Give line-count targets per
   file and justify each boundary by what the reference keeps in its two files.
2. Where the diagnostics live: `_diagnose`/`Diagnostics`/`DiagThresholds` are part of `fit`'s
   contract (status), so they may need to stay in gp.py — or `fit` returns raw draws and the
   experiment layer diagnoses. State which, and what it does to `fit`'s signature and to the
   bit-for-bit tests.
3. The public API of the core after the split (signatures with one-line docstrings), and the rule
   that the experiment layer never reaches into private names.
4. Dependency direction: core modules import nothing from the experiment layer; `sagp/__init__.py`
   lazy exports; no import of `synthobj` from gp.py (bo.py currently uses
   `synthobj.families.noise_rng` for the seeding — decide whether that stays or moves).
5. How the move is done so the git history stays reviewable: pure-move commits (no logic change,
   verified by the full suite) separated from the small edits that adjust imports and signatures;
   which tests move with which code; whether `scripts/pilot_sagp.py` needs any change.
6. Test relocation map (old file → new file) and the exact commands that prove nothing regressed at
   each step, including the three bit-for-bit tests after every move commit.

## Deliverable of this planning step

A plan document under 400 lines with: the target module map with line budgets; the core public API
(signatures); the sequence of move-then-edit commits as bite-sized tasks, each with the verifying
test command; the test relocation map; effort in hours; and at most five open questions for me. Do
not weaken any test; do not change numerics; do not touch the gate policy or the budget. Ask me only
if something blocks the plan; otherwise make the reasonable choice and flag it.

## Environment and process

- Interpreter `/opt/anaconda3/envs/saasbo/bin/python` (jax 0.10.2, numpyro 0.21.0, scipy 1.17.1,
  numpy 2.4.6); run tests as `... -m pytest -q -m "not slow"` (about 3.5 minutes; run in parts of
  ≤ 300 s), slow tests individually.
- Branch `feat/sagp-model-bo` at 0fd2c19 (22 commits from 796f35e). Another Claude session may share
  this checkout: never `git checkout`/`switch`/`stash`/`worktree` here, verify
  `git branch --show-current` before every commit, stage files by path only (the tree carries the
  user's untracked notes under `Research Context/` and pre-existing deletions that are not yours).
- After approval, execute with subagent-driven development: fresh implementer per task, a review per
  task, one ledger (continue `.superpowers/sdd/2026-09-07-sagp-model-and-bo/progress.md` or start a
  sibling for this plan), rulings recorded, never two implementers at once, and no implementer
  running a multi-minute test in the foreground.

## Answers to the plan's open questions (Quan, 2026-09-09 10:26–10:27)

1. Trim the ruling-narrating docstrings in a final docs-only commit: **yes**.
2. `experiments/` at the repo root: **root**.
3. `FittedGP` keeps the reference names as a deferred minor: **fine**.
4. Exception-policy messages through a `log` callable (default `print`) rather than `logging`: **callable**.
