# sagp core split: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce `sagp/gp.py` and `sagp/bo.py` to the reference's shape (model + inference + prediction; acquisition + one loop) by moving every piece of experiment machinery into modules that call the core through its public API, with zero numerical change.

**Architecture:** `sagp/` becomes a library of five modules (`gp`, `diagnostics`, `bo`, `references`, `readouts`); a new top-level `experiments/` package holds the study code (`identify`, `run_bo`, `runlog`). The core loop `sagp.bo.run_bo` gains two adapter slots (`surrogate`, `propose`) and one observer hook (`on_iteration`); the experiment layer supplies the method dispatch, the run directory and the CLI. Every move is a cut-and-paste commit verified by the full suite; the only real refactor is the loop (Task 6b), guarded by the reproduction test and a golden-output diff.

**Tech Stack:** unchanged — `/opt/anaconda3/envs/saasbo/bin/python`, jax 0.10.2, numpyro 0.21.0, scipy 1.17.1, numpy 2.4.6.

**Spec:** the restructuring brief given in conversation on 2026-09-09 (save verbatim as `docs/superpowers/specs/2026-09-09-sagp-core-split-brief.md` in Task 0). Binding upstream spec: `docs/superpowers/specs/2026-09-07-sagp-brief.md`; approved plan `docs/superpowers/plans/2026-09-07-sagp-model-and-bo.md` (§2 interface, §4 loop, D1); ledger `.superpowers/sdd/2026-09-07-sagp-model-and-bo/progress.md` (R11 standardize placement, R19/R21 chunking, R34/R35/R37 write order, R42–R45).

## Global Constraints

- Branch `feat/sagp-model-bo` at 0fd2c19; another session may share the checkout: run `git branch --show-current` immediately before every commit, stage by path only, never `checkout`/`switch`/`stash`/`worktree`, never touch `Research Context/`.
- Vendored `saasgp.py`, `saasbo.py`, `util.py`: byte-identical (`git diff 0fd2c19 -- saasgp.py saasbo.py util.py` empty at every commit).
- No numerical change anywhere: the three `-k bit_for_bit` tests pass after every commit; the slow reproduction test passes after Task 6b and at the end; the golden diff (Task 6b) is byte-equal on every non-timing column.
- Gate policy, budget, thresholds, chunking rule, seeding scheme, row schema, file names, resume semantics: untouched. `fit`'s signature: untouched.
- Tests are moved and renamed to follow the code, never weakened; 442 non-slow + 10 slow stay green (plus the additions listed in §6).
- Test commands (omit any file a task has not created yet): part A `python -m pytest -q -m "not slow" tests/test_sagp_*.py tests/test_identify.py tests/test_layering.py`; part B `python -m pytest -q -m "not slow" tests/test_bo_*.py tests/test_run_bo.py` (each < 300 s); `BIT = python -m pytest -q tests/test_sagp_inference.py -k bit_for_bit`; slow tests one file at a time, never in an implementer's foreground.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.

---

## 1. Decisions

**D1 — Target module map** (line budgets are for the code as it is today; see open question 1 for the prose).

| file | keeps / receives | lines now → target | reference analogue |
|---|---|---|---|
| `sagp/gp.py` | kernels, `KERNELS`, `cell_kernel_diag`, prior constants, four models, `Cell`/`CELLS`, `NUTSConfig`, `FittedGP` (posterior, alphas), `_run_nuts`, `fit`, `standardize` | 1674 → ≤ 920 | `saasgp.py`: kernels, model, `run_inference`, choleskys/predict/posterior — the same four things, times four cells |
| `sagp/diagnostics.py` (new) | `DiagThresholds`, `Diagnostics`, `_POSITIVE_SAMPLED_SITES`, `_DIAG_GROUPS`, `_group_extremes`, `diagnose` | 0 → ≤ 220 | none: the reference computes `summary` and prints; ours is a gate, so it is its own module (D2) |
| `sagp/bo.py` | `IterRNG`, `iteration_rngs`, `initial_design`, `log1mexp`, `log_h`, `log_ei`, `log_ei_sum`, `ACQUISITIONS`, `optimize_ei`, `propose_ei`, `BOState`, `Iteration`, `run_bo` | 1278 → ≤ 420 | `saasbo.py`: `ei`, `optimize_ei`, one loop with initial design, standardization and exception fallback — exactly what stays |
| `sagp/references.py` (new) | the MAP block (`_ELL_FLOOR` … `fit_map`), `propose_sobol` | 0 → ≤ 230 | none: the references are study apparatus |
| `sagp/readouts.py` (new) | `shares_from_amplitudes` … `manipulation_checks` (whole `# --- readouts ---` section) | 0 → ≤ 360 | none: analysis of a posterior, used by both studies |
| `experiments/identify.py` (new) | `identify` | 0 → ≤ 130 | SQ1 runner: a script that imports `sagp.gp`, `sagp.readouts` |
| `experiments/runlog.py` (new) | `RunConfig`, `config_hash`, `run_dir_for`, provenance helpers, row/coord field tables, `RunLogger` (+ `record`), `_complete_rows`, `_truncate_csv` | 0 → ≤ 620 | none: everything that touches a run directory |
| `experiments/run_bo.py` (new) | `METHODS`, `surrogate_for`, `propose_for`, `run`, `_nuts_config`, `_build_parser`, `resolve_config`, `_load_objective`, `main` | 0 → ≤ 380 | the SLURM entry point: `python -m experiments.run_bo …` |
| `sagp/__init__.py` | lazy exports `fit`, `FittedGP` (gp), `readouts` (readouts), `run_bo` (bo); `identify` dropped | 45 → 45 | — |
| `scripts/pilot_sagp.py` | no change (every name it imports stays in `sagp.gp`/`sagp.bo`); verified by `--help` | 707 | — |

Why top-level `experiments/` and not `sagp/experiment.py`: the dependency rule (D4) becomes a package boundary a test can check, and SQ1's future driver has a home beside SQ2's. The old plan's D1 table is unchanged except its last row: `run_saasbo`'s rewrite is now the core loop (bo.py) plus the bookkeeping (experiments/).

**D2 — Diagnostics.** They stay part of `fit`'s contract: `fit(…, thresholds=DiagThresholds())` and `FittedGP.status/attempts` are unchanged, the attempt loop (one refit, doubled warm-up, `fold_in(key, 1)`) stays in `gp.fit`, and the *verdict* (`diagnose` and its two dataclasses and site tables) moves to `sagp/diagnostics.py`, which `gp` imports. Rejected: "fit returns raw draws, the experiment layer diagnoses" — the refit needs the verdict between attempts, so that would move the gate policy into the experiment layer, the very entanglement the brief forbids; and two callers (`identify`, the loop) need the status, so by the deletion test it belongs behind `fit`. Effect on the bit-for-bit tests: none — they configure thresholds through `fit`'s unchanged signature. Effect on `fit`'s signature: none. Effect on the pending gate decision: whichever way it goes, it touches `diagnostics.py` (thresholds, sites) and the ten lines of `gp.fit`'s attempt loop, nowhere else.

**D3 — Core public API after the split** (unchanged signatures marked ✓).

```python
# sagp/gp.py
GL_NODES, GL_WEIGHTS   # gp's own leggauss(64) on [0,1]; pinned array_equal to synthobj.kernel's (Task 4)
ACTIVE_EPS = 0.02      # gp's own; pinned equal to synthobj.objective.ACTIVE_EPS (Task 4)
def kbar_all(X, Z, ell_vec, normalize) -> Array          # was _kbar_all; the readouts need it
class FittedGP:  posterior(X_test) ✓; alphas() ✓; params(s); param_sites(); noises(); columns(X)   # last four were _-prefixed
def fit(X, y, key, cell, *, alpha=None, fixed_noise=None, nuts=NUTSConfig(), thresholds=DiagThresholds(), ell_prior=ELL_PRIOR) -> FittedGP ✓
def standardize(y) -> (z, mean, std) ✓                    # stays (R11): fit's precondition, beside fit

# sagp/diagnostics.py
class DiagThresholds ✓; class Diagnostics ✓
def diagnose(flat_samples, extra, thresholds, wall_s) -> Diagnostics   # was gp._diagnose

# sagp/bo.py
def iteration_rngs(seed, t) -> IterRNG ✓; def initial_design(D, n_init, seed) ✓
def log_ei(x, y_target, gp, xi=0.0) ✓; ACQUISITIONS ✓
def optimize_ei(gp, y_target, sobol_seed, jitter_rng, xi=0.0, num_restarts_ei=5, num_init=5000, acq=log_ei) -> (x, value) ✓
def propose_ei(fitted, y_target, rngs, t, *, acq=log_ei, num_init=5000, num_restarts_ei=5) -> (np.ndarray, float)
    """optimize_ei driven by iteration t's rngs; the loop's default proposer."""
class BOState(NamedTuple): X: np.ndarray; y: np.ndarray; f: np.ndarray
class Iteration(NamedTuple):
    """What one iteration produced: t, x, y, f (this point), state (BOState after appending),
    fitted (FittedGP | None), acq_value, fit_wall_s, acq_wall_s, fit_calls, status, reason, y_mean, y_std."""
def run_bo(objective, surrogate, seed, *, T=200, n_init=20, propose=propose_ei, noiseless=False,
           state=None, on_iteration=None, log=print) -> BOState
    """The reference loop, generalized. surrogate(X, z, key) -> FittedGP, or None for a model-free
    method (fit_calls 0, fit_wall_s NaN, as today); propose(fitted, y_target, rngs, t) -> (x, acq_value);
    state resumes from a BOState (the next t is len(state.X)); on_iteration(Iteration) runs after each
    append; log(str) receives the exception-policy messages. Exception policy, seeding and the
    initial design are exactly today's _fit_with_retry/_iteration/run_bo."""

# sagp/references.py
def fit_map(X, y, *, active=None, maxiter=500) -> FittedGP ✓
def propose_sobol(fitted, y_target, rngs, t, *, sequence) -> (sequence[t], nan)

# sagp/readouts.py — shares_from_amplitudes ✓, component_means ✓, readouts ✓, manipulation_checks ✓

# experiments/identify.py — identify(objective, cell, n, seed, *, sobol_n=2048, **fit_kwargs) -> dict ✓
# experiments/runlog.py  — RunConfig ✓, config_hash ✓, run_dir_for ✓, RunLogger ✓ + record(it: Iteration)
# experiments/run_bo.py  — METHODS ✓, surrogate_for(cfg, labels), propose_for(cfg), run(objective, method, seed, *, T, n_init, out_dir, resume=True, **overrides) -> Path (today's run_bo, renamed), resolve_config ✓, main ✓
```

The rule: `experiments/*` imports only un-underscored names from `sagp.*`; a name used across two `sagp` modules is public (hence the five renames above). Enforced by `tests/test_layering.py` (Task 5).

**D4 — Dependency direction.** `sagp.gp` imports nothing from `sagp` (it *defines* the grid and `ACTIVE_EPS`; `synthobj` disappears from it in Task 4, pinned by equality tests, both values being two deterministic numpy lines). `diagnostics` imports nothing from `sagp`. `bo` imports `gp.standardize`, `gp.FittedGP` and `synthobj.families.noise_rng` — the noise stream **stays** imported: it is the objectives' seeding convention (`SeedSequence([seed, _NOISE_SALT, run])`), the loop must draw from that very stream for paired runs, and a copy in `bo.py` would be a second definition that can drift. `references` and `readouts` import `gp` only. `experiments/*` import `sagp.*` and `synthobj`; nothing in `sagp/` imports `experiments`. `sagp/__init__` lazy-exports from `sagp` modules only.

**D5 — How the move is done.** One task per destination module; each task is a *move commit* (cut/paste of a whole `# --- … ---` section, plus the import lines that make it run and the test-import updates; `git show --stat` shows one new file and shrinking sources; diff of the moved body against the source is empty) followed, where needed, by a separate *edit commit* (renames, signature changes). The order keeps every commit green and the dependency direction intact at every step. `scripts/pilot_sagp.py` needs no change.

**D6 — Test relocation map** (§6 below) and verification: part A + part B + `BIT` after every commit; slow reproduction after Task 6b and at the end; the golden diff after Task 6b.

---

## 2. Tasks

### Task 0: Save the brief; start the ledger

**Files:** Create `docs/superpowers/specs/2026-09-09-sagp-core-split-brief.md` (the brief, verbatim), `.superpowers/sdd/2026-09-09-sagp-core-split/progress.md` (git-ignored; header: plan path, base 0fd2c19, env, the branch-check rule).

- [x] Write both files; `git add` the brief and this plan; commit `docs(sagp): core-split brief and plan`.

### Task 1: `sagp/diagnostics.py` (move, then rename)

**Files:** Create `sagp/diagnostics.py`; modify `sagp/gp.py` (cut lines 529–572 `DiagThresholds`/`Diagnostics` and 861–966 `_POSITIVE_SAMPLED_SITES` … `_diagnose`); create `tests/test_sagp_diagnostics.py`; modify `tests/test_sagp_inference.py`, `tests/test_bo_loop.py`.

- [ ] **Move commit.** Paste the two blocks verbatim into `diagnostics.py` (imports: `numpy`, `jax.numpy`, `dataclass`, `numpyro.diagnostics.summary`). In `gp.py`: `from sagp.diagnostics import DiagThresholds, Diagnostics, _diagnose`. Move `test_diagnostics_fields`, `test_per_group_diagnostics_split_the_pooled_ones`, `test_positive_sampled_sites_are_exactly_the_cells_positive_sampled_sites` (and their helpers `_data`, `_stub_diagnose` copies as needed) to `tests/test_sagp_diagnostics.py`, importing from `sagp.diagnostics`. Run part A, part B, `BIT`. Commit `refactor(sagp): move the diagnostic verdict to sagp/diagnostics.py (pure move)`.
- [ ] **Edit commit.** Rename `_diagnose` → `diagnose` in `diagnostics.py`, `gp.py` (import + the two call sites in `fit`), and the monkeypatch targets `sagp.gp._diagnose` → `sagp.gp.diagnose` in `tests/test_sagp_inference.py::test_refit_and_excluded_paths` and `tests/test_bo_loop.py::test_a_refit_row_carries_both_attempts_diagnostics`. Run part A, part B, `BIT`. Commit `refactor(sagp): diagnose is public`.

### Task 2: `sagp/references.py` (move)

**Files:** Create `sagp/references.py`; modify `sagp/gp.py` (cut the whole `# --- MAP references ---` section, lines 1051–1233), `sagp/bo.py` (`fit_map` import), `sagp/__init__.py` docstring untouched; `git mv tests/test_sagp_map.py tests/test_sagp_references.py`.

- [ ] Paste verbatim; imports in `references.py`: `math`, `numpy`, `jax`, `jax.numpy`, `numpyro.distributions`, `jit`, `cho_factor`, `cho_solve`, `scipy.optimize.minimize`, and `from sagp.gp import FittedGP, kernel_product_lengthscale`. `bo.py`: `from sagp.references import fit_map`. In the renamed test: `import sagp.references as references` and `gp.<name>` → `references.<name>` for every moved name (`fit_map`, `_dsp_start`, `_dsp_neg_log_joint`, `_ell_prior_loc`, the five `_*_FLOOR/_PRIOR_*` constants, the `minimize` monkeypatch target). Run part A, part B, `BIT`. Commit `refactor(sagp): move the MAP references to sagp/references.py (pure move)`.

### Task 3: `sagp/readouts.py` (move, then public names)

**Files:** Create `sagp/readouts.py`; modify `sagp/gp.py` (cut `# --- readouts ---`, lines 1236–1567), `sagp/bo.py` (`readouts` import), `sagp/__init__.py` (`readouts` now from `sagp.readouts`), `tests/test_sagp_readouts.py`, `tests/test_sagp_cells.py` (its `component_means` test), `experiments`-free at this point.

- [ ] **Move commit.** Paste verbatim; `readouts.py` imports `from sagp.gp import ACTIVE_EPS, CELLS, GL_NODES, GL_WEIGHTS, RHO_EPS, FittedGP, _kbar_all` plus `numpy`, `jax.numpy`, `jit`, `partial`, `qmc`, `spearmanr`. Tests: `import sagp.readouts as readouts`; `gp.readouts/_sobol_*/manipulation_checks/component_means` → `readouts.<name>`; the slow tests keep `gp.fit`/`gp.standardize`. Run part A, part B, `BIT`. Commit `refactor(sagp): move the readouts to sagp/readouts.py (pure move)`.
- [ ] **Edit commit.** Renames, mechanical (`git grep -n` each old name to confirm every site): `_kbar_all` → `kbar_all` (gp, readouts, tests); `FittedGP._params/_param_sites/_noises/_columns` → `params/param_sites/noises/columns` (gp, readouts, `tests/test_sagp_cells.py:343`, the docstring at `tests/test_sagp_inference.py:424`). Run part A, part B, `BIT`. Commit `refactor(sagp): public names for what the readouts read of a fit`.

### Task 4: `gp.py` owns its grid and cutoff (no `synthobj` import)

**Files:** Modify `sagp/gp.py` lines 44–47, 57–58; `tests/test_sagp_kernels.py`.

- [ ] Write the failing test in `test_sagp_kernels.py`:
  ```python
  def test_gp_constants_are_bit_identical_to_synthobjs():
      from synthobj import kernel; from synthobj.objective import ACTIVE_EPS as EPS
      assert np.array_equal(np.asarray(GL_NODES), kernel.GL_NODES)
      assert np.array_equal(np.asarray(GL_WEIGHTS), kernel.GL_WEIGHTS)
      assert ACTIVE_EPS == EPS
      tree = ast.parse(Path(sagp.gp.__file__).read_text())
      imported = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | {
          a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
      assert not any(name.startswith("synthobj") for name in imported)
  ```
  Run it: it fails on the last assertion only. Replace the `synthobj` imports with `from numpy.polynomial.legendre import leggauss`, `_NODES, _WEIGHTS = leggauss(64)`, `GL_NODES, GL_WEIGHTS = jnp.asarray(0.5 * (_NODES + 1)), jnp.asarray(0.5 * _WEIGHTS)`, `ACTIVE_EPS: float = 0.02` (comment: the study's active cutoff, restated so the surrogate never imports the objectives; pinned to `synthobj`'s by this test). This task runs **after** Task 5, which already took `noise_rng` away with `identify`. Run part A, part B, `BIT`. Commit `refactor(sagp): gp.py owns its quadrature grid and ACTIVE_EPS`.

### Task 5: `experiments/` package, `identify`, layering test

**Files:** Create `experiments/__init__.py` (docstring only), `experiments/identify.py`, `tests/test_layering.py`; modify `sagp/gp.py` (cut `identify`, lines 1591–1674, and the `noise_rng`/`time`/`warnings`/`asdict` imports it alone used), `sagp/__init__.py` (drop `identify`); `git mv tests/test_sagp_identify.py tests/test_identify.py`; move `test_standardize_zero_mean_unit_std_and_sign_flip` into `tests/test_sagp_inference.py`.

- [ ] **Move commit.** `experiments/identify.py` imports `from sagp.gp import CELLS, fit, standardize`, `from sagp.readouts import manipulation_checks, readouts`, `from synthobj.families import noise_rng`. `tests/test_identify.py` imports `from experiments.identify import identify`; its guard becomes `python -c "import sys, sagp.gp, experiments.identify; assert 'sagp.bo' not in sys.modules"`. Run part A, part B, `BIT`. Commit `refactor(experiments): move identify() to experiments/identify.py (pure move)`.
- [ ] **Layering test** (`tests/test_layering.py`, AST over `sagp/*.py` and `experiments/*.py`): (a) no module under `sagp/` imports `experiments`; (b) every `from sagp.X import name` and `sagp.X.name` attribute in `experiments/*.py` has no leading underscore; (c) `sagp/gp.py` and `sagp/diagnostics.py` import no `sagp.bo`, `sagp.references`, `sagp.readouts`. Run part A. Commit `test(sagp): layering rules`.

### Task 6a: `experiments/runlog.py` + `experiments/run_bo.py` (move)

**Files:** Create both; modify `sagp/bo.py` (cut everything from `# --- configuration ---` (line 282) to the end), `sagp/__init__.py` (drop `run_bo` from `_BO_API`/`__all__` until Task 6b restores it: `sagp` must never import `experiments`), `tests/test_bo_loop.py` (its 16 run-directory tests, everything from `test_two_runs_at_the_same_seed_are_bit_identical` to the slow `test_oracle_beats_sobol_on_aligned3` except `test_initial_design_is_the_first_rows_of_the_runs_sobol_sequence`, move to a new `tests/test_run_bo.py` with their helpers `_objective`, `_read_rows`, `_checkpoint`, `_diagnostics`, `_LOOP_KW`, `_T`, `_N_INIT`), `tests/test_bo_config.py`, `tests/test_bo_cli.py`, `tests/test_bo_reference.py`, `README.md` (one-liner paths only, `python -m experiments.run_bo`).

- [ ] `runlog.py` receives `# --- configuration ---`, `# --- provenance ---`, `# --- the run directory ---` verbatim (`_REPO_ROOT = Path(__file__).resolve().parent.parent` still resolves to the repo root). `run_bo.py` receives `METHODS`, `# --- one iteration ---`, `# --- the loop ---`, `# --- the command line ---` verbatim, importing `from sagp.bo import ACQUISITIONS, IterRNG, initial_design, iteration_rngs, optimize_ei`, `from sagp.gp import CELLS, FittedGP, standardize, fit`, `from sagp.references import fit_map`, `from sagp.readouts import readouts`, `from experiments.runlog import …`. Tests: `from experiments.run_bo import run_bo, METHODS, resolve_config, main` / `from experiments import run_bo as run_bo_module` (for `_build_parser`, and the `monkeypatch.setattr(run_bo_module, "fit", …)` targets that were `bo.fit`); `test_bo_cli._run` uses `"-m", "experiments.run_bo"`. Run part A, part B, `BIT`. Commit `refactor(experiments): move the run directory, dispatch and CLI out of sagp/bo.py (pure move)`.

### Task 6b: the core loop (edit, TDD)

**Files:** Modify `sagp/bo.py` (add `propose_ei`, `BOState`, `Iteration`, `run_bo`; move `_fit_with_retry`, `_observe` back from `experiments/run_bo.py`), `sagp/references.py` (`propose_sobol`), `experiments/run_bo.py` (`_fit_for` → `surrogate_for`, `_propose` → `propose_for`, `run_bo` → `run`, delete `_iteration`), `experiments/runlog.py` (`RunLogger.record`, `_row`, `_iteration_line`, `statuses`), `sagp/__init__.py` (`run_bo` export back), tests as in §6.

- [ ] **Golden outputs (before touching code).** In the scratchpad run, via `experiments.run_bo.run`: `dsp_map` T=10 n_init=5 seed=1; `sobol` T=12 seed=3; `product/lengthscale` T=7 n_init=5 seed=1 `nuts=NUTSConfig(32,32,4)`; all with `num_init_candidates=256, num_restarts_ei=1, sobol_every=5` on `make_family("aligned3", 0, D=5)`. Keep the three run directories.
- [ ] **Failing test** in `tests/test_bo_loop.py`:
  ```python
  def test_run_bo_core_is_a_pure_function_of_seed_state_and_adapters():
      obj = _objective(); seen = []
      def surrogate(X, z, key):  # a valid product/lengthscale posterior, no inference
          S = {"kernel_var": jnp.ones(2), "kernel_inv_length_sq": jnp.ones((2, 5)), "kernel_noise": jnp.full(2, 0.01)}
          return FittedGP(("product", "lengthscale"), X, z, S, None, None, "ok", "", ())
      propose = partial(propose_ei, num_init=64, num_restarts_ei=1)
      a = run_bo(obj, surrogate, 0, T=8, n_init=5, propose=propose, on_iteration=seen.append)
      assert [it.t for it in seen] == [5, 6, 7] and a.X.shape == (8, 5) and seen[-1].state.X is a.X
      assert all(it.fit_calls == 1 and it.status == "ok" for it in seen)
      b = run_bo(obj, surrogate, 0, T=8, n_init=5, propose=propose)
      assert np.array_equal(a.X, b.X) and np.array_equal(a.y, b.y)
      c = run_bo(obj, surrogate, 0, T=8, n_init=5, propose=propose, state=BOState(a.X[:6], a.y[:6], a.f[:6]))
      assert np.array_equal(c.X, a.X)
      s = run_bo(obj, None, 3, T=8, n_init=5, propose=partial(propose_sobol, sequence=initial_design(5, 8, 3)), on_iteration=seen.append)
      assert np.array_equal(s.X, initial_design(5, 8, 3)) and seen[-1].fit_calls == 0 and np.isnan(seen[-1].fit_wall_s)
  ```
  Run: fails with `ImportError` (`propose_ei`, `BOState`, `propose_sobol`).
- [ ] **Implement** `sagp/bo.py` per D3 (bodies are today's `_fit_with_retry`, `_propose`, `_observe`, `_iteration`, `run_bo` with `cfg`/`logger` replaced by the adapters; `_fit_with_retry(t, surrogate, X, z, rngs, log)` returns `(None, 0, nan, None)` when `surrogate is None`; `x_next = np.asarray(x_next, dtype=float)`; `on_iteration` is called after the append). `references.propose_sobol` = today's `sobol_seq[t]` branch. `experiments/run_bo.py::run` = today's `run_bo` minus the loop body: it builds `cfg`, `logger`, resolves resume (`state = BOState(X, y, f)`, `logger.statuses = statuses`), then `sagp.bo.run_bo(objective, surrogate_for(cfg, objective.labels), cfg.seed, T=cfg.T, n_init=cfg.n_init, propose=propose_for(cfg), noiseless=cfg.noiseless, state=state, on_iteration=logger.record, log=logger.log)`. `surrogate_for`: `None` for sobol; `lambda X, z, key: fit_map(X, z)`; `… fit_map(X, z, active=np.asarray(labels.S))`; `lambda X, z, key: fit(X, z, key, cell=CELLS[…], alpha=cfg.alpha, fixed_noise=cfg.fixed_noise, nuts=cfg.nuts, thresholds=cfg.thresholds, ell_prior=cfg.ell_prior)` (a lambda, so `monkeypatch.setattr(run_bo_module, "fit", …)` still bites). `propose_for`: `partial(propose_sobol, sequence=initial_design(cfg.D, cfg.T, cfg.seed))` or `partial(propose_ei, acq=ACQUISITIONS[cfg.acq], num_init=cfg.num_init_candidates, num_restarts_ei=cfg.num_restarts_ei)`. `RunLogger.record(it)`: `compute_sobol = it.fitted is not None and (it.t % cfg.sobol_every == 0 or it.t == cfg.T - 1)`; readout; `statuses.append`; row via `_row(it, compute_sobol)` (today's dict, field for field); then today's write order: checkpoint, coords, samples, row, `log(_iteration_line(row, it.fitted))`.
- [ ] Run the new test (passes), part A, part B, `BIT`; then the slow reproduction `python -m pytest -q tests/test_bo_reference.py` (background, ≈ 2–5 min); then the **golden diff**: rerun the three runs into a second scratch directory and compare `iterations.csv` (all columns except `fit_wall_s`, `acq_wall_s`), `coords.csv`, `checkpoint.npz` (`X`, `y`, `f`, `statuses`, `config_hash`), every `samples/*.npz` array and non-timing scalar (`status`, `nuts_attempts`, `a0_*` except `a0_wall_s`), `manifest.json` less `created`/`git`/`env`/`resumed`. All byte-equal. Commit `refactor(sagp): run_bo is the loop; the run directory observes it`.

### Task 7: docs

**Files:** `README.md` (§ "Sparse additive GP cells": the module map, `python -m experiments.run_bo`, both SLURM snippets), module docstrings of `sagp/__init__.py`, `sagp/gp.py`, `sagp/bo.py`, the three new `sagp` modules and `experiments/*` (what each owns and the layering rule), `docs/superpowers/plans/2026-09-07-sagp-model-and-bo.md` (one line under §1: "superseded by 2026-09-09-sagp-core-split.md").

- [ ] Edit; `python -m experiments.run_bo --help`, `python scripts/pilot_sagp.py --help`; part A, part B, `BIT`; all slow tests one file at a time in the background. Commit `docs(sagp): module map after the core split`.

### Task 8: trim the plan-history prose in the core modules (docs-only; approved 2026-09-09)

**Files:** `sagp/gp.py`, `sagp/diagnostics.py`, `sagp/bo.py`, `sagp/references.py`, `sagp/readouts.py` — docstrings and comments only; `git diff --stat` must show no change to any line that is code (verify with `git diff -w` filtered through `grep -v '^[-+]\s*#\|^[-+]\s*"""\|^[-+]\s*$'` reading empty apart from docstring bodies).

- [ ] Rewrite each docstring and comment to what a reader of the code needs: the invariant, the shape, the reason a line is the way it is. Drop task numbers, ruling numbers, "the plan requires", "Task 7's", pilot dates and cost tables; keep every reference to the vendored code, every numerical claim that a test pins, and every warning about a trap (the `where`-NaN clamp, the `get_chunks` NameError, the ddof-0 std). Targets: `gp.py` ≤ 600, `bo.py` ≤ 300, `diagnostics.py` ≤ 150, `references.py` ≤ 160, `readouts.py` ≤ 260. Run part A, part B, `BIT` (a docstring cannot change them, which is the check). Commit `docs(sagp): core docstrings say what the code needs, not how it was built`.

---

## 3. Test relocation map

| today | after | change |
|---|---|---|
| `tests/test_sagp_env.py`, `test_sagp_kernels.py`, `test_sagp_cells.py` | same files | kernels: + `test_gp_constants_are_bit_identical_to_synthobjs`; cells: `_params` → `params` |
| `tests/test_sagp_inference.py` | same, minus 3 diagnostics tests, plus `test_standardize_*` | `sagp.gp._diagnose` → `sagp.gp.diagnose` |
| — | `tests/test_sagp_diagnostics.py` (new) | the 3 moved tests, importing `sagp.diagnostics` |
| `tests/test_sagp_map.py` | `tests/test_sagp_references.py` (git mv) | `gp.` → `references.` |
| `tests/test_sagp_readouts.py` | same file | `gp.readouts`/`gp._sobol_*` → `readouts.*`; `_kbar_all` → `kbar_all` |
| `tests/test_sagp_identify.py` | `tests/test_identify.py` (git mv) | imports `experiments.identify`; guard extended; `standardize` test moves out |
| — | `tests/test_layering.py` (new) | the three layering rules |
| `tests/test_bo_loop.py` | keeps `test_log_h_*`, `test_log_ei_*`, `test_optimize_ei_*`, `test_initial_design_*`, + the new core test | the loop/run-directory tests move → |
| — | `tests/test_run_bo.py` (new) | the 16 run-directory tests from `test_bo_loop.py`, importing `experiments.run_bo.run`, patching `run_bo_module.fit` and `sagp.gp.diagnose` |
| `tests/test_bo_config.py`, `test_bo_cli.py`, `test_bo_reference.py` | same files | `run_bo` → `run`; `bo._build_parser` → `run_bo_module._build_parser`; `-m experiments.run_bo` |
| slow: 10 tests | 10 tests, same files as their fast neighbours | none |

## 4. Effort

| task | hours |
|---|---|
| 0 brief, ledger | 0.5 |
| 1 diagnostics (move + rename) | 1.5 |
| 2 references | 1.5 |
| 3 readouts (move + renames) | 3 |
| 4 grid/cutoff decoupling | 1 |
| 5 experiments/identify + layering test | 2 |
| 6a run directory / CLI move | 2 |
| 6b core loop + golden diff + reproduction | 6 |
| 7 docs, slow suite | 1.5 |
| 8 docstring trim (docs-only) | 3 |
| reviews and ledger rulings (one review per task) | 3 |
| **total** | **≈ 25 h** |

Execution order: 0, 1, 2, 3, 5, 4, 6a, 6b, 7, 8 (Task 4's "no synthobj" assertion needs `identify` gone first).

## 5. Open questions — all answered (Quan, 2026-09-09)

1. **Prose.** About 45 % of the core files is docstring narrating rulings and task numbers. Answer: **yes**, trim it in a final docs-only commit → Task 8.
2. **`experiments/` at the repo root** versus `sagp/experiments/`. Answer: **root** (D1 as written).
3. **`FittedGP` knows the reference names** (`_MAP_REFERENCES`, `_is_map_reference`). Answer: **fine**, left as is; deferred minor.
4. **`log=print`** callable versus Python `logging` for the loop's exception-policy messages. Answer: **callable** (D3 as written): explicit in the signature, testable through it, no process-wide handler state, same shape as the reference's own `print`.
