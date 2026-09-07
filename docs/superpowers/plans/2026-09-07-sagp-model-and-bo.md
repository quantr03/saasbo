# sagp: Model and BO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two files, `sagp/gp.py` and `sagp/bo.py`, that run the thesis's 2 × 2 (additive/product × amplitude/lengthscale sparsity) on the SAASBO reference code path, with seeding, diagnostics, readouts, an offline identification runner, a checkpointed BO loop, three references, and a SLURM-ready CLI.

**Architecture:** `gp.py` owns kernels, the four NumPyro models, NUTS, prediction, readouts and `identify()`; `bo.py` owns the loop, acquisition, references, logging and the CLI. They meet at one interface: `fit(X, y, key, cell) -> FittedGP` and `FittedGP.posterior(X_test) -> (mean, var)` of shape `(S, n_test)` per retained sample, exactly the reference's shapes. Functions the reference already gets right are imported from the vendored `saasgp.py`/`saasbo.py`/`util.py`; the rest is copied with the smallest possible diff.

**Tech Stack:** Python 3.11, JAX + jaxlib 0.10.2, NumPyro 0.21.0, SciPy 1.17.1, NumPy 2.4.6, pytest 9.1.1, all in `/opt/anaconda3/envs/saasbo`; float64 everywhere. No BoTorch/GPyTorch/torch in the loop. `synthobj` supplies objectives.

**Spec:** The brief given in conversation on 2026-09-07 (to be saved verbatim as `docs/superpowers/specs/2026-09-07-sagp-brief.md` in Task 0), `Research Context/research direction/saasbo-reference-implementation.md`, `Design-brief.md` §2–4, and `Kernel normalization check.py`. This plan itself is copied to `docs/superpowers/plans/2026-09-07-sagp-model-and-bo.md` on approval.

## Context

The thesis needs the four surrogate cells to differ in nothing but the kernel/prior block, so any regret or identification difference is attributable to parameterization or structure. The only way to guarantee "everything else identical" is to make everything else literally the reference implementation's code path, with seeding added so runs are reproducible and resumable. `synthobj` is finished; this is the next module.

**Reference summary vs vendored code.** The summary file was empty when planning began and was filled in mid-session. Checked line by line against the vendored `saasgp.py`/`saasbo.py` (upstream commit `4af4a8e`), every setting it states is correct: priors, 512/256/16, tree depth 6 in the driver (class default 7), one chain, no warm start, `PRNGKey(0)` at every fit, summary computed always but printed only when verbose, EI averaged over samples with std floored at 1e-6, unseeded candidate Sobol and jitter, top-5 → L-BFGS-B `maxfun=100`, `Y.std()` with ddof 0, uniform random point on exception, abort after 3, demo budget 256/256/32 at α = 0.01 (verified in `HEAD:saasbo_demo.py`). Four facts the summary omits and the plan must handle:

1. `optimize_ei` returns only `x_best`; the EI value at the chosen point (needed for logging) is discarded → copy and return both.
2. `saasgp.matern_kernel` calls `jnp.clip(dsq, a_min=1e-12)`; the JAX changelog records the `a_min` keyword as removed in 0.10.0 (deprecated since 0.4.27), so the vendored file cannot run on the pinned stack without a one-line compatibility change (`jnp.clip(dsq, 1e-12)`, numerically identical). Task 0 confirms this on the installed stack before patching.
3. `util.get_chunks` uses `np.arange` without importing numpy: a `NameError` whenever the sample count is not a multiple of `chunk_size`. Never hit at 16/8; would be hit by a 1-sample MAP reference → use `chunk_size = min(8, S)` and leave `util.py` alone.
4. With fixed noise, `SAASGP.posterior` hard-codes `1e-6` for the predictive noise while `compute_choleskys` uses `observation_variance`; identical only when the fixed value is 1e-6. Our fixed-noise path uses the configured value in both places (equal to the reference at 1e-6, which is the value the reproduction test uses).

**Design-brief.md is out of date on three points** and this brief supersedes it: budget (512/256/16, not 128+128/8), no warm start, the reference optimizer (not BoTorch's). Its LogEI choice is back in force (decided 2026-09-07 12:00). Flagged for a doc update; nothing in this plan follows the old text.

## Global Constraints

- Python 3.11 in `/opt/anaconda3/envs/saasbo` (not the repo `.venv`); pins: `jax==0.10.2 jaxlib==0.10.2 numpyro==0.21.0 scipy==1.17.1 numpy==2.4.6`. NumPyro 0.21 requires jax ≥ 0.7 and Python ≥ 3.11 (PyPI metadata); no upper bound, so 0.10.2 is admissible and must be verified in Task 0.
- `jax_enable_x64` set at `sagp` import time; CPU platform; `numpyro.set_host_device_count(1)`.
- API audit of the vendored code on the pinned stack: the only breaking change is `jnp.clip(..., a_min=)` (removed in JAX 0.10.0). `numpyro.util.enable_x64`, `MCMC`/`NUTS` arguments, `get_samples(group_by_chain=)`, `numpyro.diagnostics.summary`, `dist.HalfCauchy/LogNormal/MultivariateNormal`, `jax.scipy.linalg.cho_factor/cho_solve/solve_triangular`, `lax.top_k`, `scipy.stats.qmc.Sobol(seed=)` and `fmin_l_bfgs_b` are all unchanged in 0.21.0 / 0.10.2 / 1.17.1. `jax.random.PRNGKey` (uint32 keys) is kept, as the reference uses it.
- NUTS budget in every cell, offline and in-loop: `num_warmup=512, num_samples=256, thinning=16, max_tree_depth=6, num_chains=1`, default adaptation, `init_to_uniform`, a fresh chain per fit. Diagnostics on the un-thinned 256 draws.
- Acquisition (decided by Quan, 2026-09-07 12:00): **LogEI**. Per retained sample, log EI_s = log σ_s + log_h(z_s) with z_s = (y_target − μ_s)/σ_s, σ_s floored at 1e-6 as in the reference, and log_h the numerically stable form of log(φ(z) + zΦ(z)) from Ament et al. 2023 (via `jax.scipy.special.log_ndtr`); the samples are combined by log-mean-exp, i.e. the exact log of the reference's sample-averaged EI. Identical in every cell and reference. `--acq ei` keeps the reference's plain EI solely for the reproduction test.
- Maximization convention (`synthobj`): `f_star` is the maximum. Internally the loop negates: the GP sees `z = -(y - mean y)/std y`, so the reference minimization code runs unchanged; all logs and regret are in `y` units.
- Vendored reference files stay byte-identical except the one `jnp.clip` compatibility line (decision 1). Never `git add -A`; the tree carries the user's untracked notes.
- Every task ends with `/opt/anaconda3/envs/saasbo/bin/python -m pytest -q` green (slow tests excluded by default via `-m "not slow"`); commit per task.

---

## 1. Module layout

```
sagp/
  __init__.py     enable_x64, platform cpu; lazy exports fit, FittedGP, identify, run_bo
  gp.py           kernels, CELLS, models, NUTS fit + diagnostics, FittedGP, fit_map, readouts, identify
  bo.py           iteration RNGs, optimize_ei (copied), run_bo, references, logging/checkpoint, CLI
saasgp.py, saasbo.py, util.py   vendored reference; imported, not copied, wherever unchanged
tests/
  test_sagp_kernels.py  test_sagp_cells.py  test_sagp_inference.py  test_sagp_readouts.py
  test_sagp_identify.py test_bo_loop.py  test_bo_reference.py  test_bo_config.py  test_bo_cli.py
docs/superpowers/specs/2026-09-07-sagp-brief.md   (brief, verbatim)
requirements-sagp.txt   the pins above (the lock written per run is generated, not this file)
```

`pytest.ini` already sets `pythonpath = .` and the `slow` marker. `setup.py`'s `find_packages()` picks up `sagp/`; the vendored root modules are importable from the repo root, so runs and the SLURM script `cd` to the repo root.

## 2. Public interface (`gp.py` ↔ `bo.py`)

```python
# ---- gp.py -------------------------------------------------------------------
CellKey = tuple[str, str]        # (structure in {"additive","product"}, prior in {"amplitude","lengthscale"})
CELLS: dict[CellKey, Cell]       # the registry; Cell is a frozen dataclass (section 3)

@dataclass(frozen=True)
class NUTSConfig:
    """The reference's sampler settings; one instance shared by identify() and run_bo()."""
    num_warmup: int = 512; num_samples: int = 256; thinning: int = 16
    max_tree_depth: int = 6; num_chains: int = 1
    # retained samples S = num_samples // thinning = 16

@dataclass(frozen=True)
class DiagThresholds:
    """Pass rule for one fit, applied to the log of every positive site (decision 4)."""
    r_hat_max: float = 1.1; n_eff_min: float = 16.0; max_divergences: int = 5

@dataclass(frozen=True)
class Diagnostics:
    """One attempt's diagnostics on the un-thinned chain (numpyro.diagnostics.summary +
    extra_fields): r_hat_max, r_hat_median, frac_r_hat_below_1_05, n_eff_min, divergences,
    num_steps_mean, wall_s, passed, reason."""

class FittedGP:
    """Posterior of one cell (or the MAP reference) on standardized, negated targets.

    Attributes: cell (CellKey or "dsp_map"/"oracle_S"), X_train (n, D) and Y_train (n,) as
    jnp float64 -- kept under the reference's names so saasbo.optimize_ei's incumbent lookup
    works unchanged; samples: dict site -> (S, ...) retained constrained values (S = 16, or 1
    for MAP); fixed_noise: float | None; active: indices used by the oracle, else None;
    status in {"ok","refit","excluded"}; status_reason: str; attempts: tuple[Diagnostics, ...]
    (one per NUTS attempt; len(attempts) is the row's nuts_attempts).
    """
    def posterior(self, X_test) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Per retained sample: posterior mean and *noisy* predictive variance at X_test,
        (S, n_test) each -- the reference's compute_choleskys + predict, with the kernel and its
        diagonal generalized to the cell. Cholesky factors are cached on first call."""

def fit(X, y, key, cell, *, alpha=None, fixed_noise=None, nuts=NUTSConfig(),
        thresholds=DiagThresholds(), ell_prior=ELL_PRIOR) -> FittedGP:
    """NUTS fit of `cell` to (X in [0,1]^D, y already standardized and negated by the caller).

    `key` is a jax PRNGKey used unchanged by the first attempt (so a caller can reproduce the
    reference bit-for-bit); the refit uses fold_in(key, 1). alpha defaults to the cell's value
    (0.1 on the rho scale, ALPHA_AMPLITUDE on the a^2 scale). fixed_noise=None learns
    kernel_noise ~ LogNormal(0,10); a float fixes it (decision 5). Runs the reference
    run_inference, computes diagnostics, refits once with doubled warm-up on failure, and
    returns status ok/refit/excluded -- never raises on a diagnostic failure, only on an
    exception from JAX/NumPyro (bo.py's failure policy handles those)."""

def fit_map(X, y, *, active=None, maxiter=500) -> FittedGP:
    """DSP-by-MAP reference: ARD Matern-5/2 (BoTorch's default is RBF; Matern-5/2 is the
    thesis's held-fixed kernel), sigma_f^2 = 1, Hvarfner's prior ell_i ~ LogNormal(sqrt2 +
    log(D_eff)/2, sqrt3) with BoTorch's floors ell_i = 0.025 + exp(u_i), noise = 1e-4 + exp(u_n),
    noise ~ LogNormal(-4, 1); objective = log marginal likelihood + log priors on the constrained
    values, no Jacobian (as BoTorch's MAP); scipy L-BFGS-B on jax value_and_grad, started at the
    prior modes (BoTorch's initial values); deterministic. active=None -> D_eff = D; active=S ->
    the oracle: X[:, S] only, D_eff = |S|. Returns a FittedGP with one 'sample' so posterior()
    has shape (1, n_test)."""

def readouts(fitted, *, eps=0.02, compute_sobol=True, sobol_n=2048, sobol_seed=0) -> dict:
    """Posterior readouts from stored samples, no refitting (decision 6, 7); compute_sobol=False
    skips sobol_hat (NaN) for the cheap in-loop rows. Keys:
    native (S, D) samples of a_i^2 or rho_i; native_median (D,); share_hat (S, D) noise-corrected
    shares (amplitude cells only); p_active (D,) = P(share_hat_i > eps) [amplitude] or
    P(rho_i > RHO_EPS) [lengthscale]; sobol_hat (D,) first-order Sobol index of the posterior
    mean (exact in three cells, QMC in product/lengthscale); active_neutral (D,) bool =
    sobol_hat > eps; active_native (D,) bool = p_active > 0.5; total_var_hat float."""

def component_means(fitted, x_grid) -> jnp.ndarray:
    """Additive cells only: per-component posterior means (S, D, len(x_grid)); sums to the
    posterior mean (test) and feeds the manipulation checks."""

def manipulation_checks(readout, labels) -> dict:
    """Spearman(native_median, labels.s), Spearman(native_median, labels.g),
    Spearman(sobol_hat, labels.s), amplitude-vs-realized pairs (share_hat median, labels.s) on S,
    and rank agreement between native_median and sobol_hat. scipy.stats.spearmanr only."""

def identify(objective, cell, n, seed, *, sobol_n=2048, **fit_kwargs) -> dict:
    """SQ1 runner: X = qmc.Sobol(D, scramble=True, seed=seed).random(n); y = objective.observe(X,
    noise_rng(seed, run=0)); standardize+negate; fit with key = PRNGKey(SeedSequence([seed, n,
    0x1D]).generate_state(1)[0]) (distinct per (seed, n), disjoint from the loop's keys);
    readouts; manipulation checks. Returns a
    flat record: scalars (family, seed, cell, n, alpha, status, reason, diagnostics fields,
    wall_s, y_mean, y_std) plus (D,) arrays (native_median, p_active, sobol_hat, labels s/g/active)
    and the retained samples dict. Must not import sagp.bo (tested)."""

# ---- bo.py -------------------------------------------------------------------
METHODS = [f"{s}/{p}" for (s, p) in CELLS] + ["sobol", "dsp_map", "oracle_S"]

@dataclass(frozen=True)
class RunConfig:
    """Every setting of a run, resolved once; serialized verbatim into manifest.json. Fields:
    family, seed, D, method, T, n_init, acq ("logei" default; "ei" = the reference's EI, kept for
    the reproduction test), alpha, fixed_noise, noiseless, nuts
    (NUTSConfig), thresholds (DiagThresholds), ell_prior, num_init_candidates=5000,
    num_restarts_ei=5, sobol_every=25, sobol_n=2048, out_dir. The constants the copied
    optimize_ei keeps hard-coded (maxfun=100, jitter sd 1e-3, xi=0) are written to the manifest
    under "reference_constants" but are not fields."""

def run_bo(objective, method, seed, *, T=200, n_init=20, out_dir, resume=True, **overrides) -> Path:
    """The reference loop generalized (section 4). Returns the run directory. Idempotent under
    resume: if <out_dir>/<run_id>/checkpoint.npz exists and its manifest matches the resolved
    config, continues from the last completed iteration bit-identically."""

def log_ei(x, y_target, gp, xi=0.0) -> jnp.ndarray:
    """LogEI with the reference ei's signature and conventions (minimization, std floored at 1e-6):
    log-mean-exp over samples of log(std) + log_h((y_target - xi - mu)/std). NaN -> -inf."""

def optimize_ei(gp, y_target, sobol_seed, jitter_rng, xi=0.0, num_restarts_ei=5,
                num_init=5000, acq=log_ei) -> tuple[np.ndarray, float]:
    """Copy of saasbo.optimize_ei with five changed lines: qmc.Sobol(dim, scramble=True,
    seed=sobol_seed); jitter from jitter_rng.standard_normal((1, dim)); `acq` in place of `ei` for
    candidate scoring and in place of `ei_grad` inside the L-BFGS-B objective; returns
    (x_best, acq_best). acq=saasbo.ei reproduces the reference exactly."""

def iteration_rngs(seed, t) -> IterRNG:
    """All randomness of iteration t from (seed, t): key = fold_in(PRNGKey(seed), t);
    sobol_seed = SeedSequence([seed, t, 0xCA]).generate_state(1)[0];
    jitter_rng = default_rng(SeedSequence([seed, t, 0x1A])); noise_rng = synthobj.families.noise_rng(seed, run=t);
    fallback_seed = SeedSequence([seed, t, 0xFA]).generate_state(1)[0]. Resume needs only (seed, t)."""

def initial_design(D, n_init, seed) -> np.ndarray:
    """qmc.Sobol(D, scramble=True, seed=seed).random(n_init) -- the reference's call, identical
    for every method at a given seed; the Sobol-search reference continues the same sequence."""

def main(argv=None) -> int:  # python -m sagp.bo
```

`bo.py` never reads `fitted.samples` or `fitted.cell` except to pass them to `gp.readouts` and to `np.savez`; `gp.py` never sees `T`, `n_init`, EI, or a budget.

## 3. Internal layout

### `gp.py`

```
# kernels (jax, jit, differentiable; inputs in [0,1]^D; all take (X, Z, params, noise, include_noise))
matern52_1d(r)                                 (1 + s + s^2/3) e^{-s}, s = sqrt5 r
GL_NODES, GL_WEIGHTS                           jnp copies of synthobj.kernel's 64-node grid
centered_matern52_1d(x, z, ell, normalize)     k~ = k - m(x) - m(z) + c; /v(ell) if normalize; x,z (n,), (m,)
v_of_ell(ell)                                  1 - sum_qq' w_q w_q' k(t_q, t_q'); reproduces the script's table
_kbar_all(X, Z, ell_vec, normalize)            (n, m, D) tensor of per-coordinate centered kernels
kernel_additive_amplitude                      sum_i a_sq[i] kbar_i          params: a_sq, kernel_ell
kernel_additive_lengthscale                    var * sum_i k~_i, ell_i = rho_i^-1/2   params: kernel_var, kernel_inv_length_sq
kernel_product_amplitude                       prod_i (1 + a_sq[i] kbar_i)   params: a_sq, kernel_ell
kernel_product_lengthscale                     saasgp.matern_kernel(X, Z, var, rho, noise, include_noise)  [imported]
                                               every kernel adds (noise + 1e-6) * I when include_noise, exactly the reference's line
cell_kernel_diag(cell, X, params)              (n,) marginal prior variance at X; + noise + 1e-6 in predict reproduces the
                                               reference's kernel_diag(var, noise) for product/lengthscale (constant var)
# models  (NumPyro; site names fixed; the product/lengthscale body is SAASGP.model verbatim)
model_product_lengthscale(X, Y, *, alpha, fixed_noise)
model_additive_lengthscale(X, Y, *, alpha, fixed_noise)
model_additive_amplitude(X, Y, *, alpha, fixed_noise, ell_prior)
model_product_amplitude(X, Y, *, alpha, fixed_noise, ell_prior)
Cell(structure, prior, model, kernel, native_site, sites, alpha_default)   frozen dataclass
CELLS                                          registry keyed by (structure, prior)
ALPHA_AMPLITUDE, RHO_EPS, ELL_EPS, ELL_PRIOR   constants of decisions 2 and 3
# inference
_run_nuts(model, X, Y, key, nuts) -> (flat_samples, summary, extra)   run_inference copied: +extra_fields, no printing
_diagnose(flat_samples, extra, thresholds, wall) -> Diagnostics          log-transform positive sites, then summary()
fit(...)                                                                   attempt loop; thinning by [::16] at the end
# prediction
FittedGP._compute_choleskys(), FittedGP._predict(...), FittedGP.posterior()   reference code with kernel/diag generalized
# MAP reference
_dsp_neg_log_joint(u, X, y, D_eff) ; fit_map(...)
# readouts
_sobol_exact_additive(fitted) ; _sobol_exact_product_amplitude(fitted) ; _sobol_qmc(fitted, n, seed)
readouts(...) ; component_means(...) ; shares_from_amplitudes(a_sq, noise) ; manipulation_checks(...)
# SQ1
identify(...)
```

### `bo.py`

```
iteration_rngs, initial_design, IterRNG
standardize(y) -> (z, mean, std)               z = -(y - mean)/std, std = y.std() (ddof 0)
log1mexp(x), log_h(z), log_ei(...), ACQUISITIONS = {"logei": log_ei, "ei": saasbo.ei}
optimize_ei(...)                               copied (5-line diff: seeds, jitter, acq in two places, return value)
_fit_for(method, cfg, X, z, key, labels) -> FittedGP | None     dispatch: cell -> gp.fit; dsp_map -> gp.fit_map; oracle_S -> gp.fit_map(active=labels.S); sobol -> None
_propose(method, fitted, z, rngs, cfg, sobol_seq) -> (x_next, acq_value, acq_wall)   passes cfg.num_restarts_ei, cfg.num_init_candidates
_iteration(...)                                fit (retry once on exception) -> propose -> observe -> row
RunLogger                                      checkpoint.npz (atomic rename, written first), then iterations.csv (one row/iteration),
                                               coords.csv (long: t, i, native_median, p_active, sobol_hat), samples/t{t:03d}.npz;
                                               manifest.json, environment.lock.txt; truncate_to(t) on resume
write_manifest(cfg, objective, out) ; _environment_lock() ; _git_commit()
run_bo(...) ; resolve_config(args) ; main(argv)
```

## 4. The loop (`run_bo`), step by step

1. `X = initial_design(D, n_init, seed)`; `y[t] = objective.observe(X[t], noise_rng(seed, run=t))` for t < n_init (or `objective(X[t])` when `noiseless`). Sobol-search reference: `sobol_seq = qmc.Sobol(D, scramble=True, seed=seed).random(T)`, of which the first `n_init` rows are exactly `X`.
2. For t = n_init … T−1: `z, m, s = standardize(y)`; `y_target = z.min()`; `rngs = iteration_rngs(seed, t)`.
3. Fit: `_fit_for(...)` with `rngs.key` (`fit()` uses it unchanged for attempt 0 and `fold_in(key, 1)` for a diagnostic refit). On exception: log it, retry once with `fold_in(rngs.key, 2)` (disjoint from the refit key); on a second exception the query is one scrambled Sobol point seeded by `rngs.fallback_seed`, `status = "excluded"`, `reason = "exception: ..."`. A fit whose diagnostics failed twice is *used* for the query (the loop must continue) and its row carries `status = "excluded"` from `fit()`.
4. Propose: `x_next, acq = optimize_ei(fitted, y_target, rngs.sobol_seed, rngs.jitter_rng, num_restarts_ei=cfg.num_restarts_ei, num_init=cfg.num_init_candidates, acq=ACQUISITIONS[cfg.acq])`; `acq_value` is logged in the chosen acquisition's units (log of the sample-averaged EI under `logei`); sobol reference: `sobol_seq[t]`, `acq = nan`.
5. Observe `y_next` with `rngs.noise_rng`; append. Two incumbents: `best_obs = y.max()` (what the loop sees) and `best_f = max objective(X[:t+1])` (noise-free, one extra evaluation per iteration, free for synthetic objectives); `regret = objective.f_star - best_f`, the Design-brief's r_t on f, which cannot go negative as the noisy version can. Under `noiseless` the two coincide.
6. Write `checkpoint.npz` (X, y, f, t, statuses) by tmp-file + `os.replace` **first**; then append the row, append `coords.csv` (native summaries `native_median`, `p_active` every iteration from stored samples via `readouts(compute_sobol=False)`; `sobol_hat` only every `sobol_every` iterations and at t = T−1, NaN otherwise, `sobol_computed` flag in the row), and save retained samples to `samples/t{t:03d}.npz` (attempt used, status).
7. Resume: on start, if `checkpoint.npz` exists, load it, require `hash(RunConfig fields)` in the manifest to equal the resolved config's (git hash and package versions are compared and *warned* about in `log.txt` and appended to the manifest's `resumed` list, not asserted), truncate `iterations.csv`, `coords.csv` and `samples/` to the checkpoint's t (a kill between checkpoint and logs cannot then duplicate rows), and continue. Bit-identity follows from step 2's per-`(seed, t)` RNGs and XLA CPU determinism; the manifest records thread settings (`XLA_FLAGS`, `OMP_NUM_THREADS`) because a different thread count is not guaranteed bit-identical.

Row schema (`iterations.csv`): `t, method, family, seed, y, f, best_obs, best_f, regret, acq_value, fit_wall_s, acq_wall_s, fit_calls, nuts_attempts, status, reason, r_hat_max, r_hat_median, frac_r_hat_below_1_05, n_eff_min, divergences, num_steps_mean, y_mean, y_std, sobol_computed, x_0 … x_{D-1}` (`fit_calls` counts exception retries, `nuts_attempts` diagnostic attempts).

## 5. The eight decisions

### D1. Reuse vs copy

| reference symbol | treatment | why |
|---|---|---|
| `saasgp.matern_kernel` | **import** (after the 1-line `jnp.clip` compat patch in `saasgp.py`) | the product/lengthscale kernel must be the reference's; importing makes the "equals reference" test trivial and the diff zero. `saasgp.kernel_diag` is not imported: `cell_kernel_diag + noise + 1e-6` reproduces it and avoids a name clash |
| `util.chunk_vmap` | **import**; call with `chunk_size=min(8, S)` | sidesteps the latent `np` NameError without touching `util.py` |
| `saasbo.ei`, `saasbo.ei_grad` | **import** unchanged | they only call `gp.posterior(x)`; our `FittedGP` satisfies that |
| `SAASGP.model` body | **copy** → `model_product_lengthscale`; `self.alpha`→`alpha`, `self.learn_noise`→`fixed_noise is None` | must be a module-level NumPyro model with the same site names in the same order (bit-for-bit test) |
| `SAASGP.run_inference` | **copy** → `_run_nuts`: `+extra_fields=("diverging","num_steps")`, `+key` argument, printing removed | diagnostics need the extra fields; the sampler call is otherwise identical |
| `compute_choleskys`, `predict`, `posterior` | **copy** into `FittedGP`; `self.kernel(...)` → `cell.kernel(X, Z, params, noise, inc)`; `kernel_diag(var, noise)` → `cell_kernel_diag(cell, X_test, params) + noise + 1e-6` | the centered kernels have non-constant diagonals; for product/lengthscale the generalization reduces to the reference line |
| `saasbo.optimize_ei` | **copy** with 5 changed lines (seeded Sobol, `jitter_rng`, `acq` callable in the candidate scoring and the L-BFGS-B objective, return the acquisition value) | the seeding the brief requires and the LogEI switch; everything else byte-identical; `acq=saasbo.ei` is the reference |
| `saasbo.run_saasbo` | **rewrite** as `run_bo`, keeping the step order (standardize → fit → optimize_ei → evaluate → append) | logging, checkpointing, references and the failure policy are new behaviour |
| `SAASGP`, `run_saasbo` | used only by tests as oracles | equivalence tests |

Recommended: apply the single compatibility line to the vendored `saasgp.py` (`jnp.clip(dsq, a_min=1.0e-12)` → `jnp.clip(dsq, 1.0e-12)`) as its own commit, and record the upstream hash `4af4a8e` in the manifest. This keeps every equivalence test literal rather than monkeypatched. Net copied-and-changed lines against the reference: about 12.

### D2. α on the a² scale — closed form, simulation as verification

Both priors are the same half-Cauchy scale mixture, θ_i = τ²λ_i with τ² ~ HC(α), λ_i ~ HC(1), so the prior-predictive count #{i : θ_i > c} depends on (α, c) only through c/α. Matching the a²-count at c = ε = 0.02 to the ρ-count at a matched threshold ρ_ε therefore gives exactly

  α_a = α_ρ · ε / ρ_ε = 0.1 · 0.02 / ρ_ε,

and the *whole* count distribution matches, not only its median. The matching rule reduces to choosing ρ_ε. Recommended: the variance-share mapping. A unit-variance component at lengthscale ℓ has ν-variance v(ℓ) (this is literally the component variance in the additive/lengthscale cell, σ_f² v(ℓ_i)); "share > ε" ⇔ v(ℓ) > 0.02 ⇔ ℓ < ℓ_ε = 2.561 ⇔ ρ > ρ_ε = 0.1524. Hence **α_a = 0.0131** (`ALPHA_AMPLITUDE`), with `RHO_EPS = 0.1524`, `ELL_EPS = 2.561` also used by the lengthscale cells' native active rule.

Simulation (10⁴ draws, D = 100, numpy, run during planning): reference count of ρ_i > ρ_ε has median 37 (q25/q75 16/64); on the a² scale, α = 0.01 → median 30, α = 0.015 → 41, so α = 0.0131 lands on 37 as the closed form predicts. The test `test_alpha_matches_reference_count` repeats this with 10⁴ draws per side and asserts equal medians (± 1) and quartiles (± 3). Sensitivity, for the open question: ρ_ε = 1 (ℓ < 1) gives α = 0.002; ρ_ε = 4 (ℓ < 0.5) gives α = 0.0005. Note that SAASBO's prior is not sparse at its median (37 of 100 coordinates "active" at the variance-matched threshold); sparsity comes from τ²'s heavy left tail and from the data. That is a property of the reference prior we inherit by construction.

### D3. Lengthscale prior in the amplitude cells

`ELL_PRIOR = (mu=0.0, sigma=1.5)`: ℓ_i ~ LogNormal(0, 1.5) on the [0,1] scale, independent per coordinate, the same in both amplitude cells. Median 1 (one wiggle across the domain), 95 % interval [0.053, 18.9]. Against the families' 0.06–3: P(ℓ < 0.06) = 3 %, P(ℓ > 3) = 23 %, so every study lengthscale is inside the central 95 % without the prior being tuned to them, and the ℓ → 0 direction (where a normalized component becomes white noise and competes with `kernel_noise`) is discouraged at 1 % below 0.03. The check script used sd 2 (95 % [0.02, 50]); sd 1.5 keeps its shape while trimming the white-noise tail, and the Gate 1 replication test asserts only the ordering, which is insensitive to this. The lengthscale cells keep the reference's HC prior on ρ and no ℓ prior at all.

### D4. Diagnostics with one chain, and the doubled-warm-up refit

Computed on all 256 post-warm-up draws (the reference already computes `summary` on the un-thinned flat samples; `numpyro.diagnostics.summary` uses `split_gelman_rubin`, which with one chain splits it into halves, and `effective_sample_size`). Applied to the log of every positive site (log ρ_i / log a_i², log τ², log σ_f², log σ_n², log ℓ_i) — the sampler's own geometry — because the constrained HC-distributed sites are so heavy-tailed that plain R̂/ESS on them is dominated by single draws. Rule, preregistered: pass iff `r_hat_max ≤ 1.1` and `n_eff_min ≥ 16` and `divergences ≤ 5`. Reasons: split-R̂ on two halves of 128 has resolution ≈ ±0.02, and 1.1 is the classical bound; 16 effective draws is the number of samples we retain, so fewer means the retained set is not 16 draws' worth; 5 divergences is 2 % of 256. `frac_r_hat_below_1_05` and `r_hat_median` are logged so the brief's R̂ < 1.05 criterion is reportable per fit without being the gate. Task 11's pilot (10 fits, aligned10, n = 100) reports the false-failure rate; the thresholds are not tuned to it.

Refit: a second `MCMC(NUTS(model, max_tree_depth=6), num_warmup=1024, num_samples=256)` with `fold_in(key, 1)` (attempt 0 uses `key` unchanged), no state reuse (a fresh chain, as the brief's "no warm start" requires). Pass → `status="refit"`; fail → `status="excluded"`, reason = the failed criteria of both attempts; the second attempt's samples are returned either way. `identify()` and every BO row carry `status`, `nuts_attempts` and both `Diagnostics`, so excluded fits are counted downstream, never averaged silently. A refit costs 1280 NUTS iterations, 1.67 × a fit, so the pilot's refit rate r enters the budget (D8); the `n_eff_min ≥ 16` criterion over ≈ 2D + 3 sites at D = 100 is the one most likely to fire, and if the pilot's r exceeds 20 % the gate is revisited before SQ1 is preregistered, not after.

### D5. Noise

Learned in every cell with the reference class's `kernel_noise ~ LogNormal(0, 10)` (`fixed_noise=None`, the default). `--fixed-noise <var>` fixes it, taking the reference's branch (`observation_variance` in the Cholesky and the predictive diagonal). For the noiseless out-of-family checks (embedded Hartmann/Levy, deferred) the fixed value is the reference's own 1e-6 for its noiseless benchmarks: the only defensible number for a function whose true noise is zero, two orders below the regret resolution reported, and identical across cells. The reproduction test also uses 1e-6 because `run_saasbo` hard-codes it.

### D6. Sobol index of the posterior mean

A simpler exact route exists for three of the four cells, so QMC is confined to the fourth. For every centered kernel Σ_q w_q k̃(t_q, x) = 0 exactly (the quadrature defines ν), hence E_{x_j}[1 + a_j² k̄(x_j, u)] = 1 for every u:

- Additive cells: m(x) = Σ_i m_i(x_i); Ŝ_i = Var_ν(m_i)/Σ_j Var_ν(m_j) with Var by the 64-node quadrature. Cost D·Q·n·S ≈ 2·10⁷ flops.
- Product/amplitude: with m_s(x) = Σ_n α_{sn} ∏_j (1 + a_{sj}² k̄_{sj}(x_j, X_{nj})): E_ν m_s = Σ_n α_{sn}; E_{x_{-i}} m_s = Σ_n α_{sn}(1 + a_{si}² k̄_{si}(x_i, X_{ni})) → Var_{x_i} by quadrature; Var_ν(m) = S⁻² Σ_{s,s'} Σ_{n,m} α_{sn}α_{s'm}[∏_j(1 + a_{sj}²a_{s'j}² G_j^{ss'}(X_{nj},X_{mj})) − 1], G_j^{ss'}(u,u') = Σ_q w_q k̄_{sj}(t_q,u) k̄_{s'j}(t_q,u'). Cost S²·n²·D·Q ≈ 7·10¹⁰ flops at n = 200, a few seconds in JAX; exact, deterministic. Implemented as a `lax.scan`/loop over the 136 pairs s ≤ s' (weight 2 off the diagonal), accumulating the (n, n) product over j per pair: 32 MB per pair, never the 8 GB (S², D, n, n) tensor.
- Product/lengthscale (ARD Matérn is not separable): Saltelli's first-order estimator on scrambled Sobol matrices A, B ∈ [0,1]^{N×D} and A_B^{(i)}, Ŝ_i = N⁻¹ Σ_j m(B_j)(m(A_B^{(i)})_j − m(A_j))/V̂; N = 2048 (a power of two, balanced), (D + 2)N ≈ 2.1·10⁵ posterior-mean evaluations, chunked at 512 points (82 MB intermediates), ≈ 10–30 s at n = 200; A, B cached per (D, N, seed) per process; Cholesky factors and α_s reused from `FittedGP`. Accuracy is pinned by a test that runs the QMC estimator on a product/amplitude posterior and compares with the exact value (≤ 0.02).

In-loop: every `sobol_every = 25` iterations plus the last (8 times per run at T = 200, ≤ 5 min total); always in `identify()`. The native summaries are logged every iteration regardless.

### D7. Standardization and the labels

The GP sees z = −(y − ȳ)/s_y with s_y the ddof-0 sample std over the design (the reference's `Y.std()`). In the additive/amplitude cell a_i² is exactly a share of Var(z) = 1, which is (Var_ν f + σ²)/s_y² up to the design's sampling error; in product/amplitude the kernel ∏(1 + a_i² k̄_i) also carries a constant term and pairwise a_i²a_j² terms, so there a_i² is the first-order share and the mapping below is approximate. Mapping back, with σ̂² the sample's `kernel_noise` in z units (or `fixed_noise`): ŝ_i = a_i² /(1 − σ̂²), i.e. the noise inflates the standardized variance by σ²/Var f and the correction divides it out; for `synthobj` (Var f = 1, σ = 0.1) it is 1 %. Original units: a_i² s_y². `p_active` and `share_hat` use ŝ_i. In-loop, s_y is over the BO trajectory, not ν, so in-loop a_i² are trajectory shares and are compared with `labels.s` only qualitatively; `y_mean`/`y_std` are logged per row so any later rescaling is possible. `labels.s_i` is the comparison target in `identify()` (Sobol design ≈ ν); `labels.g_i` for the lengthscale column.

### D8. Cost

One gradient of the product/lengthscale log-joint at (n, D) costs ≈ 3(n³/3 + 2Dn²) flops (forward + reverse), = 32 Mflop at n = 200, D = 100; the reference materializes an (n, n, D) tensor per kernel evaluation (32 MB), so the effective rate is memory-bound, 5–20 GFLOP/s on one core. NUTS at tree depth ≤ 6 takes 40–63 gradients per iteration (`num_steps` is recorded to replace the guess), 768 iterations per fit:

| effective GFLOP/s | steps/iter | t_fit(n = 200) | NUTS per run (180 fits) |
|---|---|---|---|
| 5 | 63 | 310 s | 5.3 h |
| 10 | 50 | 120 s | 2.1 h |
| 20 | 40 | 49 s | 0.85 h |

A run fits at n = 20 … 199 (180 fits), and Σ_{n=20}^{199} cost(n) = 62.0 · cost(200). Additive cells add the (n, Q, D) centering terms: ≈ 1.5×. Each diagnostic refit (D4) adds 1.67 × a fit, so multiply by (1 + 1.67 r) with r the pilot's refit rate. EI optimization ≈ 5–10 s per iteration (5000 candidates + 5 × ≤ 100 L-BFGS-B evaluations over 16 samples) ≈ 0.25–0.5 h per run. Estimate per D = 100, T = 200 run at r = 0: **1–6 core-hours (product/lengthscale), 1.5–9 (additive)**; SQ2's 160 NUTS runs: 300–1,200 core-hours before refits; references negligible. Pilot (Task 11): one fit per cell at n = 200, D = 100, aligned10, recording wall time and mean `num_steps`, plus the 10-fit refit rate; run cost = 62 · t_fit · (1 + 1.67 r) + 180 · t_EI. Preregistered fallback if the budget does not fit: `NUTSConfig(256, 256, 32)` (8 retained; the demo's setting; × 0.67 NUTS cost), and if that still does not fit, `(128, 128, 8)` (× 0.33); either applies to all four cells and both studies, chosen before SQ2 from the pilot numbers, never per cell.

## 6. Tests mapped to files

`tests/test_sagp_kernels.py`
- centered kernel integrates to zero: `Σ_q w_q k̃(t_q, x) = 0` for 20 random x, ℓ ∈ {0.06, 0.5, 3}, |·| < 1e-10.
- `v_of_ell` equals `synthobj.kernel.v` at the script's ten ℓ values to 1e-12 and the table to 1e-3; normalized kernel's quadrature-mean marginal variance = 1 ± 1e-12.
- PSD of all four kernels on 50 random points in D = 7 (min eigenvalue > −1e-10 before jitter); `kernel_product_lengthscale` equals `saasgp.matern_kernel` bit-for-bit (`np.array_equal`).
- `cell_kernel_diag(cell, X, params)` equals `diag(kernel(X, X, params, noise, False))` for each cell.
- α closed form: `test_alpha_matches_reference_count` (D2's simulation, seeds fixed).

`tests/test_sagp_cells.py`
- log-joint: `numpyro.infer.util.log_density(model_product_lengthscale, (X, Y), {...}, params)` equals that of `SAASGP(alpha=0.1).model` on the same `params` to 1e-10, learned and fixed noise.
- site names/shapes per cell via `numpyro.handlers.trace`; the additive/lengthscale trace has exactly the reference's sites.
- `posterior`: `FittedGP` built from a reference `SAASGP`'s `flat_samples` gives `np.array_equal` mean and var vs `SAASGP.posterior`.
- additive posterior mean = Σ_i `component_means` to 1e-12, both additive cells.

`tests/test_sagp_inference.py`
- bit-for-bit NUTS: D = 5, n = 30, `key = split(PRNGKey(0), 2)[0]` (the half `SAASGP.fit` uses for HMC), product/lengthscale, `NUTSConfig(64, 64, 4)` in the test, `DiagThresholds(r_hat_max=inf, n_eff_min=0, max_divergences=10**9)` so no refit can intervene, and the reference built with `verbose=False` (its `progress_bar=True` default takes a different NumPyro execution path); retained samples `np.array_equal` to `SAASGP(..., verbose=False).fit(X, y).flat_samples[::4]`. Also once with the production budget, marked slow.
- diagnostics populated; status "refit" when `DiagThresholds(r_hat_max=0.0)` fails the first attempt and a monkeypatched `_diagnose` passes the second; "excluded" when both fail; second attempt used `num_warmup = 1024` (asserted via a spy on `MCMC`).
- `fit` on the amplitude cells runs end-to-end at D = 5 with the small budget (smoke).

`tests/test_sagp_readouts.py`
- QMC vs exact on a product/amplitude posterior (D = 8, n = 40): max |Ŝ_qmc − Ŝ_exact| ≤ 0.02.
- exact additive Sobol on an exactly additive `synthobj` truth (aligned3 at D = 10 for speed; D = 100, n = 200 marked slow): |Ŝ_i − s_i| ≤ 0.03 on S, ≤ 0.03 off S.
- `shares_from_amplitudes` inverts the D7 mapping on synthetic samples; `p_active` and active rules; `manipulation_checks` returns the five statistics.
- Gate 1 replication (slow): the script's five cases at D = 1, n = 80, noise fixed at 0.01, additive/amplitude cell. The script (run 2026-09-07) gives, normalized kernel, P(var > 0.09): pure noise 0.01, strong-smooth 0.98, strong-rough 1.00, weak-smooth 0.07, weak-rough 0.05. The test asserts the robust partial order — both strong cases ≥ 0.9, both weak cases in [0.01, 0.3], noise ≤ 0.05, and min(strong) > max(weak) > noise — not the within-pair order, which the script itself resolves by 0.02.

`tests/test_sagp_identify.py`
- `identify(make_family("aligned3", 0, D=6), ("additive","amplitude"), n=25, seed=0, nuts=small)` returns every key of the flat record; status ∈ {ok, refit, excluded}; `"sagp.bo" not in sys.modules`.

`tests/test_bo_loop.py`
- same seed → identical trajectory: `dsp_map` at D = 5, T = 15 twice, `X` `np.array_equal`; the same with `product/lengthscale` and `NUTSConfig(32, 32, 4)`.
- resume: run T = 8, then `run_bo(..., T = 15, resume=True)` on the same dir; `X` equals an uninterrupted T = 15 run bit-for-bit; manifest mismatch raises.
- failure policy: monkeypatch `gp.fit` to raise twice → row `status == "excluded"`, x is the fallback Sobol point; raise once → `fit_calls == 2`, status from the fit.
- `regret` (on `best_f`) non-increasing and ≥ 0; `best_obs` can exceed `best_f`; every row has all schema columns; `coords.csv` has native summaries every iteration and `sobol_hat` exactly at the `sobol_every` multiples and the end.
- a checkpoint written but logs not (simulated by deleting the last CSV rows): resume truncates and continues without duplicate rows.
- Sobol vs oracle-S on aligned3, D = 20, T = 50 (slow): median final regret oracle < sobol over 3 seeds.

`tests/test_bo_reference.py` (slow)
- product/lengthscale, D = 5, n_init = 10, T = 25, `fixed_noise=1e-6`, `noiseless=True`, `acq="ei"`, permissive `DiagThresholds` (the reference never refits): `run_bo`'s `X` equals `saasbo.run_saasbo(lambda x: -obj(x), np.zeros(5), np.ones(5), max_evals=25, num_init_evals=10, seed=s, kernel="matern")` bit-for-bit, with the reference patched to draw its per-iteration HMC key, candidate Sobol seed and jitter from `iteration_rngs` (a `SAASGP` subclass overriding `fit` and built with `verbose=False`, `monkeypatch` on `saasbo.qmc.Sobol` and `saasbo.np.random.randn`), the only randomness the reference leaves unseeded.

`tests/test_bo_config.py`
- for every method in `METHODS`, `resolve_config` differs only in `method`; `initial_design` identical; the manifest JSON round-trips `RunConfig` and contains git hash, versions, thread env, reference file hashes, objective labels.

`tests/test_bo_cli.py`
- `--dry-run` via `subprocess` prints the resolved config and creates nothing; bad `--cell` exits non-zero.
- end-to-end `python -m sagp.bo --family aligned3 --D 5 --seed 0 --cell dsp_map --T 15 --out tmp` in < 60 s.

## 7. CLI and SLURM

```
python -m sagp.bo --family aligned10 --seed 3 --cell additive/amplitude --T 200 --out runs/
   --cell also accepts sobol | dsp_map | oracle_S     --D 100   --n-init 20   --acq logei|ei (default logei)
   --alpha F   --fixed-noise F   --noiseless   --sobol-every 25   --no-resume   --dry-run
   --nuts 512,256,16   (the preregistered fallback is passed here, identically for all cells)
```
Run directory: `<out>/<family>/<cell-with-slash-as-dash>/seed{seed:02d}/` holding `manifest.json`, `environment.lock.txt`, `iterations.csv`, `coords.csv`, `checkpoint.npz`, `samples/`, `log.txt`.

```bash
#!/bin/bash
#SBATCH --job-name=sagp --array=0-159 --cpus-per-task=1 --mem=8G --time=16:00:00
FAMILIES=(aligned3 aligned10 decoupled interaction_g0.25)
CELLS=(additive/amplitude additive/lengthscale product/amplitude product/lengthscale)
i=$SLURM_ARRAY_TASK_ID; seed=$((i % 10)); c=$(( (i / 10) % 4 )); f=$(( i / 40 ))
export OMP_NUM_THREADS=1 XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
cd $REPO && /opt/anaconda3/envs/saasbo/bin/python -m sagp.bo --family ${FAMILIES[$f]} --seed $seed \
    --cell ${CELLS[$c]} --T 200 --out runs/   # re-submitting the same array resumes
```

## 8. Tasks

Each task: write the failing tests listed for it in section 6 → run → implement → run → `pytest -q -m "not slow"` → commit. Verify commands use `/opt/anaconda3/envs/saasbo/bin/python -m pytest`.

- [ ] **Task 0 — environment, compat patch, skeleton.** `pip install jax==0.10.2 jaxlib==0.10.2 numpyro==0.21.0` into the conda env; `requirements-sagp.txt`; the one-line `jnp.clip` change in `saasgp.py` with a comment naming JAX 0.10.0; verify `python -c "import saasgp, saasbo"` and a 20-step `SAASGP.fit` runs; `sagp/__init__.py`; save the brief and this plan under `docs/superpowers/`. Verify: the two vendored modules import and fit; `git diff saasgp.py` is one line.
- [ ] **Task 1 — kernels** (`test_sagp_kernels.py` minus the α test). `GL_NODES`, `centered_matern52_1d`, `v_of_ell`, `_kbar_all`, the four kernels, `kernel_diag`. Verify: the zero-integral, table, PSD and bit-for-bit tests.
- [ ] **Task 2 — cells** (`test_sagp_cells.py` log-joint and trace tests). The four models, `Cell`, `CELLS`, constants `ALPHA_AMPLITUDE`, `RHO_EPS`, `ELL_PRIOR`; the α closed-form test. Verify: log-joint equality to 1e-10.
- [ ] **Task 3 — NUTS fit + diagnostics** (`test_sagp_inference.py`). `_run_nuts`, `_diagnose`, `fit` with the attempt loop. Verify: bit-for-bit vs reference; refit/excluded paths.
- [ ] **Task 4 — prediction** (`test_sagp_cells.py` posterior test). `FittedGP` with cached Choleskys, `chunk_size=min(8, S)`. Verify: `np.array_equal` vs `SAASGP.posterior`.
- [ ] **Task 5 — readouts** (`test_sagp_readouts.py`). Exact additive and product/amplitude Sobol, QMC Saltelli, `shares_from_amplitudes`, `readouts`, `component_means`, `manipulation_checks`. Verify: QMC vs exact ≤ 0.02; the additive recovery test at D = 10.
- [ ] **Task 6 — identify()** (`test_sagp_identify.py`). Verify: flat record; no `sagp.bo` import.
- [ ] **Task 7 — fit_map** (DSP and oracle-S). `_dsp_neg_log_joint` (floors 0.025 / 1e-4, prior-mode start), `fit_map`, `active` restriction. Verify: MAP recovers short lengthscales on the active coordinates of aligned3 at D = 10, n = 60 (ℓ_S < ℓ_off median); posterior shape (1, n_test).
- [ ] **Task 8 — loop core** (`test_bo_loop.py` determinism, resume, regret). `iteration_rngs`, `initial_design`, `standardize`, `optimize_ei` copy, `_iteration` (best_obs/best_f/regret), `RunLogger` (checkpoint first, truncate on resume), `run_bo` for cells and `dsp_map`. Verify: identical trajectories; resume bit-identical; no duplicate rows after a simulated partial write.
- [ ] **Task 9 — references, failure policy, manifest, CLI** (`test_bo_loop.py` failure tests, `test_bo_config.py`, `test_bo_cli.py`). `sobol`/`oracle_S` dispatch, exception retry and fallback point, `write_manifest`, `_environment_lock`, `_git_commit`, `resolve_config`, `main`, `--dry-run`. Verify: dry-run subprocess; e2e < 60 s.
- [ ] **Task 10 — reference-trajectory and slow tests** (`test_bo_reference.py`, slow tests of sections 6). Verify: `pytest -m slow` green (≈ 15 min).
- [ ] **Task 11 — pilot and docs.** Time one fit per cell at n = 200, D = 100 (record `num_steps`); 10 aligned10 fits at n = 100 for the diagnostic false-failure rate; the α simulation as a saved table; README section; update `Design-brief.md` §2's held-fixed paragraph (or a note beside it) to this brief's settings. Verify: numbers written into `docs/superpowers/plans/…` §5 D8 and D4.

## 9. Effort

| task | hours |
|---|---|
| 0 env/compat/skeleton | 2 |
| 1 kernels | 4 |
| 2 cells | 4 |
| 3 NUTS fit + diagnostics | 4 |
| 4 prediction | 3 |
| 5 readouts (exact + QMC) | 6 |
| 6 identify | 2 |
| 7 fit_map | 3 |
| 8 loop core + checkpoint | 6 |
| 9 references + manifest + CLI | 4 |
| 10 reference-trajectory + slow tests | 4 |
| 11 pilot + docs | 3 |
| **total** | **≈ 45 h** (+ pilot compute ≈ 4 core-hours) |

## 10. Open questions (≤ 5)

1. **α matching rule (D2).** Do you accept the variance-share mapping ρ_ε = ℓ_ε⁻² with v(ℓ_ε) = ε, giving α = 0.0131? The alternatives change α by an order of magnitude (ℓ < 1 → 0.002; ℓ < 0.5 → 0.0005), so this must be fixed before SQ1 is preregistered.
2. **Exact Ŝ_i in the product/amplitude cell (D6)** instead of QMC, with QMC kept only for product/lengthscale and as a cross-check test. The brief specified QMC for both product cells; the exact route is cheaper and noise-free.
3. **Diagnostic gate (D4):** 1.1 / 16 / 5 on log-scale sites, and using a twice-failed fit's samples for the query with the row marked `excluded`. Alternative: query a Sobol point on exclusion, which changes the trajectory.
4. **Placement:** `fit_map` (DSP and oracle-S) lives in `gp.py` because it is a GP with prediction, and `bo.py` only dispatches by name; objectives are built on the fly by `make_family(family, seed, D)` (bit-identical to `data/objectives` on the same machine), with `--objective-dir` to load a saved one instead. OK?
5. **Touching the reference:** the one-line `jnp.clip` compatibility patch to the vendored `saasgp.py` (required on JAX ≥ 0.10). The alternative is pinning JAX 0.9.x, where the keyword still works with a deprecation warning.

Decided (Quan, 2026-09-07 12:00): LogEI is the acquisition — Ament et al.'s stable per-sample log-EI combined by log-mean-exp over the retained samples, everything else unchanged; the reference-trajectory test runs under `--acq ei`, and `test_bo_loop.py` gains a test that `log_ei` equals `log(saasbo.ei)` where EI is representable and stays finite with a finite gradient where EI underflows to 0.
