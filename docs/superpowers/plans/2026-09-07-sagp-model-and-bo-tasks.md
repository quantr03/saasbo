# sagp tasks — executable briefs for docs/superpowers/plans/2026-09-07-sagp-model-and-bo.md

Each `### Task N` section below is the single source of requirements for that task. It repeats the exact values from the plan so an implementer never needs the whole plan; where it says "plan §k", read only that section of `docs/superpowers/plans/2026-09-07-sagp-model-and-bo.md`. The spec (binding authority) is `docs/superpowers/specs/2026-09-07-sagp-brief.md`.

Global constraints binding every task:
- Interpreter: `/opt/anaconda3/envs/saasbo/bin/python` (never the repo `.venv`). Run tests as `/opt/anaconda3/envs/saasbo/bin/python -m pytest -q -m "not slow"`; `pytest.ini` sets `pythonpath = .` and the `slow` marker (> 10 s). Every non-slow test < 60 s.
- Pins: jax 0.10.2, jaxlib 0.10.2, numpyro 0.21.0, scipy 1.17.1, numpy 2.4.6 (already installed).
- float64 everywhere (`sagp/__init__.py` enables it); CPU platform.
- Vendored `saasgp.py`, `saasbo.py`, `util.py` stay byte-identical except the one `jnp.clip` line Task 0 changes. Import from them where the plan says import; copy where it says copy.
- NUTS budget everywhere: warm-up 512, samples 256, thinning 16 (16 retained), max_tree_depth 6, one chain, default adaptation, a fresh chain per fit; diagnostics on the un-thinned 256 draws.
- Maximization: `synthobj` objectives are maximized (`f_star` is the maximum); the GP always sees `z = -(y - mean(y)) / std(y)` with numpy's ddof-0 `std`, so the reference minimization code runs unchanged.
- Acquisition: LogEI (user decision 2026-09-07) — per-sample stable log-EI combined by log-mean-exp over the retained samples, identical in every cell and reference; `acq="ei"` (the reference's plain EI) exists only for the reproduction test. Defined in Task 8.
- Stage files explicitly (`git add <paths>`); never `git add -A` or `git add .` — the tree carries the user's untracked notes (`Research Context/`, `CLAUDE.md`, `.DS_Store`, `.codegraph/`) and pre-existing deletions that are not yours to commit. Never touch files under `Research Context/`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Style: match `synthobj/` (docstrings that say why, type hints, `from __future__ import annotations`, small pure functions). No speculative features, no configurability beyond what a task names.

Shared vocabulary (defined by the tasks that create them):
- `CellKey = tuple[str, str]`, structure ∈ {"additive","product"}, prior ∈ {"amplitude","lengthscale"}.
- Kernel signature: `kernel(X, Z, params: dict, noise, include_noise)` → `(n, m)`; `include_noise` static; when true adds `(noise + 1.0e-6) * jnp.eye(n)` exactly as `saasgp.matern_kernel` does.
- `params` keys per cell: additive/amplitude and product/amplitude → `a_sq (D,)`, `kernel_ell (D,)`; additive/lengthscale and product/lengthscale → `kernel_var ()`, `kernel_inv_length_sq (D,)`.
- Model site names: product/lengthscale and additive/lengthscale → `kernel_var`, `kernel_noise` (only when noise is learned), `kernel_tausq`, `_kernel_inv_length_sq`, deterministic `kernel_inv_length_sq`, `Y`. Amplitude cells → `kernel_noise` (when learned), `kernel_tausq`, `_a_sq`, deterministic `a_sq`, `kernel_ell`, `Y`.
- `FittedGP` attributes: `cell` (CellKey, or the strings "dsp_map"/"oracle_S"), `X_train (n, D)`, `Y_train (n,)` as jnp float64, `samples: dict[str, jnp.ndarray]` with leading dim S, `fixed_noise: float | None`, `active: np.ndarray | None`, `status`, `status_reason`, `attempts: tuple[Diagnostics, ...]`.

---

### Task 0: Environment, compat patch, skeleton

**Files:** Modify `saasgp.py` (one line). Create `sagp/__init__.py`, `requirements-sagp.txt`, `tests/test_sagp_env.py`. Stage (already written by the controller, uncommitted): `docs/superpowers/specs/2026-09-07-sagp-brief.md`, `docs/superpowers/plans/2026-09-07-sagp-model-and-bo.md`, `docs/superpowers/plans/2026-09-07-sagp-model-and-bo-tasks.md`.

**Steps:**
1. Confirm the installed stack: `python -c "import jax, jaxlib, numpyro; print(jax.__version__, jaxlib.__version__, numpyro.__version__)"` prints `0.10.2 0.10.2 0.21.0`, and `jnp.clip(jnp.ones(1), a_min=0.0)` raises `TypeError` (the keyword was removed in JAX 0.10.0).
2. `saasgp.py` line 43: change `exponent = root_five * jnp.sqrt(jnp.clip(dsq, a_min=1.0e-12))` to `exponent = root_five * jnp.sqrt(jnp.clip(dsq, 1.0e-12))  # JAX 0.10.0 removed the a_min keyword; positional min is identical`. Nothing else in the file changes; `git diff saasgp.py` must show exactly one changed line.
3. `requirements-sagp.txt`: five lines, `jax==0.10.2`, `jaxlib==0.10.2`, `numpyro==0.21.0`, `scipy==1.17.1`, `numpy==2.4.6`.
4. `sagp/__init__.py`: module docstring (what sagp is: the thesis's four sparse-GP cells on the SAASBO reference code path); then, before anything imports jax arrays: `import numpyro; numpyro.set_platform("cpu"); numpyro.set_host_device_count(1); numpyro.enable_x64()`. Lazy exports via a module-level `__getattr__` for `fit`, `FittedGP`, `identify`, `readouts` (from `sagp.gp`) and `run_bo` (from `sagp.bo`), so that `import sagp` and `import sagp.gp` never import `sagp.bo`. `__all__` lists those five names.
5. `tests/test_sagp_env.py` (three tests, < 60 s total): (a) the versions of jax, jaxlib, numpyro, scipy, numpy equal the pins in `requirements-sagp.txt` (parse the file); (b) `import saasgp, saasbo, util` succeeds and `saasgp.matern_kernel(X, X, 1.0, jnp.ones(3), 0.1, True)` on 6 random points in D = 3 returns a finite (6, 6) array whose diagonal equals `1.0 + 0.1 + 1e-6` to 1e-12; (c) `import sagp` enables x64 (`jnp.ones(1).dtype == jnp.float64`) and a `saasgp.SAASGP(num_warmup=10, num_samples=8, thinning=4, verbose=False, kernel="matern").fit(X, y)` on 10 random points in D = 3 followed by `.posterior(X_test)` returns shapes `(2, n_test)` for both outputs — the vendored reference runs on the pinned stack.
6. Run `/opt/anaconda3/envs/saasbo/bin/python -m pytest -q -m "not slow"`: the existing 291 synthobj tests plus yours pass.
7. Two commits: `compat(saasgp): positional jnp.clip minimum (keyword removed in JAX 0.10.0)` containing only `saasgp.py`; then `chore(sagp): package skeleton, pins, env test, brief and plan docs` containing the rest (stage each path explicitly).

**Verify:** `git diff HEAD~2 -- saasgp.py` is one line; `python -c "import sagp, sys; assert 'sagp.bo' not in sys.modules"`.

---

### Task 1: Kernels

**Files:** Create `sagp/gp.py` (kernels section only; later tasks append), `tests/test_sagp_kernels.py` (all tests below except the α test, which Task 2 adds).

**Read:** plan §3 "gp.py" kernels block, plan §5 D1 (the `saasgp.matern_kernel` row), and `Research Context/research direction/Kernel normalization check.py` (read-only) / `synthobj/kernel.py` for the numpy reference of `centered`, `v`.

**Produces (exact names):**
```python
GL_NODES, GL_WEIGHTS            # jnp.asarray of synthobj.kernel.GL_NODES / GL_WEIGHTS (64-node Gauss-Legendre on [0,1]); import them, do not recompute
def matern52_1d(r):             # (1 + s + s^2/3) exp(-s), s = sqrt(5) r; r >= 0, any shape
def v_of_ell(ell):              # 1 - sum_{q,q'} w_q w_q' matern52_1d(|t_q - t_q'| / ell); scalar or (D,) ell -> same shape
def centered_matern52_1d(x, z, ell, normalize=False):   # x (n,), z (m,) -> (n, m): k - m(x) - m(z) + c, / v_of_ell(ell) if normalize
def _kbar_all(X, Z, ell_vec, normalize):  # X (n, D), Z (m, D), ell_vec (D,) -> (n, m, D) per-coordinate centered kernels
def _kbar_diag(X, ell_vec, normalize):    # (n, D): centered kernel at zero distance, 1 - 2 m_i(x_i) + c_i, normalized if asked
def kernel_additive_amplitude(X, Z, params, noise, include_noise)     # sum_i a_sq[i] * kbar_i (normalized)
def kernel_additive_lengthscale(X, Z, params, noise, include_noise)   # kernel_var * sum_i ktilde_i (unnormalized), ell_i = kernel_inv_length_sq[i] ** -0.5
def kernel_product_amplitude(X, Z, params, noise, include_noise)      # prod_i (1 + a_sq[i] * kbar_i)
def kernel_product_lengthscale(X, Z, params, noise, include_noise)    # saasgp.matern_kernel(X, Z, params["kernel_var"], params["kernel_inv_length_sq"], noise, include_noise)
KERNELS: dict[CellKey, tuple[kernel_fn, diag_fn]]                     # the four, keyed ("additive","amplitude") etc.
def cell_kernel_diag(key, X, params):     # (n,) marginal prior variance at X for KERNELS[key], WITHOUT noise/jitter; O(n D Q), never via an (n, n) matrix
```
Details: m(x) = Σ_q w_q k(|x − t_q|/ℓ), c = Σ_{q,q'} w_q w_q' k(|t_q − t_q'|/ℓ), v = 1 − c. Decorate the four kernels with `@partial(jit, static_argnums=(4,))` like the reference; `params` is a dict (a pytree). Diagonals: additive/amplitude Σ_i a_sq_i k̄_i(x_i, x_i); additive/lengthscale var Σ_i k̃_i(x_i, x_i); product/amplitude ∏_i (1 + a_sq_i k̄_i(x_i, x_i)); product/lengthscale `kernel_var * ones(n)`. All differentiable in every entry of `params` (no `jnp.abs` at exactly zero distance is a problem for Matérn-5/2: its derivative at r = 0 is 0, but avoid NaN-producing `sqrt` of 0 in gradients — use `jnp.abs` on differences, never `sqrt(square)`).

**Tests (`tests/test_sagp_kernels.py`, all < 60 s, CPU):**
- zero integral: for 20 random x in [0,1] and ℓ ∈ {0.06, 0.5, 3}: `|Σ_q w_q k̃(t_q, x)| < 1e-10` for the centered kernel, and the same after normalization.
- `v_of_ell` equals `synthobj.kernel.v` to 1e-12 at ℓ ∈ {0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 30}, and equals the check script's table to 1e-3: 0.88574, 0.78149, 0.60296, 0.28218, 0.10680, 0.03183, 0.01478, 0.00546, 0.00138, 0.00015.
- normalized kernel has quadrature-mean marginal variance 1: `Σ_q w_q k̄(t_q, t_q) = 1 ± 1e-12` for ℓ ∈ {0.06, 0.5, 3}.
- PSD: 50 random points in D = 7, random positive params (a_sq ~ U(0,1), ell ~ U(0.1, 2), var = 1.3, rho ~ U(0.1, 5)): min eigenvalue of `kernel(X, X, params, 0.0, False)` > −1e-10 for all four kernels; with `include_noise=True` and noise 0.1 the diagonal grows by exactly `0.1 + 1e-6`.
- `kernel_product_lengthscale(X, Z, {"kernel_var": v, "kernel_inv_length_sq": rho}, noise, inc)` is `np.array_equal` to `saasgp.matern_kernel(X, Z, v, rho, noise, inc)` for both `inc` values.
- `cell_kernel_diag(key, X, params)` equals `jnp.diagonal(kernel(X, X, params, 0.0, False))` to 1e-12 for each of the four keys.
- gradients exist: `jax.grad(lambda p: kernel(X, X, p, 0.0, False).sum())(params)` is finite for every kernel, including with two identical rows in X.

**Commit:** `feat(sagp): centered/normalized Matern-5/2 kernels and the four cell kernels`.

---

### Task 2: Cells — models, registry, constants

**Files:** Modify `sagp/gp.py` (models section). Create `tests/test_sagp_cells.py` (the two tests below); add the α test to `tests/test_sagp_kernels.py`.

**Read:** plan §5 D2 and D3 in full, D1 rows for `SAASGP.model`; `saasgp.py` lines 97–111 (the reference model, to copy).

**Produces:**
```python
ACTIVE_EPS = synthobj.objective.ACTIVE_EPS        # 0.02, imported so both modules share one cutoff
ALPHA_LENGTHSCALE = 0.1                           # the reference default
ELL_EPS: float      # solve v_of_ell(ell) == ACTIVE_EPS on [0.5, 50] with scipy.optimize.brentq at import (≈ 2.5613)
RHO_EPS = ELL_EPS ** -2                            # ≈ 0.15244
ALPHA_AMPLITUDE = ALPHA_LENGTHSCALE * ACTIVE_EPS / RHO_EPS   # ≈ 0.013120 (plan D2's closed form)
ELL_PRIOR = (0.0, 1.5)                             # LogNormal(mu, sigma) on each 1-D lengthscale, [0,1] scale (plan D3)

def model_product_lengthscale(X, Y, *, alpha, fixed_noise): ...
def model_additive_lengthscale(X, Y, *, alpha, fixed_noise): ...
def model_additive_amplitude(X, Y, *, alpha, fixed_noise, ell_prior): ...
def model_product_amplitude(X, Y, *, alpha, fixed_noise, ell_prior): ...

@dataclass(frozen=True)
class Cell:
    structure: str; prior: str; model: Callable; kernel: Callable; kernel_diag: Callable
    native_site: str        # "kernel_inv_length_sq" or "a_sq"
    sites: tuple[str, ...]  # every site to retain in FittedGP.samples (sampled + deterministic; kernel_noise only if learned -- filter at fit time)
    alpha_default: float
    @property
    def key(self) -> CellKey
CELLS: dict[CellKey, Cell]
```
`model_product_lengthscale` is the reference `SAASGP.model` body verbatim with `self.alpha → alpha`, `self.learn_noise → fixed_noise is None`, `self.observation_variance → fixed_noise`, and `self.kernel(...) → kernel_product_lengthscale(X, X, {"kernel_var": var, "kernel_inv_length_sq": inv_length_sq}, noise, True)` (which delegates to `saasgp.matern_kernel`, same numerics). Site order and names exactly: `kernel_var ~ LogNormal(0, 10)`, `kernel_noise ~ LogNormal(0, 10)` if learned, `kernel_tausq ~ HalfCauchy(alpha)`, `_kernel_inv_length_sq ~ HalfCauchy(jnp.ones(P))`, `kernel_inv_length_sq = numpyro.deterministic(tausq * _kernel_inv_length_sq)`, `Y ~ MultivariateNormal(zeros(N), k)` observed. `model_additive_lengthscale` is identical except the kernel. Amplitude cells: `kernel_noise` (same rule), `kernel_tausq ~ HalfCauchy(alpha)`, `_a_sq ~ HalfCauchy(jnp.ones(P))`, `a_sq = deterministic(tausq * _a_sq)`, `kernel_ell ~ LogNormal(jnp.full(P, ell_prior[0]), ell_prior[1])`, `Y` with the amplitude kernel and params `{"a_sq": a_sq, "kernel_ell": kernel_ell}`; no `kernel_var` site. `alpha_default`: 0.1 for lengthscale cells, `ALPHA_AMPLITUDE` for amplitude cells.

**Tests:**
- `tests/test_sagp_cells.py::test_product_lengthscale_log_joint_matches_reference`: X 12 random points in D = 4, Y random; params `{"kernel_var": 1.3, "kernel_noise": 0.05, "kernel_tausq": 0.2, "_kernel_inv_length_sq": array of 4 positives}`; `numpyro.infer.util.log_density(functools.partial(model_product_lengthscale, alpha=0.1, fixed_noise=None), (X, Y), {}, params)[0]` equals `log_density(saasgp.SAASGP(alpha=0.1).model, (X, Y), {}, params)[0]` to 1e-10; repeat with `fixed_noise=1e-6` vs `SAASGP(observation_variance=1e-6)` and params without `kernel_noise`.
- `tests/test_sagp_cells.py::test_trace_sites_per_cell`: for every cell, `numpyro.handlers.trace(numpyro.handlers.seed(partial(model, alpha=..., fixed_noise=None[, ell_prior=ELL_PRIOR]), PRNGKey(0))).get_trace(X, Y)` has exactly the site names listed above in that order (sample + deterministic sites, excluding `Y`'s plate internals) with the stated shapes; with `fixed_noise=1e-6` the `kernel_noise` site is absent.
- `tests/test_sagp_kernels.py::test_alpha_matches_reference_count`: numpy, `default_rng(0)`, 10⁴ draws, D = 100: reference count `#(tausq * lam > RHO_EPS)` with tausq = 0.1·|Cauchy|, lam = |Cauchy| per coordinate; amplitude count `#(tausq_a * lam > ACTIVE_EPS)` with tausq_a = ALPHA_AMPLITUDE·|Cauchy|. Assert the medians differ by ≤ 1 and the 25 %/75 % quantiles by ≤ 3, and `abs(ALPHA_AMPLITUDE - 0.01312) < 2e-4`, `abs(ELL_EPS - 2.5613) < 2e-3`.

**Commit:** `feat(sagp): the four NumPyro cells, registry and prior constants`.

---

### Task 3: NUTS fit and diagnostics

**Files:** Modify `sagp/gp.py` (inference section; the `FittedGP` class shell). Create `tests/test_sagp_inference.py`.

**Read:** plan §2 (`NUTSConfig`, `DiagThresholds`, `Diagnostics`, `FittedGP` attributes, `fit` docstring), plan §5 D4 in full, D1 rows for `run_inference`; `saasgp.py` lines 113–141 (to copy).

**Produces:**
```python
@dataclass(frozen=True)
class NUTSConfig: num_warmup: int = 512; num_samples: int = 256; thinning: int = 16; max_tree_depth: int = 6; num_chains: int = 1
@dataclass(frozen=True)
class DiagThresholds: r_hat_max: float = 1.1; n_eff_min: float = 16.0; max_divergences: int = 5
@dataclass(frozen=True)
class Diagnostics: r_hat_max: float; r_hat_median: float; frac_r_hat_below_1_05: float; n_eff_min: float; divergences: int; num_steps_mean: float; wall_s: float; passed: bool; reason: str
class FittedGP:   # plain class with __init__(cell, X_train, Y_train, samples, fixed_noise, active, status, status_reason, attempts); no methods yet (Task 4 adds them)
def _run_nuts(model, X, Y, key, nuts) -> tuple[dict, dict, float]:   # (flat_samples, extra_fields, wall_s)
def _diagnose(flat_samples, extra, thresholds, wall_s) -> Diagnostics
def fit(X, y, key, cell, *, alpha=None, fixed_noise=None, nuts=NUTSConfig(), thresholds=DiagThresholds(), ell_prior=ELL_PRIOR) -> FittedGP
```
`_run_nuts` is `SAASGP.run_inference` copied: `NUTS(model, max_tree_depth=nuts.max_tree_depth)`, `MCMC(kernel, num_warmup=..., num_samples=..., num_chains=..., progress_bar=False)`, `mcmc.run(key, X, Y, extra_fields=("diverging", "num_steps"))`, `flat = mcmc.get_samples(group_by_chain=False)`, `extra = mcmc.get_extra_fields()`; no printing. `_diagnose`: take the log of every *sampled* positive site present (`kernel_var`, `kernel_noise`, `kernel_tausq`, `_kernel_inv_length_sq`, `_a_sq`, `kernel_ell`), call `numpyro.diagnostics.summary(logs, prob=0.9, group_by_chain=False)`, pool every parameter's `r_hat`/`n_eff` (flatten vector sites): `r_hat_max`, `r_hat_median`, `frac_r_hat_below_1_05`, `n_eff_min`; `divergences = int(extra["diverging"].sum())`, `num_steps_mean = float(extra["num_steps"].mean())`; `passed` iff `r_hat_max <= thresholds.r_hat_max and n_eff_min >= thresholds.n_eff_min and divergences <= thresholds.max_divergences`; `reason` names each failed criterion with its value, "" when passed. `fit`: `cell` may be a `CellKey` or a `Cell`; `alpha = cell.alpha_default if alpha is None`; bind `model = functools.partial(cell.model, alpha=alpha, fixed_noise=fixed_noise[, ell_prior=ell_prior])` (ell_prior only for amplitude cells); attempt 0 with `key` **unchanged**; if `not diag.passed`: attempt 1 with `jax.random.fold_in(key, 1)` and `dataclasses.replace(nuts, num_warmup=2 * nuts.num_warmup)`; status "ok" (attempt 0 passed) / "refit" (attempt 1 passed) / "excluded" (both failed; reason joins both); samples from the attempt used: `{s: flat[s][::nuts.thinning] for s in cell.sites if s in flat}`; return `FittedGP(cell=cell.key, X_train=jnp.asarray(X, float64), Y_train=..., samples=..., fixed_noise=fixed_noise, active=None, status=..., status_reason=..., attempts=(diag0,) or (diag0, diag1))`. `fit` never raises on diagnostics; JAX/NumPyro exceptions propagate.

**Tests (`tests/test_sagp_inference.py`):**
- `test_nuts_reproduces_reference_bit_for_bit`: D = 5, n = 30 random X, y; `ref = saasgp.SAASGP(alpha=0.1, num_warmup=64, num_samples=64, max_tree_depth=6, num_chains=1, thinning=4, verbose=False, observation_variance=0.0, kernel="matern").fit(X, y)` (seed 0); ours: `fit(X, y, key=jax.random.split(jax.random.PRNGKey(0), 2)[0], cell=("product","lengthscale"), nuts=NUTSConfig(64, 64, 4), thresholds=DiagThresholds(r_hat_max=float("inf"), n_eff_min=0.0, max_divergences=10**9))`; for each of `kernel_var`, `kernel_noise`, `kernel_tausq`, `kernel_inv_length_sq`: `np.array_equal(ours.samples[s], np.asarray(ref.flat_samples[s])[::4])`; `ours.samples[s].shape[0] == 16`; `ours.status == "ok"`.
- `test_nuts_reproduces_reference_bit_for_bit_production_budget` (not marked slow — ruling R20 dropped the marker in Task 10: it runs in 1.9–3.8 s and `pytest.ini` defines slow as > 10 s, so reference equivalence at the real budget runs on every invocation): same with the default `NUTSConfig()` vs `SAASGP(num_warmup=512, num_samples=256, thinning=16, ...)`.
- `test_refit_and_excluded_paths`: monkeypatch `sagp.gp._diagnose` with a stub returning `passed=False` on the first call and `passed=True` on the second → status "refit", `len(attempts) == 2`, and a spy on `sagp.gp.MCMC` (wrap the class) records `num_warmup == 128` for the second construction when `nuts=NUTSConfig(64, 32, 4)`; stub always-False → status "excluded", reason non-empty, samples still present with leading dim 8; second attempt's key differs from the first (spy on `_run_nuts` args).
- `test_diagnostics_fields`: a real small fit (D = 3, n = 15, `NUTSConfig(32, 32, 4)`) has finite `r_hat_max`, `n_eff_min > 0`, integer `divergences >= 0`, `num_steps_mean > 0`, `wall_s > 0`.
- `test_amplitude_cells_fit_smoke`: both amplitude cells and additive/lengthscale run `fit` at D = 3, n = 15, `NUTSConfig(32, 32, 4)` and return samples of the right sites and shapes (`a_sq (8, 3)`, `kernel_ell (8, 3)`, no `kernel_var`).
- `test_cells_wiring` (carried from the Task 2 review): for every cell in `CELLS`, `cell.model.__name__ == f"model_{cell.structure}_{cell.prior}"`, `cell.kernel is KERNELS[cell.key][0]` and `cell.kernel_diag is KERNELS[cell.key][1]` — a transposed registry entry would otherwise pass every other test, since site names encode only the prior.
- `test_fit_rejects_fixed_noise_zero`: `fit(..., fixed_noise=0.0)` raises `ValueError` mentioning `None` — in the vendored reference `observation_variance=0.0` means "learn the noise" while in our models `fixed_noise=0.0` would fix it at zero; `fit` refuses the ambiguous value (also carried from the Task 2 review).

**Commit:** `feat(sagp): NUTS fit with diagnostics, doubled-warm-up refit and status`.

---

### Task 4: Prediction

**Files:** Modify `sagp/gp.py` (`FittedGP` methods). Add to `tests/test_sagp_cells.py`.

**Read:** plan §2 `FittedGP.posterior` docstring, D1 rows for `compute_choleskys`/`predict`/`posterior`; `saasgp.py` lines 143–198 (to copy); `util.py` (`chunk_vmap`, imported).

**Produces (methods on `FittedGP`):**
- `_kernel(self)` → `(kernel_fn, diag_fn)`: `KERNELS[self.cell]` for a `CellKey`; for the strings `"dsp_map"`/`"oracle_S"` the product/lengthscale pair.
- `_params(self, s)` → the params dict of sample s (the cell's param keys sliced from `samples`).
- `_noises(self)` → `(S,)`: `samples["kernel_noise"]` if present else `fixed_noise * ones(S)`.
- `_compute_choleskys(self, chunk_size=None)`: the reference's, over samples, via `util.chunk_vmap` with `chunk_size = min(8, S)`; caches `self._Ls (S, n, n)`.
- `_predict(self, X_test, L, params, noise)`: the reference's `predict` with `k_pX = kernel(X_test, X_train, params, noise, False)`, `mean = k_pX @ cho_solve((L, True), Y_train)`, `k_pp = diag_fn(X_test, params) + noise + 1.0e-6`, `diag_cov = k_pp - (L_kXp * L_kXp).sum(0)`.
- `posterior(self, X_test)` → `(mean, var)` each `(S, n_test)` jnp; computes Choleskys on first call; when `self.active is not None`, all kernel evaluations use `X_train[:, active]` and `X_test[:, active]` (the stored `X_train` stays full-D). Memory policy (ledger ruling R19, identical for every cell): if `n_test * n * D > 2**25` (256 MB of float64 for the `(n_test, n, D)` broadcast that two of the kernels materialize — Task 1 measured 420 MB / 775 MB peaks for additive/lengthscale and product/amplitude at 5000 × 200 × 100, versus 25 MB / 12 MB for the fused ones), evaluate `X_test` in row chunks of 256 and concatenate along the test axis; otherwise run the reference's single-call path unchanged. The diagonal term (`cell_kernel_diag`, which builds an unfused `(n_test, Q, D)` tensor — 461 MB jitted / 1.1 GB eager at 5000 × 100 per the Task 1 review) is evaluated inside the same per-chunk call, never once for the full `X_test`. Every bit-for-bit comparison against the reference in the test-suite (D ≤ 5, n ≤ 30, ≤ 5000 points → at most 7.5e5 elements) stays on the unchunked path. Add `test_posterior_chunked_equals_unchunked`: on a hand-made product/amplitude posterior (D = 6, n = 30, S = 2) force the chunked path by temporarily lowering the threshold (a module constant `_CHUNK_THRESHOLD` you monkeypatch) and compare with the unchunked result to 1e-12.
- `alphas(self)` → `(S, n)`: `cho_solve((L_s, True), Y_train)` per sample (used by Task 5).
- One-line cleanup carried from the Task 3 review: in the comment above `_POSITIVE_SAMPLED_SITES` (around `sagp/gp.py:613-614`), delete the clause claiming a deterministic site "carries no independent evidence about the chain" (split-R̂ of log ρ_i = log τ² + log λ_i is not determined by its terms' R̂s); keep only the rationale that the gate is applied in the sampler's own coordinates.

**Tests (`tests/test_sagp_cells.py`):**
- `test_posterior_matches_reference`: D = 4, n = 20; `ref = SAASGP(num_warmup=16, num_samples=32, thinning=4, verbose=False, kernel="matern").fit(X, y)` (8 retained samples — the reference's own `posterior` uses `chunk_size=8` and its `util.get_chunks` raises `NameError` whenever the retained count is not a multiple of 8; never give the reference fewer than 8 retained samples in any test); build `FittedGP(cell=("product","lengthscale"), X_train=X, Y_train=y, samples={s: ref.flat_samples[s][::4] for s in ("kernel_var","kernel_noise","kernel_tausq","kernel_inv_length_sq")}, fixed_noise=None, active=None, status="ok", status_reason="", attempts=())`; `mean, var = fitted.posterior(X_test)` for 7 random test points is `np.array_equal` to `ref.posterior(X_test)`; repeat with `observation_variance=1e-6` and `fixed_noise=1e-6`.
- `test_posterior_shapes_all_cells`: hand-made samples (S = 3) for each cell → `(3, n_test)` finite outputs, `var > 0`.
- `test_oracle_active_restricts_columns`: with `active = [0, 2]`, the posterior equals that of a `FittedGP` built on `X[:, [0, 2]]` with `active=None`, when called with `X_test[:, [0, 2]]` vs full `X_test`.

**Commit:** `feat(sagp): per-sample prediction generalized to the four kernels`.

---

### Task 5: Readouts

**Files:** Modify `sagp/gp.py` (readouts section). Create `tests/test_sagp_readouts.py`; add one test to `tests/test_sagp_cells.py`.

**Read:** plan §2 docstrings of `readouts`, `component_means`, `manipulation_checks`; plan §5 D6 and D7 in full (all formulas live there); ledger rulings R4 and R7 (`.superpowers/sdd/2026-09-07-sagp-model-and-bo/progress.md`).

**Carried cleanups from the Task 4 review (small edits in the prediction section of `gp.py`, do them first, keep them in the same commit):** (a) ruling R21 — the row-chunking gate in `posterior` becomes `n_test * (n + 64) * d_used > _CHUNK_THRESHOLD` with `_CHUNK_THRESHOLD = 2**24` (the centered cells build an `(n_test, 64, D)` quadrature tensor independent of n, which the old gate `n_test * n * d_used > 2**25` missed at small n); update the constant's comment to say the threshold is compared against that quantity and that the per-block peak is the 8-sample `chunk_vmap` batch times the block tensor; verify every reference-comparison test still runs unchunked (D ≤ 6, n ≤ 30, ≤ 600 points); (b) `_predict`'s docstring names `diag_fn`, not `cell_kernel_diag`; (c) `_kernel()` and `_param_sites()` raise `ValueError` for a string cell other than `"dsp_map"`/`"oracle_S"` instead of silently using product/lengthscale (add a two-line test).

**Produces:**
```python
def shares_from_amplitudes(a_sq, noise) -> jnp.ndarray            # (S, D) / (S, 1): a_sq / (1 - noise)
def component_means(fitted, x_grid) -> jnp.ndarray                # additive cells: (S, D, G); m_{s,i}(x) = w_{s,i} * k_i(x, X[:, i]) @ alpha_s with w = a_sq_i (amplitude, normalized kbar) or kernel_var (lengthscale, unnormalized ktilde)
def _sobol_exact_additive(fitted) -> tuple[np.ndarray, float]     # (S_hat (D,), total_var): per-component quadrature variances of the sample-averaged component means, normalized by their sum
def _sobol_exact_product_amplitude(fitted) -> tuple[np.ndarray, float]   # D6's closed form with the (s <= s') pair loop; never materialize an (S, S, D, n, n) tensor
def _sobol_qmc(fitted, n=2048, seed=0) -> tuple[np.ndarray, float]       # Saltelli first-order estimator, A/B from one qmc.Sobol(2D, scramble=True, seed=seed).random(n) split in halves, chunks of 512 points; posterior mean = mean over samples of fitted.posterior(...)[0]
def readouts(fitted, *, eps=ACTIVE_EPS, compute_sobol=True, sobol_n=2048, sobol_seed=0) -> dict
def manipulation_checks(readout, labels) -> dict
```
`readouts` keys: `native (S, D)` (the cell's `native_site` samples, i.e. `a_sq` or `kernel_inv_length_sq`; for "dsp_map"/"oracle_S" the single `kernel_inv_length_sq`), `native_median (D,)`, `share_hat (S, D)` (amplitude cells only, `shares_from_amplitudes(a_sq, noise)` with noise = per-sample `kernel_noise` or `fixed_noise`; `None` otherwise), `p_active (D,)` = mean over samples of `share_hat > eps` (amplitude) or of `kernel_inv_length_sq > RHO_EPS` (lengthscale), `sobol_hat (D,)` (exact for the two additive cells and product/amplitude; QMC for product/lengthscale, dsp_map, oracle_S — for the oracle, coordinates outside `active` get 0; NaN everywhere when `compute_sobol=False`), `total_var_hat` (float, NaN when not computed), `active_neutral = sobol_hat > eps`, `active_native = p_active > 0.5`. Numpy arrays out. `manipulation_checks(readout, labels)` returns `spearman_native_vs_s`, `spearman_native_vs_g`, `spearman_sobol_vs_s`, `spearman_native_vs_sobol` (each `scipy.stats.spearmanr(...)[0]` over all D coordinates), and `amplitude_vs_realized`: an `(|S|, 2)` array of (median `share_hat` or `native_median`, `labels.s`) rows over `labels.S`.

**Tests:**
- `tests/test_sagp_cells.py::test_additive_posterior_mean_is_sum_of_component_means`: hand-made samples (S = 3, D = 4, n = 12) for both additive cells: `component_means(fitted, x_grid).sum(1)` at grid points equals `fitted.posterior(X_grid)[0]` to 1e-12 where `X_grid` is the grid on every coordinate simultaneously (posterior mean of a sum kernel is the sum).
- `test_qmc_matches_exact_on_product_amplitude`: hand-made product/amplitude posterior (D = 8, n = 40, S = 2, one coordinate with large `a_sq`, others small): `max |_sobol_qmc(fitted, 2048, 0)[0] - _sobol_exact_product_amplitude(fitted)[0]| <= 0.02` and total variances within 5 %.
- `test_exact_product_amplitude_matches_brute_force`: D = 2, n = 3, S = 2: closed-form `total_var` equals brute-force 64 × 64 tensor-grid quadrature of the posterior mean to 1e-10; first-order indices too.
- `test_additive_sobol_recovers_labels` (D = 10, aligned3, n = 120, fast `NUTSConfig(64, 64, 4)`, additive/amplitude): `|sobol_hat - labels.s| <= 0.03` on and off `S`; the same at D = 100, n = 200 with the production budget marked slow.
- `test_shares_and_active_rules`: synthetic `a_sq`, noise 0.01 → `share_hat = a_sq / 0.99`; `p_active` and both active rules behave on a constructed readout.
- `test_manipulation_checks_keys`: returns the five statistics with finite values on a constructed readout and `make_family("aligned3", 0, D=10).labels`.
- `test_gate1_replication` (slow): the check script's five cases at D = 1, n = 80, uniform x from `default_rng(3)`, noise fixed at 0.01 (`fixed_noise=0.01`), additive/amplitude cell with the production budget; components built with `synthobj` primitives (`kernel.eigen_factor` draws rescaled to variance 0.25 / 0.02 at ℓ 1.5 / 0.08, as the script does); `P(share_hat > 0.09)` from the retained samples. Assert: both strong cases ≥ 0.9; both weak cases in [0.01, 0.3]; noise ≤ 0.05; `min(strong) > max(weak) > noise`. Script's own values for the docstring: noise 0.01, strong-smooth 0.98, strong-rough 1.00, weak-smooth 0.07, weak-rough 0.05.

**Commit:** `feat(sagp): posterior readouts — exact and QMC Sobol indices, shares, active rules`.

---

### Task 6: identify()

**Files:** Modify `sagp/gp.py`. Create `tests/test_sagp_identify.py`.

**Read:** plan §2 `identify` docstring; ledger ruling R11 (`standardize` lives in gp.py).

**Produces:**
```python
def standardize(y) -> tuple[np.ndarray, float, float]   # z = -(y - y.mean()) / y.std()  (numpy ddof 0), returns (z, mean, std)
def identify(objective, cell, n, seed, *, sobol_n=2048, **fit_kwargs) -> dict
```
`identify`: `X = scipy.stats.qmc.Sobol(objective.D, scramble=True, seed=seed).random(n)` inside `warnings.catch_warnings()` (the balance warning, as the reference suppresses it); `y = objective.observe(X, synthobj.families.noise_rng(seed, run=0))`; `z, y_mean, y_std = standardize(y)`; `key = jax.random.PRNGKey(int(numpy.random.SeedSequence([seed, n, 0x1D]).generate_state(1)[0]))`; `fitted = fit(X, z, key, cell, **fit_kwargs)`; `r = readouts(fitted, sobol_n=sobol_n)`; `checks = manipulation_checks(r, objective.labels)`. Returns a flat dict: `family` (`objective.labels.family`), `seed`, `cell` ("structure/prior"), `n`, `D`, `alpha` (the value used), `status`, `status_reason`, `nuts_attempts`, every `Diagnostics` field of the attempt used (prefixed `diag_`), `wall_s` (total), `y_mean`, `y_std`, `native_median`, `p_active`, `sobol_hat`, `total_var_hat`, `active_neutral`, `active_native`, `labels_s`, `labels_g`, `labels_active`, the `checks` entries, and `samples` (the retained dict as numpy). `sagp/gp.py` must not import `sagp.bo`.

**Tests:** `identify(make_family("aligned3", 0, D=6), ("additive","amplitude"), n=25, seed=0, nuts=NUTSConfig(32, 32, 4))` returns all keys above with the right shapes; `status in {"ok","refit","excluded"}`; the design is reproducible (`identify` twice → identical `samples`); `"sagp.bo" not in sys.modules` after `import sagp.gp` in a fresh subprocess (`sys.executable -c`).

**Commit:** `feat(sagp): offline identification runner`.

---

### Task 7: MAP references (DSP and oracle-S)

**Files:** Modify `sagp/gp.py`. Create `tests/test_sagp_map.py`.

**Read:** plan §2 `fit_map` docstring; BoTorch constants in `/opt/anaconda3/envs/saasbo/lib/python3.11/site-packages/botorch/models/utils/gpytorch_modules.py` (read-only: lengthscale prior `LogNormal(sqrt2 + log(D)/2, sqrt3)`, lengthscale floor 0.025, noise prior `LogNormal(-4, 1)`, noise floor 1e-4).

**Produces:**
```python
def _dsp_neg_log_joint(u, X, y, D_eff) -> float     # u = concat(log-ell params (D_eff,), noise param); ell = 0.025 + exp(u[:D_eff]); noise = 1e-4 + exp(u[-1]); K = kernel_product_lengthscale(X, X, {"kernel_var": 1.0, "kernel_inv_length_sq": ell**-2}, noise, True); -(log N(y | 0, K) + sum_i log LogNormal(ell_i; sqrt2 + log(D_eff)/2, sqrt3) + log LogNormal(noise; -4, 1)); no Jacobian
def fit_map(X, y, *, active=None, maxiter=500) -> FittedGP
```
`fit_map`: `active` is passed as an integer `np.ndarray` (convert with `np.asarray(active, dtype=int)` when not None) and `FittedGP.active`'s annotation is tightened to `np.ndarray | None` (carried from the Task 3 review); `Xa = X[:, active]` if active is not None; `D_eff = Xa.shape[1]`; start at the prior modes (`ell0 = exp(loc - 3)`, `noise0 = exp(-5)`, mapped through the floors); `scipy.optimize.minimize(jax.jit(jax.value_and_grad(f)), u0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter})`; return `FittedGP(cell="oracle_S" if active is not None else "dsp_map", X_train=X (full-D), Y_train=y, samples={"kernel_var": array([1.0]), "kernel_inv_length_sq": ell**-2 [None, :], "kernel_noise": array([noise])}, fixed_noise=None, active=active, status="ok", status_reason="", attempts=())`. Deterministic; no randomness.

**Tests (`tests/test_sagp_map.py`):** on `make_family("aligned3", 0, D=10)` with a 60-point Sobol design and noisy observations (`noise_rng(0)`), standardized: `fit_map` returns `posterior` shapes `(1, n_test)`; the median MAP lengthscale over `labels.S` is below the median over the other coordinates; `fit_map(X, z, active=np.array(labels.S))` works with full-D `X_test` and its `samples["kernel_inv_length_sq"].shape == (1, 3)`; the objective decreased from the start (`fun` at the optimum < at `u0`).

**Commit:** `feat(sagp): DSP-by-MAP and oracle-S references`.

---

### Task 8: The BO loop core

**Files:** Create `sagp/bo.py`, `tests/test_bo_loop.py` (the tests listed here; Task 9 adds the rest).

**Read:** plan §2 "bo.py" block (every signature and docstring), plan §3 "bo.py" block, plan §4 in full (the loop, the row schema), plan §5 D1 (`optimize_ei` row), D5, D7; `saasbo.py` lines 18–72 (`ei`, `ei_grad` are imported; `optimize_ei` is copied) and 142–188 (the loop skeleton); ledger rulings R6 and R11.

**Acquisition (decided by the user 2026-09-07: LogEI).** In `sagp/bo.py`:
- `log1mexp(x)`: log(1 − exp(x)) for x < 0, stable: `jnp.where(x > -log 2, jnp.log(-jnp.expm1(x)), jnp.log1p(-jnp.exp(x)))` (use the safe-branch idiom so neither branch produces NaN or NaN gradients on the other's inputs).
- `log_h(z)`: the stable log of h(z) = φ(z) + zΦ(z) (Ament et al. 2023, "Unexpected Improvements", the `log_h` helper). For z > −1: `jnp.log(φ(z) + z Φ(z))`; for z ≤ −1: `-z²/2 - c1 + log1mexp(jnp.log(-z) + log_ndtr(z) + z²/2 + c1)` with `c1 = log(2π)/2` and `log_ndtr = jax.scipy.special.log_ndtr` (this is the paper's erfcx form rewritten through the Mills ratio Φ(z)/φ(z), so it needs no erfcx). Combine the branches with `jnp.where` over inputs clamped to each branch's safe domain.
- `log_ei(x, y_target, gp, xi=0.0)`: the reference `saasbo.ei`'s signature and conventions — `mu, var = gp.posterior(x)`; `std = jnp.maximum(jnp.sqrt(var), 1e-6)`; `z = (y_target - xi - mu) / std`; per-sample `log(std) + log_h(z)`; `jnp.nan_to_num(..., nan=-jnp.inf)`; return `jax.scipy.special.logsumexp(values, axis=0) - jnp.log(values.shape[0])` (the exact log of the reference's sample-averaged EI). Also `log_ei_sum(x, y_target, gp, xi)` = `.sum()`, the L-BFGS-B objective analogue of `saasbo.ei_grad`.
- `ACQUISITIONS = {"logei": log_ei, "ei": saasbo.ei}`.

**Produces (in `sagp/bo.py`):** `IterRNG` (NamedTuple: `key`, `sobol_seed`, `jitter_rng`, `noise_rng`, `fallback_seed`), `iteration_rngs(seed, t)`, `initial_design(D, n_init, seed)`, `optimize_ei(gp, y_target, sobol_seed, jitter_rng, xi=0.0, num_restarts_ei=5, num_init=5000, acq=log_ei) -> (x_best, acq_best)` — a copy of `saasbo.optimize_ei` with exactly five changed lines: `qmc.Sobol(dim, scramble=True, seed=sobol_seed)`; `jitter_rng.standard_normal((1, dim))` in place of `np.random.randn(1, dim)`; `acq(X_rand, y_target, gp)` in place of `ei(X_rand, y_target, gp)` for candidate scoring; `value_and_grad(lambda x, yt, g, xi: acq(x, yt, g, xi).sum())` in place of `value_and_grad(ei_grad)` in the L-BFGS-B objective; and `return x_best, y_best`. With `acq=saasbo.ei` the function is the reference exactly. `RunConfig` (frozen dataclass with the fields in plan §2 including `acq: str = "logei"`, `nuts: NUTSConfig`, `thresholds: DiagThresholds`, `ell_prior: tuple`, `out_dir: str`), `config_hash(cfg)` (sha256 of the sorted JSON of the fields), `_fit_for`, `_propose` (passes `acq=ACQUISITIONS[cfg.acq]`; the logged `acq_value` is in that acquisition's units), `_iteration`, `RunLogger`, `run_bo(objective, method, seed, *, T=200, n_init=20, out_dir, resume=True, **overrides) -> Path`. Method dispatch: a `"structure/prior"` string → `gp.fit(..., cell=CELLS[key], alpha=cfg.alpha, fixed_noise=cfg.fixed_noise, nuts=cfg.nuts, thresholds=cfg.thresholds, ell_prior=cfg.ell_prior)`; `"dsp_map"` → `gp.fit_map(X, z)`; `"oracle_S"` → `gp.fit_map(X, z, active=np.array(objective.labels.S))`; `"sobol"` → no fit, the next row of the run's Sobol sequence. The run directory is `<out_dir>/<family>/<method with '/' -> '-'>/seed{seed:02d}/`. Follow plan §4 steps 1–7 exactly (including the `best_obs`/`best_f`/`regret` distinction, checkpoint-before-logs, per-iteration `readouts(fitted, compute_sobol=<every sobol_every and last>)`, `samples/t{t:03d}.npz` holding the retained samples plus `status`/`nuts_attempts`, and the minimal `manifest.json` with the `RunConfig` fields and `config_hash` — Task 9 extends it). Exception policy: wrap the fit in try/except; on exception log to `log.txt`, retry once with `jax.random.fold_in(rngs.key, 2)`; on a second exception propose `qmc.Sobol(D, scramble=True, seed=rngs.fallback_seed).random(1)[0]` with `status="excluded"`, `reason="exception: <type>: <msg>"`, `fit_calls=2`. `noiseless=True` uses `objective(x)` for `y`. CSV via the stdlib `csv` module; `coords.csv` columns `t, i, native_median, p_active, sobol_hat`. Resume: load `checkpoint.npz`, compare `config_hash`, truncate `iterations.csv`/`coords.csv`/`samples/` to rows with `t <= t_done`, continue. Method-specific columns: for `dsp_map`/`oracle_S` the diagnostics columns are NaN, `nuts_attempts` is 0, and the optimizer's `map_result` dict (`fun`, `fun0`, `nit`, `success`; read it with `getattr(fitted, "map_result", None)` — `FittedGP`s from `fit()` do not carry it) is written to `log.txt` per iteration; for `sobol` every model column is NaN and no `samples/` file is written.

**Tests (`tests/test_bo_loop.py`):**
- `test_log_h_matches_naive_and_stays_finite`: on 200 z values in [−6, 6], `|exp(log_h(z)) − (φ(z) + zΦ(z))| ≤ 1e-12 · max(1, h)`; at z ∈ {−10, −20, −40}, `log_h` is finite and decreasing and `jax.grad(log_h)` is finite; `log1mexp` matches `log(1 - exp(x))` to 1e-14 on x ∈ [−30, −1e-3].
- `test_log_ei_equals_log_of_reference_ei`: hand-made product/lengthscale `FittedGP` (S = 3, D = 4, n = 12) and 20 random x: `log_ei(x, y_target, gp)` vs `jnp.log(saasbo.ei(x, y_target, gp))` within 1e-9 wherever `ei > 1e-100`; with `y_target = mu.min() - 40 * std` the reference `saasbo.ei` returns exactly 0 while `log_ei` is finite and `jax.grad(lambda x: log_ei(x, ...).sum())` is finite and nonzero.
- `test_optimize_ei_reference_acq_is_within_bounds_and_improves`: `optimize_ei(gp, y_target, sobol_seed=1, jitter_rng=default_rng(1), acq=saasbo.ei)` returns x in [0,1]^D with an acquisition value ≥ the best of the 5000 candidates minus 1e-12; the same with the default `acq=log_ei`.
- determinism: `run_bo(make_family("aligned3", 0, D=5), "dsp_map", seed=1, T=15, n_init=5, out_dir=tmp_a)` and again into `tmp_b`: `X` from both checkpoints `np.array_equal`; the same for `"product/lengthscale"` with `nuts=NUTSConfig(32, 32, 4)`, `thresholds=DiagThresholds(float("inf"), 0.0, 10**9)`.
- resume: run T = 8 into `tmp`, then `run_bo(..., T=15, resume=True)` into the same `tmp`; `X` equals an uninterrupted T = 15 run bit-for-bit; changing any config field (e.g. `n_init`) on resume raises `ValueError`.
- rows: every `iterations.csv` row has all schema columns (plan §4), `regret` is non-increasing and ≥ 0, `best_obs >= best_f - 1e-12` is *not* required (they can differ either way) but `best_f == max objective(X[:t+1])`; `coords.csv` has `native_median`/`p_active` for every t ≥ n_init and finite `sobol_hat` exactly at multiples of `sobol_every` and at t = T − 1 (use `sobol_every=5`).
- partial write: after a run, delete the last two rows of `iterations.csv` and re-run with resume → no duplicate `t` values and the row count is T − n_init.

**Commit:** `feat(sagp): the BO loop — seeded reference acquisition, logging, checkpoint and resume`.

---

### Task 9: References, failure policy, manifest, CLI

**Files:** Modify `sagp/bo.py`; add to `tests/test_bo_loop.py`; create `tests/test_bo_config.py`, `tests/test_bo_cli.py`.

**Read:** plan §4 step 7 (manifest, resume warnings), plan §7 (CLI flags, run directory), plan §2 `RunConfig` docstring (the `reference_constants` block), ledger ruling R6 and R8; `synthobj/generate.py` `main` for the repo's argparse/dry-run style.

**Carried from the Task 8 review (small edits in `sagp/bo.py`, same commit):** (a) ruling R34 — per iteration, write `coords.csv` rows and `samples/t{t:03d}.npz` BEFORE appending the `iterations.csv` row (the checkpoint still goes first of all), so that a complete row implies complete artifacts and `last_logged_t` is a true completion marker; (b) ruling R35 — `_truncate_csv`/`last_logged_t` treat a final line without a trailing newline as torn and drop it (add a test that appends a truncated row without newline and checks resume redoes that iteration); (c) `log_ei`'s docstring: `jnp.nan_to_num(..., nan=-jnp.inf)` produces `finfo.min`, which behaves as −inf inside `logsumexp` — say so instead of "NaN → −inf"; (d) a one-line comment on `log1mexp`/`log_h` that the clamp idiom halves the gradient exactly at the branch points (measure zero, harmless); (e) any reader of `coords.csv` in this task treats the file's absence for `sobol` runs as legal.

**Produces:** full `write_manifest(cfg, objective, run_dir)`: the `RunConfig` fields, `config_hash`, `reference_constants` (`maxfun: 100`, `jitter_sd: 1e-3`, `xi: 0.0`, `num_init_candidates`, `num_restarts_ei`), `git` (`commit` from `git rev-parse HEAD`, `dirty` bool, "unknown" outside a repo), `versions` (python, jax, jaxlib, numpyro, numpy, scipy via `importlib.metadata`), `env` (`XLA_FLAGS`, `OMP_NUM_THREADS`, `platform.platform()`, `os.cpu_count()`), `reference_sha256` of `saasgp.py`, `saasbo.py`, `util.py`, `objective` (`family`, `seed`, `D`, `S`, `f_star`, `gamma`, `noise_sd`), `created`, `resumed` (list of `{time, commit, versions_changed}` appended on every resume; mismatches are warnings in `log.txt`, never errors); `environment.lock.txt` = sorted `name==version` for every installed distribution. `resolve_config(args) -> RunConfig` and `main(argv=None) -> int` with argparse: `--family` (required), `--seed` (required int), `--cell` (required; one of `METHODS`: the four `structure/prior` strings, `sobol`, `dsp_map`, `oracle_S`), `--T 200`, `--n-init 20`, `--D 100`, `--out` (required), `--acq` (choices `logei`, `ei`; default `logei`), `--alpha`, `--fixed-noise`, `--noiseless`, `--sobol-every 25`, `--no-resume`, `--nuts W,S,K` (three ints → `NUTSConfig`, tree depth stays 6), `--objective-dir` (load `SyntheticObjective.load(<dir>/<family>/seed{seed:02d}_D{D}.npz` stem) instead of `make_family(family, seed, D)`), `--dry-run` (print the resolved config as JSON and exit 0 without creating any file). Exit codes: 0 ok, 2 bad arguments.

**Tests:**
- `tests/test_bo_loop.py`: failure policy — monkeypatch `sagp.bo.gp.fit` to raise `RuntimeError` twice → the row has `status == "excluded"`, `fit_calls == 2`, `x` equals `qmc.Sobol(D, scramble=True, seed=iteration_rngs(seed, t).fallback_seed).random(1)[0]`; raising once → `fit_calls == 2` and status from the fit. Also `"sobol"` and `"oracle_S"` run end-to-end at D = 5, T = 12 (oracle via `fit_map`), and the sobol run's `X` equals `qmc.Sobol(5, scramble=True, seed=seed).random(12)`.
- `tests/test_bo_config.py`: for every method in `METHODS`, `resolve_config` with the same arguments differs only in `method`; `initial_design(5, 5, 3)` is identical across methods (trivially — assert the loop uses it: the first `n_init` rows of every run's `X` equal it); `manifest.json` round-trips into an equal `RunConfig` and contains the keys `git`, `versions`, `env`, `reference_sha256`, `objective`, `reference_constants`; `environment.lock.txt` contains `numpyro==0.21.0`.
- `tests/test_bo_cli.py`: `subprocess.run([sys.executable, "-m", "sagp.bo", "--family", "aligned3", "--D", "5", "--seed", "0", "--cell", "dsp_map", "--T", "8", "--n-init", "5", "--out", tmp, "--dry-run"])` exits 0, prints JSON containing `"method": "dsp_map"`, and creates nothing under `tmp`; a bad `--cell` exits 2; the end-to-end `python -m sagp.bo ... --cell dsp_map --T 15 --n-init 5` finishes in < 60 s and writes all six artifacts (`manifest.json`, `environment.lock.txt`, `iterations.csv`, `coords.csv`, `checkpoint.npz`, `samples/`).

**Commit:** `feat(sagp): references, failure policy, manifest and CLI`.

---

### Task 10: Reference-trajectory and slow tests

**Files:** Create `tests/test_bo_reference.py`; add the slow tests named in Tasks 3, 5 and here if not yet present (`test_sagp_inference.py` production-budget bit-for-bit, `test_sagp_readouts.py` D = 100 recovery and Gate 1).

**Read:** plan §6 "tests/test_bo_reference.py" bullet; `saasbo.py` lines 36–72 and 75–190 (what must be patched); `sagp/bo.py`'s `iteration_rngs` and `optimize_ei`.

**Carried from the Task 9 review (small edits in `sagp/bo.py` and `tests/test_bo_config.py`, same commit):** (a) ruling R36 — `RunLogger.note_resume` rewrites `manifest.json` atomically (write a temp file next to it, then `os.replace`), like `checkpoint`; (b) ruling R37 — `_truncate_csv` unlinks a file that holds no complete row (a torn header) so the next append rewrites the header, with a test that truncates `iterations.csv` mid-header and resumes; (c) the `--objective-dir` failure message names the missing stem path rather than blaming `--family`; (d) `test_bo_config.py`'s shared-initial-design test also runs `oracle_S` end-to-end (it is a MAP fit, as cheap as `dsp_map`).

**The reproduction test (slow):** `obj = make_family("aligned3", 0, D=5)`; seed s = 3; ours: `run_bo(obj, "product/lengthscale", seed=s, T=25, n_init=10, out_dir=tmp, acq="ei", fixed_noise=1e-6, noiseless=True, thresholds=DiagThresholds(float("inf"), 0.0, 10**9))` (`acq="ei"` because the reference has no LogEI; the production default is `logei`). Reference: `saasbo.run_saasbo(lambda x: -obj(x), np.zeros(5), np.ones(5), max_evals=25, num_init_evals=10, seed=s, kernel="matern")` with three patches active during the call: (i) `saasbo.SAASGP` replaced by a subclass whose `fit(self, X_train, Y_train, seed=0)` sets `self.rng_key_hmc, self.rng_key_predict = iteration_rngs(s, len(Y_train)).key, jax.random.PRNGKey(0)` (the iteration index t equals the number of points fitted on), `verbose=False` forced; (ii) `saasbo.qmc.Sobol` replaced by a wrapper that passes an explicit `seed` through unchanged (the initial design) and otherwise injects `seed=iteration_rngs(s, t).sobol_seed` with t taken from a counter starting at `n_init` and advanced once per call; (iii) `saasbo.np.random.randn` replaced by `lambda *shape: iteration_rngs(s, t).jitter_rng.standard_normal(shape)` using the same counter (advance the counter after the jitter draw, which happens after the Sobol call in `optimize_ei`). Assert `np.array_equal(X_ours, X_ref)` and `np.allclose(y_ours, -Y_ref)`. Document in the test docstring why each patch is needed (the only three places the reference is unseeded).

**Also (slow):** `test_bo_loop.py::test_oracle_beats_sobol_on_aligned3`: D = 20, T = 50, n_init = 10, seeds 0–2; median final regret of `oracle_S` < that of `sobol`.

**Also, carried from the Task 3 review (ledger ruling R20 and two minors), in `tests/test_sagp_inference.py`:** (i) remove the `@pytest.mark.slow` marker from the production-budget bit-for-bit test — it runs in under 4 s and `pytest.ini` defines slow as > 10 s, so reference equivalence at the real budget runs on every invocation; (ii) in `test_refit_and_excluded_paths` assert `keys[0]` equals the caller's key and `keys[1]` equals `jax.random.fold_in(key, 1)` (compare with `np.array_equal`), not merely that they differ; (iii) add `test_fit_with_fixed_noise`: `fit(..., fixed_noise=1e-6, nuts=NUTSConfig(32, 32, 4))` on the product/lengthscale cell returns samples without a `kernel_noise` key and `fitted.fixed_noise == 1e-6`.

**Commit:** `test(sagp): reference-trajectory reproduction and slow recovery tests`.

---

### Task 11: Pilot and docs

**Files:** Create `scripts/pilot_sagp.py`; modify `README.md` (a "Sparse additive GP cells (`sagp`)" section); append a "Pilot results (date)" paragraph to `docs/superpowers/plans/2026-09-07-sagp-model-and-bo.md` §5 D8 and D4. Never edit files under `Research Context/` (ledger ruling R8).

**Pilot script (`scripts/pilot_sagp.py`, re-scoped by ledger ruling R26 after the cost measurement — no fit longer than ~15 minutes anywhere in it; every stage prints its own timing line as it finishes so a partial run is still informative):**
1. `--stage grad` (≈ 2 min): for every cell and n ∈ {50, 100, 200} at D = 100, time one jitted gradient of the log-joint (`jax.grad` of `numpyro.infer.util.log_density` on random standardized data, 5 timed calls after one compile call) and print per-gradient ms plus the fit estimate `768 × steps × t_grad` for steps ∈ {40, 63}. The controller's 2026-09-07 measurement to reproduce: product/lengthscale 0.4/0.5/3.1 ms, additive/amplitude 7.6/20/45 ms, additive/lengthscale 6.6/56/173 ms, product/amplitude 7.3/29/135 ms at n = 50/100/200.
2. `--stage fit` (≈ 15–25 min): two full fits with the production `NUTSConfig()` on `make_family("aligned10", 0)` (Sobol design, noisy observations, standardized): product/lengthscale at n = 200 and additive/amplitude at n = 100; record `wall_s`, compile time (time the first `fit` minus the second at the same n — run each twice), `num_steps_mean`, `status`, `r_hat_max`, `n_eff_min`, `divergences`; then time `optimize_ei` once on each fitted model with the production 5000 candidates / 5 restarts.
3. `--stage refit` (≈ 10–20 min): ten fits of product/lengthscale at n = 100 with keys `fold_in(PRNGKey(0), k)`, k = 0…9, and ten of additive/amplitude at n = 50; report the refit and exclusion rates and the distribution of `r_hat_max`/`n_eff_min` (the D4 false-failure datum).
4. `--stage alpha` (seconds): the D2 α simulation table (10⁴ draws, D = 100; reference count at ρ_ε versus the amplitude count at α = ALPHA_AMPLITUDE: medians and quartiles).
5. `--stage all` runs 1–4 and writes `docs/superpowers/plans/2026-09-07-sagp-pilot.md` with one markdown table per stage and the run-cost formula `62 × t_fit × (1 + 1.67 r) + 180 × t_acq` evaluated per cell in core-hours, using stage 1's per-gradient times scaled by stage 2's measured steps/iteration. The controller runs `--stage all` in the background with `caffeinate`; the implementer runs only `--stage grad` and `--stage alpha` (fast) and one `--stage fit` fit at reduced size (n = 50) to prove the script works end to end.

**Plan updates (`docs/superpowers/plans/2026-09-07-sagp-model-and-bo.md`):** append a "Pilot results (2026-09-08)" paragraph under §5 D8 that points to the pilot file and states the headline: the reference cell matches D8, the three centered cells cost 15–55× more per gradient (one exponential per coordinate per pair, and two of them unfused), so a T = 200, D = 100 run of a centered cell is 40–200 core-hours at the full budget; list the options (identical fallback 128/128/8; hoisting the centering terms and fixing fusion; smaller SQ2; more cores) as the user's decision. Under D4 append the measured refit rate. Under "## 10." replace the open-question list's status with one line per question saying which ruling closed it (R2/α closed form and the D2 simulation; exact Sobol in product/amplitude — implemented; diagnostic gate 1.1/16/5 — implemented, refit rate from the pilot; placement — fit_map in gp.py, make_family on the fly; the one-line compat patch — applied).

**README section:** what `sagp` is, the four cells and the three references, the CLI with the plan §7 example and SLURM snippet (add `export OMP_NUM_THREADS=1` and the `caffeinate` note for laptop runs), the settings that supersede `Design-brief.md` §2 (512/256/16, fresh chains, LogEI as decided on 2026-09-07, the reference optimizer), α = 0.0131 with the one-line rule, where outputs land (`iterations.csv`, `coords.csv`, `samples/`, `manifest.json`, `environment.lock.txt`), how to resume (re-run the same command), the vendored-reference gotchas (retained samples must be a multiple of 8 for `SAASGP.posterior`; `jnp.clip` compat line), and the cost table with its consequence. Never edit files under `Research Context/`.

**Commit:** `docs(sagp): pilot script, pilot timings, README section`.
