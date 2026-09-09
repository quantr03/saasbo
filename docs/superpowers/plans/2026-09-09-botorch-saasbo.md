# BoTorch SAASBO Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the vendored `saasgp.py`/`saasbo.py`/`util.py` code path under `sagp/` with BoTorch 0.18.1's SAASBO — `PyroModel` subclasses fit by a line-for-line copy of `fit_fully_bayesian_model_nuts`, predicting through `SaasFullyBayesianSingleTaskGP.posterior`, analytic `LogExpectedImprovement` averaged over the draws, `optimize_acqf` with RAASP — on Papenmeier et al. (2025)'s loop protocol, with the four cells, `run_bo`'s adapters, the readouts and `experiments/` unchanged in meaning.

**Architecture:** The four cells become four `PyroModel` subclasses whose `sample()` is BoTorch's SAAS model with only the kernel/prior block swapped (the product/lengthscale one inherits `SaasPyroModel.sample` untouched); one `CellGP(SaasFullyBayesianSingleTaskGP)` binds a cell's `PyroModel` per instance. NUTS stays JAX/NumPyro (dense mass, 512/256/16, one fresh chain per fit, seeded per iteration); the un-thinned chain still feeds `diagnose`. Prediction moves to GPyTorch: three new batched `Kernel` subclasses implement the centered kernels in torch so `optimize_acqf` can differentiate through them; the JAX kernels stay as the log-density side and as the torch kernels' test oracle. `FittedGP` keeps its interface (`posterior(X) -> (S, n_test)` mean and noisy variance, `alphas()`, `samples`) on top of the BoTorch model, so `sagp.readouts` and `experiments/` change only in site names and config fields.

**Tech Stack:** Python 3.11 at `/opt/anaconda3/envs/saasbo/bin/python`; torch 2.14.0 (float64, CPU), gpytorch 1.15.2, linear_operator 0.6.1, botorch 0.18.1, jax/jaxlib 0.10.2, numpyro 0.21.0, numpy 2.4.6, scipy 1.17.1; pytest (`pytest.ini` sets `pythonpath = .`).

**Spec:** `docs/superpowers/specs/2026-09-09-botorch-saasbo-brief.md` (committed with this plan in Task 0). Approved 2026-09-09 via plan mode; the five open questions in §9 stand by recommendation.

## Global Constraints

- Branch `feat/botorch-saasbo` from the tip of `feat/sagp-model-bo`, which is **02cfb46** (the brief says c8291a5; two docs commits landed since). Ledger at `.superpowers/sdd/2026-09-09-botorch-saasbo/progress.md`.
- Shared checkout (18 peer sessions were live on 2026-09-09 16:20): `git branch --show-current` immediately before every commit; stage by path only; never `git add -A`/`.`; never `checkout`/`switch`/`stash`/`worktree` except the one HEAD-only `git switch -c` of Task 0 (open question 3); never touch `Research Context/`, the untracked `CLAUDE.md`, `.codegraph/`, `.DS_Store`, or the four unstaged demo-file deletions (`Branin100.ipynb`, `hartmann.py`, `saasbo_demo.py`, `saasgp_demo.py`). There are no uncommitted docstring edits in the tree today (the brief's note is stale).
- `python` is `/opt/anaconda3/envs/saasbo/bin/python`; ad-hoc scripts with `PYTHONPATH=.`; slow tests one file at a time, never in an implementer's foreground (`nohup … &` then poll).
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`.
- Out of scope: Ax, qLogNEI, a refit cadence, q > 1, input warping, the thesis appendix worktree, real benchmarks.
- Layering: nothing under `sagp/` imports `experiments`; `experiments/` uses only public `sagp` names; `sagp.gp` imports nothing from `synthobj`; `synthobj` stays torch-free except `botorch_adapter`; every `sagp/` module has a row in `tests/test_layering.py::_ALLOWED_SAGP_IMPORTS`.
- NUTS budget 512/256/16, tree depth 6, one chain — unchanged (Quan, 2026-09-09).

## Context

The 2026-09-07 brief pinned "inference must match the SAASBO reference implementation" (github.com/martinjankowiak/saasbo @ 4af4a8e), so `sagp/` today imports `saasgp`, `saasbo` and `util` and reproduces `run_saasbo` bit for bit. Quan decided on 2026-09-09 to run the study on BoTorch's SAASBO instead, on Papenmeier et al. (2025)'s loop protocol, so that "SAASBO" in the thesis is the implementation the field benchmarks against rather than a research prototype. Verified during planning (2026-09-09 16:15–16:45):

- Neither `hvarfner/vanilla_bo_in_highdim` nor `lpapenme/understanding-hdbo-code` contains any SAASBO code (`grep -ri saas` is empty in both); Papenmeier's original harness `LeoIV/sample-from-high-dim-GP` no longer exists on GitHub (404, no fork, no Wayback snapshot). Papenmeier's SAASBO is therefore specified by prose only (§4 p. 7: run through BoTorch, 500 iterations, 72 h cap) and BoTorch's defaults are the executable scheme — exactly the brief's fallback rule. His loop code (`bo.py`) is the protocol: standardize every iteration, fresh model every iteration, analytic `LogExpectedImprovement(best_f=y_norm.max())`, `optimize_acqf(q=1, num_restarts=5, raw_samples=512, options={"batch_limit": 1, "sample_around_best": True, "sample_around_best_sigma": 0.001})`, everything inside `gpytorch.settings.cholesky_max_tries(9)`, a uniform random point on any exception; `torch.manual_seed` only in the test harness.
- BoTorch 0.18.1 as installed: `fit_fully_bayesian_model_nuts` is 26 lines (`fit.py:375-400`): `model.train()`, `NUTS(model.pyro_model.sample, dense_mass=True, max_tree_depth)`, `MCMC(num_warmup, num_samples, progress_bar)`, `mcmc.run(jax.random.PRNGKey(seed))`, `postprocess_mcmc_samples`, `[::thinning]`, `load_mcmc_samples`, `eval()`; no `extra_fields`, the `MCMC` object discarded, `jit_compile` dead. `AbstractFullyBayesianSingleTaskGP.__init__` instantiates `self._pyro_model_class(use_input_warping, indices_to_warp)` (`fully_bayesian.py:920`), clamps `train_Yvar` at `MIN_INFERRED_NOISE_LEVEL = 1e-4` **before** `set_inputs` (`:910`), and calls `validate_input_scaling`, whose standardization check uses torch's ddof-1 std (warns for n < 52 on ddof-0 targets). `SaasPyroModel` sites in order: `outputscale ~ Gamma(2, 0.15)`, `mean ~ N(0, 1)`, `noise ~ Gamma(0.9, 10)` (+1e-4 added to the returned value; only when `train_Yvar` is None), `kernel_tausq ~ HC(0.1)`, `_kernel_inv_length_sq ~ HC(1)`, deterministic `kernel_inv_length_sq`, deterministic `lengthscale`; `Y ~ MVN(mean, K + noise·I)` with **no jitter**; `matern52_kernel` floors squared distances at 1e-30. `postprocess_mcmc_samples` deletes `kernel_tausq` and `_kernel_inv_length_sq`. `load_mcmc_samples` (`:538-611`) returns `(ConstantMean, ScaleKernel(MaternKernel(nu=2.5, ard)), GaussianLikelihood(noise ≥ 1e-4, noise = draw.clamp_min(1e-4)) | FixedNoiseGaussianLikelihood, None)`, all batched over S. `posterior(X, observation_noise)` unsqueezes `MCMC_DIM = -3` and returns a `GaussianMixturePosterior`. `LogExpectedImprovement.forward` is decorated `@average_over_ensemble_models`, which reduces the S dimension by `logmeanexp`; `_mean_and_sigma` uses the noise-free posterior and clamps the variance at 1e-12 (σ ≥ 1e-6). `optimize_acqf`: `batch_limit` defaults to `num_restarts`; `gen_batch_initial_conditions` draws `raw_samples` Sobol points seeded by `options["seed"]`, and with `sample_around_best` appends `raw_samples` more: half truncated-normal perturbations (σ) of random picks among the top 5 % of the training points, half perturbing each coordinate with probability `min(20/d, 1)` — every one of those draws (`torch.randint`, `torch.rand`, an unseeded `SobolEngine`, `torch.randperm`, the Boltzmann `torch.multinomial`) uses torch's **global** RNG; `init_batch_limit` defaults to `batch_limit` (`initializers.py:339-341`); `retry_on_optimization_warning` regenerates with the same seed. `SingleTaskGP` defaults: `outcome_transform=Standardize` and a bare RBF kernel with the DSP prior — both must be overridden for Papenmeier's model. `get_gaussian_likelihood_with_gamma_prior()` = `Gamma(1.1, 0.05)`, `GreaterThan(1e-4)`, initial value at the prior mode. `DefaultPredictionStrategy.mean_cache` is `K̂⁻¹(y − m)` per batch (`exact_prediction_strategies.py:276-283`), detached. BoTorch never enables `jax_enable_x64`; `sagp/__init__.py` does, process-wide.
- Repo today (02cfb46): `sagp/gp.py` 783 lines, `diagnostics.py` 156, `bo.py` 388, `references.py` 195, `readouts.py` 303; `experiments/run_bo.py` 403, `runlog.py` 610, `identify.py` 106. `fit` still has the doubled-warm-up refit; the ledger records Quan's "no refit" decision (2026-09-09 01:53) and its withdrawal (02:27, "pending Quan's look at how the MSR method refits"); the brief (15:56) reinstates one budget per fit (open question 1). `readouts` reads `samples["a_sq" | "kernel_ell" | "kernel_inv_length_sq" | "kernel_var"]`, `noises()`, `alphas()`, `posterior()`, `columns()`; nothing in it depends on the sign of the targets (every sign-sensitive quantity enters squared or as a ratio) and exactly one thing depends on ddof 0: `shares_from_amplitudes`' `1 − σ²` correction. The run directory stores `y`/`f`, never `z`. Three tests pin the manifest key sets and the vendored files' SHA-256 (`tests/test_bo_config.py:158-201`); one pins the row schema literally (`tests/test_run_bo.py:143-153`).

## 1. The twelve decisions

**D1 — Where the four cells live.** Four `PyroModel` subclasses in `sagp/gp.py` and **one** GP class `CellGP(SaasFullyBayesianSingleTaskGP)` parameterized by `Cell`. `ProductLengthscalePyroModel(SaasPyroModel)` overrides only `sample_lengthscale` (to pass `self.alpha`) and `postprocess_mcmc_samples` (BoTorch's minus the two `del`s), so `sample()` and the NUTS trajectory are BoTorch's to the bit. `AdditiveLengthscalePyroModel(ProductLengthscalePyroModel)` overrides `sample()` (same sites, same order, the additive JAX kernel) and `load_mcmc_samples` (a `ScaleKernel(CenteredAdditiveLengthscaleKernel)`). `AdditiveAmplitudePyroModel(PyroModel)` / `ProductAmplitudePyroModel` sample `mean, noise, kernel_tausq, _a_sq, a_sq (det), kernel_ell` and load the amplitude torch kernels. One GP class because the GP is framework glue — it sets `self._pyro_model_class = cell.pyro_model` as an instance attribute before `super().__init__` (that attribute is read on `self` at `fully_bayesian.py:920`), then sets `pyro_model.alpha`/`ell_prior`; four GP classes would repeat that plumbing and put cell knowledge in two registries. `Cell` stays the registry.

**D2 — Priors.** Adopt BoTorch's where the vendored code differed: constant mean `N(0, 1)`, outputscale `Gamma(2, 0.15)` (lengthscale cells only), noise `1e-4 + Gamma(0.9, 10)`, α = 0.1 on ρ. Amplitude cells keep `a_sq_i = tausq · lam_i` at `ALPHA_AMPLITUDE`, no outputscale, `kernel_ell ~ LogNormal(0, 1.5)`, and take the mean and noise priors. `fixed_noise=v` maps to `train_Yvar = full((n, 1), v)`; BoTorch clamps it at 1e-4 in `__init__` before the sampler sees it, so sampler and predictor agree at `max(v, 1e-4)`; `fit` therefore rejects `0 < v < 1e-4` with a `ValueError` naming the floor, and the noiseless out-of-family checks use `fixed_noise=1e-4` (the reference's 1e-6 is unreachable; recorded in README). *α survives:* the prior-predictive active count `#{i : θ_i > c}` is a function of `(tausq, lam)` alone, and both cells keep `θ_i = tausq · lam_i`, `tausq ~ HC(α)`, `lam_i ~ HC(1)`; the mean, outputscale and noise priors do not enter it, so `ALPHA_AMPLITUDE = 0.1 · ACTIVE_EPS / RHO_EPS` and `test_alpha_matches_reference_count` stand unchanged. *`shares_from_amplitudes` survives:* the constant mean is a location common to every design point, so the sample variance of z (= 1 exactly at ddof 0) is still `Σ a_i² + σ²` up to design sampling error in the additive/amplitude cell; `s_i = a_i²/(1 − σ²)` is unchanged, and `σ²` is now the loaded likelihood's `max(noise draw, 1e-4)` (what prediction uses).

**D3 — Sampler.** `sagp.gp._run_nuts(model, nuts, seed) -> (MCMC, wall_s)` is `fit.py:375-389` copied with two additions — `progress_bar=False` and `mcmc.run(PRNGKey(seed), extra_fields=("diverging", "num_steps"))` — and returns the sampler; `fit` then does the postprocess/thin/`load_mcmc_samples`/`eval()` lines itself. `dense_mass=True` adopted (the one sampler difference from the vendored path). Seeds: `IterRNG.nuts_seed` (a 32-bit int from `SeedSequence([seed, t, 0x4E])`) → `PRNGKey(nuts_seed)`; the exception retry uses `retry_seed` (salt 0x2E). Pinned by test: our copy's loaded hyperparameters equal `fit_fully_bayesian_model_nuts(seed=s)`'s bit for bit. `NUTSConfig` unchanged as the single budget object; `num_chains` must be 1. One NUTS run per fit; no diagnostic refit, no cadence knob. Precision: `sagp/__init__.py` keeps `enable_x64`, so NUTS runs in float64 here where BoTorch as shipped runs JAX in float32 — the same scheme at the study's precision; the pin test runs both sides in the same process, so it holds (open question 4).

**D4 — Diagnostics.** Gate unchanged: `diagnose` on the un-thinned `mcmc.get_samples()` and `get_extra_fields()`. Site vocabulary becomes BoTorch's (rename in `diagnostics`, `readouts`, `FittedGP`, npz): `_POSITIVE_SAMPLED_SITES = ("outputscale", "noise", "kernel_tausq", "_kernel_inv_length_sq", "_a_sq", "kernel_ell")`; the real-valued sampled site `mean` joins the `global` group on its natural scale (open question 5). `FittedGP.samples` keeps every site incl. `kernel_tausq`/`_kernel_inv_length_sq` (own `postprocess_mcmc_samples`) plus the deterministic `lengthscale`. With one attempt, `status ∈ {"ok", "excluded"}`, `attempts` has length 1 (0 for MAP); the `a0_*` row columns and npz scalars (ruling R43, refit-only) are dropped; `nuts_attempts` stays. npz versioned: `schema_version = 2` scalar added to every `samples/t*.npz`.

**D5 — Prediction.** `FittedGP.posterior(X_test, observation_noise=True) -> (mean, var)` of shape `(S, n_test)` as jnp float64 arrays, computed as `model.posterior(torch(X_test)[:, None, :], observation_noise=...)` — per-point batches, so no `(S, n_test, n_test)` joint covariance is ever formed — then `mean.squeeze((-1, -2)).T`. `alphas()` = `model.prediction_strategy.mean_cache` `(S, n)` = `K̂_s⁻¹(z − c_s)` after one warm-up `posterior` call (the strategy is created lazily); `means()` = `samples["mean"]` `(S,)` is new, the intercept the readouts' component sums now need. Torch kernels (`sagp/kernels_torch.py`): a `_CenteredKernel(gpytorch.kernels.Kernel)` base that loops over coordinates — `r = |x1[..., :, None, i] − x2[..., None, :, i]| / ell[..., 0, i]`, `kbar_i = k − m_i(x1) − m_i(x2) + (1 − v_i)`, 64-node Gauss–Legendre `m_i`, `v_i` in torch — accumulating a sum or a product in `(…, n, m)`, `diag=True` returning `(…, n)`; parameters registered with `batch_shape=(S,)`, shape `(S, 1, D)`, `Positive()` constraints, so GPyTorch broadcasts them exactly as it does `MaternKernel.lengthscale`. Subclasses: `CenteredAdditiveAmplitudeKernel` (`a_sq`, `ell`; normalized), `CenteredAdditiveLengthscaleKernel` (`lengthscale`; unnormalized; inside a `ScaleKernel`), `CenteredProductAmplitudeKernel`. Product/lengthscale uses BoTorch's own `ScaleKernel(MaternKernel)`. Memory at n = 200, D = 100, S = 16: per-coordinate temporaries `(b, S, n_test, n)` — 13 MB at b·n_test = 512 (raw-sample evaluation under `no_grad`), 26 KB at the single L-BFGS-B point (autograd graph ≈ D × that ≈ 3 MB), 52 MB at the readouts' 2048-row QMC chunks; the `(S, n, n)` train cache 5 MB; peak well under 200 MB. `_CHUNK_THRESHOLD`/`_CHUNK_ROWS`/`chunk_vmap` go away.

**D6 — Acquisition.** `LogExpectedImprovement(fitted.model, best_f=max z)`, q = 1, `maximize=True`; on the ensemble it is `logmeanexp` of per-draw analytic log EI (verified: `@average_over_ensemble_models` with `_log = True`). Two recorded differences from today's `log_ei`: (i) it uses the **noise-free** posterior variance (`_mean_and_sigma` calls `posterior(X)` without observation noise) where the vendored EI used the noisy predictive variance — BoTorch's convention is the scheme; (ii) the σ floor is the same 1e-6. `standardize` stops negating; `run_bo` uses `best_f = float(np.max(z))`. The sign flip reaches: `gp.standardize` (the line and three docstrings), `bo.run_bo` (`y_target → best_f`), `propose_ei`/`propose_sobol` parameter names, `identify.py`'s docstring, `tests/test_sagp_inference.py::test_standardize…` (asserts the flip), `tests/test_sagp_readouts.py:387` (hand-negates); not the run directory (`y`, `f`, `best_obs`, `best_f`, `regret` are raw), not the checkpoint, not `references` (the MLL depends on z through `zᵀK⁻¹z`), not the readouts. `log_h`/`log1mexp`/`log_ei` leave `sagp.bo`; the old formula lives on only inside the cross-check test.

**D7 — Acquisition maximizer.** `propose_ei` calls `optimize_acqf(acq, bounds=[[0]*D, [1]*D], q=1, num_restarts=5, raw_samples=512, options={"batch_limit": 1, "maxiter": 200, "sample_around_best": True, "sample_around_best_sigma": 1e-3, "seed": rngs.sobol_seed})` inside `torch.random.fork_rng()` + `torch.manual_seed(rngs.torch_seed)` (the perturbations, the unseeded perturbation Sobol and the Boltzmann start selection all draw from the global RNG; `options["seed"]` reaches only the main Sobol draw) and `gpytorch.settings.cholesky_max_tries(9)` (Papenmeier). `batch_limit=1` is not only speed: BoTorch batches the 5 restarts into one L-BFGS-B problem otherwise, whose joint Hessian approximation and stopping rule couple them — different iterates, not just different timing — so 1 is both Papenmeier's value and the faithful one; `init_batch_limit` is left at its default (= `batch_limit`, so raw candidates are scored one at a time; ≈ 1024 single-point posteriors per iteration, cheap). `maxiter=200` is BoTorch's default and what Papenmeier's code runs (his App. B.2 says 2000; the code sets nothing — recorded). `prob_perturb` default `min(20/D, 1)`. `RunConfig` gains `raw_samples=512`, `num_restarts=5`, `sample_around_best_sigma=1e-3`, `batch_limit=1`, `maxiter=200` (all hashed, all CLI flags) and drops `num_init_candidates`, `num_restarts_ei`, `acq`. Failure policy: the fit keeps its one retry then the seeded Sobol point and an `excluded` row; **new, flagged:** an exception from `propose` (BoTorch's optimizer can raise on a Cholesky failure) is logged and takes the same seeded Sobol fallback with status `excluded` instead of killing the run — Papenmeier's loop does the same with a uniform point.

**D8 — MAP references.** `sagp/references.py` becomes Papenmeier's `build_model(method="dsp", noiseless=False)` + `fit_mll` transcribed: `MaternKernel(nu=2.5, ard_num_dims=D_eff, lengthscale_prior=LogNormalPrior(SQRT2 + log(D_eff)/2, SQRT3))` in `ScaleKernel(outputscale_prior=GammaPrior(2.0, 0.15))`, lengthscale initialized at the prior mode `exp(loc − 3)`, `get_gaussian_likelihood_with_gamma_prior()` (Gamma(1.1, 0.05), floor 1e-4), `SingleTaskGP(X, z[:, None], covar_module, likelihood, outcome_transform=None, input_transform=FilterFeatures(active) if active is not None else None)`; fit by `fit_gpytorch_mll` under `cholesky_max_tries(9)` with the Adam fallback (500 steps, lr 0.01) and the `NotPSDError` give-up. Kernel family settled as Matérn-5/2 (passed explicitly; the class default is RBF with `Standardize`). Prior changes from today's transcription (no outputscale, LogNormal(−4, 1) noise, 2.5e-2 lengthscale floor): recorded; `tests/test_sagp_references.py`'s three prior tests re-targeted to a from-scratch construction. Location constant: `SQRT2` (the brief's, the paper's, BoTorch's) rather than Papenmeier's float32 `1.41` (open question 2). Status: `ok` after `fit_gpytorch_mll`; `ok` with reason `"fit_gpytorch_mll failed; Adam fallback"` after the fallback; `excluded` with the `NotPSDError` message if that fails too. `map_result = {"mll": float, "fallback": "none"|"adam"|"failed"}`. Oracle: the same model behind `FilterFeatures(torch.as_tensor(active, dtype=int64))`, so `optimize_acqf` searches `[0, 1]^D` and `FittedGP.active`/`columns` keep serving the readouts.

**D9 — The loop.** `run_bo` and its adapters stay: `surrogate(X, z, seed) -> FittedGP` (the third argument is now an int, the shape unchanged), `propose(fitted, best_f, rngs, t) -> (x, value)`. `initial_design` stays on scipy's scrambled Sobol. Standardization ddof 0 kept (the share mapping assumes unit sample variance exactly; the `(n − 1)/n` factor is no part of the scheme) — recorded as a deviation from Papenmeier's `torch.std()`; BoTorch's `validate_input_scaling` would warn on it for n < 52, so model construction runs under `botorch.settings.validate_input_scaling(False)`. Float64: `torch.set_default_dtype(torch.float64)` in `sagp/__init__.py` beside `enable_x64`. Threads: the sbatch lines keep `OMP_NUM_THREADS=1 XLA_FLAGS=…`; torch honours `OMP_NUM_THREADS` at start-up and `sagp/__init__.py` additionally calls `torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))` when it is set; the manifest's `env` gains `torch_num_threads`.

**D10 — The vendored reference and its tests.** Delete `saasgp.py`, `saasbo.py`, `util.py`, `tests/test_sagp_env.py`'s two vendored tests, the three `bit_for_bit` tests and every `chunk_vmap` use, in the final commit after the replacement reference test is green. No `--engine` switch. `requirements-sagp.txt` gains `torch==2.14.0`, `gpytorch==1.15.2`, `linear_operator==0.6.1`, `botorch==0.18.1`; `tests/test_sagp_env.py` checks them. `setup.py`'s stale `install_requires` is left alone (not asked).

**D11 — Layering.** `_ALLOWED_SAGP_IMPORTS` gains `"kernels_torch": set()` and `gp` → `{"sagp.diagnostics", "sagp.kernels_torch"}`; a fourth static rule: no `synthobj/*.py` except `botorch_adapter.py` imports `torch`/`botorch`/`gpytorch`/`linear_operator` (one `ast` walk, ≈ 15 lines, beside the existing subprocess guard in `tests/test_botorch_adapter.py`).

**D12 — Cost.** NUTS gradients are the same JAX log-density, so per-gradient cost is the 2026-09-08 pilot's (4.3 / 45 / 198 / 136 ms at n = 200 for product-lengthscale / additive-amplitude / additive-lengthscale / product-amplitude); dense mass changes the leapfrog count, which the pilot measures. Acquisition per iteration: ≈ 1024 single-point posteriors + 5 L-BFGS-B runs of ≤ 200 iterations, each a gradient through the torch kernel at one point — seconds, not the 59 s the 5000-candidate JAX path cost the additive/amplitude cell. Pilot task (Task 11) before any study run. Cost table in §6.

## 2. Public interface after the migration

```python
# sagp/kernels_torch.py (new)
GL_NODES_T, GL_WEIGHTS_T          # torch float64, leggauss(64) on [0,1]; pinned array_equal to gp.GL_NODES
def v_of_ell_t(ell: Tensor) -> Tensor                         # (..., D) -> (..., D), = gp.v_of_ell
def kbar_1d(x1: Tensor, x2: Tensor, ell: Tensor, normalize: bool) -> Tensor
    """Centered Matern-5/2 kernel of one coordinate; x1 (..., n), x2 (..., m), ell (..., 1, 1) -> (..., n, m)."""
class CenteredAdditiveAmplitudeKernel(Kernel):   # params a_sq, ell : (S, 1, D); k = sum_i a_sq_i kbar_i (normalized)
class CenteredAdditiveLengthscaleKernel(Kernel): # param lengthscale : (S, 1, D); k = sum_i k~_i (unnormalized); wrap in ScaleKernel
class CenteredProductAmplitudeKernel(Kernel):    # params a_sq, ell; k = prod_i (1 + a_sq_i kbar_i)
    # each: __init__(ard_num_dims, batch_shape); forward(x1, x2, diag=False, **params) -> (..., n, m) | (..., n)

# sagp/gp.py
GL_NODES, GL_WEIGHTS, ACTIVE_EPS, ALPHA_LENGTHSCALE, ELL_EPS, RHO_EPS, ALPHA_AMPLITUDE, ELL_PRIOR   ✓ unchanged
matern52_1d, v_of_ell, kbar_all, centered_matern52_1d                                             ✓ unchanged
def kernel_additive_amplitude(X, Z, params) -> Array      # (n, m), noise-free: BoTorch's sample_observations adds noise;
def kernel_additive_lengthscale(X, Z, params) -> Array    #   params keys: "a_sq","kernel_ell" | "outputscale","kernel_inv_length_sq"
def kernel_product_amplitude(X, Z, params) -> Array
def kernel_product_lengthscale(X, Z, params) -> Array     # outputscale * Matern-5/2 on sqrt(max(sum((x-z)^2/l^2), 1e-30)): BoTorch's formula
KERNELS, cell_kernel_diag                                 ✓ (signatures follow the above)
class ProductLengthscalePyroModel(SaasPyroModel)          # alpha attr; sample() inherited verbatim; postprocess keeps every site
class AdditiveLengthscalePyroModel(ProductLengthscalePyroModel)
class AdditiveAmplitudePyroModel(PyroModel)               # alpha, ell_prior attrs
class ProductAmplitudePyroModel(AdditiveAmplitudePyroModel)
@dataclass(frozen=True) class Cell: structure, prior, pyro_model: type[PyroModel], kernel, kernel_diag, native_site, sites, alpha_default; .key
CELLS: dict[CellKey, Cell]                                # sites: lengthscale ("outputscale","mean","noise","kernel_tausq","_kernel_inv_length_sq","kernel_inv_length_sq","lengthscale"); amplitude ("mean","noise","kernel_tausq","_a_sq","a_sq","kernel_ell")
class NUTSConfig ✓
class CellGP(SaasFullyBayesianSingleTaskGP):
    def __init__(self, train_X: Tensor, train_Y: Tensor, train_Yvar: Tensor | None, *, cell: Cell, alpha: float, ell_prior: tuple[float, float]) -> None:
        """BoTorch's SAAS GP with `cell.pyro_model` bound: sets `self._pyro_model_class = cell.pyro_model` before
        `super().__init__` (read on the instance at fully_bayesian.py:920), then `self.pyro_model.alpha/ell_prior`."""
class FittedGP:
    def __init__(self, cell, X_train, Y_train, samples, fixed_noise, active, status, status_reason, attempts, model) -> None:
        """`model` is the fitted, eval-mode BoTorch model (CellGP, or SingleTaskGP for the MAP references);
        `samples` the retained draws keyed by site, leading dim S (S = 1 for MAP), as jnp float64."""
    def posterior(self, X_test, observation_noise: bool = True) -> tuple[Array, Array]   ✓ shapes; noisy by default
    def alphas(self) -> Array          # (S, n): K_s^-1 (z - c_s) from prediction_strategy.mean_cache
    def means(self) -> Array           # (S,): the constant-mean draws  (new)
    def noises(self) -> Array          # (S,): loaded likelihood noise (max(draw, 1e-4)) or the clamped fixed value
    def params(self, s), param_sites(), columns(X)   ✓  (lengthscale cells: ("outputscale", "kernel_inv_length_sq"))
def fit(X, y, seed: int, cell, *, alpha=None, fixed_noise=None, nuts=NUTSConfig(), thresholds=DiagThresholds(), ell_prior=ELL_PRIOR) -> FittedGP
    """One NUTS run of `nuts` on `seed`; `status` "ok"/"excluded" by `diagnose`, draws returned either way.
    fixed_noise: None learns `noise`; v >= 1e-4 fixes it; 0 < v < 1e-4 raises (BoTorch's floor would silently raise it)."""
def _run_nuts(model: CellGP, nuts: NUTSConfig, seed: int) -> tuple[MCMC, float]   # fit.py:375-389 + extra_fields, returns the sampler
def standardize(y) -> tuple[np.ndarray, float, float]     # z = (y - mean) / std, ddof 0; no negation

# sagp/diagnostics.py — DiagThresholds ✓, Diagnostics ✓ (same 15 fields), diagnose(flat_samples, extra, thresholds, wall_s) ✓
#   _POSITIVE_SAMPLED_SITES = ("outputscale","noise","kernel_tausq","_kernel_inv_length_sq","_a_sq","kernel_ell"); _REAL_SAMPLED_SITES = ("mean",)
#   _DIAG_GROUPS["global"] = ("outputscale","noise","kernel_tausq","mean")

# sagp/bo.py
class IterRNG(NamedTuple): nuts_seed: int; retry_seed: int; sobol_seed: int; torch_seed: int; noise_rng; fallback_seed: int
def iteration_rngs(seed, t) -> IterRNG        # salts: nuts 0x4E, retry 0x2E, sobol 0xCA (kept), torch 0x70, fallback 0xFA (kept)
def initial_design(D, n_init, seed) -> np.ndarray   ✓
def propose_ei(fitted, best_f, rngs, t, *, raw_samples=512, num_restarts=5, sample_around_best_sigma=1e-3, batch_limit=1, maxiter=200) -> (np.ndarray, float)
    """LogExpectedImprovement(fitted.model, best_f) maximized by optimize_acqf under fork_rng()+manual_seed(rngs.torch_seed),
    options seed = rngs.sobol_seed, cholesky_max_tries(9). Returns x (D,) and the log EI at it."""
class BOState ✓ ; class Iteration ✓ (same fields)
def run_bo(objective, surrogate, seed, *, T=200, n_init=20, propose=propose_ei, noiseless=False, state=None, on_iteration=None, log=print) -> BOState
    """As today, with z = standardize(y)[0] (not negated), best_f = max(z), surrogate(X, z, rngs.nuts_seed), the retry on
    rngs.retry_seed, and an exception from `propose` taking the Sobol fallback with status "excluded" (D7)."""

# sagp/references.py
def fit_map(X, y, *, active=None) -> FittedGP        # Papenmeier's dsp model + fit_mll; S = 1; sites outputscale, mean, noise, kernel_inv_length_sq, lengthscale
def propose_sobol(fitted, best_f, rngs, t, *, sequence) -> (sequence[t], nan)   ✓ (param renamed)

# sagp/readouts.py — shares_from_amplitudes ✓, component_means ✓ (per-coordinate parts; the intercept is fitted.means()), readouts ✓, manipulation_checks ✓
# experiments/identify.py — identify(objective, cell, n, seed, *, sobol_n=2048, **fit_kwargs) ✓ (passes an int seed)
# experiments/runlog.py — RunConfig (fields in §4), config_hash ✓, run_dir_for ✓, RunLogger ✓ + record ✓
# experiments/run_bo.py — METHODS ✓, surrogate_for ✓, propose_for ✓, run ✓, resolve_config ✓, main ✓
```

## 3. Layout after the migration (line budgets)

| file | sections | lines now → target |
|---|---|---|
| `sagp/kernels_torch.py` (new) | quadrature constants; `v_of_ell_t`, `_quad_mean_t`, `kbar_1d`; `_CenteredKernel` (coordinate loop, `diag`, parameter registration); three subclasses | 0 → ≤ 260 |
| `sagp/gp.py` | `# --- kernels ---` (JAX, noise-free signatures; −40); `# --- cells ---` (constants ✓; four `PyroModel`s +170; `Cell`/`CELLS`); `# --- inference ---` (`NUTSConfig` ✓; `CellGP` +25; `FittedGP` on the BoTorch model −120; `_run_nuts` +30; `fit` −25); `# --- standardize ---` | 783 → ≤ 860 |
| `sagp/diagnostics.py` | site tuples renamed, `mean` group | 156 → ≤ 165 |
| `sagp/bo.py` | seeding (`IterRNG` 6 fields); `propose_ei` (+35); loop (`_fit_with_retry`, `_propose_with_fallback` +20, `run_bo`); `log_h`/`log1mexp`/`log_ei`/`optimize_ei`/`ACQUISITIONS` removed (−120) | 388 → ≤ 300 |
| `sagp/references.py` | `_dsp_model(X, z, active) -> SingleTaskGP`; `_fit_mll(mll)` (Papenmeier verbatim); `fit_map`; `propose_sobol` | 195 → ≤ 210 |
| `sagp/readouts.py` | `_centered_parts` reads `outputscale`; docstrings | 303 → ≤ 305 |
| `experiments/runlog.py` | `RunConfig`; `_VERSIONED`; `_ACQ_CONSTANTS`; `_environment`; row fields; `save_samples` | 610 → ≤ 600 |
| `experiments/run_bo.py` | adapters; five new flags, three removed | 403 → ≤ 410 |
| `experiments/identify.py` | int seed; docstring | 106 → 106 |
| `sagp/__init__.py` | torch dtype + threads | 47 → ≤ 60 |

Key code the tasks build (verbatim targets):

```python
# gp.py — the sampler copy (fit.py:375-389 with two additions)
def _run_nuts(model, nuts, seed):
    if nuts.num_chains != 1: raise ValueError(...)            # as today
    start = time.perf_counter()
    model.train()
    kernel = NUTS(model.pyro_model.sample, dense_mass=True, max_tree_depth=nuts.max_tree_depth)
    mcmc = MCMC(kernel, num_warmup=nuts.num_warmup, num_samples=nuts.num_samples, progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"))
    return mcmc, time.perf_counter() - start

# gp.py — fit, after _run_nuts
flat, extra = mcmc.get_samples(), mcmc.get_extra_fields()
diag = diagnose(flat, extra, thresholds, wall_s)
samples = model.pyro_model.postprocess_mcmc_samples(mcmc_samples=dict(flat))   # ours keeps every site
samples = {k: v[:: nuts.thinning] for k, v in samples.items()}
model.load_mcmc_samples(samples); model.eval()
return FittedGP(cell=cell.key, X_train=X, Y_train=y, samples={k: jnp.asarray(v.numpy()) for k, v in samples.items() if k in cell.sites},
                fixed_noise=fixed_noise, active=None, status="ok" if diag.passed else "excluded", status_reason=diag.reason,
                attempts=(diag,), model=model)

# gp.py — FittedGP.posterior
def posterior(self, X_test, observation_noise=True):
    X = torch.as_tensor(np.asarray(X_test), dtype=torch.float64)[:, None, :]      # (n_test, 1, D): one point per batch
    with torch.no_grad():
        post = self.model.posterior(X, observation_noise=observation_noise)
    mean, var = post.mean, post.variance                                          # (n_test, S, 1, 1) | (n_test, 1, 1) for S = 1
    mean, var = mean.reshape(X.shape[0], -1).T, var.reshape(X.shape[0], -1).T     # (S, n_test)
    return jnp.asarray(mean.numpy()), jnp.asarray(var.numpy())

# bo.py — propose_ei
bounds = torch.stack([torch.zeros(D, dtype=torch.float64), torch.ones(D, dtype=torch.float64)])
acq = LogExpectedImprovement(model=fitted.model, best_f=best_f)
options = {"batch_limit": batch_limit, "maxiter": maxiter, "sample_around_best": True,
           "sample_around_best_sigma": sample_around_best_sigma, "seed": rngs.sobol_seed}
with torch.random.fork_rng(), gpytorch.settings.cholesky_max_tries(9):
    torch.manual_seed(rngs.torch_seed)
    x, value = optimize_acqf(acq, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples, options=options)
return x.detach().numpy()[0], float(value)

# references.py — Papenmeier's model, verbatim structure
prior = LogNormalPrior(torch.tensor(SQRT2 + math.log(D_eff) / 2), torch.tensor(SQRT3))
base = MaternKernel(nu=2.5, ard_num_dims=D_eff, lengthscale_prior=prior)
covar = ScaleKernel(base_kernel=base, outputscale_prior=GammaPrior(2.0, 0.15))
covar.base_kernel.lengthscale = torch.full((D_eff,), math.exp(SQRT2 + math.log(D_eff) / 2 - 3.0))
likelihood = get_gaussian_likelihood_with_gamma_prior()
model = SingleTaskGP(X_t, z_t[:, None], covar_module=covar, likelihood=likelihood, outcome_transform=None,
                     input_transform=None if active is None else FilterFeatures(torch.as_tensor(active, dtype=torch.int64)))
```

## 4. Config, manifest, npz schema, resume

`RunConfig` after: `family, seed, D, method, T, n_init, alpha=None, fixed_noise=None, noiseless=False, nuts=NUTSConfig(), thresholds=DiagThresholds(), ell_prior=ELL_PRIOR, raw_samples=512, num_restarts=5, sample_around_best_sigma=1e-3, batch_limit=1, maxiter=200, sobol_every=25, sobol_n=2048, out_dir="runs"`. Removed: `acq`, `num_init_candidates`, `num_restarts_ei`. `config_hash` unchanged in mechanism (all fields but `out_dir`, `T`), so **every pre-migration run directory has a different hash and refuses to resume** — intended; no in-flight run survives the migration. CLI: `--raw-samples`, `--num-restarts`, `--sample-around-best-sigma`, `--batch-limit`, `--maxiter` added; `--acq`, `--num-init-candidates`, `--num-restarts-ei` removed.

Manifest: `versions` = python + `("jax", "jaxlib", "numpyro", "numpy", "scipy", "torch", "gpytorch", "botorch", "linear_operator")`; `env` gains `torch_num_threads`; `reference_sha256` removed; `reference_constants` replaced by `acquisition_constants = {"acquisition": "LogExpectedImprovement", "q": 1, "sample_around_best": True, "best_pct": 5.0, "prob_perturb": min(20 / D, 1.0), "cholesky_max_tries": 9}`. `iterations.csv`: the three `a0_*` columns removed, everything else identical (`nuts_attempts` is 1 for a cell, 0 for a MAP reference). `coords.csv` unchanged. `samples/t{t:03d}.npz`: keys are the new site names (`outputscale`, `mean`, `noise`, `kernel_tausq`, `_kernel_inv_length_sq`, `kernel_inv_length_sq`, `lengthscale` | `mean`, `noise`, `kernel_tausq`, `_a_sq`, `a_sq`, `kernel_ell`), plus `status`, `nuts_attempts`, `schema_version=2`; the `a0_*` scalars removed. `checkpoint.npz` unchanged. Resume mechanics unchanged (`note_resume` diffs the nine versions and warns).

## 5. Tests mapped to files

| test (brief's list) | file | disposition |
|---|---|---|
| torch kernels == JAX kernels (four cells + `v_of_ell` + `kbar_all` diag/full) at ℓ ∈ {0.06, 0.5, 3}, 1e-10; PSD; batched over S; `diag` == diagonal of full | `tests/test_sagp_torch_kernels.py` | new |
| product/lengthscale `sample()` log density == `SaasPyroModel.sample` via `numpyro.infer.util.log_density` on fixed constrained params (1e-10); `posterior` == `SaasFullyBayesianSingleTaskGP.posterior` on the same loaded draws; JAX `kernel_product_lengthscale` == BoTorch's `matern52_kernel` | `tests/test_sagp_cells.py` | re-targeted (was `saasgp.SAASGP.model`); chunking tests deleted |
| `_run_nuts` + `fit` reproduce `fit_fully_bayesian_model_nuts(seed=s)` bit for bit (D = 5, n = 30, budget 64/64/4 and 512/256/16); two fits at one `nuts_seed` identical; `fixed_noise` variant; `0 < fixed_noise < 1e-4` rejected | `tests/test_sagp_inference.py` | re-targeted (the three `bit_for_bit` tests rewritten); refit-path tests deleted |
| `diagnose` reads the un-thinned chain and counts divergences; a failing gate → `status == "excluded"`, `attempts` length 1, draws intact and loaded; `mean` in the global group | `tests/test_sagp_inference.py`, `tests/test_sagp_diagnostics.py` | re-targeted |
| `LogExpectedImprovement(fitted.model, best_f)` == `logmeanexp_s(log σ_s + log_h((μ_s − best_f)/σ_s))` from `fitted.posterior(X, observation_noise=False)` (1e-10); today's minimization formula on `−z` with `y_target = min(−z)` equals it | `tests/test_bo_loop.py` | re-targeted (`log_h` reimplemented in the test with scipy `erfcx`); `optimize_ei` tests replaced by `propose_ei` bounds/determinism/seed-sensitivity tests |
| additive posterior mean == `means()[s] + Σ_i component_means` ; Sobol of the posterior mean recovers `labels.s_i` within 0.03 (slow); QMC vs exact ≤ 0.02 | `tests/test_sagp_readouts.py` | one assertion updated (intercept); rest unchanged, re-run |
| replacement reference: `run_bo` on product/lengthscale == a ≤ 40-line hand-written BoTorch loop (`SaasFullyBayesianSingleTaskGP` + `fit_fully_bayesian_model_nuts(seed=rngs.nuts_seed)` + `LogExpectedImprovement` + `optimize_acqf` under the same `torch_seed`/`sobol_seed`, same `standardize`) bit for bit over 15 iterations at D = 5 (slow) | `tests/test_bo_reference.py` | rewritten |
| same seed → identical trajectory (dsp_map and a NUTS cell); resume == uninterrupted; hash changes when any of the five optimizer fields changes; torn rows | `tests/test_run_bo.py`, `tests/test_bo_config.py` | updated (`refit row carries both attempts` deleted; row schema literal updated) |
| all seven methods see the identical initial design and identical resolved optimizer config; manifest lists botorch/gpytorch/torch/linear_operator; `acquisition_constants` | `tests/test_bo_config.py` | re-targeted (`reference_sha256` assertions removed) |
| DSP model's priors/constraints/initialization == a from-scratch construction with Papenmeier's numbers (kernel class, ν, `LogNormalPrior(loc, scale)`, `GammaPrior(2, 0.15)`, `GammaPrior(1.1, 0.05)`, floor 1e-4, init `exp(loc − 3)`, no outcome transform); oracle model sees only the active columns (`FilterFeatures.feature_indices`; `ard_num_dims == |S|`); MAP lengthscales shorter on the active set; `fit_map` deterministic | `tests/test_sagp_references.py` | re-targeted |
| layering (four rules incl. the torch rule); `synthobj` torch-free subprocess guard | `tests/test_layering.py`, `tests/test_botorch_adapter.py` | extended / unchanged |
| pins: nine `name==version` lines match the installed stack; `import sagp` sets torch float64 and JAX x64 | `tests/test_sagp_env.py` | rewritten (vendored tests deleted) |
| CLI: new flags reach the config; `--acq` gone; `python -m sagp.bo` still exits non-zero | `tests/test_bo_cli.py` | updated |
| `identify` keys/shapes unchanged; reproducible under an int seed | `tests/test_identify.py` | updated (seed type) |
| α closed form; GL/ACTIVE_EPS pins; kernel math | `tests/test_sagp_kernels.py` | `saasgp.matern_kernel` equality re-targeted to BoTorch's `matern52_kernel`; rest unchanged |

Deleted outright: `tests/test_sagp_env.py::test_vendored_reference_imports…`, `::test_import_sagp_enables_x64_and_saasgp_fit_runs…`; `tests/test_sagp_inference.py` refit/attempt-1 tests; `tests/test_sagp_cells.py` `_chunk_size`/chunked/`retained count 8` tests; `tests/test_bo_loop.py::test_optimize_ei_*`; `tests/test_run_bo.py::…refit row…`.

## 6. Cost table and the pilot

| `--cell` | per gradient (2026-09-08, n = 200) | NUTS per fit (unchanged formula: 768 × steps × t_grad) | acquisition per iteration, today → after |
|---|---|---|---|
| product/lengthscale | 4.3 ms | 2.2–3.5 min | 13–15 s → ≈ 2–5 s |
| additive/amplitude | 45 ms | 23–36 min (measured 20.7 min at n = 100) | 59–63 s → ≈ 5–15 s |
| additive/lengthscale | 198 ms | 100–158 min | (not measured) → ≈ 10–30 s |
| product/amplitude | 136 ms | 68–108 min | (not measured) → ≈ 10–30 s |

The "after" column is an estimate (≈ 1024 single-point posteriors under `no_grad` plus 5 × ≤ 200 L-BFGS-B gradients through the torch kernel); dense-mass NUTS may change `num_steps_mean`. Task 11 replaces the estimates: `scripts/pilot_sagp.py --stage fit` re-pointed at the new API, `FIT_SIZES` = all four cells × n ∈ {100, 200}, one fit + one `propose_ei` per pair (the compile-time second run dropped via `--once`), written to `docs/superpowers/plans/2026-09-09-botorch-saasbo-pilot.md` with the 2026-09-08 numbers beside them; ≈ 6–8 h wall on the laptop under `caffeinate -di`, run with `nohup` and polled.

## 7. Tasks in execution order

Each task: implement → tests named in §5 green → `git branch --show-current` → stage by path → commit. `PY=/opt/anaconda3/envs/saasbo/bin/python`. The fast suite command is `$PY -m pytest -q -m "not slow" tests/`; slow files run one at a time via `nohup`.

- [ ] **Task 0 — branch, docs, pins.** `git status --porcelain` shows only the four known deletions and the untracked user files; `git switch -c feat/botorch-saasbo` (HEAD-only, same tree, at 02cfb46); copy this plan to `docs/superpowers/plans/2026-09-09-botorch-saasbo.md`; `git add` the brief and the plan; create `.superpowers/sdd/2026-09-09-botorch-saasbo/progress.md` (git-ignored) with ruling R0 (in-place branch); `requirements-sagp.txt` += the four torch-stack pins; `tests/test_sagp_env.py` rewritten to nine pins + the dtype test (vendored tests stay until Task 10, so keep the two old tests for now — delete in Task 10); `sagp/__init__.py` gains `torch.set_default_dtype(torch.float64)` and the `OMP_NUM_THREADS` → `torch.set_num_threads` line. Verify: `$PY -m pytest -q tests/test_sagp_env.py tests/test_layering.py`. ≈ 1 h.

- [ ] **Task 1 — torch kernels.** Create `sagp/kernels_torch.py` (§2, §3); add the layering row; write `tests/test_sagp_torch_kernels.py`: equality with `gp.kbar_all`/`v_of_ell`/the four JAX kernels at 1e-10 on `default_rng(0)` inputs (n = 7, m = 5, D = 4, S = 3 draws) at ℓ ∈ {0.06, 0.5, 3}, `diag=True` vs `torch.diagonal(full)`, PSD (`eigvalsh ≥ −1e-10`) of the batched full matrices, gradient finiteness through `x1`, `GL_NODES_T == gp.GL_NODES`. Verify: `$PY -m pytest -q tests/test_sagp_torch_kernels.py tests/test_layering.py`. ≈ 4 h.

- [ ] **Task 2 — the four `PyroModel`s, `Cell`, `CellGP`, JAX kernel signatures.** In `sagp/gp.py`: JAX kernels lose `(noise, include_noise)` (callers: `cell_kernel_diag` unchanged; tests updated); `kernel_product_lengthscale` on BoTorch's distance formula; the four `PyroModel` subclasses (D1; `postprocess_mcmc_samples` = torch conversion of every site + `lengthscale`/deterministics; `load_mcmc_samples` builds mean via `_build_mean_module`, likelihood by the 17 lines at `fully_bayesian.py:553-569` in a shared helper, kernel per cell, returns a 4-tuple with `None`); `Cell.pyro_model` + new `sites`; `CellGP`. Tests in `tests/test_sagp_cells.py`: log density of `ProductLengthscalePyroModel.sample` == `SaasPyroModel.sample` (both bound to the same `train_X_jax`/`train_Y_jax`, params = one draw from a 4-step MCMC, `log_density(model.sample, (), {}, params)` — raw `get_samples()` keys, not postprocessed) at 1e-10, learned and fixed noise; trace-site inventory per cell (`numpyro.handlers.trace`); `CellGP(...).pyro_model` is the cell's class with `alpha`/`ell_prior` set; JAX product/lengthscale == `matern52_kernel`. Verify: `$PY -m pytest -q tests/test_sagp_cells.py tests/test_sagp_kernels.py`. ≈ 4 h.

- [ ] **Task 3 — `_run_nuts`, `fit`, `FittedGP`, diagnostics.** `_run_nuts` copy; `fit` one attempt (§3 code); `FittedGP` on the BoTorch model (`posterior`, `alphas` via `mean_cache` after a warm-up call, `means`, `noises` from `model.likelihood.noise` (S,) / the clamped fixed value, `params`/`param_sites`/`columns`, `_MAP_REFERENCES` kept; `_Ls`, `_compute_choleskys`, `_predict`, chunking, `chunk_vmap`, `import saasgp`, `from util import` removed); `diagnostics` site tuples + `mean`. Tests (`tests/test_sagp_inference.py` rewritten): bit-for-bit vs `fit_fully_bayesian_model_nuts` — compare `covar_module.base_kernel.lengthscale`, `outputscale`, `mean_module.constant`, `likelihood.noise` with `torch.equal`, at 64/64/4 and 512/256/16 and with `fixed_noise=1e-4`; `fit` determinism; `NEVER_PASSES` thresholds → `excluded` with `attempts == 1` and a working `posterior`; `_POSITIVE_SAMPLED_SITES` pin; `posterior` == `SaasFullyBayesianSingleTaskGP.posterior` (both `observation_noise` values) on the product/lengthscale cell; `alphas()` reproduces the posterior mean (`means()[s] + k_*ᵀ alphas[s]`); shapes for all four cells and `fixed_noise`. Verify: `$PY -m pytest -q tests/test_sagp_inference.py tests/test_sagp_cells.py tests/test_sagp_diagnostics.py`. ≈ 4 h.

- [ ] **Task 4 — readouts.** `_centered_parts` reads `outputscale`; `component_means` docstring names the intercept; `tests/test_sagp_readouts.py`: the additive-mean assertion becomes `mean == means()[s] + Σ components`, `_identification_data` stops negating; slow Sobol recovery and Gate-1 tests re-run once (`nohup $PY -m pytest -q tests/test_sagp_readouts.py > /tmp/ro.log &`). Verify: fast + the slow file green. ≈ 1.5 h.

- [ ] **Task 5 — `sagp/bo.py` and `standardize`.** `standardize` without the negation (docstrings at `gp.py:487, 703, 777`, `bo.py:6, 315`); `IterRNG`/`iteration_rngs`; `propose_ei`; `run_bo` with `best_f`, int seeds, `_propose_with_fallback`; the JAX acquisition code removed. `tests/test_bo_loop.py`: LogEI cross-check (D6), `propose_ei` returns a point in `[0, 1]^D` with a finite value, is deterministic under `(seed, t)`, changes with `torch_seed`, and the `run_bo` purity test; `tests/test_sagp_inference.py::test_standardize…` asserts `z.argmax() == y.argmax()`. Verify: `$PY -m pytest -q tests/test_bo_loop.py tests/test_sagp_inference.py`. ≈ 3 h.

- [ ] **Task 6 — MAP references.** `sagp/references.py` per D8 (`_dsp_model`, `_fit_mll` verbatim from Papenmeier's `fit.py`, `fit_map`, `propose_sobol`); the `FittedGP` for S = 1 with `samples = {"outputscale": (1,), "mean": (1,), "noise": (1,), "kernel_inv_length_sq": (1, D_eff), "lengthscale": (1, D_eff)}`. `tests/test_sagp_references.py` re-targeted (§5). Verify: `$PY -m pytest -q tests/test_sagp_references.py`. ≈ 3 h.

- [ ] **Task 7 — experiments.** `runlog.py` (§4), `run_bo.py` adapters/flags/`resolve_config`, `identify.py` int seed, `README` run-directory table rows for the npz keys and the `a0_` removal. Tests: `tests/test_run_bo.py` (row schema literal; `nuts_attempts == 1` for a cell; both determinism runs), `tests/test_bo_config.py` (versions/env/`acquisition_constants`; hash changes per new field; identical resolved optimizer config across all seven methods), `tests/test_bo_cli.py`, `tests/test_identify.py`. Verify: `$PY -m pytest -q tests/test_run_bo.py tests/test_bo_config.py tests/test_bo_cli.py tests/test_identify.py`, then the whole fast suite. ≈ 3 h.

- [ ] **Task 8 — the replacement reference test.** `tests/test_bo_reference.py` rewritten: the hand-written loop (§1 D6/D7 settings, `NUTSConfig(128, 64, 4)` to keep it ≈ 2 min, `validate_input_scaling(False)`, `rngs = iteration_rngs(seed, t)` for every draw) vs `run_bo(objective, surrogate=fit on product/lengthscale, seed, T=n_init+15, n_init=5, propose=propose_ei)` on `make_family("aligned3", 0, D=5)`; `np.array_equal` on `X` and `y`. Slow. Verify: `nohup $PY -m pytest -q tests/test_bo_reference.py > /tmp/ref.log &` then read the log. ≈ 2 h.

- [ ] **Task 9 — layering, docs, pilot script.** `tests/test_layering.py` torch rule; README §sagp rewritten where it names the vendored path (lines 83–100, 116, 121–142, 150–155, 163–164, 222–229, the cost paragraph gets a pointer to the new pilot file); `docs/superpowers/specs/2026-09-07-sagp-brief.md` gets a one-line header note that its inference clause is superseded by the 2026-09-09 brief; `scripts/pilot_sagp.py`: `optimize_ei` → `propose_ei`, `stage_refit` → `stage_gate` (10 fits, `excluded` rate; `a0_` columns dropped), the cost formula without `(1 + 1.67 r)`, `--once`, new default `--out`. Verify: `$PY -m pytest -q tests/test_layering.py` and `PYTHONPATH=. $PY scripts/pilot_sagp.py --help`. ≈ 2 h.

- [ ] **Task 10 — delete the vendored path.** `git rm saasgp.py saasbo.py util.py`; delete the two vendored tests in `tests/test_sagp_env.py` and any remaining `saasgp`/`saasbo`/`util`/`chunk_vmap` mention (`grep -rn -E "saasgp|saasbo\.py|import util|chunk_vmap" sagp experiments tests scripts README.md` must be empty; the vendored-reference gotchas paragraph in README goes). Verify: the whole fast suite, then all slow files one at a time (`test_bo_reference`, `test_sagp_readouts`, `test_run_bo`, the six `synthobj` ones untouched but re-run). ≈ 1 h.

- [ ] **Task 11 — pilot.** Run `PYTHONPATH=. nohup caffeinate -di $PY scripts/pilot_sagp.py --stage fit --once --out docs/superpowers/plans/2026-09-09-botorch-saasbo-pilot.md > /tmp/pilot.log 2>&1 &`; poll; when done, add the comparison paragraph (fit wall, `num_steps_mean`, `acq_s`, diagnostics per cell at n ∈ {100, 200} against the 2026-09-08 table) and commit the file. ≈ 1 h attended, 6–8 h wall.

- [ ] **Task 12 — final review and hand-off.** Whole-branch review (spec coverage against the brief's test list; `git diff 02cfb46 --stat`); ledger closed; `superpowers:finishing-a-development-branch`. ≈ 2 h.

## 8. Effort

≈ 30 h of implementer/reviewer time over Tasks 0–10 and 12, plus 6–8 h of unattended pilot wall time (Task 11). Tasks 1–3 are the risk: the torch kernels' broadcasting against GPyTorch's `(b, S, n, D)` conventions and the bit-for-bit pin of the sampler copy; if the pin fails, the first suspects are `extra_fields` (memory says it does not perturb NumPyro's stream) and the `progress_bar` flag.

## 9. Open questions (≤ 5)

1. **No diagnostic refit — confirm.** The brief (15:56) says one budget per fit and `excluded` on a failed gate; the ledger records that same decision made at 01:53 and withdrawn at 02:27 pending a look at how Papenmeier's methods refit (they refit from scratch every iteration and never re-run a failed fit). The plan follows the brief. Recommend: confirm; if withdrawn again, Task 3 keeps today's attempt loop unchanged (cost: `a0_` columns stay, +1 h).
2. **DSP location constant.** `SQRT2` (brief, Hvarfner's paper, BoTorch's `SQRT2`) vs Papenmeier's code `1.41` built in float32 "to reproduce the published runs". Recommend `SQRT2` in float64; the difference is 0.004 in log-lengthscale. Say if you want his literal.
3. **Branch creation in the shared checkout.** The brief forbids `checkout`/`switch` but asks for a new branch; the only way is `git switch -c feat/botorch-saasbo` at 02cfb46 (HEAD-only, files untouched). Peers committing without checking `git branch --show-current` would land on the new branch. Recommend: do it, announce it in the ledger and to peers; alternative is to keep committing on `feat/sagp-model-bo`.
4. **NUTS precision.** BoTorch as shipped runs JAX in float32; `sagp` enables x64 process-wide, so the migrated sampler runs float64 (as the vendored path did). Recommend keeping float64 (Cholesky at n = 200 in float32 is fragile; the pin test is unaffected). Say if "BoTorch's scheme" should mean float32.
5. **`mean` in the gate.** BoTorch's model samples a constant mean the vendored model did not have. Recommend pooling its split-R̂/ESS on the natural scale into the `global` group (the gate reads "every scalar the sampler moved converged"); the alternative is to log it without gating. This is preregistration-relevant, hence asked.
