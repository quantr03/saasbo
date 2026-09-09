# sagp brief (verbatim, 2026-09-07)

You are planning (not yet implementing) the model and optimization code for a master's thesis on sparse additive GPs for high-dimensional Bayesian optimization. Read this whole brief, then produce an implementation plan. Do not write the implementation until I approve the plan.

## Context

The thesis compares two placements of one sparsity prior — on normalized additive amplitudes a_i² vs on inverse squared lengthscales ρ_i = ℓ_i⁻² (SAASBO) — crossed with two kernel structures (first-order additive vs product), with inference, acquisition, initialization and budget held identical. The design brief is `/Users/quantr03/Library/CloudStorage/OneDrive-AaltoUniversity/2026 Summer Internship'/saasbo/Research Context/research direction/Design-brief.md`; the kernel-centering reference is `/Users/quantr03/Library/CloudStorage/OneDrive-AaltoUniversity/2026 Summer Internship'/saasbo/Research Context/research direction/Kernel normalization check.py` (numpy: `matern52`, `centered`, `v`, 64-node Gauss–Legendre grid on [0,1]). The synthetic objectives come from the already-planned `synthobj` module: `SyntheticObjective` with `__call__(X)`, `observe(X, rng)`, `f_star`, `labels` (S, s_i, ℓ_i, g_i, γ, active indicator).

**Inference must match the SAASBO reference implementation** (github.com/martinjankowiak/saasbo): `saasgp.py` (model, NumPyro NUTS, prediction) and `saasbo.py` (EI, optimizer, loop). Its exact settings are summarized in `/Users/quantr03/Library/CloudStorage/OneDrive-AaltoUniversity/2026 Summer Internship'/saasbo/Research Context/research direction/saasbo-reference-implementation.md`. The stack is therefore JAX + NumPyro + SciPy in float64; the four cells are four variants of `SAASGP.model` that differ only in the kernel/prior block, and every other step — sampler options, budget, prediction per retained sample, standardization, EI averaging, candidate generation, L-BFGS-B — is the reference code path, reused or copied verbatim with seeding added. BoTorch/GPyTorch are not used in the loop.

Two files, one responsibility each:

- `sagp/gp.py` — kernels (JAX), the four model cells, NUTS inference, prediction, posterior readouts, and the offline identification runner.
- `sagp/bo.py` — the BO loop, the reference methods, logging/checkpointing, and the CLI.

Nothing in `bo.py` may know how a cell is parameterized; it talks to `gp.py` through one interface (`fit(X, y, key) -> FittedGP`, `FittedGP.posterior(X_test) -> (mean, var)` per retained sample, exactly the reference's shapes). Nothing in `gp.py` may know about acquisition or budgets.

## gp.py — what it must contain

**Kernels** (JAX, jit-able, differentiable in all hyperparameters, inputs in [0,1]^D):
- `centered_matern52_1d`: 1-D Matérn-5/2 centered under ν = U[0,1] by quadrature, k̃(x,x') = k − m(x) − m(x') + c with m(x) = Σ_q w_q k(x,t_q), c = Σ_{q,q'} w_q w_q' k(t_q,t_q'); optional normalization by v(ℓ) = 1 − c computed by the same quadrature. Must reproduce the v(ℓ) table of the reference script.
- Cell kernels: additive amplitude Σ_i a_i² k̄_{ℓ_i}(x_i,x_i') (normalized components); additive lengthscale σ_f² Σ_i k̃_{ℓ_i} (unnormalized, one amplitude); product amplitude ∏_i (1 + a_i² k̄_{ℓ_i}); product lengthscale = the reference `matern_kernel` (ARD Matérn-5/2, single variance) unchanged. All add noise + 1e-6 jitter on the diagonal exactly as the reference does.

**The four cells**, registered by key `(structure ∈ {additive, product}, prior ∈ {amplitude, lengthscale})`, each a NumPyro `model(X, Y)` function. Shared with the reference in every cell: zero mean on standardized targets; `kernel_noise ~ LogNormal(0, 10)` (learned — see decision 5); the global–local form `tausq ~ HalfCauchy(α)`, `θ_i = tausq · λ_i`, `λ_i ~ HalfCauchy(1)` (non-centered, as in the reference), with θ_i = ρ_i in the lengthscale column (α = 0.1, the reference default) and θ_i = a_i² in the amplitude column (α fixed by the prior-predictive rule, decision 2). Lengthscale cells keep `kernel_var ~ LogNormal(0, 10)`; amplitude cells have no global variance (the a_i² carry it) and put a weakly informative log-normal prior on each 1-D lengthscale (decision 3). The product/lengthscale cell must be the reference model verbatim.

**Inference**: the reference's `run_inference` — NumPyro `NUTS(model, max_tree_depth=6)` with default adaptation, `num_warmup = 512`, `num_samples = 256`, `thinning = 16` (16 retained samples), `num_chains = 1`, a fresh chain at every fit (no warm start), float64. Identical in every cell. Two additions: a fresh PRNG subkey per fit derived from the run seed (the reference reuses `PRNGKey(0)`), and diagnostics computed on the un-thinned chain (split R-hat, n_eff via `numpyro.diagnostics.summary`, divergence count from `extra_fields`) and stored. Every fit returns a `status` ∈ {ok, refit, excluded} with a reason: `refit` means the diagnostics failed once and the fit was repeated with doubled warm-up, `excluded` means it failed twice. The status propagates into `identify()`'s record and into every BO iteration row so exclusions are counted, never silently averaged.

**Prediction**: the reference's `compute_choleskys` + `predict` (mean and noisy predictive variance per retained sample), generalized to the four kernels; returns shape (16, n_test).

**Readouts** (from stored samples, no refitting): per-coordinate native score (a_i² or ρ_i) summaries; P(a_i² > ε) with ε = 0.02; the first-order Sobol index of the posterior mean Ŝ_i — exact per component in the additive cells (quadrature), quasi-Monte-Carlo over x_{-i} in the product cells; the active-set rule; manipulation-check helpers (posterior a_i² vs realized component variance; amplitude ranking vs Sobol ranking).

**Offline identification runner** `identify(objective, cell, n, seed, ...)`: scrambled Sobol design of size n, observe, standardize, fit, return readouts and diagnostics as a flat record. This is SQ1 of the brief and must not import `bo.py`.

## bo.py — what it must contain

- `run_bo(objective, cell_or_reference, seed, T=200, n_init=20, out_dir)`: the reference loop generalized — the same seeded 20-point scrambled Sobol initial design per seed in every cell and reference; standardize targets every iteration; refit from scratch every iteration as the reference does; acquisition and optimizer as in ACQUISITION below, identical everywhere; store the retained samples of every refit so SQ1-style readouts can be computed afterwards without refitting.
- ACQUISITION: the reference's `ei` (per-sample EI with the noisy predictive std floored at 1e-6, xi = 0, NaN → 0, averaged over the 16 samples) and `optimize_ei` (5000 scrambled Sobol candidates with the incumbent-plus-jitter point, top-5 by EI, L-BFGS-B with maxfun 100, bounds [0,1]^D), with the candidate set and the jitter seeded from the run seed. Note: the thesis maximizes; either negate internally or flip `y_target`; state which. If I say LogEI instead, the only change is the per-sample acquisition formula — everything else stays.
- References through the same `posterior` interface and the same acquisition/optimizer: Sobol search; DSP by MAP (ARD Matérn-5/2 with Hvarfner's dimension-scaled lengthscale prior, MAP by L-BFGS in JAX, a single "sample"); oracle-S (the same MAP GP restricted to the true active coordinates from `labels`).
- Logging per iteration: x, y, best-so-far, simple regret against `f_star`, fit wall-clock, R-hat/n_eff/divergences, `status`, native relevance summaries, Ŝ_i every k iterations, acquisition value at the chosen point. Outputs: a tidy per-iteration table (parquet or CSV), an npz of retained samples per refit, a JSON manifest with every setting, the git commit hash and package versions, and an environment lock file next to the manifest. Checkpoint every iteration and resume bit-identically.
- Failure policy: the reference queries a random point on exception; here an exception is logged, the fit is retried once, and if it fails again the iteration queries a Sobol point and the row is marked `excluded`.
- CLI: `python -m sagp.bo --family aligned10 --seed 3 --cell additive/amplitude --T 200 --out runs/`, usable from a SLURM array; a `--dry-run` that prints the resolved configuration and exits.

## Decisions the plan must make (with a recommendation and reason for each)

1. Reuse vs copy. Which reference functions are imported from saasbo.py and/or saasgp.py unchanged and which are copied into `sagp/` with the seeding and generalizations, so that a diff against the reference stays small and reviewable.
2. α on the a² scale: choose α so the prior-predictive median number of coordinates with a_i² > 0.02 (D = 100) equals the number the reference prior implies on the ρ scale under a matched threshold; give the simulation procedure (10⁴ prior draws per candidate α; medians, not means).
3. Lengthscale prior in the amplitude cells: log-normal on the [0,1] scale — propose the numbers and justify against the synthetic families' lengthscales (0.06–3).
4. Diagnostic thresholds with one chain (split R-hat, n_eff on the un-thinned chain, divergences) and how the doubled-warm-up refit is implemented in NumPyro.
5. Noise: learned with the reference class's LogNormal(0, 10) in every cell (recommended; the reference BO driver fixes it at 1e-6, which the interaction sweep cannot tolerate) — provide a flag to fix it, and say how the fixed value is chosen for the noiseless out-of-family checks.
6. Sobol index of the posterior mean for the product cells: QMC design size, cost, caching, and how often it is computed in-loop.
7. Standardization and the labels: a_i² are shares of standardized-y variance; give the mapping back to `labels.s_i` (noise inflates the standardized variance by σ²/Var f).
8. Cost: with 768 NUTS iterations per fit and a refit every iteration, estimate core-hours per run at D = 100, T = 200 (gradient ≈ n³/3 + 2Dn² flops), state the pilot timing to confirm it (one fit at n = 200), and the preregistered fallback if the budget does not fit — a reduced but identical budget in all cells (e.g. the demo's 256/256/32), never a per-cell change.

## Tests the plan must include (pytest, CPU, each < 60 s unless marked slow)

gp.py:
- Centered kernel integrates to zero: Σ_q w_q k̃(t_q, x) = 0 for random x, all ℓ in {0.06, 0.5, 3}, to 1e-10.
- v(ℓ) matches the reference script's table; normalized kernel has mean marginal variance 1.
- All four kernels are PSD on random inputs; the product/lengthscale kernel equals the reference `matern_kernel` to machine precision.
- The product/lengthscale cell's log-joint equals the reference `SAASGP.model`'s log-joint (via `numpyro.infer.util.log_density`) on fixed parameters, to 1e-10; its `posterior` output equals `SAASGP.posterior` on the same samples.
- With the same PRNG key, our NUTS on the product/lengthscale cell reproduces the reference's retained samples bit-for-bit on a D = 5, n = 30 problem (this pins the sampler settings).
- Additive posterior mean equals the sum of component posterior means; Sobol index of the posterior mean on an exactly additive `synthobj` truth at n = 200 recovers `labels.s_i` within 0.03 (slow).
- Gate 1 replication: on the five toy components of the reference script, the normalized amplitude cell's P(a² > 0.09) ordering matches the script's table (slow).

bo.py:
- Same seed → identical trajectory (x's bit-identical) across two runs; resume from a checkpoint reproduces the un-interrupted run.
- The product/lengthscale cell with seeded candidates reproduces the reference `run_saasbo` trajectory when the reference is patched to use the same seeds (D = 5, T = 25, slow).
- All four cells and all references receive the identical initial design and identical acquisition/optimizer configuration (assert on the resolved config).
- End-to-end at D = 5, T = 15 with the DSP-by-MAP reference in < 60 s; regret is non-increasing; manifest contains every setting.
- Sobol reference and oracle-S on a `synthobj` function give the expected regret ordering on aligned3 at T = 50 (slow).

## Deliverable of this planning step

A plan document (markdown) with: the public interface between the two files (signatures with docstrings); class/function layout inside each file; the eight decisions with recommendations; the test list mapped to test files; the CLI and a SLURM array example; the cost estimate of decision 8; estimated effort in hours; and at most five open questions for me. Keep it under 700 lines. Python 3.11, JAX + NumPyro + SciPy pinned (NumPyro ≥ 0.7 API; check the vendored code runs on the pinned version and list any API changes needed). Ask me only if something blocks the plan; otherwise make the reasonable choice and flag it. Out of scope: exact additive Thompson sampling, real benchmarks (Lasso-DNA, MOPTA08), the D = 300 spot check.
