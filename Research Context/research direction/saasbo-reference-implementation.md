# SAASBO reference implementation — exact settings
 
Source: github.com/martinjankowiak/saasbo, files `saasgp.py` (model + NUTS + prediction) and `saasbo.py` (BO loop), read 7 Sep 2026. Stack: JAX + NumPyro (≥ 0.7.2) + SciPy, float64 enabled.
 
## Model (`SAASGP.model`)
 
| element | reference code |
|---|---|
| kernel variance | `kernel_var ~ LogNormal(0, 10)` |
| observation noise | `kernel_noise ~ LogNormal(0, 10)` if `observation_variance == 0`, otherwise fixed at the given value. **The BO driver fixes it at 1e-6** (noiseless assumption). |
| global scale | `kernel_tausq ~ HalfCauchy(alpha)`, `alpha = 0.1` default (demo uses 0.01) |
| local scales | `_kernel_inv_length_sq ~ HalfCauchy(1)` per dimension (non-centered); `kernel_inv_length_sq = tausq * _kernel_inv_length_sq` — so ρ_i = τ² λ_i, λ_i ~ HC(1), i.e. ρ_i | τ² ~ HC(τ²) |
| kernel | ARD with ρ_i multiplying squared differences: d² = Σ_i ρ_i (x_i − z_i)². `matern`: var·(1 + √5 d + 5/3 d²) exp(−√5 d), d² clipped at 1e-12. `rbf`: var·exp(−d²/2). Diagonal adds noise + 1e-6 jitter. **`SAASGP` defaults to matern; `run_saasbo` defaults to rbf.** |
| mean | zero (`MultivariateNormal(loc=0)`); targets standardized first |
| likelihood | exact GP marginal likelihood |
 
## Inference (`SAASGP.run_inference`)
 
| element | reference code |
|---|---|
| sampler | NumPyro `NUTS(model, max_tree_depth=...)`, all other NUTS options default (diagonal mass adaptation, dual-averaging step size, target accept 0.8, `init_to_uniform`) |
| budget in BO | `num_warmup = 512`, `num_samples = 256`, `thinning = 16` → **16 retained samples**; `max_tree_depth = 6` hard-coded in `run_saasbo` (class default 7); `num_chains = 1` |
| demo budget | `saasbo_demo.py`: warmup 256, samples 256, thinning 32 → 8 samples, alpha 0.01 |
| warm start | **none** — a fresh `MCMC` object and a fresh chain at every BO iteration |
| RNG | `random.PRNGKey(seed)` with `seed = 0` at every `fit` call → identical HMC randomness at every iteration |
| diagnostics | `numpyro.diagnostics.summary` (r_hat, n_eff) computed; printed only if verbose; **never acted on** (BO runs with verbose = False) |
 
## Prediction (`SAASGP.posterior`)
 
Per retained sample: Cholesky of K_XX (+ noise + jitter), posterior mean k_*ᵀ K⁻¹ y, posterior variance var + noise + jitter − ‖L⁻¹ k_*‖². Returns arrays of shape (16, n_test) — the mixture is never collapsed.
 
## Acquisition and optimizer (`saasbo.py`)
 
| element | reference code |
|---|---|
| standardization | every iteration: `train_Y = (Y − Y.mean()) / Y.std()`; target `y_target = train_Y.min()` (minimization) |
| acquisition | EI per sample with the noisy predictive std (floored at 1e-6), `xi = 0`, NaNs → 0; **averaged over the 16 samples** (`values.mean(axis=0)`), not EI of the mixture |
| candidates | 5000 scrambled Sobol points (**unseeded**), first one replaced by incumbent + 0.001·N(0,1) (clipped; **unseeded** `np.random`) |
| optimizer | top-5 candidates by EI → `scipy.optimize.fmin_l_bfgs_b` with JAX gradient, bounds [0,1]^D, `maxfun = 100`; best of the 5 |
| initial design | scrambled Sobol, seeded, `num_init_evals` points (demo: 20) |
| failure policy | on any exception, query a uniform random point; abort after 3 exceptions |
| refit | every iteration, from scratch |
 
## Consequences for the thesis design
 
Matching "SAASBO's inference" means: NumPyro NUTS with exactly these options; 512 + 256 thinned to 16; tree depth 6; one chain; no warm start; zero mean on standardized targets; τ² ~ HC(α) with the non-centered local HC(1); prediction and EI-averaging per retained sample. The four cells become four `model` functions differing only in the kernel/prior block; everything else is the reference code path.
 
Deviations the design must make, to be stated explicitly:
 
1. **Kernel: Matérn-5/2** in every cell (the reference class supports it; the BO driver's default is RBF).
2. **Noise learned** (`observation_variance = 0`, prior LogNormal(0, 10)) in every cell, because the interaction sweep needs somewhere for unexplained variance to go and the synthetic observations carry noise; the reference BO driver fixes noise at 1e-6.
3. **Seeding**: candidate Sobol set and incumbent jitter seeded from the run seed; a fresh PRNG subkey per refit instead of `PRNGKey(0)` every time.
4. **Diagnostics logged and acted on** (status ok / refit / excluded) instead of ignored.
5. **Budget**: the full 512 + 256 with refit every iteration costs roughly 2–3× the reduced setting assumed in the design brief's cost table; confirm with one timed fit at n = 200, D = 100 in the pilot, and if a reduction is needed it applies identically to all cells and is preregistered.
Open choice: acquisition. The reference uses EI averaged over samples with the optimizer above; the design brief chose LogEI. Either is fine if identical in every cell and reference; matching the reference makes the product/lengthscale cell SAASBO verbatim.