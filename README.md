# SAASBO

This repository contain a Python package
for SAASBO, an algorithm for high-dimensional Bayesian optimization described in
[High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces](https://arxiv.org/abs/2103.00349).

## Abstract

Bayesian optimization (BO) is a powerful paradigm for efficient optimization of black-box objective functions. High-dimensional BO presents a particular challenge, in part because the curse of dimensionality makes it difficult to define -- as well as do inference over -- a suitable class of surrogate models. We argue that Gaussian process surrogate models defined on sparse axis-aligned subspaces offer an attractive compromise between flexibility and parsimony. We demonstrate that our approach, which relies on Hamiltonian Monte Carlo for inference, can rapidly identify sparse subspaces relevant to modeling the unknown objective function, enabling sample-efficient high-dimensional BO. In an extensive suite of experiments comparing to existing methods for high-dimensional BO we demonstrate that our algorithm, Sparse Axis-Aligned Subspace BO (SAASBO), achieves excellent performance on several synthetic and real-world problems without the need to set problem-specific hyperparameters.

### Requirements
Python 3.7, NumPy, SciPy, JAX, NumPyro


### File structure

Besides the core functionality we include:
- a script (saasbo_demo.py) that demonstrates how to run SAASBO on the Hartmann6 function embedded in D=50 dimensions 
- a notebook (Branin100.ipynb) that demonstrates how to run SAASBO on the Branin function embedded in D=100 dimensions

## Synthetic objectives (`synthobj`)

`synthobj` generates synthetic objective functions with known ground-truth sparsity structure,
for evaluating high-dimensional Bayesian optimization methods (in particular sparse additive GP
surrogates) against a function whose active coordinates, variance shares, and interactions are
known exactly rather than estimated. An objective is a sum of one-dimensional spline components
on individual coordinates, made non-additive either by product interactions on disjoint
coordinate pairs or — for the three rotated variants, which carry no interaction at all — by a
planar rotation of disjoint coordinate pairs, and centered and variance-scaled under the uniform
reference measure `nu = U[0,1]^D`: `E_nu[f] = 0`, `Var_nu[f] = 1`. Both hold exactly under the
64-node Gauss-Legendre rule the construction solves against, and for a rotated objective only
after a quadrature-computed recentering (`mu`) and rescaling (`scale`); under `nu` itself there
is a gap, small and growing with the draw's roughness, that `docs/thesis/` measures. The
returned `Labels` record which coordinates are active, their variance shares, lengthscales, and
(for the rotated family) axis-aligned Sobol shares under a change of
basis. Observation noise has standard deviation 0.1 and is drawn by the caller, not the
objective, so a paired run can reproduce its own noise stream independently of the objective's
draws (see `synthobj.families.noise_rng`).

The package's only runtime dependencies are NumPy and SciPy; PyTorch/BoTorch are needed only for
`synthobj.botorch_adapter`, which is never imported by `import synthobj` itself.

**Maximization convention.** A `SyntheticObjective`'s `f_star` is its **maximum**, and
`x_star`/`f_star` are exact for every unpaired coordinate. For a paired block (an interaction or a
rotated pair) they are not closed form: the block maximum is found by a grid scan refined with
L-BFGS-B, and re-optimizing from the returned point with tighter tolerances does not improve the
value beyond floating-point roundoff.
See `synthobj/interaction.py`'s `block_argmax` and `synthobj/rotation.py`'s
`rotated_block_argmax` docstrings, which also explain why a gap measured against a finite
reference grid bounds that grid rather than the method. This is the
opposite of the SAASBO implementation in this repository, which
*minimizes* its objective — a caller wiring `synthobj` into `saasbo_demo.py` or similar must
negate `f` (e.g. `-obj(X)`, or `SyntheticObjectiveTestFunction(obj, negate=True)` for the BoTorch
adapter) before handing it to code in this repo that expects a minimization problem. This is the
single most likely convention to get backwards when reusing an objective built here.

Minimal example:

```python
from synthobj import make_family

# One of the study's 14 named variants, at a given seed and ambient dimension.
obj = make_family("interaction_g0.25", seed=0, D=20)

print(obj.labels.S)       # the active coordinates (ground truth)
print(obj.labels.s)       # prescribed variance share by coordinate, 0 off S
print(obj.labels.s[list(obj.labels.S)])   # the same shares restricted to S, in S's order
print(obj.f_star)         # the maximum (this code's own convention: maximize) — exact for every
                          # unpaired coordinate, and a converged 2-D optimum for each of this
                          # variant's two interaction blocks

import numpy as np
X = np.random.default_rng(0).uniform(size=(5, obj.D))
f = obj(X)                                      # noise-free
y = obj.observe(X, np.random.default_rng(1))    # + N(0, 0.1^2) noise, caller's own rng
```

Generate the full study grid (14 variants x 10 seeds) to disk:

```
python -m synthobj.generate --out data/objectives
```

This writes `<out>/<variant>/seed{seed:02d}_D{D}.npz|.json` for every `(variant, seed)` pair
plus a `manifest.json` describing the resulting directory (paths, SHA-256 hashes, `f_star`,
`gamma`, and the active-coordinate count for every entry). That description is of the
**directory**, not of this run: its top-level `Ds`/`seeds` are unions over whatever is present,
its hashes are per-platform, and a non-zero exit means an objective-shaped file went
undescribed (exit 1) or the arguments were rejected (exit 2).
`SyntheticObjective.save`/`.load` round-trip an objective without ever re-drawing,
and the same seed always produces bit-identical files **on the same platform**: the zip header
records the OS and text-mode newline translation changes the JSON's hash across platforms (the
reason the manifest carries in its own `hash_note`), and the draws themselves depend on the BLAS
build. See `python -m synthobj.generate --help` for `--D`, `--seeds`, `--families`, `--dry-run`
and `--overwrite`.

The package's own documentation is the thesis appendix under `docs/thesis/`: the construction and
its derivations, the variant-by-variant reference, the API and file format, and the design
decisions behind them. `make -C docs/thesis pdf` builds it as a standalone PDF.

## Sparse additive GP cells (`sagp`)

`sagp` is the thesis's 2 x 2 of Gaussian-process surrogates for high-dimensional Bayesian
optimization -- {additive, product} kernel structure x {amplitude, lengthscale} sparsity
parameterization, under either of two sparsity priors, SAASBO's half-Cauchy or R2-D2 --
run on BoTorch 0.18.1's SAASBO (`SaasFullyBayesianSingleTaskGP`, NumPyro NUTS through a copy of
`fit_fully_bayesian_model_nuts`'s sampler lines, `LogExpectedImprovement`, `optimize_acqf`) under
Papenmeier et al. (2025)'s loop protocol, so that a difference in regret or in identification is
attributable to the cell. `sagp` is a seven-module library and `experiments` is the study code
built on it:

| module | owns |
|---|---|
| `sagp/gp.py` | the eight cells as BoTorch `PyroModel`s, `CellGP`, `NUTSConfig`, `FittedGP`, `fit`, `standardize`; the JAX kernels are the log density |
| `sagp/kernels_torch.py` | the three centered kernels in torch, batched over draws, for prediction and the acquisition's gradients |
| `sagp/diagnostics.py` | the convergence verdict on a NUTS attempt |
| `sagp/r2d2.py` | the R2-D2 prior's two forms behind `sample_log_theta`, and their Gaussian-copula map: the reference Newton solver and the loop-free table the cells run |
| `sagp/bo.py` | the loop: seeding, BoTorch's LogEI and `optimize_acqf`, `run_bo` |
| `sagp/references.py` | Papenmeier's `dsp` model at the MAP and the oracle behind `FilterFeatures`; the Sobol proposer |
| `sagp/readouts.py` | posterior readouts from a `FittedGP`'s retained draws |
| `experiments/identify.py` | SQ1's offline identification runner |
| `experiments/runlog.py` | one run directory: config, provenance, checkpointing, resume |
| `experiments/run_bo.py` | SQ2/SQ3 entry point and CLI |
| `experiments/replay.py` | stage 2's offline replay (plan D9): refits a cell on a stored run's iteration-t data into `replay.csv` (`r2d2_r2` for the amplitude cells, `first_order_r2` for the additive amplitude cells only), and `compare` reads gate G2 off the refits |
| `experiments/r2d2_prior.py` | the prior-only R2-D2 study's models (the supervisor's reference and the two forms of `sagp.r2d2`), its half-Cauchy control, and the NumPy ground truth and closed forms they are checked against |
| `experiments/r2d2_prior_study.py` | that study's harness and CLI: Tiers A-D, the calibration search, and the acceptance table in `sagp_analysis/2026-09-25-r2d2-prior/` |

Nothing under `sagp/` imports `experiments`, and the core -- `sagp.gp` and the three modules
beneath it, `sagp.diagnostics`, `sagp.kernels_torch` and `sagp.r2d2`, which import nothing from
the package -- never imports `sagp.bo`/`sagp.references`/`sagp.readouts`, so
a cell's parameterization and the loop's acquisition and budget stay on opposite sides of
`fit(X, y, seed, cell) -> FittedGP` and `FittedGP.posterior(X_test) -> (mean, var)` of shape
`(S, n_test)` per retained sample.
`python -m experiments.run_bo --help` is the entry point. `python -m sagp.bo` is no longer an
entry point and exits non-zero.

**The eight cells and the three references** are the eleven values `--cell` accepts:

| `--cell` | what it is |
|---|---|
| `additive/amplitude` | a normalized centered Matern-5/2 component per coordinate, summed; sparsity on the components' variances `a_i^2` |
| `additive/lengthscale` | the same additive structure under the reference's SAAS prior on `rho_i = 1/ell_i^2` |
| `product/amplitude` | the centered components multiplied, `prod_i (1 + a_i^2 kbar_i)`; sparsity on `a_i^2` |
| `product/lengthscale` | **SAASBO itself**: BoTorch's `SaasPyroModel` with `sample()` inherited untouched |
| `additive/amplitude_r2d2` | `additive/amplitude` with the R2-D2 prior (below) on `a_i^2` in place of the half-Cauchy |
| `additive/lengthscale_r2d2` | `additive/lengthscale` with the R2-D2 prior on `rho_i` |
| `product/amplitude_r2d2` | `product/amplitude` with the R2-D2 prior on `a_i^2` |
| `product/lengthscale_r2d2` | SAASBO's kernel with the R2-D2 prior on `rho_i`; `sample()` is still BoTorch's |
| `sobol` | a scrambled Sobol search, no model at all |
| `dsp_map` | Papenmeier et al. 2025's `dsp` model: an ARD Matern-5/2 in a `ScaleKernel` with `Gamma(2, 0.15)`, `LogNormal(sqrt(2) + log(D)/2, sqrt(3))` lengthscales started at the prior mode and `Gamma(1.1, 0.05)` noise, fitted by `fit_gpytorch_mll` with his Adam fallback |
| `oracle_S` | the same model behind a `FilterFeatures` transform on the objective's true active coordinates |

**Settings, which supersede `Design-brief.md` section 2**:

- **Budget 512 / 256 / 16** -- 512 warm-up, 256 samples, thinning 16, so 16 retained draws --
  with `max_tree_depth = 6`, one chain, a **dense mass matrix** (BoTorch's scheme) and a **fresh
  chain per fit with no warm start**. Not the brief's 128 + 128 / 8, which survives only as the
  preregistered fallback (see the cost note below), applied identically to all eight cells if it is
  ever taken. **One NUTS run per fit**: a fit whose chain fails the gate (split-R-hat `<= 1.1` and
  ESS `>= 16` over every sampled site, `mean` included, and at most 5 divergences) is marked
  `excluded` and its draws are still used.
- **LogEI is the acquisition** (decided 2026-09-07): BoTorch's analytic `LogExpectedImprovement`
  -- Ament et al. 2023's numerically stable log EI -- on the noise-free posterior, averaged over
  the 16 retained draws by log-mean-exp. That average is the exact log of the sample-averaged EI,
  so the argmax is unchanged in exact arithmetic and only the underflow behaviour differs.
- **The acquisition maximizer is `optimize_acqf`** at Papenmeier et al. (2025)'s numbers: 5
  restarts optimized one at a time (`batch_limit = 1`, because batching makes L-BFGS-B descend on
  the restarts' sum and so changes the iterates, not only the speed), started from 512 raw Sobol
  candidates plus 512 RAASP candidates, all drawn from the design's top 5 % and refined by
  L-BFGS-B for at most 200 iterations. BoTorch splits those 512 in half: 256 are truncated-normal
  perturbations at sigma = 1e-3 of *every* coordinate, and 256 perturb each coordinate with
  probability `min(20/D, 1)` -- so at D = 100 a subset candidate moves about 20 of them. Below
  D = 20 the split is skipped entirely and all 512 perturb every coordinate. Seeding is the only
  change: the Sobol scramble and the RAASP draws come from `(seed, t)`, where the reference draws
  them from global state.
- **Priors are BoTorch's SAAS priors**, so the four half-Cauchy cells differ in their kernel/prior
  block and in nothing else: constant mean `N(0, 1)`; outputscale `Gamma(2, 0.15)` in the
  lengthscale cells (the amplitude cells carry none -- it would be unidentifiable against `tausq`);
  noise `1e-4 + Gamma(0.9, 10)`, or a fixed value, which must be at least 1e-4 because BoTorch
  clamps `train_Yvar` there before the sampler sees it; the half-Cauchy scale mixture at
  `alpha = 0.1` on `rho` or `0.0131` on `a^2`; and, in the amplitude cells only,
  `kernel_ell ~ LogNormal(0, 1.5)`. Each R2-D2 cell is its half-Cauchy twin with that scale
  mixture replaced by the R2-D2 prior on the same parameter, and every other prior kept.
- **Standardization is `(y - mean) / std` with ddof 0**, recomputed every iteration, and the
  incumbent handed to LogEI is `best_f = max z`. Papenmeier's code uses torch's ddof 1: the
  difference is a recorded deviation rather than an oversight, because the readouts' share mapping
  assumes the targets have unit sample variance exactly. The model is therefore built with
  BoTorch's input-scaling check off, which reads ddof 1 and would warn on every fit with n < 52.
- **Everything runs in float64**, torch and JAX alike, fixed at package import; BoTorch as shipped
  runs its NUTS in float32.
- **alpha = 0.0131** on the amplitude cells' `a^2` scale (`ALPHA_AMPLITUDE`), against the
  reference's 0.1 on the `rho` scale. The one-line rule: both priors are the same half-Cauchy
  scale mixture, so the prior-predictive active count depends on `(alpha, cutoff)` only through
  `cutoff / alpha`, and matching the `a^2`-count at the study's active cutoff `eps = 0.02` to the
  `rho`-count at the variance-matched threshold `rho_eps = 0.1524` gives
  `alpha_a = 0.1 * 0.02 / 0.1524 = 0.0131` -- which matches the whole count distribution, not just
  its median (both have median 37 active coordinates at D = 100).
- **The R2-D2 prior** (Zhang et al. 2016; `sagp/r2d2.py`) puts `a_i^2 = omega phi_i` on the
  amplitudes, with a global `omega = R2 / (1 - R2)`, `R2 ~ Beta(a, b)`, and shares
  `phi ~ Dirichlet(k, ..., k)`, in its untied form (`R2D2_FORM = "reference"`: the paper's tie
  `a = k D` is not imposed). That `R2` is the prior's variable, `omega / (1 + omega)`, not the
  data's R^2: on a standardized target `omega` is about the first-order R^2, so `R2` is about
  `R^2 / (1 + R^2)`, at most about 1/2 (`sagp.readouts.r2d2_r2`, `first_order_r2`). NUTS
  samples `R2` itself (`r2d2_R2`) and D standard normals
  (`r2d2_z_lam`), which a Gaussian copula maps onto the Gamma(k) variables the shares normalize,
  through a loop-free table of the inverse CDF. The two lengthscale R2-D2 cells carry it on
  `rho_i = 7.62 omega phi_i`, the constant being `rho_eps / eps` (`R2D2_RHO_SCALE`): then
  `rho_i > rho_eps` exactly when `omega phi_i > eps`, draw for draw, so the two R2-D2 columns share
  one calibration as `alpha_a` makes the two half-Cauchy columns share one. There `log rho` is
  floored at -450, because below `log rho = -473` the derivative of `rho^(-1/2)` overflows and
  turns the whole gradient into NaN; the prior puts about 1e-97 of each coordinate's mass below the
  floor. On all four R2-D2 cells `--alpha` sets `k`, the Dirichlet concentration per coordinate;
  `a` and `b` are constants.
- **The R2-D2 calibration** (decided 2026-09-28 on the prior-only study in
  `sagp_analysis/2026-09-25-r2d2-prior/`, `tables/tier_d_calibration.csv`) is
  `(a, b, k) = (1.5778, 0.8044, 0.4994)` (`R2D2_A`, `R2D2_B`, `R2D2_K`), matched to the
  half-Cauchy cells on the same prior-predictive count of coordinates with `a_i^2 > eps` at
  D = 100. The median is matched exactly (37 active coordinates); the quartiles are 14 and 61
  against the half-Cauchy's 17 and 64, a residual of 3 on each (2.93 on the continuous quartiles),
  the least the proper region `a, b >= 0.5` with `k <= 1/2` allows. The bound on `k` keeps the
  paper's regime of a spike at zero: without it the match keeps improving towards uniform shares,
  a dense prior. The residual confounds the prior family with the dispersion of sparsity, not with
  its level, and every comparison of the two families inherits it.

**Running it.** One `(family, seed, cell)` per invocation:

```
python -m experiments.run_bo --family aligned10 --seed 3 --cell additive/amplitude --T 200 --out runs/
```

`--cell` also takes `sobol | dsp_map | oracle_S`; other flags are `--D 100`, `--n-init 20`,
`--alpha F`, `--fixed-noise F`, `--noiseless`, `--sobol-every 25`, `--no-resume`,
`--nuts 512,256,16` (the fallback budget is passed here, identically for every cell),
`--raw-samples 512`, `--num-restarts 5`, `--sample-around-best-sigma 0.001`, `--batch-limit 1`
and `--maxiter 200` (the acquisition maximizer's five sizes, held fixed across every method),
`--objective-dir` and `--dry-run`. See `python -m experiments.run_bo --help`.

Outputs land in `<out>/<family>/<cell with '/' as '-'>/seed{seed:02d}/`:

| file | contents |
|---|---|
| `iterations.csv` | one row per iteration: `y`, `f`, `best_obs`, `best_f`, `regret`, the acquisition value (a log EI, so it is negative whenever the improvement is small), wall times, `status`, the fit's diagnostics, `y_mean`/`y_std`, and the query point |
| `coords.csv` | long format, `t, i, native_median, p_active, sobol_hat` -- the per-coordinate readouts (absent for `sobol` runs, which fit no model) |
| `samples/t{t:03d}.npz` | that iteration's retained posterior draws (16 for a cell, 1 for a MAP reference), keyed by the model's own site names (BoTorch's, and the R2-D2 prior's `r2d2_*`), plus its `status`, `nuts_attempts` and `schema_version` (2) |
| `manifest.json` | the resolved `RunConfig` and its hash, the git commit, package versions (torch, gpytorch and botorch among them), thread environment and JAX device, `acquisition_constants` -- the maximizer's whole operating point -- the objective's labels, and a `resumed` entry per resume |
| `environment.lock.txt` | every installed distribution as `name==version` |
| `checkpoint.npz`, `log.txt` | the resume point, and the run's narrative (timings, statuses, tracebacks) |

**To resume, re-run the same command.** Every random draw of iteration `t` is a pure function of
`(seed, t)`, so a killed run continues from its checkpoint: the loop reloads `checkpoint.npz`,
refuses to continue if the configuration hash differs, truncates the two CSVs and `samples/` back
to the last complete row, and carries on. `--no-resume` starts over instead. The continuation is
bit-identical on the same CPU model with the same thread settings; on a different CPU the design
and the seeds are identical but the floating-point results may differ at the ulp level, which is
why every resume records its own `env` (machine, processor, thread settings, `jax_device`) in
`manifest.json`. A GPU gives no such guarantee even on one device: two runs of one seed on one
H200 drew different posteriors, with or without XLA's determinism flags (checked 2026-09-10), so
a GPU run reproduces in distribution, not to the bit.

On SLURM, one array task per `(family, seed, method)`; re-submitting the same array resumes every
run in it. Five arrays, because the four cells and the three references have different budgets:
one `slurm/sagp_<cell>.sbatch` per cell (40 tasks, one core, 8 GB and one GPU each -- an H200 or an
H100, whichever partition can start the task first -- `--time` sized to that cell's cost) and
`slurm/sagp_refs.sbatch` (120 tasks, 2 h, on CPUs). Stage 2's offline replay has its own array,
`slurm/r2d2_replay.sbatch` (40 tasks on ELLIS H200s: the four R2-D2 cells on their twins' runs of
the eight families, and the four half-Cauchy cells on their own `aligned10` and `decoupled` runs
as the control; `PROBE=1` runs the GPU cost probe instead), read by `python -m experiments.replay
compare` (`--probe` for the probe). `sagp` takes a visible GPU first and the CPU
otherwise, for the NUTS chain and a cell's acquisition alike (`sagp/__init__.py`, `sagp.gp.fit`);
`JAX_PLATFORMS=cpu` forces the CPU. The cell files also set `TORCH_DISABLE_NATIVE_JIT=1`, since
torch 2.14 JIT-compiles a C launcher for a few CUDA ops and the GPU nodes have no C compiler, and
`--ntasks=1`, since those nodes hand out CPUs in hyperthread pairs and `srun` would otherwise start
two copies of each run into one directory. All index `(family, seed)` from
`$SLURM_ARRAY_TASK_ID`, share the objective grid through `--objective-dir data/objectives`, and
read `REPO`, `OUT`, `T` (and, for the cells, `NUTS`) from the environment, so a smoke run is
`sbatch --export=ALL,OUT=runs_smoke/,T=22 --array=0 slurm/sagp_refs.sbatch`. On Aalto's Triton,
`REPO_URL=<remote> bash slurm/setup_triton.sh` on a login node makes the checkout under `$WRKDIR`,
the `saasbo` conda environment, the objective grid and a dry run, once.

`--time` is a **checkpoint interval, not a deadline**: a task the wall clock kills has written a
checkpoint after every completed iteration, and re-submitting the same array picks each run up
where it stopped. Chain the resubmissions rather than watching for them:

```bash
bash slurm/submit_chain.sh slurm/sagp_product-lengthscale.sbatch 3    # the array, then three afterany re-submissions
```

`afterany` rather than `afterok` on purpose -- a task killed at the wall clock exits non-zero, and
that is exactly the case the next submission exists to continue. A run already at `T` records the
resume in its manifest and exits 0 without fitting anything, so an over-long chain costs a few
seconds per task and nothing else.

`OMP_NUM_THREADS=1` matters: an array task gets one core, and letting torch or XLA try to spread
one fit over cores it does not have makes the whole node slower, not faster. It governs both --
`sagp/__init__.py` reads it into `torch.set_num_threads`, and the resulting `torch_num_threads` is
recorded in `manifest.json` beside the XLA flags. On a laptop, run under
`caffeinate -di` -- a sleeping Mac does not advance `perf_counter`, so a suspended run reports
timings that are wrong rather than merely late.

**Cost.** One gradient of the log-joint at n = 200, D = 100, and what it implies for a fit (768
NUTS iterations x 40-63 leapfrog steps) and for a `T = 200` run (`62 x t_fit`, before the
acquisition), measured 2026-09-07 on an M3 laptop:

| `--cell` | per gradient | one fit at n = 200 | one T = 200 run |
|---|---|---|---|
| `product/lengthscale` | 3.1 ms | 1.6-2.5 min | 1.7-2.6 wall-clock hours |
| `additive/amplitude` | 45 ms | 23-36 min | 24-37 wall-clock hours |
| `product/amplitude` | 135 ms | 69-109 min | 71-113 wall-clock hours |
| `additive/lengthscale` | 173 ms | 89-139 min | 92-144 wall-clock hours |

**The single-thread re-measurement agrees within 20% (see
`docs/superpowers/plans/2026-09-07-sagp-pilot-singlethread.md`), so the table above is in
core-time.** The pilot's cost stage gives, per `T = 200` run: product/lengthscale 5.2,
additive/amplitude 12.2, additive/lengthscale 205 and product/amplitude 140 core-hours (the
last three are lower bounds). The measured additive/amplitude fit at n = 100 took 20.7 min
against the formula's 3.3 min. Realistic additive/amplitude cost is therefore 60-75 core-hours
per run, and the unfused cells several hundred. Those core-hours still carry the old scheme's
`(1 + 1.67 r)` diagnostic-refit factor, which no longer exists: the gate labels a fit and never
reruns it, so a run costs `62 x t_fit + 180 x t_acq`.

The reference cell is affordable and the three centered cells are 15-55x more expensive per
gradient: each evaluates one exponential per (pair, coordinate) rather than one per pair, and two
of them additionally materialize an `(n, n, D)` tensor that XLA declines to fuse. The consequence
is that the full study grid does not fit the planned compute at 512 / 256 / 16, and the choice
between the preregistered `--nuts 128,128,8` fallback, a kernel-level optimization of the centered
cells, a smaller grid and more cores is still open. `t_acq` is now `optimize_acqf`'s rather than
the 5000-candidate reference optimizer's: 1024 candidates scored through the torch kernels, then
five L-BFGS-B restarts of at most 200 iterations, every step of which differentiates that same
kernel. Reproduce the gradient numbers with `scripts/pilot_sagp.py --stage grad`; the 2026-09-09
pilot (`docs/superpowers/plans/2026-09-09-botorch-saasbo-pilot.md`) re-measures fit and
acquisition under BoTorch's scheme, and the 2026-09-07 pilot it supersedes is
`docs/superpowers/plans/2026-09-07-sagp-pilot.md`.

**On a GPU** (measured 2026-09-10 on one H100, JAX and torch both on it, with
`scripts/pilot_sagp.py --stage fit --once`) a fit takes 32-55 s at n = 100-200 in every cell --
13x the CPU pilot's speed for product/lengthscale and 85x for additive/amplitude, the one centered
cell it reached -- and the budget question above is settled: the study runs 512 / 256 / 16, on
GPUs. The acquisition is what is left:

| `--cell` | fit, n = 100 / 200 | acquisition, n = 100 / 200 | readout |
|---|---|---|---|
| `product/lengthscale` | 41 / 46 s | 4 / 6 s | 9 s |
| `additive/amplitude` | 41 / 53 s | 267 / 306 s | 2 s |
| `additive/lengthscale` | 32 / 47 s | 214 / 206 s | 1 s |
| `product/amplitude` | 39 / 55 s | 225 / 297 s | 5 s |

The centered cells' torch kernels accumulate the D coordinates one at a time, so on a GPU their
acquisition is bound by kernel launches and only 2-3x faster than on the host CPU. A `T = 200` run
is therefore about 2.5 h for product/lengthscale and 12-17 h for a centered cell, most of it
acquisition.
