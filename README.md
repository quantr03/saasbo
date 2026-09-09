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
on individual coordinates plus product interactions on disjoint coordinate pairs, each centered
and exactly variance-scaled under the uniform reference measure `nu = U[0,1]^D`: `E_nu[f] = 0`,
`Var_nu[f] = 1`, and the returned `Labels` record which coordinates are active, their variance
shares, lengthscales, and (for the rotated family) axis-aligned Sobol shares under a change of
basis. Observation noise has standard deviation 0.1 and is drawn by the caller, not the
objective, so a paired run can reproduce its own noise stream independently of the objective's
draws (see `synthobj.families.noise_rng`).

The package's only runtime dependencies are NumPy and SciPy; PyTorch/BoTorch are needed only for
`synthobj.botorch_adapter`, which is never imported by `import synthobj` itself.

**Maximization convention.** A `SyntheticObjective`'s `f_star` is its **maximum**, and
`x_star`/`f_star` are exact for every unpaired coordinate. For a paired block (an interaction or a
rotated pair) they are not closed form: the block maximum is found by a grid scan refined with
L-BFGS-B, and re-optimizing from the returned point with tighter tolerances does not improve it.
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
print(obj.labels.s)       # each coordinate's prescribed variance share
print(obj.f_star)         # the exact/near-exact maximum (this code's own convention: maximize)

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
`gamma`, and the active-coordinate count for every entry). `SyntheticObjective.save`/`.load`
round-trip an objective without ever re-drawing, and the same seed always produces bit-identical
files. See `python -m synthobj.generate --help` for `--D`, `--seeds`, `--families`, `--dry-run`
and `--overwrite`.

## Sparse additive GP cells (`sagp`)

`sagp` is the thesis's 2 x 2 of Gaussian-process surrogates for high-dimensional Bayesian
optimization -- {additive, product} kernel structure x {amplitude, lengthscale} sparsity prior --
run on BoTorch 0.18.1's SAASBO (`SaasFullyBayesianSingleTaskGP`, NumPyro NUTS through a copy of
`fit_fully_bayesian_model_nuts`'s sampler lines, `LogExpectedImprovement`, `optimize_acqf`) under
Papenmeier et al. (2025)'s loop protocol, so that a difference in regret or in identification is
attributable to the parameterization and to nothing else. `sagp` is a six-module library and
`experiments` is the study code built on it:

| module | owns |
|---|---|
| `sagp/gp.py` | the four cells as BoTorch `PyroModel`s, `CellGP`, `NUTSConfig`, `FittedGP`, `fit`, `standardize`; the JAX kernels are the log density |
| `sagp/kernels_torch.py` | the three centered kernels in torch, batched over draws, for prediction and the acquisition's gradients |
| `sagp/diagnostics.py` | the convergence verdict on a NUTS attempt |
| `sagp/bo.py` | the loop: seeding, BoTorch's LogEI and `optimize_acqf`, `run_bo` |
| `sagp/references.py` | Papenmeier's `dsp` model at the MAP and the oracle behind `FilterFeatures`; the Sobol proposer |
| `sagp/readouts.py` | posterior readouts from a `FittedGP`'s retained draws |
| `experiments/identify.py` | SQ1's offline identification runner |
| `experiments/runlog.py` | one run directory: config, provenance, checkpointing, resume |
| `experiments/run_bo.py` | SQ2/SQ3 entry point and CLI |

Nothing under `sagp/` imports `experiments`, and `sagp.gp`/`sagp.diagnostics`/`sagp.kernels_torch`
are the bottom of the stack -- they never import `sagp.bo`/`sagp.references`/`sagp.readouts` -- so
a cell's parameterization and the loop's acquisition and budget stay on opposite sides of
`fit(X, y, seed, cell) -> FittedGP` and `FittedGP.posterior(X_test) -> (mean, var)` of shape
`(S, n_test)` per retained sample.
`python -m experiments.run_bo --help` is the entry point. `python -m sagp.bo` is no longer an
entry point and exits non-zero.

**The four cells and the three references** are the seven values `--cell` accepts:

| `--cell` | what it is |
|---|---|
| `additive/amplitude` | a normalized centered Matern-5/2 component per coordinate, summed; sparsity on the components' variances `a_i^2` |
| `additive/lengthscale` | the same additive structure under the reference's SAAS prior on `rho_i = 1/ell_i^2` |
| `product/amplitude` | the centered components multiplied, `prod_i (1 + a_i^2 kbar_i)`; sparsity on `a_i^2` |
| `product/lengthscale` | **SAASBO itself**: BoTorch's `SaasPyroModel` with `sample()` inherited untouched |
| `sobol` | a scrambled Sobol search, no model at all |
| `dsp_map` | Papenmeier et al. 2025's `dsp` model: an ARD Matern-5/2 in a `ScaleKernel` with `Gamma(2, 0.15)`, `LogNormal(sqrt(2) + log(D)/2, sqrt(3))` lengthscales started at the prior mode and `Gamma(1.1, 0.05)` noise, fitted by `fit_gpytorch_mll` with his Adam fallback |
| `oracle_S` | the same model behind a `FilterFeatures` transform on the objective's true active coordinates |

**Settings, which supersede `Design-brief.md` section 2**:

- **Budget 512 / 256 / 16** -- 512 warm-up, 256 samples, thinning 16, so 16 retained draws --
  with `max_tree_depth = 6`, one chain, a **dense mass matrix** (BoTorch's scheme) and a **fresh
  chain per fit with no warm start**. Not the brief's 128 + 128 / 8, which survives only as the
  preregistered fallback (see the cost note below), applied identically to all four cells if it is
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
- **Priors are BoTorch's SAAS priors**, so the four cells differ in their kernel/prior block and
  in nothing else: constant mean `N(0, 1)`; outputscale `Gamma(2, 0.15)` in the lengthscale cells
  (the amplitude cells carry none -- it would be unidentifiable against `tausq`); noise
  `1e-4 + Gamma(0.9, 10)`, or a fixed value, which must be at least 1e-4 because BoTorch clamps
  `train_Yvar` there before the sampler sees it; the half-Cauchy scale mixture at `alpha = 0.1` on
  `rho` or `0.0131` on `a^2`; and, in the amplitude cells only, `kernel_ell ~ LogNormal(0, 1.5)`.
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
| `samples/t{t:03d}.npz` | that iteration's retained posterior draws (16 for a cell, 1 for a MAP reference), keyed by BoTorch's own site names, plus its `status`, `nuts_attempts` and `schema_version` (2) |
| `manifest.json` | the resolved `RunConfig` and its hash, the git commit, package versions (torch, gpytorch and botorch among them), thread environment, `acquisition_constants` -- the maximizer's whole operating point -- the objective's labels, and a `resumed` entry per resume |
| `environment.lock.txt` | every installed distribution as `name==version` |
| `checkpoint.npz`, `log.txt` | the resume point, and the run's narrative (timings, statuses, tracebacks) |

**To resume, re-run the same command.** Every random draw of iteration `t` is a pure function of
`(seed, t)`, so a killed run continues from its checkpoint: the loop reloads `checkpoint.npz`,
refuses to continue if the configuration hash differs, truncates the two CSVs and `samples/` back
to the last complete row, and carries on. `--no-resume` starts over instead. The continuation is
bit-identical on the same CPU model with the same thread settings; on a different CPU the design
and the seeds are identical but the floating-point results may differ at the ulp level, which is
why every resume records its own `env` (machine, processor, thread settings) in `manifest.json`.

On SLURM, one array task per `(family, seed, method)`; re-submitting the same array resumes every
run in it. Five arrays, because the four cells and the three references have different budgets:
one `slurm/sagp_<cell>.sbatch` per cell (40 tasks, one core and 8 GB each, `--time` sized to that
cell's cost) and `slurm/sagp_refs.sbatch` (120 tasks, 2 h). All index `(family, seed)` from
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
