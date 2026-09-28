# R2-D2 amplitude prior: three NUTS parameterizations, prior-only study — brief (verbatim, 2026-09-25)

You are implementing and testing, in the repository
`/Users/quantr03/Library/CloudStorage/OneDrive-AaltoUniversity/2026 Summer Internship'/saasbo`
(the path contains an apostrophe; quote it in every command), three NumPyro implementations of one
and the same R2-D2 prior on the amplitudes of the additive GP cell, and a prior-only study that
tests each of them. Read this whole brief first; it names everything else to read. Then write a
short plan (one page: tasks, and the check that verifies each task), record it under
`docs/superpowers/plans/2026-09-25-r2d2-prior-parametrizations.md`, and execute it without waiting
for approval. Stop and ask only if a decision in the "Decisions" section cannot be made with the
recommendation given there.

## Why this exists

The study's additive-amplitude cell puts a sparsity prior on `a_sq_i`, the variance of coordinate
i's normalized component (`sagp/gp.py:365-401`). Today that prior is a half-Cauchy scale mixture,
`a_sq_i = tausq * lam_i`, `tausq ~ HC(ALPHA_AMPLITUDE)`, `lam_i ~ HC(1)`. The research question
keeps an R2-D2 prior on `a_sq` as an appendix robustness check on SQ1 only
(`Research Context/research direction/research question.md`, Q6 and section 5 "Priors"), because
with normalized components and standardized targets its Beta prior on R² is a prior on the
first-order additive R² and its Dirichlet allocation is a prior on first-order Sobol shares.

On 2026-09-21 an R2-D2 arm was planned and then dropped, because the prior alone could not be
sampled at the inference budget the study holds fixed in every cell (Design brief, section 2:
NumPyro NUTS, 512 warm-up + 256 samples thinned to 16, `max_tree_depth = 6`, one chain; the dense
mass matrix comes from `sagp/gp.py:788`, not from the brief). NUTS on the prior alone at D = 100,
parameterized as `xi ~ Gamma(b, 1)`, `lam_i ~ Gamma(a_pi, 1)`, `a_sq_i = lam_i / xi` with NumPyro's
default log transform, gave r_hat 1.42-1.44, ESS 6, every trajectory at the 63-leapfrog cap; at
depth 10 it passed (r_hat 1.03, ESS 54-114, 237-296 leapfrogs). The half-Cauchy prior passes at
depth 6 (r_hat 1.02-1.04, ESS 73-84, 62 leapfrogs). That design coupled the prior to the noise,
`a_sq_i = sigma_sq * lam_i / xi`; the probe sampled the uncoupled `lam_i / xi`. The decision note
was drafted but never written into the repository; the draft is at
`~/.claude/plans/immutable-foraging-pine.md` (outside the repo; if it is missing, everything you
need from it is in this brief).

On 2026-09-25 the supervisor said that R2-D2 needs a rigorous implementation to work well and
supplied the reference implementation reproduced verbatim in the appendix of this brief. A scouting
pass the same day (everything under `scripts/scout_r2d2/`, untracked) audited and probed it. The
findings you are building on:

1. The reference is exact. Its `numpyro.factor` term reproduces the log-Gamma law with the correct
   Jacobian; NUTS on the model with no likelihood matches an independent sampler at depth 10 on
   every statistic checked (`report-prior-nuts-probe.md`, KS p in 0.21-0.91; `report-math-audit.md`,
   item 1).
2. The reference does not change the 2026-09-21 finding. Its `raw = k * (log lam - log k)` site is
   an affine map of NumPyro's default `log lam` coordinate, and NUTS with a dense mass matrix is
   invariant to affine maps once the mass matrix is adapted (NumPyro regularizes its estimate
   toward the identity and runs the first warm-up window on the identity, so affine variants differ
   before adaptation but share the same limit). At D = 100, depth 6, the 2026-09-21 hyperparameters,
   it gives r_hat 1.38-1.40, ESS 6.7-7.0, every iteration capped at 63 leapfrogs, both seeds; the
   half-Cauchy control passes at identical settings. The intrinsic problem is the ratio of the
   left-tail scale to the wall width of the log-Gamma density at shape k, which is 1/k under any
   affine map (`report-math-audit.md`, items 1 and 6).
3. At small concentration the reference is worse than the plain log transform: at
   (a, b, alpha) = (1, 1, 1), k = 0.01, it produced divergences (14-21 per 256 draws at depth 6
   without the `z_beta` site, 26-80 with it; 33 % in a 4000-draw run) and a variance of `log a_sq`
   3.6 times below the closed-form target.
4. Forward sampling of the reference is wrong. `numpyro.factor` is an observed `Unit` site;
   `numpyro.infer.Predictive` samples `raw ~ N(0, 1)` and ignores the factor, so prior-predictive
   draws, `init_to_sample`, and any forward simulation from the model draw the proposal, not the
   prior (`report-math-audit.md`, item 5; confirmed empirically).
5. A Gaussian-copula parameterization, `z_i ~ N(0, 1)`,
   `log lam_i = log F_Gamma(k)^{-1}(Phi(z_i))` through a Newton solver in log space with an
   implicit-differentiation JVP, passes at depth 6 at both settings on the prior alone: zero
   divergences, ESS of `log a_sq` at median 80-97 % of draws (minimum over coordinates 40-67 %),
   every closed-form identity matched (`report-math-audit.md`, item 6 (v); prototype
   `scripts/scout_r2d2/icdf2.py`, verified against SciPy's `gammaincinv` to 4e-13 in log space).
6. The paper defines the prior with `a = p * a_pi` (Zhang, Naughton, Bondell and Reich, arXiv
   1609.00046v3: Proposition 5, p. 9; "We set a = p a_pi in the rest of the paper", p. 10;
   section 4.2.4, p. 17, "not a choice"). The supervisor's reference leaves `a` and `alpha` free.
   Both cases are in scope.

The question this brief answers, on the prior alone: which parameterization of the R2-D2 prior on
`a_sq` can NUTS sample exactly at the study's fixed budget at D = 100, and what does the prior
imply for the active-coordinate count that calibration will have to match. Nothing here fits data,
touches the BO loop, or registers a cell. Whether the posterior with the GP likelihood is also
sampleable is the next study, not this one; say so in the results note.

## The prior (identical in all three implementations)

With p = D coordinates and hyperparameters (a, b, alpha):

    R2 ~ Beta(a, b)
    phi ~ Dirichlet(alpha/p, ..., alpha/p)         (k := alpha/p is the per-coordinate concentration)
    omega = R2 / (1 - R2)                          (Beta-prime(a, b))
    a_sq_i = omega * phi_i                         (no sigma^2 coupling, no coefficient site)

`phi` and `R2` are independent for every (a, alpha). When `a = alpha` the prior has the paper's
Gamma-Gamma representation `xi ~ Gamma(b, 1)`, `lam_i ~ Gamma(k, 1)`, `a_sq_i = lam_i / xi`, with
closed-form marginals `a_sq_i ~ Beta-prime(k, b)`, `P(a_sq_i > eps) = 1 - I_{eps/(1+eps)}(k, b)`,
and `N_eps | xi ~ Binomial(p, Q(k, eps xi))` with Q the upper regularized Gamma function; when
`a != alpha` that representation encodes a different prior (the reference's with `a` replaced by
`alpha`) and must not be used (`report-math-audit.md`, item 4).

Closed-form identities for the harness (psi digamma, psi' trigamma; all from
`report-math-audit.md`, item 3, re-derive them before use):

    E log R2 = psi(a) - psi(a+b)                 Var log R2 = psi'(a) - psi'(a+b)
    E log omega = psi(a) - psi(b)                Var log omega = psi'(a) + psi'(b)
    E log phi_i = psi(k) - psi(alpha)            Var log phi_i = psi'(k) - psi'(alpha)
    Var(log phi_i - log phi_j) = 2 psi'(k)       (the identity to use when k is tiny)
    E log a_sq_i = psi(a) - psi(b) + psi(k) - psi(alpha)
    Var log a_sq_i = psi'(a) + psi'(b) + psi'(k) - psi'(alpha)
    E R2 = a/(a+b),  Var R2 = ab / ((a+b)^2 (a+b+1)),  E phi_i = 1/p,  Var phi_i = (1/p)(1-1/p)/(alpha+1)
    E omega is finite only for b > 1: every a_sq statistic must be log- or quantile-based.

Ground truth must never come from the NumPyro models. Use NumPy: `R2` from `Beta(a, b)`; `log G_i`
drawn in log space as `log G' + log(U)/k` with `G' ~ Gamma(k+1, 1)`, `U ~ Uniform(0, 1)` (a plain
`Gamma(k)` draw underflows to exactly 0 with frequency 6e-4 at k = 0.01 and poisons every
log statistic); `phi = softmax(log G)`; `a_sq = R2/(1-R2) * phi`; fixed seed; N = 2e5. Keep only
per-draw scalars (`logit R2`, `log phi_1`, `log a_sq_1`, `log max_i phi_i`, `N_0.02`), never the
(N, p) arrays.

## The three implementations

All three are NumPyro sampling functions with one interface,

    sample_a_sq(dim: int, *, a: float, b: float, alpha: float) -> Array     # (dim,)

creating the deterministic sites `a_sq` (dim,), `phi` (dim,), `R2` (scalar; deterministic when it
is not itself sampled) and `log_omega`, so that a later `AdditiveAmplitudePyroModel.sample_amplitudes`
override is one line. Every sampled site name carries the prefix `r2d2_` so that nothing collides
with the site vocabulary in `sagp/diagnostics.py:61-82` and `tests/test_sagp_cells.py:179-191`. For
every sampled site, document its support: positive (diagnosed on the log scale, as
`_POSITIVE_SAMPLED_SITES` are), real (natural scale), or unit-interval (diagnosed on the logit
scale, NumPyro's own unconstrained coordinate for a Beta site); `sagp/diagnostics.py` has no
category for the last, which is a fact for the results note, not something to change now.

**I1, the control (required).** The supervisor's reference with `z_beta` removed and `a_sq`
returned as `exp(log omega + log phi)`. Keep its `_log_gamma_values` construction exactly
(standardized `N(0, 1)` proposal site plus the density factor), including the factor's Jacobian
term. Do not alter I1's construction to change its sampler behaviour; whatever the harness reports
is a property of this parameterization.

**I2, the Gaussian copula in reference form (required).** `R2` keeps its `Beta(a, b)` site.
`z_i ~ N(0, 1)`, `log lam_i = log F_{Gamma(k, 1)}^{-1}(Phi(z_i))`, `log phi = log lam -
logsumexp(log lam)`. The inverse CDF is a Newton iteration on `y = log x` (never on `x`: at
k = 0.01 the median of `Gamma(k)` is 4.5e-31 and `Phi(-8)` maps to y = -3500), with `log u` from
`log_ndtr(z)` and `log(1-u)` from `log_ndtr(-z)`, a lower branch on `log gammainc` with the
small-x series `k y - lgamma(k+1) - k x/(k+1)` below x = 1e-6, an upper branch on `log gammaincc`,
the asymptote `(log u + lgamma(k+1))/k` as the lower initial guess, and a `custom_jvp` giving
`dy/dz = phi(z) / (x f(x))` in closed form, f the Gamma density. `jax.scipy.special` in jax 0.10.2
has `gammainc`, `gammaincc`, `log_ndtr` and no `gammaincinv`; TensorFlow Probability is not
installed; do not add dependencies. Start from `scripts/scout_r2d2/icdf2.py`
(`log_gamma_icdf_from_normal`), which is verified but scratch quality: it needs a fixed iteration
count justified or replaced by a convergence check, documented branches, a finite-difference
gradient test in float64, behaviour at |z| up to 8 and at k in {0.005, 0.01, 0.0333, 0.2093, 0.5,
1, 20.93} defined and tested (no NaN, no inf, monotone in z), and it must be `jit`- and
`vmap`-safe. Predictive-compatible by construction: there is no factor. Note that on the prior
alone every site of I2 but `r2d2_R2` is exactly standard normal, so Tier B below tests NUTS on a
near-Gaussian target; the geometry of the copula map is exercised only with a likelihood, in the
next study.

**I3, the tied Gamma-Gamma copula (required).** Valid only when `a = alpha`; `sample_a_sq` raises
`ValueError` otherwise. `z_xi ~ N(0, 1)`, `z_i ~ N(0, 1)`,
`log a_sq_i = log F_{Gamma(k)}^{-1}(Phi(z_i)) - log F_{Gamma(b)}^{-1}(Phi(z_xi))`. It is the
paper's own definition of the prior, every site is standard normal, there is no simplex
normalization, and its closed-form marginals make Tier D exact. It shares I2's allocation block
(same `z_i` sites, same inverse-CDF map) by design; it differs from I2 in the global coordinate
(`z_xi` through the Gamma(b) inverse CDF instead of a logit-Beta site) and in dropping the
normalization, which is the nonlinear change that makes it a distinct parameterization. Measured
in the scouting at D = 100, depth 6: ESS of `log a_sq` at median 243-254 of 256 (minimum over
coordinates 123-136), 52 leapfrogs, zero divergences, at both settings; on the prior alone its
sampled sites are exactly `N(0, I_{p+1})` for every (k, b), so it passes Tier B by construction and
only its `log a_sq` ESS carries information. If you can justify a different I3 on the same terms
(same prior; a nonlinear change of the coordinates NUTS sees; exact), you may substitute it and say
why in the plan. Not acceptable as I3: the Gamma-Gamma form with NumPyro's default log transform
and the Gamma-Gamma form with standardized log-Gamma sites, which are the same sampler as I1 on the
p allocation coordinates under an adapted dense mass matrix (they differ from I1 only nonlinearly
in the one global coordinate, logit R2 versus log xi) and were measured to fail the same way (ESS
13 and 44 at the 2026-09-21 hyperparameters, capped; 67 divergences at k = 0.01 for the
standardized one). Considered and set aside: NumPyro's `dist.Dirichlet` (stick-breaking) plus the
`Beta` site, nonlinear but order-dependent and failing (ESS 9 and capped; biased variance 6148
against 10003 at k = 0.01); and the full copula, I2 with `R2` also mapped through an inverse Beta
CDF (`jax.scipy.special.betainc` exists, `betaincinv` does not), whose gain is unproven because the
`R2` site was not the ESS-limiting site at either setting. If Tier B shows the `R2` site limiting
I2's ESS, say so in the results note as a reason to try the full copula in the next study; do not
swap I3 for it.

## What must not change

- No edits to any existing file. In particular not `sagp/gp.py`, `sagp/diagnostics.py`,
  `sagp/readouts.py`, `sagp/bo.py`, `sagp/__init__.py`, the `CELLS` and `KERNELS` registries,
  `NUTSConfig`, `DiagThresholds`, any existing file under `experiments/` or `slurm/`, `README.md`,
  or any existing test. No cell is registered. New files are allowed under `experiments/` and
  `tests/` only. A new module under `sagp/` is not allowed: `tests/test_layering.py` rule (c)
  requires a row in its D4 table for every `sagp/*.py`, and that would be an edit to an existing
  test; the results note says what moving the winner into `sagp/` would take (a D4 row, a README
  module-table row).
- Files under `experiments/` may import or reach only public `sagp` names (`tests/test_layering.py`
  rule (b)): `ACTIVE_EPS`, `ALPHA_AMPLITUDE`, `NUTSConfig`, `DiagThresholds` are fine; `_run_nuts`,
  `_POSITIVE_SAMPLED_SITES`, `_REAL_SAMPLED_SITES` are not. Reproduce the sampler call and the
  site-scale rule locally.
- The Tier-B protocol below reproduces `_run_nuts` (`sagp/gp.py:772-793`):
  `NUTS(model, dense_mass=True, max_tree_depth=6)` with NumPyro's defaults otherwise
  (`init_to_uniform`, `target_accept_prob = 0.8`), `MCMC(kernel, num_warmup=512, num_samples=256,
  progress_bar=False)` (`num_chains` left at its default of 1, as `_run_nuts` does),
  `mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"))`, float64 on
  (`import sagp` enables it at `sagp/__init__.py:41-44`; otherwise `numpyro.enable_x64()` before
  anything else).
- The gate is the one in `sagp/diagnostics.py:102-162`: `numpyro.diagnostics.summary(...,
  prob=0.9, group_by_chain=False)` on the unthinned 256 draws, on the log of every positive sampled
  site, every real sampled site on its natural scale, and the logit of every unit-interval sampled
  site (`r2d2_R2`); pooled over all coordinates (max split-R-hat, min ESS); pass iff
  `r_hat_max <= 1.1` and `n_eff_min >= 16` and `divergences <= 5`, with each criterion written as
  the negation of the pass condition so that a NaN fails (`diagnostics.py:138-144`); divergences =
  the `diverging` extra field summed. Every sampled site of every implementation is in the gate,
  and `log a_sq` is gated separately. Do not call `sagp.diagnostics.diagnose` for the verdict: it
  reads only the sites named in its module-level tuples, so on a prior-only `r2d2_` model it raises
  `ValueError` (nothing to concatenate), and in a full GP model it would silently gate on `mean` and
  `noise` alone. Compute the gate yourself.
- Vendored `saasbo.py`, `saasgp.py`, `util.py` stay byte-identical. They are untracked, so record
  `shasum -a 256 saasbo.py saasgp.py util.py` before starting and compare at the end.
- Git: `git branch --show-current` must print `feat/botorch-saasbo` before any commit; do not commit
  unless asked; never `git add -A`, `checkout`, `switch`, `stash` or `reset` (other sessions share
  this checkout). New files stay untracked until asked. `runs/` is tracked and not ignored: write
  nothing there. Caches and raw outputs go under `runs_smoke/r2d2_prior_study/` (gitignored) or
  the scratchpad.
- Environment: `/opt/anaconda3/envs/saasbo/bin/python` (Python 3.11, jax 0.10.2, numpyro 0.21.0,
  scipy 1.17.1, botorch 0.18.1). Nothing is installed. Do not run a bare `pytest tests/`: the
  untracked `tests/test_sagp_identify.py` fails collection (`ImportError: cannot import name
  'identify' from 'sagp.gp'`); run named test files, always including `tests/test_layering.py`.
- Tests taking more than 10 s carry `@pytest.mark.slow` (`pytest.ini`: "slow: takes more than 10
  seconds"); no unmarked test may exceed 60 s (the 2026-09-07 brief's ceiling). The full study is a
  script with a results note, not a test.
- Files, notes, plans, commit messages, docstrings: normal prose.

## The prior-only test protocol (identical for the three implementations)

Settings. p in {30, 100}. Hyperparameter points, written as (a, b, alpha):

- P1 = (0.2093 p, 0.5, 0.2093 p): tied; carries the 2026-09-21 values `a_pi = 0.2093`, `b = 0.5`
  (the paper's default). That `a_pi` was calibrated for the noise-coupled prior
  `a_sq = sigma_sq * lam / xi`, marginal over BoTorch's noise prior; under this brief's uncoupled
  prior it is not a calibration: `P(a_sq_i > 0.02) = 0.652` and the `N_0.02` quartiles at p = 100
  are (55, 65, 75), exact mean 65.2. At p = 100 this is (20.93, 0.5, 20.93), k = 0.2093; at p = 30
  it is (6.279, 0.5, 6.279).
- P2 = (1, 1, 1): tied, k = 1/p (0.01 at p = 100, 0.0333 at p = 30); the stress point where the
  reference diverges.
- P3 = (2, 2, 0.2093 p): untied; I1 and I2 only (I3 cannot represent it; if you substitute an I3
  that can, run it here too).

Tier A, exactness. Per implementation and point at p = 100, one long NUTS run on the prior alone:
512 warm-up, 4000 draws, dense mass, `max_tree_depth = 10` (this tier tests the density, not the
budget). Report the ESS of `log a_sq` per coordinate (median and minimum). Checks:

- Moments, per coordinate i: `mean(log a_sq_i)` within `3 sqrt(Var / ESS_i)` and `var(log a_sq_i)`
  within `3 Var sqrt((kappa - 1) / ESS_i)` of the closed forms, `kappa` the empirical kurtosis of
  that coordinate's draws (not the Gaussian `sqrt(2 / ESS)`: at k = 0.01 the law is nearly
  `-Exponential(1)/k` and `kappa - 1 = 8`); report the fraction of the p coordinates outside,
  expected at most 1 %. The same for `log R2` and `log omega` with their own ESS, and
  `Var(log phi_i - log phi_j) = 2 psi'(k)` on a few pairs.
- Distribution, against the NumPy ground truth: two-sample KS on `logit R2`, `log phi_1`,
  `log a_sq_1`, `log max_i phi_i`, run on `n = ESS_min` equally spaced draws of the 4000 (KS assumes
  independent draws; ESS_min over the `log a_sq` coordinates) against the 2e5 ground-truth draws;
  within tolerance when `p >= 0.01`.
- Count: `N_eps = #{i : a_sq_i > ACTIVE_EPS}` with `ACTIVE_EPS = 0.02` (`sagp/gp.py:214`): median
  within 1 and quartiles within 3 of ground truth, the tolerance `tests/test_sagp_kernels.py:267-285`
  uses. For a tied point also the exact `P(a_sq_i > eps)` against the empirical frequency.
- A run whose `ESS_min` over `log a_sq` is below 100 or whose divergence count exceeds 1 % of draws
  is recorded as "not evaluable" at that point, with its ESS and the tolerance it would need; it is
  neither a pass nor a fail. Do not thin, do not drop the run, do not lower the target silently.

Tier B, the fixed budget. Per implementation, point, p, and seed in {0, 1, 2}: the exact protocol in
"What must not change", at `max_tree_depth = 6` and, for context, at 10. Report r_hat max and ESS
min over sampled sites (on the scales above) and over `log a_sq`, divergences, mean `num_steps`, the
fraction of iterations at the cap (63 at depth 6, 1023 at depth 10), wall time, and the gate
verdict. Also run the current half-Cauchy amplitude prior as the control rows, once per seed at
each p and depth: a standalone prior-only function reproducing
`AdditiveAmplitudePyroModel.sample_amplitudes` (`gp.py:380-384`) with the same sites,
`kernel_tausq ~ HalfCauchy(ALPHA_AMPLITUDE)` imported from `sagp.gp` (about 0.01312) and
`_a_sq ~ HalfCauchy(ones(D))`, both positive and gated on the log scale. If the control fails the
depth-6 gate at p = 100 the harness is wrong: stop and fix the harness before reading any other row.

Tier C, forward sampling. Per implementation, `numpyro.infer.Predictive(model, num_samples=4000)`
on the prior, KS of `log phi_1` and `logit R2` against ground truth. Record whether the forward
draws match. Where they do not, give the one-paragraph reason and do not alter the model to make
them match; this is a property of the parameterization that matters wherever forward simulation
from the prior is used (prior-predictive checks, `init_to_sample`).

Tier D, the calibration readout. At p = 100, from ground truth and cross-checked from each
evaluable implementation's Tier-A draws: the distribution of `N_0.02` (mean, sd, quartiles,
5/95 %) at P1, P2, P3, at the uncoupled tied point that matches the four cells' median,
k = 0.0892 (a = alpha = 8.92 at p = 100; quartiles (28, 36, 44)), and at a small grid over
(a, b, alpha) of your choosing that brackets the four cells' prior count quartiles at D = 100,
which are (16-17, 37, 64) for the half-Cauchy prior at `ALPHA_AMPLITUDE`. The 2026-09-21
objection, made for the noise-coupled prior, was that the tied prior has one knob and matches the
median only (its quartiles were (18-19, 36-37, 54-55) with `sigma_sq` coupling); the untied prior
has three. This table is for the supervisor to choose from; do not choose.

Acceptance. An implementation passes if (i) Tier A is within tolerance at P1 and, for
implementations that represent it, at P3, and (ii) Tier B passes the depth-6 gate for all three
seeds at p = 100 at P1. Tier A and Tier B at P2 are reported, not required. Tier C is reported as a
property, not gated.

Blind comparison. The scouting's per-tier numbers are in
`scripts/scout_r2d2/report-prior-nuts-probe.md` and in item 6 of `report-math-audit.md`. Compute
every verdict from the harness before opening those two, then add a column "agrees with scouting"
per row; where they disagree, show both and say which you trust and why.

## Read before starting (in this order)

1. This brief.
2. `Research Context/research direction/Design-brief.md`: section 2 (what is held fixed),
   section 4 SQ1, section 6 item 3 "Prior scale" (the rule the calibration must serve).
3. `scripts/scout_r2d2/report-math-audit.md` items 1-5 (derivations, the paper, identities, the
   Gamma-Gamma equivalence, the `Predictive` caveat); `report-tests-env.md`; sections 3-5 of
   `report-repo-integration.md` (prior-only NUTS, constants, diagnostics). Leave
   `report-prior-nuts-probe.md` and item 6 of the math audit for the blind comparison.
4. `~/.claude/plans/immutable-foraging-pine.md` if present: "The finding that decided it",
   "Reasons", "If reopened: the design that was reached".
5. `scripts/scout_r2d2/icdf2.py`, `prior_nuts.py`, `probe/probe_r2d2.py`: prototypes; code quality
   is not to be trusted, and `probe_r2d2.py`'s printed numbers belong to the blind comparison.
6. `sagp/gp.py:365-401` (the amplitude model), `:516-544` (`NUTSConfig`), `:772-793`
   (`_run_nuts`), `:214-239` (constants); `sagp/diagnostics.py:17-27`, `:61-82`, `:102-162`;
   `tests/test_layering.py` (rules (b) and (c)); `tests/test_sagp_kernels.py:267-285`;
   `tests/test_sagp_cells.py:179-191`.
7. Only if a re-derived identity disagrees with `report-math-audit.md`:
   `Research Context/paper/Zhang-R2D2ShrinkagePrior-2016.pdf`, eq. (7)-(10) pp. 8-10,
   Propositions 4-5 p. 9, section 4.2.4 pp. 17-18, and the Gibbs step (g) p. 23 that normalizes
   Gamma variates the way the reference does.

## Decisions the work must make (with a recommendation and reason for each)

1. I3. Decided above: the tied Gamma-Gamma copula, unless you substitute one on the stated terms
   and say why in the plan.
2. Where the code lives. Recommendation: `experiments/r2d2_prior.py` (the three sampling functions,
   the inverse CDF and its JVP, the closed-form identities as pure functions, the NumPy ground-truth
   sampler; public `sagp` names only), `experiments/r2d2_prior_study.py` (the harness and its
   command line), `tests/test_r2d2_prior.py` (fast: identity functions, inverse-CDF accuracy and
   gradient, a smoke NUTS run at p = 5 with `NUTSConfig(32, 32, 4)`-sized budgets, the Tier-C
   property of each implementation; slow-marked: one depth-6 gate run per implementation at p = 30,
   seed 0, asserting the gate passes for I2 and I3 and, for I1, only that the run completes with
   finite statistics, its verdict going to the results note). Results at
   `sagp_analysis/2026-09-25-r2d2-prior/REPORT.md` with `tables/` beside it, following
   `sagp_analysis/2026-09-12-1428/`.
3. Site names. Recommendation: `r2d2_R2` (Beta site, I1 and I2), `r2d2_raw` (I1), `r2d2_z` (I2),
   `r2d2_z_xi` and `r2d2_z_lam` (I3); deterministic `a_sq`, `phi`, `R2`, `log_omega`.
4. Tier-A length. Recommendation: 4000 draws at depth 10 for all; a run that does not reach the
   evaluability bar is reported as such, not extended.
5. The Beta site in I2. Recommendation: keep it (logit-Beta is order-1 at both settings, and it is
   what keeps `a != alpha` representable); revisit only on evidence from Tier B.
6. Ground truth. Recommendation: one NumPy function shared by all tiers, log-space Gammas as above,
   seed 0, N = 2e5, per-draw scalars cached under `runs_smoke/r2d2_prior_study/` (about 8 MB per
   point).

## Deliverables

1. Code: the module, the harness, the tests; every test passing, with the commands and their
   decisive output lines in the results note (`tests/test_r2d2_prior.py` and
   `tests/test_layering.py` at least; `-m slow` run once).
2. The results note, in this order: the prior and the three parameterizations in one page; Tier A,
   B, C, D tables (one row per implementation, point, p, seed where applicable, each with the
   "agrees with scouting" column); the calibration table; the verdict per implementation against
   the acceptance rule; a recommendation of which implementation to carry into the next study, and
   what that study must establish that this one cannot (posterior geometry with the GP likelihood
   at n in {30, 100}, D = 100, under the same budget; the `readouts.py:232` `is_amplitude`
   dispatch; the diagnostics site tables and the unit-interval category; the cell registration;
   moving the winner into `sagp/`); a list of everything not done or not verified. Numbers in
   tables, not prose.
3. The plan file from the first step, with each task marked done or not done.

## Environment and process

- One shared harness, written once, before any implementation is written against it; implementers
  do not edit it. If you orchestrate with subagents: the three implementations and their Tier-A
  runs are independent; verification is adversarial and independent of implementation: one
  verifier re-derives the identities and audits the harness and the ground-truth sampler; one
  attacks the inverse-CDF solver (gradient by finite differences, |z| up to 8, the k list above,
  NaN/inf, `jit` and `vmap`, comparison with SciPy's `gammaincinv`/`gammainccinv` in log space);
  one re-runs Tier B from the results table alone and confirms the seeds reproduce the numbers.
- Compute: every prior-only run at p = 100 is seconds; the Tier-A runs are minutes; the whole study
  fits in about an hour of CPU. Do not use Triton.
- Scratch files go to your scratchpad directory, not the repository; `scripts/scout_r2d2/` is
  read-only evidence.

## Appendix: the supervisor's reference implementation (verbatim, 2026-09-25)

```python
import jax.numpy as jnp
from jax.nn import logsumexp
from jax.scipy.special import gammaln
import numpyro
import numpyro.distributions as dist


def _log_gamma_values(
    concentration: float,
    rate: float,
    p: int,
    *,
    site_name: str,
):
    """Sample independent Gamma values through standardized log sites."""
    concentration = jnp.asarray(concentration)
    rate = jnp.asarray(rate)

    location = jnp.log(concentration) - jnp.log(rate)
    proposal = dist.Normal(0.0, 1.0).expand([p])
    raw = numpyro.sample(site_name, proposal)

    log_values = location + raw / concentration

    # Transformed Gamma density, including the Jacobian.
    target_log_prob = (
        concentration * log_values
        - rate * jnp.exp(log_values)
        + concentration * jnp.log(rate)
        - gammaln(concentration)
        - jnp.log(concentration)
    )
    numpyro.factor(
        f"{site_name}_log_density",
        jnp.sum(target_log_prob - proposal.log_prob(raw)),
    )
    return log_values


def r2d2_prior(p: int, a: float, b: float, alpha: float):
    """Gaussian R2-D2 coefficient prior without external scale coupling."""
    R2 = numpyro.sample("R2_genetic", dist.Beta(a, b))

    log_allocation = _log_gamma_values(
        concentration=alpha / p,
        rate=1.0,
        p=p,
        site_name="r2d2_log_allocation_raw",
    )
    log_phi = log_allocation - logsumexp(log_allocation)

    log_global_variance = jnp.log(R2) - jnp.log1p(-R2)
    log_scale = 0.5 * (log_global_variance + log_phi)

    z = numpyro.sample("z_beta", dist.Normal(0.0, 1.0).expand([p]))
    beta = jnp.exp(log_scale) * z

    numpyro.deterministic("phi", jnp.exp(log_phi))
    return numpyro.deterministic("beta", beta)
```

The scouting's derived facts about it: the five terms of `target_log_prob` are exactly
`log p(raw)` for `raw = k (log X - log(k / rate))`, `X ~ Gamma(k, rate)`, including the `- log k`
Jacobian; `log p(raw) = raw - k exp(raw/k) + (k - 1) log k - lgamma(k)`, mode at 0, left slope 1,
curvature `-1/k` at the mode and `-(1/k) exp(raw/k)` beyond it; `E raw = k (psi(k) - log k)`,
`Var raw = k^2 psi'(k)`, which is 1 as k tends to 0 and stays order 1 for k <= 1 (1.05 at
k = 0.2093, 1.64 at k = 1) but grows like k beyond (21.4 at k = 20.93), so "standardized" means
unit-scale at the small shapes used here, not for every k. In the repository's use, `z_beta` goes
and `a_sq = exp(log_global_variance + log_phi)`.
