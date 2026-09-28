# R2-D2 cells brief (verbatim, 2026-09-27)

You are planning (not yet implementing) the R2-D2 prior as a second prior family for all four
cells of the `sagp` 2 × 2, and the staged study that decides whether the completed in-loop BO study
is repeated with it. Repository:
`/Users/quantr03/Library/CloudStorage/OneDrive-AaltoUniversity/2026 Summer Internship'/saasbo`
(the path contains an apostrophe; quote it in every command). Read this whole brief, then produce
an implementation plan. Do not write code until I approve the plan.

## Goal

The completed in-loop study ran the four cells, `{additive, product} × {amplitude, lengthscale}`,
plus three references (`sobol`, `dsp_map`, `oracle_S`), on the synthetic families with D = 100,
T = 200, 20-point Sobol initial design, 10 paired seeds. Locally, `runs/` holds `aligned3`,
`aligned10`, `decoupled`, `interaction_g0.25` (280 runs); the Triton rerun of 2026-09-21 (sagp2)
covers the eight families `aligned3`, `aligned10`, `anti_aligned`, `decoupled`,
`interaction_g0.00`, `interaction_g0.25`, `interaction_g0.50`, `interaction_g0.75` (verify which
of these exist locally and which only on Triton; results were committed there as 635278b0 on
2026-09-25). I want the same eight families run again with the R2-D2 prior in every cell, so that
prior family becomes a crossed factor: 2 × 2 × {half-Cauchy, R2-D2}. The half-Cauchy runs are
kept as they are and never rerun unless a study-level budget change forces every cell to rerun.

The program is staged with gates, and the plan must schedule all of it:

0. Prior-only study: execute the existing brief
   `docs/superpowers/specs/2026-09-25-r2d2-prior-parametrizations-brief.md`, extended with the
   rho-scale variant (the lengthscale cells put the prior on `rho = ell^-2`). Output: the chosen
   parameterization per scale and the calibration tables.
1. Four R2-D2 cells in `sagp`, registered, diagnosed, read out, tested, with the half-Cauchy cells
   untouched.
2. Offline replay on the stored trajectories of the half-Cauchy runs: refit each R2-D2 cell to the
   data a half-Cauchy run had at t in {50, 100, 199}, same seed, same NUTS budget. Output: posterior
   gate rates and identification readouts for R2-D2 against the stored half-Cauchy diagnostics on
   identical data, at about 60 GPU-hours instead of 4,000. Gate: if the R2-D2 cells fail the gate
   the way the half-Cauchy amplitude cells do, the sampler budget is decided first, by me, for all
   eight cells, and nothing in-loop is launched.
3. In-loop pilot: `aligned10` and `decoupled`, the four R2-D2 cells, seeds 0-4, the same initial
   designs as the half-Cauchy runs so that seeds stay paired. About 500 GPU-hours. Gate: a prior-
   family effect on final regret that exceeds seed noise or the preregistered smallest effect
   (0.3 in log10 regret at T = 200); otherwise the full replicate is not run and the pilot is the
   appendix result.
4. Full replicate: eight families × four R2-D2 cells × 10 seeds, about 3,900 GPU-hours at the
   current budget; analysis extended to eleven methods with the family factor; thesis tables.

## Facts you are building on

The cells today (`sagp/gp.py:365-401`, `:272-330`): amplitude cells sample `kernel_tausq ~
HalfCauchy(ALPHA_AMPLITUDE)`, `_a_sq ~ HalfCauchy(1)` per coordinate, deterministic `a_sq =
kernel_tausq * _a_sq`, then `kernel_ell ~ LogNormal(ELL_PRIOR)`; lengthscale cells sample
`outputscale`, `kernel_tausq ~ HalfCauchy(ALPHA_LENGTHSCALE)`, `_kernel_inv_length_sq ~
HalfCauchy(1)`, deterministic `kernel_inv_length_sq` and `lengthscale`. `ALPHA_AMPLITUDE =
ALPHA_LENGTHSCALE * ACTIVE_EPS / RHO_EPS` is derived so that both scales imply the same prior
distribution of the active-coordinate count (`tests/test_sagp_kernels.py:267-285`, quartiles about
(16-17, 37, 64) at D = 100). Registry: `CELLS` and `KERNELS` at `sagp/gp.py:476-510` and `:194-199`,
2-tuple keys `(structure, prior)`; `fit` keeps only `cell.sites` (`:844-846`); `NUTSConfig`
512/256/16, depth 6, one chain; `_run_nuts` (`:772-793`) uses `dense_mass=True`. Diagnostics site
tables at `sagp/diagnostics.py:61-82`, gate 1.1 / 16 / 5 at `:102-162`. Readouts dispatch on the
prior string at `sagp/readouts.py:232` (`is_amplitude = cell[1] == "amplitude"`), which misroutes
any other amplitude-parameterized prior string to the rho-scale `p_active` with no error. Method
strings `structure/prior` come from `experiments/run_bo.py:34` (a 3-tuple key breaks it); run
directories are `runs/<family>/<structure>-<prior>/seed##/` (`experiments/runlog.py:101`);
`config_hash` hashes `dataclasses.asdict(RunConfig)` (`runlog.py:83-97`), so a new `RunConfig`
field moves every stored hash. Per-iteration npz schema: `mean`, `noise`, `kernel_tausq`, `_a_sq`,
`a_sq`, `kernel_ell` (16 draws) plus `status`, `nuts_attempts`, `schema_version`. `coords.csv` holds
readouts (`t, i, native_median, p_active, sobol_hat`), not the evaluated points; the evaluated
design and observations live in `checkpoint.npz` (verify its keys and that the t-th dataset is its
first t rows). Slurm: one sbatch per cell under `slurm/`, `submit_chain.sh` for resumes, a 24 h
limit that additive-amplitude exceeded (median 17.6 h, max 26.6 h per run). Analysis:
`sagp_analysis/latest/analyze.py` hard-codes `CELLS` and `METHODS` (`:22-24`), pairs cells with
references (`:225`), and the thesis tables come from `docs/thesis/regen_tables.py`.

The completed study (`sagp_analysis/latest/REPORT.md`): every cell beats Sobol; the cell whose
structure matches the objective wins; all four cells rank the active set well (AP 0.89-0.97);
the amplitude cells are 100 % gate-excluded in-loop (r_hat median 1.50, ESS 6) and every cell
saturates depth 6, so amplitude readouts are medians of 16 draws from chains with about 6
effective draws; fit plus acquisition cost about 1,940 GPU-hours, acquisition dominating
(180-470 s per iteration against 40-58 s of fit). Point 4 of that report says raising depth or
reparameterizing the amplitude funnel is the first thing to try before trusting amplitude
readouts further. Any budget change applies identically to every cell, preregistered (Design
brief, section 5).

R2-D2 history: on 2026-09-21 an R2-D2 amplitude arm was dropped because the prior alone could not
be sampled at depth 6 (draft note at `~/.claude/plans/immutable-foraging-pine.md`, never in the
repo). On 2026-09-25 the supervisor supplied a reference implementation (standardized log-Gamma
sites with a `numpyro.factor` density correction, Beta site on R², logsumexp normalization, free
`a` and `alpha`). The scouting of that day (`scripts/scout_r2d2/`, untracked, read its README and
`report-prior-nuts-probe.md`, `report-math-audit.md`) found: the reference is exact in law; it is
an affine reparameterization of `log lam` and fails depth 6 at D = 100 exactly like the dropped
arm; its forward sampling is wrong because `Predictive` ignores `factor`; and a Gaussian-copula
parameterization, `z ~ N(0, 1)`, `log lam = log F_Gamma(k)^{-1}(Phi(z))` through a log-space Newton
solver with an implicit-differentiation JVP, samples the prior exactly at depth 6 with zero
divergences and ESS at 55-97 % of draws at both settings tested. The paper (Zhang, Naughton,
Bondell and Reich, `Research Context/paper/Zhang-R2D2ShrinkagePrior-2016.pdf`) ties `a = p a_pi`
(Proposition 5 p. 9; section 4.2.4 p. 17). Under the uncoupled prior the 2026-09-21 `a_pi = 0.2093`
is not a calibration (count quartiles 55/65/75 at D = 100); the median-matching tied point is
k = 0.0892 (28/36/44). All of this is on the prior alone; nothing is known about the posterior.

## The prior, per scale

    R2 ~ Beta(a, b);  phi ~ Dirichlet(alpha/p, ..., alpha/p);  omega = R2 / (1 - R2)
    amplitude cells:    a_sq_i = omega * phi_i        (R2 = first-order additive R2, phi = Sobol shares)
    lengthscale cells:  rho_i  = omega * phi_i        (same global-local form on rho; R2 has no meaning there, say so)

No `sigma^2` coupling (the supervisor's form; targets are standardized). Two exact
parameterizations from the scouting: the reference form (Beta site for R², copula for `lam`,
`phi` by normalization; any `(a, b, alpha)`), and the tied Gamma-Gamma form (`z_xi` and `z_lam`
through inverse Gamma CDFs, `a_sq_i = lam_i / xi`, every site N(0, 1), valid only when
`a = alpha`, closed-form marginals `Beta-prime(alpha/p, b)`). The prior-only study picks; the plan
must accommodate either behind one interface.

## What must not change

- The four half-Cauchy cells and the three references: bit-for-bit. The pinned tests stay as
  they are in substance: `tests/test_sagp_inference.py -k bit_for_bit`, the posterior equality in
  `tests/test_sagp_cells.py`, `tests/test_bo_reference.py` (slow). Every existing test keeps
  passing; tests may be extended, never weakened.
- `config_hash` of every stored `manifest.json` under `runs/` reproduces (280 locally; the 2026-09-21
  note records the check: recompute from each manifest, 0 mismatches). No new `RunConfig` field;
  new cells are new values of the existing `method` string.
- Sampler budget: one `NUTSConfig` for all eight cells. The plan does not choose a budget; it
  schedules the gate that informs my choice.
- The loop protocol (LogEI, `optimize_acqf`, seeding, standardization, refit every iteration, the
  gate policy `ok / refit / excluded`) is untouched; a new cell differs from its half-Cauchy twin
  in the prior block only.
- `runs/` is tracked and is a read-only input for the replay. Replay and pilot outputs go to new
  directories (decide where; `runs_smoke/` is gitignored, `runs/` is not).
- Layering (`tests/test_layering.py`): a new `sagp/*.py` needs a D4-table row and a README
  module-table row; `experiments/` uses public `sagp` names only. `synthobj/` never imports torch.
- Vendored `saasbo.py`, `saasgp.py`, `util.py`: untracked, byte-identical (`shasum` before and
  after).
- Git: `feat/botorch-saasbo` is the working branch; other sessions share the checkout, so never
  `checkout`, `switch`, `stash`, `reset` or `add -A`; commit only when I ask.

## Read before planning (in this order)

1. This brief; then `Research Context/research direction/Design-brief.md` (sections 2, 4, 5)
   and `Research Context/research direction/research question.md` Q6 and section 5 "Priors".
2. `docs/superpowers/specs/2026-09-25-r2d2-prior-parametrizations-brief.md` (stage 0, binding
   for the prior's definition, identities and test protocol) and `scripts/scout_r2d2/README.md`,
   `report-prior-nuts-probe.md`, `report-math-audit.md` items 1-6, `report-repo-integration.md`.
3. `docs/superpowers/specs/2026-09-09-sagp-core-split-brief.md` and
   `docs/superpowers/plans/2026-09-09-sagp-core-split.md` (the house shape of a plan: decisions,
   API sketch, tasks with verification, test map, effort) and
   `.superpowers/sdd/2026-09-09-sagp-core-split/progress.md` (the ledger convention).
4. `sagp/gp.py` (cells `:272-510`, inference `:516-851`), `sagp/diagnostics.py`,
   `sagp/readouts.py` (`:30`, `:214-283`), `experiments/run_bo.py`, `experiments/runlog.py`,
   `experiments/identify.py`, `slurm/*.sbatch`, `slurm/submit_chain.sh`,
   `sagp_analysis/latest/analyze.py` and `REPORT.md` (points 1-5 and the cost tables),
   `docs/thesis/regen_tables.py`.
5. Tests that pin names and counts: `tests/test_sagp_cells.py:179-191`,
   `tests/test_sagp_diagnostics.py:95-121`, `tests/test_sagp_inference.py:394-398`,
   `tests/test_sagp_kernels.py:105`, `:239`, `:267-285`, `tests/test_bo_config.py:93`
   (`len(configs) == 7`), `tests/test_layering.py`, `tests/test_run_bo.py`, `tests/test_bo_loop.py`.
6. `~/.claude/plans/immutable-foraging-pine.md` if present: "If reopened: the design that was
   reached" and "Side findings".

## Decisions the plan must make (with a recommendation and reason for each)

1. Parameterization per scale, behind one interface: reference form (untied allowed) or tied
   Gamma-Gamma. My expectation: tied by default, with the reference form kept available; the
   stage-0 results decide, and the plan says how the choice is switched without touching the
   cells.
2. Calibration rule: match the prior count distribution of `a_sq > ACTIVE_EPS` and `rho > RHO_EPS`
   to the half-Cauchy cells' quartiles at D = 100 with the knobs available (one knob when tied,
   three when untied); a test in the style of `test_alpha_matches_reference_count`; constants and
   their derivation next to `ALPHA_AMPLITUDE`. Say what is matched (median only, or quartiles) and
   what the residual mismatch is, because it confounds prior family with sparsity level.
3. Keys and spellings: new `prior` values in 2-tuple keys (for example `amplitude_r2d2`,
   `lengthscale_r2d2`), method strings `additive/amplitude_r2d2`, run directories
   `additive-amplitude_r2d2`; every place that enumerates methods or cells (`run_bo.py:34`,
   `test_bo_config.py:93`, `analyze.py:22-24`, `regen_tables.py`, slurm) listed with the change.
4. Module placement: `sagp/r2d2.py` at the bottom of the stack (the inverse-CDF map with its JVP,
   the two sampling functions, the calibration identities; imports nothing from `sagp`), imported
   by `sagp/gp.py`, with the D4 row and README row; or everything inside `gp.py`. Recommend and
   say why.
5. Model classes: two new `PyroModel` subclasses (amplitude R2-D2 from `AdditiveAmplitudePyroModel`,
   lengthscale R2-D2 from `ProductLengthscalePyroModel`) overriding only the prior block, with
   product variants by attribute swap as today; sites named with an `r2d2_` prefix; deterministic
   `a_sq` / `kernel_inv_length_sq` / `lengthscale` kept so kernels, loading, readouts and the npz
   schema keep working.
6. Sites, npz schema, `cell.sites`: which sampled and deterministic sites are retained (my
   expectation: the sampled `r2d2_*` sites, the deterministic natives, and `R2` as an extra
   deterministic because its posterior is the misspecification detector of Q4), whether
   `schema_version` is bumped, and what `analyze.py` and `regen_tables.py` need to read the new
   npz files alongside the old.
7. Diagnostics: `r2d2_z*` sites are real-valued (natural scale), a Beta site is unit-interval
   (logit scale, a new category), group assignment (`native`, `ell`, `global`) for the new sites,
   and the extensions to `tests/test_sagp_diagnostics.py:95-121`.
8. Readouts: replace the `readouts.py:232` dispatch on the prior string by the cell's declared
   `native_site` (equivalent for the four cells, correct for the new ones); the exact Sobol
   formulas apply unchanged to the new cells; whether `R2` gets a readout column.
9. Offline replay: a script under `experiments/` that rebuilds the dataset at t from a run's
   `checkpoint.npz` (verify), fits a given cell with the run's seed and budget, writes
   diagnostics and readouts in the `iterations.csv` / `coords.csv` vocabulary, and a comparison
   that sets each R2-D2 fit beside the stored half-Cauchy diagnostics of the same (family, seed,
   t). Which t values, which families, which seeds, CPU or GPU, expected hours, and where the
   output lives. Also whether a half-Cauchy refit is included as a control of the harness
   (reproduction is in distribution only on GPU; the manifest records the device).
10. Pilot and full replicate: the sbatch files (one per new cell, chained resumes as today), the
    24 h limit against the 26.6 h maximum run, GPU-type consistency (wall-times comparable only
    within one GPU type; regret and readouts are), expected GPU-hours per stage, and how a
    budget change (if stage 2 forces one) would be applied to all eight cells.
11. Analysis and thesis: `analyze.py` with eleven methods and a family factor (paired seeds,
    Wilcoxon as today, the two main effects and the family interaction), the `regen_tables.py`
    tables, and the preregistration text for stages 3-4: predictions (both main effects replicate
    in sign; effect sizes within a stated band), SESOI 0.3 in log10 regret, stop rules.
12. Effort and calendar against the thesis timeline (October SQ1, November-December SQ2): what
    can run in parallel, what waits on a gate, what I must decide and when.

## Deliverable of this planning step

`docs/superpowers/plans/2026-09-27-r2d2-cells.md`, in the shape of the 2026-09-09 plan: global
constraints; the twelve decisions with an API sketch (the two sampling functions, the two model
classes, the registry entries, the replay script's command line); tasks in execution order with a
verification for each (the test or command and its expected output), Task 0 being "save this brief
verbatim under `docs/superpowers/specs/` if not already there and start the ledger at
`.superpowers/sdd/2026-09-27-r2d2-cells/progress.md`"; a test map listing every existing test to
extend with its line reference and every new test; effort per task; open questions for me,
each with your recommendation. Stages 0-4 appear as tasks with their gates written as explicit
stop conditions. Planning may run second-long probes (imports, a trace of a model at D = 5, a
`checkpoint.npz` inspection) but no fits and nothing on Triton.

## Environment and process

- Python: `/opt/anaconda3/envs/saasbo/bin/python` (3.11; jax 0.10.2, numpyro 0.21.0, botorch
  0.18.1, torch 2.14.0). Nothing is installed. Triton runs use `slurm/setup_triton.sh`.
- Do not run a bare `pytest tests/`: the untracked `tests/test_sagp_identify.py` fails
  collection; run named files. Tests over 10 s carry `@pytest.mark.slow`.
- The repo has a `.codegraph` index; `codegraph explore "<symbol>"` is the fastest way to read a
  symbol with its callers.
- Files, plans, ledgers, commit messages: normal prose. Numbers in tables.
