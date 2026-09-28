# R2-D2 prior-only study: results and gate G0

Stage 0 of `docs/superpowers/plans/2026-09-27-r2d2-cells.md` (Task 4), run on 2026-09-28 at commit `d06be119c` under the binding protocol `docs/superpowers/specs/2026-09-25-r2d2-prior-parametrizations-brief.md` and ledger rulings R12, R15 and R16. Every number in the tier tables is copied, rounded, from the harness's own tables in `tables/`, written by `python -m experiments.r2d2_prior_study` (source digest `7a5110da855c4cdb`). Numbers marked *supplementary* come from the scratch analyses in the appendix, which call the harness's functions unchanged. No code was changed for this note.

This study samples the prior alone. Nothing here fits data or touches the BO loop. Whether the posterior with the GP likelihood can be sampled at the same budget is the next study (section 9).

## Summary

- **The harness check holds.** The half-Cauchy control passes the depth-6 gate at p = 100 for all three seeds: r_hat max 1.023-1.047, ESS min 55.7-83.9, no divergence.
- **I2 and I3 pass the acceptance rule; I1 does not.** All three are exact on Tier A at P1, and I1 and I2 at P3. But I1, the supervisor's reference, fails the depth-6 gate at P1 in 3 of 3 seeds (r_hat 1.31-1.40, ESS 6.5-7.0, every iteration at the 63-step cap). I2, the reference-form copula Quan chose, passes the depth-6 gate in all 36 of its depth-6 runs, at p = 30 and p = 100 and at all six points; I3 passes all 18 of its own.
- **At the calibrated points.** I2 at C_untied: Tier A **fail** (the variance check: 5 of 100 coordinates outside their 3-sigma band; null pass rate 0.835), and the Tier B depth-6 gate 3/3. I2 at C_untied_k1: Tier A pass (null 0.86), and Tier B 3/3. At C_tied, the tied point of D1's fallback, I2 and I3 both pass Tier A (null 0.84) and Tier B (3/3).
- **The C_untied fail is a false fail of the variance check on a NUTS chain, not a wrong density** (supplementary). The check's tolerance uses the ESS of the draws of log a_sq_i, median 4,767 from 4,000 draws. That is 1.67 times the ESS of their squared deviations, which is the ESS a variance estimate actually has. On the latter, 0 of 100 coordinates are outside. The i.i.d. null cannot show this, because both ESS equal the draw count there, and it never puts more than 4 coordinates outside in 1,200 replicates. Over nine further seeds, I2 fails the protocol's Tier A at P1, C_untied and C_untied_k1 alike (3, 3 and 5 of 9 seeds). With the variance check on its own ESS, 8 of 10 chains pass at each of the three points, which is the null's rate.
- **G0.** The harness condition is met. The program condition is met (I2 and I3). The calibrated-point condition is **not met at C_untied** by the harness's verdict. It is met at C_untied_k1, and at C_tied, the point of D1's fallback, which this failure triggers. G0 therefore stops here for Quan's decision.
- **Recommendation (D1-D2; a recommendation, not a choice).** Form F = reference (I2). Calibration C = C_untied: (a, b, k) = (1.578, 0.8044, 0.4994), alpha = 49.94. It matches the median (37); its quartiles are (14, 37, 61) against the half-Cauchy's (17, 37, 64), a residual of 3 on each quartile (continuous 2.93). Its Tier A fail is read as the false fail above. There are three alternatives (section G0).
  - C_untied_k1 = (3.049, 1.674, 0.9981), residual 3 (q25 3, q75 2; continuous 2.50). It passes every protocol check, but it lies outside D2's k ≤ 1/2, where its shares are near-flat (Dir(1)). Its Tier A pass is also no stronger evidence than C_untied's fail.
  - D1's own fallback: the tied form at b = 0.5, median-matched, C_tied = (9.379, 0.5, 0.09379). It passes every protocol check, in both forms. It is not recommended: its residual is 18, and it puts 0.2 % of R2's mass below 0.6.
  - Fix the harness's variance check first, then rerun Tier A over several seeds at C_untied before ruling.

## G0: the stop conditions

| condition | rule (plan, Task 4) | numbers | verdict |
|---|---|---|---|
| Harness | the half-Cauchy control passes the depth-6 gate at p = 100 | 3/3 seeds pass: r_hat max 1.023 / 1.043 / 1.047; ESS min 73.0 / 83.9 / 55.7 (`_a_sq` limiting); 0 divergences; mean steps 61.9 / 61.2 / 63.0, with 97 / 95 / 100 % of iterations at the cap. The log a_sq gate also passes 3/3 (ESS 62.9 / 120 / 86.0). The harness exited 0 with `control_passes_depth6_p100: true`. | **met** |
| Program | I2 or I3 passes acceptance: Tier A within tolerance at P1 (and at P3 where represented), and the depth-6 Tier B gate for all three seeds at p = 100 at P1 | I2: Tier A pass at P1 (null 0.88) and at P3 (null 0.91); Tier B at P1 3/3 (r_hat 1.043 / 1.043 / 1.028, ESS 102 / 94.4 / 87.3). I3: Tier A pass at P1 (null 0.88), P3 not representable; Tier B at P1 3/3 (r_hat 1.027 / 1.035 / 1.038, ESS 150 / 136 / 108). | **met** (both) |
| Calibrated point: C_untied | I2 at C_untied: Tier A within tolerance, and the depth-6 gate 3/3 at p = 100 | Tier A **fail**: var_log_a_sq, 5 of 100 coordinates outside (1 allowed), need_k_var 3.79. Every other check is within tolerance: KS min p 0.012 on 4,000 draws; count quartiles 13 / 38 / 63 against 14 / 37 / 61. Null pass rate 0.835; the null fails this check 2.5 % of the time. Tier B 3/3: r_hat 1.030 / 1.013 / 1.036; ESS 163 / 124 / 159; log a_sq ESS 150 / 150 / 89.1; 0 divergences. | **not met** (Tier A) |
| Calibrated point: C_untied_k1 | the same at C_untied_k1 | Tier A pass (null 0.86; ESS median / min 4,660 / 3,618). Tier B 3/3: r_hat 1.030 / 1.034 / 1.026; ESS 133.5 / 78.8 / 95.6; log a_sq ESS 137 / 105 / 188; 0 divergences. | **met** |
| Calibrated point: C_tied (D1's fallback: the tied form at b = 0.5, median-matched) | the same at C_tied, for I3 (the tied form) and for I2 (the reference form, which represents the tie) | I2: Tier A pass (null 0.84; ESS median / min 4,133 / 3,552). Tier B 3/3: r_hat 1.048 / 1.024 / 1.072; ESS 103 / 124 / 96.3; 0 divergences. I3: Tier A pass (null 0.84; ESS 4,377 / 3,326). Tier B 3/3: r_hat 1.027 / 1.035 / 1.038; ESS 150 / 136 / 108; 0 divergences. Against it: residual 18 (quartiles 30 / 37 / 46), and P(R2 < 0.6) = 0.0022. | **met** |

By the plan, a failure at the calibrated point means "stop and ask". D1 made the reference form conditional: "the reference (untied) form calibrated on the count quartiles, if it passes stage 0 at that point; the tied form at the paper's b = 0.5, median-matched, as the fallback". By the harness's verdict, that conditional has now triggered. Quan, with the supervisor, has four options:

1. **Keep C_untied,** reading its Tier A fail as the false fail below. This is the recommendation.
2. **Move to C_untied_k1,** R15's sensitivity point, which passes every protocol check.
3. **Take D1's fallback, C_tied,** in the tied form (I3) or the reference form (I2). It passes every protocol check in both. It is not recommended. Its residual of 18 matches the half-Cauchy count at the median only: the interquartile range is 16 against 47. And its P(R2 < 0.6) = 0.0022 is the prior-data conflict on R2 of D1's reason 2.
4. **Fix the harness's variance check first:** put the tolerance on the ESS of the squared deviations, then rerun Tier A over several seeds at C_untied before ruling.

Nothing in stage 1 that consumes C (Task 6) should start before that ruling.

### Reading the C_untied fail beside the null (ruling R16; supplementary)

R16 asks whether the failing check sits within the null's false-fail profile. At C_untied the variance check fails in 2.5 % of the harness's 200 i.i.d. replicates, never with more than 3 coordinates outside. In 1,000 further replicates (root seed 1) it never has more than 4 outside (Table S2). Against that profile alone, 5 of 100 is outside what i.i.d. draws produce. Three measurements point to the chain as the cause, not the density.

1. **The variance check's tolerance is too tight for these chains.** The harness takes the brief's `3 Var sqrt((kappa - 1) / ESS_i)` with ESS_i the ESS of log a_sq_i itself. On a near-Gaussian target, NUTS draws can be antithetic in x, with ESS above the draw count as here, but not in (x - mean)^2, and the sample variance's standard error depends on the latter. For I2 at C_untied the first ESS is 1.67 times the second (medians 4,767 and 2,879). On the second, 0 of 100 coordinates lie outside 3 sigma, with largest z 2.94 (Table S1). The same correction leaves every I2 and I3 chain of Tier A at 0 coordinates outside. It leaves I1's two variance fails, at C_tied and C_untied_k1, at 1 coordinate each, within the 1 % allowance.
2. **The global coordinate moved all 100 variances together.** The chain's sample variance of log omega is 3.36 against its closed form's 3.15. On the draws' ESS, the ESS this note argues is wrong for a variance, that is +2.64 sigma; on its own squared-deviation ESS it is +1.68 sigma. 79 of the 100 coordinates' sample variances lie above theirs. At C_untied, log omega carries 39 % of every log a_sq_i's variance (Var log omega 3.15 of 8.08). One fluctuation of that one shared coordinate therefore shifts all 100 per-coordinate variances the same way. The chain's lowest KS p, 0.012, is on logit R2, which is that same coordinate. The most direct evidence is the ten chains at C_untied pooled: the protocol chain and seeds 1-9, 40,000 draws. There, log omega's sample variance is 3.174 against 3.155, a signed z of +0.49 on the summed squared-deviation ESS. The per-coordinate ratio s²/Var is 1.0027 ± 0.0042 (between-chain standard error). Both are the review's figures, and both reproduce in a scratch recomputation (+0.498; 1.0027 ± 0.0042).
3. **I2's chains fail the protocol's Tier A at the same rate at every point.** Table S3 runs nine further seeds (1-9) at the Tier A budget: 6 pass at P1, 6 at C_untied and 4 at C_untied_k1. With seed 0, the protocol's verdict passes 7, 6 and 5 chains of 10. With the variance check on the squared deviations' ESS and every other check unchanged, 8 of 10 pass at each point, against null pass rates of 0.88, 0.835 and 0.86. C_untied behaves like P1, where the protocol verdict passed and I2 is accepted.

The review of this note (fix round 1) confirmed the argument independently, in theory and by three measurements:

- batch means on the chain give a standard error for s² 1.03 times the squared-deviation one;
- an AR(1) simulation gives an ESS ratio of 1.63, against theory's 1.62;
- on exact i.i.d. draws at C_untied the corrected check fails 0.035 and 0.048 of replicates (the 200 and the 1,000), against the harness check's 0.025 and 0.048, so the correction does not loosen the check.

Nothing in I2's construction depends on the point except two things: the table's shape k, verified against the Newton solver at every shape the study uses (worst relative error 9.95e-13, Task 2 review), and the Beta site, which is NumPyro's own. So for NUTS chains the single-seed Tier A verdict is a noisy measurement at every point (section 2).

| impl | point | ESS of log a_sq, median / min | ESS of its squared deviations, median / min | ratio (median) | coordinates outside, harness tolerance | coordinates outside, tolerance on the squared deviations' ESS | max z: harness / own ESS |
|---|---|---|---|---|---|---|---|
| I1 | P1 | 2642 / 1664 | 2309 / 1072 | 1.15 | 0 | 0 | 2.95 / 2.76 |
| I2 | P1 | 3850 / 2662 | 3279 / 2451 | 1.2 | 0 | 0 | 2.67 / 2.57 |
| I3 | P1 | 4455 / 3412 | 3228 / 2470 | 1.38 | 1 | 0 | 3 / 2.45 |
| I1 | P2 | 31.4 / 3.76 | 59.2 / 12.5 | 0.54 | 14 | 27 | 4.91 / 8.2 |
| I2 | P2 | 4460 / 3443 | 3406 / 2665 | 1.33 | 0 | 0 | 2.77 / 2.55 |
| I3 | P2 | 4326 / 3309 | 3365 / 2414 | 1.3 | 0 | 0 | 2.84 / 2.6 |
| I1 | P3 | 2508 / 1582 | 1998 / 1063 | 1.21 | 0 | 0 | 2.69 / 2.4 |
| I2 | P3 | 4438 / 3537 | 3324 / 2694 | 1.34 | 0 | 0 | 2.76 / 2.37 |
| I1 | C_tied | 2798 / 1507 | 2587 / 1080 | 1.08 | 2 | 1 | 3.22 / 3.16 |
| I2 | C_tied | 4133 / 3552 | 3342 / 2288 | 1.28 | 0 | 0 | 2.97 / 2.67 |
| I3 | C_tied | 4377 / 3326 | 3335 / 2426 | 1.33 | 0 | 0 | 2.92 / 2.53 |
| I1 | C_untied | 3094 / 2370 | 2348 / 1337 | 1.3 | 0 | 0 | 2.88 / 2.4 |
| I2 | C_untied | 4767 / 4179 | 2879 / 2174 | 1.67 | 5 | 0 | 3.79 / 2.94 |
| I1 | C_untied_k1 | 3994 / 3056 | 2694 / 1827 | 1.49 | 3 | 1 | 3.57 / 3.08 |
| I2 | C_untied_k1 | 4660 / 3618 | 2828 / 2303 | 1.67 | 0 | 0 | 2.98 / 2.38 |

Table S1 (supplementary). The variance check per Tier A chain, under the harness's tolerance (ESS of log a_sq) and under the ESS of the squared deviations. The protocol verdict is the harness's.

| point | replicates (root seed) | null pass rate | variance check failure rate | replicates by coordinates outside the variance check (0 / 1 / 2 / 3 / 4 / 5 or more) |
|---|---|---|---|---|
| P1 | 200 (20260928) | 0.88 | 0.03 | 150 / 44 / 6 / 0 / 0 / 0 |
| P2 | 200 (20260928) | 0.885 | 0.055 | 137 / 52 / 9 / 2 / 0 / 0 |
| P3 | 200 (20260928) | 0.91 | 0.045 | 142 / 49 / 8 / 1 / 0 / 0 |
| C_tied | 200 (20260928) | 0.84 | 0.06 | 142 / 46 / 11 / 1 / 0 / 0 |
| C_untied | 200 (20260928) | 0.835 | 0.025 | 157 / 38 / 4 / 1 / 0 / 0 |
| C_untied_k1 | 200 (20260928) | 0.86 | 0.05 | 163 / 27 / 8 / 2 / 0 / 0 |
| C_untied | 1000 (1) | 0.839 | 0.048 | 781 / 171 / 36 / 6 / 6 / 0 |

Table S2 (supplementary). The null profile of the variance check: the harness's own 200 replicates at every point, whose pass rates reproduce the harness's exactly, plus 1,000 fresh replicates at C_untied.

| point | seed | verdict | failing check: margin | coordinates outside the variance check, on the squared deviations' ESS | ESS log a_sq median / min | log omega z (variance) |
|---|---|---|---|---|---|---|
| C_untied | 1 | pass | - | 0 | 4536 / 3677 | 0.381 |
| C_untied | 2 | pass | - | 0 | 4749 / 3853 | 0.879 |
| C_untied | 3 | pass | - | 0 | 4352 / 3192 | 1.4 |
| C_untied | 4 | pass | - | 1 | 4130 / 3323 | 0.64 |
| C_untied | 5 | pass | - | 0 | 4503 / 3632 | 0.313 |
| C_untied | 6 | pass | - | 0 | 4475 / 3194 | 0.713 |
| C_untied | 7 | fail | mean_log_a_sq: fraction outside 0.08 > 0.01 | 0 | 4498 / 3787 | 0.689 |
| C_untied | 8 | fail | var_log_a_sq: fraction outside 0.03 > 0.01; phi_pairs: pairs outside 1 > 0 | 0 | 4466 / 3594 | 0.9 |
| C_untied | 9 | fail | var_log_a_sq: fraction outside 0.02 > 0.01 | 1 | 3855 / 2801 | 0.192 |
| C_untied_k1 | 1 | fail | var_log_a_sq: fraction outside 0.07 > 0.01; log_omega: z 3.124 > 3 | 1 | 4644 / 3602 | 3.12 |
| C_untied_k1 | 2 | pass | - | 0 | 4333 / 3368 | 1.08 |
| C_untied_k1 | 3 | fail | var_log_a_sq: fraction outside 0.02 > 0.01 | 0 | 4675 / 3818 | 1.01 |
| C_untied_k1 | 4 | pass | - | 0 | 5026 / 4164 | 0.515 |
| C_untied_k1 | 5 | fail | var_log_a_sq: fraction outside 0.02 > 0.01 | 0 | 4893 / 3597 | 0.347 |
| C_untied_k1 | 6 | pass | - | 0 | 4761 / 3776 | 1.2 |
| C_untied_k1 | 7 | pass | - | 0 | 4687 / 3714 | 1.43 |
| C_untied_k1 | 8 | fail | var_log_a_sq: fraction outside 0.03 > 0.01 | 1 | 5088 / 3789 | 1.23 |
| C_untied_k1 | 9 | fail | var_log_a_sq: fraction outside 0.02 > 0.01; ks: min p 0.009865 < 0.01 | 0 | 4594 / 3660 | 1.21 |
| P1 | 1 | pass | - | 0 | 3798 / 2897 | 0.104 |
| P1 | 2 | fail | mean_log_a_sq: fraction outside 0.02 > 0.01 | 0 | 3910 / 2927 | 1.87 |
| P1 | 3 | pass | - | 0 | 3681 / 2926 | 0.875 |
| P1 | 4 | pass | - | 0 | 3792 / 3020 | 0.245 |
| P1 | 5 | fail | var_log_a_sq: fraction outside 0.02 > 0.01 | 1 | 3997 / 3260 | 1.47 |
| P1 | 6 | fail | var_log_a_sq: fraction outside 0.02 > 0.01 | 2 | 3792 / 2956 | 1.88 |
| P1 | 7 | pass | - | 0 | 3600 / 3047 | 0.788 |
| P1 | 8 | pass | - | 0 | 4071 / 3281 | 0.965 |
| P1 | 9 | pass | - | 0 | 3665 / 2798 | 0.406 |

Table S3 (supplementary). I2 at the Tier A budget (512 + 4,000 draws, depth 10), seeds 1-9, through the harness's `tier_a_statistics`. Seed 0 is the protocol chain in Table A.

## 1. The prior and the three parameterizations

The prior has p = D coordinates, hyperparameters (a, b, alpha), and per-coordinate concentration k = alpha / p:

    R2 ~ Beta(a, b),   phi ~ Dir(k, ..., k),   omega = R2 / (1 - R2) ~ Beta-prime(a, b),   a_sq_i = omega * phi_i

There is no sigma^2 coupling. R2 and phi are independent for every (a, alpha). The paper's tie a = alpha (= p a_pi) gives the Gamma-Gamma form: xi ~ Gamma(b), lam_i ~ Gamma(k), a_sq_i = lam_i / xi, with closed-form marginals. Every a_sq statistic below is taken on the log scale or through quantiles, because E omega is infinite for b ≤ 1.

| | sampled sites (support: gate scale) | construction of log a_sq | exact | represents | what NUTS sees on the prior |
|---|---|---|---|---|---|
| **I1**, the supervisor's reference without `z_beta` (the control implementation) | `r2d2_R2` (unit interval: logit), `r2d2_raw` (real) | The `r2d2_raw` ~ N(0, 1) proposal is corrected, by the factor `r2d2_raw_log_density` (Jacobian included), to the law of raw = k (log lam - log k) with lam ~ Gamma(k). Then log phi = log lam - logsumexp(log lam) and log a_sq = logit R2 + log phi. | under NUTS, yes (NUTS reads the factor); under `Predictive`, no (section 4) | any (a, b, alpha) | a log-Gamma(k) density per coordinate, with left-tail scale 1 and wall curvature 1/k: an affine map of NumPyro's default log transform, which an adapted dense mass matrix cannot change |
| **I2**, the Gaussian copula in reference form (`sagp.r2d2.sample_reference`, form `"reference"`) | `r2d2_R2` (unit interval: logit), `r2d2_z_lam` (real) | z ~ N(0, I) and y = log F_k^-1(Phi(z)), computed by `sagp.r2d2.log_gamma_icdf`, a loop-free quintic Hermite table built per shape from the Newton solver. Then log a_sq = logit R2 + y - logsumexp(y). | yes | any (a, b, alpha) | a standard normal in p coordinates, plus one logit-Beta coordinate |
| **I3**, the tied Gamma-Gamma copula (`sagp.r2d2.sample_tied`, form `"tied"`) | `r2d2_z_xi` (real), `r2d2_z_lam` (real) | log a_sq = log F_k^-1(Phi(z_lam)) - log F_b^-1(Phi(z_xi)), with no normalization | yes | only a = alpha (it raises otherwise): P1, P2, C_tied | exactly N(0, I_{p+1}), whatever (k, b) |
| **HC**, the control (`sagp/gp.py:380-384`) | `kernel_tausq`, `_a_sq` (positive: log) | a_sq = tausq * _a_sq, with tausq ~ HC(ALPHA_AMPLITUDE = 0.01312) and _a_sq ~ HC(1) | - | - | the four half-Cauchy cells' own prior |

Several departures from the 2026-09-25 brief were made upstream and are carried here:

- I2 and I3 are the cells' own code in `sagp/r2d2.py` (plan D1, Task 2), not the brief's `experiments/`-only module.
- Inside the model, a table replaces the brief's Newton iteration. The solver builds the table and is its accuracy reference; the worst relative error is 9.95e-13.
- I2's share site is `r2d2_z_lam`, shared with I3, not the brief's `r2d2_z`.
- The models expose one deterministic site, `log_a_sq`, instead of `a_sq`, `phi`, `R2` and `log_omega`, because a_sq underflows at k = 0.01. I1 also returns log a_sq.
- `sagp/diagnostics.py` has no unit-interval category. The harness gates `r2d2_R2` on its logit, which is NumPyro's own unconstrained coordinate for a Beta site.

The points, as (a, b, alpha):

- P1 = (0.2093 p, 0.5, 0.2093 p), tied.
- P2 = (1, 1, 1), tied, with k = 1/p.
- P3 = (2, 2, 0.2093 p), untied.
- At p = 100: C_tied = (9.379, 0.5, 9.379), tied, k = 0.09379; C_untied = (1.578, 0.8044, 49.94), k = 0.4994; and C_untied_k1 = (3.049, 1.674, 99.81), k = 0.9981.

At p = 30 the C points keep k, as P1 and P3 do: C_tied keeps its tie (a = 30 k), and the untied points keep a and b.

## 2. Tier A: exactness (p = 100, seed 0, 512 + 4,000 draws, depth 10)

| impl | point | k | verdict | failing check: margin | null pass rate | ESS log a_sq median / min | div | fraction outside: mean / var | log R2 max z | log omega max z | phi pairs outside | KS min p (series, n) | count q25/q50/q75: chain vs truth | p_active z | mean steps | at cap | wall s | agrees with scouting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| I1 | P1 | 0.2093 | **pass** | - | 0.88 | 2642 / 1664 | 0 | 0 / 0 | 0.731 | 0.451 | 0 | 0.388 (log_a_sq_1, n 1663) | 55/65/75 vs 56/65/75 | 0.404 | 124 | 0 | 9.13 | agrees: the probe's A2 at P1, depth 10, two 256-draw chains: KS p 0.21-0.91, count quartiles 55/64/77 and 54/64/74 against 55/65/75 |
| I2 | P1 | 0.2093 | **pass** | - | 0.88 | 3850 / 2662 | 0 | 0 / 0 | 1.07 | 1.66 | 0 | 0.305 (log_a_sq_1, n 2662) | 56/65/76 vs 56/65/75 | 1.08 | 16.6 | 0 | 5.29 | agrees in verdict: item 6 (v) ref form, 4,000 draws at depth 6 (not 10): E log a_sq -3.198 against -3.09, Var 28.92 against 29.0, ESS median/min 3,204/1,614 |
| I3 | P1 | 0.2093 | **pass** | - | 0.88 | 4455 / 3412 | 0 | 0.01 / 0.01 | 0.247 | 1.14 | 0 | 0.375 (logit_R2, n 3412) | 55/65/74 vs 56/65/75 | 1.14 | 17.9 | 0 | 4.21 | agrees in verdict: item 6 (v) Gamma-Gamma, 256 draws at depth 6: E -3.074, Var 28.85 (targets -3.09, 29.0) |
| I1 | P2 | 0.01 | **not evaluable** | not evaluable: ESS_min 3.76 < 100 (checks at that ESS: mean_log_a_sq: fraction outside 0.05 > 0.01; var_log_a_sq: fraction outside 0.14 > 0.01; count: median gap 2 > 1 or q25 gap 1 > 3 or q75 gap 1 > 3) | 0.885 | 31.4 / 3.76 | 4 | 0.05 / 0.14 | 2.08 | 1.76 | 0 | 0.161 (log_max_phi, n 3) | 1/2/4 vs 2/4/5 | 1.48 | 1023 | 1 | 31.3 | agrees: the probe found P2 untestable by NUTS (A2 at depth 10: r_hat 2.605, ESS 3.1, at the cap); item 6 (i), 4,000 draws at depth 6: 33 % divergent, Var 2,786 against 10,003 |
| I2 | P2 | 0.01 | **pass** | - | 0.885 | 4460 / 3443 | 0 | 0 / 0 | 1.15 | 1.21 | 0 | 0.144 (log_max_phi, n 3442) | 2/4/5 vs 2/4/5 | 1.03 | 16.2 | 0 | 3.86 | agrees in verdict: item 6 (v) ref form, 4,000 draws at depth 6: E -100.15 against -99.98, Var 10,081 against 10,003, ESS 3,900/2,676 |
| I3 | P2 | 0.01 | **pass** | - | 0.885 | 4326 / 3309 | 0 | 0 / 0 | 0.634 | 1.17 | 0 | 0.135 (log_max_phi, n 3309) | 2/4/5 vs 2/4/5 | 0.958 | 17.9 | 0 | 3.49 | agrees in verdict: item 6 (v) Gamma-Gamma, 256 draws at depth 6: E -100.36, Var 10,179 |
| I1 | P3 | 0.2093 | **pass** | - | 0.91 | 2508 / 1582 | 0 | 0 / 0 | 0.355 | 0.206 | 0 | 0.354 (log_max_phi, n 1581) | 7/15/25 vs 7/15/25 | - | 122 | 0 | 6.83 | no scouting number (not run there) |
| I2 | P3 | 0.2093 | **pass** | - | 0.91 | 4438 / 3537 | 0 | 0 / 0 | 1.56 | 1.48 | 0 | 0.308 (logit_R2, n 3536) | 7/15/25 vs 7/15/25 | - | 17.3 | 0 | 3.32 | no scouting number (not run there) |
| I1 | C_tied | 0.09379 | **fail** | var_log_a_sq: fraction outside 0.02 > 0.01 | 0.84 | 2798 / 1507 | 0 | 0 / 0.02 | 2.2 | 1.75 | 0 | 0.2 (logit_R2, n 1507) | 30/38/46 vs 30/37/46 | 2.43 | 442 | 0.059 | 15.9 | no scouting number (not run there) |
| I2 | C_tied | 0.09379 | **pass** | - | 0.84 | 4133 / 3552 | 0 | 0 / 0 | 0.713 | 2.12 | 0 | 0.04 (logit_R2, n 3552) | 30/37/47 vs 30/37/46 | 1.93 | 19.3 | 0 | 3.66 | no scouting number (not run there) |
| I3 | C_tied | 0.09379 | **pass** | - | 0.84 | 4377 / 3326 | 0 | 0 / 0 | 0.495 | 1.14 | 0 | 0.0929 (logit_R2, n 3325) | 30/37/45 vs 30/37/46 | 1.23 | 17.9 | 0 | 3.39 | no scouting number (not run there) |
| I1 | C_untied | 0.4994 | **pass** | - | 0.835 | 3094 / 2370 | 0 | 0 / 0 | 1.8 | 1.48 | 0 | 0.176 (log_a_sq_1, n 2370) | 14/38/63 vs 14/37/61 | - | 49.7 | 0 | 4.91 | no scouting number (not run there) |
| I2 | C_untied | 0.4994 | **fail** | var_log_a_sq: fraction outside 0.05 > 0.01 | 0.835 | 4767 / 4179 | 0 | 0 / 0.05 | 1.77 | 2.64 | 0 | 0.012 (logit_R2, n 4000) | 13/38/63 vs 14/37/61 | - | 15.2 | 0 | 3.66 | no scouting number (not run there) |
| I1 | C_untied_k1 | 0.9981 | **fail** | var_log_a_sq: fraction outside 0.03 > 0.01 | 0.86 | 3994 / 3056 | 0 | 0 / 0.03 | 2.04 | 1.38 | 0 | 0.374 (log_phi_1, n 3055) | 14/38/61 vs 14/37/62 | - | 30 | 0 | 4.03 | no scouting number (not run there) |
| I2 | C_untied_k1 | 0.9981 | **pass** | - | 0.86 | 4660 / 3618 | 0 | 0 / 0 | 0.439 | 1.61 | 0 | 0.013 (log_max_phi, n 3618) | 14/37/62 vs 14/37/62 | - | 16.8 | 0 | 3.75 | no scouting number (not run there) |

Table A. The tolerances are binding:

- Moments per coordinate: the mean within 3 sqrt(Var / ESS_i) and the variance within 3 Var sqrt((kappa_i - 1) / ESS_i), with at most 1 % of coordinates outside either.
- log R2 and log omega within 3 sigma on their own ESS; the column gives the larger of the mean z and the variance z.
- Var(log phi_i - log phi_j) = 2 psi'(k) on 5 pairs, with none outside.
- KS p ≥ 0.01 for logit R2, log phi_1, log a_sq_1 and log max phi, each on ESS_min equally spaced draws against 2e5 ground-truth draws. The column gives the smallest p.
- The count median within 1 of the ground truth's, and its quartiles within 3.
- At a tied point, P(a_sq_i > 0.02) within 3 standard errors of the exact value.

A run with ESS_min below 100, or with more than 1 % of its draws divergent, is not evaluable. Each row carries its point's null pass rate: the same verdict on 200 exact i.i.d. replicates (R16).

Reading (R16). 11 of the 15 chains pass. Of the four that do not, one is not evaluable and three fail:

- I1 at P2 is not evaluable. Its ESS_min is 3.76 (median 31.4), 99.9 % of its iterations hit the 1,023-step cap, and it has 4 divergences. The tolerance it would need is 4.66 sigma on the means and 4.91 on the variances (need_k). Its count quartiles, (1, 2, 4) against (2, 4, 5), show a stuck chain.
- I1 at C_tied and at C_untied_k1 fails the variance check, with 2 and 3 coordinates outside. Both are within the null's false-fail profile at those points: the null has 2 or more coordinates outside in 6.0 % and 5.0 % of replicates, and 3 in 0.5 % and 1.0 % (Table S2). They are reported as false fails, not as evidence against I1's density.
- I2 at C_untied: see the G0 section.

The null pass rates run from 0.835 to 0.91. The brief's tolerances fail exact i.i.d. draws 9 to 16.5 % of the time per point: about 11 checks are ANDed, and the coordinates are correlated through log omega. For NUTS chains the rate is higher; 5 to 7 of 10 I2 chains pass the protocol verdict (Table S3).

Where I1 is evaluable it is exact: at P1, P3 and C_untied, and at C_tied and C_untied_k1 up to the variance false fail. But even at depth 10 it is slow: 124 and 122 mean steps at P1 and P3, and 442 at C_tied, against 15-19 for I2 and I3.

## 3. Tier B: the fixed budget (512 warm-up + 256 draws, dense mass, one chain)

Depth 6, the study's budget:

| impl | point | p | gate passed | log a_sq gate passed | r_hat max (s0/s1/s2) | ESS min, sites | ESS min, log a_sq | div | mean steps | at cap | wall s | limiting site | failing criteria | agrees with scouting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HC | - | 30 | **3/3** | 3/3 | 1.01/1.02/1.02 | 165/159/135 | 244/156/168 | 0/0/0 | 14.5/12/14.6 | 0/0/0 | 2.2/0.67/0.62 | _a_sq | - | no scouting number (not run there) |
| HC | - | 100 | **3/3** | 3/3 | 1.02/1.04/1.05 | 73/83.9/55.7 | 62.9/120/86 | 0/0/0 | 61.9/61.2/63 | 0.97/0.95/1 | 1.9/0.94/1 | _a_sq | - | agrees to every printed digit for seeds 0-1 (probe model C on the log scale: r_hat 1.023/1.043, ESS 73.0/83.9, log a_sq ESS 62.9/120.4, 61.9/61.2 steps); seed 2 is new |
| I1 | P1 | 30 | **2/3** | 3/3 | 1.13/1.07/1.04 | 40.8/30.4/42.3 | 37.3/29/46.8 | 0/0/0 | 40.1/37.3/40.1 | 0.29/0.23/0.3 | 2.2/0.71/0.75 | r2d2_raw | s0: r_hat_max 1.128 > 1.1 | no scouting number (not run there) |
| I1 | P2 | 30 | **0/3** | 0/3 | 1.33/1.34/1.35 | 8.53/6.27/7.54 | 8.51/6.11/7.59 | 11/8/11 | 61.6/62.2/61.5 | 0.95/0.97/0.95 | 0.78/1/0.8 | r2d2_raw | s0: r_hat_max 1.327 > 1.1; n_eff_min 8.5 < 16.0; divergences 11 > 5; s1: r_hat_max 1.337 > 1.1; n_eff_min 6.3 < 16.0; divergences 8 > 5; s2: r_hat_max 1.350 > 1.1; n_eff_min 7.5 < 16.0; divergences 11 > 5 | no scouting number (not run there) |
| I1 | P3 | 30 | **3/3** | 3/3 | 1.08/1.04/1.06 | 46.9/45.1/42.8 | 44.9/46.9/41.4 | 0/0/0 | 50.1/49.1/35.6 | 0.6/0.56/0.16 | 0.79/0.9/0.89 | r2d2_raw | - | no scouting number (not run there) |
| I1 | C_tied | 30 | **2/3** | 2/3 | 1.08/1.09/1.3 | 20.6/17.8/11.6 | 20.5/19/13.2 | 0/0/0 | 62.9/62.5/63 | 1/0.98/1 | 0.84/0.85/0.87 | r2d2_raw | s2: r_hat_max 1.297 > 1.1; n_eff_min 11.6 < 16.0 | no scouting number (not run there) |
| I1 | C_untied | 30 | **3/3** | 3/3 | 1.03/1.1/1.06 | 34.6/17.2/33.9 | 41.4/23.8/41.3 | 0/0/0 | 21.2/17/16.8 | 0.0039/0/0 | 0.77/0.76/0.88 | r2d2_raw | - | no scouting number (not run there) |
| I1 | C_untied_k1 | 30 | **3/3** | 3/3 | 1.03/1.04/1.05 | 95.5/42.3/122 | 123/52.5/139 | 0/0/0 | 15/14.6/13.7 | 0/0/0 | 0.77/0.79/0.84 | r2d2_raw | - | no scouting number (not run there) |
| I2 | P1 | 30 | **3/3** | 3/3 | 1.01/1.02/1.02 | 100/117/185 | 145/147/172 | 0/0/0 | 9.06/8.09/13.2 | 0/0/0 | 2.8/0.74/0.73 | r2d2_R2 | - | no scouting number (not run there) |
| I2 | P2 | 30 | **3/3** | 3/3 | 1.01/1/1.02 | 122/150/201 | 138/135/130 | 0/0/0 | 7.97/7.31/7.91 | 0/0/0 | 1/0.77/0.71 | r2d2_z_lam | - | no scouting number (not run there) |
| I2 | P3 | 30 | **3/3** | 3/3 | 1.01/1.02/1.03 | 220/175/185 | 173/114/122 | 0/0/0 | 7.53/7.56/8.72 | 0/0/0 | 1/0.74/0.78 | r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_tied | 30 | **3/3** | 3/3 | 1.01/1.03/1.03 | 159/127/149 | 154/117/135 | 0/0/0 | 10/7.34/7.41 | 0/0/0 | 1.2/0.74/0.85 | r2d2_R2, r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_untied | 30 | **3/3** | 3/3 | 1.01/1.01/1.02 | 236/192/133 | 246/160/124 | 0/0/0 | 7.59/8.5/7.66 | 0/0/0 | 1/0.78/0.8 | r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_untied_k1 | 30 | **3/3** | 3/3 | 1.01/1.01/1.02 | 159/199/111 | 203/160/170 | 0/0/0 | 7.41/7.81/7.66 | 0/0/0 | 0.98/0.74/0.73 | r2d2_z_lam | - | no scouting number (not run there) |
| I3 | P1 | 30 | **3/3** | 3/3 | 1.02/1/1.02 | 152/184/134 | 175/139/182 | 0/0/0 | 7.84/7.31/8.69 | 0/0/0 | 1.7/0.69/0.73 | r2d2_z_lam | - | no scouting number (not run there) |
| I3 | P2 | 30 | **3/3** | 3/3 | 1.02/1/1.02 | 152/184/134 | 177/145/136 | 0/0/0 | 7.84/7.31/8.69 | 0/0/0 | 0.97/0.87/0.85 | r2d2_z_lam | - | no scouting number (not run there) |
| I3 | C_tied | 30 | **3/3** | 3/3 | 1.02/1/1.02 | 152/184/134 | 173/140/160 | 0/0/0 | 7.84/7.31/8.69 | 0/0/0 | 0.86/0.84/0.71 | r2d2_z_lam | - | no scouting number (not run there) |
| I1 | P1 | 100 | **0/3** | 0/3 | 1.37/1.4/1.31 | 7/6.67/6.5 | 7.01/6.16/7.9 | 0/0/0 | 63/63/63 | 1/1/1 | 1.6/1.1/1.1 | r2d2_raw | s0: r_hat_max 1.375 > 1.1; n_eff_min 7.0 < 16.0; s1: r_hat_max 1.397 > 1.1; n_eff_min 6.7 < 16.0; s2: r_hat_max 1.307 > 1.1; n_eff_min 6.5 < 16.0 | agrees to every printed digit for seeds 0-1 (probe A2: r_hat 1.375/1.397, ESS 7.0/6.7, log a_sq ESS 7.0/6.2, 63 steps, all capped; item 6 (i): E log a_sq -2.33, Var 22.9, coordinate-1 ESS 20, this chain's values); seed 2 is new |
| I1 | P2 | 100 | **0/3** | 0/3 | 2.77/3.6/2.71 | 3.09/2.76/3.25 | 3.15/2.76/3.23 | 21/14/7 | 61.3/61.4/62.6 | 0.92/0.94/0.97 | 1.1/0.95/0.93 | r2d2_raw | s0: r_hat_max 2.770 > 1.1; n_eff_min 3.1 < 16.0; divergences 21 > 5; s1: r_hat_max 3.603 > 1.1; n_eff_min 2.8 < 16.0; divergences 14 > 5; s2: r_hat_max 2.707 > 1.1; n_eff_min 3.3 < 16.0; divergences 7 > 5 | agrees to every printed digit for seeds 0-1 (probe A2: r_hat 2.770/3.603, ESS 3.1/2.8, 21/14 divergences, 61.3/61.4 steps; item 6 (i): E -64.0, Var 2,783, ESS 5, this chain's values); seed 2 is new |
| I1 | P3 | 100 | **0/3** | 0/3 | 1.21/1.33/1.29 | 8.93/5.43/6.64 | 9.25/5.19/6.54 | 0/0/0 | 63/63/63 | 1/1/1 | 1.1/1.4/1 | r2d2_raw | s0: r_hat_max 1.214 > 1.1; n_eff_min 8.9 < 16.0; s1: r_hat_max 1.326 > 1.1; n_eff_min 5.4 < 16.0; s2: r_hat_max 1.286 > 1.1; n_eff_min 6.6 < 16.0 | no scouting number (not run there) |
| I1 | C_tied | 100 | **0/3** | 0/3 | 1.39/1.54/1.67 | 6.53/4.93/4.69 | 6.58/4.66/4.71 | 0/1/0 | 63/62.9/63 | 1/1/1 | 1.1/1.1/1.1 | r2d2_raw | s0: r_hat_max 1.387 > 1.1; n_eff_min 6.5 < 16.0; s1: r_hat_max 1.538 > 1.1; n_eff_min 4.9 < 16.0; s2: r_hat_max 1.675 > 1.1; n_eff_min 4.7 < 16.0 | no scouting number (not run there) |
| I1 | C_untied | 100 | **0/3** | 1/3 | 1.13/1.21/1.16 | 16/9.54/13.2 | 25.5/14.9/14.2 | 0/0/0 | 63/63/63 | 1/1/1 | 1.1/1.1/1 | r2d2_raw | s0: r_hat_max 1.126 > 1.1; s1: r_hat_max 1.208 > 1.1; n_eff_min 9.5 < 16.0; s2: r_hat_max 1.162 > 1.1; n_eff_min 13.2 < 16.0 | no scouting number (not run there) |
| I1 | C_untied_k1 | 100 | **1/3** | 2/3 | 1.05/1.19/1.1 | 48.7/13.1/21.4 | 87.3/21.3/28.5 | 0/0/0 | 63/62.9/63 | 1/1/1 | 1.2/1.1/1.1 | r2d2_raw | s1: r_hat_max 1.190 > 1.1; n_eff_min 13.1 < 16.0; s2: r_hat_max 1.103 > 1.1 | no scouting number (not run there) |
| I2 | P1 | 100 | **3/3** | 3/3 | 1.04/1.04/1.03 | 102/94.4/87.3 | 100/115/61.1 | 0/0/0 | 61.2/55.2/57.6 | 0.95/0.76/0.84 | 2.2/1.1/1.2 | r2d2_z_lam | - | agrees: item 6 (v) ref form's E log a_sq -3.090, Var 27.3, median ESS 231 and 61.2 steps are this seed-0 chain's to every digit; its minimum ESS of about 140 is not a computed value, and the same chain's minimum is 100.2 (section 8) |
| I2 | P2 | 100 | **3/3** | 3/3 | 1.02/1.02/1.04 | 122/141/102 | 143/144/130 | 0/0/0 | 48.5/51.5/47.2 | 0.58/0.68/0.53 | 1.6/1.2/1 | r2d2_z_lam | - | agrees in verdict, steps (48.5) and median ESS (255 against 254); item 6's E -98.96 and Var 9,980 differ from this seed-0 chain's -99.21 and 9,962 (section 8) |
| I2 | P3 | 100 | **3/3** | 3/3 | 1.02/1.02/1.02 | 86/71.7/95.2 | 105/95.4/116 | 0/0/0 | 53.5/50.8/55.2 | 0.72/0.63/0.75 | 1.5/1.4/1.2 | r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_tied | 100 | **3/3** | 3/3 | 1.05/1.02/1.07 | 103/124/96.3 | 122/102/93.8 | 0/0/0 | 59.8/56.6/60.6 | 0.91/0.79/0.93 | 1.5/1.3/1.1 | r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_untied | 100 | **3/3** | 3/3 | 1.03/1.01/1.04 | 163/124/159 | 150/150/89.1 | 0/0/0 | 52.5/51.9/52.4 | 0.67/0.67/0.68 | 1.1/1.1/1.1 | r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_untied_k1 | 100 | **3/3** | 3/3 | 1.03/1.03/1.03 | 133/78.8/95.6 | 137/105/188 | 0/0/0 | 59.9/47.4/52.6 | 0.9/0.52/0.67 | 1.3/1.1/1.2 | r2d2_z_lam | - | no scouting number (not run there) |
| I3 | P1 | 100 | **3/3** | 3/3 | 1.03/1.04/1.04 | 150/136/108 | 123/137/127 | 0/0/0 | 52.4/48.4/54.2 | 0.7/0.59/0.75 | 1.1/1.1/1 | r2d2_z_lam | - | agrees to every printed digit: item 6 (v) Gamma-Gamma's E -3.074, Var 28.85, median ESS 243, 52.4 steps and step size 0.40 are this seed-0 chain's |
| I3 | P2 | 100 | **3/3** | 3/3 | 1.03/1.04/1.04 | 150/136/108 | 136/123/110 | 0/0/0 | 52.4/48.4/54.2 | 0.7/0.59/0.75 | 1.1/0.98/1.1 | r2d2_z_lam | - | agrees in verdict, steps (52.4) and median ESS (254); item 6's E -100.36 and Var 10,179 differ from this seed-0 chain's -100.60 and 10,161 (section 8) |
| I3 | C_tied | 100 | **3/3** | 3/3 | 1.03/1.04/1.04 | 150/136/108 | 140/125/114 | 0/0/0 | 52.4/48.4/54.2 | 0.7/0.59/0.75 | 1/1.1/1.6 | r2d2_z_lam | - | no scouting number (not run there) |

Table B6.

Depth 10, for context:

| impl | point | p | gate passed | log a_sq gate passed | r_hat max (s0/s1/s2) | ESS min, sites | ESS min, log a_sq | div | mean steps | at cap | wall s | limiting site | failing criteria | agrees with scouting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| HC | - | 30 | **3/3** | 3/3 | 1.01/1.02/1.01 | 176/140/204 | 282/219/198 | 0/0/0 | 14.7/13.4/14 | 0/0/0 | 0.93/0.98/0.82 | _a_sq | - | no scouting number (not run there) |
| HC | - | 100 | **3/3** | 3/3 | 1.02/1.02/1.02 | 132/147/163 | 173/144/137 | 0/0/0 | 25.6/24.7/26.6 | 0/0/0 | 3.3/3.2/3.2 | _a_sq | - | no scouting number (not run there) |
| I1 | P1 | 30 | **3/3** | 3/3 | 1.09/1.09/1.1 | 33.7/18.5/29.2 | 45.6/19.7/32.1 | 0/0/0 | 44.1/35.1/40.2 | 0/0/0 | 1.1/1/1.1 | r2d2_raw | - | no scouting number (not run there) |
| I1 | P2 | 30 | **2/3** | 2/3 | 1.04/1.12/1.02 | 36/16.6/32.2 | 36.6/17.1/32.1 | 0/0/0 | 214/204/232 | 0/0/0 | 2/1.9/1.8 | r2d2_raw | s1: r_hat_max 1.118 > 1.1 | no scouting number (not run there) |
| I1 | P3 | 30 | **2/3** | 2/3 | 1.05/1.11/1.03 | 47.8/18.3/34.5 | 50.6/18.2/36.8 | 0/0/0 | 37.7/42.9/51.2 | 0/0/0 | 1.1/1.4/1.2 | r2d2_raw | s1: r_hat_max 1.108 > 1.1 | no scouting number (not run there) |
| I1 | C_tied | 30 | **2/3** | 2/3 | 1.03/1.07/1.1 | 38.1/25.5/34.2 | 36.6/26.3/33.8 | 0/0/0 | 95.5/67.5/74.4 | 0/0/0 | 1.3/1.4/1.3 | r2d2_raw | s2: r_hat_max 1.101 > 1.1 | no scouting number (not run there) |
| I1 | C_untied | 30 | **3/3** | 3/3 | 1.05/1.07/1.04 | 54.6/47/55.9 | 77.5/65.4/76.4 | 0/0/0 | 22.1/21.2/21 | 0/0/0 | 1/1.2/1 | r2d2_raw | - | no scouting number (not run there) |
| I1 | C_untied_k1 | 30 | **3/3** | 3/3 | 1.02/1.04/1.03 | 94.6/82.9/37.3 | 107/113/80.1 | 0/0/0 | 15/15.1/15.2 | 0/0/0 | 1.1/0.97/1.2 | r2d2_raw | - | no scouting number (not run there) |
| I2 | P1 | 30 | **3/3** | 3/3 | 1/1.02/1.02 | 165/158/157 | 150/129/137 | 0/0/0 | 8.62/13/11.2 | 0/0/0 | 1.1/0.94/0.96 | r2d2_R2, r2d2_z_lam | - | no scouting number (not run there) |
| I2 | P2 | 30 | **3/3** | 3/3 | 1.01/1/1.02 | 167/144/162 | 218/156/119 | 0/0/0 | 8.44/8.91/7.97 | 0/0/0 | 1.1/0.79/0.93 | r2d2_z_lam | - | no scouting number (not run there) |
| I2 | P3 | 30 | **3/3** | 3/3 | 1.01/1.01/1.01 | 204/153/211 | 217/164/204 | 0/0/0 | 7.69/8.03/7.47 | 0/0/0 | 1.1/0.94/1.6 | r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_tied | 30 | **3/3** | 3/3 | 1.02/1.01/1.01 | 86.8/139/133 | 195/115/113 | 0/0/0 | 10.3/10.7/7.94 | 0/0/0 | 0.94/1/0.86 | r2d2_R2, r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_untied | 30 | **3/3** | 3/3 | 1/1.02/1.01 | 214/105/175 | 217/123/173 | 0/0/0 | 7.53/8.75/7.12 | 0/0/0 | 1/1.1/0.86 | r2d2_R2, r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_untied_k1 | 30 | **3/3** | 3/3 | 1.01/1.03/1.02 | 175/129/160 | 245/146/201 | 0/0/0 | 9.03/9.56/7.25 | 0/0/0 | 0.86/0.97/1.1 | r2d2_z_lam | - | no scouting number (not run there) |
| I3 | P1 | 30 | **3/3** | 3/3 | 1.01/1.01/1.02 | 230/152/241 | 223/159/228 | 0/0/0 | 7.19/7.66/7.19 | 0/0/0 | 0.89/0.92/0.99 | r2d2_z_lam | - | no scouting number (not run there) |
| I3 | P2 | 30 | **3/3** | 3/3 | 1.01/1.01/1.02 | 230/152/241 | 225/144/193 | 0/0/0 | 7.19/7.66/7.19 | 0/0/0 | 0.86/0.9/0.91 | r2d2_z_lam | - | no scouting number (not run there) |
| I3 | C_tied | 30 | **3/3** | 3/3 | 1.01/1.01/1.02 | 230/152/241 | 219/150/198 | 0/0/0 | 7.19/7.66/7.19 | 0/0/0 | 0.92/0.85/0.87 | r2d2_z_lam | - | no scouting number (not run there) |
| I1 | P1 | 100 | **3/3** | 3/3 | 1.03/1.04/1.06 | 90.7/52.9/63.8 | 111/55.8/64.7 | 0/0/0 | 119/124/151 | 0/0/0 | 3.9/4.1/4.1 | r2d2_raw | - | agrees to every printed digit for seeds 0-1 (probe A2: r_hat 1.031/1.041, ESS 90.7/52.9, log a_sq ESS 111.4/55.8, 118.9/123.6 steps); seed 2 is new |
| I1 | P2 | 100 | **0/3** | 0/3 | 2.61/2.51/2.29 | 3.11/3.03/3.42 | 3.11/3.03/3.42 | 1/184/3 | 1020/578/1017 | 1/0.25/0.99 | 4.4/3.3/5.1 | r2d2_raw | s0: r_hat_max 2.605 > 1.1; n_eff_min 3.1 < 16.0; s1: r_hat_max 2.507 > 1.1; n_eff_min 3.0 < 16.0; divergences 184 > 5; s2: r_hat_max 2.287 > 1.1; n_eff_min 3.4 < 16.0 | agrees to every printed digit for seed 0 (probe A2, supplementary: r_hat 2.605, ESS 3.1, 1 divergence, 1,020 steps, 0.996 capped); seeds 1-2 are new |
| I1 | P3 | 100 | **3/3** | 3/3 | 1.03/1.05/1.04 | 60.1/70.1/50.7 | 65.2/75.1/51.4 | 0/0/0 | 126/110/132 | 0/0/0 | 4.2/4/3.8 | r2d2_raw | - | no scouting number (not run there) |
| I1 | C_tied | 100 | **3/3** | 3/3 | 1.04/1.04/1.06 | 98.2/93.8/73.8 | 102/102/66.4 | 0/0/0 | 444/712/428 | 0.07/0.44/0.039 | 5.4/4.9/4.1 | r2d2_R2, r2d2_raw | - | no scouting number (not run there) |
| I1 | C_untied | 100 | **3/3** | 3/3 | 1.04/1.04/1.05 | 89/78.9/90.8 | 90.9/90/127 | 0/0/0 | 49.8/51.3/48.9 | 0/0/0 | 3.5/3.5/3.4 | r2d2_raw | - | no scouting number (not run there) |
| I1 | C_untied_k1 | 100 | **3/3** | 3/3 | 1.03/1.02/1.02 | 97/75.6/72.5 | 111/99.3/84.4 | 0/0/0 | 30/32.6/28.6 | 0/0/0 | 3.3/3.4/3.2 | r2d2_raw | - | no scouting number (not run there) |
| I2 | P1 | 100 | **3/3** | 3/3 | 1.01/1.03/1.03 | 150/133/153 | 152/120/135 | 0/0/0 | 16.5/18.9/17.8 | 0/0/0 | 2.7/2.9/2.7 | r2d2_R2, r2d2_z_lam | - | no scouting number (not run there) |
| I2 | P2 | 100 | **3/3** | 3/3 | 1.02/1.01/1.03 | 144/167/175 | 133/140/157 | 0/0/0 | 16.1/19.1/18.4 | 0/0/0 | 3.1/2.9/3.2 | r2d2_z_lam | - | no scouting number (not run there) |
| I2 | P3 | 100 | **3/3** | 3/3 | 1.01/1.04/1.03 | 147/117/118 | 142/117/128 | 0/0/0 | 16.8/15.4/19.4 | 0/0/0 | 3/2.9/3 | r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_tied | 100 | **3/3** | 3/3 | 1.03/1.02/1.03 | 130/94.9/149 | 148/144/142 | 0/0/0 | 19.3/20.7/20.8 | 0/0/0 | 2.9/3/3.2 | r2d2_R2, r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_untied | 100 | **3/3** | 3/3 | 1.02/1.04/1.01 | 189/149/161 | 162/127/153 | 0/0/0 | 15.1/17.4/16.6 | 0/0/0 | 3.1/3.5/3.3 | r2d2_z_lam | - | no scouting number (not run there) |
| I2 | C_untied_k1 | 100 | **3/3** | 3/3 | 1.02/1.03/1.01 | 148/124/148 | 175/112/117 | 0/0/0 | 17/17.2/19.6 | 0/0/0 | 3.1/3.3/3.3 | r2d2_R2, r2d2_z_lam | - | no scouting number (not run there) |
| I3 | P1 | 100 | **3/3** | 3/3 | 1.02/1.03/1.01 | 149/149/173 | 139/136/118 | 0/0/0 | 18.4/15.8/18.5 | 0/0/0 | 3.1/2.7/2.9 | r2d2_z_lam | - | no scouting number (not run there) |
| I3 | P2 | 100 | **3/3** | 3/3 | 1.02/1.03/1.01 | 149/149/173 | 131/133/147 | 0/0/0 | 18.4/15.8/18.5 | 0/0/0 | 2.9/3/3.7 | r2d2_z_lam | - | no scouting number (not run there) |
| I3 | C_tied | 100 | **3/3** | 3/3 | 1.02/1.03/1.01 | 149/149/173 | 130/133/127 | 0/0/0 | 18.4/15.8/18.5 | 0/0/0 | 3/2.7/2.8 | r2d2_z_lam | - | no scouting number (not run there) |

Table B10.

In both tables there is one row per implementation, point and p, and each cell lists seeds 0/1/2. The gate passes when r_hat max ≤ 1.1, ESS min ≥ 16 and divergences ≤ 5, pooled over every coordinate of every sampled site on its scale. The log a_sq gate is the same rule on log a_sq alone. "At cap" is the fraction of the 256 draws whose trajectory hit 63 steps (depth 6) or 1,023 (depth 10). Wall time runs from kernel construction to `block_until_ready` on the draws, compilation included, on the laptop CPU under load (appendix). The per-seed rows are in `tables/tier_b.md`.

Reading:

- The control passes every run (12 of 12).
- **I1** at depth 6 fails the gate at p = 100 in 17 of 18 runs, the exception being C_untied_k1 seed 0 (ESS 48.7), and 92-100 % of iterations hit the cap. At p = 30 it passes 13 of 18. At depth 10 it passes 15 of 18 at each p. At p = 100 the failures are all at P2, where k = 0.01: r_hat 2.29-2.61, ESS 3.0-3.4, and 184 divergences in seed 1.
- **I2** passes all 72 of its runs (6 points × 2 p × 2 depths × 3 seeds). **I3** passes all 36 of its own.
- **The R2 site does not limit I2 at the study's p = 100, depth 6.** There `r2d2_z_lam` is the limiting site in 18 of 18 runs. `r2d2_R2` limits only at p = 30 (P1 in 3 of 3 seeds, ESS 100-185; C_tied seed 0, ESS 159) and in 7 depth-10 runs, always at ESS ≥ 86.8. Tier B therefore gives no reason to try the full copula (R2 through an inverse Beta CDF) in the next study.
- **I3's sampled-site statistics are identical** at P1, P2 and C_tied for a given seed and p. Its prior sites are exactly N(0, I_{p+1}) whatever (k, b), so NUTS runs the same z-chain, and only the map to log a_sq differs. The three I3 rows are one chain per seed.
- **At depth 6, every p = 100 chain of HC, I2 and I3 spends 52-100 % of its iterations at the 63-step cap** (HC 95-100 %). The same chains at depth 10 use 15-27 steps, none at the cap. On I3, whose target is exactly N(0, I_101), the target cannot be the cause. At the end of warm-up the step size is similar at both depths (0.34-0.45 for I2 and I3). But the dense inverse mass matrix's smallest eigenvalue is 0.004-0.005 at depth 6, against 0.09-0.12 at depth 10; for I3 the ideal inverse mass matrix is the identity. The control shows the same, 0.003-0.004 against 0.20-0.24 (Table S4). The chains pass the gate at depth 6 anyway. But this is a property of the fixed budget, shared with the half-Cauchy cells, that the next study will meet with the likelihood.

| impl | depth | seed | adapted step size | inverse mass matrix eigenvalues: min / median / max | mean steps | at cap |
|---|---|---|---|---|---|---|
| I3 | 6 | 0 | 0.399 | 0.00506 / 0.66 / 4.13 | 52.4 | 0.7 |
| I3 | 6 | 1 | 0.43 | 0.00422 / 0.689 / 4.42 | 48.4 | 0.59 |
| I3 | 10 | 0 | 0.377 | 0.0864 / 0.797 / 2.93 | 18.4 | 0 |
| I3 | 10 | 1 | 0.401 | 0.109 / 0.814 / 2.85 | 15.8 | 0 |
| I2 | 6 | 0 | 0.405 | 0.00487 / 0.692 / 4.32 | 52.5 | 0.67 |
| I2 | 6 | 1 | 0.337 | 0.00525 / 0.669 / 5.67 | 51.9 | 0.67 |
| I2 | 10 | 0 | 0.453 | 0.0969 / 0.84 / 3.69 | 15.1 | 0 |
| I2 | 10 | 1 | 0.384 | 0.115 / 0.82 / 4.24 | 17.4 | 0 |
| HC | 6 | 0 | 0.274 | 0.00309 / 1.34 / 10.2 | 61.9 | 0.97 |
| HC | 6 | 1 | 0.289 | 0.00448 / 1.46 / 16.7 | 61.2 | 0.95 |
| HC | 10 | 0 | 0.302 | 0.201 / 2.05 / 7.25 | 25.6 | 0 |
| HC | 10 | 1 | 0.294 | 0.244 / 1.99 / 6.85 | 24.7 | 0 |

Table S4 (supplementary). The adapted state at the end of warm-up, read from `mcmc.last_state` with the harness's sampler lines, at p = 100 (I3 at P1, I2 at C_untied). Its mean steps and cap fractions reproduce Table B's.

### Cost: the map against the prototype

| function (D = 100, laptop CPU, jit, blocked per call) | forward, ms | value and gradient, ms |
|---|---|---|
| prototype `scripts/scout_r2d2/icdf2.py` (Newton inside while-loops), k = 0.0892 | 15.6 | 15.0 |
| `sagp.r2d2.log_gamma_icdf` (the table), k = 0.0892 / 0.44 / 0.005 | 0.0148 / 0.0148 / 0.0154 | 0.0179 / 0.0179 / 0.0181 |

Per value and gradient, the map costs about 1/840 of the prototype. That is about 1.3 % of an n = 100, D = 100 GP log-likelihood gradient on the same machine (1.4 ms, planning §0); the prototype costs 11 times one. These were measured on 2026-09-28 with `.superpowers/sdd/2026-09-27-r2d2-cells/task2_cost.py`, 2,000 calls per entry, each blocked with `jax.block_until_ready`; Task 2 recorded 0.016-0.019 ms against 15.7 ms. In Tier B, a whole prior-only chain at depth 6 and p = 100, compilation included, takes 1.0-2.2 s for I2 against 0.94-1.9 s for the control.

## 4. Tier C: forward sampling (`Predictive`, 4,000 draws)

| impl | point | p | KS D log phi_1 | KS p log phi_1 | KS D logit R2 | KS p logit R2 | forward draws match | factor in model | agrees with scouting |
|---|---|---|---|---|---|---|---|---|---|
| I1 | P1 | 30 | 0.366 | 0 | 0.016 | 0.268 | **no** | yes | agrees in mechanism (the probe's Predictive check; math audit item 5): Predictive never evaluates the factor; not run at this point |
| I2 | P1 | 30 | 0.015 | 0.335 | 0.016 | 0.268 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I3 | P1 | 30 | 0.015 | 0.335 | 0.0122 | 0.596 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I1 | P2 | 30 | 0.479 | 0 | 0.0134 | 0.475 | **no** | yes | agrees in mechanism (the probe's Predictive check; item 5); not run at this point |
| I2 | P2 | 30 | 0.0177 | 0.167 | 0.0134 | 0.475 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I3 | P2 | 30 | 0.0177 | 0.167 | 0.0102 | 0.805 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I1 | P3 | 30 | 0.367 | 0 | 0.0114 | 0.685 | **no** | yes | agrees in mechanism (the probe's Predictive check; item 5); not run at this point |
| I2 | P3 | 30 | 0.0148 | 0.356 | 0.0114 | 0.685 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I1 | C_tied | 30 | 0.444 | 0 | 0.0143 | 0.395 | **no** | yes | agrees in mechanism (the probe's Predictive check; item 5); not run at this point |
| I2 | C_tied | 30 | 0.0181 | 0.15 | 0.0143 | 0.395 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I3 | C_tied | 30 | 0.0181 | 0.15 | 0.0114 | 0.679 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I1 | C_untied | 30 | 0.191 | 1.03e-125 | 0.0161 | 0.258 | **no** | yes | agrees in mechanism (the probe's Predictive check; item 5); not run at this point |
| I2 | C_untied | 30 | 0.0143 | 0.395 | 0.0161 | 0.258 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I1 | C_untied_k1 | 30 | 0.0604 | 7.26e-13 | 0.00985 | 0.837 | **no** | yes | agrees in mechanism (the probe's Predictive check; item 5); not run at this point |
| I2 | C_untied_k1 | 30 | 0.0129 | 0.523 | 0.00985 | 0.837 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I1 | P1 | 100 | 0.451 | 0 | 0.0159 | 0.273 | **no** | yes | agrees in mechanism (the probe's Predictive check; item 5); not run at this point |
| I2 | P1 | 100 | 0.0149 | 0.345 | 0.0159 | 0.273 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I3 | P1 | 100 | 0.0149 | 0.345 | 0.0142 | 0.402 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I1 | P2 | 100 | 0.607 | 0 | 0.0134 | 0.475 | **no** | yes | agrees: probe Predictive check of A2 at P2 (2,000 draws, PRNGKey(7)): log phi_1 KS D 0.594, p = 0 |
| I2 | P2 | 100 | 0.0154 | 0.306 | 0.0134 | 0.475 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I3 | P2 | 100 | 0.0154 | 0.306 | 0.0152 | 0.323 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I1 | P3 | 100 | 0.452 | 0 | 0.0114 | 0.685 | **no** | yes | agrees in mechanism (the probe's Predictive check; item 5); not run at this point |
| I2 | P3 | 100 | 0.0154 | 0.308 | 0.0114 | 0.685 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I1 | C_tied | 100 | 0.548 | 0 | 0.0159 | 0.269 | **no** | yes | agrees in mechanism (the probe's Predictive check; item 5); not run at this point |
| I2 | C_tied | 100 | 0.0161 | 0.259 | 0.0159 | 0.269 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I3 | C_tied | 100 | 0.0161 | 0.259 | 0.0138 | 0.443 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I1 | C_untied | 100 | 0.224 | 4.42e-173 | 0.0161 | 0.258 | **no** | yes | agrees in mechanism (the probe's Predictive check; item 5); not run at this point |
| I2 | C_untied | 100 | 0.0149 | 0.346 | 0.0161 | 0.258 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |
| I1 | C_untied_k1 | 100 | 0.0678 | 4.17e-16 | 0.00985 | 0.837 | **no** | yes | agrees in mechanism (the probe's Predictive check; item 5); not run at this point |
| I2 | C_untied_k1 | 100 | 0.0138 | 0.44 | 0.00985 | 0.837 | yes | no | agrees with item 5, which states that Predictive is valid for the copula (no factor); not measured there |

Table C. Two-sample KS against the ground truth (2e5 draws); the draws match when both p ≥ 0.01. This is a property, not a gate.

- **I2 and I3 draw the prior forward correctly** at every point and p: every KS p is at least 0.149.
- **I1 does not, at any point.** Its log phi_1 KS p is below 1e-12 everywhere, while its logit R2 matches. The reason: `numpyro.infer.Predictive` samples every latent site from its own distribution and never evaluates a factor. I1's `r2d2_raw` is declared N(0, 1) and becomes the log-Gamma law only through the factor `r2d2_raw_log_density`. NUTS reads that factor through the log density; `Predictive` ignores it. I1's forward draws are therefore the proposal, raw ~ N(0, 1) with log lam = log k + raw / k, a far narrower share distribution than Dir(k) at small k. The Beta site on R2 is a plain distribution and comes out right. This matters wherever the prior is simulated forward, as in prior-predictive checks and `init_to_sample`. The model is not altered to make the draws match.
- **Caveats.** I2 and I3 draw identical `r2d2_z_lam` values (the second site in both, under PRNGKey(0)), and every point reuses them. Their log phi_1 KS results are therefore one test per p (their D is identical at the tied points), and one unlucky sample would flag every I2 and I3 row at once. Likewise, I1 and I2 share their `r2d2_R2` draws, so their logit R2 KS values are identical at every point.

## 5. Tier D: the active count at p = 100

| point | source | mean | sd | q05/q25/q50/q75/q95 | chain evaluable | agrees with scouting |
|---|---|---|---|---|---|---|
| half-Cauchy | mixture | 41.54 | 28.67 | 3/17/37/64/93 | - | agrees with the brief's 16-17/37/64; the probe's two 256-draw HC chains gave 15/32/60 and 18/37/63 |
| half-Cauchy | ground_truth | 41.55 | 28.65 | 3/17/37/64/93 | - | agrees (as the mixture row) |
| P1 | ground_truth | 65.21 | 13.44 | 43/56/65/75/88 | - | **disagrees** by 1 at q25: the probe's 20,000-draw ground truth gave 55/65/75, mean 65.25 (section 8) |
| P1 | mixture | 65.21 | 13.46 | 43/56/65/75/88 | - | **disagrees** by 1 at q25, as the ground-truth row |
| P1 | I1 Tier A | 65.12 | 13.61 | 43/55/65/75/88 | yes | agrees: the probe's A2 depth-10 chains (256 draws) gave 55/64/77 and 54/64/74 |
| P1 | I2 Tier A | 65.47 | 13.74 | 43/56/65/76/89 | yes | no scouting number (not run there) |
| P1 | I3 Tier A | 65 | 13.31 | 44/55/65/74/88 | yes | no scouting number (not run there) |
| P2 | ground_truth | 3.854 | 2.268 | 1/2/4/5/8 | - | agrees: the probe's ground truth, mean 3.87, 2/4/5 |
| P2 | mixture | 3.856 | 2.27 | 1/2/4/5/8 | - | agrees (as the ground-truth row) |
| P2 | I1 Tier A | 2.944 | 2.254 | 0/1/2/4/7 | **no** | agrees in kind: the probe's A2 depth-10 chain was stuck too (mean 5.67, 4/5/7), off in the other direction |
| P2 | I2 Tier A | 3.897 | 2.277 | 1/2/4/5/8 | yes | no scouting number (not run there) |
| P2 | I3 Tier A | 3.812 | 2.274 | 1/2/4/5/8 | yes | no scouting number (not run there) |
| P3 | ground_truth | 16.85 | 12.44 | 0/7/15/25/40 | - | no scouting number (not run there) |
| P3 | mixture | 16.81 | 12.42 | 0/7/15/25/40 | - | no scouting number (not run there) |
| P3 | I1 Tier A | 16.8 | 12.46 | 0/7/15/25/40 | yes | no scouting number (not run there) |
| P3 | I2 Tier A | 17.05 | 12.59 | 0/7/15/25/40 | yes | no scouting number (not run there) |
| C_tied | ground_truth | 38.44 | 12.14 | 21/30/37/46/61 | - | no scouting number (not run there) |
| C_tied | mixture | 38.44 | 12.16 | 21/30/37/46/61 | - | no scouting number (not run there) |
| C_tied | I1 Tier A | 39.02 | 12.36 | 22/30/38/46/62 | yes | no scouting number (not run there) |
| C_tied | I2 Tier A | 38.87 | 12.65 | 21/30/37/47/63 | yes | no scouting number (not run there) |
| C_tied | I3 Tier A | 38.22 | 11.93 | 21/30/37/45/60 | yes | no scouting number (not run there) |
| C_untied | ground_truth | 38.87 | 27.9 | 0/14/37/61/87 | - | no scouting number (not run there) |
| C_untied | mixture | 38.95 | 27.96 | 0/14/37/61/87 | - | no scouting number (not run there) |
| C_untied | I1 Tier A | 39.61 | 28.03 | 0/14/38/63/87 | yes | no scouting number (not run there) |
| C_untied | I2 Tier A | 39.46 | 28.62 | 0/13/38/63/87 | yes | no scouting number (not run there) |
| C_untied_k1 | ground_truth | 39.23 | 27.92 | 0/14/37/62/87 | - | no scouting number (not run there) |
| C_untied_k1 | mixture | 39.28 | 27.94 | 0/14/37/62/87 | - | no scouting number (not run there) |
| C_untied_k1 | I1 Tier A | 39.27 | 27.62 | 1/14/38/61/87 | yes | no scouting number (not run there) |
| C_untied_k1 | I2 Tier A | 39.44 | 28.17 | 0/14/37/62/88 | yes | no scouting number (not run there) |
| tied_k0.0892 | ground_truth | 37 | 11.9 | 20/28/36/44/59 | - | agrees on the quartiles with the brief's 28/36/44; **disagrees** that this k matches the half-Cauchy median: its median is 36, not 37 (section 8) |
| tied_k0.0892 | mixture | 37 | 11.92 | 20/28/36/44/59 | - | as the ground-truth row |

Table D. N_0.02 = #{i : a_sq_i > 0.02} from three sources:

- the ground truth, 2e5 NumPy draws;
- the exact mixture: N | T ~ Binomial(100, Q(k, eps T)), with T = xi / B, xi ~ Gamma(b) and B ~ Beta(a, kD - a), or T = xi when tied, over 2^14 scrambled Sobol points;
- each Tier A chain, 4,000 draws.

The half-Cauchy row is the four cells' prior count at ALPHA_AMPLITUDE (its mixture is exact given tausq).

- The mixture and the ground truth agree at every study point to within 0.1 in mean and sd, and exactly in every listed quantile.
- Every evaluable chain's quantiles are within 2 of the ground truth's. The exception is I1 at P2, the chain that is not evaluable (median 2 against 4).
- The rho-scale transport identity holds on all 56 rho rows, every ground truth and every chain: #{7.6218 a_sq_i > RHO_EPS} = #{a_sq_i > ACTIVE_EPS}, draw for draw. So the rho-scale R2-D2 cells share this calibration and its residual exactly (plan D2, Q4). The rows are in `tables/tier_d_counts.md`.

## 6. The calibration table (both forms)

The target is the four half-Cauchy cells' count at D = 100: integer quartiles (17, 37, 64), continuous quartiles (16.63, 36.99, 64.40), mean 41.5. The residual is max(|q25 - q25_HC|, |q75 - q75_HC|) on integer quartiles; the continuous residual, the tie-breaker, is in parentheses. Every row matches the median (continuous 36.99). Under ruling R15, C_untied minimizes the residual over a ≥ 0.5, b ≥ 0.5 and k ≤ 1/2, and C_untied_k1, the sensitivity point, over k ≤ 1. The trend rows, with no bound on k, are reported only.

The tied form (a = alpha; one knob, k, at b = 0.5):

| point | a = alpha | b | k | quartiles | residual (q25, q75) | sd log omega | P(R2 < 0.6) |
|---|---|---|---|---|---|---|---|
| C_tied (median-matched) | 9.379 | 0.5 | 0.09379 | 30 / 37 / 46 | 18 (13, 18); continuous 18.5 | 2.25 | 0.0022 |
| the 2026-09-25 brief's tied k | 8.92 | 0.5 | 0.0892 | 28 / 36 / 44 | 20 (11, 20); median 36, not 37 | 2.25 | 0.0029 |

The untied (reference) form, the whole search as the harness wrote it (`tables/tier_d_calibration.md`):

| kind | a | b | k | alpha | quartiles | continuous quartiles | residual (continuous) | sd log omega | P(R2 < 0.6) | branch | ground-truth quartiles | agrees with scouting |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **target** | nan | nan | nan | nan | 17/37/64 | 16.63/36.99/64.4 | 0 (0) | nan | nan | - | - | agrees with the brief's 16-17/37/64 |
| **C_tied** | 9.379 | 0.5 | 0.09379 | 9.379 | 30/37/46 | 29.59/36.99/45.86 | 18 (18.5) | 2.25 | 0.00224 | rising | 30/37/46 ; roots 0.093786 rising | **disagrees** with the brief's median-matched k = 0.0892 (section 8) |
| profile_k0.5 | 0.5 | - | - | - | - | - | - | - | - | - | no b >= 0.5 with k <= 0.5 matches the median | no scouting number (not run there) |
| profile_k0.5 | 0.75 | - | - | - | - | - | - | - | - | - | no b >= 0.5 with k <= 0.5 matches the median | no scouting number (not run there) |
| profile_k0.5 | 1 | 0.5001 | 0.363 | 36.3 | 12/37/63 | 11.65/36.99/63.36 | 5 (4.98) | 2.56 | 0.368 | rising | 12/37/63 | no scouting number (not run there) |
| profile_k0.5 | 1.25 | 0.6084 | 0.3863 | 38.63 | 14/37/61 | 13.59/36.99/61.35 | 3 (3.05) | 2.18 | 0.358 | rising | 14/37/61 | no scouting number (not run there) |
| profile_k0.5 | 1.5 | 0.7578 | 0.4727 | 47.27 | 14/37/61 | 13.68/36.99/61.44 | 3 (2.96) | 1.85 | 0.368 | rising | 14/37/61 | no scouting number (not run there) |
| profile_k0.5 | 1.75 | 0.8771 | 0.5 | 50 | 15/37/60 | 14.78/36.99/60.28 | 4 (4.12) | 1.66 | 0.363 | rising | 15/37/60 | no scouting number (not run there) |
| profile_k0.5 | 2 | 0.9814 | 0.5 | 50 | 16/37/59 | 16.13/36.99/58.81 | 5 (5.58) | 1.53 | 0.354 | rising | 16/37/59 | no scouting number (not run there) |
| profile_k0.5 | 2.5 | 1.188 | 0.5 | 50 | 18/37/57 | 18.22/36.99/56.56 | 7 (7.84) | 1.33 | 0.336 | rising | 18/37/57 | no scouting number (not run there) |
| profile_k0.5 | 3 | 1.393 | 0.5 | 50 | 20/37/55 | 19.78/36.99/54.88 | 9 (9.51) | 1.19 | 0.32 | rising | 20/37/55 | no scouting number (not run there) |
| profile_k0.5 | 4 | 1.801 | 0.5 | 50 | 22/37/53 | 22/36.99/52.53 | 11 (11.9) | 1.01 | 0.294 | rising | 22/37/52 | no scouting number (not run there) |
| profile_k0.5 | 5 | 2.208 | 0.5 | 50 | 24/37/51 | 23.53/36.99/50.92 | 13 (13.5) | 0.89 | 0.271 | rising | 23/37/51 | no scouting number (not run there) |
| profile_k0.5 | 7 | 3.018 | 0.5 | 50 | 26/37/49 | 25.53/36.99/48.83 | 15 (15.6) | 0.739 | 0.234 | rising | 25/37/49 | no scouting number (not run there) |
| profile_k0.5 | 10 | 4.232 | 0.5 | 50 | 27/37/47 | 27.31/36.99/46.97 | 17 (17.4) | 0.61 | 0.192 | rising | 27/37/47 | no scouting number (not run there) |
| profile_k0.5 | 20 | 8.275 | 0.5 | 50 | 30/37/44 | 29.95/36.99/44.24 | 20 (20.2) | 0.424 | 0.109 | rising | 30/37/44 | no scouting number (not run there) |
| profile_k0.5 | 50 | 20.41 | 0.5 | 50 | 32/37/42 | 32.2/36.99/41.93 | 22 (22.5) | 0.265 | 0.0253 | rising | 32/37/42 | no scouting number (not run there) |
| **C_untied** | 1.578 | 0.8044 | 0.4994 | 49.94 | 14/37/61 | 13.71/36.99/61.46 | 3 (2.93) | 1.78 | 0.371 | rising | 14/37/61 ; roots 0.49942 rising | no scouting number (not run there) |
| profile_k1 | 0.5 | - | - | - | - | - | - | - | - | - | no b >= 0.5 with k <= 1 matches the median | no scouting number (not run there) |
| profile_k1 | 0.75 | - | - | - | - | - | - | - | - | - | no b >= 0.5 with k <= 1 matches the median | no scouting number (not run there) |
| profile_k1 | 1 | 0.5001 | 0.363 | 36.3 | 12/37/63 | 11.65/36.99/63.36 | 5 (4.97) | 2.56 | 0.368 | rising | 12/37/63 | no scouting number (not run there) |
| profile_k1 | 1.25 | 0.6084 | 0.3864 | 38.64 | 14/37/61 | 13.59/36.99/61.35 | 3 (3.05) | 2.18 | 0.358 | rising | 14/37/61 | no scouting number (not run there) |
| profile_k1 | 1.5 | 0.7579 | 0.4728 | 47.28 | 14/37/61 | 13.68/36.99/61.44 | 3 (2.96) | 1.85 | 0.368 | rising | 14/37/61 | no scouting number (not run there) |
| profile_k1 | 1.75 | 0.9073 | 0.5584 | 55.84 | 14/37/62 | 13.77/36.99/61.52 | 3 (2.87) | 1.63 | 0.375 | rising | 14/37/61 | no scouting number (not run there) |
| profile_k1 | 2 | 1.056 | 0.6436 | 64.36 | 14/37/62 | 13.85/36.99/61.61 | 3 (2.79) | 1.47 | 0.379 | rising | 14/37/62 | no scouting number (not run there) |
| profile_k1 | 2.5 | 1.352 | 0.8133 | 81.33 | 14/37/62 | 14/36.99/61.76 | 3 (2.64) | 1.25 | 0.385 | rising | 14/37/62 | no scouting number (not run there) |
| profile_k1 | 3 | 1.645 | 0.9817 | 98.17 | 14/37/62 | 14.13/36.99/61.88 | 3 (2.51) | 1.11 | 0.386 | rising | 14/37/62 | no scouting number (not run there) |
| profile_k1 | 4 | 2.151 | 1 | 100 | 17/37/59 | 16.67/36.99/58.86 | 5 (5.54) | 0.934 | 0.37 | rising | 17/37/59 | no scouting number (not run there) |
| profile_k1 | 5 | 2.651 | 1 | 100 | 19/37/57 | 18.61/36.99/56.62 | 7 (7.78) | 0.824 | 0.354 | rising | 19/37/56 | no scouting number (not run there) |
| profile_k1 | 7 | 3.65 | 1 | 100 | 21/37/54 | 21.23/36.99/53.65 | 10 (10.8) | 0.684 | 0.328 | rising | 21/37/53 | no scouting number (not run there) |
| profile_k1 | 10 | 5.146 | 1 | 100 | 24/37/51 | 23.65/36.99/50.99 | 13 (13.4) | 0.565 | 0.297 | rising | 24/37/51 | no scouting number (not run there) |
| profile_k1 | 20 | 10.13 | 1 | 100 | 27/37/47 | 27.34/36.99/47.01 | 17 (17.4) | 0.394 | 0.224 | rising | 27/37/47 | no scouting number (not run there) |
| profile_k1 | 50 | 25.08 | 1 | 100 | 31/37/44 | 30.64/36.99/43.53 | 20 (20.9) | 0.247 | 0.115 | rising | 31/37/44 | no scouting number (not run there) |
| profile_k1 | 100 | 50.01 | 1 | 100 | 32/37/42 | 32.26/36.99/41.87 | 22 (22.5) | 0.174 | 0.0446 | rising | 32/37/42 | no scouting number (not run there) |
| **C_untied_k1** | 3.049 | 1.674 | 0.9981 | 99.81 | 14/37/62 | 14.14/36.99/61.89 | 3 (2.5) | 1.09 | 0.387 | rising | 14/37/62 ; roots 0.99811 rising | no scouting number (not run there) |
| trend | 1.5 | 0.7578 | 0.4727 | 47.27 | 14/37/61 | 13.68/36.99/61.43 | 3 (2.96) | 1.85 | 0.368 | rising | 14/37/61 | no scouting number (not run there) |
| trend | 3 | 1.645 | 0.9816 | 98.16 | 14/37/62 | 14.13/36.99/61.88 | 3 (2.52) | 1.11 | 0.386 | rising | 14/37/62 | no scouting number (not run there) |
| trend | 5 | 2.794 | 1.643 | 164.3 | 14/37/62 | 14.49/36.99/62.14 | 3 (2.26) | 0.807 | 0.381 | rising | 14/37/62 | no scouting number (not run there) |
| trend | 10 | 5.573 | 3.318 | 331.8 | 15/37/63 | 14.8/36.99/62.56 | 2 (1.84) | 0.549 | 0.35 | rising | 15/37/63 | no scouting number (not run there) |
| trend | 20 | 10.96 | 6.64 | 664 | 15/37/63 | 15.05/36.99/62.81 | 2 (1.59) | 0.383 | 0.288 | falling | 15/37/63 | no scouting number (not run there) |
| trend | 50 | 26.75 | 15.92 | 1592 | 16/37/63 | 15.6/36.99/62.53 | 1 (1.87) | 0.241 | 0.171 | falling | 16/37/63 | no scouting number (not run there) |
| trend | 100 | 52.6 | 32.54 | 3254 | 16/37/63 | 15.53/36.99/62.92 | 1 (1.48) | 0.171 | 0.0774 | falling | 15/37/63 | no scouting number (not run there) |

Table F. "branch" is whether the count median rises or falls with k at the matched point. The ground-truth quartiles are from 2e5 NumPy draws at each row; each C row also lists every root in k at its (a, b).

Reading:

- **The tied form matches the median only.** Its best quartiles are (30, 37, 46), an interquartile range of 16 against the half-Cauchy's 47, at residual 18.
- **The untied form reaches residual 3** along a valley from a = 1.25 to a = 3 (k from 0.39 to 0.98).
  - C_untied, at (1.578, 0.8044, 0.4994), has quartiles (14, 37, 61) and residual 3 on both quartiles (continuous 2.93). It sits where the k ≤ 1/2 bound stops the valley: at a ≥ 1.75 the matched point lies on k = 1/2 and the residual rises.
  - C_untied_k1, at (3.049, 1.674, 0.9981), has quartiles (14, 37, 62) and residual 3 at q25 and 2 at q75 (continuous 2.50). It sits where k ≤ 1 stops the valley. Task 3's re-review found a residual-2 point 0.026 continuous-quartile units outside k ≤ 1, at (3.165, 1.0), within the QMC resolution of 2^14 points, so C_untied_k1's integer residual of 3 is borderline.
- **Without a bound on k the residual keeps falling** toward uniform shares: residual 1 at a = 50 and 100, k 16-33, on the falling root. So there is no minimizer. The limit is a dense prior with a random global scale, outside the Design brief's sparse-prior premise (R15).
- **Both C points put 37-39 % of R2's mass below 0.6** (the tied form, 0.2-0.3 %). That is the region a standardized target implies, R2 ≈ 0.5 (plan §0).
- **The residual literals for Task 6's calibration test:** C_untied q25 3, q75 3; C_untied_k1 q25 3, q75 2.

## 7. The verdict per implementation against the acceptance rule

| impl | P1 (required) | P3 (required where represented) | P2 (reported) | C_tied | C_untied | C_untied_k1 | acceptance | agrees with scouting |
|---|---|---|---|---|---|---|---|---|
| I1 | A pass (null 0.88); B 0/3 | A pass (null 0.91); B 0/3 | A not evaluable (null 0.885); B 0/3 | A fail (null 0.84; var_log_a_sq: fraction outside 0.02 > 0.01); B 0/3 | A pass (null 0.835); B 0/3 | A fail (null 0.86; var_log_a_sq: fraction outside 0.03 > 0.01); B 1/3 | **fails** | agrees: the scouting found the reference exact at depth 10 and failing at depth 6 (r_hat 1.38-1.40, ESS 6.7-7.0) |
| I2 | A pass (null 0.88); B 3/3 | A pass (null 0.91); B 3/3 | A pass (null 0.885); B 3/3 | A pass (null 0.84); B 3/3 | A fail (null 0.835; var_log_a_sq: fraction outside 0.05 > 0.01); B 3/3 | A pass (null 0.86); B 3/3 | **passes** | agrees: item 6 (v) ref form passes at depth 6 at P1 and P2, 0 divergences |
| I3 | A pass (null 0.88); B 3/3 | n/a | A pass (null 0.885); B 3/3 | A pass (null 0.84); B 3/3 | n/a | n/a | **passes** | agrees: item 6 (v) Gamma-Gamma passes at depth 6 at P1 and P2 |

Table E. An implementation passes if (i) Tier A is within tolerance at P1 and, where it represents the point, at P3, and (ii) the depth-6 Tier B gate passes for all three seeds at p = 100 at P1. P2 and the C points are reported, not required; the C points are G0's calibrated-point check. Each Tier A entry carries its point's null pass rate and, for a fail, the failing check's margin (R16). "B k/3" counts seeds passing the depth-6 gate at p = 100.

- **I1 fails** criterion (ii): 0 of 3 at P1, and 0-1 of 3 at every point. Where evaluable it is exact under NUTS, but it cannot be sampled at the budget, and forward sampling gets it wrong.
- **I2 passes.** It also passes every Tier B run, at every point, both p and both depths.
- **I3 passes** at the points it can represent. It cannot represent P3, C_untied or C_untied_k1.

## 8. Blind comparison with the scouting

The verdicts above were computed from the harness's output and written into this note before either scouting report was opened. At 06:42:16 the verdict draft's SHA-256 was `a2d757ae53e2ffa6b92c21c76218b6c3a62f8f4d8ec67adfd148a678995dbc4a`, with every "agrees with scouting" cell reading "pending". Only then were `scripts/scout_r2d2/report-prior-nuts-probe.md` and item 6 of `report-math-audit.md` read, and the columns filled. Regenerating the draft from the same inputs reproduced that hash; the draft is kept in the session scratchpad as `verdict_draft_regenerated.txt`. Outside this section, the final note differs from the draft in these ways only:

- the last cell of 185 table rows (the agreement column);
- blank lines added between tables and their captions;
- one raw line after Table D, "transport identity rows: 56, all hold: True", removed because the text under the table states it;
- four edits to the draft's own prose, each made by proofreading it against the harness's tables and cache:
  - the header now says the numbers are copied "rounded";
  - I1's cap fraction at depth 6, p = 100 reads "92-100 %" where the draft said "every iteration" (P2's chains cap at 92-97 %);
  - the count check "failed on no evaluable chain" where the draft said "on any chain" (I1 at P2, which is not evaluable, fails it);
  - the G0 reading adds the log omega variances (3.36 against 3.15) and the 79 of 100 coordinates above their closed form;
- the appendix's reproduction of the supplementary scripts, and one row for the checks of this section;
- fix round 1 after the review (ruling R17), in the decision sections only:
  - C_tied and the four G0 options;
  - the Tier A pass count, 11 of 15 where the draft said 12;
  - log omega's variance z on its own ESS, and the ten-seed pooled result;
  - the chains' failure rates, now with their intervals.

No verdict, and no number from the harness, changed after the scouting was read.

The scouting's models map onto this study as follows:

- The probe's model A2 and item 6's (i) are I1.
- Item 6's (v) "ref form" is I2, but run with the prototype Newton map instead of the table.
- Item 6's (v) "Gamma-Gamma" is I3.
- The probe's model C is the HC control.
- The probe's model A (with `z_beta`) and model B (Gamma-Gamma under NumPyro's log transform) are not in this study.

Settings 1 and 2 are P1 and P2 at p = 100. The scouting ran nothing at p = 30, at P3 or at the C points, so those rows have no scouting number.

**Every verdict agrees.** I1 is exact on the prior but fails the depth-6 gate at P1 and P2. I2 and I3 pass at depth 6 at P1 and P2. I1's forward draws are wrong. The half-Cauchy control passes.

**Where the scouting ran the same sampler with the same seeds, the numbers agree to every printed digit.** This covers I1 and HC under PRNGKey(0) and PRNGKey(1), at depth 6 and 10: r_hat, ESS, log a_sq ESS, divergences, mean steps and cap fraction all match (Tables B6 and B10). This is an independent check that the harness reproduces `_run_nuts` exactly.

At P1 the same holds for the copula chains. On the prior alone the map does not enter the log density, so the z-chain does not depend on how the map is computed. Item 6's summaries of its P1 runs equal this study's seed-0 chains to every printed digit: I2 has E log a_sq -3.090, Var 27.3, median ESS 231 and 61.2 steps; I3 has -3.074, 28.85, 243, 52.4 steps and step size 0.40. The prototype and the table each agree with the Newton solver to 3e-13 relative on z in [-6, 6] at k = 0.01, 0.2093, 0.5 and 1, and they give identical E and Var on this study's I3 chain at P2 (`scout_check.py`, appendix).

The rows that disagree:

| row | scouting | this study | which to trust, and why |
|---|---|---|---|
| I2 at P1, depth 6, p = 100, seed 0: minimum log a_sq ESS | about 140 (item 6 (v)); the brief summarizes the scouting as 40-67 % of 256 draws, 102-172 | 100.2 (39 %) on the same chain (E, Var and median ESS equal item 6's) | This study's. The scouting's script (`scripts/scout_r2d2/icdf_nuts.py:81-83`) prints coordinate 1's ESS and the median only, so "about 140" is not one of its outputs. Both are far above 16, so the verdict agrees. |
| I2 and I3 at P2, depth 6, p = 100: E and Var of log a_sq | I2 -98.96, 9,980; I3 -100.36, 10,179 (item 6 (v); steps 48.5 and 52.4, median ESS 255 and 254) | I2 -99.21, 9,962; I3 -100.60, 10,161, seed 0 (steps 48.5 and 52.4, median ESS 254 and 254) | Unresolved, and immaterial. The steps and median ESS agree, but E and Var do not, so the scouting's P2 runs were not these chains; the maps agree to 3e-13 on this chain, so the map is not the cause. The reports do not say what differed. Both sets lie within Monte Carlo error of the closed forms (-99.98, 10,003), about 0.6 in E for 256 draws. |
| the P1 ground-truth count, q25 | 55, from 20,000 draws (the brief's 55/65/75 inherits it) | 56, from 2e5 draws; the exact mixture agrees | This study's. The exact P(N ≤ 55) is 0.2499, so q25 sits on a CDF boundary, and 20,000 draws, with standard error 0.003 on that probability, decide it by chance. |
| the median-matched tied point | k = 0.0892, quartiles 28/36/44 (the brief, from the scouting) | the same quartiles, but the median is 36, not the half-Cauchy's 37; the median-matched k is 0.0938 (C_tied, 30/37/46) | This study's. The half-Cauchy median is 36.99 in continuous terms, just above the 36/37 boundary, and k = 0.0892 gives 36 in both the exact mixture and 2e5 draws. |
| the copula chains' wall time | 1.3-1.9 s per 768 iterations with the prototype map (item 6 (v)), "not the bottleneck" | 1.0-2.2 s with the table; the prototype itself costs 15.0 ms per value and gradient | Neither number is the prototype's cost. At 37,000-47,000 gradients per chain, the prototype would need 9-12 minutes, so the scouting's clock most likely stopped before JAX's asynchronously dispatched chain had run (planning §0). The script confirms it: it reads `time.time()` around `mc.run` with no `block_until_ready` (`icdf_nuts.py:77`). The probe's I1 and HC wall times were blocked (`probe/probe_r2d2.py:188`) and are comparable with this study's: 0.9-1.3 s at depth 6, against 0.93-1.9 s. |

Item 6's 4,000-draw runs are at depth 6, and this study's Tier A at depth 10, so their ESS figures are not comparable. Only their verdicts are compared.

## 9. Recommendation, and what the next study must establish

**Carry I2 (form "reference") into the next study.** It passes the acceptance rule and every Tier B run. It draws the prior forward correctly. It represents untied points, which the calibration needs: the tied form's residual is 18 at best. And it is the form Quan chose on 2026-09-28. I3 is the tied fallback and passes wherever it applies. I1 is not a candidate at this budget.

**Calibration.** C_untied is recommended under D2 and R15, subject to the G0 ruling above; the table also gives C_untied_k1 and D1's fallback, C_tied, for the other options. The constants Task 6 would take:

| constant or property | C_untied | C_untied_k1 | C_tied (D1's fallback) |
|---|---|---|---|
| `R2D2_FORM` | `"reference"` | `"reference"` | `"tied"` (D1), or `"reference"` |
| `R2D2_K` | 0.49942145555129697 | 0.9981098100451032 | 0.09378587497618585 |
| `R2D2_A` | 1.577790731256835 | 3.0486546035543407 | None under `"tied"`; 9.378587497618584 (= K D) under `"reference"` |
| `R2D2_B` | 0.8043916352200708 | 1.67377856733023 | 0.5 |
| alpha = K D | 49.94 | 99.81 | 9.379 |
| count quartiles (target 17 / 37 / 64) | 14 / 37 / 61 | 14 / 37 / 62 | 30 / 37 / 46 |
| `Q25_RESIDUAL`, `Q75_RESIDUAL` | 3, 3 | 3, 2 | 13, 18 |
| continuous residual | 2.93 | 2.50 | 18.5 |
| sd log omega; P(R2 < 0.6) | 1.78; 0.37 | 1.09; 0.39 | 2.25; 0.0022 |
| I2, Tier A (seed 0) | fail: variance check, 5 of 100 (a false fail, supplementary) | pass | pass (I3: pass) |
| I2, Tier B depth 6, p = 100 | 3/3; ESS 124-163 | 3/3; ESS 78.8-133.5 | 3/3; ESS 96.3-124 (I3: 3/3; 108-150) |

**What this study cannot establish, and the next one must:**

1. **The posterior geometry with the GP likelihood,** at n ∈ {30, 100}, D = 100, under the same budget. On the prior alone every site of I2 but `r2d2_R2` is exactly N(0, 1), so Tier B tested NUTS on a near-Gaussian target. The copula map's geometry is exercised only with a likelihood: a dominant coordinate's z at 6 or beyond, and the steep lower tail at small k. In the plan this is stage 2's replay on stored trajectories (Tasks 8 and 11, gate G2), after the cells exist.
2. **The dispatch misroutes.** `readouts.py:232` (`is_amplitude`), `readouts.py:266` and `gp.py:708-712` (`param_sites`) test prior strings, so a new cell would get the wrong sites and the wrong estimator (plan §0, Task 5).
3. **The diagnostics site tables and the unit-interval category.** `sagp/diagnostics.py` has no unit-interval category, so `r2d2_R2` needs a logit group (plan Task 6: `_UNIT_SAMPLED_SITES` and the logit union in `diagnose`).
4. **The cell registration:** the four keys, the constants above, and `test_r2d2_calibration_matches_reference_count` (plan Task 6).
5. **Moving the winner into `sagp/`.** This was done ahead of the study (Task 2: `sagp/r2d2.py`, its D4 row in `tests/test_layering.py` and its README module row).
6. **The depth-6 warm-up** (section 3): whether, with the likelihood, a mass matrix left with directions shrunk about 200-fold still yields gate-passing fits, for the R2-D2 cells as for the half-Cauchy cells that already run under it.

## 10. Not done or not verified

- **One Tier A chain per point, as the protocol prescribes.** Its verdict is a single draw from a noisy check. At the protocol tolerance, I2's chains, ten per point, failed it 3, 4 and 5 times at P1, C_untied and C_untied_k1 (Table S3). The 95 % intervals are 7-65 %, 12-74 % and 19-81 %, and the pooled rate is 12 of 30 (23-59 %). That is far above the i.i.d. null's failure rates of 12-16.5 %: the binomial tail is 4.5e-4 at the three per-point null rates. With the variance check on the corrected ESS, 8 of 10 pass at each point. Only I2, and only at three points, was run at further seeds (supplementary).
- **The null pass rate models i.i.d. draws, not chains.** No chain-level null exists beyond Table S3.
- **The variance check's ESS.** The harness uses the ESS of log a_sq for the variance tolerance, as the brief's formula reads. The tolerance on the squared deviations' ESS was computed only in the supplementary analysis; the harness and its verdicts are unchanged.
- **The Tier A count check** has a fixed median tolerance of 1. For chains with ESS well below 4,000 it fails more often than the null says. It failed on no evaluable chain here.
- **Tier C uses one PRNG key per p.** Its draws are shared across points, between I2 and I3 (`r2d2_z_lam`), and between I1 and I2 (`r2d2_R2`). The I3 rows of Tier B at P1, P2 and C_tied are one z-chain per seed.
- **The rho-scale variant** was verified by the draw-for-draw identity only, with no separate runs (plan Q4, by design).
- **C_untied_k1's integer residual of 3 is borderline** (Task 3 re-review); the QMC resolution was not raised past 2^14 for this note.
- **The brief's deterministic sites** `a_sq`, `phi`, `R2` and `log_omega` do not exist; the harness reads the one `log_a_sq` site. The brief's own plan-file deliverable is superseded by ruling R1.
- **Tests.** In this task only the fast suites of `tests/test_r2d2_prior.py`, `tests/test_sagp_r2d2.py` and `tests/test_layering.py` were run: 101 passed. The slow tests were not rerun; Task 3's run at this commit had 4 slow passed.
- **Wall times** were measured on a laptop under a 1-minute load average of 5-17 (Spotlight and OneDrive indexing), with compilation included. They are indicative only, and not comparable with GPU times.
- **The posterior with the likelihood** is out of scope (section 9).

## Appendix: how this was produced

Repository `feat/botorch-saasbo` at `d06be119c`; `/opt/anaconda3/envs/saasbo/bin/python` (Python 3.11.16, JAX 0.10.2, NumPyro 0.21.0, NumPy 2.4.6, SciPy 1.17.1), CPU only, Apple M3. `$SCRATCH` is the session scratchpad; the cache `runs_smoke/r2d2_prior_study/` is git-ignored.

The Bash tool's 10-minute limit on a foreground command meant the study could not run as one first invocation. So the tiers were first computed one at a time into the shared cache, and then the protocol command ran over the full cache and wrote `tables/`:

| # | started | command | elapsed | harness wall times, s |
|---|---|---|---|---|
| 1 | 05:56:43 | `python -m experiments.r2d2_prior_study --tiers D --out $SCRATCH/part_D --cache runs_smoke/r2d2_prior_study/` | 138 s | calibration 97.8, D 35.3 |
| 2 | 05:59:11 | the same with `--tiers B --out $SCRATCH/part_B` | 331 s | B control 19.8, B 298.6 |
| 3 | 06:04:55 | the same with `--tiers A --out $SCRATCH/part_A` | 213 s | A 208.7, including the six null pass rates |
| 4 | 06:08:38 | the same with `--tiers C --out $SCRATCH/part_C` | 40 s | C 20.6 |
| 5 | 06:09:29 | `python -m experiments.r2d2_prior_study --tiers A,B,C,D --out sagp_analysis/2026-09-25-r2d2-prior/ --cache runs_smoke/r2d2_prior_study/` | 55 s | calibration 0.0006, B control 0.15, A 1.46, B 1.47, C 18.8, D 29.5 (from the cache) |

- Every invocation exited 0. The half-Cauchy check ran first within run 2 and passed before any other Tier B row.
- The tables of runs 2-4 are byte-identical to the final `tier_b.csv`, `tier_a.csv` and `tier_c.csv`, and run 1's to `tier_d_calibration.csv`. The final `tier_d_counts.csv` adds the 30 Tier A cross-check rows, which need Tier A in the same invocation.
- `run_config.json` records run 5, with its cached wall times. The per-run `wall_s` in the Tier A and B tables is each chain's first computation, stored in the cache.
- First computation took 722 s in total.

Supplementary analyses, run after the harness and reading its cache and functions only. The scripts are reproduced below; `make_tables.py`, which only formats this note's tables from the CSVs, is not.

| analysis | what it runs | elapsed |
|---|---|---|
| S1 (`supplement.py ess`) | per cached Tier A chain: `numpyro.diagnostics.effective_sample_size` of log a_sq and of its squared deviations; the variance z under each | 5 s |
| S2 (`supplement.py null`, `null1000`) | `exact_log_theta` replicates through `tier_a_statistics`, from `SeedSequence(20260928).spawn(200)` at every point (the harness's own replicates), and from `SeedSequence(1).spawn(1000)` at C_untied | 115 s, 92 s |
| S3 (`supplement.py seeds`) | `run_nuts(prior_model("I2", 100, ...), NUTSConfig(512, 4000, max_tree_depth=10), seed)` for seeds 1-9 at P1, C_untied and C_untied_k1, through `tier_a_statistics`; run twice with identical draws | 106 s, 108 s |
| S4 (`adapt_probe.py`) | the harness's sampler lines, keeping `mcmc.last_state.adapt_state` | 33 s |
| cost | `PYTHONPATH=. python .superpowers/sdd/2026-09-27-r2d2-cells/task2_cost.py` | 128 s |
| tests | `python -m pytest -q -m "not slow" tests/test_r2d2_prior.py tests/test_sagp_r2d2.py tests/test_layering.py`: "101 passed, 4 deselected, 1 warning in 36.35s" | 40 s |
| blind-comparison checks (`scout_check.py`) | item 6's summaries on the cached seed-0 chains; the prototype map against the table and the solver | 13 s |

<details><summary><code>supplement.py</code>: Tables S1-S3</summary>

```python
"""Supplementary Tier A analysis for the Task 4 note (read-only use of the committed harness).

Run from the repository root with the study's Python:

    python supplement.py ess      # per chain: ESS of log a_sq vs ESS of its squared deviations
    python supplement.py null     # the harness's own 200 null replicates, per-check statistics
    python supplement.py null1000 # 1,000 further null replicates at C_untied (root seed 1)
    python supplement.py seeds    # I2 Tier A chains at seeds 1-9 at P1, C_untied, C_untied_k1

Nothing here writes to the harness cache or to the study's tables; every statistic goes through
`tier_a_statistics`, the harness's own function, so a verdict here is computed exactly as there.
"""
import glob
import json
import sys

import numpy as np
from numpyro.diagnostics import effective_sample_size
from scipy.stats import kurtosis

from experiments.r2d2_prior import closed_forms, exact_log_theta, ground_truth, prior_model
from experiments.r2d2_prior_study import (
    NULL_SEED, hyperparameters, run_nuts, tier_a_statistics,
)
from sagp.gp import NUTSConfig

CACHE = "runs_smoke/r2d2_prior_study"
CAL = json.load(open("sagp_analysis/2026-09-25-r2d2-prior/tables/run_config.json"))["calibration"]
NAMED = {name: CAL[name] for name in ("C_tied", "C_untied", "C_untied_k1")}
POINTS = ("P1", "P2", "P3", "C_tied", "C_untied", "C_untied_k1")
DIM = 100


def point(name):
    a, b, alpha = hyperparameters(name, DIM, NAMED)
    return a, b, alpha, alpha / DIM


def truth_at(name):
    a, b, alpha, k = point(name)
    return ground_truth(a, b, k, dim=DIM, n=200_000)


def ess_table():
    for name in POINTS:
        a, b, alpha, k = point(name)
        var = closed_forms(a, b, k, dim=DIM)["var_log_a_sq"]
        for impl in ("I1", "I2", "I3"):
            paths = glob.glob(f"{CACHE}/run_A_{impl}_{name}_p100_s0_d10_*.npz")
            if not paths:
                continue
            with np.load(paths[0]) as run:
                x = run["s__log_a_sq"].astype(float)
            ess_x = effective_sample_size(x[None])
            dev2 = (x - x.mean(axis=0)) ** 2
            ess_sq = effective_sample_size(dev2[None])
            kappa = kurtosis(x, axis=0, fisher=False)
            s2 = x.var(axis=0, ddof=1)
            z_harness = np.abs(s2 - var) / (var * np.sqrt((kappa - 1) / ess_x))
            z_own = np.abs(s2 - var) / (var * np.sqrt((kappa - 1) / ess_sq))
            print(json.dumps({
                "impl": impl, "point": name,
                "ess_x_median": float(np.median(ess_x)), "ess_x_min": float(ess_x.min()),
                "ess_sq_median": float(np.median(ess_sq)), "ess_sq_min": float(ess_sq.min()),
                "ratio_median": float(np.median(ess_x / ess_sq)),
                "outside_harness": int(np.sum(~(z_harness <= 3))),
                "outside_own_ess": int(np.sum(~(z_own <= 3))),
                "z_harness_max": float(z_harness.max()), "z_own_max": float(z_own.max()),
                "mean_s2_over_var": float(np.mean(s2) / var),
            }))


def null_profile(name, replicates, root, n_draws=4_000):
    a, b, alpha, k = point(name)
    truth = truth_at(name)
    quiet = {"diverging": np.zeros(n_draws, dtype=bool)}
    rows = []
    for child in np.random.SeedSequence(root).spawn(replicates):
        log_theta = exact_log_theta(a, b, k, dim=DIM, n=n_draws, seed=child)
        s = tier_a_statistics(log_theta, quiet, a=a, b=b, alpha=alpha, truth=truth)
        rows.append((s["verdict"], s["failed_checks"], s["frac_outside_var"],
                     s["frac_outside_mean"], s["log_omega_z_var"]))
    verdicts = [r[0] for r in rows]
    var_frac = np.array([r[2] for r in rows])
    out = {
        "point": name, "replicates": replicates, "root": root,
        "pass_rate": verdicts.count("pass") / replicates,
        "var_fail_rate": float(np.mean(var_frac > 0.01)),
        "var_frac_hist": {f"{v:.2f}": int(np.sum(np.isclose(var_frac, v)))
                          for v in sorted(set(np.round(var_frac, 2)))},
        "P_var_frac_ge_0.05": float(np.mean(var_frac >= 0.05 - 1e-12)),
        "P_var_frac_ge_0.03": float(np.mean(var_frac >= 0.03 - 1e-12)),
        "P_var_frac_ge_0.02": float(np.mean(var_frac >= 0.02 - 1e-12)),
    }
    print(json.dumps(out), flush=True)


def seed_chains(names, seeds):
    nuts = NUTSConfig(num_warmup=512, num_samples=4_000, max_tree_depth=10)
    for name in names:
        a, b, alpha, k = point(name)
        truth = truth_at(name)
        for seed in seeds:
            run = run_nuts(prior_model("I2", DIM, a=a, b=b, alpha=alpha), nuts, seed)
            s = tier_a_statistics(run.samples["log_a_sq"], run.extra, a=a, b=b, alpha=alpha,
                                  truth=truth)
            x = np.asarray(run.samples["log_a_sq"], dtype=float)
            var = closed_forms(a, b, k, dim=DIM)["var_log_a_sq"]
            ess_sq = effective_sample_size(((x - x.mean(axis=0)) ** 2)[None])
            kappa = kurtosis(x, axis=0, fisher=False)
            z_own = np.abs(x.var(axis=0, ddof=1) - var) / (var * np.sqrt((kappa - 1) / ess_sq))
            print(json.dumps({
                "var_outside_own_ess": int(np.sum(~(z_own <= 3))),
                "ess_sq_median": float(np.median(ess_sq)),
                "impl": "I2", "point": name, "seed": seed, "verdict": s["verdict"],
                "failed_checks": s["failed_checks"], "margins": s["margins"],
                "frac_outside_var": s["frac_outside_var"],
                "frac_outside_mean": s["frac_outside_mean"],
                "ess_min": s["ess_log_a_sq_min"], "ess_median": s["ess_log_a_sq_median"],
                "log_omega_z_var": s["log_omega_z_var"], "divergences": s["divergences"],
                "wall_s": run.wall_s,
            }), flush=True)


if __name__ == "__main__":
    task = sys.argv[1]
    if task == "ess":
        ess_table()
    elif task == "null":
        for name in POINTS:
            null_profile(name, 200, NULL_SEED)
    elif task == "null1000":
        null_profile("C_untied", 1_000, 1)
    elif task == "seeds":
        seed_chains(sys.argv[2].split(","), range(1, 10))
```

</details>

<details><summary><code>adapt_probe.py</code>: Table S4</summary>

```python
"""The adapted step size and dense inverse mass matrix at the end of warm-up, depth 6 vs 10.

The run_nuts lines of the harness, keeping mcmc.last_state, at p = 100: I3 at P1 (its sites are
exactly N(0, I_101) on the prior, so the ideal inverse mass matrix is the identity), I2 at C_untied
and the half-Cauchy control; seeds 0 and 1. Run from the repository root with PYTHONPATH=.
"""
import json
import jax
import numpy as np
from numpyro.infer import MCMC, NUTS
from experiments.r2d2_prior import prior_model

for impl, a, b, alpha in (("I3", 20.93, 0.5, 20.93), ("I2", 1.577790731256835, 0.8043916352200708, 49.9421455551297), ("HC", None, None, None)):
    for depth in (6, 10):
        for seed in (0, 1):
            kernel = NUTS(prior_model(impl, 100, a=a, b=b, alpha=alpha), dense_mass=True, max_tree_depth=depth)
            mcmc = MCMC(kernel, num_warmup=512, num_samples=256, progress_bar=False)
            mcmc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"))
            state = mcmc.last_state.adapt_state
            inv = jax.block_until_ready(state.inverse_mass_matrix)
            inv = np.asarray(next(iter(inv.values())) if isinstance(inv, dict) else inv)
            ev = np.linalg.eigvalsh(inv)
            steps = np.asarray(mcmc.get_extra_fields()["num_steps"])
            print(json.dumps({"impl": impl, "depth": depth, "seed": seed,
                              "step_size": float(state.step_size),
                              "inv_mass_eig_min": float(ev.min()), "inv_mass_eig_max": float(ev.max()),
                              "inv_mass_eig_median": float(np.median(ev)),
                              "mean_steps": float(steps.mean()),
                              "frac_at_cap": float(np.mean(steps >= 2**depth - 1))}), flush=True)
```

</details>

<details><summary><code>scout_check.py</code>: the blind-comparison checks of section 8</summary>

```python
"""Blind-comparison checks, run after the verdicts were written (repository root, PYTHONPATH=.).

1. item 6's summaries (E, Var, coordinate-1 and median ESS of log a_sq) on the harness's cached
   depth-6, seed-0 chains at p = 100;
2. the scouting prototype map against the table and the Newton solver on z in [-6, 6], and both
   maps on the harness's I3 chain at P2.
"""
import glob

import sagp  # noqa: F401  (float64)
import jax.numpy as jnp
import numpy as np
from numpyro.diagnostics import effective_sample_size

from sagp.r2d2 import log_gamma_icdf, log_gamma_icdf_newton

CACHE = "runs_smoke/r2d2_prior_study"
for impl, point in (("I2", "P1"), ("I2", "P2"), ("I3", "P1"), ("I3", "P2"), ("I1", "P1"), ("I1", "P2")):
    with np.load(glob.glob(f"{CACHE}/run_B_{impl}_{point}_p100_s0_d6_*.npz")[0]) as run:
        la = run["s__log_a_sq"].astype(float)
    ess = effective_sample_size(la[None])
    print(impl, point, f"E {la.mean():.3f} Var {la.var():.2f} ESS coord1 "
          f"{float(effective_sample_size(la[None, :, 0])):.0f} median {np.median(ess):.0f} min {ess.min():.1f}")

source = open("scripts/scout_r2d2/icdf2.py").read().split("# accuracy at k = 0.01")[0]
namespace: dict = {}
exec(source, namespace)  # the prototype's definitions only, not its module-level experiments
prototype = namespace["log_gamma_icdf_from_normal"]
z = jnp.linspace(-6, 6, 2401)
for k in (0.01, 0.2093, 0.5, 1.0):
    newton = np.asarray(log_gamma_icdf_newton(z, k))
    scale = np.maximum(1, np.abs(newton))
    print(f"k={k}: max rel |prototype - newton| {np.max(np.abs(np.asarray(prototype(z, k)) - newton) / scale):.2e}, "
          f"|table - newton| {np.max(np.abs(np.asarray(log_gamma_icdf(z, k)) - newton) / scale):.2e}")
with np.load(glob.glob(f"{CACHE}/run_B_I3_P2_p100_s0_d6_*.npz")[0]) as run:
    z_lam, z_xi = jnp.asarray(run["s__r2d2_z_lam"]), jnp.asarray(run["s__r2d2_z_xi"])
for name, f in (("table", log_gamma_icdf), ("prototype", prototype)):
    log_theta = np.asarray(f(z_lam, 0.01)) - np.asarray(f(z_xi, 1.0))[:, None]
    print(f"{name} on the I3 P2 chain: E {log_theta.mean():.3f} Var {log_theta.var():.2f}")
```

</details>
