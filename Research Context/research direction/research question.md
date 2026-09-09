# From the supervisor's note to one thesis question

**Sparse additive GPs for high-dimensional BO — candidate research questions, a recommendation, the weakest link, and scope cuts.**
Prepared 31 Aug 2026 from `supervisor/sparse-additive-gp-note.md`, `direction/sparse-additive-gp-hdbo.md`, and the papers in `references/high-dimensional-bo-papers.md` (SAASBO, OAK, Hvarfner 2024, Papenmeier ICML 2025, Doumont 2026 and Tang & Paulson 2026 were read in full).

---

## 0. Verdict in five sentences

The supervisor's idea is sound but the note's framing has three defects that would each sink a four-month thesis if left in place. First, the headline contrast "sparse inverse squared lengthscales (SAASBO) vs sparse additive amplitudes (this project)" changes two things at once — the kernel family (product → additive) and the parameter carrying the prior — so a head-to-head measures the additive restriction, not the parameterization. Second, "a coordinate is relevant when its amplitude stays large" is only true if the orthogonalized component kernel is normalized to unit variance under a fixed reference measure; with the note's construction as written, the amplitude of a component is confounded with its lengthscale and the active-set rule A_t = {i : P(a_i > ε) ≥ ρ} has no fixed meaning (Section 1, with numbers). Third, the acquisition claim is weak as stated — LogEI with BoTorch's local candidate sampling already removes most of the acquisition-optimization failure at these dimensions (Papenmeier et al. 2025), so the note's surrogates address a largely solved problem — but it has a sharp form the note misses: an additive posterior makes Thompson sampling *exact*, which is precisely the failure mode of TS in high dimensions identified in 2025–26. The recommended thesis (Section 3) keeps the supervisor's contrast but makes it identifiable with a crossed design, puts the identification study first as a gate, and treats real benchmarks as out-of-distribution stress tests rather than as the place where the thesis is won.

**Recommended primary question.** *Holding kernel family (crossed), inference, acquisition, initialization and budget fixed, does moving the sparsity prior from inverse squared lengthscales to normalized additive amplitudes change which coordinates a fully Bayesian GP treats as relevant and how sample-efficiently BO optimizes sparse objectives — and where does any such difference end as first-order additivity fails?*

---

## 1. What the amplitude-vs-lengthscale contrast actually is

This section is the analysis everything else rests on; it is short because the mechanism is simple.

**Component variance is a product of amplitude and a lengthscale-dependent factor.** For a 1-D Matérn-5/2 kernel centered under the uniform measure on [0,1] (the note's eq. 2–3, which is the projection variant of OAK's construction; the conditioning variant OAK itself uses gives the same numbers to within 3 %), the average marginal variance of the centered component is v(ℓ) = 1 − E_{x,x'}[k(x,x'; ℓ)]. Numerically v(0.1) = 0.78, v(0.5) = 0.28, v(1) = 0.107, v(3) = 0.015, v(10) = 0.0014; for large ℓ, v(ℓ) → 5/(36ℓ²) and the centered kernel collapses to the rank-one linear kernel 5/(3ℓ²) · (x − ½)(x' − ½). So a component with hyperparameters (a_i, ℓ_i) has variance a_i² v(ℓ_i), and there are two ways for a component to be "off": a_i → 0, or ℓ_i → ∞.

**Consequence for the readout.** If relevance is read from a_i, an irrelevant coordinate can escape the sparsity prior through ℓ_i while a_i stays wherever its prior puts it, and a weak but smooth component needs a *large* a_i to represent even a small variance. In a one-component toy (n = 80, noise sd 0.1, half-Cauchy(0.1) on a, lognormal prior on ℓ with median 1 and sd 2 in log space, posterior computed exactly on a grid), the unnormalized centered kernel gives a weak *smooth* component (variance 0.02 on a unit-variance output scale, ℓ = 1.5) a 45 % posterior probability of exceeding a = 0.3, and gives two components of *equal* variance 0.25 amplitudes of 1.66 (ℓ = 1.5) and 0.55 (ℓ = 0.08). Dividing the centered kernel by v(ℓ) — so that a_i² *is* the component's variance under the reference measure — reduces the false-positive probability to 7 % and the two amplitudes to 0.70 and 0.52. The effect is specific to smoothness, as the mechanism predicts: for a weak *rough* component the two parameterizations give 9 % and 5 %, and for pure noise 7 % and 1 %. (Script: `direction/kernel_normalization_check.py`, which prints both tables.) Under normalization, ℓ_i → ∞ no longer switches a component off; it makes it linear. That is the parameterization in which "lengthscales are explicitly not relevance indicators" is actually true.

OAK itself (Lu et al. 2022) does not read relevance from the variance hyperparameters at all — it ranks components by the Sobol index of the *posterior mean*, Var_x[m_u(x)]/Var_x[m(x)], having argued that variance hyperparameters of the kind Duvenaud et al. used are unidentifiable. The thesis should do both: normalized amplitudes as the prior's target, posterior first-order Sobol indices (computable per NUTS sample) as the primary relevance statistic, and a preregistered check that the two agree.

**Consequence for the comparison.** Once the kernel is normalized, "sparsity on inverse squared lengthscales" means different things in the two families. In SAASBO's product kernel, ρ_i → 0 removes dimension i from every interaction; in the normalized additive kernel, ρ_i → 0 leaves a linear main effect with amplitude a_i — a smoothness preference, not sparsity. The additive analogue of SAASBO's switch-off is a_i → 0, and the product analogue of amplitude sparsity is Π_i(1 + a_i² k̃_i), the full-order OAK kernel with a shrinkage prior on a. So the real difference between the two priors is not *which* coordinates they can switch off but *what they tie together*: lengthscale shrinkage couples "small effect" to "smooth effect" (and, given a bounded global amplitude, "large effect" to "rough effect"); amplitude shrinkage leaves shape free. This is the mechanism hypothesis the thesis can test, with predictions in both directions:

- Where relevant coordinates carry strong effects of heterogeneous roughness, or where many irrelevant coordinates carry weak rough noise, amplitude sparsity should identify the right set and lengthscale sparsity should over-weight rough components.
- Where many coordinates carry weak, smooth, monotone effects — which Papenmeier et al. (ICML 2025, App. D) show is the actual structure of Lasso-DNA and MOPTA08 (110/180 and 93/124 "secondary" dimensions whose optimum sits on the boundary and which the GP models with long lengthscales) — lengthscale-based treatment gets those trends almost for free while amplitude sparsity zeroes them. Amplitude sparsity is *predicted to lose* on those benchmarks.

**The honest near-equivalence statement.** On the standard axis-aligned embedded benchmarks (Branin/Hartmann/Levy in D distractors, the LassoBench synthetics) the active coordinates are both strong and rough relative to distractors that are exactly zero, so both parameterizations will select the same set and any regret difference will come from structure, inference or acquisition details — not from the prior. A thesis that hopes to see the parameterization matter *there* will see nothing. The design below therefore predicts equivalence on aligned ground truths as a positive control and looks for divergence only where the mechanism says it should occur.

---

## 2. Candidate questions

Six candidates, materially different in kind. Cost estimates use the model in Appendix A (a budget of roughly 5–15 core-hours per 200-evaluation NUTS run at D = 100 depending on chain count; cheap MAP baselines are negligible).

### Q1 — Identification: which relevance criterion does each prior implement?

**Question.** Under matched fully Bayesian inference on fixed Sobol designs, does the posterior over normalized additive amplitudes rank coordinates by first-order variance share while the SAAS posterior over inverse squared lengthscales ranks them by roughness-weighted strength, so that the two rankings coincide on aligned ground truths and diverge, in the predicted direction, when strength and roughness are decoupled?

**Mechanism and confounds.** Isolates the parameter that carries the prior. No BO loop, so no adaptivity or acquisition confound; same designs, n, D, base kernel and NUTS settings across models; kernel family crossed (additive/product × amplitude/lengthscale) so the criterion is attributed to the parameterization rather than the structure.

**Minimal experiment.** D = 100. Three ground-truth families: *aligned* (5 active coordinates, equal variance share, similar lengthscale, distractors zero); *decoupled* (a 2 × 2 of strength × roughness: strong-smooth, strong-rough, weak-smooth, weak-rough, three coordinates each, remaining coordinates zero); *dense-weak* (all 100 coordinates carry small smooth monotone effects). n ∈ {30, 60, 100, 200}, 10 seeds. Four models. Measured: precision/recall of the active set at the note's cutoff rule; Spearman correlation of the posterior ranking with true variance share and with true roughness; AUROC active-vs-inactive; calibration of P(a_i > ε | D). Manipulation checks: posterior a_i² tracks the realized component variance across ℓ (no distortion), posterior Sobol indices agree with amplitude ranking, R-hat/ESS/divergences within bounds.

**Result that answers NO.** On decoupled families the rankings from the two parameterizations are indistinguishable (difference in rank correlations inside the seed-to-seed spread), or the amplitude model fails to recover even the aligned set at n ≤ 200 (recall below 0.8 where SAAS reaches about 1).

**Null-result thesis.** "For fully Bayesian GPs, amplitude and lengthscale shrinkage are operationally the same relevance detector; what differs is the function class they are attached to." This is publishable: it is the first matched head-to-head, and it comes with the kernel-normalization result, which stands regardless.

**Feasibility.** About 500 NUTS fits at n ≤ 200 — roughly 100–200 core-hours even with long chains. Three to four weeks of student time including the ground-truth generator. The cheapest question on the list.

**Closest prior work.** SAASBO §5.2 and App. D.2 (relevant dimensions "found" via posterior-median ρ on Branin-100 and SVM); GTBO (Hellsten et al. 2025: explicit active set by group testing, 6/1180 false positives at D = 300); OAK (Sobol ranking of the posterior mean, no sparsity prior, no BO); Hvarfner et al. 2024 Fig. 8 (DSP does not consistently identify active dimensions on SVM, yet optimizes well). Left open: nobody has asked which *criterion* a sparse GP posterior implements, nor compared parameterizations under matched inference, nor tested amplitude posteriors of an orthogonal additive GP under a sparsity prior at all.

### Q2 — Sample efficiency: does the parameterization have a main effect once structure is crossed?

**Question.** With kernel family crossed as a factor and inference, acquisition, initialization and budget identical, does the sparsity parameterization have a main effect on simple regret at fixed budget on sparse first-order-additive objectives, and is it larger or smaller than the structure main effect?

**Mechanism and confounds.** The 2 × 3 design below separates parameterization from structure and separates *sparsity* from *complexity control*, the confound Hvarfner et al. raise: the non-sparse controls carry the same total prior complexity as vanilla BO (√D-scaled lengthscales in the product family; amplitudes scaled 1/√D so the total prior signal variance is 1 in the additive family) but no shrinkage.

| | sparsity on amplitude | sparsity on inverse lengthscale | no sparsity (complexity-matched) |
|---|---|---|---|
| **additive** Σ a_i² k̃_i | SAAS-form prior on a_i² (the proposal) | unnormalized k̃ with common amplitude, SAAS prior on ρ | a_i ~ HalfNormal(1/√D), DSP-style ℓ prior |
| **product** | Π(1 + a_i² k̃_i), SAAS-form prior on a_i² | SAASBO itself (ρ_i ~ HC(τ), τ ~ HC(0.1)) | DSP prior on ℓ, σ_f² = 1, fit by NUTS |

The lengthscale-sparse additive cell necessarily uses the *unnormalized* centered kernel, because that is the only additive parameterization in which lengthscale shrinkage removes a component (its variance scales as ρ for long lengthscales); this is exactly the "small effect ⇔ smooth effect" coupling Section 1 identifies, now as an experimental cell rather than an accident. All cells: Matérn-5/2 base, same NUTS budget and warm-start schedule, LogEI averaged over the same number of posterior samples, the same BoTorch optimizer (Sobol raw samples + sample-around-best + L-BFGS-B, same restarts), the same 20-point Sobol initialization per seed, budget 200. References outside the factorial: Sobol, DSP by MAP (the actual vanilla baseline), and an oracle GP on the true active coordinates (the upper bound on what any variable selection can deliver; see Q5). The same global–local prior form — SAASBO's own, relevance parameter ~ HC(τ), τ ~ HC(α), with α rescaled to the parameter's natural scale — is used in both sparse columns, so the lengthscale column *is* SAASBO and the amplitude column differs from it only in the parameter the prior sits on. (The note's horseshoe variant is an appendix robustness check; see Section 5.)

**Minimal experiment.** D = 100, four objectives (aligned |S| = 3, aligned |S| = 10, decoupled, embedded Hartmann-6 for comparability with Hvarfner/Doumont), 10 seeds. Measured: log simple regret at 50/100/200 evaluations and area under the regret curve; analysed as a two-factor design over seeds with effect sizes and confidence intervals against a preregistered smallest effect of interest (say 0.3 in log10 regret at budget 200).

**Result that answers NO.** The parameterization main effect and the parameterization × structure interaction are inside the seed noise and exclude the smallest effect of interest, while the structure effect is large.

**Null-result thesis.** "The additive restriction, not the sparsity prior, buys sample efficiency on sparse-additive objectives; where the prior sits is immaterial." An equivalence result of this kind, with matched everything, is exactly what an identification-style thesis is for.

**Feasibility.** Six NUTS cells × 4 objectives × 10 seeds = 240 runs ≈ 1,500–3,000 core-hours at D = 100; a D = 300 spot check (2 cells, 1 objective, 5 seeds) adds ≈ 400–750. Two to four weeks of calendar time on a modest cluster share. This is the most expensive question but it is affordable.

**Closest prior work.** SAASBO Fig. 9 (SAAS vs dense priors, product family only, NUTS both); Hvarfner et al. Fig. 5 (SAASBO beats DSP on embedded Levy/Hartmann at D = 100/300 — but MAP vs NUTS, so prior and inference are confounded); Papenmeier ICML 2025 (MSR/DSP/SAASBO on real tasks only); Kandasamy 2015 and Ziomek & Bou-Ammar 2023 (additive without sparsity, MLE). Left open: no one has crossed structure with the sparsity parameterization under matched inference, and every published SAASBO-vs-vanilla comparison confounds the prior with the inference scheme.

### Q3 — Acquisition: does additivity make Thompson sampling exact, and does that matter?

**Question.** Given the same sparse additive model, does exact Thompson sampling — one joint pathwise sample of all component functions, then D one-dimensional maximizations — achieve lower regret than candidate-based TS (Sobol or RAASP candidates) and match LogEI on sparse-additive objectives at D = 100?

**Why this and not the note's surrogates.** The posterior mean decomposes but the posterior variance does not, so the note's sparse Add-GP-UCB and linearized LogEI are approximations of unquantified quality. Thompson sampling is different: a jointly sampled path f̃ = Σ_i f̃_i is a *fixed* separable function, so its argmax is exactly the vector of one-dimensional argmaxes. With pathwise sampling (random Fourier features plus Matheron's update, per component) the components are analytic and each 1-D maximization is trivial. This removes precisely the failure mode that makes TS worse than random search in high dimensions — candidate coverage collapsing with D (Papenmeier et al. UAI 2025; Fan & Pleiss AISTATS 2026, who show 10⁶ Sobol candidates are insufficient and propose gradient-aligned cones as a fix).

**Mechanism and confounds.** Acquisition *optimization*, holding the model fixed: same model, prior, NUTS settings, seeds and initial design in every treatment, inactive coordinates set to the incumbent in all treatments, equal budgets. (In the loop the posteriors necessarily diverge once the acquisitions differ; only the offline check below shares the identical NUTS draws.)

**Minimal experiment.** Model: the amplitude-sparse additive cell. Treatments: LogEI (BoTorch default optimizer), candidate TS (Sobol + RAASP candidates), exact additive TS. Two objectives, 10 seeds, D = 100. Measured: regret; plus, offline on stored posteriors with identical draws, the sample-path value attained by each TS optimizer (exact should dominate by construction — this is a manipulation check).

**Result that answers NO.** Exact additive TS is no better than candidate TS in regret, or both trail LogEI by more than the seed spread.

**Null-result thesis.** "Additive structure helps modelling, not acquisition: with LogEI and gradient-based multistart, acquisition optimization is not the bottleneck at D ≤ 300." That is a clean negative on the note's second claim and worth stating.

**Feasibility.** Reuses the Q2 model; 40 extra runs ≈ 400–600 core-hours. TS itself is cheap. Two weeks.

**Closest prior work.** Kandasamy et al. 2015 (additive UCB decomposition, in the DiRect era); Ziomek & Bou-Ammar 2023 (random decompositions suffice, so the acquisition-side benefit of additivity is generic); Papenmeier et al. UAI 2025 and Fan & Pleiss 2026 (the coverage failure of TS); Doumont et al. 2026 App. D.4 (TS hurts both linear and vanilla BO). Left open: the exactness of TS under an additive posterior has not been exploited or tested against LogEI.

### Q4 — Failure of the assumption: where is the phase boundary?

**Question.** As the fraction γ of objective variance carried by pairwise interactions among the active coordinates rises from 0, at what γ* does the sparse additive model's regret advantage over the matched product-kernel model and over DSP vanish, and is γ* the same for both sparsity parameterizations?

**Mechanism and confounds.** Misspecification of first-order additivity, with sparsity held fixed: same active set, same total variance, only the interaction share varies. Two further misspecification axes — rotation of the active subspace by θ ∈ {0°, 15°, 45°} (this thesis's protocol; SAASBO App. D.4 instead used random Gaussian projections of Hartmann-6 with projection dimension 6, 18 and 30) and dense-weak effects — are run as separate one-factor sweeps, not crossed.

**Minimal experiment.** D = 100, |S| = 5, γ ∈ {0, 0.1, 0.25, 0.5, 0.75}, methods: the amplitude-sparse additive cell, SAASBO-with-LogEI, DSP by MAP, Sobol; 10 seeds; budget 200. γ* estimated by interpolation with a bootstrap CI. For the dense corner use Tang & Paulson's Schwefel, Rastrigin and Michalewicz at D = 50: these are *first-order additive but dense*, so the additive kernel is correctly specified and the sparsity prior is wrong — the complexity-matched additive control should beat the amplitude-sparse additive cell (those two NUTS cells only) — and published AdaScale-TuRBO, DSP and Linear-BO curves exist at matched dimensions (budget capped at 300 evaluations for NUTS; refit every 10 iterations as they do).

**Result that answers NO.** γ* ≈ 0 — the additive model already loses to the product model at γ = 0.1 — or the additive model never wins even at γ = 0 (no headroom, see Q5).

**Null-result thesis.** "First-order additive sparsity is brittle; its win region is confined to nearly exactly additive objectives, and here is the map." A measured phase boundary is a contribution even when the region is small; it is also the honest answer to the benchmark-bias objection.

**Feasibility.** 100 NUTS runs for the γ sweep ≈ 1,000–1,500 core-hours; dense corner ≈ 250–400 (refits every 10 iterations); rotation sweep ≈ 300–450 at 5 seeds. Three to four weeks.

**Closest prior work.** SAASBO App. D.4 (rotated Hartmann; SAAS robust at projection dimension 6–30); Doumont et al. 2026 App. D.5 (sparsity and nonlinearity sweeps for linear vs vanilla); Papenmeier ICML 2025 App. C.3/D (function structure decides the winner; secondary dimensions on the boundary); Hvarfner §6.1. Left open: no controlled interaction-share sweep exists for sparse additive GPs and no one has estimated the boundary.

### Q5 — Headroom: is there anything left for sparsity to win after the √D prior?

**Question.** At D ∈ {50, 100, 300} and budgets 100–300, how large is the regret gap between an oracle GP restricted to the true active coordinates and DSP-vanilla BO on the full space, and what fraction of that gap does any sparse model close?

**Mechanism and confounds.** The value of knowing the active set, with acquisition and budget fixed; the oracle bounds every variable-selection method that could ever be built.

**Minimal experiment.** Oracle-S and DSP by MAP (cheap), SAASBO and the amplitude-sparse additive cell by NUTS (reused from Q2 at D = 100; about 30 extra NUTS runs at D = 50 and for embedded Levy-4), on aligned sparse-additive and embedded Hartmann-6/Levy-4 objectives, 10 seeds.

**Result that answers NO.** The oracle–DSP gap is within two standard errors at D ≤ 100 for the budgets the thesis can afford: there is nothing to win below D = 300, which NUTS cannot reach at scale.

**Null-result thesis.** "Under a √D-scaled prior, variable selection has negligible value at feasible dimensions; sparse priors are an interpretability device, not a sample-efficiency device." Sobering but clear, and it redirects the thesis to Q1.

**Feasibility.** Cheap except for the NUTS cells, most of which Q2 already pays for; ≈ 300–450 core-hours extra.

**Closest prior work.** Hvarfner et al. Fig. 5 shows SAASBO beating DSP on most of the embedded Levy/Hartmann tasks at D ≤ 300 — evidence that headroom exists on axis-aligned sparse tasks — while on Lasso-DNA, SVM, Ant and Humanoid DSP beats SAASBO and MOPTA08 is the one real task where SAASBO keeps an edge (their Fig. 7); Doumont Fig. 11 shows the linear–vanilla gap closing as sparsity grows. Left open: the oracle bound and the fraction of it closed have never been reported.

### Q6 — Which sparsity prior: horseshoe, R2-D2 or ℓ1-ball?

**Question.** Holding model and inference fixed, do the three amplitude priors differ in active-set recovery and regret?

**Mechanism and confounds.** The tail and near-zero behaviour of the shrinkage prior, with model, inference and acquisition fixed. But it is a bundle comparison of three families each with its own hyperparameters, whose behaviour near zero is deliberately similar; the expected differences are second-order and hard to interpret, and the ℓ1-ball's projection is non-smooth, which is a NUTS liability rather than a scientific variable. R2-D2 is the one prior with a genuine conceptual fit here: with normalized components and standardized outputs, its Beta prior on R² is a prior on the *first-order additive R²* and its Dirichlet allocation is a prior on the first-order Sobol shares — which also makes its posterior R² a built-in misspecification detector for Q4.

**Minimal experiment.** The SQ1 offline recovery study repeated with R2-D2 (and, if insisted upon, the ℓ1-ball) in place of the horseshoe, same designs and seeds; recovery metrics and NUTS diagnostics compared. Attaching it to an in-loop experiment is not affordable.

**Result that answers NO / null thesis.** No differences beyond seed noise (likely); "prior family is a second-order design choice."

**Feasibility.** Multiplies the NUTS cost of whatever it is attached to by three; not affordable as a crossed factor.

**Closest prior work.** SAASBO App. A.3 explicitly leaves the prior family open; Carvalho 2009; Zhang et al. 2022; Xu & Duan 2023. Recommendation: keep R2-D2 as an appendix robustness check on Q1 only; drop the ℓ1-ball.

---

## 3. Recommendation

**Primary question.** *Holding kernel family (crossed), inference, acquisition, initialization and budget fixed, does moving the sparsity prior from inverse squared lengthscales to normalized additive amplitudes change which coordinates a fully Bayesian GP treats as relevant and how sample-efficiently BO optimizes sparse objectives — and where does any such difference end as first-order additivity fails?*

**Sub-questions forming the arc.**

*SQ1 (Q1, offline, gate).* Which relevance criterion does each prior implement? Preregistered predictions: P1a, on aligned ground truths all four cells select the same active set (Jaccard > 0.9 at n = 100) — the positive control for equivalence; P1b, on decoupled ground truths the amplitude ranking tracks variance share (Spearman > 0.8) and the lengthscale ranking tracks roughness-weighted strength better than variance share. SQ1 also carries the manipulation checks on which everything downstream depends (normalization, Sobol agreement, NUTS health).

*SQ2 (Q2 with Q5 folded in, in-loop, confirmatory).* Does the parameterization have a main effect once structure is crossed, and how much of the oracle–DSP headroom does each cell close? Preregistered prediction P2: the structure main effect exceeds the parameterization main effect on aligned objectives; the parameterization effect appears, if at all, on the decoupled objective. The headroom gate (oracle-S vs DSP gap larger than two standard errors at D = 100, budget 200) is checked in the pilot before confirmatory runs and decides whether D = 100 or D = 300 is the confirmatory dimension.

*SQ3 (Q4, in-loop, boundary).* Where does the advantage end? Preregistered prediction P3: γ* lies in (0.1, 0.5) at budget 200, is the same for both parameterizations, and the sparse cells lose to the complexity-matched additive control on the dense first-order functions. Lasso-DNA and MOPTA08 enter here as out-of-distribution real tasks with a *predicted* outcome (amplitude sparsity loses to DSP because of their many weak smooth boundary-seeking coordinates); confirming that prediction is a result, not a failure.

*Optional SQ4 (Q3).* Exact additive Thompson sampling versus LogEI — only if SQ1–SQ3 finish by mid-January.

**Against the five criteria.**

1. *Falsifiable.* Each sub-question has a numeric prediction and a stated NO; P1a is a prediction of *no difference* and P1b/P2/P3 predict directions and magnitudes.
2. *Yields a thesis if negative.* If P1b fails, the thesis is "amplitude and lengthscale shrinkage are the same detector; here is the matched evidence and the normalization result." If P2 shows no parameterization effect, the thesis is "structure, not prior placement, is what matters." If P3 gives γ* ≈ 0, the thesis is a phase diagram showing that first-order additive sparsity has no practical win region. All three are honest, complete theses.
3. *Isolates a mechanism.* The crossed design attributes effects to parameterization versus structure; complexity-matched controls separate sparsity from complexity control; the same prior form sits on different parameters; inference, acquisition, initialization and budget are identical by construction; the identification study has no BO loop.
4. *Feasible.* Appendix A budgets roughly 4,500–7,500 core-hours across SQ1–SQ3 plus the two real tasks — at most about 300 core-days, i.e. a few weeks of calendar time on a modest share of a CPU cluster, with D = 300 and the real tasks as the first things to drop.
5. *Not already answered.* The nearest results — Hvarfner Fig. 5, SAASBO Fig. 9, Papenmeier's real-task comparisons — each confound prior with inference or never cross structure with parameterization; no published work has put a sparsity prior on the amplitudes of an orthogonal additive GP, measured which relevance criterion a sparse GP posterior implements, or estimated the interaction-share boundary.

**Timeline (Sep 2026 – Feb 2027).** September: normalized orthogonal Matérn-5/2 kernel by quadrature, NumPyro models for all six cells, BoTorch glue, unit tests, Gate 1 on toys; preregister SQ1. October: SQ1 runs and analysis; pilot BO runs to calibrate cost; preregister SQ2/SQ3 with the smallest effect of interest; supervisor sign-off. November–December: SQ2 at D = 100, headroom gate, D = 300 spot check. December–January: SQ3 sweeps, dense corner, two real tasks; SQ4 if time. February: writing. The plan is deliberately front-loaded with the cheap experiment so that a slip in the expensive ones still leaves a thesis.

**What this gives up.** Beating SAASBO or DSP on real benchmarks as the headline claim; the horseshoe-vs-R2-D2-vs-ℓ1-ball comparison; the Add-GP-UCB and linearized-LogEI surrogates; a method-by-benchmark grid at D ≥ 300; MuJoCo, SVM and trajectory planning; TuRBO-classic, BAxUS/Bounce, MCTS-VS and GTBO as run baselines. In exchange the thesis makes claims it can actually defend.

---

## 4. The weakest link

**The assumption most likely to break the thesis is the note's definition of relevance: "a coordinate is considered relevant when its amplitude remains large under the posterior."** As Section 1 shows, this is false for the kernel the note describes. With centering by empirical measure or quadrature but no normalization, the amplitude of a component depends on its lengthscale (v(ℓ) spans three orders of magnitude over the lengthscales a weakly informative prior allows), irrelevant coordinates can switch off through ℓ rather than a, weak smooth effects inflate a, and the active-set rule A_t = {i : P(a_i > ε | D_t) ≥ ρ} has a threshold with no fixed meaning. In the toy, that rule gives a weak smooth component with 2 % of the output variance a 45 % chance of being "active". The recovery and active-set-acquisition questions read relevance off a_i directly, and the sample-efficiency question depends on the prior shrinking what it is meant to shrink, so this would damage all three at once, and it would fail *quietly*: the recovery plots would look noisy rather than wrong, and weeks would go into tuning ε and ρ.

Two further details in the note make this worse. Empirical centering under the BO loop's adaptively sampled data makes the reference measure drift toward the incumbent region, so the "constant" being projected out changes from iteration to iteration and amplitudes become non-stationary across the loop. And centering under a Gaussian measure after a normalizing flow, as OAK does for data features, is unnecessary here: the domain is [0,1]^D by construction, so the uniform measure with a fixed Gauss–Legendre grid (32–64 nodes) is exact, cheap and stable.

**How the recommended question survives it.** Normalization by v(ℓ) is a one-line change once known, and it is built into every cell of the design (including the product-family amplitude cell). SQ1 runs first, offline, and contains the manipulation check that would expose the problem: posterior a_i² must track the realized component variance across lengthscales, and the amplitude ranking must agree with the posterior Sobol-index ranking, which is the readout OAK actually trusts and which is well defined in every parameterization. If the amplitude readout still misbehaves after normalization, the thesis switches its relevance statistic to the Sobol index without changing a single experiment, and the sparsity prior remains what it is — a prior on the component variances.

The runner-up risk is the framing one: that the method must *win* somewhere real. Hvarfner Fig. 5 shows headroom over DSP on axis-aligned sparse synthetics at D ≤ 300, but on Lasso-DNA, SVM and the MuJoCo tasks vanilla BO beats SAASBO outright, and Papenmeier's analysis of Lasso-DNA and MOPTA08 suggests why amplitude sparsity in particular should do worse there. The recommended design handles this by making the truth of the additive-sparse assumption a *controlled factor* and by preregistering degradation on the real tasks as the expected outcome.

---

## 5. Scope cuts

**Priors.** Use one global–local prior form in every sparse cell, and let it be SAASBO's (relevance parameter ~ HC(τ), τ ~ HC(α), with α rescaled to the parameter's natural scale): in the lengthscale cells it sits on ρ_i, so that cell *is* SAASBO and connects to the literature; in the amplitude cells it sits on a_i², so the only difference between the sparse columns is the parameter the prior sits on. The note's horseshoe variant (a_i ~ HalfNormal(τλ_i), λ_i ~ HC(1)) and R2-D2 are appendix robustness checks on SQ1 only — R2-D2 because its R² and Dirichlet shares have exact meaning in the normalized additive model. Drop the ℓ1-ball (exact zeros buy nothing in BO, which thresholds anyway, and its projection is a NUTS hazard). Spike-and-slab stays deferred, as the note says.

**Acquisition.** LogEI everywhere, with the identical BoTorch optimizer across cells. Drop sparse Add-GP-UCB (needs a β schedule, is a loose upper bound because √Var(Σf_i) ≤ Σ√Var(f_i), and adds a second confounded mechanism) and the linearized LogEI surrogate (an approximation of unquantified error, replacing an exact acquisition that is already tractable because the mean and its gradient decompose). Exact additive Thompson sampling is the only acquisition idea worth the budget, as an optional fourth sub-question.

**Inference.** NUTS in every cell — global–local shrinkage priors force it (their MAP is degenerate at zero, and SAASBO's Fig. 2 shows NUTS beating MAP even when MAP finds the right dimensions). Use SAASBO's reduced budget (128 warm-up + 128 samples, thinned to 8, tree depth ≤ 6), warm-started; 4 chains in pilots for diagnostics, 2 in confirmatory runs; refit every iteration until n = 100, then every 2–4 iterations, on the same schedule in every cell. Log R-hat, ESS and divergences per iteration and preregister the acceptance bounds.

**Baselines.** Sobol; DSP by MAP as the vanilla reference; SAASBO as a cell of the factorial (re-run with LogEI, not EI, so acquisition is matched); oracle-S as the bound. AdaScale-TuRBO and Linear BO are cheap and belong only on the Tang & Paulson dense first-order functions, where published curves at matched dimensions exist. Drop TuRBO-classic (superseded by AdaScale within its own lineage), BAxUS/Bounce, MCTS-VS and GTBO as run baselines; cite them.

**Benchmarks.** Keep: the synthetic families of SQ1–SQ3 (aligned, decoupled, dense-weak, interaction sweep, rotation sweep); embedded Hartmann-6 at D = 100 inside the factorial and embedded Levy-4 in the headroom references only, for comparability with Hvarfner and Doumont; Schwefel/Rastrigin/Michalewicz at D = 50 as the dense corner; Lasso-DNA (180D) and MOPTA08 (124D) as the two real out-of-distribution tasks, ≤ 300 evaluations, 5 seeds, two cells. Drop: SVM (388D, slow evaluations, heavy NUTS), MuJoCo Ant/Humanoid (888/6392D — NUTS cannot go there and the objectives are noisy), trajectory planning/Rover (non-additive, low priority). D = 300 is a spot check of two cells; D = 1000 is never run.

**Metrics.** Log simple regret and area under the regret curve on synthetics; best observed value on real tasks; active-set precision/recall, rank correlations and calibration on SQ1; effect sizes with confidence intervals against a preregistered smallest effect of interest; observation traveling-salesman distance split by active versus inactive coordinates as a cheap diagnostic of whether inferred relevance changes search behaviour (Papenmeier et al. UAI 2025 — use OTSD, not observation entropy, above 20 dimensions).

---

## Appendix A — Compute arithmetic

Cost model: one NUTS gradient costs about n³/3 + 2Dn² flops (Cholesky plus per-parameter kernel derivatives for the additive kernel; SAASBO's O(N³D) bound is comparable). Anchor: SAASBO reports 26.5 s per iteration at D = 100, n ≤ 50 with 768 NUTS steps on an 8-core CPU, and 19 s with the reduced 256-step budget, of which about three quarters is EI optimization — so a reduced-budget fit costs about 5 s per chain at n = 50. Scaling by the cost model gives about 100 s at n = 200 and 250 s at n = 300 for D = 100 (ratios 20× and 50×). Hvarfner et al. report that a single SAASBO fit on SVM (D = 388) takes upwards of 5 minutes on 4 CPUs in later iterations, which is in the same range.

Per run at D = 100, budget 200: summing the fit cost over n = 20 … 200 with a refit at every iteration gives about 1.7 chain-hours; with the planned schedule (every iteration to n = 100, then every 2–4) about 1 chain-hour. Acquisition optimization over 8–16 posterior samples adds roughly 1 hour per run. Doubling for JAX compilation, Python glue, deeper NUTS trees than the model assumes, and I/O gives the budget used below: **about 5–10 core-hours per run with 2 chains, 8–15 with 4.** At D = 300 the kernel-derivative term triples and the ~900-parameter posterior mixes more slowly: budget 40–75 core-hours per run. SQ1's offline fits are run with full-length chains (4 × 1,024 steps) for clean diagnostics and are therefore costed higher than the in-loop fits.

| Block | NUTS runs | Core-hours (budgeted) |
|---|---|---|
| SQ1 identification (offline fits, long chains) | ~500 fits | 100–200 |
| SQ2 factorial, D = 100 | 240 | 1,500–3,000 |
| SQ2 headroom references (D = 50 cells, embedded Levy-4) | 30 | 300–450 |
| SQ2 spot check, D = 300 | 10 | 400–750 |
| SQ3 interaction sweep | 100 | 1,000–1,500 |
| SQ3 dense corner (D = 50, ≤ 300 evals, refit every 10) | 60 | 250–400 |
| SQ3 rotation sweep (5 seeds) | 30 | 300–450 |
| Real tasks (Lasso-DNA, MOPTA08; D = 124–180, ≤ 300 evals) | 20 | 400–600 |
| SQ4 exact additive TS (optional) | 40 | 400–600 |
| **Total** | | **≈ 4,700–8,000** |

That is roughly a week of continuous use of 50 cores. Allowing for queueing and 4-core jobs, plan two to four weeks of calendar time inside November–January, with the D = 300 check, the rotation sweep and SQ4 as the first things to drop if the cluster is slow. Section 3's "about 300 core-days" is the top of this range.

## Appendix B — The kernel-normalization check

`direction/kernel_normalization_check.py` (numpy only, about a minute to run) computes v(ℓ) for the centered Matérn-5/2 kernel under U[0,1] by 64-node Gauss–Legendre quadrature, and evaluates the exact grid posterior over (a, ℓ) for one component (n = 80, noise sd 0.1) under a half-Cauchy(0.1) prior on a and a lognormal prior on ℓ with median 1 and sd 2 in log space, for five realized components (pure noise; strong-smooth; strong-rough; weak-smooth; weak-rough, each rescaled to an exact variance under U[0,1]), with and without normalization by v(ℓ). Run it before writing the NumPyro model: the same two tables should reproduce with the model's own kernel code, which is Gate 1.