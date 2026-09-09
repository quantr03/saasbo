# Design brief: amplitude vs lengthscale sparsity in fully Bayesian high-dimensional BO
 
**The 2 × 2 design, the synthetic objectives, the three studies and their preregistered predictions.**
Quan, 2 September 2026 — for discussion before SQ1 is preregistered. Longer versions: `direction/research-question.md` (candidates, weakest link, scope cuts) and `direction/primer-primary-question.md` (background).
 
---
 
## 1. The question
 
> *Holding kernel family (crossed), inference, acquisition, initialization and budget fixed, does moving the sparsity prior from inverse squared lengthscales to normalized additive amplitudes change which coordinates a fully Bayesian GP treats as relevant and how sample-efficiently BO optimizes sparse objectives — and where does any such difference end as first-order additivity fails?*
 
Three things changed relative to the June note.
 
1. **Normalization.** The centered component kernel $\tilde k_\ell$ has average marginal variance $v(\ell) = 1 - \mathbb E_{x,x'} k_\ell(x,x')$, which runs from $0.78$ at $\ell = 0.1$ to $0.0014$ at $\ell = 10$ (asymptotically $5/(36\ell^2)$). Without dividing by it, a component's amplitude $a_i$ is not its variance and "amplitude remains large" has no fixed meaning. Every amplitude cell therefore uses $\bar k_i = \tilde k_{\ell_i}/v(\ell_i)$, so that $\mathbb E\,\mathrm{Var}_\nu f_i = a_i^2$ and $a_i^2/\sum_j a_j^2$ is the first-order Sobol index. This is the one-line change that makes the sparsity prior a prior on variance shares.
2. **Crossed design.** The note's headline contrast (additive + amplitude sparsity vs SAASBO) changes structure and parameterization at once. I cross them.
3. **Additivity as a controlled factor.** Whether the objective is first-order additive is a dial ($\gamma$, below), not something the benchmarks smuggle in.
## 2. Design: $2 \times 2$, everything else held fixed
 
| | sparsity on normalized amplitude $a_i^2$ | sparsity on inverse squared lengthscale $\rho_i = \ell_i^{-2}$ |
|---|---|---|
| **additive** | $k = \sum_i a_i^2\,\bar k_i$; $a_i^2 \sim \mathrm{HC}(\tau)$ *(the proposal)* | $k = \sigma_f^2 \sum_i \tilde k_{\ell_i}$ (unnormalized, one amplitude); $\rho_i \sim \mathrm{HC}(\tau)$ |
| **product** | $k = \prod_i (1 + a_i^2\,\bar k_i)$; $a_i^2 \sim \mathrm{HC}(\tau)$ | Matérn-5/2 ARD, $\rho_i \sim \mathrm{HC}(\tau)$ *(SAASBO itself)* |
 
One global–local prior form in all four cells — SAASBO's own, $\theta_i \sim \mathrm{HC}(\tau)$, $\tau \sim \mathrm{HC}(\alpha)$ — so the lengthscale column *is* SAASBO and the amplitude column differs from it only in the parameter the prior sits on. Lengthscales in the amplitude cells carry a weakly informative log-normal prior on the $[0,1]$ scale; the same one in both. The additive–lengthscale cell must use the unnormalized kernel, because it is the only additive parameterization in which shrinking $\rho_i$ removes a component — that is the "small ⇔ smooth" coupling made into a cell.
 
**Held fixed in every cell:** Matérn-5/2 base kernel; inference exactly as in the SAASBO reference implementation (`martinjankowiak/saasbo`: NumPyro NUTS, 512 warm-up + 256 samples thinned to 16 retained, tree depth 6, one chain, a fresh chain at every refit, refit every iteration, zero mean on standardized targets, noise learned with the reference's prior); acquisition and optimizer also from the reference — EI averaged over the 16 retained samples, 5000 Sobol candidates plus the incumbent, top-5 L-BFGS-B — with seeding added; the same 20-point Sobol initialization per seed; $D = 100$; budget $T = 200$; 10 seeds. Two deliberate departures from the reference's BO driver, stated once: Matérn-5/2 instead of its RBF default, and learned rather than fixed ($10^{-6}$) noise, because the interaction sweep needs somewhere for unexplained variance to go.
 
**References outside the factorial:** Sobol search (floor); DSP by MAP (the vanilla baseline as it is used in the literature); an oracle GP given the true active set (ceiling for any variable selection).
 
**What I am dropping for now.** The third, non-sparse complexity-matched column. I take sparsity as a premise (low effective dimension; a sparse prior chosen for interpretability), so the thesis will claim *parameterization* and *structure* effects, not sparse-vs-dense. The cost of this cut is that "sparse priors are more interpretable than a well-scaled dense prior" stays an inherited premise, not a finding. If you would rather it be a finding, the cheapest way back is one cell — additive kernel, $a_i \sim \mathrm{HalfNormal}(1/\sqrt D)$, NUTS — in the offline study only, where it costs a few core-hours.
 
## 3. Synthetic objectives
 
I generate the objectives rather than borrow them, so that the ground truth is stated in the quantities the posteriors will be read in.
 
**Components.** For each active coordinate $i$, draw $g_i \sim \mathrm{GP}(0, \bar k_{\ell_i})$ on a fixed 1-D grid (1024 points plus 64 Gauss–Legendre nodes), then center and rescale under $\nu = U[0,1]$ to an *exact* variance:
 
$$
f_i(x_i) = \sqrt{\frac{s_i}{\widehat{\mathrm{Var}}_\nu(g_i)}}\;\bigl(g_i(x_i) - \widehat{\mathbb E}_\nu g_i\bigr),
\qquad
\widehat{\mathrm{Var}}_\nu(g_i) = \sum_q w_q\, g_i(t_q)^2 - \Bigl(\sum_q w_q\, g_i(t_q)\Bigr)^2 .
$$
 
Rescaling is not optional: a single draw's variance is $\sim a^2\chi^2_1$, so trusting the prior amplitude would put the ground truth off by a random factor. Each component is stored and evaluated by cubic spline, so $f$ is a fixed deterministic function usable inside the BO loop, and $f^\star$ is exact ($\mu + \sum_i \max f_i$ for additive parts; a pairwise grid for interaction terms).
 
**Interactions.** $h_{ij}(x_i, x_j) = c_{ij}\,u_i(x_i)\,u_j(x_j)$ with $u_i, u_j$ centered 1-D draws (generated as above, unit variance). Because the factors are centered and $\nu$ is a product measure, $h_{ij}$ is orthogonal to every main effect and $\mathrm{Var}_\nu h_{ij} = c_{ij}^2\,\mathrm{Var}_\nu u_i\,\mathrm{Var}_\nu u_j = c_{ij}^2$. Its variance share $s_{ij} := \mathrm{Var}_\nu h_{ij}$ is therefore set directly by $c_{ij} = \sqrt{s_{ij}}$; the first-order indices stay $s_i$, and the interaction share is exactly $\gamma = \sum_{(i,j) \in I} s_{ij}$ (split equally across the chosen pairs).
 
**Assembly.** $f = \sum_{i \in S} f_i + \sum_{(i,j) \in I} h_{ij}$ with $\sum_i s_i + \gamma = 1$, so $\mathrm{Var}_\nu f = 1$; observations $y = f(x) + \varepsilon$, $\varepsilon \sim \mathcal N(0, 0.1^2)$. $S$ is a random subset per seed, so index position cannot leak.
 
**Labels recorded per function:** $S$; $s_i$ (variance share); $\ell_i$; the realized slope share $g_i = \mathbb E_\nu[(f_i')^2] / \sum_j \mathbb E_\nu[(f_j')^2]$; $\gamma$; $f^\star$. Variance share and slope share are the two senses of relevance the two columns are hypothesized to track: $\mathbb E[(\partial_i f)^2] = \tfrac53 a_i^2 \rho_i$ versus $\mathbb E\,\mathrm{Var}_\nu f_i = a_i^2 v(\ell_i)$. They agree up to a constant for $\ell \gtrsim 1$ and diverge as $\ell \to 0$, which is what the decoupled family exploits.
 
| family | active set and shares | lengthscales | what it isolates |
|---|---|---|---|
| aligned | $\lvert S\rvert \in \{3, 10\}$, $s_i = 1/\lvert S\rvert$ | all $0.5$ | positive control: both columns should agree |
| decoupled | 2 coords per cell of $\{s = 0.21, 0.04\} \times \{\ell = 1.5, 0.08\}$ (strength × roughness); $4(0.21) + 4(0.04) = 1$ | $1.5$ / $0.08$ | the two senses of relevance disagree |
| anti-aligned | $\lvert S\rvert = 6$, $s = (0.35, 0.25, 0.18, 0.12, 0.07, 0.03)$, $\ell = (2, 1.2, 0.7, 0.4, 0.2, 0.1)$ | weak ones rough | sharpest version: share and slope rankings reversed |
| interaction sweep | aligned $\lvert S\rvert = 5$ plus pairs inside $S$, $\gamma \in \{0, .1, .25, .5, .75\}$ | $0.5$ | additivity failing |
| rotated | aligned family composed with a rotation of the active subspace, $\theta \in \{0°, 15°, 45°\}$ | $0.5$ | axis alignment failing |
| dense-weak | all $D$ coordinates, $s_i = 1/D$, monotone | $3$ | sparsity failing (Lasso-DNA-like); all shares below $\epsilon$ by design |
 
"Active" is defined once, for every metric and for the prior-scale rule below: a coordinate is active if its share exceeds $\epsilon = 0.02$, half the smallest active share in any sparse family.
 
Two out-of-family checks, because the generator is additive and Matérn and the additive cells are therefore in-family: embedded Hartmann-6 and Levy-4 at $D = 100$, and one family generated from rescaled sinusoids rather than a GP.
 
## 4. Studies, readouts, predictions
 
**SQ1 — identification (offline, cheap, first).** Fit all four cells to the same datasets: Sobol designs with $n \in \{30, 60, 100, 200\}$, $D = 100$, 10 seeds per family. Primary readout is parameterization-neutral: the first-order Sobol index of the posterior mean, $\hat S_i$ (exact per component in the additive cells; quasi-Monte Carlo in the product cells). Secondary: the native scores $a_i^2$ and $\rho_i$. Metrics: AUROC and precision/recall of the active set; Spearman of $\hat S_i$ against $s_i$ and against $g_i$; calibration of $P(a_i^2 > \epsilon)$ against the active indicator. Manipulation checks that gate everything downstream: posterior $a_i^2$ tracks realized component variance across lengthscales; amplitude ranking agrees with the Sobol ranking; $\hat R < 1.05$, ESS and divergence bounds.
*Predictions.* P1a: on the aligned family all four cells select the same set (Jaccard $> 0.9$ at $n = 100$). P1b: on the decoupled and anti-aligned families the amplitude column tracks $s_i$ (Spearman $> 0.8$) and the lengthscale column tracks $g_i$ better than $s_i$. If both columns track the same label even at $\ell = 0.08$, the two priors differ in parameterization only, and that is the result.
 
**SQ2 — sample efficiency (in-loop, confirmatory).** The $2 \times 2$ on aligned $\lvert S\rvert \in \{3, 10\}$, decoupled, and embedded Hartmann-6. Readout: simple regret $r_t = f^\star - \max_{s \le t} f(x_s)$, median and bootstrap interval over seeds, at $t \in \{50, 100, 200\}$ plus area under the curve; analysed as a two-factor design with effect sizes against a preregistered smallest effect of interest ($0.3$ in $\log_{10}$ regret at $T = 200$). Gate before spending: oracle-$S$ vs DSP-by-MAP headroom at $D = 100$ must exceed two standard errors, otherwise no selection method can matter at this budget and SQ1 becomes the core.
*Prediction* P2: the structure main effect exceeds the parameterization main effect on aligned objectives; a parameterization effect appears, if at all, on decoupled.
 
**SQ3 — boundary (in-loop).** Interaction sweep with the additive–amplitude cell, SAASBO-with-LogEI, DSP-by-MAP and Sobol; $\gamma^\star$ = the share at which the additive advantage over the matched product cell vanishes, by interpolation with a bootstrap interval. Rotation sweep and dense-weak as one-factor sweeps. Lasso-DNA and MOPTA08 enter only here, as out-of-distribution tasks with a predicted direction (amplitude sparsity loses to DSP because of their many weak, smooth, boundary-seeking coordinates).
*Prediction* P3: $\gamma^\star \in (0.1, 0.5)$ at $T = 200$, the same for both parameterizations.
 
## 5. Cost and timeline
 
Cost model: one NUTS gradient $\approx n^3/3 + 2Dn^2$ flops. With the reference's full budget (768 NUTS iterations per fit, refit every iteration, no warm start) a $D = 100$, $T = 200$ run is roughly $2$–$3\times$ the $5$–$15$ core-hours I assumed with a reduced budget, i.e. $\approx 15$–$40$ core-hours; one timed fit at $n = 200$ in the pilot settles it. At that rate SQ1 $\approx 200$–$400$ core-hours; SQ2 with four NUTS cells $\approx 2{,}500$–$6{,}000$; SQ3 $\approx 3{,}000$–$5{,}000$. If this does not fit the allocation, the fallback is a reduced budget applied identically in every cell (e.g. the reference demo's 256 + 256 thinned to 8), preregistered — never a per-cell change. The $D = 300$ spot check and the two real tasks are the first things to drop.
 
September: normalized orthogonal kernel by quadrature, NumPyro models for the four cells, BoTorch glue, Gate 1 on toys, preregister SQ1. October: SQ1 and pilot BO runs; preregister SQ2/SQ3. November–December: SQ2, headroom gate. December–January: SQ3. February: writing.
 
## 6. Where I would like your view
 
1. **Priority.** If the aim is identification (which coordinates a sparse GP treats as relevant, and in which sense), SQ1 is the core and SQ2 can shrink to two objectives; if the aim is sample efficiency, the allocation above stands. This decides where the compute goes and I would like to settle it before the end of September.
2. **The dropped column.** Whether "sparse priors are more interpretable" may stay a premise, or whether you want the one offline dense cell that would make it a finding.
3. **Prior scale.** $\alpha$ for $\tau \sim \mathrm{HC}(\alpha)$ on the $a_i^2$ scale (SAASBO uses $0.1$ on the $\rho$ scale); I plan to fix it in the pilot from the prior-predictive number of coordinates with $a_i^2 > \epsilon = 0.02$ (matched to the count SAASBO's own prior implies on the $\rho$ scale), and would rather agree the rule now than tune it later.