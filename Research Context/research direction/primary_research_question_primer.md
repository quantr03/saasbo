# Reading the primary question, clause by clause

**A primer and background for the recommended thesis question in `direction/research-question.md`.**
Prepared 31 Aug 2026 (revised the same day with typeset mathematics). Assumes a master's-level statistics background and no prior familiarity with Bayesian optimization; each section builds exactly what the corresponding clause of the question needs and points to the project paper that covers it in depth.

---

## The question

> *Holding kernel family (crossed), inference, acquisition, initialization and budget fixed, does moving the sparsity prior from inverse squared lengthscales to normalized additive amplitudes change which coordinates a fully Bayesian GP treats as relevant and how sample-efficiently BO optimizes sparse objectives — and where does any such difference end as first-order additivity fails?*

Read it as four claims wrapped around one experimental discipline. The discipline is the opening clause: everything that is not under study is held identical, and the one thing that could be confused with the thing under study — the kernel family — is crossed with it so its effect can be measured separately. The four claims are then about (i) *relevance*: what a Gaussian process posterior says about which input coordinates matter; (ii) the *prior* that is being moved, and the two places it can sit; (iii) *sample efficiency* on objectives that really are sparse; and (iv) the *boundary* of the whole idea, when the objective stops being a sum of one-dimensional pieces. The sections below take these in the order needed to understand them, which is not the order in which they appear in the sentence.

Notation used throughout: $D$ is the ambient dimension, $x \in [0,1]^D$ an input with coordinates $x_1, \dots, x_D$, $S \subset \{1,\dots,D\}$ the set of active coordinates, $n$ the number of observations so far and $T$ the evaluation budget.

---

## 1. The object: a Gaussian process inside a Bayesian optimization loop

**The optimization problem.** We want $x^\star = \arg\max_{x \in [0,1]^D} f(x)$, where $f$ is expensive to evaluate, has no available gradients, and $D$ is large (50–300 here). We may afford a few hundred evaluations. "Sample efficiency" means how good the best point found is after $T$ evaluations.

**The surrogate.** Bayesian optimization (BO) keeps a probabilistic model of $f$ — a Gaussian process (GP) — and uses it to choose each next evaluation. A GP prior $f \sim \mathcal{GP}(m, k)$ says that for any finite set of inputs the function values are jointly Gaussian, with mean $m(x)$ (a constant here, since outputs are standardized) and covariance $k(x, x')$. The kernel $k$ encodes what functions are plausible before seeing data: how smooth, how large in amplitude, how quickly correlation decays with distance in each coordinate.

Given $n$ observations $y = f(X) + \varepsilon$ with $\varepsilon \sim \mathcal{N}(0, \sigma_n^2 I)$, the posterior at a new point $x$ is Gaussian with

$$
\begin{gathered}
\mu(x) = k(x, X)\,\bigl(K + \sigma_n^2 I\bigr)^{-1} y, \\[4pt]
\sigma^2(x) = k(x, x) - k(x, X)\,\bigl(K + \sigma_n^2 I\bigr)^{-1} k(X, x),
\end{gathered}
$$

where $K = k(X, X)$ is the $n \times n$ kernel matrix. The kernel's hyperparameters $\theta$ (lengthscales, amplitudes, noise) are learned from the data through the log marginal likelihood

$$
\log p(y \mid X, \theta) = -\tfrac12\, y^{\top}\bigl(K_\theta + \sigma_n^2 I\bigr)^{-1} y \;-\; \tfrac12 \log\bigl|K_\theta + \sigma_n^2 I\bigr| \;-\; \tfrac{n}{2}\log 2\pi,
$$

whose first term rewards fitting the data and whose second term penalizes complexity. Everything in the question about "priors on lengthscales" or "priors on amplitudes" is a prior $p(\theta)$ added to this marginal likelihood.

**The loop.** Start with $n_0$ points from a space-filling design (a Sobol sequence — this is the *initialization* in the question). Then repeat: fit $\theta$, compute an *acquisition function* from the posterior, maximize it to pick the next $x$, evaluate $f$, append. The *budget* is the total number of evaluations $T$. Compute time per iteration is a separate cost, and it is the binding one for the fully Bayesian models here (Section 5).

**Regret.** For synthetic objectives whose optimum is known, performance is measured by simple regret

$$
r_T = f(x^\star) - \max_{t \le T} f(x_t),
$$

usually on a log scale, as a curve in $T$, averaged over seeds. "More sample-efficient" means lower regret at the same $T$.

*Read:* SAASBO §3 (`2103.00349v2.pdf`) for a compact statement of exactly this setup; Rasmussen & Williams for GPs generally.

---

## 2. "Which coordinates a fully Bayesian GP treats as relevant"

This clause is the heart of the question, and it hides a subtlety: *relevance is not a single concept*. A GP can encode "coordinate $i$ matters" in two structurally different ways, and the thesis is about whether the difference between them matters.

### 2.1 Relevance through lengthscales (automatic relevance determination)

The standard high-dimensional kernel couples all coordinates through one scaled distance (automatic relevance determination, ARD). With the Matérn-5/2 base used throughout the project,

$$
k(x, x') = \sigma_f^2 \Bigl(1 + \sqrt{5}\, r + \tfrac{5}{3} r^2\Bigr) e^{-\sqrt{5}\, r},
\qquad
r^2 = \sum_{i=1}^{D} \frac{(x_i - x_i')^2}{\ell_i^2}.
$$

Each coordinate has its own lengthscale $\ell_i$: the distance along coordinate $i$ over which $f$ is expected to change appreciably. A very long lengthscale ($\ell_i \to \infty$) makes the kernel insensitive to $x_i$, so sample functions are constant along that coordinate — the coordinate is *switched off*. This is MacKay and Neal's automatic relevance determination (ARD): after fitting, coordinates with long lengthscales are read as irrelevant.

SAASBO parameterizes this with inverse squared lengthscales $\rho_i = 1/\ell_i^2$ and puts a sparsity prior on $\rho$ (Section 3), so that $\rho_i \approx 0$ for most coordinates. This is what the question calls "the sparsity prior on inverse squared lengthscales."

Notice what relevance means here. There is one global amplitude $\sigma_f^2$. The kernel does not have a separate knob for *how much* $f$ varies along coordinate $i$; it only has $\ell_i$, which sets how *quickly* $f$ varies along $i$. A coordinate carrying a large but very smooth effect (say a strong linear trend) has to be represented with a moderate $\rho_i$, because with $\sigma_f^2$ fixed the size of variation along $i$ over the unit interval is $\sigma_f^2$ times an increasing function of $\rho_i$; a coordinate carrying a small but rapidly oscillating effect needs a large $\rho_i$ even though it contributes little variance. Lengthscale relevance is therefore a *roughness-weighted* notion of strength, and it cannot separate "large" from "rough."

### 2.2 Relevance through amplitudes (additive kernels)

A first-order additive model writes

$$
f(x) = \mu + \sum_{i=1}^{D} f_i(x_i), \qquad f_i \sim \mathcal{GP}\bigl(0,\, a_i^2\, \tilde{k}_i\bigr),
\qquad\text{so}\qquad
k(x, x') = \sum_{i=1}^{D} a_i^2\, \tilde{k}_i(x_i, x_i'),
$$

a sum of one-dimensional GPs, each with its own amplitude $a_i$ and its own lengthscale $\ell_i$ inside $\tilde{k}_i$. Now each coordinate has *two* knobs: $a_i$ sets how much $f$ varies along $i$, $\ell_i$ sets the shape of that variation. A coordinate is switched off by $a_i \to 0$, regardless of $\ell_i$. This is "the sparsity prior on additive amplitudes": put the shrinkage prior on $a = (a_1, \dots, a_D)$.

Provided the component kernels are *normalized* (Section 4.3), $a_i^2$ is exactly the variance of the $i$-th component under the reference measure on $[0,1]$, so amplitude relevance is a *variance-share* notion: coordinate $i$ is relevant in proportion to how much of $\operatorname{Var} f$ it explains. This is the first-order Sobol index of global sensitivity analysis, and it is why the additive model connects to functional ANOVA (Section 7).

### 2.3 Why "fully Bayesian" belongs in this clause

With a sparsity prior, the *posterior* over $a_i$ (or $\rho_i$) is the object that says whether coordinate $i$ is relevant — for example through $\Pr(a_i > \epsilon \mid \mathcal{D})$, the note's active-set rule. A point estimate cannot deliver that. Worse, the global–local shrinkage priors used here have a density that goes to infinity at zero, so the maximum a posteriori estimate collapses to "everything off" and is useless (Section 3). Inference therefore has to produce posterior samples — Markov chain Monte Carlo, specifically NUTS (Section 5.2) — and "which coordinates the GP treats as relevant" is a statement about those samples: their medians, their tail probabilities, and, more robustly, the posterior distribution of each component's Sobol index.

### 2.4 The mechanism hypothesis the thesis tests

Putting 2.1 and 2.2 together: the two priors do not merely sit on different parameters; they *tie different things together*. Lengthscale shrinkage couples "small effect" to "smooth effect" (a nearly-off coordinate is necessarily nearly linear) and, with a bounded global amplitude, "large effect" to "rough effect." Amplitude shrinkage leaves shape free. On the standard benchmarks, where the relevant coordinates are both strong and rough relative to distractors that are exactly zero, both priors will select the same set and the difference is invisible. The difference should show where strength and roughness are *decoupled* — strong-smooth effects, weak-rough distractors — and, in the other direction, where many coordinates carry weak, smooth, monotone effects, which lengthscale-based models get almost for free and amplitude shrinkage zeroes. Papenmeier et al. (ICML 2025, App. D) show that Lasso-DNA and MOPTA08, two of the most used real benchmarks, have exactly that structure: most of their coordinates are "secondary," best set at the boundary, and modelled with long lengthscales. That is why the recommendation predicts amplitude sparsity will *lose* there, and treats it as a result rather than a failure.

*Read:* SAASBO §4.1 for the ARD view; OAK §3–4 (`2206.09861v1.pdf`) for the additive/Sobol view; Hvarfner et al. Fig. 8 (`2402.02229v5.pdf`) for a vanilla model that optimizes well without confidently identifying relevant coordinates at all.

---

## 3. "Sparsity prior": from spike-and-slab to the horseshoe to SAAS

**What sparsity means for a GP.** Of $D$ coordinates, only $|S| \ll D$ matter. A prior that says so has to put most of its mass at "off" while allowing a few coordinates to be strongly "on."

**Spike-and-slab** (Mitchell & Beauchamp 1988; George & McCulloch 1993) does this literally: each parameter is either drawn from a point mass at zero (the spike) or from a diffuse distribution (the slab), governed by a binary inclusion variable. It is the conceptual ideal and the reason the supervisor's note lists it, but the binary variables make posterior sampling a search over $2^D$ models, which is why the note defers it.

**Global–local shrinkage** replaces the binary switch with a continuous scale. The horseshoe (Carvalho, Polson & Scott 2009) writes, for a parameter $\beta_i$,

$$
\beta_i \mid \lambda_i, \tau \;\sim\; \mathcal{N}\bigl(0,\, \lambda_i^2 \tau^2\bigr),
\qquad
\lambda_i \sim \mathrm{C}^{+}(0, 1),
\qquad
\tau \sim \mathrm{C}^{+}(0, \tau_0).
$$

$\tau$ is the *global* scale — how sparse the whole vector is — and $\lambda_i$ the *local* scale that lets an individual coordinate escape. The half-Cauchy's heavy tail lets $\lambda_i$ become very large when the data demand it, while its density near zero pulls everything else in. The induced shrinkage weight

$$
\kappa_i = \frac{1}{1 + \lambda_i^2 \tau^2}
$$

has a $\mathrm{Beta}(\tfrac12, \tfrac12)$ distribution when $\tau = 1$ — a horseshoe shape with mass piled at both ends, "shrink to zero or leave alone." Two properties matter for this thesis: the prior is continuous (so gradient-based samplers work), and its density is unbounded at zero (so MAP estimation is degenerate and full posterior sampling is mandatory).

**SAASBO's prior** is the same idea applied to inverse squared lengthscales, in a slightly simpler form:

$$
\begin{gathered}
\sigma_f^2 \sim \mathrm{LogNormal}(0, 10^2), \qquad \tau \sim \mathrm{C}^{+}(0, \alpha), \\[4pt]
\rho_i \sim \mathrm{C}^{+}(0, \tau)\quad (i = 1, \dots, D), \qquad \alpha = 0.1 \ \text{by default}.
\end{gathered}
$$

Most $\rho_i$ sit below $\tau$, which itself concentrates near zero, so most coordinates are off; as data accumulate the posterior on $\tau$ grows and more coordinates turn on. Eriksson & Jankowiak note that this places a shrinkage prior "on inverse squared lengthscales in a non-linear kernel and not on variances," and explicitly leave open whether other distributional choices would be better (their App. A.3). The recommended design uses this exact functional form in every sparse cell — on $\rho_i$ in the lengthscale cells, on $a_i^2$ in the amplitude cells — so that the only difference between the sparse columns is the parameter the prior sits on. The note's horseshoe variant and the R2-D2 prior (a Beta prior on the model's $R^2$, distributed to coordinates by a Dirichlet, which in the normalized additive model is literally a prior on the first-order Sobol shares) are kept as appendix robustness checks.

**What "sparse" buys and what it doesn't.** In SAASBO's Fig. 1 a dense GP fit by either maximum likelihood or NUTS with weak priors reverts to a trivial mean prediction on a 124- or 388-dimensional real task, while the sparse prior gives a usable fit: with $n \approx 100$ points and $D \approx 100$ lengthscales, the model is hopelessly over-parameterized unless something regularizes it. Hvarfner et al. (2024) later argued that what matters is the *reduction in prior complexity* rather than sparsity as such — a point that turns into the complexity-matched controls of Section 6.

*Read:* Carvalho et al. (`carvalho09a.pdf`) §1–2; SAASBO §4.1 and App. A.3; the R2-D2 paper (`1609.00046v3.pdf`) §2 for the $R^2$ construction.

---

## 4. "Moving the prior from inverse squared lengthscales to normalized additive amplitudes"

This clause names the two kernel families and one non-obvious technical requirement. The families first.

### 4.1 Product kernels versus additive kernels

The ARD kernel of Section 2.1 couples all coordinates in a single distance, so $f$ can have interactions of every order — the function value can depend on coordinates jointly in arbitrary ways. (For the squared-exponential base this kernel is literally a product of per-coordinate factors, $\prod_i e^{-\rho_i d_i^2/2}$; for the Matérn base it is not separable, but the interaction structure is the same.) Switching a coordinate off ($\rho_i \to 0$) removes it from every interaction at once.

The first-order additive kernel of Section 2.2 is a *sum* over coordinates: $f$ is a sum of one-dimensional functions and has no interactions at all. This is a much smaller function class. Kandasamy et al. (2015) showed that under exact additivity the cumulative regret of BO grows only linearly in $D$ rather than exponentially, and Duvenaud et al. (2011) built kernels with all interaction orders as a middle ground. Whether the restriction helps depends entirely on whether the objective is close to additive — which is what the final clause of the question is about.

The point that matters for the design: the supervisor's headline contrast "SAASBO vs this project" changes *both* the family (product $\to$ additive) *and* the prior's parameter ($\rho \to a$). If the proposal wins a head-to-head, one cannot tell whether the additive restriction or the parameterization did it. The question's opening clause fixes this by crossing the two: four sparse cells (product/additive $\times$ $\rho$-prior/$a$-prior), where the product-family amplitude cell is the full-order orthogonal kernel

$$
k(x, x') = \prod_{i=1}^{D} \bigl(1 + a_i^2\, \tilde{k}_i(x_i, x_i')\bigr)
$$

with shrinkage on $a$. Multiplying it out gives $1 + \sum_i a_i^2\tilde k_i + \sum_{i<j} a_i^2 a_j^2\,\tilde k_i\tilde k_j + \cdots$, a sum over all interaction orders in which the interaction among a set $u$ of coordinates has variance $\prod_{i\in u} a_i^2$; so $a_i \to 0$ removes coordinate $i$ from every interaction, exactly as $\rho_i \to 0$ does in SAASBO. "Product family" in the design therefore means "all interaction orders, axis-aligned switch-off"; the two cells in that row differ slightly in separability unless the squared-exponential base (SAASBO's own default) is used for both.

### 4.2 Orthogonalization: why the component kernels are centered

A sum of functions is not identifiable: $f_1(x_1) + f_2(x_2) = \bigl(f_1(x_1) + c\bigr) + \bigl(f_2(x_2) - c\bigr)$ for any constant $c$. In the GP model this means constant offsets can move freely between components (and into the mean $\mu$), and amplitudes would absorb them. OAK (Lu, Boukouvalas & Hensman 2022, following Durrande et al. 2012) fixes this by constraining each component to integrate to zero under a reference measure $\nu_i$ on its coordinate,

$$
\int f_i(x_i)\, \mathrm{d}\nu_i(x_i) = 0,
$$

which makes the decomposition unique and turns it into the functional-ANOVA decomposition.

The constrained kernel is obtained from the base kernel $k$ by removing the constant. Two constructions appear in the project. The supervisor's note (eq. 2–3) uses the *projection* form,

$$
\tilde{k}(u, v) = k(u, v) - \int k(u, z)\, \mathrm{d}\nu(z) - \int k(z, v)\, \mathrm{d}\nu(z) + \iint k(z, z')\, \mathrm{d}\nu(z)\, \mathrm{d}\nu(z'),
$$

which is the covariance of $f - \int f \,\mathrm{d}\nu$. The OAK paper uses the *conditioning* form,

$$
\begin{gathered}
\tilde{k}(u, v) = k(u, v) - \frac{m(u)\, m(v)}{c}, \\[4pt]
m(u) = \int k(u, z)\, \mathrm{d}\nu(z),
\qquad
c = \iint k(z, z')\, \mathrm{d}\nu(z)\, \mathrm{d}\nu(z'),
\end{gathered}
$$

which is the covariance of $f$ given $\int f \,\mathrm{d}\nu = 0$. Both produce components that integrate to zero, and for the uniform measure on $[0,1]$ their average variances differ by under 3 % at every lengthscale, so either serves; the code has to pick one and say which. The integrals are computed once per lengthscale by Gauss–Legendre quadrature on $[0,1]$ — OAK's normalizing flows and Gaussian measures are for data features of unknown distribution, and are unnecessary when the domain is the unit cube by construction. Centering on the *empirical* measure of the observed inputs, which the note offers as an option, should be avoided in a BO loop: the observed inputs drift toward the incumbent, so the "constant" being removed would change from iteration to iteration.

### 4.3 Normalization: why "normalized" is in the question

Here is the technical requirement that the note misses and that the whole relevance story depends on. Define

$$
v(\ell) = \mathbb{E}_{x \sim \nu}\bigl[\tilde{k}(x, x)\bigr] = 1 - \mathbb{E}_{x, x' \sim \nu}\bigl[k(x, x';\, \ell)\bigr],
$$

the average marginal variance of a centered component with unit base amplitude. Numerically, for the Matérn-5/2 base under the uniform measure,

| $\ell$ | 0.1 | 0.5 | 1 | 3 | 10 |
|---|---|---|---|---|---|
| $v(\ell)$ | 0.78 | 0.28 | 0.107 | 0.015 | 0.0014 |

and for large $\ell$,

$$
v(\ell) \;\to\; \frac{5}{36\,\ell^2},
\qquad
\tilde{k}(x, x') \;\to\; \frac{5}{3\,\ell^2}\,\bigl(x - \tfrac12\bigr)\bigl(x' - \tfrac12\bigr),
$$

because the centered kernel collapses to a rank-one linear kernel. (Sketch: for small distances $k \approx 1 - \tfrac{5}{6}\, r^2/\ell^2$; centering a quadratic in $x - x'$ leaves only the cross term.)

Consequently a component with hyperparameters $(a_i, \ell_i)$ has variance

$$
\operatorname{Var}_\nu f_i = a_i^2\, v(\ell_i), \quad\text{not}\quad a_i^2 .
$$

A smooth component needs a *large* amplitude to represent even a small variance, and an irrelevant coordinate can be switched off by $\ell_i \to \infty$ with $a_i$ left wherever the prior puts it. Reading relevance from $a_i$, or thresholding $\Pr(a_i > \epsilon \mid \mathcal{D})$, is then not well defined: in a one-component toy with a weakly informative lengthscale prior, a smooth component carrying 2 % of the output variance gets a 45 % posterior probability of exceeding $a = 0.3$, and two components of *equal* variance get amplitudes of 1.66 and 0.55 purely because of their lengthscales. Dividing the centered kernel by $v(\ell)$,

$$
\bar{k}_i(u, v) = \frac{\tilde{k}_i(u, v)}{v(\ell_i)}
\qquad\text{so that}\qquad
\mathbb{E}_{x \sim \nu}\bigl[\bar{k}_i(x, x)\bigr] = 1
\ \text{ and }\
\operatorname{Var}_\nu f_i = a_i^2,
$$

brings those numbers to 7 % and to 0.70 versus 0.52. Under normalization, $\ell_i \to \infty$ no longer switches a component off; it makes the component linear — the normalized limit kernel is $12\,(x - \tfrac12)(x' - \tfrac12)$ — which is exactly the reading in which "lengthscales are not relevance indicators" is true. The script `direction/kernel_normalization_check.py` reproduces both tables and is the first thing to run against the thesis's own kernel code.

Normalization also changes what "lengthscale sparsity" means inside the additive family. With the normalized kernel, shrinking $\rho_i$ only makes component $i$ linear — a smoothness preference, not sparsity. So the lengthscale-sparse *additive* cell of the design has to use the *unnormalized* centered kernel, where the component variance scales like $\rho_i$ for long lengthscales (since $v(\ell) \approx \tfrac{5}{36}\rho$) and shrinking $\rho_i$ does remove the component. That cell is precisely the "small effect $\Leftrightarrow$ smooth effect" coupling of Section 2.4, now as an experimental condition.

*Read:* OAK §3 and Appendix D–G; the note's §4.1; `research-question.md` §1 and §4.

---

## 5. "Holding inference … fixed": what fully Bayesian inference costs and why it is needed

### 5.1 Why not MAP

Maximum a posteriori estimation maximizes $\log p(y \mid X, \theta) + \log p(\theta)$. With a global–local prior whose density diverges at zero, the maximizer is at zero for every coordinate: MAP cannot be used with these priors as they stand. SAASBO's MAP variant sidesteps this by fixing $\tau$ on a small grid and selecting by leave-one-out likelihood; even so, their Fig. 2 shows NUTS beating MAP by a wide margin on an embedded Branin problem *although MAP finds the same two relevant coordinates*. Their explanation is that averaging the acquisition function over posterior samples of the hyperparameters — including samples from different modes — makes the acquisition far more robust than any single point estimate. For this thesis that is a design fact: fully Bayesian inference is part of the method in every cell, not a luxury.

### 5.2 NUTS in one paragraph

Hamiltonian Monte Carlo simulates a particle moving over the negative log posterior $-\log p(\theta \mid y)$ using its gradient, which lets it take long, well-directed steps in hundreds of dimensions where random-walk samplers stall. The No-U-Turn Sampler (Hoffman & Gelman 2014) chooses the trajectory length automatically and, during a warm-up phase, adapts the step size and a diagonal mass matrix. The posterior here has about $2D + 2$ parameters (amplitudes or inverse squared lengthscales, lengthscales, $\tau$, noise) and a funnel-shaped geometry typical of global–local priors; SAASBO handles the funnel with the non-centered parameterization

$$
\rho_i = \tau\, \tilde{\rho}_i, \qquad \tilde{\rho}_i \sim \mathrm{C}^{+}(0, 1),
$$

and the thesis should do the same for $a_i$. Diagnostics to log at every BO iteration: $\hat{R}$ (agreement between chains, want $< 1.05$), effective sample size, and the number of divergent transitions (numerical failures that flag geometry problems; want near zero). Warm-starting each iteration's chains from the previous iteration's adapted state cuts warm-up cost and is standard.

### 5.3 What "held fixed" means and why it is the binding cost

All six cells use NUTS with identical settings: the same warm-up and sampling lengths (SAASBO's reduced budget of 128 + 128 steps thinned to 8 retained samples, tree depth $\le 6$), the same chain count, the same warm-start rule, the same refit schedule (every iteration until $n = 100$, then every 2–4). The cost of one gradient scales like

$$
\text{cost} \;\approx\; \tfrac13 n^3 + 2 D n^2
$$

(a Cholesky factorization plus one kernel derivative per hyperparameter), which is why runs are budgeted at 5–15 core-hours each at $D = 100$ and 40–75 at $D = 300$, and why $D = 300$ is a spot check rather than a grid. Hvarfner et al. report that a single SAASBO fit on the 388-dimensional SVM task takes upwards of five minutes on four CPUs late in a run; this is the practical reason the "vanilla" papers use MAP, and the reason the thesis cannot run every baseline on every benchmark.

*Read:* SAASBO §4.2 and App. A; the NumPyro NUTS documentation; `research-question.md` App. A for the arithmetic.

---

## 6. "Holding acquisition, initialization and budget fixed" and the complexity-matched controls

### 6.1 Acquisition: LogEI everywhere

Expected improvement is the classic acquisition: with incumbent value $y^\star$ and posterior $(\mu(x), \sigma(x))$,

$$
\mathrm{EI}(x) = \bigl(\mu(x) - y^\star\bigr)\,\Phi(z) + \sigma(x)\,\varphi(z),
\qquad
z = \frac{\mu(x) - y^\star}{\sigma(x)},
$$

the expected gain over the best value seen so far ($\Phi$ and $\varphi$ are the standard normal CDF and density). Ament et al. (2023) showed that EI and its gradient underflow to exactly zero over most of a high-dimensional space, crippling gradient-based maximization, and that computing $\log \mathrm{EI}$ with a numerically stable expression (LogEI) fixes this at no cost. The 2025–26 papers in the project benchmark against LogEI, and SAASBO, which used plain EI, is re-run with LogEI in the design so that acquisition is matched. With a fully Bayesian model the acquisition is averaged over the $L$ retained posterior samples of $\theta$:

$$
\overline{\mathrm{EI}}(x) = \frac{1}{L} \sum_{l=1}^{L} \mathrm{EI}\bigl(x;\, \theta^{(l)}\bigr).
$$

Maximization uses BoTorch's standard routine — a Sobol batch of candidates plus perturbations of the best observed points ("sample around best," the RAASP idea from trust-region BO), then multi-start L-BFGS-B — with identical settings in every cell. Papenmeier et al. (2025) show that this local candidate set is what rescues acquisition optimization from vanishing gradients in high dimensions, which is why the question does not treat acquisition optimization as the thing under study (the note's additive acquisition surrogates are cut; exact additive Thompson sampling is an optional fourth sub-question).

### 6.2 Initialization and budget

Every run in a seed starts from the same $n_0 = 20$ Sobol points, so comparisons between cells are paired and seed-to-seed variance partly cancels. The budget is $T = 200$ evaluations at $D = 100$ ($\le 300$ on the real tasks). Ten seeds per cell is the compromise between the cost of NUTS and the standard errors needed to detect a preregistered smallest effect of interest (0.3 in $\log_{10}$ regret at $T = 200$).

### 6.3 Complexity control: the √D prior and why the design needs non-sparse controls

Hvarfner et al. (2024) made an argument that reframed the field. In $[0,1]^D$ the expected distance between two random points grows like $\sqrt{D}$ (it lies between $\tfrac13\sqrt{D}$ and $\tfrac{1}{\sqrt6}\sqrt{D}$ for large $D$), so with fixed lengthscales the kernel matrix of any realistic dataset approaches the identity as $D$ grows: the model assumes so complex a function that no two observations inform each other, and BO degenerates (they quantify this with the maximal information gain of Srinivas et al.). Every high-dimensional BO method — embeddings, additive kernels, trust regions, sparsity priors — can be read as a way of lowering that assumed complexity. Their fix needs no structure at all: scale the lengthscale prior with dimension,

$$
\ell_i \sim \mathrm{LogNormal}\Bigl(\sqrt{2} + \tfrac12 \log D,\; \sqrt{3}\Bigr),
\qquad
\sigma_f^2 = 1 \ \text{(fixed)},
$$

which shifts the prior mode by a factor of $\sqrt{D}$. This "DSP" model, fit by MAP, matches or beats SAASBO on most real benchmarks, and Xu et al. (ICLR 2025) and Papenmeier et al. (2025) reach similar conclusions through initialization and maximum-likelihood fitting. Doumont et al. (2026) push it further: a linear model on spherically projected inputs is competitive from 60 to 6,000 dimensions.

The consequence for the question: if a sparse cell beats a dense one, that could be sparsity *or* mere complexity reduction. So the design has a third column, "no sparsity, complexity-matched": in the product family the DSP prior fit by NUTS (the fully Bayesian analogue of vanilla BO); in the additive family

$$
a_i \sim \mathrm{HalfNormal}\bigl(1/\sqrt{D}\bigr)
\qquad\Longrightarrow\qquad
\sum_{i=1}^{D} \mathbb{E}\bigl[a_i^2\bigr] = D \cdot \frac{1}{D} = 1,
$$

which makes the total prior signal variance equal to the standardized output variance — the additive analogue of the $\sqrt{D}$ scaling — with no shrinkage. The full design is:

| | sparsity on amplitude | sparsity on inverse squared lengthscale | no sparsity (complexity-matched) |
|---|---|---|---|
| **additive** $\sum_i a_i^2 \tilde{k}_i$ | SAAS-form prior on $a_i^2$ (the proposal) | unnormalized $\tilde{k}$, common amplitude, SAAS prior on $\rho$ | $a_i \sim \mathrm{HalfNormal}(1/\sqrt{D})$, DSP-style $\ell$ prior |
| **product** | $\prod_i (1 + a_i^2 \tilde{k}_i)$, SAAS-form prior on $a_i^2$ | SAASBO itself ($\rho_i \sim \mathrm{C}^{+}(\tau)$, $\tau \sim \mathrm{C}^{+}(0.1)$) | DSP prior on $\ell$, $\sigma_f^2 = 1$, fit by NUTS |

Outside the factorial sit three references: Sobol search (the floor), DSP by MAP (the actual vanilla baseline as used in the literature), and an *oracle* GP that is told the true active coordinates (the ceiling on what any variable selection could deliver — Section 7.2).

*Read:* Hvarfner et al. §2.3, §4–5; LogEI (`2310.20708v3.pdf`) §1–3; Papenmeier et al. ICML 2025 §3.2 (`2502.09198v2 1.pdf`).

---

## 7. "How sample-efficiently BO optimizes sparse objectives" and "where any difference ends"

### 7.1 Functional ANOVA: the language for "sparse" and "first-order additive"

Any square-integrable function on a product space with a product measure decomposes uniquely as

$$
f(x) = f_\varnothing + \sum_{i} f_i(x_i) + \sum_{i < j} f_{ij}(x_i, x_j) + \cdots,
$$

where each component integrates to zero over each of its own variables (the same constraint OAK imposes on the GP components). The components are orthogonal, so the variance splits,

$$
\operatorname{Var} f = \sum_{u \ne \varnothing} \operatorname{Var} f_u,
\qquad
S_u = \frac{\operatorname{Var} f_u}{\operatorname{Var} f},
$$

and the Sobol index $S_u$ measures the share of variance carried by the subset $u$ of coordinates. In this language:

- The objective is **sparse** if only a small set $S$ of coordinates appears in any component with non-negligible variance.
- It is **first-order additive** if all the variance is in the main effects, i.e. the interaction share vanishes:

$$
\gamma \;:=\; \sum_{|u| \ge 2} S_u \;=\; 0 .
$$

- The additive GP assumes $\gamma = 0$; SAASBO's product kernel does not.
- The interaction share $\gamma$ is the single knob that turns "first-order additivity fails" into a controlled experimental factor.

### 7.2 The synthetic families and the headroom check

The sample-efficiency sub-question runs on objectives built to have known structure. *Aligned*: $|S| \in \{3, 10\}$ active coordinates with equal variance and similar lengthscale, distractors exactly zero — the case in which both priors should agree (the preregistered positive control). *Decoupled*: a $2 \times 2$ of strength $\times$ roughness among the active coordinates, where the mechanism of Section 2.4 predicts the priors diverge. *Embedded Hartmann-6*: a non-additive but sparse standard test function, for comparability with Hvarfner and Doumont. Each is generated by sampling one-dimensional components from a GP, rescaling them to exact variance shares under the uniform measure, and summing.

Before spending the confirmatory budget, one cheap check decides whether there is anything to win: compare an oracle GP restricted to the true active coordinates against DSP-vanilla on the full space at $D = 100$. If that gap is inside two standard errors, no variable-selection method can matter at this dimension and budget, and the thesis pivots to the identification sub-question as its core. Hvarfner's Fig. 5 (SAASBO beating vanilla on most embedded Levy/Hartmann tasks at 25–300 dimensions) is the published evidence that the gap exists on axis-aligned sparse tasks; nobody has reported the oracle bound.

### 7.3 The phase boundary

The final sub-question takes the aligned family with $|S| = 5$ and adds pairwise interactions among the active coordinates,

$$
f(x) = \sum_{i \in S} g_i(x_i) + \sum_{(i,j) \in P} h_{ij}(x_i, x_j),
$$

with the $h_{ij}$ built as products of centered one-dimensional functions so they are orthogonal to the main effects, and scaled so that $\gamma \in \{0, 0.1, 0.25, 0.5, 0.75\}$. As $\gamma$ grows, the additive cells are increasingly misspecified: the interaction variance has nowhere to go except into the noise term and into distorted main effects. The product cells and DSP are not misspecified. The quantity of interest is $\gamma^\star$, the interaction share at which the additive model's advantage over the matched product model disappears, estimated by interpolation with a bootstrap interval. The preregistered prediction is $\gamma^\star \in (0.1, 0.5)$ at $T = 200$, the same for both parameterizations. Two more one-factor sweeps probe the assumption differently: rotating the active subspace so that the sparse structure is no longer axis-aligned (SAASBO's App. D.4 did a random-projection version and found SAAS robust), and dense first-order objectives — Tang & Paulson's Schwefel, Rastrigin and Michalewicz functions at $D = 50$, which are exactly additive but not sparse, so the complexity-matched additive control should beat the sparse one.

Real benchmarks (Lasso-DNA, MOPTA08) enter only here, as out-of-distribution tests with a predicted direction. This is the design's answer to the objection, raised in `references/high-dimensional-bo-papers.md`, that benchmark choice is not neutral: the truth of the assumption is a factor the experiment controls, not something the benchmarks smuggle in.

*Read:* OAK §4 for FANOVA and Sobol indices; SAASBO App. D.4; Tang & Paulson App. D (`2604.22967v1.pdf`) for the dense functions; Papenmeier et al. ICML 2025 App. D for the structure of the real benchmarks.

---

## 8. The landscape the question sits in

Read chronologically, the project's literature tells a single story that the question has to survive.

**2015–2022: structure.** Kandasamy et al. (additive kernels, linear-in-$D$ regret), Duvenaud et al. (all-order additive kernels), OAK (identifiable additive components via orthogonality), and SAASBO (sparsity on inverse squared lengthscales with NUTS) all attack high dimensions by restricting or regularizing the function class. SAASBO became the standard strong baseline.

**2024–2026: pushback.** Hvarfner et al. (ICML 2024): scale the lengthscale prior with $\sqrt{D}$ and vanilla BO matches or beats structural methods; the problem was assumed complexity, not dimension. Xu et al. (ICLR 2025): the same conclusion via initialization — vanishing gradients of the marginal likelihood, not the model class, explained past failures. Papenmeier et al. (ICML 2025): maximum likelihood with scaled initialization plus local candidate sampling suffices, and the popular benchmarks are simpler than their dimension suggests. Doumont et al. (AISTATS 2026): even Bayesian linear regression, after a spherical projection, matches the state of the art. Tang & Paulson (2026): the same prior-complexity argument applied inside trust regions.

**What is left open, and what the question claims.** None of these papers crosses kernel family with sparsity parameterization; every published SAASBO-versus-vanilla comparison confounds the prior with the inference scheme (NUTS versus MAP); no one has placed a sparsity prior on the amplitudes of an orthogonal additive GP, asked which *criterion* of relevance a sparse GP posterior implements, reported the oracle bound on the value of variable selection, or measured the interaction share at which additive sparsity stops paying. The question is built so that whichever way each of these goes, the thesis says something the literature does not yet say.

---

## 9. What the answers would look like

The question resolves into three preregistered predictions, each with a stated failure.

*P1 (identification, offline).* On aligned objectives, all four sparse cells select the same active set (Jaccard $> 0.9$ at $n = 100$). On decoupled objectives, the amplitude ranking tracks variance share (Spearman $> 0.8$) while the lengthscale ranking tracks roughness-weighted strength better than variance share. Failure: the rankings are indistinguishable on decoupled objectives, or amplitude sparsity fails to recover even the aligned set. Null thesis: amplitude and lengthscale shrinkage are the same relevance detector; here is the matched evidence, and the normalization result stands regardless.

*P2 (sample efficiency, in-loop).* The structure main effect exceeds the parameterization main effect on aligned objectives; a parameterization effect appears, if at all, on the decoupled objective. Failure: no parameterization effect anywhere against the smallest effect of interest. Null thesis: the additive restriction, not where the prior sits, is what buys sample efficiency.

*P3 (boundary).* $\gamma^\star \in (0.1, 0.5)$ at $T = 200$, the same for both parameterizations; the sparse additive cell loses to the complexity-matched additive control on dense objectives and to DSP on Lasso-DNA and MOPTA08. Failure: $\gamma^\star \approx 0$ (the additive model loses already at $\gamma = 0.1$) or no advantage even at $\gamma = 0$. Null thesis: first-order additive sparsity has no practical win region, and here is the map.

The reason the question is worth four months is that each of the three null theses is a complete, honest thesis, and none of them requires beating a baseline on a real benchmark.

---

## 10. Glossary of symbols and terms

| Symbol / term | Meaning |
|---|---|
| $D$, $S$, $|S|$ | ambient dimension; set of active coordinates; its size |
| $f_i$, $a_i$, $\ell_i$ | $i$-th additive component; its amplitude; its lengthscale |
| $\rho_i = 1/\ell_i^2$ | inverse squared lengthscale, SAASBO's relevance parameter |
| $\tilde{k}_i$, $\bar{k}_i$ | centered (orthogonalized) component kernel; the same normalized by $v(\ell_i)$ |
| $v(\ell)$ | average marginal variance of a centered unit-amplitude component under the reference measure |
| $\nu$, quadrature | the uniform reference measure on $[0,1]$; Gauss–Legendre nodes used to compute its integrals |
| $\tau$, $\lambda_i$, $\alpha$ | global scale, local scale, and the scale of the half-Cauchy on $\tau$ in global–local priors |
| $\mathrm{C}^{+}$, LogNormal, HalfNormal | half-Cauchy, log-normal, half-normal distributions |
| DSP | Hvarfner et al.'s dimension-scaled lengthscale prior, $\mathrm{LogNormal}(\sqrt{2} + \tfrac12\log D, \sqrt{3})$, $\sigma_f^2 = 1$ |
| SAAS, SAASBO | the sparse axis-aligned subspace prior; BO with it and NUTS |
| OAK | orthogonal additive kernel (Lu et al. 2022) |
| NUTS, MAP | No-U-Turn Sampler (posterior sampling); maximum a posteriori (point estimate) |
| $\hat{R}$, ESS, divergences | MCMC diagnostics: chain agreement, effective sample size, failed transitions |
| EI, LogEI | expected improvement; its numerically stable logarithmic form |
| RAASP / sample-around-best | candidate generation by perturbing the best observed points |
| FANOVA, $S_u$ | functional ANOVA decomposition; Sobol index (variance share) of subset $u$ |
| $\gamma$, $\gamma^\star$ | interaction share of the objective; the share at which the additive advantage vanishes |
| simple regret $r_T$ | $f(x^\star)$ minus the best value found within budget $T$ |
| oracle-$S$ | a GP restricted to the true active coordinates; the ceiling for variable selection |
| complexity-matched control | a non-sparse cell with the same total prior complexity as the sparse cells |
| manipulation check / gate | a preregistered test that the mechanism was actually manipulated as intended, run before confirmatory experiments |

## Reading order mapped to the project

1. SAASBO, `2103.00349v2.pdf` — §3–4 for the setup and prior, App. A for inference, App. D for the experiments the design reuses.
2. OAK, `2206.09861v1.pdf` — §3 for orthogonalization, §4 for Sobol indices, App. D–E for the kernel formulas.
3. Hvarfner et al., `2402.02229v5.pdf` — §2.3 and §4 for the complexity argument, §5 for the DSP prior, Fig. 5 and Fig. 8 for the results that frame Sections 6–7 above.
4. Papenmeier et al. ICML 2025, `2502.09198v2 1.pdf` — §3 for vanishing gradients and RAASP, App. D for the structure of Lasso-DNA and MOPTA08.
5. Carvalho et al., `carvalho09a.pdf` — the horseshoe; then the R2-D2 paper if the appendix check is pursued.
6. LogEI, `2310.20708v3.pdf` — §1–3.
7. Doumont et al., `2512.00170v2 1.pdf`, and Tang & Paulson, `2604.22967v1.pdf` — for the current state of the pushback and the dense synthetic functions.
8. `direction/research-question.md` — the design itself, with the compute arithmetic and the Gate 1 script.