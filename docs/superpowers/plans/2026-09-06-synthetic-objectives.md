# Synthetic Objectives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A numpy/scipy package `synthobj` that generates deterministic sparse-additive ground-truth objectives on $[0,1]^D$ with exact variance shares, an exact optimum, recorded labels, save/load, a study-grid CLI, and a thin BoTorch adapter.

**Architecture:** One-dimensional components are GP draws (Matérn-5/2, centered under $U[0,1]$ by 64-node Gauss–Legendre quadrature, normalized by $v(\ell)$) on a fixed grid, rescaled by quadrature to exact variance shares and evaluated by cubic splines. An objective is a sum of stored components plus product interactions on disjoint pairs, optionally composed with planar rotations of active pairs; $f^\star$ is exact per 1-D component and per 2-D block. Every random choice is derived from one master seed through independent `SeedSequence` streams so the family knobs are separable.

**Tech Stack:** Python 3.11 (uv-managed venv at `.venv/`), numpy, scipy (`CubicSpline`, `leggauss`, `qmc.Sobol`, `optimize.minimize`), pytest. torch/botorch only inside `synthobj/botorch_adapter.py`.

**Spec:** `docs/superpowers/specs/2026-09-06-synthetic-objectives-brief.md` (the brief, verbatim) and `Research Context/research direction/Design-brief.md` §3. Kernel code reused verbatim from `Research Context/research direction/Kernel normalization check.py`.

## Global Constraints

- Python 3.11; runtime dependencies numpy + scipy only; torch/botorch imported only in `botorch_adapter.py`; pytest for tests.
- Fixed: $\nu = U[0,1]^D$; $\mathrm{Var}_\nu f = 1$; noise sd $0.1$; 64 Gauss–Legendre nodes; 1024-point grid; distractors exactly zero; $\mu = 0$ (all components centered).
- Active cutoff `ACTIVE_EPS = 0.02` (`families.py`), recorded as `Labels.active_eps`: every sparse family's smallest active share is ≥ 0.02 (decoupled weak 0.04, anti-aligned 0.03, interaction at γ = 0.75 gives 0.05) and dense-weak's $1/D = 0.01$ sits below it by design.
- Maximization convention: $f^\star$ is the maximum (the design brief's regret $r_t = f^\star - \max f$). The SAASBO code in this repo minimizes; negate at the call site.
- Determinism: same seed → bit-identical saved files on the same machine; `load` never re-draws.
- Draws use `eigh` with eigenvalue clipping, never jitter (jitter adds white noise of sd $\approx 10^{-4}$ that corrupts slope labels of smooth components).
- Commit after every task; each task ends with `pytest -q` green.

---

## 1. Module layout

```
synthobj/
  __init__.py          public API: SyntheticObjective, Labels, FamilySpec, make_family, load, FAMILIES, STUDY_GRID
  kernel.py            GL64 nodes/weights on [0,1]; matern52, centered, v, normalized; cached eigen-factor per (grid, ell)
  draws.py             raw 1-D draws on a joint grid: gp_draw, sinusoid_draw, draw_until_monotone, omega_for_ell
  component.py         Component: center + rescale to exact share, CubicSpline eval/derivative, slope energy, exact 1-D max
  interaction.py       Interaction: c·u_i(x_i)·u_j(x_j); block_argmax for a paired block
  rotation.py          PairRotation: Givens maps, extended grid, 2-D quadrature for Var and axis-aligned Sobol shares
  objective.py         Labels dataclass; SyntheticObjective (__call__, observe, f_star assembly, save/load)
  families.py          FamilySpec, seeding streams, FAMILIES registry, STUDY_GRID, make_family, build
  generate.py          CLI: python -m synthobj.generate → files + manifest.json
  botorch_adapter.py   SyntheticObjectiveTestFunction(SyntheticTestFunction)
tests/
  conftest.py          fixtures: small-D specs (D=20), objective cache per (variant, seed), `slow` marker
  test_kernel.py  test_draws.py  test_component.py  test_interaction.py  test_rotation.py
  test_objective.py  test_families.py  test_seeding.py  test_io.py  test_cli.py  test_botorch_adapter.py
pytest.ini             [pytest] pythonpath = .  markers = slow: >10 s
```

Repo edits: `.gitignore` += `.venv/`, `data/`. `README.md` gets a short "Synthetic objectives" section. `setup.py` is untouched (`find_packages()` already picks up `synthobj/`); no jax/numpyro needed for this package.

## 2. Data model

### 2.1 `Labels` — the ground truth the posteriors are scored against (`objective.py`)

```python
@dataclass(frozen=True)
class Labels:
    D: int
    S: tuple[int, ...]            # active coordinates, sorted
    active: np.ndarray            # (D,) bool
    s: np.ndarray                 # (D,) first-order variance share of the (pre-rotation) main effect; 0 off S
    ell: np.ndarray               # (D,) lengthscale; nan off S
    g: np.ndarray                 # (D,) realized slope share E_nu[(f_i')^2] / sum_j (main effects only); 0 off S
    pairs: tuple[tuple[int, int], ...]   # interaction pairs in coordinate space; disjoint
    s_pairs: tuple[float, ...]    # variance share per pair
    gamma: float                  # sum(s_pairs)
    f_star: float
    x_star: np.ndarray            # (D,) one argmax; inactive coordinates set to 0.5
    rotation_deg: float           # 0.0 when no rotation
    rotation_pairs: tuple[tuple[int, int], ...]
    s_axis: np.ndarray            # (D,) first-order Sobol share in x-coordinates (== s unless rotated)
    gamma_axis: float             # 1 - sum(s_axis)  (== gamma unless rotated)
    scale: float                  # global renormalization factor (1.0 unless rotated)
    generator: str                # "matern" | "sinusoid"
    monotone: bool
    family: str
    seed: int
    noise_sd: float
    active_eps: float             # global active cutoff for scoring posteriors (ACTIVE_EPS = 0.02)
```

### 2.2 `FamilySpec` (`families.py`)

```python
@dataclass(frozen=True)
class FamilySpec:
    name: str
    D: int = 100
    n_active: int | None = None                  # None => all D coordinates (dense)
    shares: tuple[float, ...] | None = None      # per active rank, sum = 1 - gamma; None => equal
    ells: float | tuple[float, ...] = 0.5        # scalar broadcast or per active rank
    n_pairs: int = 0                             # interaction pairs (disjoint, inside S)
    gamma: float = 0.0                           # split equally over pairs
    theta_deg: float | None = None               # None => no rotation; 0.0 => identity rotation on the extended grid (sweep control)
    generator: str = "matern"                    # "matern" | "sinusoid"
    monotone: bool = False
    noise_sd: float = 0.1
    grid_n: int = 1024
```

### 2.3 Saved function: `<stem>.npz` + `<stem>.json`

| file | key | shape / content |
|---|---|---|
| npz | `grid` | (G,) sorted, deduped knots: uniform grid ∪ GL nodes; G = 1088 (or 1532 on the extended grid) |
| npz | `main_values` | (n_active, G) centered values with $\mathrm{Var}_\nu = s_k$ exactly |
| npz | `inter_values` | (2·n_pairs, G) centered unit-variance $u$ draws, rows `2p`, `2p+1` for pair p |
| npz | `active` | (n_active,) int, rank order (row k of `main_values` belongs to coordinate `active[k]`) |
| npz | `pairs`, `c_pairs` | (n_pairs, 2) int coordinates; (n_pairs,) $c = \sqrt{s_{ij}}$ |
| npz | `s`, `ell`, `sign` | (n_active,) per rank (sign = ±1 monotone direction, else +1) |
| npz | `x_star` | (D,) |
| json | `version` | 1 |
| json | `spec` | `asdict(FamilySpec)` |
| json | `labels` | `Labels` with arrays as lists, tuples as lists |
| json | `rotation` | `{"theta_deg": float, "pairs": [[i, j], ...]}` or null |
| json | `grid_lo`, `grid_hi`, `grid_n` | grid definition (for the eigen-factor cache key on rebuild) |

`load(stem)` rebuilds `Component`/`Interaction` objects from the arrays and passes the stored `Labels` through; no rng, no `eigh`, no $f^\star$ recomputation. JSON is written with `sort_keys=True, indent=1`; floats via `repr` (round-trip exact).

## 3. Signatures (docstrings condensed to one line)

```python
# kernel.py
GL_NODES: np.ndarray; GL_WEIGHTS: np.ndarray      # 64-node Gauss–Legendre on [0,1]; weights sum to 1
def matern52(r: np.ndarray) -> np.ndarray:        "Unit-variance Matérn-5/2 in scaled distance r (check script, verbatim)."
def centered(X, Y, ell: float) -> np.ndarray:     "Projection-form centered kernel under U[0,1] (check script, verbatim)."
def v(ell: float) -> float:                       "Average marginal variance of the centered unit-amplitude component."
def normalized(X, Y, ell: float) -> np.ndarray:   "centered / v(ell); E_nu[k(x,x)] = 1."
def make_grid(lo: float, hi: float, n: int) -> np.ndarray:  "np.linspace(lo, hi, n)."
def joint_points(grid) -> np.ndarray:             "np.concatenate([grid, GL_NODES]) — draw points; nodes last."
def eigen_factor(lo: float, hi: float, n: int, ell: float) -> np.ndarray:
    "F with F Fᵀ = normalized(Z, Z, ell), Z = joint_points(make_grid(lo, hi, n)); eigh, eigenvalues < 1e-12·λmax → 0; cached in a module dict keyed (lo, hi, n, ell)."
def nu_mean(values_at_nodes) -> float; def nu_var(values_at_nodes) -> float   "quadrature under U[0,1]"

# draws.py
def omega_for_ell(ell: float) -> float:           "sqrt(5 / (3 ell² v(ell))) / (2π): cycles/unit matching the Matérn slope energy per unit variance."
def gp_draw(F: np.ndarray, rng) -> np.ndarray:    "F @ rng.standard_normal(F.shape[1]) on the joint points."
def sinusoid_draw(Z: np.ndarray, ell: float, rng, n_terms: int = 3) -> np.ndarray:
    "Σ_k w_k sin(2π ω_k x + φ_k): w~N(0,1), ω~LogUniform(ω(ell)/1.25, 1.25 ω(ell)), φ~U[0,2π); values at Z."
def draw_until_monotone(draw_fn, n_grid: int, rng, max_tries: int = 200) -> np.ndarray:
    "Repeat draw_fn(rng) until values[:n_grid] are non-decreasing or non-increasing; RuntimeError after max_tries."

# component.py
@dataclass(frozen=True)
class Component:
    coord: int; ell: float; share: float; grid: np.ndarray; values: np.ndarray   # values on sorted deduped knots
    spline: CubicSpline = field(init=False, repr=False, compare=False)          # not-a-knot, extrapolate=False
    @classmethod
    def from_raw(cls, coord, ell, share, grid, raw_joint_values, sign: float = 1.0) -> "Component":
        "Center by quadrature on the trailing 64 node values, rescale to Var_nu = share exactly, apply sign, merge knots (dedupe < 1e-12), build spline."
    def __call__(self, x) -> np.ndarray;  def derivative(self, x) -> np.ndarray
    def nu_mean(self) -> float;  def nu_var(self) -> float                        "quadrature of spline at GL nodes"
    def slope_energy(self) -> float:      "E_nu[(f')²] = Σ_q w_q f'(t_q)²."
    def argmax(self, lo: float = 0.0, hi: float = 1.0) -> tuple[float, float]:
        "Exact: candidates = real roots of spline.derivative() in [lo, hi] ∪ {lo, hi}; returns (x*, f*)."

# interaction.py
@dataclass(frozen=True)
class Interaction:
    i: int; j: int; c: float; u_i: Component; u_j: Component    # u_* built with share=1
    def __call__(self, xi, xj) -> np.ndarray:                     "c · u_i(xi) · u_j(xj)"
    def nu_var(self) -> float:                                    "c² · u_i.nu_var() · u_j.nu_var()"
def block_value_and_grad(f_i, f_j, inter, xy: np.ndarray) -> tuple[float, np.ndarray]:
    "φ(x,y) = f_i(x)+f_j(y)+c u_i(x) u_j(y) and its analytic gradient from spline derivatives."
def block_argmax(f_i, f_j, inter, top_k: int = 20) -> tuple[np.ndarray, float]:
    "Max of φ on [0,1]²: evaluate on the 1024×1024 knot grid, L-BFGS-B (ftol 1e-15, gtol 1e-12) from the top_k cells; returns ((x*,y*), φ*)."

# rotation.py
EXT_LO = 0.5 - np.sqrt(2) / 2 - 0.01;  EXT_HI = 1.0 - EXT_LO       # [-0.217, 1.217] covers the rotated cube for any θ
@dataclass(frozen=True)
class PairRotation:
    theta_deg: float; pairs: tuple[tuple[int, int], ...]
    def forward(self, X: np.ndarray) -> np.ndarray:  "z = ½ + R_θ (x − ½) on each pair; other coordinates unchanged; returns a copy."
    def R(self) -> np.ndarray:                       "2×2 rotation matrix"
def rotated_block_stats(f_a, f_b, rot) -> tuple[float, float, float]:
    "64×64 GL quadrature of φ(x_a,x_b) = f_a(z_a) + f_b(z_b): (Var φ, Var_{x_a} E_{x_b} φ, Var_{x_b} E_{x_a} φ)."
def rotated_block_argmax(f_a, f_b, rot, top_k: int = 20) -> tuple[np.ndarray, float]:
    "As block_argmax but in x-space with the chain rule through R; bounds [0,1]²."

# objective.py
class SyntheticObjective:
    def __init__(self, D, components, interactions, rotation, spec, seed, labels=None, allow_overlap=False):
        "Assemble; if labels is None compute f_star/x_star, g, s_axis, gamma_axis, scale and build Labels."
    D: int; noise_sd: float; f_star: float; labels: Labels; spec: FamilySpec
    def __call__(self, X: np.ndarray) -> np.ndarray:  "Noise-free f at X of shape (n, D) or (D,); ValueError if outside [0,1] ± 1e-9 or non-finite."
    def observe(self, X, rng: np.random.Generator) -> np.ndarray:  "self(X) + noise_sd · rng.standard_normal(n)"
    def save(self, stem) -> tuple[Path, Path];  @classmethod def load(cls, stem) -> "SyntheticObjective"
    def component_variances(self) -> np.ndarray:  "per-block Var_nu by quadrature (1-D for main effects, 2-D for rotated pairs, c² for interactions), times scale²"

# families.py
class Streams(NamedTuple): select: SeedSequence; main: SeedSequence; inter: SeedSequence
def streams(seed: int) -> Streams:               "SeedSequence(seed).spawn(3)"
def noise_rng(seed: int, run: int = 0) -> np.random.Generator:  "default_rng(SeedSequence([seed, 0x4E4F, run])) — for paired BO runs"
def select_active(D, n_active, ss) -> np.ndarray: "default_rng(ss).permutation(D)[:n_active] in rank order (nested across n_active)."
def select_pairs(n_active, n_pairs, ss) -> tuple[tuple[int, int], ...]:
    "Disjoint rank pairs from default_rng(ss).permutation(n_active); ValueError if 2·n_pairs > n_active."
def build(spec: FamilySpec, seed: int) -> SyntheticObjective:
    "Streams → S, per-rank component draws (one child stream each), pair selection, per-pair u draws, rotation → SyntheticObjective."
FAMILIES: dict[str, Callable[..., FamilySpec]]    # aligned, decoupled, anti_aligned, interaction, rotated, dense_weak
STUDY_GRID: list[tuple[str, str, dict]]           # (variant, family, overrides) — 14 variants, §7
def make_family(name: str, seed: int, D: int = 100, **overrides) -> SyntheticObjective:
    "build(FAMILIES[name](D=D, **overrides), seed); unknown name or override → KeyError/TypeError."

# botorch_adapter.py
class SyntheticObjectiveTestFunction(SyntheticTestFunction):
    def __init__(self, obj: SyntheticObjective, noise_std: float | None = None, negate: bool = False):
        "Set dim, _bounds=[(0,1)]*D, continuous_inds, _optimal_value=obj.f_star, _optimizers=[tuple(x_star)]; then super().__init__(noise_std, negate)."
    def _evaluate_true(self, X: Tensor) -> Tensor:  "X (..., D) → obj(X.reshape(-1, D).double().cpu().numpy()) reshaped to X.shape[:-1], float64."
```

Family builders (all return `FamilySpec`; shares sum to $1-\gamma$):

| name | n_active | shares | ells | other |
|---|---|---|---|---|
| `aligned` | 3 (override 10) | equal | 0.5 | `generator` override for the sinusoid variant |
| `decoupled` | 8 | 2 × 0.21, 2 × 0.21, 2 × 0.04, 2 × 0.04 (sum 1) | 1.5, 0.08, 1.5, 0.08 | cell order: strong-smooth, strong-rough, weak-smooth, weak-rough |
| `anti_aligned` | 6 | (0.35, 0.25, 0.18, 0.12, 0.07, 0.03) | (3, 0.5, 0.27, 0.18, 0.11, 0.06) | expected slope shares increase as variance shares decrease |
| `interaction` | 5 | $(1-\gamma)/5$ each | 0.5 | `n_pairs=2`, `gamma` override |
| `rotated` | 10 | equal | 0.5 | `theta_deg` override (0.0 included); rotation pairs = `select_pairs(10, 5, ·)`; components on the extended grid |
| `dense_weak` | D | $1/D$ | 3 | `monotone=True`, random sign per coordinate |

## 4. Design decisions

**D1 — Exact $f^\star$.** Main effects: `Component.argmax` is exact — a cubic spline's derivative is piecewise quadratic, `spline.derivative().roots()` returns every real critical point, and the max over those plus the endpoints is the global max to floating point. Interactions: `select_pairs` returns a matching (disjoint pairs inside $S$), so $f^\star = \sum_{\text{pairs}} \max \varphi_p + \sum_{\text{unpaired}} \max f_i$ with $\varphi_p(x,y) = f_i(x)+f_j(y)+c\,u_i(x)u_j(y)$. `block_argmax` evaluates $\varphi$ on the 1024×1024 knot grid (1M evaluations, ~0.1 s), then runs L-BFGS-B with the analytic gradient from the top 20 cells; the grid max is within $O(h^2 |\nabla^2\varphi|) \approx 10^{-5}$ of the true max and refinement takes it below $10^{-10}$. With $|S| = 5$ a matching has at most 2 pairs, so the sweep uses `n_pairs=2` and $\gamma$ split 50/50 (flagged default). Overlapping pairs: `SyntheticObjective(..., allow_overlap=True)` computes $f^\star$ by joint grid search plus L-BFGS-B over each connected component of the pair graph with grid $64^m$ for $m \le 4$ and raises `NotImplementedError` above that; without the flag overlapping pairs raise `ValueError`. No family uses overlap. $x^\star$ is recorded with inactive coordinates at 0.5.

**D2 — Rotation.** Rotate disjoint pairs of active coordinates by a Givens rotation of angle $\theta$ about the cube centre: $z_a = \tfrac12 + \cos\theta\,(x_a-\tfrac12) - \sin\theta\,(x_b-\tfrac12)$, $z_b = \tfrac12 + \sin\theta\,(x_a-\tfrac12) + \cos\theta\,(x_b-\tfrac12)$, and evaluate $f_a(z_a)+f_b(z_b)$. Rotated components are drawn on the extended grid $[-0.217, 1.217]$ at the same spacing $1/1023$ (1468 points ∪ GL nodes) so $z$ never leaves the components' domain for any $x$ in the cube and any $\theta$: no clipping, no contraction, lengthscales exact — $f_{\text{rot}}$ is a genuine rotation of an additive function on a larger box. Centering and exact-share rescaling stay under $\nu = U[0,1]$ on the pre-rotation coordinate. $\mathrm{Var}_\nu f_{\text{rot}} \ne 1$ in general, so `rotated_block_stats` computes $\mathrm{Var}\,\varphi_p$ per pair by 64×64 quadrature (pairs are independent under $\nu$, so variances add) and the objective is multiplied by `scale` $= 1/\sqrt{\mathrm{Var}}$; `FamilySpec.theta_deg=None` means no rotation, `0.0` the identity rotation on the extended grid. `Labels` record `scale`, the pre-rotation shares `s`, the axis-aligned first-order Sobol shares `s_axis` $= \mathrm{Var}_{x_a}\mathbb E_{x_b}\varphi_p \cdot$ `scale`$^2$, and `gamma_axis` $= 1-\sum$ `s_axis` — the interaction share a first-order additive model actually faces. $f^\star$ per pair by `rotated_block_argmax` in $x$-space. $\theta = 0$ is the sweep's own control: it also uses the extended grid, so it is not bit-identical to `aligned10` at the same seed; the test checks that `forward` is the identity, `scale` $= 1$ and `s_axis = s`. Rejected: contracting the rotated square into the cube (multiplies effective lengthscales by $|\cos\theta|+|\sin\theta|$, confounding rotation with smoothness). Default $|S| = 10$, five rotated pairs (confirmed in review, Q2).

**D3 — Storage: grid + cubic spline (recommended).** Measured in a scratch run with the check script's kernel (4 unit-variance draws at $\ell = 0.08$, reference = the same joint draw on a 4093-point grid, spline from every 4th point ∪ GL nodes): max $|$error$|$ 4.4e-5 to 8.6e-5, RMS 5e-6 to 1.2e-5; derivative error at GL nodes max 0.26 to 0.49 against an RMS slope of 16 to 32 (1 to 2 % pointwise); relative error of the slope energy $\le 1.0\times10^{-3}$. Acceptable: value error is $10^{-3}$ of the noise sd, and $g_i$ labels are accurate to 0.1 %. Test tolerances: max $|$err$| < 2\times10^{-4}$, RMS $< 3\times10^{-5}$, relative slope-energy error $< 3\times10^{-3}$. Re-draw-free test design: one joint draw on grid ∪ nodes ∪ 400 fine points in $[0.30, 0.40]$ (1488 points, ~2 s `eigh`); build the spline from grid ∪ nodes; measure on the fine patch. Random-Fourier-feature paths rejected: $\sim 10^3$ features per component for comparable accuracy at $\ell = 0.08$, slower evaluation, and no exact quadrature centering. Draw method: `eigh` of $\bar k(Z,Z)$ with eigenvalues below $10^{-12}\lambda_{\max}$ clipped to zero (0.8 s at $\ell=0.5$, 1.7 s at $\ell = 3$ on this machine, cached per (grid, $\ell$); at $\ell = 3$ only 67 of 1088 eigenvalues survive), which is exact PSD sampling without jitter noise. The check script's `multivariate_normal(..., K + 1e-8 I)` must not be reused for the same reason. Knots closer than $10^{-12}$ (a GL node coinciding with a grid point) are deduped before `CubicSpline`.

**D4 — Monotone dense-weak: rejection sampling (recommended).** Measured acceptance of "monotone on the grid" for normalized centered draws: 0.66 at $\ell = 3$ (4000 draws; 0.41 at $\ell = 1.5$, 0.79 at $\ell = 5$), i.e. 1.5 draws per component and ~150 matrix-vector products for $D = 100$ after one cached `eigh` (<1 s in total). Monotonicity is checked on the stored knots; at spacing $10^{-3}$ and $\ell = 3$ a path cannot reverse between knots, and the test confirms the spline derivative keeps one sign at $10^4$ points. Sign is random per coordinate from the component's own stream, so the optimum sits at a random corner (confirmed in review, Q4); centering and rescaling are affine with positive scale and preserve monotonicity. Parametric monotone shapes rejected: they leave the Matérn family, which is the sinusoid generator's job, not this family's.

**D5 — Sinusoid generator.** A unit-variance sinusoid at $\omega$ cycles per unit has $\mathbb E[(u')^2] = (2\pi\omega)^2$; the normalized Matérn component has $\mathbb E[(f')^2]/\mathrm{Var} = 5/(3\ell^2 v(\ell))$. Matching gives $\omega(\ell) = \sqrt{5/(3\ell^2 v(\ell))}/(2\pi)$, computed: $\ell = 0.08 \to 2.83$, $0.1 \to 2.32$, $0.2 \to 1.32$, $0.5 \to 0.77$, $1.5 \to 0.59$, $3 \to 0.56$, $\ell\to\infty \to \sqrt{12}/2\pi = 0.55$ (the linear limit). Draw $\sum_{k=1}^{3} w_k \sin(2\pi\omega_k x+\phi_k)$ with $w_k \sim N(0,1)$, $\omega_k \sim \mathrm{LogUniform}(\omega(\ell)/1.25,\ 1.25\,\omega(\ell))$, $\phi_k \sim U[0,2\pi)$, then the shared centering/rescaling. The match holds in expectation only; the recorded $g_i$ is realized, so no exactness is needed. For $\ell \ge 0.5$ the band is below one cycle per unit, so those components are smooth arcs — the intended "same slope share, different shape" contrast.

**D6 — Seeding.** `SeedSequence(seed).spawn(3)` → `select`, `main`, `inter`. `select`: `permutation(D)[:n_active]` in rank order, so $S$ is nested across `n_active` and independent of shares, $\ell$, $\gamma$, $\theta$ and generator. `main.spawn(n_active)`: one child per rank, so changing $\ell_k$ or $s_k$ leaves every other component's Gaussian vector untouched, and rejection sampling consumes only its own stream. `inter.spawn(2)`: pair selection (a permutation of ranks cut into disjoint pairs, reused as the rotation pairs) and `spawn(n_pairs)` with two $u$ draws each; $c_{ij} = \sqrt{s_{ij}}$ is deterministic. Consequence: across the interaction sweep at one seed, $S$, the pairs and every shape are identical and only the shares move. Noise is not part of the objective: `observe(X, rng)` takes the caller's generator; `noise_rng(seed, run)` is provided for paired BO runs.

## 5. Tasks

Each task: write the failing tests first, run them (expect failure), implement, run `pytest -q`, commit. Test bodies are one-line assertions here; executors expand them.

### Task 0: Environment and skeleton
**Files:** `pytest.ini`, `.gitignore`, `synthobj/__init__.py`, `tests/conftest.py`
- [ ] `uv venv --python 3.11 .venv && uv pip install -p .venv/bin/python numpy scipy pytest`
- [ ] `pytest.ini` with `pythonpath = .` and the `slow` marker; add `.venv/` and `data/` to `.gitignore`
- [ ] `tests/conftest.py`: fixture `small_objective(variant, seed)` cached per session at `D=20`
- [ ] `.venv/bin/pytest -q` → "no tests ran"; commit `chore: scaffold synthobj package and test env`

### Task 1: `kernel.py`
**Files:** `synthobj/kernel.py`, `tests/test_kernel.py`
- [ ] Tests: weights sum to 1 and nodes in (0,1); `v(0.1)≈0.7815`, `v(1)≈0.107`, `v(10)≈0.00138` (rel 1e-3) and `v(30)` within 5 % of `5/(36·900)`; `Σ_q w_q centered(x, t_q) = 0` to 1e-12 for x ∈ {0, 0.3, 1}; `Σ_q w_q normalized(t_q, t_q) = 1` to 1e-10; `F Fᵀ` matches `normalized(Z,Z,ℓ)` to Frobenius rel 1e-8 at ℓ ∈ {0.08, 0.5, 3}; second `eigen_factor` call returns the same object.
- [ ] Implement (copy `matern52`, `centered`, `v` verbatim from the check script; add the rest). Commit `feat(synthobj): centered normalized Matérn kernel and cached eigen factor`.

### Task 2: `draws.py`
**Files:** `synthobj/draws.py`, `tests/test_draws.py`
- [ ] Tests: `gp_draw` shape; mean over 2000 draws of `nu_var(values[-64:])` within 5 % of 1 at ℓ = 0.5; `omega_for_ell(0.5)` ≈ 0.774 and `omega_for_ell(0.08)` ≈ 2.83 (rel 1e-2); sinusoid slope-energy/variance ratio averaged over 200 draws within factor 1.5 of `5/(3ℓ²v)` at ℓ ∈ {0.08, 0.5, 3}; `draw_until_monotone` returns monotone values at ℓ = 3 and raises `RuntimeError` with a draw function that returns `sin(20πx)`.
- [ ] Implement. Commit `feat(synthobj): GP, sinusoid and monotone 1-D draws`.

### Task 3: `component.py`
**Files:** `synthobj/component.py`, `tests/test_component.py`
- [ ] Tests: for ℓ ∈ {0.08, 0.5, 1.5, 3} × share ∈ {0.02, 0.25, 1}: `nu_mean` = 0 to 1e-12 and `nu_var` = share to 1e-8; `sign=-1` flips values; spline error vs the fine patch at ℓ = 0.08 (D3 tolerances) and at ℓ = 0.06, now the roughest lengthscale (max |err| < 5e-4, RMS < 1e-4, slope-energy rel err < 5e-3, extrapolated from the ℓ = 0.08 measurement by (h/ℓ) scaling; mark `slow`); `argmax`: value ≥ spline on 10⁵ grid points and ≥ f at 10⁶ uniform points, attained to 1e-12; `slope_energy` of the exact linear component `√12(x−½)` = 12 to 1e-6; extended-domain component evaluates at z = −0.2 and 1.2 without error.
- [ ] Implement. Commit `feat(synthobj): exact-share spline components with exact 1-D argmax`.

### Task 4: `interaction.py`
**Files:** `synthobj/interaction.py`, `tests/test_interaction.py`
- [ ] Tests: `nu_var` = s_ij to 1e-8 for s_ij ∈ {0.05, 0.375}; `Σ_q w_q h(t, t_q) = 0` to 1e-10 at every node t and symmetric in (i, j); `ΣΣ w_p w_q h(t_p,t_q) f_i(t_p) = 0` to 1e-10; `block_argmax` ≥ φ at 10⁶ random pairs and attained to 1e-6; gradient check of `block_value_and_grad` by central differences (1e-6).
- [ ] Implement. Commit `feat(synthobj): product interactions with exact block argmax`.

### Task 5: `objective.py` (assembly, no rotation, no I/O)
**Files:** `synthobj/objective.py`, `tests/test_objective.py`
- [ ] Tests on a hand-built objective (D = 20, 4 components ℓ ∈ {0.08, 0.5, 1.5, 3}, 1 pair, γ = 0.25): `Σ s + Σ s_pairs = 1` to 1e-12; `component_variances().sum()` = 1 to 1e-8 and QMC variance (`qmc.Sobol(D, scramble=True, seed=0).random(2**17)`) within 1 %; QMC mean within 1e-2 of 0; `f_star ≥ f` at 10⁶ uniform points and `f(x_star) = f_star` to 1e-6; perturbing an inactive coordinate leaves f bit-identical; `observe` noise sd within 2 % of 0.1 over 10⁵ draws; `ValueError` for X = 1.1 and for NaN; `(D,)` input returns a scalar array; `labels.g[S].sum()` = 1 and `g` = 0 off S.
- [ ] Implement `Labels`, `SyntheticObjective.__init__/__call__/observe/component_variances/_compute_labels`. Commit `feat(synthobj): SyntheticObjective assembly with exact f_star and labels`.

### Task 6: `rotation.py` + objective integration
**Files:** `synthobj/rotation.py`, `synthobj/objective.py`, `tests/test_rotation.py`
- [ ] Tests: `forward` at θ = 0 is the identity; at θ ∈ {15, 45}, `forward` of 10⁶ uniform points stays inside `[EXT_LO, EXT_HI]`; rotated objective (D = 20, 4 active → 2 pairs, ℓ = 0.5): QMC variance within 1 % of 1 and `scale ≠ 1`; θ = 0 → `scale = 1`, `s_axis = s`, `gamma_axis = 0`; `s_axis.sum() + gamma_axis = 1` to 1e-8; `f_star ≥ f` at 10⁶ points and attained to 1e-6; `x_star` inside the cube; gradient check of the chain rule.
- [ ] Implement; extend `SyntheticObjective` to apply `forward` and `scale`. Commit `feat(synthobj): planar rotation of active pairs with renormalization and axis shares`.

### Task 7: `families.py`
**Files:** `synthobj/families.py`, `synthobj/__init__.py`, `tests/test_families.py`, `tests/test_seeding.py`
- [ ] `test_families.py`: every `STUDY_GRID` variant builds at D = 20 (dense_weak at D = 20 has 20 components) with the |S|, shares and ℓ of the table in §3; shares sum to 1 − γ; decoupled at D = 100, seeds 0–9: argmax of `g` lies in the strong-rough cell and argmin in the weak-smooth cell for every seed (expected cell slope energies 66.6 / 12.7 / 2.9 / 0.55 for strong-rough / weak-rough / strong-smooth / weak-smooth; the 3-per-cell scratch run passed 10/10; mark `slow`); anti_aligned at D = 100, seeds 0–9: Spearman(`g[S]`, `s[S]`) ≤ −0.6 for every seed and median over seeds ≤ −0.8 (realized `g`; mark `slow`); every sparse variant has min active share ≥ `ACTIVE_EPS` = 0.02 and dense_weak's 1/D is below it; dense_weak: all D active and every component monotone (derivative sign constant at 10⁴ points); rotated θ = 0 vs θ = 45 share `S` and pairs; `make_family("nope", 0)` → `KeyError`; unknown override → `TypeError`.
- [ ] `test_seeding.py`: same seed, γ ∈ {0, 0.5} → identical `S`, pairs, and `main_values` rows equal up to the share scaling (allclose 1e-12 after dividing by √s); changing one coordinate's ℓ leaves the other rows bit-identical; `n_active=3` ⊂ `n_active=10` at the same seed; seeds 0 and 1 give different `S`; `noise_rng(0, 0)` ≠ `noise_rng(0, 1)`.
- [ ] Implement `FamilySpec`, streams, builders, `STUDY_GRID`, `make_family`, `build`; export the public API. Commit `feat(synthobj): family registry, seeding streams and study grid`.

### Task 8: save/load
**Files:** `synthobj/objective.py`, `tests/test_io.py`
- [ ] Tests: build twice at the same seed, save to two stems → `.npz` and `.json` bytes identical; `load` returns equal `Labels` (arrays `array_equal`, scalars `==`) and `f` equal to 1e-15 at 10⁴ points for an interaction and a rotated variant; loading does not call `eigen_factor` (monkeypatch it to raise); `version` present; different seed → different `S`.
- [ ] Implement `save`/`load`. Commit `feat(synthobj): npz+json save/load without re-drawing`.

### Task 9: `generate.py` CLI
**Files:** `synthobj/generate.py`, `tests/test_cli.py`
- [ ] Tests (via `subprocess` and via the `main(argv)` function): `--out tmp --D 20 --seeds 0-1 --families aligned3,interaction_g0.25` writes 4 stems plus `manifest.json` whose sha256 entries match the files and whose `f_star` equals the loaded labels; `--dry-run` prints 140 rows for the default grid and writes nothing; rerun skips existing stems unless `--overwrite`; bad family name exits 2 with a message.
- [ ] Implement (argparse; `families` accepts `all` or a comma list of variants; seeds accept `a-b` or a comma list). Commit `feat(synthobj): study-grid generator CLI with manifest`.

### Task 10: `botorch_adapter.py`
**Files:** `synthobj/botorch_adapter.py`, `tests/test_botorch_adapter.py`
- [ ] `uv pip install -p .venv/bin/python "botorch>=0.12"`; tests `pytest.importorskip("botorch")`: input `(3, 4, D)` → output `(3, 4)` float64; values equal to numpy to 1e-12; `optimal_value == f_star`; `negate=True` flips sign; `noise_std=0.1` makes `forward` differ from `evaluate_true`; `bounds` is a `(2, D)` tensor of zeros and ones.
- [ ] Implement. Commit `feat(synthobj): BoTorch SyntheticTestFunction adapter`.

### Task 11: docs and full run
- [ ] README section (what the package is, `make_family` example, CLI line, maximization convention); docstring pass; `pytest -q -m "not slow"` then `pytest -q`; time `python -m synthobj.generate --out data/objectives` (expect < 5 min: ~11 cached `eigh` calls plus ~25 block maximizations per seed set). Commit `docs: synthetic objectives usage`.

## 6. Tests mapped to files

| requirement (brief) | file | test |
|---|---|---|
| $\mathrm{Var}_\nu f_i = s_i$ to 1e-8; $\sum s_i + \sum s_{ij} = 1$ | test_component, test_objective, test_families | `test_component_variance_exact`, `test_shares_sum_to_one` |
| $\mathbb E[h_{ij}\mid x_i] = 0$, $\mathbb E[h_{ij} f_i] = 0$ | test_interaction | `test_conditional_mean_zero`, `test_orthogonal_to_main_effects` |
| quadrature vs Sobol MC total variance within 1 % | test_objective, test_rotation | `test_total_variance_qmc` |
| $f^\star \ge f$ at $10^6$ points; attained to 1e-6 | test_component, test_interaction, test_objective, test_rotation | `test_f_star_dominates`, `test_f_star_attained` |
| spline vs finer reference at ℓ = 0.08 | test_component | `test_spline_error_rough` (`slow`) |
| determinism; different seed → different S | test_io, test_seeding | `test_bit_identical_files`, `test_seed_changes_S` |
| labels round-trip | test_io | `test_labels_roundtrip` |
| decoupled slope-share ordering | test_families | `test_decoupled_slope_ordering` (`slow`) |
| kernel/quadrature correctness | test_kernel | `test_v_table`, `test_centered_integrates_to_zero`, `test_eigen_factor_reconstructs` |
| generator sanity | test_draws | `test_sinusoid_slope_matches_matern`, `test_monotone_rejection` |
| seeding separability | test_seeding | `test_gamma_sweep_shares_shapes`, `test_nested_active_sets` |
| CLI + manifest | test_cli | `test_generate_small_grid`, `test_dry_run` |
| BoTorch adapter | test_botorch_adapter | `test_shapes_dtype_values`, `test_optimal_value` |

## 7. CLI and study grid

```
python -m synthobj.generate --out data/objectives [--D 100] [--seeds 0-9] [--families all] [--dry-run] [--overwrite]
```

Output: `data/objectives/<variant>/seed{seed:02d}_D{D}.npz|.json` and `data/objectives/manifest.json` = `{"version": 1, "created": iso, "D": …, "seeds": […], "entries": [{"variant", "family", "overrides", "seed", "D", "npz", "json", "sha256_npz", "sha256_json", "f_star", "gamma", "n_active"}]}`.

`STUDY_GRID` (14 variants × 10 seeds = 140 files): `aligned3`, `aligned10`, `decoupled`, `anti_aligned`, `interaction_g0.00`, `interaction_g0.10`, `interaction_g0.25`, `interaction_g0.50`, `interaction_g0.75`, `rotated_t0`, `rotated_t15`, `rotated_t45`, `dense_weak`, `aligned10_sin` (the one sinusoid out-of-family check). Embedded Hartmann-6/Levy-4 are deferred to a follow-up plan (review decision Q3).

## 8. Effort

| task | hours |
|---|---|
| 0 env + skeleton | 1 |
| 1 kernel | 2.5 |
| 2 draws | 2.5 |
| 3 component | 3 |
| 4 interaction | 3 |
| 5 objective assembly | 3 |
| 6 rotation | 4 |
| 7 families + seeding | 3 |
| 8 save/load | 2 |
| 9 CLI | 2 |
| 10 BoTorch adapter | 1.5 |
| 11 docs + full run | 1.5 |
| **total** | **≈ 29 h (about four working days)** |

## 9. Review decisions (2026-09-06) and remaining questions

Resolved by Quan:

1. **Q1a decoupled.** 2 coordinates per cell, strong $s = 0.21$, weak $s = 0.04$, $\ell = 1.5 / 0.08$; $4(0.21) + 4(0.04) = 1$. Rationale: the normalized 12.5:1 version would put a weak-rough component at 1.2 % of variance against 1 % noise, undetectable for every cell at $n = 100$, so the false-positive prediction could not show. Global active cutoff $\varepsilon = 0.02$ (`ACTIVE_EPS`), half the smallest active share; dense-weak sits below it by design. Note: anti-aligned's smallest share 0.03 is $1.5\varepsilon$, not $2\varepsilon$, still above the cutoff.
2. **Q1b anti-aligned.** Reversing set $\ell = (3, 0.5, 0.27, 0.18, 0.11, 0.06)$ adopted. Test on realized $g$: Spearman$(g, s) \le -0.6$ per seed and median over the 10 seeds $\le -0.8$. $\ell = 0.06$ is about 60 knots per lengthscale on the 1024 grid.
3. **Q2 rotated.** 10 active coordinates, 5 rotated pairs, extended-domain construction with renormalization (D2).
4. **Q3 embedded Hartmann-6 / Levy-4.** Deferred to a follow-up plan.

5. **Q4 dense-weak sign.** Random monotone direction per coordinate, drawn from the component's own stream (D4).
6. **Q5 placement.** Top-level package `synthobj/` in this SAASBO fork with `tests/` beside it; `setup.py` untouched.

Status: plan approved on 2026-09-06; execution not yet triggered (Quan will start it later, subagent-driven or inline).

Flagged defaults not raised as questions: maximization convention; interaction sweep uses 2 disjoint pairs for $|S| = 5$; `rotated_t0` is not bit-identical to `aligned10`; BoTorch adapter subclasses `SyntheticTestFunction` (needs `botorch>=0.12` for `_evaluate_true`).
