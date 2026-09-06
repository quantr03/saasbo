# Synthetic objectives — brief (verbatim from Quan, 2026-09-06)

Design brief this implements: `Research Context/research direction/Design-brief.md` §3.
Kernel code to reuse: `Research Context/research direction/Kernel normalization check.py`
(`centered`, `v`, `matern52`, the 64-node Gauss–Legendre grid).

## Context

The thesis compares sparsity priors on normalized additive amplitudes vs on inverse squared lengthscales (SAASBO) in a 2×2 design (additive/product kernel × amplitude/lengthscale sparsity). The synthetic objectives are the ground truth for (a) an offline identification study and (b) in-loop BO with BoTorch.

## What the module must produce

A deterministic function `f: [0,1]^D → R` with known structure:

    f(x) = Σ_{i∈S} f_i(x_i) + Σ_{(i,j)∈I} h_ij(x_i, x_j),   Var_ν f = 1,   ν = U[0,1]^D

Components:
- For each i ∈ S, draw g_i ~ GP(0, k̄_{ℓ_i}) where k̄_ℓ is the Matérn-5/2 kernel centered under U[0,1] and divided by v(ℓ) (so E Var_ν = 1). Draw on a fixed 1-D grid (1024 points ∪ the 64 GL nodes), center under ν by quadrature, rescale to an EXACT variance share s_i by quadrature. Store grid values; evaluate by cubic spline.
- Interactions: h_ij = c_ij · u_i(x_i) · u_j(x_j) with u_i, u_j fresh centered unit-variance 1-D draws (same generator), c_ij = sqrt(s_ij). Then h_ij is orthogonal to all main effects under ν and Var_ν h_ij = s_ij exactly.
- Constraint Σ s_i + Σ s_ij = 1, so γ = Σ s_ij is the interaction share.
- Noisy observations y = f(x) + N(0, 0.1²) via a separate method that takes an rng.

Parameters set per function: D (default 100), S (size and members — random subset per seed), s_i, ℓ_i, I and s_ij, an optional rotation of the active subspace (θ ∈ {0°,15°,45°}), generator type (Matérn GP default; rescaled random sinusoids as an out-of-family alternative), seed. Fixed: ν, Var f = 1, noise 0.1, 64 GL nodes, 1024-point grid, distractors exactly zero.

Families (each a named parameter setting; 10 seeds each):
- aligned: |S| ∈ {3,10}, s_i = 1/|S|, ℓ_i = 0.5
- decoupled: 3 coordinates in each cell of {s=0.25, 0.02} × {ℓ=1.5, 0.08}
- anti-aligned: |S| = 6, s = (0.35,0.25,0.18,0.12,0.07,0.03), ℓ = (2,1.2,0.7,0.4,0.2,0.1)
- interaction sweep: aligned |S| = 5, ℓ = 0.5, pairs inside S, γ ∈ {0,0.1,0.25,0.5,0.75} split equally over pairs
- rotated: aligned family composed with a rotation of the active subspace
- dense-weak: all D coordinates, s_i = 1/D, ℓ = 3, monotone components (rejection sampling)

Labels recorded per function (the ground truth the posteriors are scored against): S; s_i; ℓ_i; realized slope share g_i = E_ν[(f_i')²] / Σ_j E_ν[(f_j')²] (spline derivative at GL nodes, quadrature); γ and the pair list; f* (exact optimum); the active indicator.

## Interfaces I want

- `SyntheticObjective`: `__call__(X: np.ndarray[n, D]) -> np.ndarray[n]` (noise-free), `observe(X, rng) -> y` (noisy), attributes `f_star`, `labels` (dataclass), `noise_sd`, `D`; `save(path)` / `load(path)` (npz + json, fully reproducible without re-drawing).
- `make_family(name: str, seed: int, D: int = 100, **overrides) -> SyntheticObjective`.
- A registry/CLI that generates the whole study grid (family × seed × D) to disk with a manifest.
- A thin BoTorch adapter (torch tensors in/out, double precision, bounds [0,1]^D). No GP library dependency in the generator itself; numpy + scipy only.

## Design decisions the plan must make (with a recommendation and reason for each)

1. Exact f*. Additive part is Σ_i max f_i (grid + spline refinement). With interactions, propose choosing the pairs in I to be disjoint so f* = Σ_{pairs} max_{(x_i,x_j)} [f_i + f_j + h_ij] (2-D grid) + Σ_{unpaired} max f_i, and say what to do if I ever contains overlapping pairs.
2. Rotation. Rotating the active subspace about the centre of the cube maps some points outside [0,1]^|S|. Propose how the rotated family is defined so that it stays well-defined on the cube, keeps Var_ν f = 1 (or documents that it does not), and keeps f* computable.
3. Component storage vs pathwise sampling. Grid + cubic spline (recommended) vs random-Fourier-feature paths. State the interpolation error at the roughest lengthscale (ℓ = 0.08 on a 1024 grid) and whether it is acceptable.
4. Monotone components for dense-weak: rejection sampling at ℓ = 3 vs a parametric monotone shape. Estimate the acceptance rate.
5. The sinusoid generator: how to map a nominal ℓ to a frequency band so that its slope share is comparable to the Matérn generator.
6. Seeding: one master seed → independent streams for S selection, component draws, pair draws, noise, so that changing one family knob does not change the others' draws.

## Tests the plan must include (pytest)

- Var_ν f_i = s_i to 1e-8 by quadrature for every component; Σ s_i + Σ s_ij = 1.
- Orthogonality: E_ν[h_ij | x_i] = 0 and E_ν[h_ij · f_i] = 0 numerically.
- Total variance: quadrature (component-wise) agrees with a 10^5-point Sobol Monte-Carlo estimate to within 1 %.
- f* is ≥ f at 10^6 random points and is attained to 1e-6 at the reported argmax.
- Spline evaluation matches an independent re-draw-free reference (e.g., finer grid) to the stated tolerance at ℓ = 0.08.
- Determinism: same seed → bit-identical files; different seed → different S.
- Labels round-trip through save/load.
- Slope share g_i sanity: for the decoupled family the rough-strong coordinate has the largest g_i and the smooth-weak the smallest.

## Deliverable of this planning step

A plan document (markdown) with: module layout and file names; the data model for a saved function; function signatures with docstrings; the six decisions above with recommendations; the test list mapped to files; the CLI for the study grid; estimated effort in hours; and at most five open questions for me. Keep it under 600 lines. Python 3.11, numpy, scipy, pytest; torch only in the adapter. Ask me only if something blocks the plan; otherwise make the reasonable choice and flag it.
