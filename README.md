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
- a script (saasgp_demo.py) that demonstrates how to fit a GP equipped with a SAAS prior
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
