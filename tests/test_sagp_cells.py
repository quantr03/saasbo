"""Tests for sagp.gp's four NumPyro cells: the reference log joint and the per-cell site lists.

Two properties carry the 2x2 comparison. ("product", "lengthscale") must be the vendored
`SAASGP.model` -- its body is copied into `model_product_lengthscale` rather than imported, so
nothing but a test can keep the copy honest; comparing the two log joints on the same parameter
values checks the priors, the kernel and the likelihood in one number. And every cell must expose
exactly the sites the plan names, in the plan's order and shapes, because `Cell.sites` is what
inference retains and every readout downstream indexes those samples by name.

`sagp.gp` is imported first, before this module creates any JAX array, so `sagp/__init__.py`'s
enable_x64 is in force for every array below.
"""
from sagp.gp import (
    CELLS,
    ELL_PRIOR,
    model_product_lengthscale,
)

from functools import partial

import jax.numpy as jnp
import numpy as np
import numpyro
import pytest
from numpyro.infer.util import log_density

import saasgp

N = 12
P = 4

CELL_KEYS = list(CELLS)
CELL_IDS = ["/".join(key) for key in CELL_KEYS]

# The plan's site list per sparsity prior: every site inference retains, sampled and
# deterministic, in the order the model declares them. Stated here as literals rather than read
# off `Cell.sites` so that the trace pins the *names*, not merely the registry's agreement with
# itself; the registry is then checked against the same literals.
SITES = {
    "lengthscale": (
        "kernel_var",
        "kernel_noise",
        "kernel_tausq",
        "_kernel_inv_length_sq",
        "kernel_inv_length_sq",
    ),
    "amplitude": ("kernel_noise", "kernel_tausq", "_a_sq", "a_sq", "kernel_ell"),
}

# Scalar hyperparameters versus the (P,) per-coordinate ones -- the distinction the whole sparse
# parameterization rests on, and the one a mis-shaped prior would silently break.
SHAPES = {
    "kernel_var": (),
    "kernel_noise": (),
    "kernel_tausq": (),
    "_kernel_inv_length_sq": (P,),
    "kernel_inv_length_sq": (P,),
    "_a_sq": (P,),
    "a_sq": (P,),
    "kernel_ell": (P,),
}


def _data() -> tuple[jnp.ndarray, jnp.ndarray]:
    """N points uniform on [0,1]^P with standard-normal responses: the shape a fit sees."""
    rng = np.random.default_rng(0)
    return jnp.asarray(rng.uniform(0.0, 1.0, (N, P))), jnp.asarray(rng.normal(size=N))


@pytest.mark.parametrize("fixed_noise", [None, 1.0e-6], ids=["learned_noise", "fixed_noise"])
def test_product_lengthscale_log_joint_matches_reference(fixed_noise):
    # Both noise branches, because `fixed_noise` replaces the reference's `learn_noise` /
    # `observation_variance` pair and a wrong branch would change the site set and the kernel.
    X, Y = _data()
    params = {
        "kernel_var": 1.3,
        "kernel_noise": 0.05,
        "kernel_tausq": 0.2,
        "_kernel_inv_length_sq": jnp.asarray(np.random.default_rng(1).uniform(0.1, 5.0, P)),
    }
    if fixed_noise is not None:
        del params["kernel_noise"]
    reference = saasgp.SAASGP(alpha=0.1, observation_variance=fixed_noise or 0.0).model
    ours = partial(model_product_lengthscale, alpha=0.1, fixed_noise=fixed_noise)

    # `log_density` substitutes the sampled sites and recomputes the deterministic ones, so this
    # is the full log joint -- priors plus likelihood -- at one point of the parameter space.
    ours_log_joint, _ = log_density(ours, (X, Y), {}, params)
    reference_log_joint, _ = log_density(reference, (X, Y), {}, params)

    assert abs(float(ours_log_joint) - float(reference_log_joint)) < 1.0e-10


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
@pytest.mark.parametrize("fixed_noise", [None, 1.0e-6], ids=["learned_noise", "fixed_noise"])
def test_trace_sites_per_cell(key, fixed_noise):
    cell = CELLS[key]
    X, Y = _data()
    kwargs = {"alpha": cell.alpha_default, "fixed_noise": fixed_noise}
    if cell.prior == "amplitude":
        kwargs["ell_prior"] = ELL_PRIOR
    model = numpyro.handlers.seed(partial(cell.model, **kwargs), rng_seed=0)

    trace = numpyro.handlers.trace(model).get_trace(X, Y)

    assert cell.sites == SITES[cell.prior]
    # `kernel_noise` is the only site the noise branch adds or removes; everything else is the
    # cell's own parameterization and must be there whatever the noise is.
    expected = [site for site in cell.sites if fixed_noise is None or site != "kernel_noise"]
    latent = [
        name
        for name, site in trace.items()
        if site["type"] in {"sample", "deterministic"} and name != "Y"
    ]
    assert latent == expected
    assert [trace[name]["value"].shape for name in latent] == [SHAPES[name] for name in latent]
    assert trace["Y"]["is_observed"]  # the one site excluded above is the likelihood, not a latent
