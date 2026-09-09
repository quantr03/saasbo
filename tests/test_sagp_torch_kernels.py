"""Tests for sagp.kernels_torch: the three centered kernels restated in torch, against the JAX ones.

Prediction and acquisition move to GPyTorch so BoTorch's `optimize_acqf` can differentiate through
the posterior, but the model has to stay the one NUTS sampled in JAX: every number the torch
kernels produce must be `sagp.gp`'s number. That is what this file pins -- the JAX side is the
oracle throughout, at 1e-10, at the three lengthscales the centering behaves differently at -- plus
the two things the torch side has that the JAX side does not. One kernel object carries all S
retained draws in a leading batch dimension and broadcasts them against `optimize_acqf`'s
`(b, 1, q, D)` candidate batches, and the kernel matrix is differentiable with respect to its
inputs at zero distance, where a repeated training point puts it.

`sagp` is imported first, before this module creates any tensor, so `sagp/__init__.py`'s
`torch.set_default_dtype(torch.float64)` is in force for every tensor below.
"""
import sagp.gp
from sagp import gp
from sagp.kernels_torch import (
    GL_NODES_T,
    GL_WEIGHTS_T,
    CenteredAdditiveAmplitudeKernel,
    CenteredAdditiveLengthscaleKernel,
    CenteredProductAmplitudeKernel,
    kbar_1d,
    v_of_ell_t,
)

import gpytorch
import jax.numpy as jnp
import numpy as np
import pytest
import torch

D = 4
N, M, S = 7, 5, 3

# The three cells built out of the centered component; ("product", "lengthscale") is GPyTorch's
# own `ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=D))` and has nothing to test here.
CELL_KEYS = [("additive", "amplitude"), ("additive", "lengthscale"), ("product", "amplitude")]
CELL_IDS = ["/".join(key) for key in CELL_KEYS]

# Rough, middling, and so smooth that v(ell) is down at 0.015 and the centering is a near-total
# cancellation -- the regime where a reassociated sum would show up first.
ELLS = [0.06, 0.5, 3.0]
ELL_IDS = ["draws", *[f"ell{ell}" for ell in ELLS]]


def _inputs() -> tuple[np.ndarray, np.ndarray]:
    """The (7, 4) and (5, 4) point sets every test uses; n != m is the acquisition path."""
    rng = np.random.default_rng(0)
    return rng.uniform(size=(N, D)), rng.uniform(size=(M, D))


def _draws(ell: float | None = None) -> dict[str, np.ndarray]:
    """S draws of every cell's parameters, on the scales `sagp.gp` parameterizes them on.

    Lengthscales are log-uniform over the range NUTS explores unless `ell` pins every one of them.
    `kernel_inv_length_sq` is rho_i = ell_i^-2, and `ell_rho` is the lengthscale the lengthscale
    cell's kernel actually sees, formed here by the same rho^-0.5 that
    `gp.kernel_additive_lengthscale` forms it by, so the two sides start from the same float.
    """
    rng = np.random.default_rng(1)
    ells = np.exp(rng.uniform(np.log(0.05), np.log(3.0), (S, D)))
    if ell is not None:
        ells = np.full((S, D), ell)
    rho = ells**-2.0
    return {
        "kernel_ell": ells,
        "kernel_inv_length_sq": rho,
        "ell_rho": rho**-0.5,
        "a_sq": np.exp(rng.uniform(np.log(1.0e-3), np.log(3.0), (S, D))),
        "kernel_var": rng.uniform(0.3, 3.0, S),
    }


def _jax_params(draws: dict[str, np.ndarray], s: int) -> dict:
    """Draw `s` as `sagp.gp`'s kernels take it: one dict carrying every cell's parameters."""
    return {
        "a_sq": jnp.asarray(draws["a_sq"][s]),
        "kernel_ell": jnp.asarray(draws["kernel_ell"][s]),
        "kernel_var": float(draws["kernel_var"][s]),
        "kernel_inv_length_sq": jnp.asarray(draws["kernel_inv_length_sq"][s]),
    }


def _torch_kernel(key: tuple[str, str], draws: dict[str, np.ndarray]) -> gpytorch.kernels.Kernel:
    """Cell `key` as one torch kernel carrying all S draws in a leading batch dimension.

    The lengthscale cell's `kernel_var` is an outputscale outside the kernel, which is the whole
    reason `CenteredAdditiveLengthscaleKernel` has no amplitude of its own; the amplitude cells
    carry theirs as `a_sq`.
    """
    batch_shape = torch.Size([S])
    if key == ("additive", "lengthscale"):
        base = CenteredAdditiveLengthscaleKernel(ard_num_dims=D, batch_shape=batch_shape)
        base.lengthscale = torch.as_tensor(draws["ell_rho"])[:, None, :]
        kernel = gpytorch.kernels.ScaleKernel(base, batch_shape=batch_shape)
        kernel.outputscale = torch.as_tensor(draws["kernel_var"])
        return kernel
    amplitude_cls = (
        CenteredAdditiveAmplitudeKernel if key[0] == "additive" else CenteredProductAmplitudeKernel
    )
    kernel = amplitude_cls(ard_num_dims=D, batch_shape=batch_shape)
    kernel.lengthscale = torch.as_tensor(draws["kernel_ell"])[:, None, :]
    kernel.a_sq = torch.as_tensor(draws["a_sq"])[:, None, :]
    return kernel


def _dense(value) -> torch.Tensor:
    """A kernel call's result as a plain tensor: `diag=True` returns one, the matrix a lazy one."""
    return (value if torch.is_tensor(value) else value.to_dense()).detach()


def test_the_quadrature_grid_is_the_jax_grid():
    # Same nodes and weights to the bit, or the two sides center against different measures.
    assert np.array_equal(GL_NODES_T.numpy(), np.asarray(gp.GL_NODES))
    assert np.array_equal(GL_WEIGHTS_T.numpy(), np.asarray(gp.GL_WEIGHTS))


def test_v_of_ell_matches_the_jax_original():
    # The (D,) path is what a kernel's forward uses; the scalar path is what a single component
    # uses. Both are the number the normalization divides by, so both are pinned.
    ell = _draws()["kernel_ell"][0]

    np.testing.assert_allclose(
        v_of_ell_t(torch.as_tensor(ell)).numpy(),
        np.asarray(gp.v_of_ell(jnp.asarray(ell))),
        atol=1.0e-10,
        rtol=0.0,
    )
    for value in ELLS:
        got = v_of_ell_t(torch.as_tensor(value))
        assert got.shape == torch.Size([])
        assert abs(float(got) - float(gp.v_of_ell(value))) < 1.0e-10


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("ell", [None, *ELLS], ids=ELL_IDS)
def test_kbar_1d_matches_kbar_all_coordinate_by_coordinate(ell, normalize):
    X, Z = _inputs()
    ell_vec = _draws(ell)["kernel_ell"][0]
    expected = np.asarray(
        gp.kbar_all(jnp.asarray(X), jnp.asarray(Z), jnp.asarray(ell_vec), normalize)
    )

    for i in range(D):
        got = kbar_1d(
            torch.as_tensor(X[:, i]),
            torch.as_tensor(Z[:, i]),
            torch.as_tensor(ell_vec[i]).reshape(1, 1),
            normalize,
        )

        np.testing.assert_allclose(got.numpy(), expected[:, :, i], atol=1.0e-10, rtol=0.0)


@pytest.mark.parametrize("ell", [None, *ELLS], ids=ELL_IDS)
@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_torch_cell_matches_the_jax_cell_draw_by_draw(key, ell):
    X, Z = _inputs()
    draws = _draws(ell)
    kernel = _torch_kernel(key, draws)

    K = _dense(kernel(torch.as_tensor(X), torch.as_tensor(Z)))

    assert K.shape == torch.Size([S, N, M])
    for s in range(S):
        expected = gp.KERNELS[key][0](
            jnp.asarray(X), jnp.asarray(Z), _jax_params(draws, s), 0.0, False
        )
        np.testing.assert_allclose(K[s].numpy(), np.asarray(expected), atol=1.0e-10, rtol=0.0)


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_torch_cell_diagonal_matches_the_jax_diagonal(key):
    # `cell_kernel_diag` is prediction's O(n D Q) marginal variance, and it is the JAX side's own
    # independent statement of the diagonal -- so agreeing with it *and* with the matrix pins both.
    X, _ = _inputs()
    draws = _draws()
    kernel, X_t = _torch_kernel(key, draws), torch.as_tensor(X)

    diagonal = _dense(kernel(X_t, X_t, diag=True))
    full = _dense(kernel(X_t, X_t))

    assert diagonal.shape == torch.Size([S, N])
    for s in range(S):
        expected = gp.cell_kernel_diag(key, jnp.asarray(X), _jax_params(draws, s))
        np.testing.assert_allclose(
            diagonal[s].numpy(), np.asarray(expected), atol=1.0e-10, rtol=0.0
        )
        np.testing.assert_allclose(
            diagonal[s].numpy(), torch.diagonal(full[s]).numpy(), atol=1.0e-12, rtol=0.0
        )


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_the_kernel_broadcasts_a_candidate_batch_against_the_draws(key):
    # `optimize_acqf` evaluates b restarts of a q-batch as (b, 1, q, D) against the S draws; the
    # kernel has to answer with (b, S, q, m) and answer the same thing for every restart.
    X, Z = _inputs()
    kernel = _torch_kernel(key, _draws())
    b = 2
    X_t, Z_t = torch.as_tensor(X), torch.as_tensor(Z)

    unbatched = _dense(kernel(X_t, Z_t))
    batched = _dense(kernel(X_t[None, None].expand(b, 1, N, D), Z_t))

    assert unbatched.shape == torch.Size([S, N, M])
    assert batched.shape == torch.Size([b, S, N, M])
    assert _dense(kernel(X_t, X_t, diag=True)).shape == torch.Size([S, N])
    for j in range(b):
        np.testing.assert_allclose(
            batched[j].numpy(), unbatched.numpy(), atol=1.0e-10, rtol=0.0
        )


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_torch_cell_is_positive_semidefinite(key):
    X = np.random.default_rng(2).uniform(size=(30, D))
    kernel = _torch_kernel(key, _draws())

    eigenvalues = torch.linalg.eigvalsh(_dense(kernel(torch.as_tensor(X))))

    assert float(eigenvalues.min()) >= -1.0e-10


@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_gradients_through_the_candidate_are_finite(key):
    # `optimize_acqf` differentiates the posterior with respect to the candidate, and x1's first
    # three rows repeat X's, so zero distance occurs -- where a sqrt(sum of squares) formulation
    # would hand the optimizer NaN rather than the true derivative 0.
    X, _ = _inputs()
    kernel = _torch_kernel(key, _draws())
    x1 = torch.tensor(X[:3], requires_grad=True)

    (grad,) = torch.autograd.grad(kernel(x1, torch.as_tensor(X)).to_dense().sum(), x1)

    assert bool(torch.isfinite(grad).all())


@pytest.mark.parametrize("ell", ELLS)
@pytest.mark.parametrize("key", CELL_KEYS, ids=CELL_IDS)
def test_the_diagonal_matches_the_full_matrix_at_every_lengthscale(key, ell):
    # The diag path is a separate formula (elementwise |x1 - x2|, no (n, m) block), so it gets its
    # own check at each lengthscale, including the one where the centering nearly cancels.
    X, _ = _inputs()
    kernel, X_t = _torch_kernel(key, _draws(ell)), torch.as_tensor(X)

    diagonal = _dense(kernel(X_t, X_t, diag=True))
    full = _dense(kernel(X_t, X_t))

    np.testing.assert_allclose(
        diagonal.numpy(),
        torch.diagonal(full, dim1=-2, dim2=-1).numpy(),
        atol=1.0e-12,
        rtol=0.0,
    )
