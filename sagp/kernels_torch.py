"""sagp.kernels_torch: the three centered Matern-5/2 cells as GPyTorch kernels, batched over draws.

Acquisition needs a posterior BoTorch's `optimize_acqf` can differentiate through, which the JAX
kernels cannot supply, so `sagp.gp`'s `matern52_1d`, `v_of_ell`, `_quad_mean` and `kbar_all` and
the three centered `kernel_*` are restated here in torch and tested equal to them at 1e-10. The
fourth cell needs nothing here: ("product", "lengthscale") is GPyTorch's own
`ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=D))`.

One kernel object carries all S retained posterior draws: `lengthscale` and `a_sq` are GPyTorch
parameter tables of shape `batch_shape + (1, D)`, so a call broadcasts the draws against
`optimize_acqf`'s `(b, 1, q, D)` candidate batches and returns `(b, S, q, m)`. The per-coordinate
terms are accumulated one at a time rather than stacked, so the (n, m, D) tensor `kbar_all` forms
is never built here -- 800 MB at 5000 candidates, n = 200, D = 100.

Imports no `sagp` module: `sagp.gp` imports this one.
"""
from __future__ import annotations

import functools
import math

import gpytorch
import torch
from numpy.polynomial.legendre import leggauss
from torch import Tensor

# 64-node Gauss-Legendre quadrature on [0,1], the reference measure U[0,1]; the same affine map of
# the same nodes as `sagp.gp`'s, which a test pins these equal to bit for bit.
_NODES, _WEIGHTS = leggauss(64)
GL_NODES_T = torch.as_tensor(0.5 * (_NODES + 1))
GL_WEIGHTS_T = torch.as_tensor(0.5 * _WEIGHTS)


@functools.cache
def _gl(device: torch.device) -> tuple[Tensor, Tensor]:
    """The nodes and weights on `device`: copied to a GPU once, not on every kernel call.

    On the CPU `.to` returns the tensors themselves, so the CPU path computes exactly what it did.
    """
    return GL_NODES_T.to(device), GL_WEIGHTS_T.to(device)


_ROOT_FIVE = math.sqrt(5.0)


def matern52_1d_t(r: Tensor) -> Tensor:
    """Unit-variance Matern-5/2 covariance at scaled distance r >= 0: (1 + s + s^2/3) e^-s, s = sqrt(5) r.

    Elementwise; `sagp.gp.matern52_1d` in torch. In r rather than r^2, whose sqrt has an infinite
    derivative at zero distance -- and the acquisition optimizer starts candidates at training
    points, where the distance is zero.
    """
    s = _ROOT_FIVE * r
    return (1.0 + s + s * s / 3.0) * torch.exp(-s)


def v_of_ell_t(ell: Tensor) -> Tensor:
    """Average marginal variance of the centered unit-amplitude component under U[0,1].

    v(ell) = 1 - sum_{q,q'} w_q w_q' k(|t_q - t_q'| / ell), `sagp.gp.v_of_ell` in torch. `ell` may
    have any shape -- scalar, (D,), or a (..., 1, 1) block shaped to broadcast against a kernel
    matrix -- and the result has that same shape.
    """
    nodes, node_weights = _gl(ell.device)
    r = torch.abs(nodes[:, None] - nodes) / ell[..., None, None]  # (..., Q, Q)
    weights = node_weights[:, None] * node_weights  # (Q, Q)
    return 1.0 - (weights * matern52_1d_t(r)).sum(dim=(-2, -1))


def quad_mean_t(x: Tensor, ell: Tensor) -> Tensor:
    """m(x) = sum_q w_q k(|x - t_q| / ell); x (..., n), `ell` broadcastable to (..., 1) -> (..., n).

    `sagp.gp._quad_mean` for one coordinate: the component integrated against the constants under
    U[0,1], which is what the centering subtracts off.
    """
    nodes, weights = _gl(x.device)
    r = torch.abs(x[..., :, None] - nodes) / ell[..., None]  # (..., n, Q)
    return (weights * matern52_1d_t(r)).sum(dim=-1)


def kbar_1d(x1: Tensor, x2: Tensor, ell: Tensor, normalize: bool, diag: bool = False) -> Tensor:
    """Centered Matern-5/2 kernel of ONE coordinate: the [:, :, i] slice of `sagp.gp.kbar_all`.

    x1 (..., n) and x2 (..., m) against `ell` broadcastable to (..., 1, 1) give (..., n, m); with
    `diag` x1 and x2 are both (..., n) against `ell` broadcastable to (..., 1) and the result is
    the elementwise (..., n) kernel, which is the matrix's diagonal when x1 is x2. `ell` therefore
    needs the trailing unit dimensions it will broadcast through, one per axis of the result.

    kbar = k(|x1 - x2| / ell) - m(x1) - m(x2) + (1 - v(ell)), the component projected onto the
    orthogonal complement of the constants under U[0,1] (Lu et al. 2022, eq. 8), divided by v(ell)
    when `normalize`.
    """
    if diag:
        k = matern52_1d_t(torch.abs(x1 - x2) / ell)
        centering = quad_mean_t(x1, ell) + quad_mean_t(x2, ell)
    else:
        ell_mean = ell[..., 0]  # (..., 1): one axis fewer, the means being vectors not matrices
        k = matern52_1d_t(torch.abs(x1[..., :, None] - x2[..., None, :]) / ell)
        centering = (
            quad_mean_t(x1, ell_mean)[..., :, None] + quad_mean_t(x2, ell_mean)[..., None, :]
        )
    v = v_of_ell_t(ell)
    kbar = k - centering + (1.0 - v)
    return kbar / v if normalize else kbar


class _CenteredKernel(gpytorch.kernels.Kernel):
    """What the three centered cells share: per-coordinate `kbar_1d`, combined by the subclass.

    `has_lengthscale` has GPyTorch register `raw_lengthscale` of shape `batch_shape + (1, D)` under
    a `Positive` constraint, so `self.lengthscale` is the (S, 1, D) table of per-draw,
    per-coordinate lengthscales; `has_amplitude` registers `raw_a_sq` the same way, for the two
    cells an amplitude parameterizes. Subclasses set `normalize`, supply each coordinate's term
    through `_term` and reduce them with `_reduce`; the reduction runs term by term so no
    (..., n, m, D) tensor exists. `last_dim_is_batch` is deprecated in GPyTorch and unsupported
    here.
    """

    has_lengthscale = True
    has_amplitude = False
    normalize = True

    def __init__(self, ard_num_dims: int, batch_shape: torch.Size = torch.Size([]), **kwargs):
        super().__init__(ard_num_dims=ard_num_dims, batch_shape=batch_shape, **kwargs)
        if self.has_amplitude:
            self.register_parameter(
                name="raw_a_sq",
                parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, ard_num_dims)),
            )
            self.register_constraint("raw_a_sq", gpytorch.constraints.Positive())

    @property
    def a_sq(self) -> Tensor | None:
        """The per-draw, per-coordinate amplitudes (S, 1, D), or None on a cell without them."""
        if not self.has_amplitude:
            return None
        return self.raw_a_sq_constraint.transform(self.raw_a_sq)

    @a_sq.setter
    def a_sq(self, value: Tensor) -> None:
        if not self.has_amplitude:
            raise RuntimeError(f"{type(self).__name__} has no amplitude.")
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_a_sq)
        self.initialize(raw_a_sq=self.raw_a_sq_constraint.inverse_transform(value))

    @staticmethod
    def _coordinate(table: Tensor, i: int, diag: bool) -> Tensor:
        """Column i of a (..., 1, D) parameter table, shaped to broadcast against the output."""
        column = table[..., 0, i]
        return column[..., None] if diag else column[..., None, None]

    def _term(self, kbar: Tensor, i: int, diag: bool) -> Tensor:
        """Coordinate i's contribution, given its centered kernel."""
        raise NotImplementedError

    def forward(self, x1: Tensor, x2: Tensor, diag: bool = False, **params) -> Tensor:
        if params.get("last_dim_is_batch"):
            raise NotImplementedError(f"{type(self).__name__} does not support last_dim_is_batch.")
        total = None
        for i in range(self.ard_num_dims):
            kbar = kbar_1d(
                x1[..., i],
                x2[..., i],
                self._coordinate(self.lengthscale, i, diag),
                self.normalize,
                diag=diag,
            )
            term = self._term(kbar, i, diag)
            total = term if total is None else self._reduce(total, term)
        return total


class CenteredAdditiveAmplitudeKernel(_CenteredKernel):
    """k(x, z) = sum_i a_sq_i kbar_i(x_i, z_i); `sagp.gp.kernel_additive_amplitude` in torch.

    Normalized, so a_sq_i is coordinate i's variance under the reference measure and the amplitude
    threshold that labels a coordinate active is well defined.
    """

    has_amplitude = True
    _reduce = staticmethod(torch.add)

    def _term(self, kbar: Tensor, i: int, diag: bool) -> Tensor:
        return self._coordinate(self.a_sq, i, diag) * kbar


class CenteredAdditiveLengthscaleKernel(_CenteredKernel):
    """k(x, z) = sum_i kbar_i(x_i, z_i); `sagp.gp.kernel_additive_lengthscale` without its variance.

    Deliberately *not* normalized: sparsity is rho_i -> 0, which drives v(ell_i) and the component
    to zero, so dividing v out would undo the shrinkage. The cell's `outputscale` site is the
    outputscale of the `ScaleKernel` this is wrapped in, not a parameter here.
    """

    normalize = False
    _reduce = staticmethod(torch.add)

    def _term(self, kbar: Tensor, i: int, diag: bool) -> Tensor:
        return kbar


class CenteredProductAmplitudeKernel(_CenteredKernel):
    """k(x, z) = prod_i (1 + a_sq_i kbar_i(x_i, z_i)); `sagp.gp.kernel_product_amplitude` in torch.

    Normalized, as the additive amplitude cell is; a_sq_i = 0 removes coordinate i from every
    interaction rather than only from its own main effect.
    """

    has_amplitude = True
    _reduce = staticmethod(torch.mul)

    def _term(self, kbar: Tensor, i: int, diag: bool) -> Tensor:
        return 1.0 + self._coordinate(self.a_sq, i, diag) * kbar
