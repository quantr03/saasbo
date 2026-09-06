"""Raw, uncentered 1-D function draws on a fixed grid: GP, sinusoid, and a monotone-rejection wrapper.

Three generators sit on top of `synthobj.kernel`'s covariance machinery:

- `gp_draw` — the default generator, a single draw from a cached PSD factor of the normalized
  centered Matern-5/2 kernel (`kernel.eigen_factor`).
- `sinusoid_draw` — an out-of-family alternative used by exactly one study variant
  (`aligned10_sin`), a sum of a few random sinusoids with frequencies matched to the Matern
  kernel's slope-energy-to-variance ratio via `omega_for_ell`.
- `draw_until_monotone` — a rejection wrapper around either generator, used by the dense-weak
  family, that keeps redrawing until a path is monotone on its uniform-grid portion.

Every function here returns raw, uncentered values. Centering and exact-variance-share rescaling
under the uniform reference measure U[0,1] is the job of the `Component` that consumes these
draws (a later task), not this module.
"""
import numpy as np

from synthobj.kernel import v


def omega_for_ell(ell: float) -> float:
    """sqrt(5 / (3 * ell**2 * v(ell))) / (2*pi): cycles/unit matching the Matern slope energy
    per unit variance (Matern E[(f')^2]/Var = 5/(3 ell^2 v(ell)); a unit-variance sinusoid at
    omega cycles/unit has E[(u')^2] = (2*pi*omega)^2; equating the two and solving for omega
    gives this closed form).

    The match holds only in expectation, and only where the sinusoid completes at least one
    full cycle on [0,1] (omega(ell) >= 1, i.e. ell up to ~0.2-0.3): for smoother lengthscales
    omega(ell) drops below 1 and centering under U[0,1] removes proportionally more variance
    than slope energy, so a realized sinusoid draw's slope-energy-to-variance ratio runs
    increasingly above this target, from about 1.02-1.04x at ell=0.08 up to about 1.80-1.86x
    at ell=3.
    """
    return np.sqrt(5.0 / (3.0 * ell**2 * v(ell))) / (2.0 * np.pi)


def gp_draw(F: np.ndarray, rng) -> np.ndarray:
    """A single GP draw at the joint points behind `F`: `F @ rng.standard_normal(F.shape[1])`.

    `F` is a PSD square-root factor of the normalized centered Matern-5/2 kernel on the joint
    points (see `kernel.eigen_factor`), so the returned vector has that kernel as its exact
    covariance. Raw and uncentered.
    """
    return F @ rng.standard_normal(F.shape[1])


def sinusoid_draw(Z: np.ndarray, ell: float, rng, n_terms: int = 3) -> np.ndarray:
    """Sum of `n_terms` random sinusoids evaluated at `Z`, raw and uncentered.

    Each term k draws w_k ~ N(0,1), omega_k ~ LogUniform(omega(ell)/1.25, 1.25*omega(ell)),
    phi_k ~ U[0, 2*pi), and contributes w_k * sin(2*pi*omega_k*z + phi_k); the returned array
    is their sum at each point of `Z`. `Z` must be one-dimensional.
    """
    omega_center = omega_for_ell(ell)
    log_bounds = np.log(omega_center / 1.25), np.log(1.25 * omega_center)
    w = rng.standard_normal(n_terms)
    omega = np.exp(rng.uniform(log_bounds[0], log_bounds[1], size=n_terms))
    phi = rng.uniform(0.0, 2.0 * np.pi, size=n_terms)
    terms = w[:, None] * np.sin(2.0 * np.pi * omega[:, None] * Z[None, :] + phi[:, None])
    return terms.sum(axis=0)


def draw_until_monotone(draw_fn, n_grid: int, rng, max_tries: int = 200) -> np.ndarray:
    """Repeat `draw_fn(rng)` until `values[:n_grid]` is non-decreasing or non-increasing.

    Monotonicity is checked only on the leading uniform-grid portion of the draw, not on any
    trailing quadrature-node values. Raises `RuntimeError` if no monotone draw is found in
    `max_tries` attempts.
    """
    for _ in range(max_tries):
        values = draw_fn(rng)
        diffs = np.diff(values[:n_grid])
        if np.all(diffs >= 0) or np.all(diffs <= 0):
            return values
    raise RuntimeError(f"draw_until_monotone: no monotone draw found in {max_tries} tries")
