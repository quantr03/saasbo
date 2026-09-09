"""Gate 1 for the sparse additive GP thesis: amplitude-lengthscale coupling in a centered (OAK-style) kernel.

Two tables, numpy only (~1 minute):
  Table 1: v(ell) = E_x[k_perp(x,x)] = 1 - E_{x,x'}[k(x,x')] for a Matern-5/2 kernel centered under U[0,1],
           against the large-ell limit 5/(36 ell^2).
  Table 2: exact grid posterior over (a, ell) for ONE additive component, with and without normalizing the
           centered kernel by v(ell). Prior: a ~ HalfCauchy(0.1), ell ~ LogNormal(median 1, sd 2 in log space).
           Data: n = 80 uniform inputs, noise sd 0.1, five realized components rescaled to an exact variance under U[0,1].

The point: without normalization the "amplitude" of a component depends on its lengthscale, so reading relevance off a_i
(or thresholding P(a_i > eps)) is not well defined. With normalization a_i^2 is the component's variance under the
reference measure. Reproduce these two tables with the thesis's own kernel/model code before running anything else.
"""
import numpy as np
from numpy.polynomial.legendre import leggauss


def matern52(r):
    s = np.sqrt(5.0) * r
    return (1 + s + s * s / 3.0) * np.exp(-s)


# Gauss-Legendre quadrature on [0,1] = the uniform reference measure (fixed, NOT the empirical measure of the data)
nodes, w = leggauss(64)
nodes, w = 0.5 * (nodes + 1), 0.5 * w


def centered(X, Y, ell):
    """k_perp(x,y) = k - E_z k(x,z) - E_z k(z,y) + E_{z,z'} k(z,z')  (Lu et al. 2022, eq. 8; the note's eq. 2-3)."""
    K = matern52(np.abs(X[:, None] - Y[None, :]) / ell)
    kx = (matern52(np.abs(X[:, None] - nodes[None, :]) / ell) * w).sum(1)
    ky = (matern52(np.abs(Y[:, None] - nodes[None, :]) / ell) * w).sum(1)
    kk = (w[:, None] * matern52(np.abs(nodes[:, None] - nodes[None, :]) / ell) * w[None, :]).sum()
    return K - kx[:, None] - ky[None, :] + kk


def v(ell):
    """Average marginal variance of the centered component under U[0,1]."""
    kk = (w[:, None] * matern52(np.abs(nodes[:, None] - nodes[None, :]) / ell) * w[None, :]).sum()
    return 1.0 - kk


# ---------------------------------------------------------------- Table 1
print("Table 1: v(ell) for the U[0,1]-centered Matern-5/2 kernel")
print(f"{'ell':>6}  {'v(ell)':>10}  {'5/(36 ell^2)':>13}")
for ell in [0.05, 0.1, 0.2, 0.5, 1, 2, 3, 5, 10, 30]:
    print(f"{ell:6.2f}  {v(ell):10.5f}  {5 / (36 * ell**2):13.5f}")

# ---------------------------------------------------------------- Table 2
rng = np.random.default_rng(3)
n, noise = 80, 0.1
x = rng.uniform(0, 1, n)

amps = np.exp(np.linspace(np.log(1e-3), np.log(10), 160))
ells = np.exp(np.linspace(np.log(0.03), np.log(300), 140))
HC_SCALE, LN_MEDIAN, LN_SD = 0.1, 1.0, 2.0


def logprior(a, ell):
    half_cauchy = np.log(2 / (np.pi * HC_SCALE * (1 + (a / HC_SCALE) ** 2)))
    lognormal = -(np.log(ell) - np.log(LN_MEDIAN)) ** 2 / (2 * LN_SD**2) - np.log(ell)
    return half_cauchy + lognormal


def mll(K, y):
    L = np.linalg.cholesky(K + 1e-9 * np.eye(n))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    return -0.5 * y @ alpha - np.log(np.diag(L)).sum()


def posterior(y, normalize):
    LP = np.zeros((len(amps), len(ells)))
    for j, ell in enumerate(ells):
        Kp = centered(x, x, ell)
        if normalize:
            Kp = Kp / v(ell)
        for i, a in enumerate(amps):
            LP[i, j] = mll(a * a * Kp + noise**2 * np.eye(n), y) + logprior(a, ell) + np.log(a) + np.log(ell)
    P = np.exp(LP - LP.max())
    return P / P.sum()


def realized_component(ell, target_var):
    """One draw of a centered component on data + quadrature nodes, rescaled to an exact variance under U[0,1]."""
    Z = np.concatenate([x, nodes])
    Kz = centered(Z, Z, ell) / v(ell)
    g = rng.multivariate_normal(np.zeros(len(Z)), Kz + 1e-8 * np.eye(len(Z)))
    gq = g[n:]
    var = (w * gq**2).sum() - ((w * gq).sum()) ** 2
    return g[:n] * np.sqrt(target_var / var)


cases = [("pure noise (irrelevant)", rng.normal(0, noise, n))]
for ell_t, var_t, name in [(1.5, 0.25, "strong-smooth: var 0.25, ell 1.5"),
                           (0.08, 0.25, "strong-rough:  var 0.25, ell 0.08"),
                           (1.5, 0.02, "weak-smooth:   var 0.02, ell 1.5"),
                           (0.08, 0.02, "weak-rough:    var 0.02, ell 0.08")]:
    cases.append((name, realized_component(ell_t, var_t) + rng.normal(0, noise, n)))

print("\nTable 2: one-component grid posterior, n=80, noise sd 0.1, a~HalfCauchy(0.1), ell~LogNormal(median 1, sd 2)")
print(f"{'case':36s} {'kernel':13s} {'med a':>6} {'P(a>0.3)':>9} {'P(var>0.09)':>12} {'med ell':>8} {'P(ell>5)':>9}")
for case, y in cases:
    for label, normalize in [("unnormalized", False), ("normalized", True)]:
        P = posterior(y, normalize)
        pa, pe = P.sum(1), P.sum(0)
        med_a = amps[np.searchsorted(np.cumsum(pa), 0.5)]
        med_e = ells[np.searchsorted(np.cumsum(pe), 0.5)]
        A, E = np.meshgrid(amps, ells, indexing="ij")
        var = A**2 * (1 if normalize else np.vectorize(v)(E))
        print(f"{case:36s} {label:13s} {med_a:6.3f} {pa[amps > 0.3].sum():9.2f} {P[var > 0.09].sum():12.2f} "
              f"{med_e:8.2f} {pe[ells > 5].sum():9.2f}")