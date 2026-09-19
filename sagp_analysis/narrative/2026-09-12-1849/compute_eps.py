"""Reproduce sagp.gp's ELL_EPS / RHO_EPS / ALPHA_AMPLITUDE with numpy only (no jax): v(ell) = 1 - sum w_q w_q' k(|t_q - t_q'|/ell),
64-node Gauss-Legendre on [0,1], k the unit Matern-5/2; ELL_EPS solves v(ell) = ACTIVE_EPS = 0.02 (bracket [0.5, 50] as in gp.py)."""
import sys, json
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq
n, w = leggauss(64); t = 0.5 * (n + 1); w = 0.5 * w
def k(r):
    s = np.sqrt(5.0) * r
    return (1 + s + s * s / 3) * np.exp(-s)
def v(ell):
    r = np.abs(t[:, None] - t[None, :]) / ell
    return 1.0 - float(w @ k(r) @ w)
ACTIVE_EPS = 0.02
ELL_EPS = brentq(lambda e: v(e) - ACTIVE_EPS, 0.5, 50.0)
RHO_EPS = ELL_EPS ** -2
out = dict(ACTIVE_EPS=ACTIVE_EPS, ELL_EPS=ELL_EPS, RHO_EPS=RHO_EPS, ALPHA_LENGTHSCALE=0.1, ALPHA_AMPLITUDE=0.1 * ACTIVE_EPS / RHO_EPS,
           v_0_5=v(0.5), v_50=v(50.0), v_0_08=v(0.08), v_1_5=v(1.5), check_v_0_5_is_0_282=abs(v(0.5) - 0.282) < 1e-3)
json.dump(out, open(sys.argv[1] + "/tables/eps_constants.json", "w"), indent=1)
print(json.dumps(out, indent=1))
