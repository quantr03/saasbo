"""The migration's central pin: `run_bo` *is* BoTorch's SAASBO on Papenmeier et al. (2025)'s loop.

Every other test in this suite pins one piece -- `fit` against `fit_fully_bayesian_model_nuts`,
`propose_ei`'s seeding and its LogEI, the row schema, the resume. This one pins their
*composition*: a loop written from scratch on BoTorch's public API, in under forty lines, must
choose the same points and observe the same values as `run_bo` -- bit for bit, over fifteen fitted
iterations. If any of the standardization, the model construction, the postprocess/thin/load
sequence, the incumbent, the acquisition, the maximizer's two seeds or the bookkeeping had drifted
from BoTorch's own by one ulp, the two trajectories would separate at the first iteration where
the drift changes an argmax and never rejoin.

What the two sides deliberately share is only what is not under test: the seeds
(`sagp.bo.iteration_rngs`), the initial design (`sagp.bo.initial_design`, a scrambled Sobol
sequence) and the standardization (`sagp.gp.standardize`, ddof 0). Those three are the run's
definition, not its implementation -- a hand-written loop that guessed at them would be pinning
nothing. What the hand-written side shares with `sagp` is nothing else: no `sagp` model, no
`sagp` acquisition, no `sagp` optimizer call, no `run_bo`. It builds a stock
`SaasFullyBayesianSingleTaskGP`, fits it with `fit_fully_bayesian_model_nuts`, and maximizes
`LogExpectedImprovement` with `optimize_acqf` under the protocol's options.

Two settings are reduced from the study's, and neither weakens the pin:

* `NUTSConfig(128, 64, 4)` rather than the production 512/256/16. The claim here is that the two
  loops are the same computation, and a computation that agrees to the bit at one sampler budget
  agrees at every other -- both sides run the same NUTS on the same seed either way. Equivalence
  *at* 512/256/16 is what `test_sagp_inference.py` pins, on every invocation of the suite; this
  test buys fifteen fitted iterations for the price of two minutes instead of half an hour.
* `thresholds=NEVER_FAILS`, so the gate always reports "ok". The gate is a label, not a branch:
  `run_bo` queries with an excluded fit exactly as it queries with a passing one (only an
  *exception* out of the fit changes what is queried), so the trajectory is the same either way.
  Silencing it keeps this test about the loop rather than about whether a 64-draw chain converged.
"""
from __future__ import annotations

from sagp.bo import initial_design, iteration_rngs, propose_ei, run_bo
from sagp.diagnostics import DiagThresholds
from sagp.gp import NUTSConfig, fit, standardize

import botorch.settings
import gpytorch.settings
import numpy as np
import pytest
import torch
from botorch.acquisition.analytic import LogExpectedImprovement
from botorch.fit import fit_fully_bayesian_model_nuts
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from botorch.optim import optimize_acqf

from synthobj.families import make_family

_SEED = 3
_D = 5
_N_INIT = 5
_T = 20  # fifteen fitted iterations
# A reduced sampler budget; see this module's docstring for why the pin does not depend on it.
_NUTS = NUTSConfig(num_warmup=128, num_samples=64, thinning=4)
# Diagnostics nothing can fail, so every fit reports "ok" on both sides.
_NEVER_FAILS = DiagThresholds(r_hat_max=float("inf"), n_eff_min=0.0, max_divergences=10**9)


def _botorch_loop(objective, seed, T, n_init, nuts):
    """SAASBO on Papenmeier's protocol, written directly on BoTorch's public API.

    Shares `initial_design`, `iteration_rngs` and `standardize` with `sagp`; nothing else. The
    `optimize_acqf` options and the two seeds around it are the protocol's, documented in
    `sagp.bo.propose_ei`. Returns the design and its observations.
    """
    D = objective.D
    X = initial_design(D, n_init, seed)
    y = np.array([
        float(objective.observe(X[t][None, :], iteration_rngs(seed, t).noise_rng)[0])
        for t in range(n_init)
    ])
    for t in range(n_init, T):
        z, _, _ = standardize(y)
        rngs = iteration_rngs(seed, t)
        with botorch.settings.validate_input_scaling(False):
            gp = SaasFullyBayesianSingleTaskGP(torch.as_tensor(X), torch.as_tensor(z)[:, None])
        fit_fully_bayesian_model_nuts(
            gp, max_tree_depth=nuts.max_tree_depth, warmup_steps=nuts.num_warmup,
            num_samples=nuts.num_samples, thinning=nuts.thinning, disable_progbar=True,
            seed=rngs.nuts_seed,
        )
        acq = LogExpectedImprovement(model=gp, best_f=float(z.max()))
        bounds = torch.stack(
            [torch.zeros(D, dtype=torch.float64), torch.ones(D, dtype=torch.float64)]
        )
        options = {
            "batch_limit": 1, "maxiter": 200, "sample_around_best": True,
            "sample_around_best_sigma": 1e-3, "seed": rngs.sobol_seed,
        }
        with torch.random.fork_rng(), gpytorch.settings.cholesky_max_tries(9):
            torch.manual_seed(rngs.torch_seed)
            x, _ = optimize_acqf(
                acq, bounds=bounds, q=1, num_restarts=5, raw_samples=512, options=options
            )
        x = x.detach().numpy()[0]
        X = np.vstack([X, x])
        y = np.append(y, float(objective.observe(x[None, :], rngs.noise_rng)[0]))
    return X, y


@pytest.mark.slow
def test_run_bo_reproduces_a_hand_written_botorch_loop():
    """`run_bo` and `_botorch_loop` choose the same 20 points and observe the same 20 values.

    On the cell that *is* SAASBO (`product/lengthscale`), which is BoTorch's `SaasPyroModel` with
    its two postprocessing deletions undone. `np.array_equal`, never `allclose`: an approximate
    agreement would mean the two loops are running different computations that have not yet
    diverged, which is the thing this test exists to rule out.
    """
    objective = make_family("aligned3", 0, D=_D)
    ref_X, ref_y = _botorch_loop(objective, _SEED, _T, _N_INIT, _NUTS)

    def surrogate(X, z, seed):
        return fit(
            X, z, seed, cell=("product", "lengthscale"), nuts=_NUTS, thresholds=_NEVER_FAILS
        )

    ours = run_bo(objective, surrogate, _SEED, T=_T, n_init=_N_INIT, propose=propose_ei)

    assert np.array_equal(ours.X, ref_X)
    assert np.array_equal(ours.y, ref_y)
