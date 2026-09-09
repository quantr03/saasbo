"""The reproduction test: `sagp.bo`'s loop *is* `saasbo.run_saasbo`, once both are given the
same random numbers.

Every other test in this suite pins one piece -- the sampler's draws, the posterior, the
acquisition, the row schema -- against the vendored reference or against itself. This one pins the
whole design at once: on the cell that is SAASBO (`product/lengthscale`), with the reference's own
EI, the reference's fixed 1e-6 observation variance and noise-free evaluations, our loop must
produce the reference's query points *bit for bit* over 15 fitted iterations. Nothing but seeding
is allowed to differ, so if any of the standardization, the fit, the candidate set, the L-BFGS-B
restarts or the bookkeeping had drifted from the reference by one ulp, the two trajectories would
separate at the first iteration where the drift changes an argmax and never rejoin.

The reference draws from global random state in exactly three places, and each is patched here so
that it draws what `iteration_rngs(seed, t)` would have given our loop instead:

1. `SAASGP.fit` splits `PRNGKey(0)` and hands the first half to NUTS, so *every* iteration's chain
   runs from the same key. `SeededSAASGP` replaces that one line with `iteration_rngs(s, t).key`,
   where `t = len(Y_train)` is the number of points already evaluated -- our loop's own index for
   the point it is about to choose.
2. `saasbo.optimize_ei` builds its 5000 candidates from an unseeded `qmc.Sobol(dim, scramble=True)`.
   `SobolShim` injects `iteration_rngs(s, t).sobol_seed` when no seed is given and passes an
   explicit one through unchanged -- which is what leaves `run_saasbo`'s *initial design*
   (`qmc.Sobol(len(lb), scramble=True, seed=seed)`) alone, since that one is already seeded and is
   already `sagp.bo.initial_design`.
3. `saasbo.optimize_ei` jitters the incumbent with `np.random.randn(1, dim)`. The replacement
   draws from `iteration_rngs(s, t).jitter_rng` and then advances the counter, because within one
   `optimize_ei` the Sobol call comes first and the jitter second: the counter therefore starts at
   `n_init` and moves once per iteration, after the jitter.

`saasbo.qmc` is `scipy.stats.qmc` and `saasbo.np` is `numpy`, so patches 2 and 3 are global for as
long as they are installed. They are installed inside a `monkeypatch.context()` around the
reference call alone, after our own run has finished, so that `sagp.bo`'s Sobol calls are the real
ones.
"""
from __future__ import annotations

import types

import jax
import numpy as np
import pytest
from scipy.stats import qmc

import saasbo
import saasgp
from experiments.run_bo import run
from sagp.bo import iteration_rngs
from sagp.diagnostics import DiagThresholds
from synthobj.families import make_family

# The run seed, shared by both sides: `SeededSAASGP` and the two shims read it to rebuild the very
# streams `run` used, so it is a module constant rather than an argument threaded through them.
_SEED = 3
_D = 5
_T = 25
_N_INIT = 10
# The reference has no refit path, so ours must not take one either: thresholds nothing can fail.
_NEVER_FAILS = DiagThresholds(float("inf"), 0.0, 10**9)


class SeededSAASGP(saasgp.SAASGP):
    """`SAASGP` whose `fit` takes its HMC key from `iteration_rngs` rather than from `PRNGKey(0)`.

    `SAASGP.fit`'s body verbatim but for the key line. `rng_key_predict` is set to a fixed key
    instead of the discarded half of the reference's split: `SAASGP.predict` accepts an `rng_key`
    and never uses it, so no number in the run depends on it. `verbose` is not forced here --
    `run_saasbo` already constructs the class with `verbose=False`.
    """

    def fit(self, X_train, Y_train, seed=0):
        self.X_train, self.Y_train = X_train.copy(), Y_train.copy()
        self.rng_key_hmc = iteration_rngs(_SEED, len(Y_train)).key
        self.rng_key_predict = jax.random.PRNGKey(0)
        self.chain_samples, self.flat_samples, self.summary = self.run_inference(
            self.rng_key_hmc, X_train, Y_train
        )
        return self


@pytest.mark.slow
def test_our_loop_reproduces_the_reference_trajectory(tmp_path, monkeypatch):
    """Our loop and `run_saasbo` choose the same 25 points and observe the same 25 values.

    See this module's docstring for what the three patches are and why they are the only three.
    Our run goes first, unpatched; the reference follows inside a `monkeypatch.context()`, since
    patching `qmc.Sobol` reaches `sagp.bo`'s own candidate sets too.
    """
    objective = make_family("aligned3", 0, D=_D)

    run_dir = run(
        objective,
        "product/lengthscale",
        seed=_SEED,
        T=_T,
        n_init=_N_INIT,
        out_dir=tmp_path,
        acq="ei",
        fixed_noise=1.0e-6,
        noiseless=True,
        thresholds=_NEVER_FAILS,
    )
    with np.load(run_dir / "checkpoint.npz") as data:
        X_ours, y_ours = data["X"], data["y"]

    counter = types.SimpleNamespace(t=_N_INIT)
    real_sobol = qmc.Sobol

    def SobolShim(dim, scramble=True, seed=None, **kwargs):
        return real_sobol(
            dim,
            scramble=scramble,
            seed=seed if seed is not None else int(iteration_rngs(_SEED, counter.t).sobol_seed),
            **kwargs,
        )

    def randn(*shape):
        draw = iteration_rngs(_SEED, counter.t).jitter_rng.standard_normal(shape)
        counter.t += 1
        return draw

    with monkeypatch.context() as patched:
        patched.setattr(saasbo, "SAASGP", SeededSAASGP)
        patched.setattr(saasbo.qmc, "Sobol", SobolShim)
        patched.setattr(saasbo.np.random, "randn", randn)
        X_ref, Y_ref = saasbo.run_saasbo(
            lambda x: -objective(x),
            np.zeros(_D),
            np.ones(_D),
            max_evals=_T,
            num_init_evals=_N_INIT,
            seed=_SEED,
            kernel="matern",
        )

    # One jitter draw per fitted iteration and none from the initial design. Checked first because
    # it localizes the failure: a counter that never moved, or moved twice, would leave every fit
    # on the right key and every candidate set on the wrong one, which looks like a drifted argmax.
    assert counter.t == _T

    # `run_saasbo` minimizes -f and reports its query points in the original box, which is the unit
    # box here (`lb + (ub - lb) * X` with lb = 0 and ub = 1 is exact), so the two compare directly.
    assert np.array_equal(X_ours, X_ref)
    assert np.allclose(y_ours, -Y_ref, rtol=0.0, atol=1.0e-12)
