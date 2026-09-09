"""experiments.identify: the SQ1 identification run -- one design, one fit, one flat record.

`identify` fits one `sagp` cell to a fixed Sobol design of a `synthobj` objective and flattens
every readout, diagnostic, ground-truth label and manipulation check into a single dict, with no
acquisition loop anywhere in it. It lives here rather than in `sagp/` because it is study code: it
composes the library's public API instead of extending it, and nothing in `sagp/` may import it.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import asdict

import numpy as np
from scipy.stats import qmc

from sagp.gp import CELLS, CellKey, fit, standardize
from sagp.readouts import manipulation_checks, readouts
from synthobj.families import noise_rng


def identify(
    objective: object,
    cell: CellKey,
    n: int,
    seed: int,
    *,
    sobol_n: int = 2048,
    **fit_kwargs: object,
) -> dict[str, object]:
    """SQ1: fit `cell` once to a fixed Sobol design of `objective` and flatten every readout and
    diagnostic into one record -- no acquisition, no loop, just identification.

    The design is one `n`-point scrambled Sobol sequence on [0,1]^`objective.D`, observed through
    `objective.observe` with its own noise stream (`synthobj.families.noise_rng(seed, run=0)`) --
    both seeded from `seed` alone, so two calls at the same `(objective, cell, n, seed)` fit the
    same data and `samples` compares equal draw for draw. `qmc.Sobol`'s balance-property warning is
    suppressed the way the reference suppresses it, since a caller-chosen `n` need not be a power
    of two. `y` is standardized by `standardize` before it reaches `fit`, as every cell expects;
    the fit's int seed comes from `(seed, n)` salted by `0x1D` so it can collide with neither the
    design's own Sobol seed nor `synthobj`'s own streams, which are seeded independently of it.

    `**fit_kwargs` (`alpha`, `fixed_noise`, `nuts`, `thresholds`, `ell_prior`) reaches `fit`
    verbatim, so the same design can be run through every cell and every sampler budget the study
    needs without this function knowing about any of them. The record merges the run's identity
    (`family`, `seed`, `cell` as `"structure/prior"`, `n`, `D`, the `alpha` actually used), the
    fit's outcome (`status`, `status_reason`, `nuts_attempts`, and the retained attempt's
    `Diagnostics`, prefixed `diag_`), the total wall time, the standardization constants, the
    coordinate-level `readouts` (`native` and `share_hat` are per-sample and left out; their
    summaries `native_median`/`p_active` are not), the ground truth (`labels_s`, `labels_g`,
    `labels_active`), the `manipulation_checks` entries, and `samples`, the retained posterior
    draws as numpy -- a plain `dict`, ready for a run log or an npz row with no schema of its own.
    """
    start = time.perf_counter()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        X = qmc.Sobol(objective.D, scramble=True, seed=seed).random(n)
    y = objective.observe(X, noise_rng(seed, run=0))
    z, y_mean, y_std = standardize(y)

    # 0x1D salts (seed, n) so this seed can never collide with the design's own Sobol seed or with
    # synthobj's own streams (`streams`, `noise_rng`), which are seeded independently of it.
    fit_seed = int(np.random.SeedSequence([seed, n, 0x1D]).generate_state(1)[0])
    fitted = fit(X, z, fit_seed, cell, **fit_kwargs)
    r = readouts(fitted, sobol_n=sobol_n)
    checks = manipulation_checks(r, objective.labels)

    diag = {f"diag_{field}": value for field, value in asdict(fitted.attempts[-1]).items()}
    samples = {site: np.asarray(draws) for site, draws in fitted.samples.items()}
    wall_s = time.perf_counter() - start

    # Resolved the same way `fit` resolves it (`fit`'s own `if alpha is None: alpha =
    # cell.alpha_default`): `fit_kwargs.get("alpha", ...)` would misreport `None` as the alpha
    # actually used whenever a caller passes `alpha=None` explicitly, since the key is then
    # present and `dict.get`'s default never fires.
    alpha = fit_kwargs.get("alpha")
    alpha = CELLS[cell].alpha_default if alpha is None else alpha

    return {
        "family": objective.labels.family,
        "seed": seed,
        "cell": "/".join(cell),
        "n": n,
        "D": objective.D,
        "alpha": alpha,
        "status": fitted.status,
        "status_reason": fitted.status_reason,
        "nuts_attempts": len(fitted.attempts),
        **diag,
        "wall_s": wall_s,
        "y_mean": y_mean,
        "y_std": y_std,
        "native_median": r["native_median"],
        "p_active": r["p_active"],
        "sobol_hat": r["sobol_hat"],
        "total_var_hat": r["total_var_hat"],
        "active_neutral": r["active_neutral"],
        "active_native": r["active_native"],
        "labels_s": np.asarray(objective.labels.s),
        "labels_g": np.asarray(objective.labels.g),
        "labels_active": np.asarray(objective.labels.active),
        **checks,
        "samples": samples,
    }
