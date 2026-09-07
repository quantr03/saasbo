"""Tests for `synthobj.botorch_adapter`, the package's only torch/botorch-importing module.

`pytest.importorskip("botorch")` below gates the whole module (torch comes with it): on a machine
without botorch installed, collection of this file is skipped rather than erroring, so the rest of
the suite stays green. `test_import_synthobj_does_not_import_torch` is the actual guard that this
adapter module's existence has not compromised the package's own torch-free import -- it must run
in a fresh subprocess: this file has already imported torch by the time any in-process assertion
here could run (pytest imports every collected test file before running any test), which is exactly
the sibling pitfall `tests/test_objective.py`'s own subprocess guard test documents.

`make_family("interaction_g0.25", ...)` at D = 20 (never D = 100, per the brief) is used throughout:
it has both main effects and a product interaction, so `x_star`/`f_star` exercise the paired-block
assembly path in `SyntheticObjective`, not just independent per-coordinate maxima.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("botorch")

import torch

from synthobj.botorch_adapter import SyntheticObjectiveTestFunction
from synthobj.families import make_family

D = 20
SEED = 3


@pytest.fixture(scope="module")
def obj():
    return make_family("interaction_g0.25", SEED, D=D)


def test_import_synthobj_does_not_import_torch(repo_root: Path) -> None:
    """A fresh interpreter that imports only `synthobj` must not pull in torch or botorch.

    Mirrors `tests/test_objective.py`'s `test_family_spec_annotation_is_deferred` subprocess guard:
    an in-process `"torch" in sys.modules` check would be meaningless here, since this very test
    module already imported torch above.
    """
    result = subprocess.run(
        [sys.executable, "-c", 'import sys, synthobj; print("torch" in sys.modules, "botorch" in sys.modules)'],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False False"


def test_batched_input_shape_and_values_match_numpy(obj) -> None:
    """`(3, 4, D)` in -> `(3, 4)` out, float64, matching an independent numpy recomputation.

    Recomputing via `obj(X_np.reshape(-1, D)).reshape(3, 4)` calls the same public numpy evaluator
    on the same flattened points independently of the adapter's own reshape logic, so a mutant that
    flattened rather than restored the batch shape, or that reordered points, would not coincide
    with this by chance.
    """
    rng = np.random.default_rng(0)
    X_np = rng.uniform(size=(3, 4, D))
    tf = SyntheticObjectiveTestFunction(obj)

    out = tf(torch.tensor(X_np, dtype=torch.double))

    assert out.shape == (3, 4)
    assert out.dtype == torch.float64
    expected = obj(X_np.reshape(-1, D)).reshape(3, 4)
    np.testing.assert_allclose(out.detach().numpy(), expected, atol=1e-12, rtol=0)


def test_bounds_is_zero_one_tensor(obj) -> None:
    tf = SyntheticObjectiveTestFunction(obj)
    assert tf.bounds.shape == (2, D)
    assert torch.equal(tf.bounds[0], torch.zeros(D, dtype=torch.double))
    assert torch.equal(tf.bounds[1], torch.ones(D, dtype=torch.double))


def test_optimal_value_and_minimization_flag_follow_our_maximization_convention(obj) -> None:
    """`optimal_value == f_star` un-negated, and the adapter reports itself a maximization problem.

    `is_minimization_problem` depends on `_is_minimization_by_default`, a class attribute
    `optimal_value` itself never reads -- a mutant that left `_is_minimization_by_default` at
    botorch's own `True` default would still pass an `optimal_value`-only check, so both are
    asserted here.
    """
    tf = SyntheticObjectiveTestFunction(obj)
    assert tf.optimal_value == pytest.approx(obj.f_star)
    assert tf.is_minimization_problem is False

    tf_negated = SyntheticObjectiveTestFunction(obj, negate=True)
    assert tf_negated.optimal_value == pytest.approx(-obj.f_star)
    assert tf_negated.is_minimization_problem is True


def test_negate_flips_an_actual_evaluated_value(obj) -> None:
    """`negate=True` changes what `forward`/`__call__` actually returns, not just a stored flag."""
    X = torch.tensor(np.random.default_rng(1).uniform(size=(5, D)), dtype=torch.double)
    out = SyntheticObjectiveTestFunction(obj)(X)
    out_negated = SyntheticObjectiveTestFunction(obj, negate=True)(X)

    np.testing.assert_array_equal(out_negated.detach().numpy(), -out.detach().numpy())
    assert torch.any(out != 0)  # guards against a vacuous pass if every value happened to be 0


def test_optimizers_reproduces_x_star_and_attains_f_star(obj) -> None:
    """`_optimizers` round-trips `obj.labels.x_star`, and evaluating there attains `f_star`.

    The second assertion calls the public numpy evaluator directly on the recorded optimizer -- an
    independent recomputation that would fail if `x_star` and `f_star` were ever mismatched.
    """
    tf = SyntheticObjectiveTestFunction(obj)
    x_opt = tf.optimizers[0].numpy()

    np.testing.assert_allclose(x_opt, obj.labels.x_star, atol=1e-12)
    assert obj(x_opt) == pytest.approx(obj.f_star, abs=1e-8)


def test_evaluate_true_accepts_a_gradient_tracked_input(obj) -> None:
    """`.detach()` guards a `requires_grad=True` input: `Tensor.numpy()` raises outright on one
    that still requires grad, and the numpy round trip through `obj(...)` cannot participate in
    autograd regardless. Without `.detach()` this call would raise instead of returning a value.
    """
    X = torch.tensor(np.random.default_rng(2).uniform(size=(5, D)), dtype=torch.double, requires_grad=True)
    tf = SyntheticObjectiveTestFunction(obj)

    out = tf.evaluate_true(X)

    assert out.shape == (5,)
    np.testing.assert_allclose(out.detach().numpy(), obj(X.detach().numpy()), atol=1e-12, rtol=0)


def test_noise_std_perturbs_forward_with_the_given_scale(obj) -> None:
    """`noise_std=0.1` makes `forward` differ from `evaluate_true`, by roughly that much noise.

    Repeating one point 5000 times and checking the empirical residual std against 0.1 (rather
    than only checking "differs") also catches a mutant that wires in noise of the wrong scale or
    ignores the `noise_std` argument in favor of a hardcoded value.
    """
    torch.manual_seed(0)
    X = torch.full((5000, D), 0.5, dtype=torch.double)
    tf = SyntheticObjectiveTestFunction(obj, noise_std=0.1)

    true_val = tf.evaluate_true(X)
    noisy_val = tf.forward(X)
    residual = (noisy_val - true_val).detach().numpy()

    assert not np.allclose(residual, 0.0)
    assert 0.08 < residual.std() < 0.12
    # evaluate_true itself must stay exact -- the noise lives only in forward's own draw.
    assert torch.equal(tf.evaluate_true(X), true_val)
