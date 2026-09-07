"""Thin BoTorch adapter: wraps a `SyntheticObjective` as a `botorch.test_functions.SyntheticTestFunction`.

This is the **only** module in `synthobj` permitted to import `torch` or `botorch`. Nothing else in
the package (`kernel.py`, `draws.py`, `component.py`, `interaction.py`, `rotation.py`,
`objective.py`, `families.py`, `generate.py`) may gain such an import, and `synthobj/__init__.py`
does not import this module at package-import time, so `import synthobj` keeps working on a
machine with neither torch nor botorch installed. Import this module directly:
`import synthobj.botorch_adapter`.

## Maximization convention

This project's `f_star` (`SyntheticObjective.f_star`) is a **maximum**: the thesis measures simple
regret as `f_star - max_t f(x_t)`. BoTorch's `SyntheticTestFunction`/`BaseTestProblem` (checked
against the installed botorch 0.18.1) carry their own, independent sign conventions, established
here by reading `botorch/test_functions/base.py` and by direct experiment against two stock
functions (`Ackley`, `Branin`) rather than assumed:

- `BaseTestProblem.forward(X)` computes `f = self.evaluate_true(X)` (plus noise) and negates
  *that result* iff `self.negate`. `evaluate_true`/`_evaluate_true` never see `negate` at all --
  confirmed empirically: `Ackley(negate=True).evaluate_true(x)` and
  `Ackley(negate=False).evaluate_true(x)` are bit-identical, while their `forward(x)` differ by
  sign. Overriding `_evaluate_true` to also flip sign under `negate` would double-negate under
  `forward`; no stock botorch test function does that, and this adapter does not either.
- `SyntheticTestFunction.optimal_value` returns `-self._optimal_value if self.negate else
  self._optimal_value`. `_optimal_value` is therefore always stored in the same raw, un-negated
  frame `_evaluate_true` returns values in -- never pre-negated.
- Whether that raw optimum is a minimum or a maximum is `_is_minimization_by_default`, a class
  attribute most stock functions (`Ackley`, `Branin`) leave at its `True` default, but which
  `Cosine8` and `Labs` (both installed with botorch) set to `False` for a function whose raw
  `_evaluate_true` is naturally maximized -- e.g. `Cosine8._optimal_value = 0.8` is exactly
  `_evaluate_true`'s value at its stored maximizer, not a minimum. This project's `f_star` is
  exactly that shape, so `SyntheticObjectiveTestFunction` follows the `Cosine8`/`Labs` pattern:
  `_is_minimization_by_default = False`, `_optimal_value = obj.f_star`. There is no
  library/project convention conflict to reconcile -- botorch's own base class already
  accommodates a naturally-maximized problem, this project's objectives are exactly that shape,
  and two of botorch's own bundled test functions already use the same setting.

Net effect: with `negate=False` (the default), `_evaluate_true`, `evaluate_true` and `forward`
(net of any noise) all equal `obj(X)`; `optimal_value == obj.f_star`; and `is_minimization_problem`
is `False` -- identical to this project's own convention, with nothing to translate. With
`negate=True`, `forward`/`__call__` return `-obj(X)` and `optimal_value == -obj.f_star`
(`is_minimization_problem` becomes `True`), matching botorch's ordinary use of `negate` to make a
callable's maximization solve the original problem -- applied here to a function that is already a
maximization problem, so `negate=True` deliberately inverts it into a minimization one instead of
the usual direction. `_evaluate_true`/`evaluate_true` stay unaffected by `negate` either way.
"""
from __future__ import annotations

import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from torch import Tensor

from synthobj.objective import SyntheticObjective


class SyntheticObjectiveTestFunction(SyntheticTestFunction):
    """Wraps one `SyntheticObjective` as a BoTorch `SyntheticTestFunction` on `[0, 1]^D`.

    See the module docstring for the maximization convention `negate` and `optimal_value` follow.
    `obj.labels.x_star` is recorded as the sole known global maximizer (`_optimizers`); it is
    frequently on the cube boundary, which is fine here since this class never passes a custom
    `bounds=` to `super().__init__`, so `SyntheticTestFunction.__init__`'s optional
    optimizers-within-bounds check (only triggered when `bounds` is explicitly given) never runs.
    """

    _is_minimization_by_default = False

    def __init__(
        self,
        obj: SyntheticObjective,
        noise_std: float | None = None,
        negate: bool = False,
    ) -> None:
        self.obj = obj
        self.dim = obj.D
        self._bounds = [(0.0, 1.0)] * obj.D
        self.continuous_inds = list(range(obj.D))
        self._optimal_value = float(obj.f_star)
        self._optimizers = [tuple(float(x) for x in obj.labels.x_star)]
        super().__init__(noise_std=noise_std, negate=negate)

    def _evaluate_true(self, X: Tensor) -> Tensor:
        """`X` of shape `(..., D)` -> `obj(X)` evaluated in numpy, reshaped back to `(...)`.

        `evaluate_true` (the public, non-underscore method this overrides `_evaluate_true` for)
        has already validated `X`'s shape and bounds before this runs, so every point here is
        already known to lie in `[0, 1]^D`. `.detach()` guards the (uncommon, but legal) case of a
        gradient-tracked `X`: the numpy round trip below cannot participate in autograd regardless,
        and `Tensor.numpy()` raises outright on a tensor that still requires grad.
        """
        shape = X.shape[:-1]
        points = X.detach().double().cpu().numpy().reshape(-1, self.dim)
        values = self.obj(points)
        return torch.tensor(values, dtype=torch.float64, device=X.device).reshape(shape)
