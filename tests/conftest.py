from pathlib import Path

import pytest

# NOTE: the `small_objective(variant, seed)` fixture described in the plan
# (cached per session at D=20) depends on `synthobj.make_family`, which does
# not exist until a later task. Adding it here would break test collection
# for every task in between, so it is added once `make_family` lands.


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent
