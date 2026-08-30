from __future__ import annotations

import pytest

from src.alpha.baselines import BASELINES


def test_p11_registered_in_baselines() -> None:
    assert "P11" in BASELINES
    model = BASELINES["P11"]()
    assert getattr(model, "name", None) == "P11"
    assert getattr(model, "scores_path_independent", False) is True
    assert hasattr(model, "score")
    assert hasattr(model, "allocate")


@pytest.mark.parametrize("scenario_id", ["test_p11_registered_in_baselines"])
def test_scenario_wrapper(scenario_id: str) -> None:
    test_p11_registered_in_baselines()
