from __future__ import annotations

import pytest

from src.strategies.registry import STRATEGIES as BASELINES


def test_p11_registered_in_baselines() -> None:
    assert "portfolio.momentum_confidence" in BASELINES
    model = BASELINES["portfolio.momentum_confidence"]()
    assert getattr(model, "name", None) == "portfolio.momentum_confidence"
    assert getattr(model, "scores_path_independent", False) is True
    assert hasattr(model, "score")
    assert hasattr(model, "allocate")


@pytest.mark.parametrize("scenario_id", ["test_p11_registered_in_baselines"])
def test_scenario_wrapper(scenario_id: str) -> None:
    test_p11_registered_in_baselines()
