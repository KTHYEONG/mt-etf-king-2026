"""SCENARIO-06-03 SCENARIO-06-04"""
from __future__ import annotations

import pytest

from src.portfolio.constraints import WeightViolationError, normalize_weights
from src.portfolio.sizing import SizingScheme, weights_from_scores


def test_SCENARIO_06_03_weights_from_scores() -> None:
    """SCENARIO-06-03"""
    scores = {"A": 0.9, "B": 0.5, "C": 0.1}
    assert weights_from_scores(scores, SizingScheme.TOP3_50_30_20) == {"A": 0.5, "B": 0.3, "C": 0.2}
    assert weights_from_scores(scores, SizingScheme.TOP1) == {"A": 1.0}

    tied = weights_from_scores({"B": 0.5, "A": 0.5}, SizingScheme.TOP2_70_30)
    assert tied == {"A": 0.7, "B": 0.3}

    shortfall = weights_from_scores({"A": 0.9, "B": 0.5}, SizingScheme.TOP3_50_30_20)
    assert shortfall == {"A": 0.5, "B": 0.3}


def test_SCENARIO_06_04_normalize_weights() -> None:
    """SCENARIO-06-04"""
    ok = normalize_weights({"A": 0.6, "B": 0.3})
    assert abs(sum(ok.values()) - 0.9) < 1e-6
    assert abs(1.0 - sum(ok.values())) > 0.0

    with pytest.raises(WeightViolationError):
        normalize_weights({"A": 0.7, "B": 0.5})

    with pytest.raises(WeightViolationError):
        normalize_weights({"A": -0.1})

    with pytest.raises(WeightViolationError):
        normalize_weights({"A": 0.8}, max_weight=0.5)
