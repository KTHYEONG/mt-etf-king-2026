from __future__ import annotations


def test_p31_in_sticky_adoption_models() -> None:
    from src.cli._impl import STICKY_ADOPTION_MODELS

    assert "P31" in STICKY_ADOPTION_MODELS
    assert "P30" in STICKY_ADOPTION_MODELS


def test_p31_membership_uses_p27_exposure() -> None:
    from src.alpha.baselines import BASELINES
    from src.portfolio.constraints import (
        load_p27_exposure_limits,
        resolve_exposure_limits_for_model,
    )

    assert "P31" in BASELINES
    assert resolve_exposure_limits_for_model("P31", comparison_mode="full_strategy_own") == load_p27_exposure_limits()
