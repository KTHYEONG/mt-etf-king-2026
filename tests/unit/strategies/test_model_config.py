"""Facade fidelity for the sticky model config split (P5)."""

from __future__ import annotations


def test_model_config_canonical_home() -> None:
    import src.strategies.sticky.model as facade
    from src.strategies.sticky.model_config import DEFAULT_EXCLUDE_NAME_TOKENS, StickyLeaderConfig

    assert facade.StickyLeaderConfig is StickyLeaderConfig
    assert facade.DEFAULT_EXCLUDE_NAME_TOKENS is DEFAULT_EXCLUDE_NAME_TOKENS
    assert StickyLeaderConfig.__module__ == "src.strategies.sticky.model_config"


def test_sticky_config_parses_inactive_participation_fields() -> None:
    from src.strategies.sticky.model_config import StickyLeaderConfig

    # 기본값
    base = StickyLeaderConfig()
    assert base.inactive_participation is False
    assert base.inactive_score_col == "rv_20"
    assert base.inactive_min_weight == 0.30
    assert base.inactive_stop_drawdown == 0.15

    # 유효값 채택
    cfg = StickyLeaderConfig.from_yaml(
        {
            "inactive_participation": True,
            "inactive_score_col": "rv_5",
            "inactive_min_weight": 0.50,
            "inactive_stop_drawdown": 0.20,
        }
    )
    assert cfg.inactive_participation is True
    assert cfg.inactive_score_col == "rv_5"
    assert cfg.inactive_min_weight == 0.50
    assert cfg.inactive_stop_drawdown == 0.20

    # 무효값 -> 기본값 fail-closed (bool이 아닌 truthy는 강제 변환하지 않는다)
    bad = StickyLeaderConfig.from_yaml(
        {
            "inactive_participation": "yes",
            "inactive_score_col": "",
            "inactive_min_weight": 0.0,
            "inactive_stop_drawdown": 1.5,
        }
    )
    assert bad.inactive_participation is False
    assert bad.inactive_score_col == "rv_20"
    assert bad.inactive_min_weight == 0.30
    assert bad.inactive_stop_drawdown == 0.15

    worse = StickyLeaderConfig.from_yaml(
        {"inactive_min_weight": float("nan"), "inactive_stop_drawdown": -0.1}
    )
    assert worse.inactive_min_weight == 0.30
    assert worse.inactive_stop_drawdown == 0.15

