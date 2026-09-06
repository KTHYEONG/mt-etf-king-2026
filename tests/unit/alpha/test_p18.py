def test_p18_registered_in_baselines() -> None:
    from pathlib import Path

    from src.strategies.registry import STRATEGIES as BASELINES
    from src.portfolio.constraints import load_rebalance_threshold
    from src.portfolio.policy import PortfolioPolicy

    assert "portfolio.convexity_variant" in BASELINES
    model = BASELINES["portfolio.convexity_variant"]()
    assert isinstance(model, PortfolioPolicy)
    assert getattr(model, "name", "") == "portfolio.convexity_variant"
    threshold = load_rebalance_threshold(Path("configs/portfolio.yaml"))
    assert float(getattr(model, "min_rebalance_delta", 0.0)) == float(threshold)
    assert getattr(model, "lottery_config") is not None  # noqa: B009
    assert getattr(model, "convexity_config") is not None  # noqa: B009
