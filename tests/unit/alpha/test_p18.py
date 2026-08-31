def test_p18_registered_in_baselines() -> None:
    from pathlib import Path

    from src.alpha.baselines import BASELINES
    from src.portfolio.constraints import load_rebalance_threshold
    from src.portfolio.policy import PortfolioPolicy

    assert "P18" in BASELINES
    model = BASELINES["P18"]()
    assert isinstance(model, PortfolioPolicy)
    assert getattr(model, "name", "") == "P18"
    threshold = load_rebalance_threshold(Path("configs/portfolio.yaml"))
    assert float(getattr(model, "min_rebalance_delta", 0.0)) == float(threshold)
    assert getattr(model, "lottery_config") is not None  # noqa: B009
    assert getattr(model, "convexity_config") is not None  # noqa: B009
