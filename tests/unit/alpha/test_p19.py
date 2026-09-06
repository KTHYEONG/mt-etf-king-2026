def test_p19_registered_in_baselines() -> None:
    from pathlib import Path

    from src.strategies.registry import STRATEGIES as BASELINES
    from src.portfolio.constraints import load_rebalance_threshold

    assert "portfolio.lottery_rebalance" in BASELINES
    model = BASELINES["portfolio.lottery_rebalance"]()
    assert getattr(model, "name", "") == "portfolio.lottery_rebalance"
    threshold = load_rebalance_threshold(Path("configs/portfolio.yaml"))
    assert float(getattr(model, "min_rebalance_delta", 0.0)) == float(threshold)
    assert threshold > 0.0
    assert getattr(model, "lottery_config") is not None
    assert getattr(model, "lottery_config").enabled is True
    assert getattr(model, "convexity_config", "UNSET") is None

def test_p19_is_p14_plus_deadband_not_p16() -> None:
    import inspect

    from src.portfolio import builders_convexity as mod
    from src.strategies.registry import STRATEGIES as BASELINES

    src = inspect.getsource(mod.make_portfolio_lottery_rebalance)
    assert "make_portfolio_lottery_exposure" in src
    assert "make_portfolio_convexity_hold" not in src
    assert "ConvexityHoldConfig" not in src
    p14 = BASELINES["portfolio.lottery_exposure"]()
    p19 = BASELINES["portfolio.lottery_rebalance"]()
    assert p14.name == "portfolio.lottery_exposure"
    assert p19.name == "portfolio.lottery_rebalance"
    assert float(p19.min_rebalance_delta) > float(getattr(p14, "min_rebalance_delta", 0.0))
