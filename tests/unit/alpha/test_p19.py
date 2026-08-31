def test_p19_registered_in_baselines() -> None:
    from pathlib import Path

    from src.alpha.baselines import BASELINES
    from src.portfolio.constraints import load_rebalance_threshold

    assert "P19" in BASELINES
    model = BASELINES["P19"]()
    assert getattr(model, "name", "") == "P19"
    threshold = load_rebalance_threshold(Path("configs/portfolio.yaml"))
    assert float(getattr(model, "min_rebalance_delta", 0.0)) == float(threshold)
    assert threshold > 0.0
    assert getattr(model, "lottery_config") is not None
    assert getattr(model, "lottery_config").enabled is True
    assert getattr(model, "convexity_config", "UNSET") is None

def test_p19_is_p14_plus_deadband_not_p16() -> None:
    import inspect

    from src.alpha import baselines as mod
    from src.alpha.baselines import BASELINES

    src = inspect.getsource(mod._make_p19)
    assert "_make_p14" in src
    assert "_make_p16" not in src
    assert "ConvexityHoldConfig" not in src
    p14 = BASELINES["P14"]()
    p19 = BASELINES["P19"]()
    assert p14.name == "P14"
    assert p19.name == "P19"
    assert float(p19.min_rebalance_delta) > float(getattr(p14, "min_rebalance_delta", 0.0))
