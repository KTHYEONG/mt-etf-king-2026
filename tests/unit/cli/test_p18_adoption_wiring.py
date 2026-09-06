def test_p18_adoption_wiring() -> None:
    import inspect
    from src.cli import CONVEXITY_ADOPTION_MODELS
    from src.cli.commands.backtest.families.portfolio import _hook_convexity

    assert "portfolio.convexity_variant" in CONVEXITY_ADOPTION_MODELS
    source = inspect.getsource(_hook_convexity)
    assert "evaluate_p16_adoption_report" in source
    assert "portfolio.convexity_variant" in source or "convexity" in source
