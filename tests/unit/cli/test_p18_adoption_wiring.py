def test_p18_adoption_wiring() -> None:
    from src.cli import CONVEXITY_ADOPTION_MODELS

    assert "P18" in CONVEXITY_ADOPTION_MODELS
    import inspect
    from src.cli import cmd_backtest

    source = inspect.getsource(cmd_backtest)
    assert "evaluate_p16_adoption_report" in source
    assert "P18" in source
