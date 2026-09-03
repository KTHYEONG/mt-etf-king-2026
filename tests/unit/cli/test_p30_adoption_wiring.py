def test_p30_adoption_wiring() -> None:
    import inspect

    from src.cli import STICKY_ADOPTION_MODELS, cmd_backtest, cmd_decide
    from src.cli.constants import STICKY_ADOPTION_MODELS as SEMANTIC_STICKY_ADOPTION_MODELS
    from src.cli.dispatch import STICKY_BACKTEST_HANDLERS, STICKY_DECIDE_HANDLERS
    from src.strategies.ids import STICKY_FILLABLE_MOM60

    assert "P30" in STICKY_ADOPTION_MODELS
    assert STICKY_FILLABLE_MOM60 in SEMANTIC_STICKY_ADOPTION_MODELS
    assert STICKY_FILLABLE_MOM60 in STICKY_DECIDE_HANDLERS
    assert STICKY_FILLABLE_MOM60 in STICKY_BACKTEST_HANDLERS
    bt = inspect.getsource(cmd_backtest)
    assert '("P29", "P29V", "P30")' in bt or "('P29', 'P29V', 'P30')" in bt
    assert "oneshot_independent_window_returns" in bt
    assert "load_p27_exposure_limits" in bt
    assert "evaluate_championship_adoption" in bt
    dec = inspect.getsource(cmd_decide)
    assert "P30" in dec
    assert "overlay_should_cash" in dec
