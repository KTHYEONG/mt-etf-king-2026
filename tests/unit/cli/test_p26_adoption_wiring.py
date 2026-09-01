def test_p26_adoption_wiring() -> None:
    import inspect

    from src.cli import STICKY_ADOPTION_MODELS, cmd_backtest, cmd_decide

    assert "P26" in STICKY_ADOPTION_MODELS
    assert "P25" in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(cmd_backtest)
    assert "P26" in bt
    assert "load_p26_exposure_limits" in bt
    assert "load_p26_arm" in bt
    assert "load_p26_lock_remaining" in bt
    assert "evaluate_championship_adoption" in bt
    assert "execution_faithful_late_lock_returns" in bt
    assert "set_portfolio_exposure_limits" in bt
    dec = inspect.getsource(cmd_decide)
    assert "load_p26_arm" in dec
    assert "house_money_should_cash" in dec
