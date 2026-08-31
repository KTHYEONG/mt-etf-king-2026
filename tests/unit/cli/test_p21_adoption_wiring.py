def test_p21_adoption_wiring() -> None:
    import inspect
    from src.cli import STICKY_ADOPTION_MODELS, cmd_backtest, cmd_decide

    assert "P21" in STICKY_ADOPTION_MODELS
    assert "P20" in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(cmd_backtest)
    assert "locked_window_returns" in bt
    assert "P21" in bt
    dec = inspect.getsource(cmd_decide)
    assert "peak_lock_active" in dec
