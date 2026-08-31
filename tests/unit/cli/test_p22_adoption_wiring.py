def test_p22_adoption_wiring() -> None:
    import inspect
    from src.cli import STICKY_ADOPTION_MODELS, cmd_backtest, cmd_decide

    assert "P22" in STICKY_ADOPTION_MODELS
    assert "P21" in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(cmd_backtest)
    assert "P22" in bt
    assert "locked_window_returns" in bt
    assert "_p22_lock" in bt
    assert "lock_level" in bt
    dec = inspect.getsource(cmd_decide)
    assert "peak_lock_active" in dec
    assert "load_p22_lock_level" in dec
    assert "_p22_lock" in dec
