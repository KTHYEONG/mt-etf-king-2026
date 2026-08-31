def test_p23_adoption_wiring() -> None:
    import inspect

    from src.cli import STICKY_ADOPTION_MODELS, cmd_backtest, cmd_decide

    assert "P23" in STICKY_ADOPTION_MODELS
    assert "P21" in STICKY_ADOPTION_MODELS
    assert "P22" in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(cmd_backtest)
    assert "P23" in bt
    assert "score_max_order_to_adv" in bt
    assert "locked_window_returns" in bt
    dec = inspect.getsource(cmd_decide)
    assert "peak_lock_active" in dec
    assert "P23" in dec or "split_residual_plus2" in dec
