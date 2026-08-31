def test_p24_adoption_wiring() -> None:
    import inspect
    from src.cli import STICKY_ADOPTION_MODELS, cmd_backtest, cmd_decide

    assert 'P24' in STICKY_ADOPTION_MODELS
    assert 'P21' in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(cmd_backtest)
    assert 'P24' in bt
    assert 'championship_lock_returns' in bt
    assert 'evaluate_p24_adoption_gates' in bt
    assert 'load_p24_lock_level' in bt or 'load_p24_trail' in bt
    dec = inspect.getsource(cmd_decide)
    assert 'peak_lock_active' in dec
    assert 'load_p24_lock_level' in dec
