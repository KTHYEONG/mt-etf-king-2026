def test_p22_adoption_wiring() -> None:
    import inspect
    from src.cli import STICKY_ADOPTION_MODELS, cmd_decide
    from src.cli.commands.backtest.families.sticky_locks import _hook_family_peak_lock
    from src.cli.commands.decide import cmd_decide as _decide_fn

    assert "sticky.family_peak_lock" in STICKY_ADOPTION_MODELS
    assert "sticky.impulse_crash" in STICKY_ADOPTION_MODELS
    _ = _decide_fn
    bt = inspect.getsource(_hook_family_peak_lock)
    assert "locked_window_returns" in bt
    assert "_p22_lock" in bt
    assert "lock_level" in bt
    dec = inspect.getsource(cmd_decide)
    assert "peak_lock_active" in dec
    assert "overlay_param" in dec
    assert "_p22_lock" in dec
