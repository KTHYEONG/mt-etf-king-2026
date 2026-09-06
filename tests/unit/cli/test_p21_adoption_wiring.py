def test_p21_adoption_wiring() -> None:
    import inspect
    from src.cli import STICKY_ADOPTION_MODELS, cmd_decide
    from src.cli.commands.backtest.families.sticky_core import _hook_impulse_crash

    assert "sticky.impulse_crash" in STICKY_ADOPTION_MODELS
    assert "sticky.leader_base" in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(_hook_impulse_crash)
    assert "locked_window_returns" in bt
    dec = inspect.getsource(cmd_decide)
    assert "peak_lock_active" in dec
