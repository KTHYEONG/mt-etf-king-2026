def test_p24_adoption_wiring() -> None:
    import inspect
    from src.cli import STICKY_ADOPTION_MODELS, cmd_decide
    from src.cli.commands.backtest.families.sticky_mom60 import _hook_mom60_peak_lock
    from src.cli.commands.decide.models import _OVERLAY_HOOKS

    assert 'sticky.mom60_peak_lock' in STICKY_ADOPTION_MODELS
    assert 'sticky.impulse_crash' in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(_hook_mom60_peak_lock)
    assert 'championship_lock_returns' in bt
    assert 'evaluate_p24_adoption_gates' in bt
    assert 'overlay_param' in bt
    dec = inspect.getsource(cmd_decide)
    assert 'peak_lock_active' in dec
    assert 'overlay_param' in dec
    assert 'sticky.mom60_peak_lock' in _OVERLAY_HOOKS
