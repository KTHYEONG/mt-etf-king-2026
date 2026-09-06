def test_p23_adoption_wiring() -> None:
    import inspect

    from src.cli import STICKY_ADOPTION_MODELS, cmd_decide
    from src.cli.commands.backtest import _core
    from src.cli.commands.backtest.families.sticky_split import _hook_split_fill_lock
    from src.cli.commands.decide.models import _ALLOCATE_HOOKS, _OVERLAY_HOOKS

    assert "sticky.split_fill_lock" in STICKY_ADOPTION_MODELS
    assert "sticky.impulse_crash" in STICKY_ADOPTION_MODELS
    assert "sticky.family_peak_lock" in STICKY_ADOPTION_MODELS
    core_src = inspect.getsource(_core.run_cells)
    assert "score_max_order_to_adv" in core_src
    bt = inspect.getsource(_hook_split_fill_lock)
    assert "locked_window_returns" in bt
    dec = inspect.getsource(cmd_decide)
    assert "peak_lock_active" in dec
    assert "sticky.split_fill_lock" in _OVERLAY_HOOKS
    assert "sticky.split_fill_lock" in _ALLOCATE_HOOKS
    assert "p23_model" in inspect.getsource(_ALLOCATE_HOOKS["sticky.split_fill_lock"])
