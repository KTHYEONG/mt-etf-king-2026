def test_p30_adoption_wiring() -> None:
    import inspect

    from src.cli import STICKY_ADOPTION_MODELS, cmd_backtest, cmd_decide
    from src.cli.commands.backtest import FAMILY_RUNNERS
    from src.cli.commands.backtest.families.sticky import run_sticky_family
    from src.cli.commands.backtest.families.sticky_house import EQUITY_GROUP_IDS, _hook_equity_group
    from src.cli.commands.decide.models import _OVERLAY_HOOKS
    from src.cli.constants import STICKY_ADOPTION_MODELS as SEMANTIC_STICKY_ADOPTION_MODELS
    from src.strategies.ids import STICKY_FILLABLE_MOM60

    assert "sticky.fillable_mom60" in STICKY_ADOPTION_MODELS
    assert STICKY_FILLABLE_MOM60 in SEMANTIC_STICKY_ADOPTION_MODELS
    assert STICKY_FILLABLE_MOM60 in EQUITY_GROUP_IDS
    assert FAMILY_RUNNERS["sticky"] is run_sticky_family
    assert STICKY_FILLABLE_MOM60 in _OVERLAY_HOOKS
    hook_src = inspect.getsource(_hook_equity_group)
    assert "oneshot_independent_window_returns" in hook_src
    assert "load_p27_exposure_limits" in hook_src
    assert "evaluate_championship_adoption" in hook_src
    _ = cmd_backtest
    dec = inspect.getsource(cmd_decide)
    decide_hook_src = inspect.getsource(_OVERLAY_HOOKS[STICKY_FILLABLE_MOM60])
    assert "sticky.fillable_mom60" in decide_hook_src
    assert "overlay_should_cash" in dec or "overlay_should_cash" in decide_hook_src
