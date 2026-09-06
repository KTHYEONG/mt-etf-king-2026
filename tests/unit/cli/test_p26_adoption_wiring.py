def test_p26_adoption_wiring() -> None:
    import inspect

    from src.cli import STICKY_ADOPTION_MODELS
    from src.cli.commands.backtest import _prep
    from src.cli.commands.backtest.families.sticky_mom60 import _hook_mom60_concentrated
    from src.cli.commands.decide.models import _OVERLAY_HOOKS

    assert "sticky.mom60_concentrated" in STICKY_ADOPTION_MODELS
    assert "sticky.house_money" in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(_hook_mom60_concentrated)
    assert "load_p26_exposure_limits" in bt
    assert "overlay_param" in bt
    assert "evaluate_championship_adoption" in bt
    assert "execution_faithful_late_lock_returns" in bt
    assert "set_portfolio_exposure_limits" in inspect.getsource(_prep.prepare_run)
    dec = inspect.getsource(_OVERLAY_HOOKS["sticky.mom60_concentrated"])
    assert "overlay_param" in dec
    assert "house_money_should_cash" in dec
