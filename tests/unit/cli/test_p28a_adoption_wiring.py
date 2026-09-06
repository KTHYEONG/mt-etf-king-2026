def test_p28a_cli_championship_wires_p27_champion() -> None:
    import inspect
    import re

    from src.cli import STICKY_ADOPTION_MODELS
    from src.cli.commands.backtest.families.sticky_cash import _hook_mom60_hold
    from src.cli.commands.decide.models import _OVERLAY_HOOKS

    assert "sticky.mom60_hold" in STICKY_ADOPTION_MODELS
    p28_src = inspect.getsource(_hook_mom60_hold)
    assert "evaluate_championship_adoption" in p28_src
    assert "sticky.mom60_raw" in p28_src
    assert "run_rolling" in p28_src
    assert "field_relative_report" in p28_src
    assert '"sticky.impulse_crash"' in p28_src or "'sticky.impulse_crash'" in p28_src
    assert 'getattr(rolling, "diagnostics"' in p28_src
    assert re.search(r"^\s*_ = diagnostics\b", p28_src, flags=re.M) is None
    assert "sticky.mom60_hold" in _OVERLAY_HOOKS
    dec = inspect.getsource(_OVERLAY_HOOKS["sticky.mom60_hold"])
    assert "overlay_should_cash" in dec
