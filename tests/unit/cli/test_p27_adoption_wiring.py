def test_p27_adoption_wiring() -> None:
    import inspect

    from src.cli import STICKY_ADOPTION_MODELS
    from src.cli.commands.backtest.families.sticky_champion import _hook_mom60_raw
    from src.cli.commands.decide.models import _OVERLAY_HOOKS

    assert "sticky.mom60_raw" in STICKY_ADOPTION_MODELS
    assert "sticky.mom60_concentrated" in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(_hook_mom60_raw)
    assert "load_p27_exposure_limits" in bt
    assert "field_relative_report" in bt
    assert "oneshot_anchor_starts" in bt
    assert "oneshot_window_returns" in bt
    assert "evaluate_championship_adoption" in bt
    assert "rolling.returns" in bt
    dec = inspect.getsource(_OVERLAY_HOOKS["sticky.mom60_raw"])
    assert "overlay_should_cash" in dec
    assert "load_overlay_mode" in dec


def test_p27_decide_identity_overlay_does_not_force_cash() -> None:
    import inspect

    from src.cli.commands.decide.models import _OVERLAY_HOOKS
    from src.tournament.policy import overlay_should_cash

    src = inspect.getsource(_OVERLAY_HOOKS["sticky.mom60_raw"])
    assert "overlay_should_cash" in src
    assert "load_overlay_mode" in src
    assert overlay_should_cash("identity", 0.99, 0, 0.50, 5) is False


def test_p27_cli_independent_window_eval_wiring() -> None:
    import inspect

    from src.cli.commands.backtest.families.sticky_champion import _hook_mom60_raw
    from src.cli.commands.backtest.families.sticky_house import _hook_house_money
    from src.cli.commands.backtest.families.sticky_mom60 import _hook_mom60_concentrated

    p27_src = inspect.getsource(_hook_mom60_raw)
    p26_src = inspect.getsource(_hook_mom60_concentrated)
    p25_src = inspect.getsource(_hook_house_money)
    assert "oneshot_independent_window_returns" in p27_src
    assert "oneshot_independent_window_returns(" in p27_src
    assert "path_dependent=_b21_flags.path_dependent" in p27_src
    assert "path_dependent=False" not in p27_src
    assert "path_dependent=_b21_flags_p26.path_dependent" in p26_src
    assert "path_dependent=_b21_flags_p25.path_dependent" in p25_src


def test_p27_cli_incumbent_no_slow_override() -> None:
    import inspect

    from src.cli.commands.backtest.families.sticky_champion import _hook_mom60_raw

    p27_src = inspect.getsource(_hook_mom60_raw)
    assert "path_dependent_mode=('slow'" not in p27_src
    assert "path_dependent_mode=_path_mode" in p27_src or "resolve_path_dependent_mode" in p27_src
    assert "exposure_limits=_p21_alpha_limits_p27" in p27_src
    assert "resolve_exposure_limits_for_model" in p27_src


def test_sticky_shared_session_cache_wiring() -> None:
    import inspect

    import src.cli.commands.backtest._core as _core
    from src.cli.commands.backtest import _prep

    core_src = inspect.getsource(_core.run_cells) + inspect.getsource(_prep.prepare_run)
    assert "is_pd" in core_src
    assert "session_cache=shared_cache" in core_src


def test_p27_cli_gross_diagnostics_avoids_undefined_name() -> None:
    import inspect
    import re

    from src.cli import STICKY_ADOPTION_MODELS
    from src.cli.commands.backtest.families.sticky_champion import _hook_mom60_raw

    p27_src = inspect.getsource(_hook_mom60_raw)
    assert 'getattr(rolling, "diagnostics"' in p27_src
    assert re.search(r"^\s*_ = diagnostics\b", p27_src, flags=re.M) is None
    assert "gross_violation_count" in p27_src
    assert "sticky.mom60_raw" in STICKY_ADOPTION_MODELS
