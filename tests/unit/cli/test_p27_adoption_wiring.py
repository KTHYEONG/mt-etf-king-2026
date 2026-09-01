def test_p27_adoption_wiring() -> None:
    import inspect

    from src.cli import STICKY_ADOPTION_MODELS, cmd_backtest, cmd_decide

    assert "P27" in STICKY_ADOPTION_MODELS
    assert "P26" in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(cmd_backtest)
    assert "P27" in bt
    assert "load_p27_exposure_limits" in bt
    assert "field_relative_report" in bt
    assert "oneshot_anchor_starts" in bt
    assert "oneshot_window_returns" in bt
    assert "evaluate_championship_adoption" in bt
    p27_idx = bt.find('if model_key == "P27"')
    assert p27_idx >= 0
    p27_src = bt[p27_idx:p27_idx + 8000]
    assert "rolling.returns" in p27_src
    assert "field_relative_report" in p27_src
    assert "oneshot_window_returns" in p27_src
    dec = inspect.getsource(cmd_decide)
    assert "P27" in dec
    assert "overlay_should_cash" in dec
    assert "load_p27_overlay_mode" in dec


def test_p27_decide_identity_overlay_does_not_force_cash() -> None:
    import inspect
    import re

    from src.cli import cmd_decide
    from src.tournament.policy import overlay_should_cash

    src = inspect.getsource(cmd_decide)
    assert "if _model_arg == \"P27\"" in src
    block = src.split("if _model_arg == \"P27\"", 1)[1]
    next_model = re.search(r"\nif _model_arg == \"P2[0-9]\"", block)
    if next_model is not None:
        block = block[: next_model.start()]
    assert "overlay_should_cash" in block
    assert "load_p27_overlay_mode" in block
    assert overlay_should_cash("identity", 0.99, 0, 0.50, 5) is False


def test_p27_cli_independent_window_eval_wiring() -> None:
    import inspect

    from src.cli import cmd_backtest

    bt = inspect.getsource(cmd_backtest)
    end_p27 = bt.find("if model_key in CONVEXITY_ADOPTION_MODELS")
    assert end_p27 > 0
    p27_idx = bt.rfind('if model_key == "P27"', 0, end_p27)
    p26_idx = bt.rfind('if model_key == "P26"', 0, p27_idx)
    p25_idx = bt.rfind('if model_key == "P25"', 0, p26_idx)
    assert p27_idx > 0 and p26_idx > 0 and p25_idx > 0
    p27_src = bt[p27_idx:end_p27]
    p26_src = bt[p26_idx:p27_idx]
    p25_src = bt[p25_idx:p26_idx]
    assert "oneshot_independent_window_returns" in p27_src
    assert "oneshot_independent_window_returns(" in p27_src
    assert "path_dependent=_b21_flags.path_dependent" in p27_src
    assert "path_dependent=False" not in p27_src
    assert "path_dependent=_b21_flags_p26.path_dependent" in p26_src
    assert "path_dependent=_b21_flags_p25.path_dependent" in p25_src


def test_p27_cli_incumbent_no_slow_override() -> None:
    import inspect

    from src.cli import cmd_backtest

    bt = inspect.getsource(cmd_backtest)
    end_p27 = bt.find("if model_key in CONVEXITY_ADOPTION_MODELS")
    assert end_p27 > 0
    p27_idx = bt.rfind('if model_key == "P27"', 0, end_p27)
    assert p27_idx > 0
    p27_src = bt[p27_idx:end_p27]
    assert "path_dependent_mode=('slow'" not in p27_src
    assert "path_dependent_mode=_path_mode" in p27_src or "resolve_path_dependent_mode" in p27_src


def test_sticky_shared_session_cache_wiring() -> None:
    import inspect

    from src.cli import cmd_backtest

    bt = inspect.getsource(cmd_backtest)
    assert "if _is_pd and _scores_pi:" not in bt
    assert "if _is_pd" in bt
    assert "session_cache=_shared_cache" in bt
