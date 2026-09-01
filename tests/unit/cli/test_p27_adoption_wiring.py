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
