def test_p28a_cli_championship_wires_p27_champion() -> None:
    import inspect
    import re

    from src.cli import STICKY_ADOPTION_MODELS, cmd_backtest, cmd_decide

    assert "P28A" in STICKY_ADOPTION_MODELS
    bt = inspect.getsource(cmd_backtest)
    assert 'if model_key == "P28A"' in bt
    end_block = bt.find("if model_key in CONVEXITY_ADOPTION_MODELS")
    assert end_block > 0
    p28_idx = bt.rfind('if model_key == "P28A"', 0, end_block)
    assert p28_idx > 0
    p28_src = bt[p28_idx:end_block]
    assert "evaluate_championship_adoption" in p28_src
    assert 'BASELINES["P27"]' in p28_src or "BASELINES['P27']" in p28_src
    assert "run_rolling" in p28_src
    assert "field_relative_report" in p28_src
    assert '"P21"' in p28_src or "'P21'" in p28_src
    assert 'getattr(rolling, "diagnostics"' in p28_src
    assert re.search(r"^\s*_ = diagnostics\b", p28_src, flags=re.M) is None
    assert 'model_key in ("P21", "P26", "P27", "P28A")' in bt
    dec = inspect.getsource(cmd_decide)
    assert 'if _model_arg == "P28A"' in dec
    assert "overlay_should_cash" in dec
