def test_p20_adoption_wiring() -> None:
    import inspect
    from src.cli import STICKY_ADOPTION_MODELS, cmd_backtest

    assert {"P20"}.issubset(STICKY_ADOPTION_MODELS)
    assert "P21" in STICKY_ADOPTION_MODELS
    source = inspect.getsource(cmd_backtest)
    assert "STICKY_ADOPTION_MODELS" in source
    assert "evaluate_adoption_gates" in source
    assert "evaluate_objective_gates" in source
    assert "objective_gate_status" in source
    assert '_make_eval_control_model("B0"' in source or "_make_eval_control_model(\"B0\"" in source
    assert '_make_eval_control_model("B1"' in source or "_make_eval_control_model(\"B1\"" in source
