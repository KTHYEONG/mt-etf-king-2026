def test_p20_adoption_wiring() -> None:
    import inspect
    from src.cli import STICKY_ADOPTION_MODELS
    from src.cli.commands.backtest.families.sticky_locks import _hook_leader_base

    assert {"sticky.leader_base"}.issubset(STICKY_ADOPTION_MODELS)
    assert "sticky.impulse_crash" in STICKY_ADOPTION_MODELS
    import src.cli.commands.backtest._core as _core_mod

    assert "STICKY_ADOPTION_MODELS" in inspect.getsource(_core_mod)
    source = inspect.getsource(_hook_leader_base)
    assert "evaluate_adoption_gates" in source
    assert "evaluate_objective_gates" in source
    assert "objective_gate_status" in source
    assert '_make_eval_control_model("baseline.buy_hold"' in source
    assert '_make_eval_control_model("baseline.mom20_top1"' in source
