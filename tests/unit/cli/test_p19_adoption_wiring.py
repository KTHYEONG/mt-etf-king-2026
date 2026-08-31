def test_lottery_adoption_models_includes_p19() -> None:
    import inspect

    from src.cli import LOTTERY_ADOPTION_MODELS, cmd_backtest

    assert LOTTERY_ADOPTION_MODELS == frozenset({"P14", "P19"})
    source = inspect.getsource(cmd_backtest)
    assert "LOTTERY_ADOPTION_MODELS" in source
    assert "evaluate_adoption_gates" in source
    assert '_make_eval_control_model("P14"' in source or "_make_eval_control_model(\"P14\"" in source
