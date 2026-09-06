def test_lottery_adoption_models_includes_p19() -> None:
    import inspect

    import src.cli.commands.backtest._core as _core_mod
    from src.cli import LOTTERY_ADOPTION_MODELS
    from src.cli.commands.backtest.families.portfolio import _hook_lottery_rebalance

    assert LOTTERY_ADOPTION_MODELS == frozenset({"portfolio.lottery_exposure", "portfolio.lottery_rebalance"})
    source = inspect.getsource(_hook_lottery_rebalance)
    assert "LOTTERY_ADOPTION_MODELS" in inspect.getsource(_core_mod)
    assert "evaluate_adoption_gates" in source
    assert '_make_eval_control_model("portfolio.lottery_exposure"' in source
