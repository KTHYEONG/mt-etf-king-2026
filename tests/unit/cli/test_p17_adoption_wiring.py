def test_make_eval_control_model_resolves_p14_adoption_flags() -> None:
    from src.cli import _make_eval_control_model
    from src.tournament.simulator import model_requires_path_dependent

    p14 = _make_eval_control_model("portfolio.lottery_exposure", "adoption")

    assert p14.path_dependent is False
    assert p14.state_enabled is False
    assert model_requires_path_dependent(p14) is False


def test_p17_uses_convexity_preflight_and_adoption_wiring() -> None:
    import inspect

    import src.cli.commands.backtest._core as _core_mod
    from src.cli import CONVEXITY_ADOPTION_MODELS
    from src.cli.commands.backtest.families.portfolio import _hook_convexity, _hook_lottery_rebalance

    source = inspect.getsource(_hook_convexity)
    adoption_block = source.split("evaluate_p16_adoption_report", 1)[0]

    assert CONVEXITY_ADOPTION_MODELS == frozenset({"portfolio.convexity_hold", "portfolio.convexity_rebalance", "portfolio.convexity_variant"})  # noqa: SIM300
    assert "CONVEXITY_ADOPTION_MODELS" in inspect.getsource(_core_mod)
    assert '_make_eval_control_model("portfolio.lottery_exposure", eval_mode)' in inspect.getsource(_hook_lottery_rebalance)
    assert "evaluate_p16_adoption_report" in source
    assert "b0_dist = dist" not in adoption_block
    assert 'adoption_gate model={model_key}' in source
