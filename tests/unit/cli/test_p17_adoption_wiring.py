def test_make_eval_control_model_resolves_p14_adoption_flags() -> None:
    from src.cli import _make_eval_control_model
    from src.tournament.simulator import model_requires_path_dependent

    p14 = _make_eval_control_model("P14", "adoption")

    assert p14.path_dependent is False
    assert p14.state_enabled is False
    assert model_requires_path_dependent(p14) is False


def test_p17_uses_convexity_preflight_and_adoption_wiring() -> None:
    import inspect

    from src.cli import CONVEXITY_ADOPTION_MODELS, cmd_backtest

    source = inspect.getsource(cmd_backtest)
    adoption_block = source.split("evaluate_p16_adoption_report", 1)[0]

    assert CONVEXITY_ADOPTION_MODELS == frozenset({"P16", "P17"})  # noqa: SIM300
    assert source.count("model_key in CONVEXITY_ADOPTION_MODELS") >= 2
    assert '_make_eval_control_model("P14", eval_mode)' in source
    assert "evaluate_p16_adoption_report" in source
    assert "b0_dist = dist" not in adoption_block
    assert 'adoption_gate model={model_key}' in source
