def test_strategy_id_constant_registered() -> None:
    from src.strategies.ids import STICKY_P27_COMPLEMENT_SWITCH
    from src.strategies.registry import resolve_strategy_id

    assert STICKY_P27_COMPLEMENT_SWITCH == "sticky.p27_complement_switch"
    assert resolve_strategy_id(STICKY_P27_COMPLEMENT_SWITCH) == "sticky.p27_complement_switch"


def test_registry_builds_complement_switch() -> None:
    from src.strategies.ids import STICKY_P27_COMPLEMENT_SWITCH
    from src.strategies.registry import STRATEGIES
    from src.tournament.p27_complement_sleeve import P27ComplementHardSwitchModel

    factory = STRATEGIES[STICKY_P27_COMPLEMENT_SWITCH]
    model = factory()
    assert isinstance(model, P27ComplementHardSwitchModel)
    assert model.name == "sticky.p27_complement_switch"


def test_prep_engine_limits_include_complement_switch() -> None:
    from src.cli.commands.backtest import _prep

    assert _prep._ENGINE_LIMIT_LOADERS["sticky.p27_complement_switch"] == "p27"
    assert "sticky.p27_complement_switch" in _prep._EXPOSURE_LIMIT_IDS
