def test_resolve_strategy_id_maps_legacy_p27() -> None:
    from src.strategies.ids import STICKY_MOM60_RAW
    from src.strategies.registry import resolve_strategy_id

    assert resolve_strategy_id("P27") == STICKY_MOM60_RAW
    assert resolve_strategy_id(STICKY_MOM60_RAW) == STICKY_MOM60_RAW
    assert resolve_strategy_id("p27") == STICKY_MOM60_RAW

def test_legacy_aliases_cover_all_baselines_keys() -> None:
    from src.alpha.baselines import BASELINES
    from src.strategies.registry import LEGACY_ALIASES, STRATEGIES, resolve_strategy_id

    for legacy_key in BASELINES:
        semantic = resolve_strategy_id(legacy_key)
        assert semantic in STRATEGIES
        assert LEGACY_ALIASES[legacy_key] == semantic
        assert callable(STRATEGIES[semantic])

def test_strategy_model_name_uses_semantic_id() -> None:
    from src.strategies.ids import STICKY_MOM60_RAW
    from src.strategies.registry import STRATEGIES

    model = STRATEGIES[STICKY_MOM60_RAW]()
    # Champion path contract: P27 keeps its legacy factory name.
    assert getattr(model, "name") == "P27"

def test_baselines_proxy_accepts_legacy_and_semantic() -> None:
    from src.alpha.baselines import BASELINES
    from src.strategies.ids import STICKY_MOM60_RAW

    legacy = BASELINES["P27"]()
    semantic = BASELINES[STICKY_MOM60_RAW]()
    # Champion path contract: P27 keeps its legacy factory name.
    assert getattr(legacy, "name") == "P27"
    assert getattr(semantic, "name") == "P27"

def test_p27_behavior_unchanged_after_rename() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel, load_p27_overlay_mode
    from src.portfolio.constraints import load_p26_exposure_limits, load_p27_exposure_limits
    from src.strategies.ids import STICKY_MOM60_RAW

    p27 = BASELINES["P27"]()
    raw = BASELINES[STICKY_MOM60_RAW]()
    assert isinstance(p27, StickyLeaderModel)
    assert isinstance(raw, StickyLeaderModel)
    assert str(p27.config.mom_col) == "mom_60"
    assert str(raw.config.mom_col) == "mom_60"
    assert float(p27.config.min_gap) == 0.04
    assert int(p27.config.min_hold) == 2
    assert load_p27_overlay_mode() == "identity"
    assert load_p27_exposure_limits() == load_p26_exposure_limits()
    assert load_p27_exposure_limits() == (0.95, 1.90, 0.05)
