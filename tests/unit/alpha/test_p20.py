def test_p20_registered_in_baselines() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel

    assert "P20" in BASELINES
    model = BASELINES["P20"]()
    assert isinstance(model, StickyLeaderModel)
    assert getattr(model, "name", "") == "P20"
    cfg = getattr(model, "config")
    assert cfg.only_plus_2 is True
    assert cfg.no_inverse is True
    assert float(cfg.min_gap) == 0.08
    assert int(cfg.min_hold) == 3
    assert not hasattr(model, "allocate") or not callable(getattr(model, "allocate", None))


def test_filter_plus2_scores_keeps_only_plus2() -> None:
    import polars as pl
    from src.alpha.sticky import StickyLeaderConfig, filter_plus2_scores

    snap = pl.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "name": ["KODEX 200", "KODEX 레버리지", "KODEX 인버스2X"],
            "mom_20": [0.10, 0.18, 0.25],
        }
    )
    out = filter_plus2_scores(snap, StickyLeaderConfig())
    assert out == {"B": 0.18}


def test_apply_sticky_leader_holds_within_gap() -> None:
    from src.alpha.sticky import StickyLeaderConfig, apply_sticky_leader
    from src.portfolio.sizing import SizingScheme, weights_from_scores

    cfg = StickyLeaderConfig(min_gap=0.08, min_hold=0)
    scores = {"HOLD": 0.10, "CHAL": 0.15}
    out = apply_sticky_leader(scores, "HOLD", cfg, hold_len=10)
    w = weights_from_scores(out, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {"HOLD"}
    assert scores["HOLD"] == 0.10


def test_apply_sticky_leader_switches_when_gap_exceeded() -> None:
    from src.alpha.sticky import StickyLeaderConfig, apply_sticky_leader
    from src.portfolio.sizing import SizingScheme, weights_from_scores

    cfg = StickyLeaderConfig(min_gap=0.08, min_hold=0)
    scores = {"HOLD": 0.10, "CHAL": 0.20}
    out = apply_sticky_leader(scores, "HOLD", cfg, hold_len=10)
    w = weights_from_scores(out, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {"CHAL"}


def test_apply_sticky_leader_min_hold_blocks_switch() -> None:
    from src.alpha.sticky import StickyLeaderConfig, apply_sticky_leader
    from src.portfolio.sizing import SizingScheme, weights_from_scores

    cfg = StickyLeaderConfig(min_gap=0.08, min_hold=3)
    scores = {"HOLD": 0.10, "CHAL": 0.50}
    out = apply_sticky_leader(scores, "HOLD", cfg, hold_len=2)
    w = weights_from_scores(out, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {"HOLD"}
    out2 = apply_sticky_leader(scores, "HOLD", cfg, hold_len=3)
    w2 = weights_from_scores(out2, SizingScheme.TOP1, k=1)
    assert set(w2.keys()) == {"CHAL"}


def test_apply_sticky_leader_fail_closed_empty() -> None:
    import polars as pl
    from src.alpha.sticky import StickyLeaderConfig, apply_sticky_leader, filter_plus2_scores

    cfg = StickyLeaderConfig()
    assert apply_sticky_leader({}, "X", cfg, 1) == {}
    assert filter_plus2_scores(pl.DataFrame(), cfg) == {}


def test_sticky_leader_config_from_yaml_fail_closed() -> None:
    from src.alpha.sticky import StickyLeaderConfig

    bad = StickyLeaderConfig.from_yaml("nope")  # type: ignore[arg-type]
    assert bad.min_gap == 0.08
    assert bad.min_hold == 3
    nan = StickyLeaderConfig.from_yaml({"min_gap": float("nan"), "min_hold": -2})
    assert nan.min_gap == 0.08
    assert nan.min_hold == 3
    ok = StickyLeaderConfig.from_yaml({"min_gap": 0.04, "min_hold": 5, "only_plus_2": True})
    assert ok.min_gap == 0.04
    assert ok.min_hold == 5
