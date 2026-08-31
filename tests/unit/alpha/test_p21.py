def test_p21_registered_in_baselines() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel

    assert "P21" in BASELINES
    p21 = BASELINES["P21"]()
    assert isinstance(p21, StickyLeaderModel)
    assert getattr(p21, "name", "") == "P21"
    cfg = getattr(p21, "config")
    assert float(cfg.impulse_gap) == 0.04
    assert cfg.impulse_require_volx is True
    assert float(cfg.cash_drawdown) == -0.12
    assert not hasattr(p21, "allocate") or not callable(getattr(p21, "allocate", None))
    p20 = BASELINES["P20"]()
    p20_cfg = getattr(p20, "config")
    assert float(p20_cfg.impulse_gap) == 0.0
    assert float(p20_cfg.cash_drawdown) == 0.0


def test_apply_impulse_switch_overrides_when_gap_and_volx() -> None:
    import polars as pl
    from src.alpha.sticky import StickyLeaderConfig, apply_impulse_switch
    from src.portfolio.sizing import SizingScheme, weights_from_scores

    cfg = StickyLeaderConfig(impulse_gap=0.04, impulse_require_volx=True)
    scores = {"HOLD": 0.20, "CHAL": 0.16}
    snap = pl.DataFrame(
        {
            "ticker": ["HOLD", "CHAL"],
            "mom_5": [0.01, 0.06],
            "volume_expansion": [0.1, 0.2],
        }
    )
    out = apply_impulse_switch(scores, "HOLD", snap, cfg)
    w = weights_from_scores(out, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {"CHAL"}
    assert scores["HOLD"] == 0.20


def test_apply_impulse_switch_blocked_without_volx() -> None:
    import polars as pl
    from src.alpha.sticky import StickyLeaderConfig, apply_impulse_switch
    from src.portfolio.sizing import SizingScheme, weights_from_scores

    cfg = StickyLeaderConfig(impulse_gap=0.04, impulse_require_volx=True)
    scores = {"HOLD": 0.20, "CHAL": 0.16}
    snap = pl.DataFrame(
        {
            "ticker": ["HOLD", "CHAL"],
            "mom_5": [0.01, 0.10],
            "volume_expansion": [0.1, 0.0],
        }
    )
    out = apply_impulse_switch(scores, "HOLD", snap, cfg)
    w = weights_from_scores(out, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {"HOLD"}


def test_apply_impulse_switch_requires_positive_mom5() -> None:
    import polars as pl
    from src.alpha.sticky import StickyLeaderConfig, apply_impulse_switch
    from src.portfolio.sizing import SizingScheme, weights_from_scores

    cfg = StickyLeaderConfig(impulse_gap=0.04, impulse_require_volx=True)
    scores = {"HOLD": 0.20, "CHAL": 0.16}
    snap = pl.DataFrame(
        {
            "ticker": ["HOLD", "CHAL"],
            "mom_5": [0.10, -0.01],
            "volume_expansion": [0.1, 0.5],
        }
    )
    out = apply_impulse_switch(scores, "HOLD", snap, cfg)
    w = weights_from_scores(out, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {"HOLD"}


def test_apply_impulse_switch_disabled_when_gap_zero() -> None:
    import polars as pl
    from src.alpha.sticky import StickyLeaderConfig, apply_impulse_switch
    from src.portfolio.sizing import SizingScheme, weights_from_scores

    cfg = StickyLeaderConfig(impulse_gap=0.0, impulse_require_volx=True)
    scores = {"HOLD": 0.20, "CHAL": 0.16}
    snap = pl.DataFrame(
        {
            "ticker": ["HOLD", "CHAL"],
            "mom_5": [0.01, 0.20],
            "volume_expansion": [0.1, 0.5],
        }
    )
    out = apply_impulse_switch(scores, "HOLD", snap, cfg)
    w = weights_from_scores(out, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {"HOLD"}


def test_apply_crash_cash_empties_on_drawdown() -> None:
    import polars as pl
    from src.alpha.sticky import StickyLeaderConfig, apply_crash_cash

    cfg = StickyLeaderConfig(cash_drawdown=-0.12)
    snap = pl.DataFrame({"ticker": ["HOLD"], "drawdown_20": [-0.15]})
    out = apply_crash_cash({"HOLD": 0.2}, "HOLD", snap, cfg)
    assert out == {}


def test_apply_crash_cash_keeps_when_above_threshold() -> None:
    import polars as pl
    from src.alpha.sticky import StickyLeaderConfig, apply_crash_cash

    cfg = StickyLeaderConfig(cash_drawdown=-0.12)
    snap = pl.DataFrame({"ticker": ["HOLD"], "drawdown_20": [-0.05]})
    out = apply_crash_cash({"HOLD": 0.2}, "HOLD", snap, cfg)
    assert out == {"HOLD": 0.2}
    disabled = StickyLeaderConfig(cash_drawdown=0.0)
    deep = pl.DataFrame({"ticker": ["HOLD"], "drawdown_20": [-0.50]})
    assert apply_crash_cash({"HOLD": 0.2}, "HOLD", deep, disabled) == {"HOLD": 0.2}


def test_sticky_leader_config_impulse_from_yaml_fail_closed() -> None:
    from src.alpha.sticky import StickyLeaderConfig

    nan = StickyLeaderConfig.from_yaml({"impulse_gap": float("nan"), "cash_drawdown": float("nan")})
    assert nan.impulse_gap == 0.0
    assert nan.cash_drawdown == 0.0
    neg = StickyLeaderConfig.from_yaml({"impulse_gap": -1.0, "cash_drawdown": 0.3})
    assert neg.impulse_gap == 0.0
    assert neg.cash_drawdown == 0.0
    ok = StickyLeaderConfig.from_yaml({"impulse_gap": 0.04, "impulse_require_volx": True, "cash_drawdown": -0.12})
    assert ok.impulse_gap == 0.04
    assert ok.impulse_require_volx is True
    assert ok.cash_drawdown == -0.12
