def test_p24_registered_mom60_keeps_p21_impulse() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel

    assert 'P24' in BASELINES
    p24 = BASELINES['P24']()
    p21 = BASELINES['P21']()
    assert isinstance(p24, StickyLeaderModel)
    assert p24.name == 'P24'
    cfg = p24.config
    p21_cfg = p21.config
    assert str(cfg.mom_col) == 'mom_60'
    assert str(p21_cfg.mom_col) == 'mom_20'
    assert float(cfg.impulse_gap) == 0.04
    assert cfg.impulse_require_volx is True
    assert float(cfg.cash_drawdown) == -0.12
    assert float(cfg.min_gap) == 0.08
    assert int(cfg.min_hold) == 3
    assert cfg.only_plus_2 is True
    assert float(cfg.lock_level) == 0.50
    assert not hasattr(p24, 'allocate') or not callable(getattr(p24, 'allocate', None))
    assert float(p21_cfg.impulse_gap) == 0.04
    assert float(p21_cfg.lock_level) in {0.0, 0.40}


def test_p24_score_uses_mom60_column() -> None:
    from datetime import date
    import polars as pl
    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.portfolio.sizing import SizingScheme, weights_from_scores
    from src.universe.tournament import TournamentRules

    snap = pl.DataFrame(
        {
            'ticker': ['SLOW', 'FAST'],
            'name': ['KODEX 레버리지', 'KODEX 코스닥150레버리지'],
            'mom_20': [0.50, 0.10],
            'mom_60': [0.10, 0.40],
            'mom_5': [0.01, 0.02],
            'volume_expansion': [0.1, 0.1],
            'drawdown_20': [0.0, 0.0],
        }
    )
    rules = TournamentRules(
        name='t',
        start_date=date(2026, 9, 21),
        end_date=date(2026, 11, 13),
        initial_capital=1_000_000_000,
        category='autonomous',
        leverage_allowed=True,
        inverse_allowed=True,
        max_weight=1.0,
        cash_allowed=True,
        sponsor_etf_only=True,
        manifest_path=None,
        issuer_whitelist=None,
        commission_bps=3.0,
        slippage_bps=5.0,
        max_order_to_adv=0.01,
        stress_grid=(0.01, 0.02, 0.05),
    )
    ctx = DecisionContext(decision_date=date(2026, 1, 2), regime=None, capital=1.0e9, held={}, rules=rules)
    p24 = BASELINES['P24']()
    scores = p24.score(snap, ctx)
    w = weights_from_scores(scores, SizingScheme.TOP1, k=1)
    assert set(w.keys()) == {'FAST'}
    p21 = BASELINES['P21']()
    w21 = weights_from_scores(p21.score(snap, ctx), SizingScheme.TOP1, k=1)
    assert set(w21.keys()) == {'SLOW'}


def test_load_p24_config_fail_closed() -> None:
    from src.alpha.sticky import load_p24_lock_level, load_p24_mom_col, load_p24_trail, resolve_lock_level

    assert load_p24_lock_level() == 0.50
    assert load_p24_trail() == 0.0
    assert load_p24_mom_col() == 'mom_60'
    assert resolve_lock_level(float('nan'), default=0.50) == 0.50
    assert resolve_lock_level(-1.0, default=0.50) == 0.50
    assert resolve_lock_level('bad', default=0.50) == 0.50
