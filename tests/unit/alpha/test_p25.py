def test_p25_registered_keeps_p24_alpha() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel, load_p25_arm, load_p25_lock_remaining

    assert 'P25' in BASELINES
    p25 = BASELINES['P25']()
    p24 = BASELINES['P24']()
    assert isinstance(p25, StickyLeaderModel)
    assert p25.name == 'P25'
    assert p24.name == 'P24'
    cfg = p25.config
    cfg24 = p24.config
    assert str(cfg.mom_col) == 'mom_60'
    assert str(cfg24.mom_col) == 'mom_60'
    assert float(cfg.impulse_gap) == 0.04
    assert cfg.impulse_require_volx is True
    assert float(cfg.cash_drawdown) == -0.12
    assert float(cfg.min_gap) == 0.08
    assert int(cfg.min_hold) == 3
    assert cfg.only_plus_2 is True
    assert cfg.no_inverse is True
    assert cfg.collapse_family is False
    assert load_p25_arm() == 0.50
    assert load_p25_lock_remaining() == 5
    assert not hasattr(p25, 'allocate') or not callable(getattr(p25, 'allocate', None))
