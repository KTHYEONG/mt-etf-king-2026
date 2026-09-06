def test_p25_registered_keeps_p24_alpha() -> None:
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.alpha.sticky import StickyLeaderModel
    from src.strategies.ids import STICKY_HOUSE_MONEY
    from src.strategies.sticky.overlays import overlay_param

    assert 'sticky.house_money' in BASELINES
    p25 = BASELINES['sticky.house_money']()
    p24 = BASELINES['sticky.mom60_peak_lock']()
    assert isinstance(p25, StickyLeaderModel)
    assert p25.name == 'sticky.house_money'
    assert p24.name == 'sticky.mom60_peak_lock'
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
    assert overlay_param(STICKY_HOUSE_MONEY, "arm", default=0.50) == 0.50
    assert overlay_param(STICKY_HOUSE_MONEY, "lock_remaining", default=5) == 5
    assert not hasattr(p25, 'allocate') or not callable(getattr(p25, 'allocate', None))
