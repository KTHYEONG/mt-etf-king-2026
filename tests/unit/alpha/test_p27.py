def test_p27_registered_matches_p26_alpha() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel, load_p27_overlay_mode
    from src.portfolio.constraints import load_p26_exposure_limits, load_p27_exposure_limits

    assert "P27" in BASELINES
    p26 = BASELINES["P26"]()
    p27 = BASELINES["P27"]()
    assert isinstance(p27, StickyLeaderModel)
    assert p27.name == "P27"
    assert p26.name == "P26"
    c26 = p26.config
    c27 = p27.config
    assert str(c27.mom_col) == "mom_60"
    assert float(c27.cash_drawdown) == 0.0
    assert float(c27.min_gap) == 0.04
    assert int(c27.min_hold) == 2
    assert float(c27.impulse_gap) == 0.0
    assert c27.only_plus_2 is True
    assert c27.no_inverse is True
    assert c27.collapse_family is False
    assert str(c27.mom_col) == str(c26.mom_col)
    assert float(c27.cash_drawdown) == float(c26.cash_drawdown)
    assert float(c27.min_gap) == float(c26.min_gap)
    assert int(c27.min_hold) == int(c26.min_hold)
    assert float(c27.impulse_gap) == float(c26.impulse_gap)
    assert load_p27_overlay_mode() == "identity"
    assert load_p27_exposure_limits() == load_p26_exposure_limits()
    assert load_p27_exposure_limits() == (0.95, 1.90, 0.05)
    assert not hasattr(p27, "allocate") or not callable(getattr(p27, "allocate", None))


def test_sticky_leader_declares_path_dependent_and_reset_trackers() -> None:
    from src.alpha.baselines import BASELINES
    from src.alpha.sticky import StickyLeaderModel
    from src.tournament.simulator import model_requires_path_dependent

    p27 = BASELINES["P27"]()
    p21 = BASELINES["P21"]()
    p26 = BASELINES["P26"]()
    assert isinstance(p27, StickyLeaderModel)
    for model in (p27, p21, p26):
        assert model.path_dependent is True
        assert model.scores_path_independent is False
        assert model_requires_path_dependent(model) is True
        model._held = "X"
        model._hold_len = 7
        model.reset_trackers()
        assert model._held is None
        assert int(model._hold_len) == 0
