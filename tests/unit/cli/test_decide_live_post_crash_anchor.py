# ruff: noqa
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl


def test_hook_post_crash_anchor_allocate_uses_own_model_and_state_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    from src.cli.commands.decide.models import (
        _ALLOCATE_HOOKS,
        _LIVE_STICKY_STRATEGIES,
        _OVERLAY_HOOKS,
        _DecideState,
    )
    from src.portfolio.intent import PortfolioIntent

    d = date(2026, 9, 18)
    panel = pl.DataFrame(
        {"date": [d], "ticker": ["233740"], "mom_60": [-0.10], "name": ["Test ETF"], "trading_value": [1e12]},
        schema={"date": pl.Date, "ticker": pl.String, "mom_60": pl.Float64, "name": pl.String, "trading_value": pl.Float64},
    )
    index_frame = pl.DataFrame(
        {"date": [d], "index_name": ["코스피"], "close": [7051.64]},
        schema={"date": pl.Date, "index_name": pl.String, "close": pl.Float64},
    )
    fake_model = SimpleNamespace(
        reset_trackers=lambda: None,
        restore_state=lambda held, hold_len: None,
        score=lambda snapshot, ctx: {"233740": 0.2},
    )

    def _raw_factory() -> object:
        raise AssertionError("raw P27 model must not be instantiated for the anchor strategy")

    state = _DecideState(
        args=SimpleNamespace(capital=None, held=None),
        model_arg="sticky.mom60_post_crash_anchor",
        decision_date=d,
        panel_loaded=panel,
        policy=SimpleNamespace(),
        master=None,
        rules=SimpleNamespace(initial_capital=1_000_000_000),
        regime_str=None,
        lev_allowed=None,
        inv_allowed=None,
    )

    assert "sticky.mom60_post_crash_anchor" in _ALLOCATE_HOOKS
    assert _LIVE_STICKY_STRATEGIES == frozenset({"sticky.mom60_raw", "sticky.mom60_post_crash_anchor"})
    assert _OVERLAY_HOOKS["sticky.mom60_post_crash_anchor"] is _OVERLAY_HOOKS["sticky.mom60_raw"]

    with (
        patch(
            "src.strategies.registry.STRATEGIES",
            {"sticky.mom60_post_crash_anchor": lambda: fake_model, "sticky.mom60_raw": _raw_factory},
        ),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
        patch("polars.read_parquet", return_value=index_frame),
    ):
        _ALLOCATE_HOOKS["sticky.mom60_post_crash_anchor"](state)

    assert isinstance(state.decision_weights, PortfolioIntent)
    assert state.weights == {"233740": 0.95}
    assert (tmp_path / "state" / "sticky_mom60_post_crash_anchor_position.json").exists()
    assert not (tmp_path / "state" / "sticky_mom60_raw_position.json").exists()



from datetime import date
from types import SimpleNamespace

import pytest


def test_hook_sticky_live_allocate_rejects_unknown_strategy() -> None:
    from src.cli.commands.decide.models import _DecideState, _hook_sticky_live_allocate

    state = _DecideState(
        args=SimpleNamespace(capital=None, held=None),
        model_arg="sticky.mom60_hold",
        decision_date=date(2026, 9, 18),
        panel_loaded=None,
        policy=SimpleNamespace(),
        master=None,
        rules=None,
        regime_str=None,
        lev_allowed=None,
        inv_allowed=None,
    )

    with pytest.raises(ValueError):
        _hook_sticky_live_allocate(state, strategy_id="sticky.mom60_hold")



from datetime import date
from unittest.mock import patch

import polars as pl


def test_cmd_decide_post_crash_anchor_path_computes_order_estimates(capsys) -> None:  # type: ignore[no-untyped-def]
    import argparse

    from src.cli import cmd_decide
    from src.cli.commands.decide import models as decide_models

    d = date(2026, 8, 27)
    panel = pl.DataFrame(
        {"date": [d], "ticker": ["233740"], "close": [12_345.0]},
        schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64},
    )

    def _stub_allocate(state) -> None:  # type: ignore[no-untyped-def]
        state.weights = {"233740": 0.95}
        state.panel_loaded = panel

    args = argparse.Namespace(
        model="sticky.mom60_post_crash_anchor", date="2026-08-27", panel=None, capital=None, output=None, trace=False
    )

    with (
        patch.dict(decide_models._ALLOCATE_HOOKS, {"sticky.mom60_post_crash_anchor": _stub_allocate}),
        patch("src.cli.commands.decide._load_panel_for_backtest", return_value=panel),
        patch("src.cli.commands.decide._scores_from_deployment_universe", return_value={"233740": 0.5}),
    ):
        rc = cmd_decide(args)

    assert rc == 0
    out = capsys.readouterr().out
    assert "233740" in out
    assert "추정" in out

