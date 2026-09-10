import argparse
from datetime import date
from types import SimpleNamespace
from unittest.mock import patch

import polars as pl
import pytest


def test_hook_mom60_raw_allocate_wires_real_scoring_into_decide_state() -> None:
    from src.cli.commands.decide.models import _ALLOCATE_HOOKS, _DecideState
    from src.portfolio.intent import PortfolioIntent

    d = date(2026, 8, 27)
    panel = pl.DataFrame({"date": [d], "ticker": ["999"], "mom_60": [0.05]}, schema={"date": pl.Date, "ticker": pl.String, "mom_60": pl.Float64})
    # Given: a valid index row AT the decision date so assert_sleeve_inputs_fresh passes (R14)
    index_frame = pl.DataFrame(
        {"date": [d], "index_name": ["코스피"], "close": [7051.64]},
        schema={"date": pl.Date, "index_name": pl.String, "close": pl.Float64},
    )
    state = _DecideState(
        args=SimpleNamespace(capital=None),
        model_arg="sticky.mom60_raw",
        decision_date=d,
        panel_loaded=panel,
        policy=SimpleNamespace(),
        master=None,
        rules=SimpleNamespace(initial_capital=1_000_000_000),
        regime_str=None,
        lev_allowed=None,
        inv_allowed=None,
    )

    fake_model = SimpleNamespace(
        reset_trackers=lambda: None,
        score=lambda snapshot, ctx: {"999": 0.5},
    )

    # When: hook is registered for sticky.mom60_raw and drives the real score/size pipeline
    assert "sticky.mom60_raw" in _ALLOCATE_HOOKS
    with (
        patch("src.strategies.registry.STRATEGIES", {"sticky.mom60_raw": lambda: fake_model}),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root="data")),
        patch("polars.read_parquet", return_value=index_frame),
    ):
        _ALLOCATE_HOOKS["sticky.mom60_raw"](state)

    # Then: real model-driven weights (TOP1 on the only scored ticker), not a generic mom_20 scorer
    assert isinstance(state.decision_weights, PortfolioIntent)
    assert state.weights == {"999": 1.0}


def test_hook_mom60_raw_allocate_surfaces_explicit_cash_intent() -> None:
    from src.cli.commands.decide.models import _ALLOCATE_HOOKS, _DecideState
    from src.portfolio.intent import CASH_INTENT

    d = date(2026, 8, 27)
    panel = __import__("polars").DataFrame(
        {"date": [d], "ticker": ["999"], "mom_60": [0.05]},
        schema={"date": __import__("polars").Date, "ticker": __import__("polars").String, "mom_60": __import__("polars").Float64},
    )
    state = _DecideState(
        args=SimpleNamespace(capital=None),
        model_arg="sticky.mom60_raw",
        decision_date=d,
        panel_loaded=panel,
        policy=SimpleNamespace(),
        master=None,
        rules=SimpleNamespace(initial_capital=1_000_000_000),
        regime_str=None,
        lev_allowed=None,
        inv_allowed=None,
    )
    cash_model = SimpleNamespace(reset_trackers=lambda: None, score=lambda snapshot, ctx: CASH_INTENT)
    index_frame = pl.DataFrame(
        {"date": [d], "index_name": ["코스피"], "close": [7051.64]},
        schema={"date": pl.Date, "index_name": pl.String, "close": pl.Float64},
    )

    with (
        patch("src.strategies.registry.STRATEGIES", {"sticky.mom60_raw": lambda: cash_model}),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root="data")),
        patch("polars.read_parquet", return_value=index_frame),
    ):
        _ALLOCATE_HOOKS["sticky.mom60_raw"](state)

    # Then: explicit CASH surfaces via the existing peak-lock rationale path (R11)
    assert state.weights == {}
    assert state.decision_weights is CASH_INTENT
    assert state.peak_is_locked is True


def test_hook_mom60_raw_allocate_empty_panel_yields_empty_weights() -> None:
    from src.cli.commands.decide.models import _ALLOCATE_HOOKS, _DecideState

    d = date(2026, 8, 27)
    empty_panel = pl.DataFrame({"date": [], "ticker": [], "mom_60": []}, schema={"date": pl.Date, "ticker": pl.String, "mom_60": pl.Float64})
    state = _DecideState(
        args=SimpleNamespace(capital=None),
        model_arg="sticky.mom60_raw",
        decision_date=d,
        panel_loaded=empty_panel,
        policy=SimpleNamespace(),
        master=None,
        rules=SimpleNamespace(initial_capital=1_000_000_000),
        regime_str=None,
        lev_allowed=None,
        inv_allowed=None,
    )

    # When: panel has zero rows -> early return, no scoring attempted
    _ALLOCATE_HOOKS["sticky.mom60_raw"](state)

    # Then
    assert state.weights == {}
    assert state.decision_weights is None

    # Then: panel_loaded is None -> the same early return
    state2 = _DecideState(
        args=SimpleNamespace(capital=None),
        model_arg="sticky.mom60_raw",
        decision_date=d,
        panel_loaded=None,
        policy=SimpleNamespace(),
        master=None,
        rules=SimpleNamespace(initial_capital=1_000_000_000),
        regime_str=None,
        lev_allowed=None,
        inv_allowed=None,
    )
    _ALLOCATE_HOOKS["sticky.mom60_raw"](state2)
    assert state2.weights == {}
    assert state2.decision_weights is None


def test_hook_mom60_raw_allocate_falls_back_to_default_capital_on_bad_rules() -> None:
    from src.cli.commands.decide.models import _ALLOCATE_HOOKS, _DecideState
    from src.portfolio.intent import PortfolioIntent

    d = date(2026, 8, 27)
    panel = pl.DataFrame({"date": [d], "ticker": ["999"], "mom_60": [0.05]}, schema={"date": pl.Date, "ticker": pl.String, "mom_60": pl.Float64})
    state = _DecideState(
        args=SimpleNamespace(capital=None),
        model_arg="sticky.mom60_raw",
        decision_date=d,
        panel_loaded=panel,
        policy=SimpleNamespace(),
        master=None,
        rules=SimpleNamespace(initial_capital="not-a-number"),  # forces the TypeError/ValueError fallback
        regime_str=None,
        lev_allowed=None,
        inv_allowed=None,
    )
    captured_capital: list[float] = []

    def _score(snapshot, ctx):  # type: ignore[no-untyped-def]
        captured_capital.append(ctx.capital)
        return {"999": 0.5}

    fake_model = SimpleNamespace(reset_trackers=lambda: None, score=_score)
    index_frame = pl.DataFrame(
        {"date": [d], "index_name": ["코스피"], "close": [7051.64]},
        schema={"date": pl.Date, "index_name": pl.String, "close": pl.Float64},
    )

    with (
        patch("src.strategies.registry.STRATEGIES", {"sticky.mom60_raw": lambda: fake_model}),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root="data")),
        patch("polars.read_parquet", return_value=index_frame),
    ):
        _ALLOCATE_HOOKS["sticky.mom60_raw"](state)

    # Then: malformed rules.initial_capital falls back to the 1B default (models.py's except branch)
    assert captured_capital == [1_000_000_000.0]
    assert isinstance(state.decision_weights, PortfolioIntent)


def test_cmd_decide_p27_path_computes_order_estimates_end_to_end(capsys) -> None:
    import argparse

    from src.cli import cmd_decide
    from src.cli.commands.decide import models as decide_models
    from src.tournament.live_decision import LiveOrderEstimate

    d = date(2026, 8, 27)
    panel = pl.DataFrame(
        {"date": [d], "ticker": ["412570"], "close": [726_500.0]},
        schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64},
    )

    def _stub_allocate(state) -> None:  # type: ignore[no-untyped-def]
        state.weights = {"412570": 1.0}
        state.panel_loaded = panel

    args = argparse.Namespace(model="sticky.mom60_raw", date="2026-08-27", panel=None, capital=None, output=None, trace=False)

    with (
        patch.dict(decide_models._ALLOCATE_HOOKS, {"sticky.mom60_raw": _stub_allocate}),
        patch("src.cli.commands.decide._load_panel_for_backtest", return_value=panel),
        patch("src.cli.commands.decide._scores_from_deployment_universe", return_value={"412570": 0.5}),
    ):
        rc = cmd_decide(args)

    # Then: the real __init__.py wiring resolved capital from rules and computed order estimates (R14)
    assert rc == 0
    out = capsys.readouterr().out
    assert "412570" in out
    assert "추정" in out


def test_cmd_decide_p27_path_fails_closed_when_price_missing(capsys) -> None:
    import argparse

    from src.cli import cmd_decide
    from src.cli.commands.decide import models as decide_models

    d = date(2026, 8, 27)
    # Panel has no row for "412570" on decision_date -> order estimate cannot price the position.
    panel = pl.DataFrame({"date": [d], "ticker": ["999"], "close": [100.0]}, schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64})

    def _stub_allocate(state) -> None:  # type: ignore[no-untyped-def]
        state.weights = {"412570": 1.0}
        state.panel_loaded = panel

    args = argparse.Namespace(model="sticky.mom60_raw", date="2026-08-27", panel=None, capital=None, output=None, trace=False)

    with (
        patch.dict(decide_models._ALLOCATE_HOOKS, {"sticky.mom60_raw": _stub_allocate}),
        patch("src.cli.commands.decide._load_panel_for_backtest", return_value=panel),
        patch("src.cli.commands.decide._scores_from_deployment_universe", return_value={"412570": 0.5}),
    ):
        rc = cmd_decide(args)

    # Then: the command fails closed (R6 propagated through __init__.py's ValueError catch)
    assert rc == 1


def test_cmd_decide_legacy_path_skips_order_estimate_and_stays_green(capsys) -> None:
    import argparse
    from unittest.mock import patch

    from src.cli import cmd_decide

    # Given: no --model (legacy/default path), a decision_date with zero real panel coverage
    args = argparse.Namespace(date="2026-10-07")

    # When: estimate_live_order_quantities must never even be called for this path (R14)
    with patch(
        "src.tournament.live_decision.estimate_live_order_quantities",
        side_effect=AssertionError("must not be called for the legacy/default decide path"),
    ):
        rc = cmd_decide(args)

    # Then: unchanged legacy behaviour - still succeeds via the synthetic fallback scores
    assert rc == 0
    out = capsys.readouterr().out
    assert "PORTFOLIO" in out


def test_hook_mom60_raw_allocate_fails_closed_on_stale_index() -> None:
    from src.cli.commands.decide.models import _ALLOCATE_HOOKS, _DecideState
    from src.tournament.live_decision import StaleSleeveInputError

    d = date(2026, 9, 21)
    panel = pl.DataFrame(
        {"date": [d], "ticker": ["999"], "mom_60": [0.05]},
        schema={"date": pl.Date, "ticker": pl.String, "mom_60": pl.Float64},
    )
    state = _DecideState(
        args=SimpleNamespace(capital=None),
        model_arg="sticky.mom60_raw",
        decision_date=d,
        panel_loaded=panel,
        policy=SimpleNamespace(),
        master=None,
        rules=SimpleNamespace(initial_capital=1_000_000_000),
        regime_str=None,
        lev_allowed=None,
        inv_allowed=None,
    )
    fake_model = SimpleNamespace(reset_trackers=lambda: None, score=lambda snapshot, ctx: {"999": 0.5})

    # When: the index parquet is unreadable -> empty frame -> gate must fire (R11)
    with (
        patch("src.strategies.registry.STRATEGIES", {"sticky.mom60_raw": lambda: fake_model}),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root="data")),
        patch("polars.read_parquet", side_effect=OSError("no index file")),
        pytest.raises(StaleSleeveInputError),
    ):
        _ALLOCATE_HOOKS["sticky.mom60_raw"](state)


def test_cmd_decide_returns_1_on_stale_sleeve_input(capsys) -> None:
    from src.cli import cmd_decide
    from src.cli.commands.decide import models as decide_models
    from src.tournament.live_decision import StaleSleeveInputError

    d = date(2026, 9, 21)
    panel = pl.DataFrame(
        {"date": [d], "ticker": ["412570"], "close": [1191.0]},
        schema={"date": pl.Date, "ticker": pl.String, "close": pl.Float64},
    )

    def _stale_hook(state):
        raise StaleSleeveInputError("sleeve inputs missing at 2026-09-21")

    args = argparse.Namespace(model="sticky.mom60_raw", date="2026-09-21", panel=None, capital=None, output=None, trace=False)

    with (
        patch.dict(decide_models._ALLOCATE_HOOKS, {"sticky.mom60_raw": _stale_hook}),
        patch("src.cli.commands.decide._load_panel_for_backtest", return_value=panel),
        patch("src.cli.commands.decide._scores_from_deployment_universe", return_value={"412570": 0.5}),
    ):
        rc = cmd_decide(args)

    # Then: loud failure, not a silent CASH dashboard (R12)
    assert rc == 1
    assert "PORTFOLIO" not in capsys.readouterr().out
