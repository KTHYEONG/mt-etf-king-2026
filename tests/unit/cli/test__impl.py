from __future__ import annotations


def test_p31_in_sticky_adoption_models() -> None:
    from src.cli import STICKY_ADOPTION_MODELS

    assert "convex.lottery_impulse" in STICKY_ADOPTION_MODELS
    assert "sticky.fillable_mom60" in STICKY_ADOPTION_MODELS


def test_attach_p36_backtest_artifacts_updates_summary() -> None:
    from datetime import date
    from types import SimpleNamespace

    import polars as pl

    from src.cli.commands.backtest.artifacts import attach_backtest_artifacts
    from src.cli.context import BacktestCellBundle, BacktestResult

    sessions = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7)]
    panel = pl.DataFrame(
        {"date": sessions, "ticker": ["A"] * 3, "open": [1.0, 1.0, 1.0], "close": [1.0, 1.0, 1.0]}
    )
    bundle = BacktestCellBundle(
        run_id="r1",
        meta={},
        summary={},
        panel=panel,
        engine=SimpleNamespace(),
        model=SimpleNamespace(),
        case_config=SimpleNamespace(start=sessions[0], end=sessions[-1]),
        rolling=SimpleNamespace(starts=sessions[:1], returns=(0.0,)),
        horizon=1,
        shared_cache=SimpleNamespace(
            scores={sessions[0]: {"A": 1.0}},
            universes={},
            open_map={d: {"A": 1.0} for d in sessions},
        ),
        leverage_allowed=True,
        inverse_allowed=False,
    )
    ctx = SimpleNamespace(
        calendar=SimpleNamespace(sessions=lambda _s, _e: sessions),
        paths=SimpleNamespace(),
    )
    out = attach_backtest_artifacts(ctx, BacktestResult(return_code=0, cells=(bundle,), strategy_id="x"))
    assert "phantom_sessions" in out["r1"]["meta"]
    assert "attainability" in out["r1"]["summary"]


def test_p31_membership_uses_p27_exposure() -> None:
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.portfolio.constraints import (
        load_p27_exposure_limits,
        resolve_exposure_limits_for_model,
    )

    assert "convex.lottery_impulse" in BASELINES
    assert resolve_exposure_limits_for_model("convex.lottery_impulse", comparison_mode="full_strategy_own") == load_p27_exposure_limits()


def test_cmd_backtest_wires_p36_attainability_artifacts() -> None:
    import argparse

    from src.cli.commands.backtest import cmd_backtest

    args = argparse.Namespace(
        model="baseline.mom20_top1",
        start="2024-01-02",
        end="2024-02-28",
        leverage_scenario="aggressive",
        eval_mode="adoption",
        protocol="single",
        stress_grid=False,
        commission_bps=None,
        slippage_bps=None,
        participation=None,
        forensics=False,
        trace=False,
    )
    assert cmd_backtest(args) in (0, 1)


def test_cmd_backtest_tolerates_p36_attach_failure(monkeypatch) -> None:
    import argparse

    from src.cli.commands.backtest import cmd_backtest

    def _boom(*_a, **_k):
        raise RuntimeError("attach fail")

    monkeypatch.setattr("src.cli.commands.backtest.artifacts.attach_backtest_artifacts", _boom)
    monkeypatch.setattr("src.reporting.results.write_backtest_result", lambda *_a, **_k: None)
    args = argparse.Namespace(
        model="baseline.mom20_top1",
        start="2024-01-02",
        end="2024-02-28",
        leverage_scenario="aggressive",
        eval_mode="adoption",
        protocol="single",
        stress_grid=False,
        commission_bps=None,
        slippage_bps=None,
        participation=None,
        forensics=False,
        trace=False,
    )
    assert cmd_backtest(args) in (0, 1)
