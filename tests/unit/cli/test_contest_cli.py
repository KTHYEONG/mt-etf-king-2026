"""CLI guards for contest-weekly (idempotency, cards, state) and the daily-refresh skip."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.cli.commands.contest as contest_cmd
from src.contest.decision import ContestAction, ContestDecision


def _weekly_config(tmp_path: Path) -> dict:
    return {
        "nickname": "tester",
        "start_date": "2026-09-21",
        "end_date": "2026-11-13",
        "vehicles": {"K2": {"ticker": "T_K2", "name": "K2"}},
        "inference": {"match_tol_pct": 0.02, "min_weight": 0.90},
        "leaderboard": {"archive_dir": "contest/leaderboard"},
        "reference": {"single_stock_path": "reference/single_stock_daily.parquet"},
        "simulation": {"synthetic_fee_annual": 0.01},
        "decision": {
            "candidates": ["HOLD"],
            "min_adv_krw": 0,
            "target_weight": 0.999,
            "switch_min_p1_gain": 0.05,
            "endgame_sessions": 8,
            "eras": {},
            "profiles": [],
            "n_worlds": 2,
            "mean_block": 10,
            "seed": 1,
            "output_dir": str(tmp_path / "out"),
            "state_name": "contest_position",
        },
    }


def _card(action: ContestAction = ContestAction.HOLD) -> ContestDecision:
    from src.contest.decision import CandidateScore

    return ContestDecision(
        decision_session=date(2026, 9, 23), execution_session=None, action=action,
        current_alias="K2", target_alias="K2", target_ticker="T_K2", target_weight=0.999,
        our_rank=30, our_total_return_pct=3.0, leaderboard_base_date=date(2026, 9, 23),
        sessions_remaining=30,
        scores=(CandidateScore(alias="HOLD", ticker="T_K2", p1_mean=0.1, p1_min=0.05, p2_mean=0.2,
                               p10_mean=0.5, median_return=0.03, p_loss30=0.0),),
        inferred_leaders={}, warnings=(),
    )


def _k2_score() -> tuple:
    from src.contest.decision import CandidateScore

    return (CandidateScore(alias="K2", ticker="T_K2", p1_mean=0.2, p1_min=0.1, p2_mean=0.3,
                           p10_mean=0.6, median_return=0.04, p_loss30=0.0),)


def _run_weekly(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, **ns: object) -> argparse.Namespace:
    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: _weekly_config(tmp_path))
    monkeypatch.setattr(
        "src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path)
    )
    base = {"session": "2026-09-23", "force": False, "worlds": None}
    base.update(ns)
    return argparse.Namespace(**base)


def test_contest_weekly_idempotent_skips_existing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An existing card without --force returns 0 with the file untouched."""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True)
    card_path = out_dir / "2026-09-23.json"
    card_path.write_text('{"action": "HOLD"}\n', encoding="utf-8")
    before = card_path.stat().st_mtime_ns

    def _boom(*a: object, **k: object) -> object:
        raise AssertionError("must not build when skipping")

    monkeypatch.setattr(contest_cmd, "_build_weekly_panel", _boom)
    args = _run_weekly(monkeypatch, tmp_path)
    with caplog.at_level("INFO"):
        assert contest_cmd.cmd_contest_weekly(args) == 0
    assert card_path.read_text(encoding="utf-8") == '{"action": "HOLD"}\n'
    assert card_path.stat().st_mtime_ns == before
    assert any("exists_skip" in r.message for r in caplog.records)


def test_contest_weekly_writes_card_and_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A SWITCH card persists JSON, markdown, and the position state."""
    import src.contest.decision as decision_mod

    switch = ContestDecision(
        decision_session=date(2026, 9, 23), execution_session=date(2026, 9, 24),
        action=ContestAction.SWITCH, current_alias="K2", target_alias="HY2",
        target_ticker="T_HY", target_weight=0.999, our_rank=30, our_total_return_pct=3.0,
        leaderboard_base_date=date(2026, 9, 23), sessions_remaining=30,
        scores=_k2_score(), inferred_leaders={}, warnings=(),
    )
    monkeypatch.setattr(contest_cmd, "_build_weekly_panel", lambda *a, **k: object())
    monkeypatch.setattr(
        "src.contest.leaderboard.latest_snapshot_on_or_before", lambda *a, **k: None
    )
    monkeypatch.setattr(decision_mod, "decide_week", lambda *a, **k: switch)
    args = _run_weekly(monkeypatch, tmp_path, force=True)
    with caplog.at_level("INFO"):
        assert contest_cmd.cmd_contest_weekly(args) == 0
    card = json.loads((tmp_path / "out" / "2026-09-23.json").read_text(encoding="utf-8"))
    assert card["action"] == "SWITCH"
    assert (tmp_path / "out" / "2026-09-23.md").is_file()
    state = json.loads((tmp_path / "state" / "contest_position.json").read_text(encoding="utf-8"))
    assert state["recommended_alias"] == "HY2"
    assert state["source"] == "recommendation"
    assert any("action=SWITCH" in r.message and "p1_best=" in r.message for r in caplog.records)


def test_contest_weekly_no_state_for_non_trade_cards(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """CONTEST_OVER writes the card but never touches the position state."""
    import src.contest.decision as decision_mod

    over = _card(ContestAction.CONTEST_OVER)
    monkeypatch.setattr(contest_cmd, "_build_weekly_panel", lambda *a, **k: object())
    monkeypatch.setattr(
        "src.contest.leaderboard.latest_snapshot_on_or_before", lambda *a, **k: None
    )
    monkeypatch.setattr(decision_mod, "decide_week", lambda *a, **k: over)
    args = _run_weekly(monkeypatch, tmp_path, force=True)
    assert contest_cmd.cmd_contest_weekly(args) == 0
    assert (tmp_path / "out" / "2026-09-23.json").is_file()
    assert not (tmp_path / "state" / "contest_position.json").exists()


def test_contest_weekly_scoreless_switch_logs_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A switch without scored rows logs none sentinels and still returns 0."""
    import src.contest.decision as decision_mod

    bare = ContestDecision(
        decision_session=date(2026, 9, 23), execution_session=date(2026, 9, 24),
        action=ContestAction.SWITCH, current_alias="K2", target_alias="HY2",
        target_ticker="T_HY", target_weight=0.999, our_rank=None, our_total_return_pct=None,
        leaderboard_base_date=None, sessions_remaining=30,
        scores=(), inferred_leaders={}, warnings=("OUTSIDE_TOP50",),
    )
    monkeypatch.setattr(contest_cmd, "_build_weekly_panel", lambda *a, **k: object())
    monkeypatch.setattr(
        "src.contest.leaderboard.latest_snapshot_on_or_before", lambda *a, **k: None
    )
    monkeypatch.setattr(decision_mod, "decide_week", lambda *a, **k: bare)
    args = _run_weekly(monkeypatch, tmp_path, force=True)
    with caplog.at_level("INFO"):
        assert contest_cmd.cmd_contest_weekly(args) == 0
    assert any("p1_current=none" in r.message for r in caplog.records)


def test_contest_weekly_unexpected_error_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A panel build failure fails closed with rc 1."""

    def _boom(*a: object, **k: object) -> object:
        raise RuntimeError("no data")

    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: _weekly_config(tmp_path))
    monkeypatch.setattr(
        "src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path)
    )
    monkeypatch.setattr(contest_cmd, "_build_weekly_panel", _boom)
    args = argparse.Namespace(session="2026-09-23", force=True, worlds=None)
    assert contest_cmd.cmd_contest_weekly(args) == 1


def test_contest_weekly_no_session_resolves_to_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No --session and an empty calendar fails closed with rc 1."""
    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: _weekly_config(tmp_path))
    monkeypatch.setattr(contest_cmd, "_resolve_target_session", lambda *a, **k: None)
    args = argparse.Namespace(session=None, force=True, worlds=None)
    assert contest_cmd.cmd_contest_weekly(args) == 1


def test_contest_weekly_persist_failure_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An unwritable output dir fails closed with rc 1 after deciding."""
    import src.contest.decision as decision_mod

    out_file = tmp_path / "out"
    out_file.write_text("blocking", encoding="utf-8")
    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: _weekly_config(tmp_path))
    monkeypatch.setattr(
        "src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path)
    )
    monkeypatch.setattr(contest_cmd, "_build_weekly_panel", lambda *a, **k: object())
    monkeypatch.setattr(
        "src.contest.leaderboard.latest_snapshot_on_or_before", lambda *a, **k: None
    )
    monkeypatch.setattr(decision_mod, "decide_week", lambda *a, **k: _card())
    args = argparse.Namespace(session="2026-09-23", force=True, worlds=None)
    assert contest_cmd.cmd_contest_weekly(args) == 1


def test_resolve_target_session_picks_friday_for_saturday() -> None:
    """A Saturday run decides on Friday's close."""
    from src.core.calendar import get_calendar

    assert contest_cmd._resolve_target_session(get_calendar(), date(2026, 10, 3)) == date(2026, 10, 2)
    empty_cal = MagicMock()
    empty_cal.sessions.return_value = []
    assert contest_cmd._resolve_target_session(empty_cal, date(2026, 10, 3)) is None


def test_read_state_alias_reads_recorded_intent(tmp_path: Path) -> None:
    """Valid, corrupt, and mistyped state files resolve to an alias or None."""
    import json

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"recommended_alias": "HY2"}), encoding="utf-8")
    assert contest_cmd._read_state_alias(good) == "HY2"
    assert contest_cmd._read_state_alias(tmp_path / "missing.json") is None
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert contest_cmd._read_state_alias(bad) is None
    typed = tmp_path / "typed.json"
    typed.write_text(json.dumps({"recommended_alias": 5}), encoding="utf-8")
    assert contest_cmd._read_state_alias(typed) is None
    listed = tmp_path / "listed.json"
    listed.write_text(json.dumps([1, 2]), encoding="utf-8")
    assert contest_cmd._read_state_alias(listed) is None


def test_build_weekly_panel_reads_silver_and_reference(tmp_path: Path) -> None:
    """The weekly panel assembles from the normalized tables plus the reference file."""
    import polars as pl

    from src.contest.panel import VehiclePanel

    dates = [date(2026, 9, 18), date(2026, 9, 21), date(2026, 9, 22), date(2026, 9, 23)]
    n = len(dates)
    norm = tmp_path / "normalized"
    norm.mkdir(parents=True)
    pl.DataFrame(
        {"date": dates * 2, "index_name": ["코스피 200"] * n + ["코스피"] * n,
         "open": [100.0] * (2 * n), "close": [101.0] * (2 * n)},
        schema={"date": pl.Date, "index_name": pl.String, "open": pl.Float64, "close": pl.Float64},
    ).write_parquet(norm / "index_daily.parquet")
    pl.DataFrame(
        {"date": dates, "ticker": ["T_K2"] * n, "open": [10.0] * n, "close": [10.1] * n},
        schema={"date": pl.Date, "ticker": pl.String, "open": pl.Float64, "close": pl.Float64},
    ).write_parquet(norm / "etf_daily.parquet")
    ref_dir = tmp_path / "reference"
    ref_dir.mkdir(parents=True)
    pl.DataFrame(
        {"date": [], "symbol": [], "open": [], "close": []},
        schema={"date": pl.Date, "symbol": pl.String, "open": pl.Float64, "close": pl.Float64},
    ).write_parquet(ref_dir / "single_stock_daily.parquet")
    contest = {
        "vehicles": {"K2": {"ticker": "T_K2", "returns": "listed", "tradable": True, "crowd_only": False}},
        "reference": {"single_stock_path": "reference/single_stock_daily.parquet"},
        "simulation": {"synthetic_fee_annual": 0.01},
    }
    panel = contest_cmd._build_weekly_panel(contest, tmp_path)
    assert isinstance(panel, VehiclePanel)
    assert panel.names == ("K2",)


def test_contest_mode_check_fails_closed_on_config_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unreadable contest config leaves the champion path untouched."""
    from src.cli.commands.pipeline import _contest_mode_active

    def _boom(name: str) -> object:
        raise RuntimeError("no config")

    monkeypatch.setattr("src.core.config.load_config", _boom)
    assert _contest_mode_active() is False


def test_daily_refresh_skips_champion_decide_in_contest_mode(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """With contest.enabled and --decide, daily-refresh skips cmd_decide with rc 0."""
    import polars as pl
    from unittest.mock import patch

    from src.cli.commands.pipeline import cmd_daily_refresh

    silver_dir = tmp_path / "normalized"
    silver_dir.mkdir(parents=True)
    pl.DataFrame(
        {"date": [date(2018, 1, 2), date(2026, 9, 10)]},
        schema={"date": pl.Date},
    ).write_parquet(silver_dir / "etf_daily.parquet")

    decide_mock = MagicMock(side_effect=AssertionError("cmd_decide must not run in contest mode"))
    args = argparse.Namespace(dataset=None, as_of="2026-09-15", lookback_days=5, decide=True, output_dir=None)
    with (
        patch("src.cli.commands.data.cmd_ingest", MagicMock(return_value=0)),
        patch("src.cli.commands.data.cmd_normalize", MagicMock(return_value=0)),
        patch("src.cli.commands.features.cmd_features", MagicMock(return_value=0)),
        patch("src.cli.commands.decide.cmd_decide", decide_mock),
        patch("src.core.settings.get_settings", return_value=SimpleNamespace(data_root=tmp_path)),
        caplog.at_level("INFO"),
    ):
        rc = cmd_daily_refresh(args)
    assert rc == 0
    decide_mock.assert_not_called()
    assert any("contest_mode weekly decision active" in r.message for r in caplog.records)
