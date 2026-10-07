"""CLI guards and invariants for contest-segment (daily segment card & equity ledger)."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.cli.commands.contest_segment as contest_segment_cmd
from src.cli.commands.contest_segment import cmd_contest_segment
from src.contest.daily import QuoteBar, archive_quotes
from src.contest.leaderboard import archive_leaderboard
from src.contest.segment import (
    IntradaySide,
    SegmentSettings,
    session_policy_multiple,
)
from src.core.calendar import get_calendar

CAL = get_calendar()
SESSION = date(2026, 10, 7)
PREV_SESSION = CAL.previous_session(SESSION)


def _config(out: Path, enabled: bool = True) -> dict[str, Any]:
    return {
        "nickname": "tester",
        "start_date": "2026-09-21",
        "end_date": "2026-11-13",
        "leaderboard": {"archive_dir": "contest/leaderboard"},
        "shadow": {
            "enabled": True,
            "quotes_dir": "contest/quotes",
            "yahoo_suffix": ".KS",
            "timeout_s": 20,
        },
        "fade": {"enabled": False},
        "segment": {
            "enabled": enabled,
            "long_ticker": "0193T0",
            "long_alias": "HY2",
            "inverse_ticker": "0197X0",
            "inverse_alias": "HY2I",
            "vote": {
                "HY2": "0193T0",
                "SEMI2": "494310",
                "K2": "122630",
            },
            "vote_min_down": 2,
            "open_window": "09:05-09:15",
            "close_window": "15:10-15:19",
            "cost_bps_round_trip": 12.0,
            "prize_rank": 3,
            "endgame_sessions": 5,
            "ledger_state_name": "contest_segment_equity",
            "output_dir": str(out),
        },
    }


def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: dict[str, Any]) -> None:
    monkeypatch.setattr(contest_segment_cmd, "_contest_config", lambda: cfg)
    monkeypatch.setattr("src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))


def _args(**over: Any) -> argparse.Namespace:
    base: dict[str, Any] = {"session": SESSION.isoformat(), "force": False, "our_return": None}
    base.update(over)
    return argparse.Namespace(**base)


def _put_snapshot(root: Path, day: date, rows: list[tuple[int, str, float, float]]) -> None:
    archive_leaderboard(
        {
            "etfRankTotal": {
                "baseDt": day.strftime("%Y%m%d"),
                "reqDateTime": day.strftime("%Y%m%d") + "160500",
                "data": [
                    {"rank": r, "userName": u, "totalReturnRate": t, "dailyReturnRate": d}
                    for r, u, t, d in rows
                ],
            }
        },
        root,
    )


def _put_archive(root: Path, day: date, bars: dict[str, QuoteBar]) -> None:
    archive_quotes(root, day, bars)


def _sample_bars_down() -> dict[str, QuoteBar]:
    return {
        "0193T0": QuoteBar("0193T0", 100.0, 100.0, 98.0),
        "0197X0": QuoteBar("0197X0", 50.0, 50.0, 51.0),
        "494310": QuoteBar("494310", 100.0, 100.0, 97.0),
        "122630": QuoteBar("122630", 100.0, 100.0, 101.0),
    }


def _sample_bars_up() -> dict[str, QuoteBar]:
    return {
        "0193T0": QuoteBar("0193T0", 100.0, 100.0, 102.0),
        "0197X0": QuoteBar("0197X0", 50.0, 50.0, 49.0),
        "494310": QuoteBar("494310", 100.0, 100.0, 101.0),
        "122630": QuoteBar("122630", 100.0, 100.0, 101.5),
    }


def _write_ledger(
    root: Path, as_of: date, equity: float, source: str = "USER", name: str = "contest_segment_equity"
) -> None:
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{name}.json").write_text(
        json.dumps({"as_of": as_of.isoformat(), "equity": equity, "source": source}), encoding="utf-8"
    )


def test_archived_quotes_build_card_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Archived quotes build card without network."""
    out = tmp_path / "out"
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)
    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, _sample_bars_down())

    def _boom(*args: Any, **kwargs: Any) -> dict[str, QuoteBar]:
        raise AssertionError("fetch must not be called when quotes are archived")

    monkeypatch.setattr("src.contest.daily.fetch_quotes", _boom)

    assert cmd_contest_segment(_args()) == 0
    card_json = out / f"{SESSION.isoformat()}.json"
    card_md = out / f"{SESSION.isoformat()}.md"
    assert card_json.is_file()
    assert card_md.is_file()

    payload = json.loads(card_json.read_text(encoding="utf-8"))
    assert payload["action"] == "HOLD_LONG"


def test_missing_archive_fetches_in_memory_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing archive fetches in memory only without writing to archive."""
    out = tmp_path / "out"
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)

    bars = _sample_bars_down()
    fetch_called = False

    def _fetch(tickers: list[str], session: date, suffix: str, timeout_s: float) -> dict[str, QuoteBar]:
        nonlocal fetch_called
        fetch_called = True
        return bars

    monkeypatch.setattr("src.contest.daily.fetch_quotes", _fetch)

    assert cmd_contest_segment(_args()) == 0
    assert fetch_called
    assert (out / f"{SESSION.isoformat()}.json").is_file()
    assert not (tmp_path / "contest" / "quotes" / f"{SESSION.strftime('%Y%m%d')}.json").exists()


def test_fetch_failure_is_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetch failure is fail-closed, yielding NO_DATA with QUOTES_MISSING and leaving ledger unchanged."""
    out = tmp_path / "out"
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)
    _write_ledger(tmp_path, PREV_SESSION, 1.0)
    ledger_path = tmp_path / "state" / "contest_segment_equity.json"
    ledger_before = ledger_path.read_text(encoding="utf-8")

    def _boom(*args: Any, **kwargs: Any) -> dict[str, QuoteBar]:
        raise RuntimeError("network failure")

    monkeypatch.setattr("src.contest.daily.fetch_quotes", _boom)

    assert cmd_contest_segment(_args()) == 0
    payload = json.loads((out / f"{SESSION.isoformat()}.json").read_text(encoding="utf-8"))
    assert payload["action"] == "NO_DATA"
    assert "QUOTES_MISSING" in payload["warnings"]
    assert ledger_path.read_text(encoding="utf-8") == ledger_before


def test_disabled_segment_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabled segment is a no-op, returning 0 without creating output directory."""
    out = tmp_path / "out"
    cfg = _config(out, enabled=False)
    _setup(monkeypatch, tmp_path, cfg)

    assert cmd_contest_segment(_args()) == 0
    assert not out.exists()


def test_existing_card_skipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing non-NO_DATA card is skipped unless --force is given."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    target = out / f"{SESSION.isoformat()}.json"
    target.write_text('{"action": "HOLD_LONG"}\n', encoding="utf-8")
    before_bytes = target.read_bytes()

    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)

    def _boom(*args: Any, **kwargs: Any) -> dict[str, QuoteBar]:
        raise AssertionError("must not fetch quotes when card is skipped")

    monkeypatch.setattr("src.contest.daily.fetch_quotes", _boom)

    assert cmd_contest_segment(_args(force=False)) == 0
    assert target.read_bytes() == before_bytes


def test_no_data_card_rebuilt_on_rerun(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing NO_DATA card is rebuilt on rerun without --force."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    target = out / f"{SESSION.isoformat()}.json"
    target.write_text('{"action": "NO_DATA"}\n', encoding="utf-8")

    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)
    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, _sample_bars_down())

    assert cmd_contest_segment(_args(force=False)) == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["action"] == "HOLD_LONG"


def test_our_return_anchors_ledger_and_rebuilds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--our-return anchors ledger and rebuilds even when an existing card is present."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    target = out / f"{SESSION.isoformat()}.json"
    target.write_text('{"action": "HOLD_LONG"}\n', encoding="utf-8")

    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)
    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, _sample_bars_down())

    assert cmd_contest_segment(_args(our_return=-11.5)) == 0

    ledger_path = tmp_path / "state" / "contest_segment_equity.json"
    assert ledger_path.is_file()
    ledger_data = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger_data["as_of"] == SESSION.isoformat()
    assert ledger_data["equity"] == pytest.approx(0.885)
    assert ledger_data["source"] == "USER"

    card_data = json.loads(target.read_text(encoding="utf-8"))
    assert card_data["standing"]["our_equity"] == pytest.approx(0.885)


def test_ledger_compounds_from_previous_card(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ledger compounds from previous card decision and current session bars."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    # 10/06 card with action SWITCH_SHORT
    prev_card = out / f"{PREV_SESSION.isoformat()}.json"
    prev_card.write_text('{"action": "SWITCH_SHORT"}\n', encoding="utf-8")

    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)
    _write_ledger(tmp_path, PREV_SESSION, 0.9, source="LEDGER")

    quotes_root = tmp_path / "contest" / "quotes"
    bars = _sample_bars_down()
    _put_archive(quotes_root, SESSION, bars)

    settings = SegmentSettings.load(cfg)
    expected_mult = session_policy_multiple(bars, IntradaySide.SHORT, session_is_final=False, settings=settings)
    assert expected_mult is not None

    assert cmd_contest_segment(_args()) == 0

    ledger_path = tmp_path / "state" / "contest_segment_equity.json"
    ledger_data = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger_data["as_of"] == SESSION.isoformat()
    assert ledger_data["equity"] == pytest.approx(0.9 * expected_mult)
    assert ledger_data["source"] == "LEDGER"


def test_missing_previous_card_leaves_ledger_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing previous card leaves ledger stale (no guessing)."""
    out = tmp_path / "out"
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)
    _write_ledger(tmp_path, PREV_SESSION, 0.9, source="LEDGER")
    ledger_path = tmp_path / "state" / "contest_segment_equity.json"
    before_ledger = ledger_path.read_text(encoding="utf-8")

    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, _sample_bars_down())

    assert cmd_contest_segment(_args()) == 0
    assert ledger_path.read_text(encoding="utf-8") == before_ledger

    card_data = json.loads((out / f"{SESSION.isoformat()}.json").read_text(encoding="utf-8"))
    assert "LEDGER_STALE" in card_data["warnings"]


def test_no_data_previous_card_leaves_ledger_stale(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A NO_DATA previous card may have been executed via the manual SHORT fallback, so the ledger must not compound."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    (out / f"{PREV_SESSION.isoformat()}.json").write_text('{"action": "NO_DATA"}\n', encoding="utf-8")
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)
    _write_ledger(tmp_path, PREV_SESSION, 0.9, source="LEDGER")
    ledger_path = tmp_path / "state" / "contest_segment_equity.json"
    before_ledger = ledger_path.read_text(encoding="utf-8")
    _put_archive(tmp_path / "contest" / "quotes", SESSION, _sample_bars_down())

    assert cmd_contest_segment(_args()) == 0
    assert ledger_path.read_text(encoding="utf-8") == before_ledger
    card_data = json.loads((out / f"{SESSION.isoformat()}.json").read_text(encoding="utf-8"))
    assert "LEDGER_STALE" in card_data["warnings"]


def test_rerun_does_not_double_compound(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rerun with --force does not double-compound an already-updated ledger."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    prev_card = out / f"{PREV_SESSION.isoformat()}.json"
    prev_card.write_text('{"action": "SWITCH_SHORT"}\n', encoding="utf-8")

    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)
    _write_ledger(tmp_path, PREV_SESSION, 0.9, source="LEDGER")

    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, _sample_bars_down())

    assert cmd_contest_segment(_args()) == 0
    ledger_path = tmp_path / "state" / "contest_segment_equity.json"
    first_equity = json.loads(ledger_path.read_text(encoding="utf-8"))["equity"]

    # Second run with --force
    assert cmd_contest_segment(_args(force=True)) == 0
    second_equity = json.loads(ledger_path.read_text(encoding="utf-8"))["equity"]
    assert first_equity == pytest.approx(second_equity)


def test_leaderboard_listing_anchors_ledger(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Same-day snapshot listing configured nickname anchors the ledger."""
    out = tmp_path / "out"
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)

    leaderboard_root = tmp_path / "contest" / "leaderboard"
    _put_snapshot(leaderboard_root, SESSION, [(1, "tester", 3.0, 0.5)])

    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, _sample_bars_down())

    assert cmd_contest_segment(_args()) == 0

    ledger_path = tmp_path / "state" / "contest_segment_equity.json"
    ledger_data = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger_data["as_of"] == SESSION.isoformat()
    assert ledger_data["equity"] == pytest.approx(1.03)
    assert ledger_data["source"] == "LEADERBOARD"


def test_standing_uses_same_day_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Standing computes prize_line_equity (3rd best competitor) and required_multiple."""
    out = tmp_path / "out"
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)

    leaderboard_root = tmp_path / "contest" / "leaderboard"
    _put_snapshot(
        leaderboard_root,
        SESSION,
        [
            (1, "rival1", 10.0, 1.0),
            (2, "rival2", 8.0, 0.8),
            (3, "rival3", 5.0, 0.5),
            (4, "rival4", 3.0, 0.3),
            (5, "rival5", 1.0, 0.1),
        ],
    )

    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, _sample_bars_down())

    assert cmd_contest_segment(_args(our_return=0.0)) == 0

    card_data = json.loads((out / f"{SESSION.isoformat()}.json").read_text(encoding="utf-8"))
    standing = card_data["standing"]
    assert standing["our_equity"] == pytest.approx(1.0)
    assert standing["prize_line_equity"] == pytest.approx(1.05)  # 3rd best is rival3 at 5.0%
    assert standing["required_multiple"] == pytest.approx(1.05 / 1.0)


def test_persist_failure_returns_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Persistence failure returns 1 and logs reason=persist."""
    out = tmp_path / "out"
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)

    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, _sample_bars_down())

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(contest_segment_cmd, "atomic_write_text", _boom)

    with caplog.at_level(logging.ERROR):
        rc = cmd_contest_segment(_args())
    assert rc == 1
    assert any("[SYS] contest_segment status=fail reason=persist" in r.getMessage() for r in caplog.records)


def test_parser_and_registry_agree() -> None:
    """Parser and registry dispatch agree on contest-segment."""
    from src.cli.main import _SHARED_DATA_COMMANDS, SUBCOMMANDS as MAIN_SUBCOMMANDS
    from src.cli.parser import SUBCOMMANDS as PARSER_SUBCOMMANDS, build_parser

    parser = build_parser()
    args = parser.parse_args(["contest-segment", "--our-return", "-11.5", "--session", "2026-10-07"])
    assert args.func is cmd_contest_segment
    assert args.our_return == -11.5
    assert args.session == "2026-10-07"

    assert "contest-segment" in PARSER_SUBCOMMANDS
    assert "contest-segment" in MAIN_SUBCOMMANDS
    assert MAIN_SUBCOMMANDS["contest-segment"] is cmd_contest_segment
    assert "contest-segment" in _SHARED_DATA_COMMANDS


def test_config_block_validates() -> None:
    """Repository configs/contest.yaml validates and has fade disabled."""
    from src.core.config import load_config

    cfg = load_config("contest")
    contest = cfg.get("contest", {})
    settings = SegmentSettings.load(contest)
    assert settings.prize_rank == 3
    assert contest.get("fade", {}).get("enabled") is False
    assert contest.get("segment", {}).get("enabled") is True


def test_no_resolvable_session_returns_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No resolvable target session fails closed with rc 1."""
    out = tmp_path / "out"
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)
    monkeypatch.setattr(contest_segment_cmd, "_resolve_target_session", lambda cal, today: None)
    assert cmd_contest_segment(argparse.Namespace(session=None, force=False, our_return=None)) == 1


def test_non_mapping_shadow_and_leaderboard_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-mapping shadow and leaderboard sections fall back to defaults."""
    out = tmp_path / "out"
    cfg = _config(out)
    cfg["shadow"] = "not_a_dict"
    cfg["leaderboard"] = "not_a_dict"
    _setup(monkeypatch, tmp_path, cfg)
    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, _sample_bars_down())
    assert cmd_contest_segment(_args()) == 0


def test_corrupted_ledger_and_prev_card_degrades(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupted ledger file and prev card degrade without raising, and the unreadable ledger is logged."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    prev_card = out / f"{PREV_SESSION.isoformat()}.json"
    prev_card.write_text("invalid json", encoding="utf-8")

    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)

    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True)
    (state_dir / "contest_segment_equity.json").write_text("invalid json", encoding="utf-8")

    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, _sample_bars_down())

    with caplog.at_level("WARNING"):
        assert cmd_contest_segment(_args()) == 0
    assert "ledger_unreadable" in caplog.text


def test_calendar_failure_on_previous_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Calendar raising on previous_session degrades executed_side to None."""
    out = tmp_path / "out"
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)

    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, _sample_bars_down())

    real_cal = CAL

    class _BadCal:
        def next_session(self, day: date) -> date:
            return real_cal.next_session(day)

        def previous_session(self, day: date) -> date:
            raise RuntimeError("no prev")

        def sessions(self, start: date, end: date) -> list[date]:
            return real_cal.sessions(start, end)

    monkeypatch.setattr("src.core.calendar.get_calendar", lambda *a, **k: _BadCal())
    assert cmd_contest_segment(_args()) == 0

