"""CLI guards for contest-fade (archived quotes, fail-closed fetch, idempotency)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.cli.commands.contest as contest_cmd
from src.cli.commands.contest import cmd_contest_fade
from src.contest.daily import QuoteBar, archive_quotes
from src.core.calendar import get_calendar

CAL = get_calendar()
SESSION = date(2026, 9, 29)
PREV_SESSION = CAL.previous_session(SESSION)


def _fade(out: Path, enabled: bool = True) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "long_ticker": "0193T0",
        "inverse_ticker": "0197X0",
        "base_alias": "HY2",
        "inverse_alias": "HY2I",
        "threshold_pct": 6.0,
        "borderline_pct": 0.15,
        "mode": "close_return",
        "close_route": "continuous_1515",
        "endgame_sessions": 5,
        "output_dir": str(out),
    }


def _config(out: Path, enabled: bool = True) -> dict[str, Any]:
    return {
        "nickname": "tester",
        "start_date": "2026-09-21",
        "end_date": "2026-11-13",
        "leaderboard": {"archive_dir": "contest/leaderboard"},
        "decision": {"state_name": "contest_position"},
        "shadow": {
            "enabled": True,
            "quotes_dir": "contest/quotes",
            "yahoo_suffix": ".KS",
            "timeout_s": 20,
        },
        "fade": _fade(out, enabled),
    }


def _setup(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: dict[str, Any]) -> None:
    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: cfg)
    monkeypatch.setattr("src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))


def _args(**over: Any) -> argparse.Namespace:
    base: dict[str, Any] = {"session": SESSION.isoformat(), "force": False}
    base.update(over)
    return argparse.Namespace(**base)


def _put_archive(root: Path, day: date, prev: float, close: float) -> None:
    archive_quotes(root, day, {"0193T0": QuoteBar("0193T0", prev, prev, close)})


def _write_state(root: Path, alias: str, name: str = "contest_position") -> None:
    state_dir = root / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / f"{name}.json").write_text(
        json.dumps({"recommended_alias": alias}), encoding="utf-8"
    )


def test_archived_quotes_build_card_without_network(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Archived quotes build card without network."""
    out = tmp_path / "out"
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)
    _write_state(tmp_path, "HY2")
    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, 100.0, 101.0)
    _put_archive(quotes_root, PREV_SESSION, 100.0, 100.5)

    def _boom(*args: Any, **kwargs: Any) -> dict[str, QuoteBar]:
        raise AssertionError("fetch must not be called")

    monkeypatch.setattr("src.contest.daily.fetch_quotes", _boom)
    assert cmd_contest_fade(_args()) == 0
    assert (out / f"{SESSION.isoformat()}.json").is_file()
    assert (out / f"{SESSION.isoformat()}.md").is_file()


def test_fetch_failure_writes_no_data_not_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetch failure writes NO_DATA not error."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    _write_state(tmp_path, "HY2")
    monkeypatch.setattr("src.contest.daily.fetch_quotes", lambda *a, **k: {})
    assert cmd_contest_fade(_args()) == 0
    payload = json.loads((out / f"{SESSION.isoformat()}.json").read_text(encoding="utf-8"))
    assert payload["action"] == "NO_DATA"


def test_shadow_archive_never_written(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Shadow archive never written by contest-fade."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    bars = {"0193T0": QuoteBar("0193T0", 100.0, 100.0, 101.0), "0197X0": QuoteBar("0197X0", 50.0, 50.0, 49.0)}
    monkeypatch.setattr("src.contest.daily.fetch_quotes", lambda *a, **k: bars)
    assert cmd_contest_fade(_args()) == 0
    assert not (tmp_path / "contest" / "quotes" / f"{SESSION.strftime('%Y%m%d')}.json").exists()


def test_existing_non_no_data_card_skipped_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing non-NO_DATA card skipped without force."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    target = out / f"{SESSION.isoformat()}.json"
    target.write_text('{"action": "FADE"}\n', encoding="utf-8")
    before = target.read_bytes()
    _setup(monkeypatch, tmp_path, _config(out))

    def _boom(*args: Any, **kwargs: Any) -> dict[str, QuoteBar]:
        raise AssertionError("must not fetch when skipped")

    monkeypatch.setattr("src.contest.daily.fetch_quotes", _boom)
    assert cmd_contest_fade(_args(force=False)) == 0
    assert target.read_bytes() == before


def test_disabled_block_is_noop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Disabled block is a no-op."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out, enabled=False))
    assert cmd_contest_fade(_args()) == 0
    assert not out.exists()


def test_no_resolvable_session_returns_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No resolvable target session fails closed with rc 1."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    monkeypatch.setattr(contest_cmd, "_resolve_target_session", lambda cal, today: None)
    assert cmd_contest_fade(argparse.Namespace(session=None, force=False)) == 1


def test_non_mapping_shadow_falls_back_to_defaults(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Non-mapping shadow config falls back to default quote paths."""
    out = tmp_path / "out"
    cfg = _config(out)
    cfg["shadow"] = ["not-a-mapping"]
    _setup(monkeypatch, tmp_path, cfg)
    _write_state(tmp_path, "HY2")
    monkeypatch.setattr("src.contest.daily.fetch_quotes", lambda *a, **k: {})
    assert cmd_contest_fade(_args()) == 0
    payload = json.loads((out / f"{SESSION.isoformat()}.json").read_text(encoding="utf-8"))
    assert payload["action"] == "NO_DATA"


def test_fetch_exception_writes_no_data_not_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fetch exception writes NO_DATA not error."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    _write_state(tmp_path, "HY2")

    def _boom(*args: Any, **kwargs: Any) -> dict[str, QuoteBar]:
        raise RuntimeError("no network")

    monkeypatch.setattr("src.contest.daily.fetch_quotes", _boom)
    assert cmd_contest_fade(_args()) == 0
    payload = json.loads((out / f"{SESSION.isoformat()}.json").read_text(encoding="utf-8"))
    assert payload["action"] == "NO_DATA"


def test_missing_previous_session_still_builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Calendar without previous session still builds with unknown history."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    _write_state(tmp_path, "HY2")
    monkeypatch.setattr("src.contest.daily.fetch_quotes", lambda *a, **k: {})

    real_cal = CAL

    class _NoPrev:
        def next_session(self, day: date) -> date:
            return real_cal.next_session(day)

        def previous_session(self, day: date) -> date:
            raise ValueError("no previous session")

        def sessions(self, start: date, end: date) -> list[date]:
            return real_cal.sessions(start, end)

    monkeypatch.setattr("src.core.calendar.get_calendar", lambda *a, **k: _NoPrev())
    assert cmd_contest_fade(_args()) == 0
    payload = json.loads((out / f"{SESSION.isoformat()}.json").read_text(encoding="utf-8"))
    assert payload["action"] == "NO_DATA"


def test_persist_failure_returns_1(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unwritable output directory fails closed with rc 1."""
    out = tmp_path / "out"
    out.write_text("not a dir", encoding="utf-8")
    _setup(monkeypatch, tmp_path, _config(out))
    monkeypatch.setattr("src.contest.daily.fetch_quotes", lambda *a, **k: {})
    assert cmd_contest_fade(_args()) == 1


def test_cli_reads_same_state_file_as_daily(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """CLI reads the same state file contest-daily reads; mismatch forces INACTIVE."""
    import logging

    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    _write_state(tmp_path, "SEMI2")
    quotes_root = tmp_path / "contest" / "quotes"
    _put_archive(quotes_root, SESSION, 100.0, 112.0)
    _put_archive(quotes_root, PREV_SESSION, 100.0, 100.5)

    def _boom(*args: Any, **kwargs: Any) -> dict[str, QuoteBar]:
        raise AssertionError("fetch must not be required for the gate assertion")

    monkeypatch.setattr("src.contest.daily.fetch_quotes", _boom)
    with caplog.at_level(logging.INFO):
        assert cmd_contest_fade(_args()) == 0
    payload = json.loads((out / f"{SESSION.isoformat()}.json").read_text(encoding="utf-8"))
    assert payload["action"] == "INACTIVE"
    assert payload["state_alias"] == "SEMI2"


def test_missing_state_file_yields_inactive_not_crash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing state file yields INACTIVE, not a crash."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    monkeypatch.setattr("src.contest.daily.fetch_quotes", lambda *a, **k: {})
    assert cmd_contest_fade(_args()) == 0
    payload = json.loads((out / f"{SESSION.isoformat()}.json").read_text(encoding="utf-8"))
    assert payload["action"] == "INACTIVE"
    assert "STATE_UNKNOWN" in payload["warnings"]


def test_log_line_includes_state_alias(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Log line includes the state alias."""
    import logging

    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    _write_state(tmp_path, "HY2")
    monkeypatch.setattr("src.contest.daily.fetch_quotes", lambda *a, **k: {})
    with caplog.at_level(logging.INFO):
        assert cmd_contest_fade(_args()) == 0
    records = [r for r in caplog.records if "[PORTFOLIO] contest_fade" in r.getMessage()]
    assert records
    assert "state_alias=HY2" in records[-1].getMessage()
