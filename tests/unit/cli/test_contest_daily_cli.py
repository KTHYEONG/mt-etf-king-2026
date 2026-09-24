"""CLI guards for contest-daily (shadow writes, idempotency, state isolation)."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import src.cli.commands.contest as contest_cmd
from src.cli.commands.contest import cmd_contest_daily
from src.contest.daily import DailyShadowCard, GateVerdict, QuoteBar, ShadowAction, archive_quotes

SESSION = date(2026, 9, 29)


def _shadow(out: Path, enabled: bool = True) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "start_session": "2026-09-28",
        "gate_session": "2026-10-08",
        "adopt_from_session": "2026-10-12",
        "output_dir": str(out),
        "quotes_dir": "contest/quotes",
        "yahoo_suffix": ".KS",
        "timeout_s": 20,
        "long_alias": "HY2",
        "inverse_alias": "HY2I",
        "exposures": {"HY2": ["0193T0"], "HY2I": ["0197X0"]},
        "min_streak": 2,
        "review_streak": 3,
        "top_n": 20,
        "gates": {
            "g1_min_identified_share": 0.5,
            "g1_window": 5,
            "g1_min_days": 3,
            "g2_max_daily_churn": 0.15,
            "g3_max_no_data_days": 2,
        },
    }


def _config(out: Path, enabled: bool = True) -> dict[str, Any]:
    return {
        "nickname": "tester",
        "start_date": "2026-09-21",
        "end_date": "2026-11-13",
        "leaderboard": {"archive_dir": "contest/leaderboard"},
        "inference": {"match_tol_pct": 0.02, "min_weight": 0.90},
        "decision": {"target_weight": 0.999},
        "shadow": _shadow(out, enabled),
    }


def _fake_card() -> DailyShadowCard:
    return DailyShadowCard(
        session=SESSION, execution_session=SESSION, action=ShadowAction.RECORD,
        mirror_target="HY2I", long_leader=("a", 1.101), inverse_leader=("b", 1.078),
        our_equity=1.03, our_equity_source="leaderboard",
        identified_top_share=0.6, daily_churn=0.05, review=False,
        gate=GateVerdict.PENDING, gate_detail={"g1": {"value": 1, "pass": True}},
        holders=(), warnings=(),
    )


def _setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cfg: dict[str, Any]
) -> None:
    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: cfg)
    monkeypatch.setattr(
        "src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path)
    )
    monkeypatch.setattr("src.contest.daily.build_daily_card", lambda *a, **k: _fake_card())


def _args(**over: Any) -> argparse.Namespace:
    base: dict[str, Any] = {"session": SESSION.isoformat(), "force": True, "our_return": None}
    base.update(over)
    return argparse.Namespace(**base)


def test_contest_daily_writes_card_and_upserts_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A forced run writes one json/md pair and keeps exactly one metrics row for the session."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    quotes_root = tmp_path / "contest" / "quotes"
    archive_quotes(quotes_root, SESSION, {"0193T0": QuoteBar("0193T0", 100.0, 100.0, 103.0)})
    metrics_path = out / "metrics.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        '{"session": "2026-09-28", "action": "RECORD"}\nbad line\n[1, 2]\n', encoding="utf-8"
    )
    assert contest_cmd.cmd_contest_daily(_args()) == 0
    assert (out / f"{SESSION.isoformat()}.json").is_file()
    assert (out / f"{SESSION.isoformat()}.md").is_file()
    rows = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(1 for row in rows if row["session"] == SESSION.isoformat()) == 1
    assert sum(1 for row in rows if row["session"] == "2026-09-28") == 1
    assert contest_cmd.cmd_contest_daily(_args()) == 0
    rerun = [
        json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(1 for row in rerun if row["session"] == SESSION.isoformat()) == 1


def test_contest_daily_idempotent_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An existing card without --force returns 0 without building."""
    out = tmp_path / "out"
    out.mkdir(parents=True)
    (out / f"{SESSION.isoformat()}.json").write_text("{}", encoding="utf-8")
    _setup(monkeypatch, tmp_path, _config(out))
    monkeypatch.setattr(
        "src.contest.daily.build_daily_card",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not build")),
    )
    assert contest_cmd.cmd_contest_daily(_args(force=False)) == 0


def test_contest_daily_never_writes_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live position state file is byte-identical after a shadow run."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    state_path = tmp_path / "state" / "contest_position.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_bytes(b'{"recommended_alias": "HY2"}')
    assert contest_cmd.cmd_contest_daily(_args()) == 0
    assert state_path.read_bytes() == b'{"recommended_alias": "HY2"}'


def test_contest_daily_disabled_is_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Disabled or misshapen shadow config returns 0 and writes nothing."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out, enabled=False))
    assert contest_cmd.cmd_contest_daily(_args()) == 0
    assert not out.exists()
    bad = _config(out)
    bad["shadow"] = ["not-a-mapping"]
    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: bad)
    assert contest_cmd.cmd_contest_daily(_args()) == 0


def test_contest_daily_parser_registers_subcommand() -> None:
    """contest-daily parses --our-return and dispatches to cmd_contest_daily."""
    from src.cli.parser import build_parser

    args = build_parser().parse_args(["contest-daily", "--our-return", "-3.5"])
    assert args.our_return == -3.5
    assert args.func is cmd_contest_daily


def test_contest_daily_fetch_failure_still_writes_card(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A quotes download failure still writes a card but archives nothing, so a later run can retry the capture."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))

    def _boom(*args: Any, **kwargs: Any) -> dict[str, QuoteBar]:
        raise RuntimeError("no network")

    monkeypatch.setattr("src.contest.daily.fetch_quotes", _boom)
    assert contest_cmd.cmd_contest_daily(_args()) == 0
    assert not (tmp_path / "contest" / "quotes" / f"{SESSION.strftime('%Y%m%d')}.json").exists()
    assert (out / f"{SESSION.isoformat()}.json").is_file()


def test_contest_daily_no_data_card_is_retried(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An existing NO_DATA card does not block the next scheduled run (no --force needed)."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{SESSION.isoformat()}.json").write_text('{"action": "NO_DATA"}\n', encoding="utf-8")
    calls: list[int] = []
    real = contest_cmd.__dict__.get("_read_daily_metrics")

    def _spy(*a: Any, **k: Any) -> Any:
        calls.append(1)
        return real(*a, **k) if real else []

    monkeypatch.setattr(contest_cmd, "_read_daily_metrics", _spy)
    assert contest_cmd.cmd_contest_daily(_args()) == 0
    assert calls


def test_contest_daily_no_session_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No resolvable target session fails closed with rc 1."""
    out = tmp_path / "out"
    _setup(monkeypatch, tmp_path, _config(out))
    monkeypatch.setattr(contest_cmd, "_resolve_target_session", lambda cal, today: None)
    assert contest_cmd.cmd_contest_daily(argparse.Namespace(session=None, force=False, our_return=None)) == 1


def test_contest_daily_persist_failure_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unwritable output directory fails closed with rc 1."""
    out = tmp_path / "out"
    out.write_text("not a dir", encoding="utf-8")
    _setup(monkeypatch, tmp_path, _config(out))
    assert contest_cmd.cmd_contest_daily(_args()) == 1


def test_contest_daily_archives_quotes_covering_both_mirror_sides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A capture quoting both the long and inverse mirror exposures is archived once for the session."""
    out = tmp_path / "out"
    cfg = _config(out)
    _setup(monkeypatch, tmp_path, cfg)
    shadow = cfg["shadow"]
    long_t = shadow["exposures"][shadow["long_alias"]][0]
    inv_t = shadow["exposures"][shadow["inverse_alias"]][0]
    bars = {long_t: QuoteBar(long_t, 100.0, 100.0, 103.0), inv_t: QuoteBar(inv_t, 50.0, 50.0, 48.5)}
    monkeypatch.setattr("src.contest.daily.fetch_quotes", lambda *a, **k: bars)
    assert contest_cmd.cmd_contest_daily(_args()) == 0
    from src.contest.daily import load_quotes

    got = load_quotes(tmp_path / "contest" / "quotes", SESSION)
    assert got is not None and set(got) == {long_t, inv_t}
