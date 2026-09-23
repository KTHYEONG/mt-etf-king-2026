"""CLI guards for contest-archive (success, disabled, fail-closed) and seed-reference."""

from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import polars as pl
import pytest

import src.cli.commands.contest as contest_cmd
from src.contest.leaderboard import LeaderboardFetchError, LeaderboardSnapshot
from src.contest.reference import ReferenceFetchError, seed_single_stock_reference


def _contest_dict(enabled: bool = True) -> dict:
    return {
        "enabled": enabled,
        "nickname": "lkthl",
        "leaderboard": {
            "base_url": "https://www.mt.co.kr/etf/array",
            "endpoints": ["etfRankTotal"],
            "timeout_s": 20,
            "archive_dir": "contest/leaderboard",
        },
    }


def _snapshot() -> LeaderboardSnapshot:
    from src.contest.leaderboard import LeaderboardEntry

    return LeaderboardSnapshot(
        base_date=date(2026, 9, 23),
        requested_at="20260923160500",
        entries=(LeaderboardEntry(rank=22, user_name="lkthl", total_return_pct=3.71417, daily_return_pct=0.5),),
        purchases={},
    )


def test_contest_archive_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Successful archive logs our rank and returns 0."""
    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: _contest_dict())
    monkeypatch.setattr(
        "src.core.settings.get_settings",
        lambda: SimpleNamespace(data_root=tmp_path),
    )
    monkeypatch.setattr(
        "src.contest.leaderboard.fetch_leaderboard_payloads",
        lambda *a, **k: {"etfRankTotal": {"baseDt": "20260923"}},
    )
    monkeypatch.setattr(
        "src.contest.leaderboard.archive_leaderboard",
        lambda payloads, root: (date(2026, 9, 23), True),
    )
    monkeypatch.setattr("src.contest.leaderboard.load_snapshot", lambda root, day: _snapshot())
    with caplog.at_level("INFO"):
        rc = contest_cmd.cmd_contest_archive(argparse.Namespace())
    assert rc == 0
    assert any("contest_archive" in r.message and "our_rank=22" in r.message for r in caplog.records)


def test_contest_archive_disabled(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    """Disabled contest short-circuits with rc 0."""
    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: _contest_dict(enabled=False))
    with caplog.at_level("INFO"):
        rc = contest_cmd.cmd_contest_archive(argparse.Namespace())
    assert rc == 0
    assert any("contest disabled" in r.message for r in caplog.records)


def test_contest_archive_fetch_failure_returns_1(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Fetch failure is fail-closed with rc 1."""

    def _boom(*a: object, **k: object) -> dict:
        raise LeaderboardFetchError("down")

    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: _contest_dict())
    monkeypatch.setattr("src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr("src.contest.leaderboard.fetch_leaderboard_payloads", _boom)
    rc = contest_cmd.cmd_contest_archive(argparse.Namespace())
    assert rc == 1


def test_contest_config_loads_real_file() -> None:
    """The real contest.yaml parses to the expected nickname and endpoints."""
    contest = contest_cmd._contest_config()
    assert contest["nickname"] == "lkthl"
    assert "etfRankTotal" in list(contest["leaderboard"]["endpoints"])


def test_contest_config_missing_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    """A contest.yaml without the contest mapping raises."""
    monkeypatch.setattr("src.core.config.load_config", lambda name: {})
    with pytest.raises(ValueError, match=r"contest.*mapping"):
        contest_cmd._contest_config()


def test_contest_archive_config_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unreadable contest config fails closed with rc 1."""

    def _boom() -> dict:
        raise ValueError("bad config")

    monkeypatch.setattr(contest_cmd, "_contest_config", _boom)
    assert contest_cmd.cmd_contest_archive(argparse.Namespace()) == 1


def test_contest_archive_bad_leaderboard_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A non-mapping leaderboard section fails closed with rc 1."""
    bad = _contest_dict()
    bad["leaderboard"] = ["not-a-mapping"]  # type: ignore[assignment]
    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: bad)
    monkeypatch.setattr("src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    assert contest_cmd.cmd_contest_archive(argparse.Namespace()) == 1


def test_contest_archive_unexpected_fetch_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A non-fetch exception during download still fails closed with rc 1."""

    def _boom(*a: object, **k: object) -> dict:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: _contest_dict())
    monkeypatch.setattr("src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr("src.contest.leaderboard.fetch_leaderboard_payloads", _boom)
    assert contest_cmd.cmd_contest_archive(argparse.Namespace()) == 1


def test_contest_archive_schema_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Schema violations during archiving fail closed with rc 1."""
    from src.contest.leaderboard import LeaderboardSchemaError

    def _boom(payloads: object, root: object) -> tuple:
        raise LeaderboardSchemaError("bad rows")

    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: _contest_dict())
    monkeypatch.setattr("src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr("src.contest.leaderboard.fetch_leaderboard_payloads", lambda *a, **k: {})
    monkeypatch.setattr("src.contest.leaderboard.archive_leaderboard", _boom)
    assert contest_cmd.cmd_contest_archive(argparse.Namespace()) == 1


def test_contest_archive_write_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unexpected archive errors fail closed with rc 1."""

    def _boom(payloads: object, root: object) -> tuple:
        raise OSError("disk full")

    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: _contest_dict())
    monkeypatch.setattr("src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr("src.contest.leaderboard.fetch_leaderboard_payloads", lambda *a, **k: {})
    monkeypatch.setattr("src.contest.leaderboard.archive_leaderboard", _boom)
    assert contest_cmd.cmd_contest_archive(argparse.Namespace()) == 1


def test_contest_archive_load_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A snapshot that cannot be reloaded after archiving fails closed with rc 1."""

    def _boom(root: object, day: object) -> LeaderboardSnapshot:
        raise FileNotFoundError("gone")

    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: _contest_dict())
    monkeypatch.setattr("src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr("src.contest.leaderboard.fetch_leaderboard_payloads", lambda *a, **k: {})
    monkeypatch.setattr("src.contest.leaderboard.archive_leaderboard", lambda p, r: (date(2026, 9, 23), True))
    monkeypatch.setattr("src.contest.leaderboard.load_snapshot", _boom)
    assert contest_cmd.cmd_contest_archive(argparse.Namespace()) == 1


def _yahoo_payload(rows: list[tuple[str, float | None, float | None]]) -> dict[str, Any]:
    ts = [int(datetime.fromisoformat(d).replace(tzinfo=UTC).timestamp()) for d, _, _ in rows]
    return {
        "chart": {
            "result": [
                {
                    "timestamp": ts,
                    "indicators": {
                        "quote": [
                            {"open": [o for _, o, _ in rows], "close": [c for _, _, c in rows]},
                        ]
                    },
                }
            ]
        }
    }


def _patch_yahoo(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)


def test_seed_single_stock_reference_writes_parquet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Downloaded OHLC lands in parquet with KST dates at or before the cutoff."""
    payload = _yahoo_payload(
        [
            ("2026-05-24T00:00:00", 100.0, 101.0),
            ("2026-05-25T00:00:00", None, 102.0),
            ("2026-05-26T00:00:00", 102.0, 103.0),
            ("2026-05-27T00:00:00", 103.0, 104.0),
        ]
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    _patch_yahoo(monkeypatch, handler)
    out = tmp_path / "ref" / "single_stock_daily.parquet"
    n = seed_single_stock_reference({"HYNIX": "000660.KS"}, date(2026, 5, 26), out, 30.0)
    assert n == 2
    frame = pl.read_parquet(out)
    assert frame.columns == ["date", "symbol", "open", "close"]
    assert frame.get_column("symbol").unique().to_list() == ["HYNIX"]
    assert frame.get_column("date").max() == date(2026, 5, 26)
    assert list(tmp_path.rglob("*.tmp")) == [] and list(out.parent.iterdir()) == [out]


def test_seed_single_stock_reference_fetch_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An HTTP error fails closed with nothing written."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={})

    _patch_yahoo(monkeypatch, handler)
    out = tmp_path / "single_stock_daily.parquet"
    with pytest.raises(ReferenceFetchError):
        seed_single_stock_reference({"HYNIX": "000660.KS"}, date(2026, 5, 26), out, 30.0)
    assert not out.exists()


def test_seed_single_stock_reference_bad_payload(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Malformed chart JSON and empty symbol maps fail closed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"chart": {}})

    _patch_yahoo(monkeypatch, handler)
    out = tmp_path / "single_stock_daily.parquet"
    with pytest.raises(ReferenceFetchError):
        seed_single_stock_reference({"HYNIX": "000660.KS"}, date(2026, 5, 26), out, 30.0)
    with pytest.raises(ReferenceFetchError, match=r"no symbols"):
        seed_single_stock_reference({}, date(2026, 5, 26), out, 30.0)
    assert not out.exists()


def test_seed_single_stock_reference_no_usable_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A cutoff before all data fails closed with nothing written."""
    payload = _yahoo_payload([("2026-05-27T00:00:00", 103.0, 104.0)])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    _patch_yahoo(monkeypatch, handler)
    out = tmp_path / "single_stock_daily.parquet"
    with pytest.raises(ReferenceFetchError, match=r"no usable rows"):
        seed_single_stock_reference({"HYNIX": "000660.KS"}, date(2026, 1, 1), out, 30.0)
    assert not out.exists()


def test_seed_single_stock_reference_transport_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Unexpected client errors surface as ReferenceFetchError."""

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        raise RuntimeError("no network")

    monkeypatch.setattr(httpx, "Client", factory)
    with pytest.raises(ReferenceFetchError):
        seed_single_stock_reference({"HYNIX": "000660.KS"}, date(2026, 5, 26), tmp_path / "x.parquet", 30.0)


def test_seed_single_stock_reference_write_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A parquet write failure cleans up the temp file and raises."""
    payload = _yahoo_payload([("2026-05-24T00:00:00", 100.0, 101.0)])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    _patch_yahoo(monkeypatch, handler)

    def _boom(self: object, *args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", _boom)
    out = tmp_path / "single_stock_daily.parquet"
    with pytest.raises(ReferenceFetchError, match=r"write failed"):
        seed_single_stock_reference({"HYNIX": "000660.KS"}, date(2026, 5, 26), out, 30.0)
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def _seed_dict() -> dict:
    base = _contest_dict()
    base["reference"] = {
        "single_stock_path": "reference/single_stock_daily.parquet",
        "symbols": {"HYNIX": "000660.KS"},
        "cutoff": "2026-05-26",
        "timeout_s": 30,
    }
    return base


def test_contest_seed_reference_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Successful seeding returns 0 and logs the row count."""
    monkeypatch.setattr(contest_cmd, "_contest_config", _seed_dict)
    monkeypatch.setattr("src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr("src.contest.reference.seed_single_stock_reference", lambda *a, **k: 42)
    with caplog.at_level("INFO"):
        assert contest_cmd.cmd_contest_seed_reference(argparse.Namespace()) == 0
    assert any("rows=42" in r.message for r in caplog.records)


def test_contest_seed_reference_fetch_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A download failure returns 1."""

    def _boom(*a: object, **k: object) -> int:
        raise ReferenceFetchError("down")

    monkeypatch.setattr(contest_cmd, "_contest_config", _seed_dict)
    monkeypatch.setattr("src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr("src.contest.reference.seed_single_stock_reference", _boom)
    assert contest_cmd.cmd_contest_seed_reference(argparse.Namespace()) == 1


def test_contest_seed_reference_bad_config(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing reference mapping and unexpected errors return 1."""
    bad = _contest_dict()
    bad["reference"] = ["not-a-mapping"]
    monkeypatch.setattr(contest_cmd, "_contest_config", lambda: bad)
    assert contest_cmd.cmd_contest_seed_reference(argparse.Namespace()) == 1

    def _boom() -> dict:
        raise ValueError("bad config")

    monkeypatch.setattr(contest_cmd, "_contest_config", _boom)
    assert contest_cmd.cmd_contest_seed_reference(argparse.Namespace()) == 1


def test_contest_seed_reference_unexpected_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A non-fetch exception during seeding still fails closed with rc 1."""

    def _boom(*a: object, **k: object) -> int:
        raise RuntimeError("unexpected")

    monkeypatch.setattr(contest_cmd, "_contest_config", _seed_dict)
    monkeypatch.setattr("src.core.settings.get_settings", lambda: SimpleNamespace(data_root=tmp_path))
    monkeypatch.setattr("src.contest.reference.seed_single_stock_reference", _boom)
    assert contest_cmd.cmd_contest_seed_reference(argparse.Namespace()) == 1
