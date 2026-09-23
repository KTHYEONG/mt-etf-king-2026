"""Invariant guards for the contest leaderboard data layer."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.contest.leaderboard import (
    LeaderboardConflictError,
    LeaderboardFetchError,
    LeaderboardSchemaError,
    archive_leaderboard,
    fetch_leaderboard_payloads,
    infer_single_vehicle_holders,
    latest_snapshot_on_or_before,
    load_snapshot,
)

ENDPOINTS = [
    "etfRankTotal",
    "etfRankGroupTop",
    "etfRankProductPurchase",
    "etfRankGroupTopMonth",
    "etfGuideArticle",
]


def _row(rank: int, name: str, total: float, daily: float) -> dict[str, Any]:
    return {"rank": rank, "userName": name, "totalReturnRate": total, "dailyReturnRate": daily}


def _payloads(
    base_dt: str = "20260923",
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    if rows is None:
        rows = [
            _row(1, "leader", 8.5, 1.25),
            _row(22, "lkthl", 3.71417, 0.5),
        ]
    return {
        "etfRankTotal": {"baseDt": base_dt, "reqDateTime": f"{base_dt}160500", "data": rows},
        "etfRankGroupTop": {"baseDt": base_dt, "data": []},
        "etfRankProductPurchase": {"baseDt": base_dt, "data": {}},
        "etfRankGroupTopMonth": {"baseDt": base_dt, "data": []},
        "etfGuideArticle": {"baseDt": base_dt, "data": []},
    }


def test_archive_writes_once_per_base_dt(tmp_path: Path) -> None:
    """Archive writes once per baseDt; second identical archive is a no-op."""
    payloads = _payloads()
    first = archive_leaderboard(payloads, tmp_path)
    assert first == (date(2026, 9, 23), True)
    second = archive_leaderboard(payloads, tmp_path)
    assert second == (date(2026, 9, 23), False)
    day_dir = tmp_path / "20260923"
    assert sorted(p.name for p in day_dir.iterdir()) == sorted(f"{e}.json" for e in ENDPOINTS)


def test_archive_refuses_conflicting_rewrite(tmp_path: Path) -> None:
    """Conflicting rewrite of the same baseDt is refused and originals stay intact."""
    payloads = _payloads()
    archive_leaderboard(payloads, tmp_path)
    day_dir = tmp_path / "20260923"
    before = {p.name: p.read_bytes() for p in day_dir.iterdir()}
    other = _payloads(rows=[_row(1, "leader", 9.99, 2.0), _row(22, "lkthl", 3.0, 0.1)])
    with pytest.raises(LeaderboardConflictError):
        archive_leaderboard(other, tmp_path)
    after = {p.name: p.read_bytes() for p in day_dir.iterdir()}
    assert before == after


def test_archive_schema_violation_fails_closed(tmp_path: Path) -> None:
    """Missing data key fails closed without creating any directory."""
    payloads = _payloads()
    payloads["etfRankTotal"] = {"baseDt": "20260923"}
    with pytest.raises(LeaderboardSchemaError):
        archive_leaderboard(payloads, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_fetch_failure_is_all_or_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    """One failing endpoint aborts the whole fetch with no partial dict."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "etfRankGroupTop.json" in str(request.url):
            return httpx.Response(500, json={})
        return httpx.Response(200, json={"baseDt": "20260923", "data": []})

    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)
    with pytest.raises(LeaderboardFetchError):
        fetch_leaderboard_payloads("https://www.mt.co.kr/etf/array", ENDPOINTS, 20.0)


def _patch_client(monkeypatch: pytest.MonkeyPatch, handler: Any) -> None:
    transport = httpx.MockTransport(handler)
    real_client = httpx.Client

    def factory(*args: Any, **kwargs: Any) -> httpx.Client:
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)


def test_fetch_success_returns_all_payloads(monkeypatch: pytest.MonkeyPatch) -> None:
    """All endpoints returning objects yield the full payload dict."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"baseDt": "20260923", "data": []})

    _patch_client(monkeypatch, handler)
    out = fetch_leaderboard_payloads("https://www.mt.co.kr/etf/array", ENDPOINTS, 20.0)
    assert sorted(out) == sorted(ENDPOINTS)


def test_fetch_non_json_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-JSON body fails the whole fetch closed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    _patch_client(monkeypatch, handler)
    with pytest.raises(LeaderboardFetchError):
        fetch_leaderboard_payloads("https://www.mt.co.kr/etf/array", ENDPOINTS, 20.0)


def test_fetch_non_object_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """A JSON array body fails the whole fetch closed."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=[1, 2])

    _patch_client(monkeypatch, handler)
    with pytest.raises(LeaderboardFetchError):
        fetch_leaderboard_payloads("https://www.mt.co.kr/etf/array", ENDPOINTS, 20.0)


def test_archive_missing_total_payload(tmp_path: Path) -> None:
    """Payloads without etfRankTotal fail closed."""
    payloads = _payloads()
    del payloads["etfRankTotal"]
    with pytest.raises(LeaderboardSchemaError):
        archive_leaderboard(payloads, tmp_path)


@pytest.mark.parametrize("base_dt", ["", "2026-9-3", "20269999", "abcdefgh"])
def test_archive_invalid_base_dt(tmp_path: Path, base_dt: str) -> None:
    """Malformed baseDt values fail closed without creating directories."""
    payloads = _payloads(base_dt=base_dt)
    with pytest.raises(LeaderboardSchemaError):
        archive_leaderboard(payloads, tmp_path)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "rows",
    [
        ["not-a-mapping"],
        [_row(1, "a", 1.0, 0.5), {"rank": 2, "userName": "b", "totalReturnRate": 1.0}],
        [_row(1, "a", 1.0, 0.5), {"rank": "x", "userName": "b", "totalReturnRate": 1.0, "dailyReturnRate": 0.1}],
        [],
    ],
)
def test_archive_row_violations(tmp_path: Path, rows: list[Any]) -> None:
    """Non-object, incomplete, mistyped, or empty rows fail closed."""
    payloads = _payloads(rows=rows)  # type: ignore[arg-type]
    with pytest.raises(LeaderboardSchemaError):
        archive_leaderboard(payloads, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_archive_conflict_other_endpoint(tmp_path: Path) -> None:
    """Identical etfRankTotal but differing sidecar content is a conflict."""
    archive_leaderboard(_payloads(), tmp_path)
    (tmp_path / "20260923" / "etfRankGroupTop.json").write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(LeaderboardConflictError):
        archive_leaderboard(_payloads(), tmp_path)


def test_archive_dir_missing_total_file(tmp_path: Path) -> None:
    """A pre-existing baseDt dir without etfRankTotal is a conflict, never adopted."""
    day_dir = tmp_path / "20260923"
    day_dir.mkdir(parents=True)
    (day_dir / "etfRankGroupTop.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(LeaderboardConflictError):
        archive_leaderboard(_payloads(), tmp_path)


def test_our_entry_lookup(tmp_path: Path) -> None:
    """entry_for returns our exact userName match and None when absent."""
    archive_leaderboard(_payloads(), tmp_path)
    snapshot = load_snapshot(tmp_path, date(2026, 9, 23))
    entry = snapshot.entry_for("lkthl")
    assert entry is not None
    assert entry.rank == 22
    assert entry.total_return_pct == pytest.approx(3.71417)
    assert snapshot.entry_for("absent") is None


def test_single_vehicle_inference_matches_pure_holders(tmp_path: Path) -> None:
    """Pure single-vehicle dailies map to the right vehicle with implied weight."""
    rows = [
        _row(1, "semi_holder", 10.0, 6.13349),
        _row(2, "hy2i_holder", 5.0, -2.50895),
        _row(3, "k2_holder", 4.0, 2.13954),
    ]
    archive_leaderboard(_payloads(rows=rows), tmp_path)
    snapshot = load_snapshot(tmp_path, date(2026, 9, 23))
    changes = {"SEMI2": 6.13358, "HY2I": -2.50896, "K2": 2.25087}
    out = infer_single_vehicle_holders(snapshot, changes, tol_pct=0.02, min_weight=0.90)
    assert out["semi_holder"][0] == "SEMI2"
    assert out["semi_holder"][1] == pytest.approx(1.0, abs=0.002)
    assert out["hy2i_holder"][0] == "HY2I"
    assert out["hy2i_holder"][1] == pytest.approx(1.0, abs=0.002)
    assert out["k2_holder"][0] == "K2"
    assert out["k2_holder"][1] == pytest.approx(0.9505, abs=0.002)


def test_ambiguous_entries_omitted(tmp_path: Path) -> None:
    """A daily explainable by two vehicles is omitted from the result."""
    rows = [_row(1, "mixed", 7.0, 2.0)]
    archive_leaderboard(_payloads(rows=rows), tmp_path)
    snapshot = load_snapshot(tmp_path, date(2026, 9, 23))
    out = infer_single_vehicle_holders(snapshot, {"A": 2.0, "B": 2.01}, tol_pct=0.02, min_weight=0.90)
    assert "mixed" not in out


def test_latest_snapshot_lookup_respects_session(tmp_path: Path) -> None:
    """Latest lookup returns the newest snapshot on or before the session."""
    for day in ("20260921", "20260922", "20260923"):
        archive_leaderboard(_payloads(base_dt=day), tmp_path)
    got = latest_snapshot_on_or_before(tmp_path, date(2026, 9, 22))
    assert got is not None
    assert got.base_date == date(2026, 9, 22)
    assert latest_snapshot_on_or_before(tmp_path, date(2026, 9, 20)) is None


def test_load_snapshot_missing_date(tmp_path: Path) -> None:
    """Loading a never-archived baseDt raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_snapshot(tmp_path, date(2026, 9, 23))


def test_load_snapshot_non_object_total(tmp_path: Path) -> None:
    """An archived non-object etfRankTotal raises a schema error on load."""
    archive_leaderboard(_payloads(), tmp_path)
    (tmp_path / "20260923" / "etfRankTotal.json").write_text("[1, 2]\n", encoding="utf-8")
    with pytest.raises(LeaderboardSchemaError):
        load_snapshot(tmp_path, date(2026, 9, 23))


def test_purchases_shapes_loaded(tmp_path: Path) -> None:
    """Division-mapped purchases load per division; malformed rows are skipped."""
    payloads = _payloads()
    payloads["etfRankProductPurchase"] = {
        "baseDt": "20260923",
        "data": {
            "auto": [
                {"productName": "X", "purchaseCount": 100.0, "priceChangeRate": 1.5},
                {"productName": "broken"},
                "junk",
            ],
            "free": "not-a-list",
        },
    }
    archive_leaderboard(payloads, tmp_path)
    snapshot = load_snapshot(tmp_path, date(2026, 9, 23))
    assert snapshot.purchases == {"auto": (("X", 100.0, 1.5),)}


def test_purchases_list_shape_loaded(tmp_path: Path) -> None:
    """A list-shaped purchases payload loads under the catch-all division."""
    payloads = _payloads()
    payloads["etfRankProductPurchase"] = {
        "baseDt": "20260923",
        "data": [{"productName": "Y", "purchaseCount": 10.0, "priceChangeRate": -0.5}],
    }
    archive_leaderboard(payloads, tmp_path)
    snapshot = load_snapshot(tmp_path, date(2026, 9, 23))
    assert snapshot.purchases == {"all": (("Y", 10.0, -0.5),)}


@pytest.mark.parametrize("data", [{}, "junk"])
def test_purchases_empty_shapes_loaded(tmp_path: Path, data: Any) -> None:
    """Empty or scalar purchases payloads load as no purchases."""
    payloads = _payloads()
    payloads["etfRankProductPurchase"] = {"baseDt": "20260923", "data": data}
    archive_leaderboard(payloads, tmp_path)
    snapshot = load_snapshot(tmp_path, date(2026, 9, 23))
    assert snapshot.purchases == {}


def test_purchases_non_mapping_file_loaded(tmp_path: Path) -> None:
    """A non-mapping purchases file loads as no purchases."""
    archive_leaderboard(_payloads(), tmp_path)
    (tmp_path / "20260923" / "etfRankProductPurchase.json").write_text("[1]\n", encoding="utf-8")
    snapshot = load_snapshot(tmp_path, date(2026, 9, 23))
    assert snapshot.purchases == {}


def test_latest_snapshot_missing_root(tmp_path: Path) -> None:
    """Lookup under a nonexistent archive root returns None."""
    assert latest_snapshot_on_or_before(tmp_path / "nope", date(2026, 9, 23)) is None


def test_latest_snapshot_ignores_stray_dirs(tmp_path: Path) -> None:
    """Stray files and non-calendar directories never shadow real snapshots."""
    archive_leaderboard(_payloads(base_dt="20260922"), tmp_path)
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "20269999").mkdir()
    (tmp_path / "20260999").mkdir()
    got = latest_snapshot_on_or_before(tmp_path, date(2026, 9, 23))
    assert got is not None
    assert got.base_date == date(2026, 9, 22)
