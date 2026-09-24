"""Invariant guards for the contest leaderboard data layer."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone, UTC
from pathlib import Path
from typing import Any

import httpx
import pytest

from src.contest.leaderboard import (
    ArchiveOutcome,
    LeaderboardConflictError,
    LeaderboardFetchError,
    LeaderboardSchemaError,
    archive_leaderboard,
    fetch_leaderboard_payloads,
    infer_single_vehicle_holders,
    latest_entry_on_or_before,
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
        "etfRankTotal": {"baseDt": base_dt, "reqDateTime": "2026/09/23 16:00", "data": rows},
        "etfRankGroupTop": {"baseDt": base_dt, "reqDateTime": "2026/09/23 16:00", "data": []},
        "etfRankProductPurchase": {"baseDt": base_dt, "reqDateTime": "2026/09/23 16:00", "data": {}},
        "etfRankGroupTopMonth": {"baseDt": "202412", "reqDateTime": "2026/09/23 16:00", "data": []},
        "etfGuideArticle": {"data": []},
    }


def _with_req_date_time(
    payloads: dict[str, dict[str, Any]], req: str
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for endpoint, body in payloads.items():
        copied = dict(body)
        if "reqDateTime" in copied:
            copied["reqDateTime"] = req
        out[endpoint] = copied
    return out


def test_archive_writes_once_per_base_dt(tmp_path: Path) -> None:
    """Archive writes once per baseDt; second identical archive is a no-op."""
    payloads = _payloads()
    first = archive_leaderboard(payloads, tmp_path)
    assert first == (date(2026, 9, 23), ArchiveOutcome.WRITTEN)
    second = archive_leaderboard(payloads, tmp_path)
    assert second == (date(2026, 9, 23), ArchiveOutcome.UNCHANGED)
    day_dir = tmp_path / "20260923"
    assert sorted(p.name for p in day_dir.iterdir()) == sorted(f"{e}.json" for e in ENDPOINTS)


def test_volatile_metadata_refetch_is_unchanged(tmp_path: Path) -> None:
    """Holiday re-fetch differing only in reqDateTime is unchanged with untouched bytes."""
    payloads = _payloads()
    archive_leaderboard(payloads, tmp_path)
    day_dir = tmp_path / "20260923"
    before = {p.name: p.read_bytes() for p in day_dir.iterdir()}
    refetched = _with_req_date_time(_payloads(), "2026/09/24 16:00")
    outcome = archive_leaderboard(refetched, tmp_path)
    assert outcome == (date(2026, 9, 23), ArchiveOutcome.UNCHANGED)
    assert not (day_dir / "revisions").exists()
    after = {p.name: p.read_bytes() for p in day_dir.iterdir()}
    assert before == after


def test_holiday_refetch_without_req_on_one_endpoint(tmp_path: Path) -> None:
    """etfGuideArticle carries no reqDateTime; changing others' reqDateTime is still unchanged."""
    archive_leaderboard(_payloads(), tmp_path)
    refetched = _with_req_date_time(_payloads(), "2026/09/24 16:00")
    assert "reqDateTime" not in refetched["etfGuideArticle"]
    assert archive_leaderboard(refetched, tmp_path) == (date(2026, 9, 23), ArchiveOutcome.UNCHANGED)


def test_differing_rows_stored_as_revision(tmp_path: Path) -> None:
    """Genuinely different data for the same baseDt is stored as a timestamped revision."""
    payloads = _payloads()
    archive_leaderboard(payloads, tmp_path)
    day_dir = tmp_path / "20260923"
    before = {p.name: p.read_bytes() for p in day_dir.iterdir() if p.is_file()}
    other = _with_req_date_time(
        _payloads(rows=[_row(1, "leader", 9.99, 2.0), _row(22, "lkthl", 3.0, 0.1)]),
        "2026/09/24 16:00",
    )
    fetched_at = datetime(2026, 9, 24, 7, 41, 15, tzinfo=UTC)
    result = archive_leaderboard(other, tmp_path, fetched_at=fetched_at)
    assert result == (date(2026, 9, 23), ArchiveOutcome.REVISION_STORED)
    revision_dir = day_dir / "revisions" / "20260924T074115Z"
    assert revision_dir.is_dir()
    assert sorted(p.name for p in revision_dir.iterdir()) == sorted(f"{e}.json" for e in ENDPOINTS)
    for endpoint, body in other.items():
        stored = json.loads((revision_dir / f"{endpoint}.json").read_text(encoding="utf-8"))
        assert stored == body
        assert stored.get("reqDateTime") == body.get("reqDateTime")
    for name, raw in before.items():
        assert (day_dir / name).read_bytes() == raw
    snapshot = load_snapshot(tmp_path, date(2026, 9, 23))
    assert snapshot.entries[0].total_return_pct == pytest.approx(8.5)


def test_repeated_identical_revision_deduplicated(tmp_path: Path) -> None:
    """Re-archiving stored revision content (new reqDateTime, later fetched_at) is unchanged."""
    archive_leaderboard(_payloads(), tmp_path)
    other = _payloads(rows=[_row(1, "leader", 9.99, 2.0), _row(22, "lkthl", 3.0, 0.1)])
    archive_leaderboard(other, tmp_path, fetched_at=datetime(2026, 9, 24, 7, 41, 15, tzinfo=UTC))
    again = _with_req_date_time(other, "2026/09/24 17:41")
    result = archive_leaderboard(again, tmp_path, fetched_at=datetime(2026, 9, 24, 8, 41, 15, tzinfo=UTC))
    assert result == (date(2026, 9, 23), ArchiveOutcome.UNCHANGED)
    revisions = list((tmp_path / "20260923" / "revisions").iterdir())
    assert len(revisions) == 1


def test_distinct_second_revision(tmp_path: Path) -> None:
    """A third distinct content set stores a second revision directory."""
    archive_leaderboard(_payloads(), tmp_path)
    second = _payloads(rows=[_row(1, "leader", 9.99, 2.0), _row(22, "lkthl", 3.0, 0.1)])
    third = _payloads(rows=[_row(1, "leader", 7.77, 1.0), _row(22, "lkthl", 3.0, 0.1)])
    archive_leaderboard(second, tmp_path, fetched_at=datetime(2026, 9, 24, 7, 41, 15, tzinfo=UTC))
    result = archive_leaderboard(third, tmp_path, fetched_at=datetime(2026, 9, 24, 9, 0, 0, tzinfo=UTC))
    assert result == (date(2026, 9, 23), ArchiveOutcome.REVISION_STORED)
    assert len(list((tmp_path / "20260923" / "revisions").iterdir())) == 2


def test_auxiliary_only_change_becomes_revision(tmp_path: Path) -> None:
    """Identical etfRankTotal but differing sidecar content is stored as a revision."""
    archive_leaderboard(_payloads(), tmp_path)
    day_dir = tmp_path / "20260923"
    before_others = {
        p.name: p.read_bytes() for p in day_dir.iterdir() if p.name != "etfGuideArticle.json"
    }
    changed = _payloads()
    changed["etfGuideArticle"] = {"data": [{"title": "new"}]}
    result = archive_leaderboard(changed, tmp_path, fetched_at=datetime(2026, 9, 24, 7, 0, 0, tzinfo=UTC))
    assert result == (date(2026, 9, 23), ArchiveOutcome.REVISION_STORED)
    for name, raw in before_others.items():
        assert (day_dir / name).read_bytes() == raw


def test_tampered_sidecar_is_preserved_as_revision(tmp_path: Path) -> None:
    """An altered archived sidecar is never repaired; the canonical payload becomes a revision."""
    archive_leaderboard(_payloads(), tmp_path)
    (tmp_path / "20260923" / "etfRankGroupTop.json").write_text('{"tampered": true}\n', encoding="utf-8")
    result = archive_leaderboard(
        _payloads(), tmp_path, fetched_at=datetime(2026, 9, 24, 7, 0, 0, tzinfo=UTC)
    )
    assert result == (date(2026, 9, 23), ArchiveOutcome.REVISION_STORED)
    assert (tmp_path / "20260923" / "etfRankGroupTop.json").read_text(encoding="utf-8") == '{"tampered": true}\n'


def test_unparseable_sidecar_becomes_revision(tmp_path: Path) -> None:
    """A sidecar with invalid JSON counts as differing and is preserved via revision."""
    archive_leaderboard(_payloads(), tmp_path)
    (tmp_path / "20260923" / "etfRankGroupTop.json").write_text("not json", encoding="utf-8")
    result = archive_leaderboard(
        _payloads(), tmp_path, fetched_at=datetime(2026, 9, 24, 7, 0, 0, tzinfo=UTC)
    )
    assert result == (date(2026, 9, 23), ArchiveOutcome.REVISION_STORED)
    assert (tmp_path / "20260923" / "etfRankGroupTop.json").read_text(encoding="utf-8") == "not json"


def test_non_mapping_sidecar_becomes_revision(tmp_path: Path) -> None:
    """A sidecar with valid non-object JSON counts as differing content."""
    archive_leaderboard(_payloads(), tmp_path)
    (tmp_path / "20260923" / "etfRankProductPurchase.json").write_text("[1, 2]\n", encoding="utf-8")
    result = archive_leaderboard(
        _payloads(), tmp_path, fetched_at=datetime(2026, 9, 24, 8, 0, 0, tzinfo=UTC)
    )
    assert result == (date(2026, 9, 23), ArchiveOutcome.REVISION_STORED)


def test_missing_endpoint_in_original_counts_as_differing(tmp_path: Path) -> None:
    """An endpoint absent from the original archive counts as differing content."""
    partial = _payloads()
    del partial["etfRankGroupTop"]
    archive_leaderboard(partial, tmp_path)
    result = archive_leaderboard(
        _payloads(), tmp_path, fetched_at=datetime(2026, 9, 24, 7, 0, 0, tzinfo=UTC)
    )
    assert result == (date(2026, 9, 23), ArchiveOutcome.REVISION_STORED)


def test_revision_name_collision_fails_closed(tmp_path: Path) -> None:
    """A revision directory name taken by different content raises without overwriting."""
    archive_leaderboard(_payloads(), tmp_path)
    other = _payloads(rows=[_row(1, "leader", 9.99, 2.0), _row(22, "lkthl", 3.0, 0.1)])
    fetched_at = datetime(2026, 9, 24, 7, 41, 15, tzinfo=UTC)
    archive_leaderboard(other, tmp_path, fetched_at=fetched_at)
    clash = _payloads(rows=[_row(1, "leader", 1.11, 0.1), _row(22, "lkthl", 3.0, 0.1)])
    with pytest.raises(LeaderboardConflictError):
        archive_leaderboard(clash, tmp_path, fetched_at=fetched_at)
    assert len(list((tmp_path / "20260923" / "revisions").iterdir())) == 1


def test_new_base_dt_appears_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A crash mid-write never leaves a partial baseDt directory behind."""
    from pathlib import Path as _Path

    real_write = _Path.write_text
    calls = {"n": 0}

    def _boom(self: Path, *args: Any, **kwargs: Any) -> Any:
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("disk full")
        return real_write(self, *args, **kwargs)

    monkeypatch.setattr(_Path, "write_text", _boom)
    with pytest.raises(OSError, match="disk full"):
        archive_leaderboard(_payloads(), tmp_path)
    assert not (tmp_path / "20260923").exists()
    monkeypatch.undo()
    assert archive_leaderboard(_payloads(), tmp_path) == (date(2026, 9, 23), ArchiveOutcome.WRITTEN)


def test_staging_and_revisions_invisible_to_readers(tmp_path: Path) -> None:
    """Leftover staging dirs and revisions never shadow reader lookups."""
    archive_leaderboard(_payloads(base_dt="20260922"), tmp_path)
    archive_leaderboard(_payloads(base_dt="20260923"), tmp_path)
    (tmp_path / ".staging-20260923-xyz").mkdir()
    (tmp_path / ".staging-20260923-xyz" / "etfRankTotal.json").write_text("{}\n", encoding="utf-8")
    other = _payloads(rows=[_row(1, "leader", 9.99, 2.0), _row(22, "lkthl", 3.0, 0.1)])
    archive_leaderboard(other, tmp_path, fetched_at=datetime(2026, 9, 24, 7, 41, 15, tzinfo=UTC))
    got = latest_snapshot_on_or_before(tmp_path, date(2026, 9, 23))
    assert got is not None
    assert got.base_date == date(2026, 9, 23)
    assert got.entries[0].total_return_pct == pytest.approx(8.5)
    hit = latest_entry_on_or_before(tmp_path, "lkthl", date(2026, 9, 23))
    assert hit is not None
    assert hit[0] == date(2026, 9, 23)


def test_base_dt_path_as_file_fails_closed(tmp_path: Path) -> None:
    """A baseDt path that is a file (not a directory) is a structural conflict."""
    (tmp_path / "20260923").write_text("junk", encoding="utf-8")
    with pytest.raises(LeaderboardConflictError):
        archive_leaderboard(_payloads(), tmp_path)


def test_new_base_dt_rename_race_reevaluates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Losing the staging rename race re-evaluates against the now-existing directory."""
    import os as _os

    real_rename = _os.rename

    def _race(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        target = tmp_path / "20260923"
        if str(dst) == str(target) and not target.exists():
            target.mkdir(parents=True)
            (target / "etfRankTotal.json").write_text("not json", encoding="utf-8")
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr("src.contest.leaderboard.os.rename", _race)
    with pytest.raises(LeaderboardConflictError):
        archive_leaderboard(_payloads(), tmp_path)


def test_naive_fetched_at_names_revision(tmp_path: Path) -> None:
    """A naive fetched_at is treated as UTC when naming the revision directory."""
    archive_leaderboard(_payloads(), tmp_path)
    other = _payloads(rows=[_row(1, "leader", 9.99, 2.0), _row(22, "lkthl", 3.0, 0.1)])
    result = archive_leaderboard(other, tmp_path, fetched_at=datetime(2026, 9, 24, 7, 41, 15))
    assert result == (date(2026, 9, 23), ArchiveOutcome.REVISION_STORED)
    assert (tmp_path / "20260923" / "revisions" / "20260924T074115Z").is_dir()


def test_revision_rename_race_conflicts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A revision rename losing a race to an existing name fails closed."""
    import os as _os

    archive_leaderboard(_payloads(), tmp_path)
    other = _payloads(rows=[_row(1, "leader", 9.99, 2.0), _row(22, "lkthl", 3.0, 0.1)])
    real_rename = _os.rename

    def _race(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        revisions_root = tmp_path / "20260923" / "revisions"
        if str(dst).startswith(str(revisions_root)) and ".staging-" in str(src):
            revisions_root.mkdir(parents=True, exist_ok=True)
            target = _os.path.basename(str(dst))
            (revisions_root / target).mkdir(parents=True, exist_ok=True)
            (revisions_root / target / "etfRankTotal.json").write_text("{}\n", encoding="utf-8")
        return real_rename(src, dst, *args, **kwargs)

    monkeypatch.setattr("src.contest.leaderboard.os.rename", _race)
    with pytest.raises(LeaderboardConflictError):
        archive_leaderboard(other, tmp_path, fetched_at=datetime(2026, 9, 24, 7, 41, 15, tzinfo=UTC))


def test_new_base_dt_rename_unexpected_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A rename failure without an existing target surfaces the original error."""
    def _boom(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        raise OSError("rename down")

    monkeypatch.setattr("src.contest.leaderboard.os.rename", _boom)
    with pytest.raises(OSError, match="rename down"):
        archive_leaderboard(_payloads(), tmp_path)
    assert not (tmp_path / "20260923").exists()


def test_revision_rename_unexpected_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A revision rename failure without an existing target surfaces the original error."""
    archive_leaderboard(_payloads(), tmp_path)
    other = _payloads(rows=[_row(1, "leader", 9.99, 2.0), _row(22, "lkthl", 3.0, 0.1)])

    def _boom(src: Any, dst: Any, *args: Any, **kwargs: Any) -> None:
        target = tmp_path / "20260923"
        if str(dst).startswith(str(target / "revisions")):
            raise OSError("revision rename down")
        import os as _os

        return _os.rename(src, dst, *args, **kwargs)

    monkeypatch.setattr("src.contest.leaderboard.os.rename", _boom)
    with pytest.raises(OSError, match="revision rename down"):
        archive_leaderboard(other, tmp_path, fetched_at=datetime(2026, 9, 24, 7, 41, 15, tzinfo=UTC))


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


def test_archive_dir_missing_total_file(tmp_path: Path) -> None:
    """A pre-existing baseDt dir without etfRankTotal is a conflict, never adopted."""
    day_dir = tmp_path / "20260923"
    day_dir.mkdir(parents=True)
    (day_dir / "etfRankGroupTop.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(LeaderboardConflictError):
        archive_leaderboard(_payloads(), tmp_path)
    assert not (day_dir / "revisions").exists()


def test_archive_dir_corrupt_total_file(tmp_path: Path) -> None:
    """A baseDt dir with non-JSON etfRankTotal is a structural conflict with nothing created."""
    day_dir = tmp_path / "20260923"
    day_dir.mkdir(parents=True)
    (day_dir / "etfRankTotal.json").write_text("not json", encoding="utf-8")
    with pytest.raises(LeaderboardConflictError):
        archive_leaderboard(_payloads(), tmp_path)
    assert not (day_dir / "revisions").exists()


def test_archive_dir_non_object_total_file(tmp_path: Path) -> None:
    """A baseDt dir with non-object etfRankTotal JSON is a structural conflict."""
    day_dir = tmp_path / "20260923"
    day_dir.mkdir(parents=True)
    (day_dir / "etfRankTotal.json").write_text("[1, 2]\n", encoding="utf-8")
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


def test_wrappers_of_one_exposure_not_ambiguous(tmp_path: Path) -> None:
    """Two wrapper keys of one exposure matching the same daily count once for that exposure."""
    rows = [_row(1, "holder", 7.0, 2.714)]
    archive_leaderboard(_payloads(rows=rows), tmp_path)
    snapshot = load_snapshot(tmp_path, date(2026, 9, 23))
    changes = {"HY2": 2.727, "HY2@0195S0": 2.696}
    exposure_of = {"HY2": "HY2", "HY2@0195S0": "HY2"}
    out = infer_single_vehicle_holders(snapshot, changes, tol_pct=0.02, min_weight=0.90, exposure_of=exposure_of)
    assert out["holder"][0] == "HY2"
    assert 0.9 <= out["holder"][1] <= 1.0


def test_different_exposures_stay_ambiguous(tmp_path: Path) -> None:
    """One fitting exposure identifies; two fitting exposures omit the entry."""
    archive_leaderboard(_payloads(rows=[_row(1, "holder", 7.0, 2.714)]), tmp_path)
    snapshot = load_snapshot(tmp_path, date(2026, 9, 23))
    only_hy2 = infer_single_vehicle_holders(
        snapshot, {"HY2": 2.727, "Q2": 3.2}, tol_pct=0.02, min_weight=0.90
    )
    assert only_hy2["holder"][0] == "HY2"
    both = infer_single_vehicle_holders(
        snapshot, {"HY2": 2.727, "Q2": 3.026}, tol_pct=0.02, min_weight=0.90
    )
    assert "holder" not in both


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


def test_latest_entry_on_or_before_skips_snapshots_without_us(tmp_path: Path) -> None:
    """The newest visible entry at or before the session wins; later or missing-us snapshots are skipped."""
    archive_leaderboard(_payloads("20260923"), tmp_path)
    archive_leaderboard(_payloads("20260928", rows=[_row(1, "leader", 20.0, 3.0)]), tmp_path)
    archive_leaderboard(_payloads("20261001", rows=[_row(5, "lkthl", 9.0, 1.0)]), tmp_path)
    hit = latest_entry_on_or_before(tmp_path, "lkthl", date(2026, 9, 30))
    assert hit is not None
    assert hit[0] == date(2026, 9, 23)
    assert hit[1].total_return_pct == pytest.approx(3.71417)
    assert latest_entry_on_or_before(tmp_path, "nobody", date(2026, 10, 1)) is None
    assert latest_entry_on_or_before(tmp_path / "absent", "lkthl", date(2026, 10, 1)) is None
