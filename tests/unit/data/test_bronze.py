from __future__ import annotations

from datetime import date, datetime, UTC
from pathlib import Path

from src.core.paths import DataPaths
from src.data.bronze import BronzeRecord, BronzeStore


def test_scenario_02_06_bronze_write_once_and_revision(tmp_path: Path) -> None:
    """SCENARIO-02-06."""
    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)

    rec = BronzeRecord(
        endpoint="etp/etf_bydd_trd",
        bas_dd=date(2026, 8, 27),
        fetched_at=datetime(2026, 8, 28, 0, 57, 12, tzinfo=UTC),
        http_status=200,
        row_count=1,
        rows=[{"BAS_DD": "20260827", "NAV": ""}],
    )
    res = store.write(rec)
    assert res == "written"
    p = paths.bronze("etp/etf_bydd_trd", date(2026, 8, 27))
    assert p.exists()
    assert p == tmp_path / "raw/krx/etp/etf_bydd_trd/2026/20260827.json.gz"

    # second write different record same date -> skipped
    rec2 = BronzeRecord(
        endpoint="etp/etf_bydd_trd",
        bas_dd=date(2026, 8, 27),
        fetched_at=datetime(2026, 8, 28, 1, 0, 0, tzinfo=UTC),
        http_status=200,
        row_count=0,
        rows=[],
    )
    b1 = p.read_bytes()
    res2 = store.write(rec2)
    assert res2 == "skipped"
    assert p.read_bytes() == b1

    # allow_revision
    res3 = store.write(rec2, allow_revision=True)
    assert res3 == "revised"
    assert p.read_bytes() == b1
    siblings = list(p.parent.glob("*"))
    assert len(siblings) == 2
    # revision is sibling  # noqa: RUF015
    rev = next(x for x in siblings if x != p)  # noqa: RUF015
    assert ".rev." in rev.name

    # read round-trips
    r = store.read("etp/etf_bydd_trd", date(2026, 8, 27))
    assert r.endpoint == "etp/etf_bydd_trd"
    assert r.bas_dd == date(2026, 8, 27)
    assert r.row_count == 1
    assert r.rows[0]["NAV"] == ""

    assert store.has_session("etp/etf_bydd_trd", date(2026, 8, 27)) is True
    assert store.has_session("etp/etf_bydd_trd", date(2026, 8, 26)) is False

    assert date(2026, 8, 27) in store.available_sessions("etp/etf_bydd_trd")


def test_SCENARIO_DSR_01_gzip_write_roundtrip(tmp_path: Path) -> None:  # noqa: N802
    """SCENARIO_DSR_01: write creates .json.gz; read returns identical payload."""
    import gzip
    import json

    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)
    rec = BronzeRecord(
        endpoint="etp/etf_bydd_trd",
        bas_dd=date(2026, 8, 28),
        fetched_at=datetime(2026, 8, 28, 0, 57, 12, tzinfo=UTC),
        http_status=200,
        row_count=2,
        rows=[{"BAS_DD": "20260828", "NAV": "100"}, {"BAS_DD": "20260828", "NAV": "200"}],
    )
    res = store.write(rec)
    assert res == "written"
    p = paths.bronze("etp/etf_bydd_trd", date(2026, 8, 28))
    assert p.name.endswith(".json.gz")
    assert p.suffixes == [".json", ".gz"] or p.name.endswith(".json.gz")
    assert p.exists()
    # plain absent
    plain = p.with_suffix("")
    assert not plain.exists() or plain == p  # if with_suffix removes .gz only, plain is .json; ensure plain not exist as legacy standalone? For gz case, plain path is .json
    # ensure no plain .json file separate from gz (legacy) exists when we just wrote gz
    legacy_plain = Path(str(p)[:-3])  # strip .gz
    # after gz write, legacy plain should not exist as separate new plain write
    # we check that the legacy .json file does not exist independently (if plain == legacy, it's same check)
    # For our implementation, gz write does not create plain, so legacy plain may not exist unless we created before
    # So assert that reading via gzip round-trip matches
    r = store.read("etp/etf_bydd_trd", date(2026, 8, 28))
    assert r.row_count == 2
    assert r.rows == rec.rows
    # verify gzip content is compact (no indent) by reading raw
    with gzip.open(p, "rt", encoding="utf-8") as f:
        raw = f.read()
        data = json.loads(raw)
        assert data["rows"] == rec.rows
        assert "\n  " not in raw or raw.count("\n") <= 1  # compact


def test_SCENARIO_DSR_02_migrate_plain(tmp_path: Path) -> None:  # noqa: N802
    """SCENARIO_DSR_02: legacy plain readable; migrate produces gz and deletes plain."""
    import gzip
    import json

    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)
    # create legacy plain .json manually
    plain_path = tmp_path / "raw/krx/etp/etf_bydd_trd/2026/20260829.json"
    plain_path.parent.mkdir(parents=True, exist_ok=True)
    rec_plain = {
        "endpoint": "etp/etf_bydd_trd",
        "bas_dd": "20260829",
        "fetched_at": datetime(2026, 8, 29, 0, 0, 0, tzinfo=UTC).isoformat(),
        "http_status": 200,
        "row_count": 1,
        "rows": [{"BAS_DD": "20260829", "NAV": "999"}],
    }
    plain_path.write_text(json.dumps(rec_plain, ensure_ascii=False, indent=2), encoding="utf-8")
    # has_session should be true via legacy fallback
    assert store.has_session("etp/etf_bydd_trd", date(2026, 8, 29)) is True
    r = store.read("etp/etf_bydd_trd", date(2026, 8, 29))
    assert r.rows[0]["NAV"] == "999"
    assert r.row_count == 1
    # migrate
    res = store.migrate_plain_to_gzip("etp/etf_bydd_trd", delete_plain=True)
    assert res["failed"] == 0
    assert res["deleted_plain"] == 1
    assert res["migrated"] == 1
    gz_path = paths.bronze("etp/etf_bydd_trd", date(2026, 8, 29))
    assert gz_path.exists()
    assert not plain_path.exists()
    # has_session still true and available_sessions deduped
    assert store.has_session("etp/etf_bydd_trd", date(2026, 8, 29)) is True
    sessions = store.available_sessions("etp/etf_bydd_trd")
    assert sessions.count(date(2026, 8, 29)) == 1
    assert date(2026, 8, 29) in sessions
    # verify gz content round-trip
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        gz_data = json.load(f)
    assert gz_data["rows"] == rec_plain["rows"]
