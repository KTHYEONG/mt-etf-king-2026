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
    assert p == tmp_path / "raw/krx/etp/etf_bydd_trd/2026/20260827.json"

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
