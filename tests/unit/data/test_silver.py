from __future__ import annotations

import tempfile
from datetime import date, datetime, UTC
from pathlib import Path

import polars as pl

import pytest

from src.core.calendar import TradingCalendar
from src.core.paths import DataPaths
from src.data.bronze import BronzeRecord, BronzeStore
from src.data.schema import ETF_DAILY_SCHEMA
from src.data.silver import SilverBuilder
from src.data.validation import PanelValidator, Severity, ValidationIssue, ValidationReport


def _make_row(ticker: str, price: float, date_str: str) -> dict[str, str]:
    return {
        "BAS_DD": date_str,
        "ISU_CD": ticker,
        "ISU_NM": "Test",
        "TDD_CLSPRC": str(int(price)) if price == int(price) else str(price),
        "NAV": str(int(price)) if price == int(price) else str(price),
        "TDD_OPNPRC": str(int(price)),
        "TDD_HGPRC": str(int(price + 1)),
        "TDD_LWPRC": str(int(price - 1)),
        "ACC_TRDVOL": "1000",
        "ACC_TRDVAL": "1000000",
        "MKTCAP": str(int(price * 1000)),
        "INVSTASST_NETASST_TOTAMT": str(int(price * 1000)),
        "LIST_SHRS": "1000",
        "IDX_IND_NM": "Index",
        "OBJ_STKPRC_IDX": "1000",
    }


def test_scenario_03_07_build_excludes_holiday() -> None:
    """SCENARIO-03-07"""
    tmp = Path(tempfile.mkdtemp(dir="tmp"))
    paths = DataPaths(root=tmp)
    store = BronzeStore(paths)
    cal = TradingCalendar()
    validator = PanelValidator(cal)
    # 14 populated, 15 holiday, 18 populated
    rows14 = [_make_row("451060", 35000, "20260814"), _make_row("069500", 30000, "20260814")]
    rows18 = [_make_row("451060", 35100, "20260818"), _make_row("069500", 30100, "20260818")]
    holidays = []
    for ticker in ["451060", "069500"]:
        r = _make_row(ticker, 0, "20260815")
        r["TDD_CLSPRC"] = ""
        r["TDD_OPNPRC"] = ""
        r["TDD_HGPRC"] = ""
        r["TDD_LWPRC"] = ""
        r["ACC_TRDVOL"] = ""
        r["ACC_TRDVAL"] = ""
        r["MKTCAP"] = ""
        r["NAV"] = ""
        r["OBJ_STKPRC_IDX"] = ""
        holidays.append(r)
    for d, rows in [(date(2026, 8, 14), rows14), (date(2026, 8, 15), holidays), (date(2026, 8, 18), rows18)]:
        rec = BronzeRecord(endpoint="etp/etf_bydd_trd", bas_dd=d, fetched_at=datetime.now(UTC), http_status=200, row_count=len(rows), rows=rows)
        store.write(rec)
    builder = SilverBuilder(store, paths, validator)
    result = builder.build("etf_daily", mode="full")
    assert result.path == paths.silver("etf_daily")
    assert result.path.exists()
    df = pl.read_parquet(result.path)
    distinct = sorted(df.select(pl.col("date").unique()).to_series().to_list())
    assert distinct == [date(2026, 8, 14), date(2026, 8, 18)]
    expected_cols = {c.column for c in ETF_DAILY_SCHEMA.columns} | {"is_tradable"}
    assert expected_cols.issubset(set(df.columns))


def test_scenario_03_08_incremental_idempotence_and_fatal() -> None:
    """SCENARIO-03-08"""
    tmp = Path(tempfile.mkdtemp(dir="tmp"))
    paths = DataPaths(root=tmp)
    store = BronzeStore(paths)
    cal = TradingCalendar()
    validator = PanelValidator(cal)
    # initial 14 and 18
    for d in [date(2026, 8, 14), date(2026, 8, 18)]:
        rows = [_make_row("451060", 35000, d.strftime("%Y%m%d")), _make_row("069500", 30000, d.strftime("%Y%m%d"))]
        rec = BronzeRecord(endpoint="etp/etf_bydd_trd", bas_dd=d, fetched_at=datetime.now(UTC), http_status=200, row_count=2, rows=rows)
        store.write(rec)
    builder = SilverBuilder(store, paths, validator)
    builder.build("etf_daily", mode="full")
    # add later session
    d = date(2026, 8, 19)
    rows = [_make_row("451060", 35200, d.strftime("%Y%m%d")), _make_row("069500", 30200, d.strftime("%Y%m%d"))]
    rec = BronzeRecord(endpoint="etp/etf_bydd_trd", bas_dd=d, fetched_at=datetime.now(UTC), http_status=200, row_count=2, rows=rows)
    store.write(rec)
    result_inc = builder.build("etf_daily", mode="incremental")
    df_inc = pl.read_parquet(paths.silver("etf_daily"))
    # fresh full rebuild to compare
    tmp2 = Path(tempfile.mkdtemp(dir="tmp"))  # noqa: F841
    # Simpler: capture inc, then rebuild full and compare
    result_full = builder.build("etf_daily", mode="full")
    df_full = pl.read_parquet(paths.silver("etf_daily"))
    assert df_inc.sort(["date", "ticker"]).equals(df_full.sort(["date", "ticker"]))

    # fatal case
    tmp3 = Path(tempfile.mkdtemp(dir="tmp"))
    paths3 = DataPaths(root=tmp3)
    store3 = BronzeStore(paths3)
    for d in [date(2026, 8, 14)]:
        rows = [_make_row("451060", 35000, d.strftime("%Y%m%d"))]
        rec = BronzeRecord(endpoint="etp/etf_bydd_trd", bas_dd=d, fetched_at=datetime.now(UTC), http_status=200, row_count=1, rows=rows)
        store3.write(rec)

    class FatalValidator:
        def validate(self, dataset: str, frame: pl.DataFrame) -> ValidationReport:
            return ValidationReport(dataset=dataset, rows=frame.height, sessions=0, issues=(ValidationIssue(gate="V2", severity=Severity.CRITICAL, count=1, detail="fatal"),))

    builder_fatal = SilverBuilder(store3, paths3, FatalValidator())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        builder_fatal.build("etf_daily", mode="full")
    assert not paths3.silver("etf_daily").exists()


def test_SCENARIO_03A_03_future_dates_blocks_silver_write() -> None:  # noqa: N802
    """SCENARIO-03A-03"""
    tmp = Path(tempfile.mkdtemp(dir="tmp"))
    paths = DataPaths(root=tmp)
    store = BronzeStore(paths)
    cal = TradingCalendar()
    validator = PanelValidator(cal, today=lambda: date(2026, 8, 28))
    d = date(2026, 8, 29)
    rows = [_make_row("451060", 35000, d.strftime("%Y%m%d")), _make_row("069500", 30000, d.strftime("%Y%m%d"))]
    rec = BronzeRecord(endpoint="etp/etf_bydd_trd", bas_dd=d, fetched_at=datetime.now(UTC), http_status=200, row_count=2, rows=rows)
    store.write(rec)
    builder = SilverBuilder(store, paths, validator)
    with pytest.raises(RuntimeError):
        builder.build("etf_daily", mode="full")
    assert not paths.silver("etf_daily").exists()


globals()["test SCENARIO-03A-03"] = test_SCENARIO_03A_03_future_dates_blocks_silver_write  # noqa: E402, F401


def _write_trading_day(store: BronzeStore, day: date) -> None:
    rows = [_make_row("451060", 35000, day.strftime("%Y%m%d")), _make_row("069500", 30000, day.strftime("%Y%m%d"))]
    rec = BronzeRecord(
        endpoint="etp/etf_bydd_trd",
        bas_dd=day,
        fetched_at=datetime.now(UTC),
        http_status=200,
        row_count=len(rows),
        rows=rows,
    )
    store.write(rec)


def test_build_heals_late_bronze_session_incrementally(tmp_path: Path) -> None:
    """A bronze session older than the silver max date heals without a full rebuild."""
    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)
    builder = SilverBuilder(store, paths, PanelValidator(TradingCalendar()))
    _write_trading_day(store, date(2026, 8, 27))
    _write_trading_day(store, date(2026, 8, 31))
    first = builder.build("etf_daily", mode="full")
    assert first.written is True
    _write_trading_day(store, date(2026, 8, 28))
    healed = builder.build("etf_daily", mode="incremental")
    assert healed.written is True
    assert healed.report is not None
    frame = pl.read_parquet(paths.silver("etf_daily"))
    assert frame.select(pl.col("date").unique().sort()).to_series().to_list() == [
        date(2026, 8, 27),
        date(2026, 8, 28),
        date(2026, 8, 31),
    ]
    assert frame.sort(["date", "ticker"]).equals(frame)


def test_build_incremental_noop_leaves_file_untouched(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """No new bronze session means no recompute, no rewrite, and a None report."""
    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)
    builder = SilverBuilder(store, paths, PanelValidator(TradingCalendar()))
    _write_trading_day(store, date(2026, 8, 27))
    _write_trading_day(store, date(2026, 8, 28))
    built = builder.build("etf_daily", mode="full")
    silver_path = paths.silver("etf_daily")
    before_bytes = silver_path.read_bytes()
    before_mtime = silver_path.stat().st_mtime_ns
    with caplog.at_level("INFO", logger="src.data.silver"):
        again = builder.build("etf_daily", mode="incremental")
    assert again.written is False
    assert again.report is None
    assert silver_path.read_bytes() == before_bytes
    assert silver_path.stat().st_mtime_ns == before_mtime
    assert (again.rows, again.sessions) == (built.rows, built.sessions)
    assert any("status=unchanged" in record.message for record in caplog.records)


def test_build_non_trading_bronze_stays_excluded(tmp_path: Path) -> None:
    """Non-trading bronze never enters silver; repeated builds stay no-ops."""
    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)
    builder = SilverBuilder(store, paths, PanelValidator(TradingCalendar()))
    _write_trading_day(store, date(2026, 8, 27))
    assert builder.build("etf_daily", mode="full").written is True
    closed = [_make_row("451060", 0, "20260901"), _make_row("069500", 0, "20260901")]
    for row in closed:
        row["TDD_CLSPRC"] = ""
    rec = BronzeRecord(
        endpoint="etp/etf_bydd_trd",
        bas_dd=date(2026, 9, 1),
        fetched_at=datetime.now(UTC),
        http_status=200,
        row_count=len(closed),
        rows=closed,
    )
    store.write(rec)
    for _ in range(2):
        result = builder.build("etf_daily", mode="incremental")
        assert result.written is False
        assert result.report is None
    dates = pl.read_parquet(paths.silver("etf_daily")).select(pl.col("date").unique()).to_series().to_list()
    assert date(2026, 9, 1) not in dates


def test_build_fatal_keeps_prior_file(tmp_path: Path) -> None:
    """A CRITICAL validation on new rows raises with the existing file untouched."""
    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)
    builder = SilverBuilder(store, paths, PanelValidator(TradingCalendar()))
    _write_trading_day(store, date(2026, 8, 27))
    builder.build("etf_daily", mode="full")
    silver_path = paths.silver("etf_daily")
    before = silver_path.read_bytes()

    class FatalValidator:
        def validate(self, dataset: str, frame: pl.DataFrame) -> ValidationReport:
            return ValidationReport(dataset=dataset, rows=frame.height, sessions=0, issues=(ValidationIssue(gate="V2", severity=Severity.CRITICAL, count=1, detail="fatal"),))

    _write_trading_day(store, date(2026, 8, 28))
    builder_fatal = SilverBuilder(store, paths, FatalValidator())  # type: ignore[arg-type]
    with pytest.raises(RuntimeError):
        builder_fatal.build("etf_daily", mode="incremental")
    assert silver_path.read_bytes() == before


def test_build_incremental_recovers_from_unreadable_silver(tmp_path: Path) -> None:
    """A corrupt existing file falls back to processing every bronze session."""
    paths = DataPaths(root=tmp_path)
    store = BronzeStore(paths)
    builder = SilverBuilder(store, paths, PanelValidator(TradingCalendar()))
    _write_trading_day(store, date(2026, 8, 27))
    builder.build("etf_daily", mode="full")
    paths.silver("etf_daily").write_bytes(b"not a parquet file")
    _write_trading_day(store, date(2026, 8, 28))
    result = builder.build("etf_daily", mode="incremental")
    assert result.written is True
    dates = pl.read_parquet(paths.silver("etf_daily")).select(pl.col("date").unique()).to_series().to_list()
    assert sorted(dates) == [date(2026, 8, 27), date(2026, 8, 28)]


def test_silver_session_dates(tmp_path: Path) -> None:
    """The date helper returns the distinct set, or empty when no file exists."""
    from src.data.silver import silver_session_dates

    paths = DataPaths(root=tmp_path)
    assert silver_session_dates(paths, "etf_daily") == frozenset()
    store = BronzeStore(paths)
    builder = SilverBuilder(store, paths, PanelValidator(TradingCalendar()))
    _write_trading_day(store, date(2026, 8, 27))
    _write_trading_day(store, date(2026, 8, 28))
    builder.build("etf_daily", mode="full")
    assert silver_session_dates(paths, "etf_daily") == frozenset({date(2026, 8, 27), date(2026, 8, 28)})
