import fcntl
from collections.abc import Callable, Mapping
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from src.arena import panel
from src.arena.panel import (
    SOURCE_COLUMNS,
    Panel,
    PanelConfig,
    PanelSourceError,
    build_panel,
    load_etf_silver,
    load_panel_config,
)
from src.core.config import ConfigError
from src.core.paths import DataPaths

CFG = PanelConfig(lock_timeout_s=900.0, nav_divergence_max=0.10)
DAYS = [date(2026, 1, 1) + timedelta(days=i) for i in range(8)]
D1, D2, D3, D4, D5 = DAYS[:5]
SCHEMA: dict[str, pl.DataType] = {
    "date": pl.Date,
    "ticker": pl.String,
    "name": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Int64,
    "trading_value": pl.Int64,
    "nav": pl.Float64,
    "shares_outstanding": pl.Int64,
    "underlying_index_name": pl.String,
}
OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "name",
    "listed",
    "traded",
    "open",
    "high",
    "low",
    "close",
    "close_ref",
    "nav",
    "volume",
    "trading_value",
    "r_cc",
    "r_on",
    "r_id",
    "r_ref",
    "r_nav",
    "nav_divergence",
]


def _row(day: date, ticker: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "date": day,
        "ticker": ticker,
        "name": f"{ticker} fund",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 1000,
        "trading_value": 100000,
        "nav": 100.0,
        "shares_outstanding": 1000,
        "underlying_index_name": "IDX",
    }
    row.update(overrides)
    return row


def _frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=SCHEMA)


def _build(
    rows: list[dict[str, Any]],
    tickers: tuple[str, ...] = ("A",),
    start: date = D1,
    end: date = D4,
) -> Panel:
    return build_panel(_frame(rows), tickers, start, end, CFG)


def _get(p: Panel, ticker: str, day: date) -> dict[str, Any]:
    matched = p.frame.filter((pl.col("ticker") == ticker) & (pl.col("date") == day))
    assert matched.height == 1
    return matched.row(0, named=True)


def _load_with(monkeypatch: pytest.MonkeyPatch, section: Mapping[str, Any]) -> PanelConfig:
    data: Mapping[str, Any] = {"panel": dict(section)}
    monkeypatch.setattr(panel, "load_config", lambda name: data)
    return load_panel_config()


def test_panel_config_loads() -> None:
    assert load_panel_config() == PanelConfig(900.0, 0.10)


@pytest.mark.parametrize(
    "section",
    [
        {"lock_timeout_s": 900.0, "nav_divergence_max": 0.10, "extra": 1},
        {"lock_timeout_s": 900.0},
        {"lock_timeout_s": 900.0, "nav_divergence_max": True},
        {"lock_timeout_s": "900", "nav_divergence_max": 0.10},
        {"lock_timeout_s": 0, "nav_divergence_max": 0.10},
        {"lock_timeout_s": 900.0, "nav_divergence_max": -0.1},
    ],
)
def test_panel_config_rejects_bad_sections(monkeypatch: pytest.MonkeyPatch, section: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        _load_with(monkeypatch, section)


def test_panel_config_accepts_int_and_rejects_non_mapping(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _load_with(monkeypatch, {"lock_timeout_s": 900, "nav_divergence_max": 1}) == PanelConfig(900.0, 1.0)
    monkeypatch.setattr(panel, "load_config", lambda name: {"panel": []})
    with pytest.raises(ConfigError):
        load_panel_config()
    monkeypatch.setattr(panel, "load_config", lambda name: {"other": {}})
    with pytest.raises(ConfigError):
        load_panel_config()


def test_load_reads_columns_under_shared_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = DataPaths(root=tmp_path)
    source = paths.silver("etf_daily")
    source.parent.mkdir(parents=True)
    _frame([_row(D1, "A")]).with_columns(extra=pl.lit(1)).write_parquet(source)
    lock_path = paths.lock("pipeline")
    original = pl.read_parquet
    observed: list[bool] = []

    def probe(*args: Any, **kwargs: Any) -> pl.DataFrame:
        with lock_path.open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                observed.append(True)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)
                observed.append(False)
        with lock_path.open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            observed.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", probe)
    result = load_etf_silver(paths, CFG)
    assert observed == [True, True]
    assert result.columns == list(SOURCE_COLUMNS)


def test_load_missing_file_or_column(tmp_path: Path) -> None:
    paths = DataPaths(root=tmp_path)
    with pytest.raises(PanelSourceError):
        load_etf_silver(paths, CFG)
    source = paths.silver("etf_daily")
    source.parent.mkdir(parents=True)
    _frame([_row(D1, "A")]).drop("nav").write_parquet(source)
    with pytest.raises(PanelSourceError, match="nav"):
        load_etf_silver(paths, CFG)


def test_calendar_from_all_tickers() -> None:
    rows = [_row(d, "A") for d in DAYS[:4]] + [_row(d, "B") for d in (D3, D4)]
    p = _build(rows, tickers=("B",))
    assert p.sessions == (D1, D2, D3, D4)
    assert p.frame.filter(pl.col("ticker") == "B")["date"].to_list() == [D3, D4]


def test_explicit_gap_row() -> None:
    rows = [_row(D1, "A"), _row(D2, "A"), _row(D4, "A"), _row(D3, "C")]
    p = _build(rows)
    gap = _get(p, "A", D3)
    assert gap["listed"] is False
    assert gap["traded"] is False
    assert gap["open"] is None and gap["close"] is None and gap["close_ref"] is None
    assert gap["name"] is None
    after = _get(p, "A", D4)
    for key in ("r_cc", "r_on", "r_ref"):
        assert gap[key] is None
        assert after[key] is None


def test_zero_volume_is_not_a_trade() -> None:
    rows = [
        _row(D1, "A"),
        _row(D2, "A", volume=0, open=0.0, high=0.0, low=0.0, close=101.0),
        _row(D3, "A", close=102.0),
    ]
    p = _build(rows)
    d2 = _get(p, "A", D2)
    assert d2["traded"] is False
    assert d2["open"] is None and d2["high"] is None and d2["low"] is None and d2["close"] is None
    assert d2["close_ref"] == pytest.approx(101.0)
    assert d2["r_cc"] is None
    assert d2["r_id"] is None
    assert d2["r_ref"] == pytest.approx(101.0 / 100.0 - 1.0)
    assert _get(p, "A", D3)["r_cc"] is None


@pytest.mark.parametrize("override", [{"open": 0.0}, {"high": 0.0}, {"low": 0.0}])
def test_open_zero_with_volume_is_not_a_trade(override: dict[str, float]) -> None:
    p = _build([_row(D1, "A"), _row(D2, "A", volume=4, **override)])
    assert _get(p, "A", D2)["traded"] is False


def test_return_definitions() -> None:
    rows = [
        _row(D1, "A", close=100.0, nav=100.0),
        _row(D2, "A", open=102.0, high=102.0, low=98.0, close=99.0, nav=98.0),
    ]
    d2 = _get(_build(rows), "A", D2)
    assert d2["r_cc"] == pytest.approx(-0.01)
    assert d2["r_on"] == pytest.approx(0.02)
    assert d2["r_id"] == pytest.approx(99.0 / 102.0 - 1.0)
    assert d2["r_nav"] == pytest.approx(-0.02)


def test_first_session_uses_prior_close() -> None:
    rows = [_row(D1, "A", close=100.0), _row(D2, "A", close=110.0), _row(D3, "A", close=121.0)]
    p = _build(rows, start=D2, end=D3)
    assert p.frame["date"].to_list() == [D2, D3]
    assert _get(p, "A", D2)["r_cc"] == pytest.approx(0.10)


def test_nav_divergence_flag() -> None:
    rows = [
        _row(D1, "A", close=100.0, nav=100.0),
        _row(D2, "A", close=160.0, nav=96.0),
        _row(D3, "A", close=220.8, nav=132.48),
        _row(D4, "A", close=240.0, nav=None),
    ]
    p = _build(rows)
    assert _get(p, "A", D2)["nav_divergence"] is True
    assert _get(p, "A", D3)["nav_divergence"] is False
    assert _get(p, "A", D4)["r_nav"] is None
    assert _get(p, "A", D4)["nav_divergence"] is False


def test_future_rows_ignored() -> None:
    base = [_row(d, "A") for d in DAYS[:3]]
    future = [
        *base,
        _row(D4, "A", close=5.0),
        _row(D4, "A", close=6.0),
        _row(D5, "Z"),
    ]
    clean = _build(base, tickers=("A", "Z"), start=D1, end=D3)
    dirty = _build(future, tickers=("A", "Z"), start=D1, end=D3)
    assert dirty.frame.equals(clean.frame)
    assert dirty.sessions == clean.sessions
    assert dirty.missing_tickers == ("Z",)


def test_delisted_ticker_kept_until_last_row() -> None:
    rows = [_row(d, "A") for d in DAYS[:4]] + [_row(D1, "X"), _row(D2, "X")]
    p = _build(rows, tickers=("A", "X"))
    assert p.frame.filter(pl.col("ticker") == "X")["date"].to_list() == [D1, D2]
    assert p.missing_tickers == ()


def test_missing_tickers_reported() -> None:
    p = _build([_row(d, "A") for d in DAYS[:3]], tickers=("A", "Q"), end=D3)
    assert p.missing_tickers == ("Q",)
    assert p.frame.filter(pl.col("ticker") == "Q").height == 0


@pytest.mark.parametrize(
    ("build", "match"),
    [
        (lambda: build_panel(_frame([_row(D1, "A")]), ("A",), D2, D1, CFG), "after end"),
        (lambda: build_panel(_frame([_row(D1, "A")]), (), D1, D1, CFG), "non-empty"),
        (lambda: build_panel(_frame([_row(D1, "A")]).drop("volume"), ("A",), D1, D1, CFG), "lack column volume"),
        (
            lambda: build_panel(
                _frame([_row(D1, "A")]).with_columns(pl.col("close").cast(pl.Int64)), ("A",), D1, D1, CFG
            ),
            "column close",
        ),
        (lambda: build_panel(_frame([_row(D1, "A"), _row(D1, "A")]), ("A",), D1, D1, CFG), "duplicate"),
        (lambda: build_panel(_frame([_row(D1, "A", close=0.0)]), ("A",), D1, D1, CFG), "close at"),
        (lambda: build_panel(_frame([_row(D1, "A", close=None)]), ("A",), D1, D1, CFG), "close at"),
    ],
)
def test_bad_inputs_rejected(build: Callable[[], Panel], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        build()


def test_deterministic_and_column_order() -> None:
    rows = [_row(D1, "A"), _row(D3, "A", volume=0, open=0.0, high=0.0, low=0.0, close=99.0), _row(D2, "B")]
    first = _build(rows, tickers=("A", "B"), end=D3)
    second = _build(list(reversed(rows)), tickers=("B", "A"), end=D3)
    assert first.frame.equals(second.frame)
    assert first.frame.columns == OUTPUT_COLUMNS
    assert first.frame.select("ticker", "date").rows() == sorted(first.frame.select("ticker", "date").rows())
