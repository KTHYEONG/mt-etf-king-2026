from dataclasses import replace
from datetime import date
from typing import Any

import polars as pl
import pytest

from src.arena import universe as un
from src.arena.contest import EditionSpec, load_edition
from src.arena.universe import (
    EmptyHistoryError,
    IssuerTable,
    UniverseSnapshot,
    UnsupportedTrackError,
    eligible_universe,
    load_issuer_table,
    resolve_issuer,
)
from src.core.config import ConfigError

D1 = date(2026, 10, 1)
D2 = date(2026, 10, 2)
D3 = date(2026, 10, 5)


def _table() -> IssuerTable:
    return load_issuer_table()


def _spec() -> EditionSpec:
    return load_edition(2026)


def _rows(*records: tuple[date, str, str | None]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [r[0] for r in records],
            "ticker": [r[1] for r in records],
            "name": [r[2] for r in records],
        },
        schema={"date": pl.Date, "ticker": pl.Utf8, "name": pl.Utf8},
    )


def _snapshot(rows: pl.DataFrame, as_of: date, spec: EditionSpec | None = None) -> UniverseSnapshot:
    return eligible_universe(rows, spec or _spec(), "autonomous", _table(), as_of)


def _set_issuers(monkeypatch: pytest.MonkeyPatch, data: dict[str, Any]) -> None:
    monkeypatch.setattr(un, "load_config", lambda name: data)


def test_issuer_table_rejects_duplicates_and_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    bad_data: list[dict[str, Any]] = [
        {"issuers": {"A": ["KODEX"], "B": ["kodex"]}, "other_brands": []},
        {"issuers": {"A": ["KODEX"]}, "other_brands": ["kodex"]},
        {"issuers": {"A": ["KODEX"]}, "other_brands": ["WON", "won"]},
        {"issuers": {"A": ["KO DEX"]}, "other_brands": []},
        {"issuers": {"A": [""]}, "other_brands": []},
        {"issuers": {"A": [5]}, "other_brands": []},
        {"issuers": {"A": []}, "other_brands": []},
        {"issuers": {"A": "KODEX"}, "other_brands": []},
        {"issuers": {"": ["KODEX"]}, "other_brands": []},
        {"issuers": {5: ["KODEX"]}, "other_brands": []},
        {"issuers": {}, "other_brands": []},
        {"issuers": ["KODEX"], "other_brands": []},
        {"issuers": {"A": ["KODEX"]}, "other_brands": "WON"},
        {"issuers": {"A": ["KODEX"]}, "other_brands": [""]},
        {"issuers": {"A": ["KODEX"]}, "other_brands": ["WON"], "extra": 1},
        {"issuers": {"A": ["KODEX"]}},
    ]
    for data in bad_data:
        _set_issuers(monkeypatch, data)
        with pytest.raises(ConfigError):
            load_issuer_table()


def test_issuer_table_is_longest_first_and_deterministic() -> None:
    table = _table()
    assert table.brands_longest_first == tuple(sorted(table.brand_to_issuer, key=lambda b: (-len(b), b)))
    assert table.brand_to_issuer["koact"] == "삼성액티브자산운용"
    assert table.brand_to_issuer["won"] == ""


def test_resolves_real_name_variants() -> None:
    table = _table()
    sponsors = _spec().sponsors
    cases: list[tuple[str | None, str, str | None]] = [
        ("KODEX 200", "SPONSOR", "삼성자산운용"),
        ("KODEx 미국테크TOP3플러스", "SPONSOR", "삼성자산운용"),
        ("ACE미국500데일리타겟커버드콜(합성)", "SPONSOR", "한국투자신탁운용"),
        ("TIMEFOLIO BBIG액티브", "SPONSOR", "타임폴리오자산운용"),
        ("TIME 미국나스닥100채권혼합50액티브", "SPONSOR", "타임폴리오자산운용"),
        ("KoAct 바이오헬스케어액티브", "NON_SPONSOR", "삼성액티브자산운용"),
        ("WON AI ESG액티브", "NON_SPONSOR", None),
        ("KBKBSTAR 우량업종대표주", "UNKNOWN", None),
        (None, "UNKNOWN", None),
        ("", "UNKNOWN", None),
        ("   ", "UNKNOWN", None),
    ]
    for name, status, issuer in cases:
        result = resolve_issuer(name, table, sponsors)
        assert result.status == status, name
        assert result.issuer == issuer, name


def test_non_sponsor_issuer_outside_sponsor_list() -> None:
    table = _table()
    result = resolve_issuer("KODEX 200", table, frozenset({"미래에셋자산운용"}))
    assert result.status == "NON_SPONSOR"
    assert result.issuer == "삼성자산운용"


def test_no_mid_name_match() -> None:
    result = resolve_issuer("미국 KODEX 추종", _table(), _spec().sponsors)
    assert result.status == "UNKNOWN"
    assert result.issuer is None


def test_autonomous_snapshot_basic() -> None:
    rows = _rows(
        (D1, "069500", "KODEX 200"),
        (D1, "449000", "KoAct 바이오헬스케어액티브"),
        (D1, "999999", "미지정상품"),
    )
    snap = _snapshot(rows, D1)
    assert snap.tickers == ("069500",)
    assert dict(snap.issuer_by_ticker) == {"069500": "삼성자산운용"}
    assert snap.excluded_non_sponsor == 1
    assert snap.unknown == (("999999", "미지정상품"),)
    assert snap.session == D1
    assert snap.track == "autonomous"


def test_future_rows_ignored() -> None:
    base = _rows(
        (D1, "069500", "KODEX 200"),
        (D1, "102110", "TIGER 200"),
        (D2, "069500", "KODEX 200"),
        (D2, "102110", "TIGER 200"),
    )
    future = pl.concat(
        [
            base,
            _rows(
                (D3, "069500", "WON 인버스"),
                (D3, "700000", "KODEX 신규상장"),
                (D3, "102110", "KoAct 신규"),
            ),
        ]
    )
    assert _snapshot(future, D2) == _snapshot(base, D2)


def test_delisted_absent_and_historical_present() -> None:
    rows = _rows(
        (D1, "111111", "KODEX 구형"),
        (D1, "102110", "TIGER 200"),
        (D2, "102110", "TIGER 200"),
    )
    assert "111111" in _snapshot(rows, D1).tickers
    assert "111111" not in _snapshot(rows, D2).tickers


def test_name_glitch_uses_latest_classified() -> None:
    rows = _rows(
        (D1, "Y", "KBSTAR 국고채"),
        (D2, "Y", "KBKBSTAR 국고채"),
    )
    snap = _snapshot(rows, D2)
    assert "Y" in snap.tickers
    assert snap.issuer_by_ticker["Y"] == "KB자산운용"
    assert all(t != "Y" for t, _ in snap.unknown)


def test_transfer_to_non_sponsor_takes_effect() -> None:
    rows = _rows(
        (D1, "Z", "KODEX 자산"),
        (D2, "Z", "WON 자산"),
    )
    snap = _snapshot(rows, D2)
    assert "Z" not in snap.tickers
    assert snap.excluded_non_sponsor == 1


def test_no_classified_row_reported_unknown_with_name_at_session() -> None:
    rows = _rows(
        (D1, "Q", None),
        (D2, "Q", None),
        (D2, "R", "KODEX 자산"),
    )
    snap = _snapshot(rows, D2)
    assert snap.unknown == (("Q", None),)
    assert snap.tickers == ("R",)


def test_null_name_with_classified_history_stays_eligible() -> None:
    rows = _rows(
        (D1, "N", "KODEX 자산"),
        (D2, "N", None),
    )
    snap = _snapshot(rows, D2)
    assert snap.tickers == ("N",)
    assert snap.unknown == ()


def test_as_of_between_sessions() -> None:
    rows = _rows((D1, "069500", "KODEX 200"), (D3, "069500", "KODEX 200"))
    snap = _snapshot(rows, D2)
    assert snap.session == D1
    assert snap.as_of == D2


def test_unsupported_track_fails_closed() -> None:
    rows = _rows((D1, "069500", "KODEX 200"))
    spec = _spec()
    with pytest.raises(UnsupportedTrackError):
        eligible_universe(rows, spec, "domestic_equity", _table(), D1)
    with pytest.raises(KeyError):
        eligible_universe(rows, spec, "nope", _table(), D1)


def test_empty_history() -> None:
    rows = _rows((D2, "069500", "KODEX 200"))
    with pytest.raises(EmptyHistoryError):
        _snapshot(rows, D1)


def test_bad_frame_rejected() -> None:
    good = _rows((D1, "069500", "KODEX 200"))
    with pytest.raises(ValueError, match="column 'name'"):
        _snapshot(good.drop("name"), D1)
    with pytest.raises(ValueError, match="column 'ticker'"):
        _snapshot(good.with_columns(pl.col("ticker").cast(pl.Int64)), D1)
    with pytest.raises(ValueError, match="column 'date'"):
        _snapshot(good.with_columns(pl.col("date").cast(pl.Utf8)), D1)
    with pytest.raises(ValueError, match="unique on"):
        _snapshot(pl.concat([good, good]), D1)


def test_sponsors_from_edition() -> None:
    spec = _spec()
    custom = replace(spec, sponsors=frozenset(spec.sponsors - {"삼성자산운용"}))
    rows = _rows((D1, "069500", "KODEX 200"))
    assert _snapshot(rows, D1, custom).tickers == ()
    assert _snapshot(rows, D1, custom).excluded_non_sponsor == 1
