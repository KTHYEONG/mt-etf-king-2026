"""Point-in-time eligible ETF universe of an edition track, classified by ETF name prefix."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from types import MappingProxyType
from typing import Any, Final, Literal

import polars as pl

from src.arena.contest import EditionSpec
from src.core.config import ConfigError, load_config

Status = Literal["SPONSOR", "NON_SPONSOR", "UNKNOWN"]

_ISSUERS_KEY: Final = "issuers"
_OTHER_BRANDS_KEY: Final = "other_brands"
_OTHER_ISSUER: Final = ""
_UNKNOWN: Final[Status] = "UNKNOWN"
_NON_SPONSOR: Final[Status] = "NON_SPONSOR"
_SPONSOR: Final[Status] = "SPONSOR"
_ANY_SCOPE: Final = "any"


@dataclass(frozen=True)
class IssuerTable:
    brand_to_issuer: Mapping[str, str]
    brands_longest_first: tuple[str, ...]


@dataclass(frozen=True)
class Resolution:
    issuer: str | None
    status: Status


@dataclass(frozen=True)
class UniverseSnapshot:
    as_of: date
    session: date
    track: str
    tickers: tuple[str, ...]
    issuer_by_ticker: Mapping[str, str]
    excluded_non_sponsor: int
    unknown: tuple[tuple[str, str | None], ...]


class UnsupportedTrackError(ValueError):
    """Raised for a track whose asset-class classification is not implemented yet."""


class EmptyHistoryError(ValueError):
    """Raised when no row is dated on or before `as_of`."""


def _normalize(name: str) -> str:
    return "".join(name.split()).casefold()


def _add_brand(table: dict[str, str], raw: Any, issuer: str, where: str) -> None:
    if not isinstance(raw, str) or not raw or any(c.isspace() for c in raw):
        raise ConfigError(f"{where} must be a non-empty brand without whitespace: {raw!r}")
    brand = raw.casefold()
    if brand in table:
        raise ConfigError(f"brand {brand!r} is listed more than once")
    table[brand] = issuer


def load_issuer_table(name: str = "arena_issuers") -> IssuerTable:
    """Return the casefolded brand table of `configs/<name>.yaml`; raises ConfigError on duplicates or malformed entries."""
    data = load_config(name)
    if set(data) != {_ISSUERS_KEY, _OTHER_BRANDS_KEY}:
        raise ConfigError(f"config {name} must have exactly the keys issuers and other_brands")
    issuers = data[_ISSUERS_KEY]
    other_brands = data[_OTHER_BRANDS_KEY]
    if not isinstance(issuers, Mapping) or not issuers:
        raise ConfigError("issuers must be a non-empty mapping")
    if not isinstance(other_brands, list):
        raise ConfigError("other_brands must be a list")

    table: dict[str, str] = {}
    for issuer, brands in issuers.items():
        if not isinstance(issuer, str) or not issuer or not isinstance(brands, list) or not brands:
            raise ConfigError(f"issuer {issuer!r} must map to a non-empty list of brands")
        for raw in brands:
            _add_brand(table, raw, issuer, f"issuers.{issuer}")
    for raw in other_brands:
        _add_brand(table, raw, _OTHER_ISSUER, _OTHER_BRANDS_KEY)

    return IssuerTable(
        brand_to_issuer=MappingProxyType(table),
        brands_longest_first=tuple(sorted(table, key=lambda brand: (-len(brand), brand))),
    )


def resolve_issuer(name: str | None, table: IssuerTable, sponsors: frozenset[str]) -> Resolution:
    """Classify an ETF name by its longest brand prefix after removing whitespace and casefolding."""
    if not name:
        return Resolution(None, _UNKNOWN)
    normalized = _normalize(name)
    brand = next((b for b in table.brands_longest_first if normalized.startswith(b)), None)
    if brand is None:
        return Resolution(None, _UNKNOWN)
    issuer = table.brand_to_issuer[brand]
    if issuer == _OTHER_ISSUER:
        return Resolution(None, _NON_SPONSOR)
    if issuer not in sponsors:
        return Resolution(issuer, _NON_SPONSOR)
    return Resolution(issuer, _SPONSOR)


def _check_schema(etf_rows: pl.DataFrame) -> None:
    schema = etf_rows.schema
    if "date" not in schema or schema["date"] != pl.Date:
        raise ValueError("etf_rows requires a pl.Date column 'date'")
    if "ticker" not in schema or schema["ticker"] != pl.Utf8:
        raise ValueError("etf_rows requires a pl.Utf8 column 'ticker'")
    if "name" not in schema or schema["name"] != pl.Utf8:
        raise ValueError("etf_rows requires a pl.Utf8 column 'name'")


def eligible_universe(
    etf_rows: pl.DataFrame, spec: EditionSpec, track: str, table: IssuerTable, as_of: date
) -> UniverseSnapshot:
    """Return the sponsor-issued tickers of `track` at `as_of`, using only rows dated on or before it.

    Raises KeyError for an unknown track, UnsupportedTrackError for a non-`any` scope,
    ValueError for a malformed frame, and EmptyHistoryError when no row precedes `as_of`.
    """
    rule = spec.tracks[track]
    if rule.scope != _ANY_SCOPE:
        raise UnsupportedTrackError(f"track {track!r} scope {rule.scope!r} needs asset-class classification")
    _check_schema(etf_rows)

    visible = etf_rows.filter(pl.col("date") <= as_of)
    if visible.is_empty():
        raise EmptyHistoryError(f"no ETF rows on or before {as_of}")
    if visible.height != visible.unique(subset=["date", "ticker"]).height:
        raise ValueError("etf_rows must be unique on (date, ticker)")
    session: date = visible.select(pl.col("date").max()).item()

    distinct_names = visible.get_column("name").fill_null("").unique().to_list()
    resolutions = [resolve_issuer(n, table, spec.sponsors) for n in distinct_names]
    lookup = pl.DataFrame(
        {
            "name_key": distinct_names,
            "status": [r.status for r in resolutions],
            "issuer": [r.issuer for r in resolutions],
        },
        schema={"name_key": pl.Utf8, "status": pl.Utf8, "issuer": pl.Utf8},
    )
    classified = (
        visible.with_columns(pl.col("name").fill_null("").alias("name_key"))
        .join(lookup, on="name_key", how="left")
        .filter(pl.col("status") != _UNKNOWN)
        .sort(["ticker", "date"])
        .unique(subset=["ticker"], keep="last")
        .select(["ticker", "status", "issuer"])
    )
    candidates = (
        visible.filter(pl.col("date") == session)
        .select(["ticker", "name"])
        .join(classified, on="ticker", how="left")
        .sort("ticker")
    )

    tickers: list[str] = []
    issuer_by_ticker: dict[str, str] = {}
    excluded_non_sponsor = 0
    unknown: list[tuple[str, str | None]] = []
    for row in candidates.iter_rows(named=True):
        status = row["status"]
        if status is None:
            unknown.append((row["ticker"], row["name"]))
        elif status == _SPONSOR:
            tickers.append(row["ticker"])
            issuer_by_ticker[row["ticker"]] = row["issuer"]
        elif status == _NON_SPONSOR:
            excluded_non_sponsor += 1

    return UniverseSnapshot(
        as_of=as_of,
        session=session,
        track=track,
        tickers=tuple(tickers),
        issuer_by_ticker=MappingProxyType(issuer_by_ticker),
        excluded_non_sponsor=excluded_non_sponsor,
        unknown=tuple(unknown),
    )
