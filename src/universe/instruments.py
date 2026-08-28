from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

import polars as pl
import yaml

from src.universe.families import resolve_family_key
from src.universe.taxonomy import Taxonomy, normalize_index_key


class Confidence(StrEnum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


def resolve_leverage(name: str) -> tuple[int, Confidence]:
    # Ordered first-match: inverse2X -> inverse1X -> leverage
    # Contradictory detection: more than one category present => LOW
    # Handle overlapping substrings carefully.
    has_inv2 = any(tok in name for tok in ("인버스2X", "-2X", "곱버스"))
    # For inv1/leverage detection, strip inv2 tokens to avoid double counting
    temp = name
    for tok in ("인버스2X", "-2X", "곱버스"):
        temp = temp.replace(tok, "")
    has_inv1 = ("인버스" in temp) or ("-1X" in temp)
    has_lev = ("레버리지" in temp) or ("2배" in temp) or ("2X" in temp)
    # contradictory if >=2 categories
    contradictory = sum((has_inv2, has_inv1, has_lev)) >= 2

    # Determine leverage multiple by ordered first-match on original name
    lev: int
    if has_inv2:
        lev = -2
    elif "인버스" in name or "-1X" in name:
        # need to ensure inv2 already handled
        # but if name is "인버스2X", has_inv2 true already, so this branch not reached
        lev = -1
    elif "레버리지" in name or "2X" in name or "2배" in name:
        lev = 2
    else:
        lev = 1

    conf = Confidence.LOW if contradictory else Confidence.HIGH
    return lev, conf


def load_sponsor_brand_map(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    asset_managers = data.get("asset_managers") or {}
    out: dict[str, str] = {}
    for issuer, info in asset_managers.items():
        brands: list[str] = []
        if isinstance(info, dict):
            brands = list(info.get("brands") or [])
        elif isinstance(info, list):
            brands = list(info)
        for b in brands:
            out[str(b)] = str(issuer)
    return out


def resolve_issuer(name: str, brand_map: Mapping[str, str]) -> str:
    # brand is first token of ISU_NM
    if not name:
        return "UNKNOWN"
    first = name.strip().split()[0] if name.strip() else ""
    return brand_map.get(first, "UNKNOWN")


@dataclass(frozen=True)
class InstrumentAttributes:
    ticker: str
    name: str
    issuer: str
    leverage_multiple: int
    leverage_family_key: str
    is_synthetic: bool
    is_hedged: bool
    is_active: bool
    index_key: str
    theme: str
    first_seen: date
    last_seen: date
    left_censored: bool
    confidence: Confidence


class InstrumentMaster:
    def __init__(self, attributes: Mapping[str, InstrumentAttributes], panel_start: date) -> None:
        self._attributes: dict[str, InstrumentAttributes] = dict(attributes)
        self._panel_start = panel_start

    @property
    def attributes(self) -> Mapping[str, InstrumentAttributes]:
        return self._attributes

    @property
    def panel_start(self) -> date:
        return self._panel_start

    @classmethod
    def build(
        cls,
        panel: pl.DataFrame,
        taxonomy: Taxonomy,
        brand_map: Mapping[str, str],
        family_overrides: Mapping[str, str] | None = None,
        overrides: Mapping[str, Mapping[str, object]] | None = None,
    ) -> InstrumentMaster:
        if panel.height == 0:
            raise ValueError("panel is empty")
        # panel expected columns: date, ticker, name, underlying_index_name etc.
        # Determine panel start/end
        dates = panel.select(pl.col("date")).to_series().to_list()
        # filter None
        dates = [d for d in dates if d is not None]
        if not dates:
            raise ValueError("panel has no dates")
        panel_start = min(dates)
        panel_end = max(dates)

        # Group by ticker
        # Use polars group_by to get first_seen/last_seen
        grouped = panel.group_by("ticker").agg(
            [
                pl.col("date").min().alias("first_seen"),
                pl.col("date").max().alias("last_seen"),
            ]
        )
        # Build mapping from ticker to first/last
        first_map: dict[str, date] = {}
        last_map: dict[str, date] = {}
        for row in grouped.iter_rows(named=True):
            t = str(row["ticker"])
            first_map[t] = row["first_seen"]
            last_map[t] = row["last_seen"]

        # For each ticker, get representative latest row for name/index
        # Sort panel by date ascending, then take last per ticker
        # Use group_by to get last name
        # Simpler: iterate tickers, filter panel for ticker and take row with max date
        attributes: dict[str, InstrumentAttributes] = {}
        # Precompute for performance: sort panel?
        # We'll use python loop over tickers
        for ticker, first_seen in first_map.items():
            last_seen = last_map[ticker]
            left_censored = first_seen == panel_start
            # get rows for ticker
            sub = panel.filter(pl.col("ticker") == ticker)
            # pick row with max date (last_seen); if multiple rows same date, take last
            # Sort by date
            sub_sorted = sub.sort("date")
            # take last row
            last_row = sub_sorted.row(sub_sorted.height - 1, named=True) if sub_sorted.height > 0 else {}
            name = str(last_row.get("name") or "")
            idx_nm = last_row.get("underlying_index_name")
            # fallback column name may be 'underlying_index_name' or 'IDX_IND_NM'
            if idx_nm is None:
                idx_nm = last_row.get("IDX_IND_NM")
            # index_key
            index_key = normalize_index_key(idx_nm if isinstance(idx_nm, str) or idx_nm is None else str(idx_nm))
            leverage_family_key = resolve_family_key(index_key, family_overrides)
            # leverage
            lev, conf = resolve_leverage(name)
            # issuer
            issuer = resolve_issuer(name, brand_map)
            # is_synthetic / is_hedged
            is_synthetic = "(합성" in name
            is_hedged = "(H)" in name
            is_active = last_seen == panel_end
            # theme via taxonomy
            theme = taxonomy.classify(name, idx_nm if isinstance(idx_nm, str) else None)
            # overrides handling
            if overrides is not None and ticker in overrides:
                ov = overrides[ticker]
                # ov is Mapping[str, object]
                # each entry takes precedence and sets confidence HIGH
                if "name" in ov:
                    name = str(ov["name"])
                if "issuer" in ov:
                    issuer = str(ov["issuer"])
                if "leverage_multiple" in ov:
                    lev = int(ov["leverage_multiple"])  # type: ignore[call-overload]
                if "leverage_family_key" in ov:
                    leverage_family_key = str(ov["leverage_family_key"])
                if "is_synthetic" in ov:
                    is_synthetic = bool(ov["is_synthetic"])
                if "is_hedged" in ov:
                    is_hedged = bool(ov["is_hedged"])
                if "is_active" in ov:
                    is_active = bool(ov["is_active"])
                if "index_key" in ov:
                    index_key = str(ov["index_key"])
                if "theme" in ov:
                    theme = str(ov["theme"])
                # confidence HIGH for overridden
                conf = Confidence.HIGH
                # also if ticker override provides first_seen/last_seen?
                if "first_seen" in ov:
                    first_seen = ov["first_seen"]  # type: ignore[assignment]
                if "last_seen" in ov:
                    last_seen = ov["last_seen"]  # type: ignore[assignment]
                if "left_censored" in ov:
                    left_censored = bool(ov["left_censored"])
                if "confidence" in ov:
                    # allow override confidence as string
                    cval = ov["confidence"]
                    if isinstance(cval, str):
                        try:
                            conf = Confidence(cval)
                        except ValueError:
                            conf = Confidence.HIGH
                    elif isinstance(cval, Confidence):
                        conf = cval
                # for any other override, also ensure confidence HIGH unless explicitly set
                if "confidence" not in ov:
                    conf = Confidence.HIGH
            attr = InstrumentAttributes(
                ticker=ticker,
                name=name,
                issuer=issuer,
                leverage_multiple=lev,
                leverage_family_key=leverage_family_key,
                is_synthetic=is_synthetic,
                is_hedged=is_hedged,
                is_active=is_active,
                index_key=index_key,
                theme=theme,
                first_seen=first_seen,
                last_seen=last_seen,
                left_censored=left_censored,
                confidence=conf,
            )
            attributes[ticker] = attr
        return cls(attributes=attributes, panel_start=panel_start)

    def as_of(self, day: date) -> dict[str, InstrumentAttributes]:
        return {ticker: attr for ticker, attr in self._attributes.items() if attr.first_seen <= day <= attr.last_seen}
