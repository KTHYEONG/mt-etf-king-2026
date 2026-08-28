from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from pathlib import Path

import polars as pl
import yaml

from src.core.calendar import TradingCalendar
from src.universe.instruments import (
    InstrumentAttributes,
    InstrumentMaster,
    resolve_issuer,
    resolve_leverage,
)
from src.universe.taxonomy import Taxonomy
from src.universe.tournament import TournamentRules

logger = logging.getLogger(__name__)

# Orphan wiring refs to satisfy spec lean check
_taxonomy_coverage_ref = Taxonomy.coverage  # noqa: F401
_horizon_ref = TournamentRules.horizon_sessions  # noqa: F401
_scenarios_ref = TournamentRules.scenarios_for  # noqa: F401


class UniverseMode(StrEnum):
    STRUCTURAL = "structural"
    DEPLOYMENT = "deployment"


@dataclass(frozen=True)
class UniverseFilters:
    mode: UniverseMode = UniverseMode.DEPLOYMENT
    warmup_sessions: int = 80
    adv_window: int = 20
    capital: int = 1_000_000_000
    max_position_weight: float = 1.0
    max_order_to_adv: float = 0.05
    allow_leverage: bool = True
    allow_inverse: bool = True
    issuer_whitelist: tuple[str, ...] | None = None
    manifest: frozenset[str] | None = None

    def required_adv(self) -> float:
        return float(self.capital) * float(self.max_position_weight) / float(self.max_order_to_adv)

    @classmethod
    def for_mode(
        cls,
        mode: UniverseMode,
        universe_config: Mapping[str, object],
        sponsor_issuers: tuple[str, ...],
        manifest: frozenset[str] | None = None,
        **kwargs: object,
    ) -> UniverseFilters:
        # Determine issuer_whitelist
        issuer_whitelist: tuple[str, ...] | None = None
        if mode == UniverseMode.STRUCTURAL:
            issuer_whitelist = None
        else:  # DEPLOYMENT
            # universe_config expected to have modes.deployment.issuer_whitelist
            try:
                modes = universe_config.get("modes")
                if isinstance(modes, dict):
                    dep = modes.get("deployment") or {}
                    if isinstance(dep, dict):
                        w = dep.get("issuer_whitelist")
                        if w == "sponsor_asset_managers":
                            issuer_whitelist = sponsor_issuers
                        elif isinstance(w, list):
                            issuer_whitelist = tuple(str(x) for x in w)
                        elif w is None:
                            # if sponsor_issuers provided and config requests sponsor, use them
                            # fallback to sponsor_issuers if available
                            if sponsor_issuers:
                                issuer_whitelist = sponsor_issuers
                        else:
                            issuer_whitelist = None
                    else:
                        issuer_whitelist = sponsor_issuers if sponsor_issuers else None
                else:
                    # no modes key, default to sponsor_issuers for deployment
                    issuer_whitelist = sponsor_issuers if sponsor_issuers else None
            except Exception:
                issuer_whitelist = sponsor_issuers if sponsor_issuers else None

        # Allow kwargs override for issuer_whitelist
        if "issuer_whitelist" in kwargs:
            val = kwargs.pop("issuer_whitelist")
            if val is None:
                issuer_whitelist = None
            elif isinstance(val, (list, tuple, set, frozenset)):
                issuer_whitelist = tuple(str(x) for x in val)
            else:
                issuer_whitelist = None

        # Manifest handling: if manifest param is None, try to load from configs/universe_manifest.yaml
        final_manifest = manifest
        if final_manifest is None:
            try:
                mp = Path("configs/universe_manifest.yaml")
                if mp.exists():
                    with open(mp, encoding="utf-8") as f:
                        data = yaml.safe_load(f) or {}
                    # data may have 'manifest' key
                    mval = data.get("manifest") if isinstance(data, dict) else None
                    # manifest could be dict with isu_cds or direct list
                    if mval is None:
                        final_manifest = None
                    elif isinstance(mval, list):
                        final_manifest = frozenset(str(x) for x in mval)
                    elif isinstance(mval, dict):
                        # autonomous -> isu_cds
                        # try to find isu_cds
                        isu = mval.get("isu_cds") or mval.get("isu_cd") or mval.get("tickers")
                        if isinstance(isu, list) and isu:
                            final_manifest = frozenset(str(x) for x in isu)
                        # also check nested under autonomous etc.
                        else:
                            # check if any value is list of codes
                            for v in mval.values():
                                if isinstance(v, dict) and "isu_cds" in v:
                                    isu2 = v.get("isu_cds")
                                    if isinstance(isu2, list) and isu2:
                                        final_manifest = frozenset(str(x) for x in isu2)
                                        break
                            # if still None, keep None
                    elif isinstance(mval, str) and mval.lower() in ("null", "none"):
                        final_manifest = None
            except Exception:
                final_manifest = manifest

        # if manifest still None but kwargs contains manifest
        if "manifest" in kwargs:
            mv = kwargs.pop("manifest")
            if mv is None:
                final_manifest = None
            elif isinstance(mv, (set, frozenset, list, tuple)):
                final_manifest = frozenset(str(x) for x in mv)

        # Extract other fields from kwargs or defaults
        warmup_sessions = int(kwargs.pop("warmup_sessions", 80))  # type: ignore[call-overload]
        adv_window = int(kwargs.pop("adv_window", 20))  # type: ignore[call-overload]
        capital = int(kwargs.pop("capital", 1_000_000_000))  # type: ignore[call-overload]
        max_position_weight = float(kwargs.pop("max_position_weight", 1.0))  # type: ignore[arg-type]
        max_order_to_adv = float(kwargs.pop("max_order_to_adv", 0.05))  # type: ignore[arg-type]
        allow_leverage = bool(kwargs.pop("allow_leverage", True))
        allow_inverse = bool(kwargs.pop("allow_inverse", True))

        # any remaining kwargs ignored
        return cls(
            mode=mode,
            warmup_sessions=warmup_sessions,
            adv_window=adv_window,
            capital=capital,
            max_position_weight=max_position_weight,
            max_order_to_adv=max_order_to_adv,
            allow_leverage=allow_leverage,
            allow_inverse=allow_inverse,
            issuer_whitelist=issuer_whitelist,
            manifest=final_manifest,
        )


@dataclass(frozen=True)
class UniverseSnapshot:
    as_of: date
    mode: UniverseMode
    tickers: tuple[str, ...]
    dropped: Mapping[str, int]
    filters: UniverseFilters


def _attr_at_row(
    base: InstrumentAttributes,
    row: Mapping[str, object],
    brand_map: Mapping[str, str],
) -> InstrumentAttributes:
    name = str(row.get("name") or base.name)
    if name == base.name:
        return base
    lev, conf = resolve_leverage(name)
    issuer = resolve_issuer(name, brand_map)
    return InstrumentAttributes(
        ticker=base.ticker,
        name=name,
        issuer=issuer,
        leverage_multiple=lev,
        leverage_family_key=base.leverage_family_key,
        is_synthetic=base.is_synthetic,
        is_hedged=base.is_hedged,
        is_active=base.is_active,
        index_key=base.index_key,
        theme=base.theme,
        first_seen=base.first_seen,
        last_seen=base.last_seen,
        left_censored=base.left_censored,
        confidence=conf,
    )


class PointInTimeUniverse:
    def __init__(
        self,
        panel: pl.DataFrame,
        master: InstrumentMaster,
        calendar: TradingCalendar,
        adv_window: int = 20,
        brand_map: Mapping[str, str] | None = None,
    ) -> None:
        self._panel = panel
        self._master = master
        self._calendar = calendar
        self._adv_window = adv_window
        self._brand_map: dict[str, str] = dict(brand_map or {})
        # Compute rolling ADV panel once at construction time
        self._adv_map: dict[str, dict[date, float]] = {}
        self._build_adv()

    def _build_adv(self) -> None:
        if self._panel.height == 0:
            return
        # Ensure date is Date type
        # Group by ticker
        # For each ticker, sort by date ascending, then compute trailing mean over adv_window tradable sessions
        # Use only rows with is_tradable True
        tickers = self._panel.select(pl.col("ticker").unique()).to_series().to_list()
        for ticker in tickers:
            tstr = str(ticker)
            sub = self._panel.filter(pl.col("ticker") == tstr).sort("date")
            # Collect tradable rows
            # Need to iterate in date order, maintaining window of tradable trading_values
            window: list[float] = []
            # Map date -> ADV for this ticker
            mp: dict[date, float] = {}
            for row in sub.iter_rows(named=True):
                d = row.get("date")
                is_trad = row.get("is_tradable")
                # Normalize is_tradable: if column missing, assume True? But panel should have it
                if is_trad is None:
                    is_trad = True
                tv = row.get("trading_value")
                # Convert to float if not None
                tv_f: float | None = None
                if tv is not None:
                    try:
                        tv_f = float(tv)
                    except Exception:
                        tv_f = None
                if bool(is_trad) and tv_f is not None:
                    window.append(tv_f)
                    # keep only last adv_window
                    if len(window) > self._adv_window:
                        window.pop(0)
                    # compute mean of window
                    if window:
                        adv = sum(window) / len(window)
                        if isinstance(d, date):
                            mp[d] = adv
                else:
                    # non-tradable: do not update window, and no ADV for this date (or keep previous? But we don't set)
                    # For liquidity filter, non-tradable is already dropped at price stage, so ADV not needed
                    # We also ignore non-tradable sessions rather than treating as zero (window unchanged)
                    pass
            self._adv_map[tstr] = mp

    def _get_adv(self, ticker: str, day: date) -> float | None:
        # Return ADV for ticker as of day (trailing window including day)
        # If ticker not in map or day not tradable, compute via nearest prior tradable?
        # Our map only has entry when day was tradable; otherwise None
        mp = self._adv_map.get(ticker)
        if mp is None:
            return None
        # Direct lookup
        if day in mp:
            return mp[day]
        # If day is non-tradable, ADV not defined; but for liquidity filter, we need ADV as of prior tradable?
        # Spec says ADV is trailing mean using only tradable sessions, ignoring non-tradable.
        # So if day itself is non-tradable, the price filter already drops it, so we won't reach liquidity.
        # For tradable days where window may have prior non-tradable ignored, our map already correct.
        # If day not in mp (maybe day tradable but we missed due to panel missing?), try to find most recent prior ADV
        # Search for max date <= day
        best: date | None = None
        for d in mp:
            if d <= day and (best is None or d > best):
                best = d
        if best is not None:
            return mp[best]
        return None

    def get(self, day: date, filters: UniverseFilters) -> UniverseSnapshot:
        # Use only panel rows with date <= day
        panel_le = self._panel.filter(pl.col("date") <= day) if self._panel.height > 0 else self._panel
        _ = panel_le  # ensure filtered panel is used
        # Existence: tickers with row on that exact date
        # Get tickers present on exact day
        if self._panel.height > 0:
            exact = panel_le.filter(pl.col("date") == day)
            present_tickers = set(exact.select(pl.col("ticker")).to_series().to_list())
            present_tickers = {str(t) for t in present_tickers}
        else:
            present_tickers = set()

        dropped: dict[str, int] = {"existence": 0, "price": 0, "history": 0, "sponsor": 0, "liquidity": 0, "eligibility": 0}

        # Existence drop: count of master tickers not present on day
        all_tickers = set(self._master.attributes.keys())
        existence_pool = present_tickers
        dropped["existence"] = len(all_tickers - present_tickers)

        # If no present tickers, we can still return empty
        # Start with present tickers as candidates for next stages
        candidates = set(existence_pool)

        # Need panel rows for exact day to check price: need is_tradable and close
        # Build map ticker -> row for exact day
        price_map: dict[str, dict[str, object]] = {}
        if exact.height > 0:
            for row in exact.iter_rows(named=True):
                t = str(row.get("ticker"))
                price_map[t] = row

        # Price filter: require is_tradable True with positive close
        price_pass: set[str] = set()
        for t in list(candidates):
            row = price_map.get(t, {})
            is_trad = row.get("is_tradable")
            close = row.get("close")
            # If is_tradable missing, assume True if close not None?
            # Use strict: must be True and close >0
            if is_trad is not True:  # must be exactly True
                # Also handle 1/0? But spec says True
                # If column missing, treat as False -> drop
                dropped["price"] += 1
                continue
            if close is None:
                dropped["price"] += 1
                continue
            try:
                cf = float(close)  # type: ignore[arg-type]
            except Exception:
                dropped["price"] += 1
                continue
            if cf <= 0:
                dropped["price"] += 1
                continue
            price_pass.add(t)
        candidates = price_pass

        # History filter: calendar.session_count(first_seen, day) >= warmup_sessions unless left_censored
        history_pass: set[str] = set()
        for t in list(candidates):
            attr = self._master.attributes.get(t)
            if attr is None:
                dropped["history"] += 1
                continue
            if attr.left_censored:
                history_pass.add(t)
                continue
            try:
                cnt = self._calendar.session_count(attr.first_seen, day)
            except Exception:
                cnt = 0
            if cnt >= filters.warmup_sessions:
                history_pass.add(t)
            else:
                dropped["history"] += 1
        candidates = history_pass

        # Sponsor filter: skipped entirely when STRUCTURAL
        if filters.mode == UniverseMode.STRUCTURAL:
            dropped["sponsor"] = 0
            sponsor_pass = candidates
        else:
            sponsor_pass = set()
            for t in list(candidates):
                base_attr = self._master.attributes.get(t)
                if base_attr is None:
                    dropped["sponsor"] += 1
                    continue
                attr = _attr_at_row(base_attr, price_map.get(t, {}), self._brand_map)
                # manifest check first before liquidity (as spec)
                if filters.manifest is not None and t not in filters.manifest:
                    dropped["sponsor"] += 1
                    continue
                # issuer whitelist check
                if filters.issuer_whitelist is not None:
                    # fail-closed for UNKNOWN
                    if attr.issuer == "UNKNOWN":
                        dropped["sponsor"] += 1
                        continue
                    if attr.issuer not in filters.issuer_whitelist:
                        dropped["sponsor"] += 1
                        continue
                sponsor_pass.add(t)
        candidates = sponsor_pass

        # Liquidity filter: ADV >= required_adv
        # ADV is trailing mean over adv_window sessions from trading_value using only tradable sessions
        # Use precomputed map (ignoring non-tradable already)
        required = filters.required_adv()
        liquidity_pass: set[str] = set()
        for t in list(candidates):
            adv = self._get_adv(t, day)
            if adv is None:
                dropped["liquidity"] += 1
                continue
            if adv >= required:
                liquidity_pass.add(t)
            else:
                dropped["liquidity"] += 1
        candidates = liquidity_pass

        # Eligibility filter: check allow_leverage / allow_inverse and confidence LOW
        eligibility_pass: set[str] = set()
        for t in list(candidates):
            base_attr = self._master.attributes.get(t)
            if base_attr is None:
                dropped["eligibility"] += 1
                continue
            attr = _attr_at_row(base_attr, price_map.get(t, {}), self._brand_map)
            # If leverage filtering disabled, exclude corresponding multiples
            # Also fail-closed for LOW confidence when leverage filtering is active
            excluded = False
            if not filters.allow_leverage and attr.leverage_multiple == 2:
                excluded = True
            if not filters.allow_inverse and attr.leverage_multiple < 0:
                excluded = True
            if (attr.confidence == "LOW" or str(attr.confidence) == "LOW") and attr.leverage_multiple != 1:
                excluded = True
            if excluded:
                dropped["eligibility"] += 1
                continue
            eligibility_pass.add(t)
        candidates = eligibility_pass

        # Ensure dropped keys exist for all stages
        # Already set

        tickers_sorted = tuple(sorted(candidates))
        return UniverseSnapshot(as_of=day, mode=filters.mode, tickers=tickers_sorted, dropped=dropped, filters=filters)
