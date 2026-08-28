from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

import yaml

from src.core.calendar import TradingCalendar


class _UnknownRule:
    def __repr__(self) -> str:
        return "UNKNOWN"

    def __str__(self) -> str:
        return "UNKNOWN"

    def __bool__(self) -> bool:
        return False

    def __eq__(self, other: object) -> bool:
        return isinstance(other, _UnknownRule)

    def __hash__(self) -> int:
        return hash("UNKNOWN")


UNKNOWN: Final[_UnknownRule] = _UnknownRule()


@dataclass(frozen=True)
class TournamentRules:
    name: str
    start_date: date
    end_date: date
    initial_capital: int
    category: str
    leverage_allowed: bool | _UnknownRule
    inverse_allowed: bool | _UnknownRule
    max_weight: float | _UnknownRule
    cash_allowed: bool | _UnknownRule
    sponsor_etf_only: bool
    manifest_path: Path | None
    issuer_whitelist: tuple[str, ...] | None
    commission_bps: float | _UnknownRule
    slippage_bps: float | _UnknownRule
    max_order_to_adv: float
    stress_grid: tuple[float, ...]

    @classmethod
    def from_yaml(cls, path: Path) -> TournamentRules:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        # tournament may be nested under 'tournament'
        data = raw.get("tournament") if isinstance(raw, dict) and "tournament" in raw else raw
        if not isinstance(data, dict):
            data = {}

        def parse_unknown(val: object) -> object:
            if isinstance(val, str) and val.strip().lower() == "unknown":
                return UNKNOWN
            return val

        def to_bool_unknown(val: object) -> bool | _UnknownRule:
            parsed = parse_unknown(val)
            if parsed is UNKNOWN:
                return UNKNOWN
            if isinstance(parsed, bool):
                return parsed
            if isinstance(parsed, str):
                low = parsed.lower()
                if low == "true":
                    return True
                if low == "false":
                    return False
            if parsed is None:
                return UNKNOWN
            return bool(parsed)

        def to_float_unknown(val: object) -> float | _UnknownRule:
            parsed = parse_unknown(val)
            if parsed is UNKNOWN:
                return UNKNOWN
            if parsed is None:
                return UNKNOWN
            try:
                return float(parsed)  # type: ignore[arg-type]
            except Exception:
                return UNKNOWN

        start_raw = data.get("start_date")
        end_raw = data.get("end_date")
        # handle date strings
        start_date = date.fromisoformat(str(start_raw)) if start_raw else date(2026, 9, 21)
        end_date = date.fromisoformat(str(end_raw)) if end_raw else date(2026, 11, 13)
        capital = int(data.get("capital") or data.get("initial_capital") or 1_000_000_000)

        # category
        cat_data = data.get("category") or {}
        if isinstance(cat_data, dict):
            category = str(cat_data.get("name") or "autonomous")
            lev_raw = cat_data.get("leverage_allowed")
            inv_raw = cat_data.get("inverse_allowed")
            max_w_raw = cat_data.get("max_weight")
            cash_raw = cat_data.get("cash_allowed")
        else:
            category = str(cat_data)
            lev_raw = data.get("leverage_allowed")
            inv_raw = data.get("inverse_allowed")
            max_w_raw = data.get("max_weight")
            cash_raw = data.get("cash_allowed")

        leverage_allowed = to_bool_unknown(lev_raw) if lev_raw is not None else UNKNOWN
        inverse_allowed = to_bool_unknown(inv_raw) if inv_raw is not None else UNKNOWN
        max_weight = to_float_unknown(max_w_raw) if max_w_raw is not None else UNKNOWN
        cash_allowed = to_bool_unknown(cash_raw) if cash_raw is not None else UNKNOWN

        # sponsor_etf_only
        rules = data.get("rules") or {}
        sponsor_etf_only = bool(rules.get("sponsor_etf_only", True)) if isinstance(rules, dict) else True

        # manifest_path
        manifest_raw = data.get("manifest")
        manifest_path: Path | None
        if manifest_raw is None or (isinstance(manifest_raw, str) and manifest_raw.lower() in ("null", "none", "")):
            manifest_path = None
        elif isinstance(manifest_raw, str):
            manifest_path = Path(manifest_raw)
        else:
            manifest_path = None

        # issuer whitelist from sponsors.asset_managers
        sponsors = data.get("sponsors") or {}
        issuer_whitelist: tuple[str, ...] | None = None
        if isinstance(sponsors, dict):
            am = sponsors.get("asset_managers")
            if isinstance(am, list) and am:
                issuer_whitelist = tuple(str(x) for x in am)

        # commission/slippage
        commission_raw = data.get("commission_bps")
        if commission_raw is None:
            commission_raw = (data.get("rules") or {}).get("commission_bps") if isinstance(data.get("rules"), dict) else None
        slippage_raw = data.get("slippage_bps")
        if slippage_raw is None:
            slippage_raw = (data.get("rules") or {}).get("slippage_bps") if isinstance(data.get("rules"), dict) else None
        commission_bps: float | _UnknownRule = to_float_unknown(commission_raw) if commission_raw is not None else UNKNOWN
        slippage_bps: float | _UnknownRule = to_float_unknown(slippage_raw) if slippage_raw is not None else UNKNOWN

        # max_order_to_adv
        mota = data.get("max_order_to_adv")
        if mota is None:
            mota = (rules.get("max_order_to_adv") if isinstance(rules, dict) else None)
        max_order_to_adv = float(mota) if mota is not None else 0.05

        # stress_grid
        grid_raw = data.get("stress_grid")
        if grid_raw is None:
            grid_raw = data.get("participation_grid")
        if isinstance(grid_raw, list) and grid_raw:
            stress_grid = tuple(float(x) for x in grid_raw)
        else:
            stress_grid = (0.01, 0.02, 0.05, 0.10)

        return cls(
            name=str(data.get("name") or category or "tournament"),
            start_date=start_date,
            end_date=end_date,
            initial_capital=capital,
            category=category,
            leverage_allowed=leverage_allowed,
            inverse_allowed=inverse_allowed,
            max_weight=max_weight,
            cash_allowed=cash_allowed,
            sponsor_etf_only=sponsor_etf_only,
            manifest_path=manifest_path,
            issuer_whitelist=issuer_whitelist,
            commission_bps=commission_bps,
            slippage_bps=slippage_bps,
            max_order_to_adv=max_order_to_adv,
            stress_grid=stress_grid,
        )

    def horizon_sessions(self, calendar: TradingCalendar) -> int:
        return calendar.session_count(self.start_date, self.end_date)

    def scenarios_for(self, field: str) -> tuple[object, ...]:
        val = getattr(self, field, None)
        if val is UNKNOWN:
            # boolean unknowns -> (True, False), numeric unknowns -> stress_grid
            if field in ("leverage_allowed", "inverse_allowed", "cash_allowed"):
                return (True, False)
            if field in ("max_weight", "commission_bps", "slippage_bps"):
                return self.stress_grid
            return (True, False)
        if isinstance(val, bool):
            return (val,)
        if isinstance(val, (int, float)):
            return (val,)
        if val is None:
            return ()
        if isinstance(val, tuple):
            return val
        return (val,)
