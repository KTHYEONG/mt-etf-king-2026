from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.core.calendar import TradingCalendar


@dataclass(frozen=True)
class Fill:
    ticker: str
    execution_date: date
    price: float
    target_weight: float


class NextOpenExecution:
    def __init__(self, calendar: TradingCalendar) -> None:
        self.calendar = calendar

    def resolve(
        self,
        target: Mapping[str, float],
        panel: pl.DataFrame,
        decision_date: date,
    ) -> tuple[list[Fill], tuple[str, ...]]:
        if not target:
            return [], ()
        # Need next session strictly after decision_date
        try:
            execution_date = self.calendar.next_session(decision_date)
        except Exception:
            return [], tuple(sorted(target.keys()))
        # If execution_date not in panel dates? We need to check panel's execution date rows
        # Filter panel to execution_date
        if panel.height == 0 or "date" not in panel.columns:
            return [], tuple(sorted(target.keys()))
        exec_rows = panel.filter(pl.col("date") == execution_date)
        # Build lookup ticker -> (open, is_tradable)
        # Use iter_rows for robustness
        open_map: dict[str, float | None] = {}
        tradable_map: dict[str, bool] = {}
        if exec_rows.height > 0:
            for row in exec_rows.iter_rows(named=True):
                t = str(row.get("ticker"))
                # open column may be named 'open'
                o = row.get("open")
                # also handle 'open' vs maybe case? Use 'open' key directly
                # If column named differently, try alternatives but spec says open column
                # Capture null
                try:
                    if o is None:
                        open_map[t] = None
                    else:
                        # Polars may return None for null, otherwise float
                        open_map[t] = float(o)
                        if open_map[t] is not None and (open_map[t] != open_map[t]):  # NaN
                            open_map[t] = None
                except Exception:
                    open_map[t] = None
                is_trad = row.get("is_tradable")
                # If column missing, assume True? But spec says must check is_tradable False -> unfilled
                # Default to True if not present? Safer to treat missing as True for backcompat but tests set it explicitly
                if is_trad is None:
                    tradable_map[t] = True
                else:
                    tradable_map[t] = bool(is_trad)
        fills: list[Fill] = []
        unfilled: list[str] = []
        for ticker, weight in target.items():
            tstr = str(ticker)
            # Check if ticker has row on execution_date
            if tstr not in open_map:
                unfilled.append(tstr)
                continue
            is_trad = tradable_map.get(tstr, True)
            if not is_trad:
                unfilled.append(tstr)
                continue
            price = open_map.get(tstr)
            if price is None:
                unfilled.append(tstr)
                continue
            try:
                pf = float(price)
            except Exception:
                unfilled.append(tstr)
                continue
            if pf != pf or pf <= 0:  # NaN or non-positive considered unfilled? null case already
                unfilled.append(tstr)
                continue
            fills.append(Fill(ticker=tstr, execution_date=execution_date, price=pf, target_weight=float(weight)))
        # Sort fills/unfilled for determinism
        fills.sort(key=lambda f: f.ticker)
        unfilled_sorted = tuple(sorted(unfilled))
        return fills, unfilled_sorted
