from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.core.calendar import TradingCalendar


def is_open_fillable(open_price: float | None) -> bool:
    if open_price is None:
        return False
    try:
        v = float(open_price)
    except Exception:
        return False
    if not math.isfinite(v):
        return False
    return v > 0


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
        try:
            execution_date = self.calendar.next_session(decision_date)
        except Exception:
            return [], tuple(sorted(target.keys()))
        if panel.height == 0 or "date" not in panel.columns:
            return [], tuple(sorted(target.keys()))
        exec_rows = panel.filter(pl.col("date") == execution_date)
        open_map: dict[str, float | None] = {}
        if exec_rows.height > 0:
            for row in exec_rows.iter_rows(named=True):
                t = str(row.get("ticker"))
                o = row.get("open")
                try:
                    if o is None:
                        open_map[t] = None
                    else:
                        v = float(o)
                        open_map[t] = v
                except Exception:
                    open_map[t] = None
        fills: list[Fill] = []
        unfilled: list[str] = []
        for ticker, weight in target.items():
            tstr = str(ticker)
            if tstr not in open_map:
                unfilled.append(tstr)
                continue
            price = open_map.get(tstr)
            if not is_open_fillable(price):
                unfilled.append(tstr)
                continue
            pf = float(price)  # type: ignore[arg-type]
            fills.append(Fill(ticker=tstr, execution_date=execution_date, price=pf, target_weight=float(weight)))
        fills.sort(key=lambda f: f.ticker)
        unfilled_sorted = tuple(sorted(unfilled))
        return fills, unfilled_sorted
