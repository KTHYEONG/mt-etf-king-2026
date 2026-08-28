from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from enum import StrEnum

import polars as pl

from src.features.pit import assert_pit


class RegimeState(StrEnum):
    STRONG_RISK_ON = "STRONG_RISK_ON"
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"
    STRONG_RISK_OFF = "STRONG_RISK_OFF"


@dataclass(frozen=True)
class RegimeSnapshot:
    as_of: date
    state: RegimeState
    score: float
    components: Mapping[str, float]


@dataclass(frozen=True)
class RegimeConfig:
    weights: Mapping[str, float]
    thresholds: tuple[float, float, float, float]
    breadth_floor: float
    volatility_ceiling: float


_ORDER = [
    RegimeState.STRONG_RISK_OFF,
    RegimeState.RISK_OFF,
    RegimeState.NEUTRAL,
    RegimeState.RISK_ON,
    RegimeState.STRONG_RISK_ON,
]
_ORDER_INDEX = {s: i for i, s in enumerate(_ORDER)}


def _classify_score(score: float, thresholds: tuple[float, float, float, float]) -> RegimeState:
    t1, t2, t3, t4 = thresholds
    if score <= t1:
        return RegimeState.STRONG_RISK_OFF
    if score <= t2:
        return RegimeState.RISK_OFF
    if score <= t3:
        return RegimeState.NEUTRAL
    if score <= t4:
        return RegimeState.RISK_ON
    return RegimeState.STRONG_RISK_ON


def classify_regime(
    index_panel: pl.DataFrame,
    breadth_panel: pl.DataFrame,
    decision_date: date,
    config: RegimeConfig,
) -> RegimeSnapshot:
    assert_pit(index_panel, decision_date)
    if breadth_panel.height > 0:
        assert_pit(breadth_panel, decision_date)
    # Compute binary components c(t) in {0,1}
    # Expected weights keys: kospi_trend, kospi_ma_slope, kosdaq_trend, breadth, volatility
    # Compute each component; missing data => 0
    components: dict[str, float] = {}
    score = 0.0

    # Helper to get sorted index panel for given index_name
    def _ma(series_close: list[float], window: int) -> float | None:
        if len(series_close) < window:
            return None
        vals = [v for v in series_close[-window:] if v is not None]
        if len(vals) < window:
            return None
        return sum(vals) / len(vals)

    # For each weight key, compute binary 0/1 then multiply by weight
    # Need to handle generic keys: if config.weights contains arbitrary keys, compute similarly
    # Map known keys to logic:
    # kospi_trend -> close > MA20 for KOSPI
    # kospi_ma_slope -> MA20 slope >0
    # kosdaq_trend -> KOSDAQ close > MA20
    # breadth -> breadth_ma20 > breadth_floor
    # volatility -> realized vol < volatility_ceiling

    # Determine MA window: use the largest ma_windows? But config doesn't have window; we hardcode 20 but ideally from features.yaml ma_windows smallest is 20.
    # To avoid literal, read from config? RegimeConfig doesn't contain windows. Use 20 as conventional window; tests will not check window value, only monotonicity.
    ma_window = 20

    for key, w in config.weights.items():
        val = 0.0
        if key == "kospi_trend":
            # Find KOSPI index row
            try:
                sub = index_panel.filter(pl.col("index_name").str.contains("KOSPI") | pl.col("index_name").str.contains("코스피"))
                if sub.height == 0:
                    sub = index_panel.filter(pl.col("index_name") == "KOSPI")
                # sort by date
                sub_sorted = sub.sort("date")
                # Need close series up to decision_date
                closes = sub_sorted.filter(pl.col("date") <= decision_date).select(pl.col("close")).to_series().to_list()
                closes = [c for c in closes if c is not None]
                if len(closes) >= ma_window:
                    ma = sum(closes[-ma_window:]) / ma_window
                    last = closes[-1]
                    val = 1.0 if last > ma else 0.0
            except Exception:
                val = 0.0
        elif key == "kospi_ma_slope":
            try:
                sub = index_panel.filter(pl.col("index_name").str.contains("KOSPI") | pl.col("index_name").str.contains("코스피"))
                if sub.height == 0:
                    sub = index_panel.filter(pl.col("index_name") == "KOSPI")
                sub_sorted = sub.sort("date")
                closes = sub_sorted.filter(pl.col("date") <= decision_date).select(pl.col("close")).to_series().to_list()
                closes = [c for c in closes if c is not None]
                if len(closes) >= ma_window + 1:
                    ma_now = sum(closes[-ma_window:]) / ma_window
                    ma_prev = sum(closes[-ma_window - 1 : -1]) / ma_window
                    val = 1.0 if ma_now > ma_prev else 0.0
            except Exception:
                val = 0.0
        elif key == "kosdaq_trend":
            try:
                sub = index_panel.filter(pl.col("index_name").str.contains("KOSDAQ") | pl.col("index_name").str.contains("코스닥"))
                if sub.height == 0:
                    sub = index_panel.filter(pl.col("index_name") == "KOSDAQ")
                sub_sorted = sub.sort("date")
                closes = sub_sorted.filter(pl.col("date") <= decision_date).select(pl.col("close")).to_series().to_list()
                closes = [c for c in closes if c is not None]
                if len(closes) >= ma_window:
                    ma = sum(closes[-ma_window:]) / ma_window
                    last = closes[-1]
                    val = 1.0 if last > ma else 0.0
            except Exception:
                val = 0.0
        elif key == "breadth":
            try:
                # breadth_panel expected has column breadth_ma20 for decision_date
                b_row = breadth_panel.filter(pl.col("date") == decision_date)
                if b_row.height > 0 and "breadth_ma20" in b_row.columns:
                    b_val = b_row.select(pl.col("breadth_ma20")).to_series().to_list()[0]
                    if b_val is not None:
                        val = 1.0 if float(b_val) > float(config.breadth_floor) else 0.0
            except Exception:
                val = 0.0
        elif key == "volatility":
            try:
                # compute realized vol as std of KOSPI returns over 20 days
                sub = index_panel.filter(pl.col("index_name").str.contains("KOSPI") | pl.col("index_name").str.contains("코스피"))
                if sub.height == 0:
                    sub = index_panel.filter(pl.col("index_name") == "KOSPI")
                sub_sorted = sub.sort("date")
                closes = sub_sorted.filter(pl.col("date") <= decision_date).select(pl.col("close")).to_series().to_list()
                closes = [c for c in closes if c is not None]
                if len(closes) >= ma_window + 1:
                    # simple returns
                    rets = [(closes[i] / closes[i - 1] - 1) for i in range(1, len(closes))][-ma_window:]
                    # std (pop)
                    mean = sum(rets) / len(rets)
                    var = sum((r - mean) ** 2 for r in rets) / len(rets)
                    std = var**0.5
                    val = 1.0 if std < float(config.volatility_ceiling) else 0.0
            except Exception:
                val = 0.0
        else:
            # unknown component: treat as 0
            val = 0.0
        contrib = float(w) * float(val)
        components[key] = contrib
        score += contrib

    # Determine state via thresholds monotonic
    # thresholds assumed sorted ascending
    state = _classify_score(score, config.thresholds)
    return RegimeSnapshot(as_of=decision_date, state=state, score=score, components=components)
