from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from src.features.builder import FeatureBuilder, FeatureConfig
from src.features.regime import RegimeConfig, RegimeState, classify_regime
from src.core.calendar import get_calendar
from tests.unit.features.conftest import session_dates


def _config() -> RegimeConfig:
    return RegimeConfig(
        weights={
            "kospi_trend": 0.25,
            "kospi_ma_slope": 0.20,
            "kosdaq_trend": 0.20,
            "breadth": 0.20,
            "volatility": 0.15,
        },
        thresholds=(0.25, 0.45, 0.65, 0.85),
        breadth_floor=0.5,
        volatility_ceiling=0.025,
    )


def _index_panel(sessions: list[date], kospi_closes: list[float], kosdaq_closes: list[float]) -> pl.DataFrame:
    kospi = pl.DataFrame({"date": sessions, "index_name": ["KOSPI"] * len(sessions), "close": kospi_closes})
    kosdaq = pl.DataFrame({"date": sessions, "index_name": ["KOSDAQ"] * len(sessions), "close": kosdaq_closes})
    return pl.concat([kospi, kosdaq])


def _risk_rank(state: RegimeState) -> int:
    order = [
        RegimeState.STRONG_RISK_OFF,
        RegimeState.RISK_OFF,
        RegimeState.NEUTRAL,
        RegimeState.RISK_ON,
        RegimeState.STRONG_RISK_ON,
    ]
    return order.index(state)


def test_scenario_05_07_classify_regime_monotone() -> None:
    """SCENARIO-05-07"""
    sessions = session_dates(date(2026, 1, 2), 30)
    decision = sessions[-1]
    config = _config()

    flat = [100.0] * len(sessions)
    off_index = _index_panel(sessions, flat, flat)
    off_breadth = pl.DataFrame({"date": [decision], "breadth_ma20": [0.2]})
    off = classify_regime(off_index, off_breadth, decision, config)
    assert off.state == RegimeState.STRONG_RISK_OFF
    assert off.score == pytest.approx(sum(off.components.values()))

    rising = [80.0 + i * 2.0 for i in range(len(sessions))]
    on_index = _index_panel(sessions, rising, rising)
    on_breadth = pl.DataFrame({"date": [decision], "breadth_ma20": [0.9]})
    on = classify_regime(on_index, on_breadth, decision, config)
    assert on.state == RegimeState.STRONG_RISK_ON
    assert on.score == pytest.approx(sum(on.components.values()))

    variants = {
        "kospi_trend": _index_panel(sessions, rising, flat),
        "kospi_ma_slope": _index_panel(sessions, rising, flat),
        "kosdaq_trend": _index_panel(sessions, flat, rising),
        "breadth": off_index,
        "volatility": _index_panel(sessions, flat, flat),
    }
    breadth_high = pl.DataFrame({"date": [decision], "breadth_ma20": [0.9]})
    breadth_low = off_breadth

    base = classify_regime(off_index, off_breadth, decision, config)
    for key in config.weights:
        index_panel = variants.get(key, off_index)
        breadth_panel = breadth_high if key == "breadth" else breadth_low
        flipped = classify_regime(index_panel, breadth_panel, decision, config)
        assert flipped.score >= base.score - 1e-12
        assert _risk_rank(flipped.state) >= _risk_rank(base.state)


def test_SCENARIO_07P_03_build_regime_series() -> None:  # noqa: N802
    """SCENARIO-07P-03"""
    sessions = session_dates(date(2026, 1, 2), 30)
    config = _config()
    cal = get_calendar()
    rising = [80.0 + i * 2.0 for i in range(len(sessions))]
    index_panel = _index_panel(sessions, rising, rising)
    breadth_panel = pl.DataFrame({"date": sessions, "breadth_ma20": [0.9] * len(sessions)})
    fconfig = FeatureConfig(momentum_horizons=(20,), ma_windows=(20,), breakout_windows=(20,), volatility_windows=(5,), flow_windows=(5,), regime=config)
    builder = FeatureBuilder(cal, fconfig)
    result = builder.build_regime_series(index_panel, breadth_panel, sessions)
    assert set(result.keys()) == set(sessions)
    assert result[sessions[-1]].state == RegimeState.STRONG_RISK_ON
    # future row ignored
    import datetime as _dt

    extra = pl.DataFrame({"date": [sessions[-1] + _dt.timedelta(days=1)], "index_name": ["KOSPI"], "close": [9999.0]})
    index_plus = pl.concat([index_panel, extra])
    result2 = builder.build_regime_series(index_plus, breadth_panel, sessions)
    assert set(result2.keys()) == set(sessions)
    assert result[sessions[0]].state != RegimeState.STRONG_RISK_ON
