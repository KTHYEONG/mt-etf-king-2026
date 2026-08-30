# ruff: noqa
from datetime import date

import polars as pl

from src.reporting.exposure_metrics import RealisedExposureSummary, summarise_realised_exposure
from src.universe.instruments import InstrumentAttributes, InstrumentMaster
from src.universe.taxonomy import Taxonomy


def _master_with_family() -> InstrumentMaster:
    panel = pl.DataFrame(
        [
            {"date": date(2026, 1, 2), "ticker": "T1", "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 2), "ticker": "T2", "name": "KODEX 레버리지", "underlying_index_name": "KOSPI 200"},
        ]
    )
    taxonomy = Taxonomy(rules=[])
    master = InstrumentMaster.build(panel, taxonomy, {})
    attrs = dict(master.attributes)
    a = attrs["T1"]
    b = attrs["T2"]
    new_b = InstrumentAttributes(
        ticker=b.ticker,
        name=b.name,
        issuer=b.issuer,
        leverage_multiple=2,
        leverage_family_key=a.leverage_family_key,
        is_synthetic=b.is_synthetic,
        is_hedged=b.is_hedged,
        is_active=b.is_active,
        index_key=b.index_key,
        theme=b.theme,
        first_seen=b.first_seen,
        last_seen=b.last_seen,
        left_censored=b.left_censored,
        confidence=b.confidence,
    )
    new_a = InstrumentAttributes(
        ticker=a.ticker,
        name=a.name,
        issuer=a.issuer,
        leverage_multiple=1,
        leverage_family_key=a.leverage_family_key,
        is_synthetic=a.is_synthetic,
        is_hedged=a.is_hedged,
        is_active=a.is_active,
        index_key=a.index_key,
        theme=a.theme,
        first_seen=a.first_seen,
        last_seen=a.last_seen,
        left_censored=a.left_censored,
        confidence=a.confidence,
    )
    return InstrumentMaster(attributes={"T1": new_a, "T2": new_b}, panel_start=master.panel_start)


def test_realised_exposure_empty_dates_returns_zeros() -> None:
    master = _master_with_family()
    summary = summarise_realised_exposure([], pl.DataFrame(), [], master)
    assert isinstance(summary, RealisedExposureSummary)
    assert summary.active_name_mean == 0.0
    assert summary.turnover == 0.0


def test_realised_exposure_metrics_ignore_zero_weights() -> None:
    master = _master_with_family()
    dates = [date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 4)]
    trades = pl.DataFrame(
        [
            {"decision_date": dates[0], "execution_date": dates[1], "ticker": "T1", "weight_after": 0.5, "weight_before": 0.0, "delta_weight": 0.5},
            {"decision_date": dates[1], "execution_date": dates[2], "ticker": "T1", "weight_after": 1e-10, "weight_before": 0.5, "delta_weight": -0.5},
            {"decision_date": dates[1], "execution_date": dates[2], "ticker": "T2", "weight_after": 0.0, "weight_before": 0.0, "delta_weight": 0.0},
        ]
    )
    trades = trades.with_columns(pl.col("decision_date").cast(pl.Date), pl.col("execution_date").cast(pl.Date))
    summary = summarise_realised_exposure(dates, trades, [], master, epsilon=1e-9)
    assert summary.active_name_mean < 1.0
    assert abs(summary.effective_gross_mean - (0.5 / 3)) < 1e-9
    assert abs(summary.turnover - 1.0) < 1e-9
    assert summary.active_family_mean <= summary.active_name_mean + 1e-9
