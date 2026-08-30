# ruff: noqa
from datetime import date

import polars as pl

from src.core.trace import CandidateTrace, InMemoryTraceSink
from src.reporting.exposure_metrics import summarise_realised_exposure
from src.reporting.timeseries import build_window_timeseries
from src.reporting.trace_store import frames_from_sink
from src.tournament.simulator import RollingResult
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
    new_b = InstrumentAttributes(ticker=b.ticker, name=b.name, issuer=b.issuer, leverage_multiple=2, leverage_family_key=a.leverage_family_key, is_synthetic=b.is_synthetic, is_hedged=b.is_hedged, is_active=b.is_active, index_key=b.index_key, theme=b.theme, first_seen=b.first_seen, last_seen=b.last_seen, left_censored=b.left_censored, confidence=b.confidence)
    new_a = InstrumentAttributes(ticker=a.ticker, name=a.name, issuer=a.issuer, leverage_multiple=1, leverage_family_key=a.leverage_family_key, is_synthetic=a.is_synthetic, is_hedged=a.is_hedged, is_active=a.is_active, index_key=a.index_key, theme=a.theme, first_seen=a.first_seen, last_seen=a.last_seen, left_censored=a.left_censored, confidence=a.confidence)
    return InstrumentMaster(attributes={"T1": new_a, "T2": new_b}, panel_start=master.panel_start)


def test_candidate_trace_preserves_source_vehicle_lineage() -> None:
    master = _master_with_family()
    sink = InMemoryTraceSink()
    # Simulate a remapped source T1 to vehicle T2
    cand = CandidateTrace(
        decision_date=date(2026, 1, 2),
        ticker="T1",
        score=1.0,
        rank=1,
        selected=True,
        reject_reason="",
        weight_raw=1.0,
        weight_target=1.0,
        weight_after_adv=1.0,
        weight_fill=1.0,
        source_ticker="T1",
        vehicle_ticker="T2",
        family_key=master.attributes["T1"].leverage_family_key,
        multiple=2,
        route_reason="CAPACITY_OK",
        lottery_active=True,
        weight_intended=1.0,
        weight_after_capacity=1.0,
        weight_filled=1.0,
    )
    sink.emit_candidates([cand])
    # need at least one session
    from src.core.trace import SessionTrace

    sink.emit_session(SessionTrace(decision_date=date(2026, 1, 2), n_universe=2, n_scores=1, n_selected=1, n_fills=1, n_unfilled=0, n_candidates_written=1, n_candidates_truncated=0, regime="RISK_ON", equity=1_000_000.0))
    sessions, candidates, gates = frames_from_sink(sink)
    assert candidates.height == 1
    row = candidates.row(0, named=True)
    assert row["source_ticker"] == "T1"
    assert row["vehicle_ticker"] == "T2"
    assert row["family_key"] == master.attributes["T1"].leverage_family_key
    assert row["multiple"] == 2
    assert row["route_reason"] == "CAPACITY_OK"
    assert row["weight_intended"] == 1.0
    assert row["weight_after_capacity"] == 1.0
    assert row["weight_filled"] == 1.0
    # no false TOPK_CUT label
    assert row["reject_reason"] != "TOPK_CUT"
    # ensure selected
    assert row["selected"] is True


def test_realised_exposure_metrics_ignore_zero_weights() -> None:
    master = _master_with_family()
    dates = [date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 4)]
    # trades: holdings for T1 weight 0.5 on first date, 0.0 residual on second (should be ignored)
    trades = pl.DataFrame(
        [
            {"decision_date": dates[0], "execution_date": dates[1], "ticker": "T1", "weight_after": 0.5, "weight_before": 0.0, "delta_weight": 0.5},
            {"decision_date": dates[1], "execution_date": dates[2], "ticker": "T1", "weight_after": 1e-10, "weight_before": 0.5, "delta_weight": -0.5},
            {"decision_date": dates[1], "execution_date": dates[2], "ticker": "T2", "weight_after": 0.0, "weight_before": 0.0, "delta_weight": 0.0},
        ]
    )
    try:
        trades = trades.with_columns(pl.col("decision_date").cast(pl.Date), pl.col("execution_date").cast(pl.Date))
    except Exception:
        pass
    summary = summarise_realised_exposure(dates, trades, [], master, epsilon=1e-9)
    # Zero and <=1e-9 residual keys do not increase active-name/family counts
    # On dates[2], active should be 0 because 1e-10 filtered
    assert summary.active_name_mean < 1.0  # includes zero session
    # Reconstructed effective gross: T1 mult 1, weight 0.5 => gross 0.5 ; other dates 0
    # mean = (0 +0.5+0)/3 =0.166...
    assert abs(summary.effective_gross_mean - (0.5 / 3)) < 1e-9
    # turnover: first hold 0->0.5 =>0.5, second 0.5->0 =>0.5, total 1.0
    assert abs(summary.turnover - 1.0) < 1e-9
    # active family mean should be <= active name mean
    assert summary.active_family_mean <= summary.active_name_mean + 1e-9


def test_window_artifact_aligns_rolling_outcomes() -> None:
    dates = [date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 5), date(2026, 1, 6)]
    horizon = 2
    rolling = RollingResult(name="P15", horizon=horizon, starts=(dates[0], dates[1], dates[2]), returns=(0.35, -0.30, 0.45), drawdowns=(0.1, 0.2, 0.05), givebacks=(0.02, 0.03, 0.01))
    df = build_window_timeseries(rolling, dates, ruin_threshold=-0.25)
    assert df.height == 3
    # One row per RollingResult.start ; terminal_return matches tuple
    for i, s in enumerate(rolling.starts):
        row = df.filter(pl.col("window_start") == s).row(0, named=True)
        assert abs(row["terminal_return"] - rolling.returns[i]) < 1e-12
        assert abs(row["max_drawdown"] - rolling.drawdowns[i]) < 1e-12
        assert abs(row["giveback"] - rolling.givebacks[i]) < 1e-12
        assert row["gt_30"] == (rolling.returns[i] > 0.30)
        assert row["gt_40"] == (rolling.returns[i] > 0.40)
        assert row["gt_50"] == (rolling.returns[i] > 0.50)
        assert row["ruin"] == (rolling.returns[i] < -0.25)
        # calendar-derived end date
        s_idx = dates.index(s)
        expected_end = dates[s_idx + horizon - 1]
        assert row["window_end"] == expected_end
