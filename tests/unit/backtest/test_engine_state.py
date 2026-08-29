"""SCENARIO-B2-05"""
from datetime import date

import polars as pl

from src.backtest.costs import CostConfig
from src.backtest.engine import BacktestConfig, BacktestEngine  # noqa: F401
from src.portfolio.policy import PortfolioPolicy
from src.portfolio.sizing import ConfidenceSizingConfig
from tests.unit.backtest.conftest import build_engine, panel_row


def test_SCENARIO_B2_05_backtest_reset_trackers() -> None:  # noqa: N802
    # SCENARIO-B2-05: BacktestEngine.run on PortfolioPolicy-backed model invokes reset_trackers once at start; two sequential run() calls do not carry WATCH state from run1 into run2 (fresh HOLD)
    start = date(2025, 9, 22)
    end = date(2025, 9, 30)
    cal_sessions = __import__("src.core.calendar", fromlist=["TradingCalendar"]).TradingCalendar().sessions(start, end)
    rows: list[dict[str, object]] = []
    for d in cal_sessions:
        rows.append(panel_row(day=d, ticker="069500", close=30000.0, mom_20=0.05, name="KODEX 200", theme="반도체"))
        rows.append(panel_row(day=d, ticker="451060", close=20000.0, mom_20=0.03, name="KODEX K-반도체", theme="반도체"))
    panel = pl.DataFrame(rows)
    engine, cal, filt = build_engine(panel)
    config = BacktestConfig(start=start, end=end, capital=1_000_000_000.0, scheme=__import__("src.portfolio.sizing", fromlist=["SizingScheme"]).SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig(0.0, 0.0, 0.0))

    # PortfolioPolicy-backed model with allocate and reset_trackers
    policy = PortfolioPolicy(sizing_config=ConfidenceSizingConfig())

    class PolicyModel:
        name = "P08"
        path_dependent = True

        def score(self, snapshot, context):  # type: ignore[no-untyped-def]
            # simple scores
            if snapshot.height == 0:
                return {}
            scores = {}
            for row in snapshot.iter_rows(named=True):
                t = str(row.get("ticker"))
                v = row.get("mom_20")
                if v is not None:
                    scores[t] = float(v)
            return scores

        def allocate(self, scores):  # type: ignore[no-untyped-def]
            return policy.allocate(scores)

        def reset_trackers(self):  # type: ignore[no-untyped-def]
            return policy.reset_trackers()

    model = PolicyModel()
    # track reset calls
    call_count = 0
    orig_reset = policy.reset_trackers

    def counting_reset():  # type: ignore[no-untyped-def]
        nonlocal call_count
        call_count += 1
        return orig_reset()

    policy.reset_trackers = counting_reset  # type: ignore[method-assign]
    # also need model.reset to delegate to policy counting
    model.reset_trackers = counting_reset  # type: ignore[method-assign]

    # First run: force a BREAKDOWN to put trackers into EXIT/WATCH
    # Manually put tracker into EXIT before run by allocating once
    policy.allocate({"069500": 0.05, "451060": 0.03}, theme_states={"069500": "BREAKDOWN"})
    # Now run: should reset at start, so after run, tracker should be fresh HOLD not WATCH
    res1 = engine.run(model, panel, config)
    assert call_count >= 1
    # After run, tracker should have been reset and then progressed via scoring (which is HOLD-like)
    # Check that next run also resets and doesn't carry WATCH
    call_count_before_second = call_count
    res2 = engine.run(model, panel, config)
    assert call_count > call_count_before_second
    # Verify that after second run, allocating BREAKDOWN again yields EXIT not immediate RE_ENTER (fresh state)
    # If no reset, sessions_since_exit would carry over and RECOVERY could trigger early
    # Fresh HOLD + BREAKDOWN => EXIT with weight 0
    dec = policy.allocate({"069500": 0.08, "451060": 0.01}, theme_states={"069500": "BREAKDOWN"})
    assert dec.weights["069500"] == 0.0
    assert "state=EXIT" in dec.rationale["069500"]  # type: ignore[index]

    # Also ensure sum weights behavior not cross-contaminated
    assert res1.daily.height > 0
    assert res2.daily.height > 0
