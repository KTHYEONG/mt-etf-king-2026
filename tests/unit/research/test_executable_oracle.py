from datetime import date

import polars as pl

from src.research.decision_population import WindowPolicy, build_decision_population
from src.research.executable_oracle import build_deployment_oracle_opens, executable_open_to_open_ceiling


def test_build_deployment_oracle_opens_filters_non_sponsor_rows() -> None:
    sessions = [date(2026, 1, d) for d in (2, 3, 5, 6, 7, 8)]
    panel = pl.DataFrame(
        {
            "date": [sessions[0], sessions[1], sessions[2], sessions[3]],
            "ticker": ["A", "A", "B", "B"],
            "name": ["삼성 레버리지", "삼성 레버리지", "타사 ETF", "타사 ETF"],
            "open": [100.0, 101.0, 50.0, 51.0],
            "trading_value": [200_000_000.0, 200_000_000.0, 200_000_000.0, 200_000_000.0],
            "is_tradable": [True, True, True, True],
        }
    )
    opens = build_deployment_oracle_opens(
        panel,
        sessions=sessions,
        sponsor_issuers=frozenset({"삼성자산운용"}),
        brand_map={"삼성": "삼성자산운용"},
        min_adv=0.0,
        min_history_sessions=0,
    )
    sponsor_rows = opens.filter(pl.col("ticker") == "A")
    other_rows = opens.filter(pl.col("ticker") == "B")
    assert sponsor_rows["eligible"].all()
    assert not other_rows["eligible"].any()


def test_executable_open_to_open_ceiling_is_vectorized_group_max() -> None:
    sessions = [date(2026, 1, d) for d in (2, 5, 6, 7, 8)]
    panel = pl.DataFrame({"date": sessions, "ticker": ["X"] * 5})
    pop = build_decision_population(calendar_sessions=sessions, panel=panel, horizon=2, policy=WindowPolicy.EXECUTABLE)
    opens = pl.DataFrame(
        {
            "date": sessions + sessions,
            "ticker": ["A"] * 5 + ["B"] * 5,
            "open": [100.0, 100.0, 110.0, 120.0, 130.0, 100.0, 100.0, 101.0, 102.0, 103.0],
            "eligible": [True] * 10,
        }
    )
    out = executable_open_to_open_ceiling(opens, pop.windows, cost_rate=0.001)
    assert out.height == pop.n_windows
    row0 = out.filter(pl.col("decision_date") == sessions[0]).row(0, named=True)
    assert row0["best_ticker"] == "A"
    assert abs(float(row0["ceil_exec_raw"]) - (120.0 / 100.0 - 1.0 - 0.001)) < 1e-12


from datetime import timedelta
from unittest.mock import patch




def test_oracle_first_seen_matches_per_ticker_min_and_keeps_eligibility() -> None:
    # 12 sessions so the adv_20 rolling_mean(min_samples=5) is defined for both tickers;
    # OLD is listed on all of them, NEW only on the last 6, so NEW can only ever be
    # excluded by the first_seen history gate (its ADV is defined from its 5th row).
    sessions = [date(2026, 1, 5) + timedelta(days=i) for i in range(12)]
    rows = []
    for i, d in enumerate(sessions):
        rows.append({"date": d, "ticker": "OLD", "name": "KODEX 200", "open": 100.0 + i,
                     "is_tradable": True, "trading_value": 5e11})
    for i, d in enumerate(sessions[6:]):
        rows.append({"date": d, "ticker": "NEW", "name": "KODEX 배당", "open": 200.0 + i,
                     "is_tradable": True, "trading_value": 5e11})
    panel = pl.DataFrame(rows)
    brand_map = {"KODEX": "삼성자산운용"}
    sponsor_issuers = frozenset({"삼성자산운용"})

    # Reference: the per-ticker minimum the old loop computed
    expected_first_seen = {
        t: panel.filter(pl.col("ticker") == t)["date"].min()
        for t in panel["ticker"].unique().to_list()
    }
    assert expected_first_seen["OLD"] == sessions[0]
    assert expected_first_seen["NEW"] == sessions[6]

    # When: min_history_sessions=8 admits OLD (first_seen index 0) but never NEW (first_seen index 6)
    original_filter = pl.DataFrame.filter
    with patch.object(pl.DataFrame, "filter", autospec=True, side_effect=original_filter) as spy:
        frame = build_deployment_oracle_opens(
            panel, sessions=sessions, sponsor_issuers=sponsor_issuers,
            brand_map=brand_map, min_adv=1.0, min_history_sessions=8,
        )

    # Then: the first_seen-driven eligibility gate is unchanged
    assert set(frame.columns) == {"date", "ticker", "open", "eligible"}
    eligible = frame.filter(pl.col("eligible")).select(["date", "ticker"]).to_dicts()
    assert sorted({r["ticker"] for r in eligible}) == ["OLD"]
    assert sorted(r["date"] for r in eligible if r["ticker"] == "OLD") == sessions[8:]
    # And: no per-ticker filtering happened inside the builder
    assert spy.call_count == 0

