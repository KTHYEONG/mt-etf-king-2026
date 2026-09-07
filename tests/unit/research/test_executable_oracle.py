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
