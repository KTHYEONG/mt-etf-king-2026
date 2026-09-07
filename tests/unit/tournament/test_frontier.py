# ruff: noqa

def test_frontier_runs_every_flat_window_and_preserves_cash() -> None:
    from datetime import date
    import polars as pl
    from src.core.calendar import get_calendar
    from src.tournament.frontier import run_frontier_research
    sessions=get_calendar().sessions(date(2024,1,2),date(2024,6,28))
    panel=pl.DataFrame({"date":sessions,"ticker":["069500"]*len(sessions),"name":["KODEX 200"]*len(sessions),"underlying_index_name":["KOSPI 200"]*len(sessions),"open":[100.0]*len(sessions),"close":[100.0]*len(sessions),"trading_value":[1e12]*len(sessions),"is_tradable":[True]*len(sessions)})
    result=run_frontier_research(panel,sessions,capital=1e9,participation=0.01,cost_bps=8.0)
    assert result.height == len(sessions)-35
    assert result["window_start"].to_list() == sessions[:-35]
    assert result["candidate_return"].to_list() == [0.0]*result.height
    assert result["baseline_return"].to_list() == [0.0]*result.height
    assert result["candidate_cash_min"].min() == 1e9
    assert result["candidate_missing_mark_count"].sum() == 0

def test_frontier_uptrend_fills_next_open_and_is_deterministic() -> None:
    from datetime import date
    import polars as pl
    from src.core.calendar import get_calendar
    from src.tournament.frontier import run_frontier_research
    sessions=get_calendar().sessions(date(2024,1,2),date(2024,6,28))
    panel=pl.DataFrame({"date":sessions,"ticker":["069500"]*len(sessions),"name":["KODEX 200"]*len(sessions),"underlying_index_name":["KOSPI 200"]*len(sessions),"open":[100.0]*len(sessions),"close":[100.0]*len(sessions),"trading_value":[1e12]*len(sessions),"is_tradable":[True]*len(sessions)})
    prices=[100.0*(1.01**i) for i in range(len(sessions))]
    panel=panel.with_columns(pl.Series("open",prices),pl.Series("close",prices))
    first=run_frontier_research(panel,sessions,capital=1e9,participation=0.01,cost_bps=8.0)
    second=run_frontier_research(panel,sessions,capital=1e9,participation=0.01,cost_bps=8.0)
    assert first.equals(second)
    assert first["candidate_return"][-1] > 0.0
    assert first["candidate_cash_min"].min() >= 0.0
    assert first["candidate_execution_violation_count"].sum() == 0
    assert first["candidate_first_fill"][0] is None

def test_frontier_prefix_invariance_and_rejects_misaligned_grid() -> None:
    from datetime import date
    import polars as pl
    from src.core.calendar import get_calendar
    from src.tournament.frontier import run_frontier_research
    sessions=get_calendar().sessions(date(2024,1,2),date(2024,6,28))
    panel=pl.DataFrame({"date":sessions,"ticker":["069500"]*len(sessions),"name":["KODEX 200"]*len(sessions),"underlying_index_name":["KOSPI 200"]*len(sessions),"open":[100.0]*len(sessions),"close":[100.0]*len(sessions),"trading_value":[1e12]*len(sessions),"is_tradable":[True]*len(sessions)})
    import pytest
    cut=90
    short=run_frontier_research(panel.filter(pl.col("date")<=sessions[cut-1]),sessions[:cut],capital=1e9,participation=0.01,cost_bps=8.0)
    full=run_frontier_research(panel,sessions,capital=1e9,participation=0.01,cost_bps=8.0)
    assert short.equals(full.head(short.height))
    for frame,grid in ((pl.concat([panel,panel]),sessions),(panel,sessions+[sessions[-1]]),(panel,sessions[:-1]),(panel.head(0),sessions)):
        with pytest.raises(ValueError):
            run_frontier_research(frame,grid,capital=1e9,participation=0.01,cost_bps=8.0)

def test_frontier_one_x_rules_exclude_leveraged_vehicle() -> None:
    from datetime import date
    import polars as pl
    from src.core.calendar import get_calendar
    from src.tournament.frontier import run_frontier_research
    sessions=get_calendar().sessions(date(2024,1,2),date(2024,6,28))
    panel=pl.DataFrame({"date":sessions,"ticker":["069500"]*len(sessions),"name":["KODEX 200"]*len(sessions),"underlying_index_name":["KOSPI 200"]*len(sessions),"open":[100.0]*len(sessions),"close":[100.0]*len(sessions),"trading_value":[1e12]*len(sessions),"is_tradable":[True]*len(sessions)})
    prices=[100.0*(1.01**i) for i in range(len(sessions))]
    panel=panel.with_columns(pl.lit("122630").alias("ticker"),pl.lit("KODEX 레버리지").alias("name"),pl.Series("open",prices),pl.Series("close",prices))
    denied=run_frontier_research(panel,sessions,capital=1e9,participation=0.01,cost_bps=8.0,allow_leverage=False)
    allowed=run_frontier_research(panel,sessions,capital=1e9,participation=0.01,cost_bps=8.0,allow_leverage=True)
    assert denied["candidate_return"].to_list() == [0.0]*denied.height
    assert allowed["candidate_return"][-1] > 0.0
