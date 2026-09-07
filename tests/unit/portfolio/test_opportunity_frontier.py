# ruff: noqa

def test_capacity_frontier_remaining_horizon_changes_ranking() -> None:
    import polars as pl
    from src.portfolio.opportunity_frontier import capacity_frontier_scores
    frame = pl.DataFrame({"ticker":["SLOW","FAST"],"mom_60":[0.6,0.3],"adv20":[400.0,1000.0],"leverage_multiple":[1,2],"confidence":["HIGH","HIGH"],"eligible":[True,True]})
    short=capacity_frontier_scores(frame,capital=100.0,remaining_sessions=1,participation=0.01,cost_bps=0.0)
    long=capacity_frontier_scores(frame,capital=100.0,remaining_sessions=36,participation=0.01,cost_bps=0.0)
    assert short["FAST"] > short["SLOW"]
    assert long["SLOW"] > long["FAST"]
    assert capacity_frontier_scores(frame,capital=100.0,remaining_sessions=0,participation=0.01,cost_bps=0.0) == {}

def test_capacity_frontier_finite_rows_confidence_and_costs() -> None:
    import polars as pl
    from src.portfolio.opportunity_frontier import capacity_frontier_scores
    frame = pl.DataFrame({"ticker":["SLOW","FAST"],"mom_60":[0.6,0.3],"adv20":[400.0,1000.0],"leverage_multiple":[1,2],"confidence":["HIGH","HIGH"],"eligible":[True,True]})
    invalid=frame.with_columns(pl.Series("confidence",["LOW","LOW"]),pl.Series("leverage_multiple",[1,2]))
    result=capacity_frontier_scores(invalid,capital=100.0,remaining_sessions=36,participation=0.01,cost_bps=0.0)
    assert set(result) == {"SLOW"}
    blocked=frame.with_columns(pl.lit(False).alias("eligible"))
    assert capacity_frontier_scores(blocked,capital=100.0,remaining_sessions=36,participation=0.01,cost_bps=0.0) == {}
    assert capacity_frontier_scores(frame,capital=100.0,remaining_sessions=1,participation=0.01,cost_bps=10000.0) == {}
    nanframe=frame.with_columns(pl.lit(float("nan")).alias("adv20"))
    assert capacity_frontier_scores(nanframe,capital=100.0,remaining_sessions=36,participation=0.01,cost_bps=0.0) == {}

def test_capacity_frontier_rejects_bad_config_and_duplicate_tickers() -> None:
    import polars as pl
    from src.portfolio.opportunity_frontier import capacity_frontier_scores
    frame = pl.DataFrame({"ticker":["SLOW","FAST"],"mom_60":[0.6,0.3],"adv20":[400.0,1000.0],"leverage_multiple":[1,2],"confidence":["HIGH","HIGH"],"eligible":[True,True]})
    import pytest
    kw=dict(capital=100.0,remaining_sessions=36,participation=0.01,cost_bps=0.0)
    for changes in ({"capital":0.0},{"remaining_sessions":-1},{"remaining_sessions":37},{"participation":float("nan")},{"cost_bps":-1.0}):
        with pytest.raises(ValueError):
            capacity_frontier_scores(frame,**(kw | changes))
    with pytest.raises(ValueError):
        capacity_frontier_scores(pl.concat([frame,frame]),**kw)


def test_capacity_frontier_skips_non_integral_or_unknown_multiple() -> None:
    import polars as pl
    from src.portfolio.opportunity_frontier import capacity_frontier_scores

    frame = pl.DataFrame({"ticker": ["FLOAT", "UNKNOWN"], "mom_60": [0.5, 0.5], "adv20": [1000.0, 1000.0], "leverage_multiple": [1.5, 3], "confidence": ["HIGH", "HIGH"], "eligible": [True, True]})
    assert capacity_frontier_scores(frame, capital=100.0, remaining_sessions=10, participation=0.01, cost_bps=0.0) == {}
