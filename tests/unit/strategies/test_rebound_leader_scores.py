# ruff: noqa
def test_rebound_leader_scores_uses_mom20_plus2_fail_closed() -> None:
    import polars as pl

    from src.strategies.sticky.model_scores import rebound_leader_scores

    snap = pl.DataFrame(
        {
            "ticker": ["FAST", "SLOW", "SYN", "ONE", "INV"],
            "name": [
                "KODEX 반도체레버리지",
                "TIGER 200IT레버리지",
                "TIGER 차이나전기차레버리지(합성)",
                "KODEX 200",
                "KODEX 인버스",
            ],
            "mom_20": [0.10, 0.10, 0.90, 0.80, 0.70],
            "mom_60": [0.01, 0.50, 0.90, 0.80, 0.70],
            "volume_expansion": [2.0, 1.0, 9.0, 9.0, 9.0],
        }
    )
    scores = rebound_leader_scores(snap)
    assert set(scores) == {"FAST", "SLOW"}
    assert scores["FAST"] > scores["SLOW"]
    missing = pl.DataFrame({"ticker": ["FAST"], "name": ["KODEX 반도체레버리지"], "mom_60": [0.50]})
    assert rebound_leader_scores(missing) == {}
    assert rebound_leader_scores({}) == {}
    assert rebound_leader_scores(pl.DataFrame()) == {}
    empty_names = pl.DataFrame(
        {"ticker": [""], "name": [""], "mom_20": [0.10]},
    )
    assert rebound_leader_scores(empty_names) == {}
    nan_row = pl.DataFrame(
        {"ticker": ["NAN"], "name": ["KODEX 반도체레버리지"], "mom_20": [float("nan")]},
    )
    assert rebound_leader_scores(nan_row) == {}
    bad_type = pl.DataFrame(
        {"ticker": ["BAD"], "name": ["KODEX 반도체레버리지"], "mom_20": ["x"]},
        schema={"ticker": pl.Utf8, "name": pl.Utf8, "mom_20": pl.Object},
    )
    assert rebound_leader_scores(bad_type) == {}
    null_row = pl.DataFrame(
        {"ticker": [None], "name": ["KODEX 반도체레버리지"], "mom_20": [0.10]},
        schema={"ticker": pl.Utf8, "name": pl.Utf8, "mom_20": pl.Float64},
    )
    assert rebound_leader_scores(null_row) == {}
    blank_ticker = pl.DataFrame(
        {"ticker": [""], "name": ["KODEX 반도체레버리지"], "mom_20": [0.10]},
    )
    assert rebound_leader_scores(blank_ticker) == {}
