# ruff: noqa
def test_resolve_adoption_vehicle_rate_top1_without_allocate() -> None:
    from datetime import date
    import polars as pl
    from src.features.regime import RegimeSnapshot, RegimeState
    from src.tournament.distribution import measure_vehicle_activity_from_top1_scores

    class _M:
        name = "P20"

        def score(self, snapshot: pl.DataFrame, context: object) -> dict[str, float]:
            return {"LEV": 0.2}

    class _Cache:
        dates = (date(2024, 1, 2), date(2024, 1, 3))
        snapshots = {
            date(2024, 1, 2): pl.DataFrame({"ticker": ["LEV"], "name": ["KODEX 레버리지"]}),
            date(2024, 1, 3): pl.DataFrame({"ticker": ["LEV"], "name": ["KODEX 레버리지"]}),
        }
        scores = {}
        rules = None

    regimes = {
        date(2024, 1, 2): RegimeSnapshot(as_of=date(2024, 1, 2), state=RegimeState.RISK_ON, score=0.7, components={}),
        date(2024, 1, 3): RegimeSnapshot(as_of=date(2024, 1, 3), state=RegimeState.RISK_ON, score=0.7, components={}),
    }
    rate = measure_vehicle_activity_from_top1_scores(_M(), _Cache(), regimes, True)
    assert rate == 1.0
