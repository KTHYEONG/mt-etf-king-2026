from __future__ import annotations


def test_cached_filtered_scores_reuses_immutable_snapshot() -> None:
    import polars as pl

    from src.strategies.sticky.capacity import cached_filtered_scores

    snapshot = pl.DataFrame({"ticker": ["A"]})
    calls = 0

    def scorer(frame: pl.DataFrame) -> dict[str, float]:
        nonlocal calls
        calls += 1
        assert frame is snapshot
        return {"A": 1.0}

    cache: dict[int, dict[str, float]] = {}
    assert cached_filtered_scores(cache, snapshot, scorer) == {"A": 1.0}
    assert cached_filtered_scores(cache, snapshot, scorer) == {"A": 1.0}
    assert calls == 1
