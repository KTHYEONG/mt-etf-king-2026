from __future__ import annotations


from datetime import date

import polars as pl


def test_cached_filtered_scores_keys_by_explicit_key() -> None:
    from src.strategies.sticky.capacity import cached_filtered_scores

    cache: dict[object, dict[str, float]] = {}
    calls: list[int] = []

    def scorer(frame: pl.DataFrame) -> dict[str, float]:
        calls.append(1)
        return {"A": 1.0}

    snapshot_1 = pl.DataFrame({"ticker": ["A"]})
    snapshot_2 = pl.DataFrame({"ticker": ["A"]})  # distinct object, same content
    key = date(2020, 1, 2)

    result_1 = cached_filtered_scores(cache, key, snapshot_1, scorer)
    result_2 = cached_filtered_scores(cache, key, snapshot_2, scorer)

    assert result_1 == {"A": 1.0}
    assert result_2 == {"A": 1.0}
    assert len(calls) == 1, "second call with the same key must be a cache hit, not a recompute"


def test_cached_filtered_scores_different_keys_never_collide_on_reused_address() -> None:
    """Deterministic regression guard for the id(snapshot)-keyed bug.

    A del()+gc.collect() cycle does not reliably force CPython to reuse a freed
    polars.DataFrame's address (verified empirically: 15/15 runs passed against the
    deliberately-reverted id()-based implementation), so it cannot prove discriminating
    power. Instead, this seeds the cache with the exact artifact the old buggy code would
    have left behind -- a stale entry keyed by this snapshot's CURRENT id() holding wrong
    data from a hypothetical earlier, unrelated session -- and asserts the fix ignores it
    unconditionally because it looks up by the caller-supplied key, never by id().
    """
    from src.strategies.sticky.capacity import cached_filtered_scores

    cache: dict[object, dict[str, float]] = {}
    calls: list[tuple[str, ...]] = []

    def scorer(frame: pl.DataFrame) -> dict[str, float]:
        tickers = tuple(frame["ticker"].to_list())
        calls.append(tickers)
        return {t: float(i) for i, t in enumerate(tickers, start=1)}

    snapshot = pl.DataFrame({"ticker": ["BBB"]})
    # Simulate a stale leftover the OLD `key = id(snapshot)` code would have produced: an
    # entry keyed by THIS snapshot's current address, holding wrong data from a different
    # (already garbage-collected) session that happened to reuse the same address.
    cache[id(snapshot)] = {"WRONG": 999.0}

    result = cached_filtered_scores(cache, date(2020, 1, 3), snapshot, scorer)

    assert result == {"BBB": 1.0}, "a real key lookup must never return a stale id()-keyed entry"
    assert calls == [("BBB",)], "scorer must run for the real key, not be skipped by a stale id() hit"
    assert cache[date(2020, 1, 3)] == {"BBB": 1.0}
    assert cache[id(snapshot)] == {"WRONG": 999.0}, "the stale int-keyed entry is left untouched, simply unreachable"


def test_cached_filtered_scores_rejects_none_key() -> None:
    import pytest

    from src.strategies.sticky.capacity import cached_filtered_scores

    cache: dict[object, dict[str, float]] = {}

    def scorer(frame: pl.DataFrame) -> dict[str, float]:
        return dict()  # noqa: C408

    with pytest.raises(ValueError, match="key"):
        cached_filtered_scores(cache, None, pl.DataFrame({"ticker": ["A"]}), scorer)


from src.alpha.base import DecisionContext
from src.strategies.sticky.model_runner import StickyLeaderModel


def _plus2_snapshot() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "name": ["KODEX 200", "KODEX 레버리지", "KODEX 인버스2X"],
            "mom_20": [0.10, 0.18, 0.25],
        }
    )


def test_sticky_model_score_passes_decision_date_as_cache_key() -> None:
    model = StickyLeaderModel(name="sticky.leader_base")
    ctx_day1 = DecisionContext(decision_date=date(2020, 1, 2), regime=None, capital=1_000_000_000.0, held={}, rules=None, championship_sleeve="LOTTERY_ON")  # type: ignore[arg-type]
    ctx_day2 = DecisionContext(decision_date=date(2020, 1, 3), regime=None, capital=1_000_000_000.0, held={}, rules=None, championship_sleeve="LOTTERY_ON")  # type: ignore[arg-type]

    scores_day1 = model.score(_plus2_snapshot(), ctx_day1)
    assert scores_day1 == {"B": 0.18}
    assert date(2020, 1, 2) in model._filtered_scores_by_snapshot

    # a distinct snapshot object for a distinct date must be scored independently
    scores_day2 = model.score(_plus2_snapshot(), ctx_day2)
    assert scores_day2 == {"B": 0.18}
    assert date(2020, 1, 3) in model._filtered_scores_by_snapshot
    assert len(model._filtered_scores_by_snapshot) == 2, "each decision_date must get its own cache entry"


def test_sticky_model_score_raises_on_missing_decision_date() -> None:
    import pytest

    model = StickyLeaderModel(name="sticky.leader_base")
    ctx_missing = DecisionContext(decision_date=None, regime=None, capital=1_000_000_000.0, held={}, rules=None)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="key"):
        model.score(_plus2_snapshot(), ctx_missing)


def test_reset_trackers_clears_filtered_scores_cache() -> None:
    model = StickyLeaderModel(name="sticky.leader_base")
    snap = pl.DataFrame({"ticker": ["A", "B"], "name": ["KODEX 200", "KODEX 레버리지"], "mom_20": [0.1, 0.2]})
    ctx = DecisionContext(decision_date=date(2020, 1, 2), regime=None, capital=1_000_000_000.0, held={}, rules=None)  # type: ignore[arg-type]

    model.score(snap, ctx)
    assert len(model._filtered_scores_by_snapshot) == 1

    model.reset_trackers()

    assert model._filtered_scores_by_snapshot == {}
    assert model._held is None
    assert model._hold_len == 0


def test_resolve_capacity_params_returns_none_when_filter_disabled() -> None:
    from types import SimpleNamespace

    from src.strategies.sticky.capacity import resolve_capacity_params

    # Given: a context whose rules are well-formed, so only min_fill_ratio decides
    ctx = SimpleNamespace(rules=SimpleNamespace(initial_capital=1_000_000_000.0, max_order_to_adv=0.01))

    # When / Then: every disabled shape resolves to None, never to a substituted tuple
    assert resolve_capacity_params(ctx, SimpleNamespace()) is None
    assert resolve_capacity_params(ctx, SimpleNamespace(min_fill_ratio=0.0)) is None
    assert resolve_capacity_params(ctx, SimpleNamespace(min_fill_ratio=-0.25)) is None
    assert resolve_capacity_params(ctx, SimpleNamespace(min_fill_ratio=float("nan"))) is None
    assert resolve_capacity_params(ctx, SimpleNamespace(min_fill_ratio=float("inf"))) is None
    assert resolve_capacity_params(ctx, SimpleNamespace(min_fill_ratio="x")) is None
    assert resolve_capacity_params(ctx, SimpleNamespace(min_fill_ratio=None)) is None


def test_resolve_capacity_params_resolves_capital_and_participation() -> None:
    from types import SimpleNamespace

    from src.strategies.sticky.capacity import TOURNAMENT_INITIAL_CAPITAL_DEFAULT, resolve_capacity_params

    cfg = SimpleNamespace(min_fill_ratio=0.25)

    # Given: fully specified rules -> values are taken verbatim
    ctx = SimpleNamespace(rules=SimpleNamespace(initial_capital=2_000_000_000.0, max_order_to_adv=0.02))
    params = resolve_capacity_params(ctx, cfg)
    assert params is not None
    capital, phi, mfr = params
    assert capital == 2_000_000_000.0
    assert phi == 0.02
    assert mfr == 0.25

    # When: participation is missing / non-finite / non-positive -> fixed 0.01 fallback
    for bad in (None, 0.0, -0.5, float("nan"), float("inf"), "x"):
        ctx_bad = SimpleNamespace(rules=SimpleNamespace(initial_capital=2_000_000_000.0, max_order_to_adv=bad))
        got = resolve_capacity_params(ctx_bad, cfg)
        assert got is not None
        assert got[1] == 0.01, bad

    # And: capital falls back to the tournament default when rules cannot supply one
    ctx_nocap = SimpleNamespace(rules=SimpleNamespace(max_order_to_adv=0.01))
    got_nocap = resolve_capacity_params(ctx_nocap, cfg)
    assert got_nocap is not None
    assert got_nocap[0] == TOURNAMENT_INITIAL_CAPITAL_DEFAULT
