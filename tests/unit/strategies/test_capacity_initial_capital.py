from __future__ import annotations


def test_resolve_capacity_capital_uses_initial_not_mark_to_market() -> None:
    from types import SimpleNamespace

    from src.strategies.sticky.capacity import TOURNAMENT_INITIAL_CAPITAL_DEFAULT, resolve_capacity_capital

    ctx = SimpleNamespace(
        capital=1.5e9,
        rules=SimpleNamespace(initial_capital=1_000_000_000),
    )
    got = resolve_capacity_capital(ctx)
    assert got == 1_000_000_000.0
    assert got == TOURNAMENT_INITIAL_CAPITAL_DEFAULT
    assert got != float(ctx.capital)


def test_resolve_capacity_capital_fail_closed_missing_or_invalid_initial() -> None:
    from types import SimpleNamespace

    from src.strategies.sticky.capacity import TOURNAMENT_INITIAL_CAPITAL_DEFAULT, resolve_capacity_capital

    assert resolve_capacity_capital(SimpleNamespace(capital=2.0e9, rules=None)) == TOURNAMENT_INITIAL_CAPITAL_DEFAULT
    assert resolve_capacity_capital(SimpleNamespace(capital=2.0e9)) == TOURNAMENT_INITIAL_CAPITAL_DEFAULT
    assert resolve_capacity_capital(SimpleNamespace(capital=2.0e9, rules=SimpleNamespace())) == TOURNAMENT_INITIAL_CAPITAL_DEFAULT
    assert resolve_capacity_capital(SimpleNamespace(capital=2.0e9, rules=SimpleNamespace(initial_capital=0))) == TOURNAMENT_INITIAL_CAPITAL_DEFAULT
    assert resolve_capacity_capital(SimpleNamespace(capital=2.0e9, rules=SimpleNamespace(initial_capital=-1))) == TOURNAMENT_INITIAL_CAPITAL_DEFAULT
    assert resolve_capacity_capital(SimpleNamespace(capital=2.0e9, rules=SimpleNamespace(initial_capital=float("nan")))) == TOURNAMENT_INITIAL_CAPITAL_DEFAULT
    assert resolve_capacity_capital(SimpleNamespace(capital=2.0e9, rules=SimpleNamespace(initial_capital=None))) == TOURNAMENT_INITIAL_CAPITAL_DEFAULT
    assert resolve_capacity_capital(SimpleNamespace(capital=2.0e9, rules=SimpleNamespace(initial_capital="x"))) == TOURNAMENT_INITIAL_CAPITAL_DEFAULT
    assert resolve_capacity_capital(None) == TOURNAMENT_INITIAL_CAPITAL_DEFAULT
    assert resolve_capacity_capital(SimpleNamespace(rules=SimpleNamespace(initial_capital=1.0)), default=float("nan")) == 1.0
    assert resolve_capacity_capital(SimpleNamespace(rules=None), default=-5.0) == TOURNAMENT_INITIAL_CAPITAL_DEFAULT


def test_resolve_capacity_capital_source_never_reads_context_capital() -> None:
    import inspect

    from src.strategies.sticky.capacity import resolve_capacity_capital

    src = inspect.getsource(resolve_capacity_capital)
    assert "initial_capital" in src
    assert 'getattr(context, "capital"' not in src
    assert "context.capital" not in src
    assert "except Exception" not in src


def test_apply_capacity_filter_hard_cut_unchanged_at_fixed_capital() -> None:
    import polars as pl

    from src.strategies.sticky.capacity import apply_capacity_filter

    snap = pl.DataFrame(
        {
            "ticker": ["ILLIQ", "LIQ"],
            "name": ["TIGER 차이나전기차레버리지(합성)", "KODEX 반도체레버리지"],
            "trading_value": [4.56e8, 4.56e10],
            "mom_60": [0.70, 0.53],
        }
    )
    scores = {"ILLIQ": 0.70, "LIQ": 0.53}
    out = apply_capacity_filter(
        scores,
        snap,
        capital=1_000_000_000.0,
        max_order_to_adv=0.01,
        min_fill_ratio=0.25,
        sleeve_weight=0.95,
    )
    assert "ILLIQ" not in out
    assert out.get("LIQ") == 0.53
