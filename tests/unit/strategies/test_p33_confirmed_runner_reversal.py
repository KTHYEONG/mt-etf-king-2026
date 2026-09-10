from __future__ import annotations


def test_p33_exits_only_after_confirmed_runner_reversal() -> None:
    from datetime import date, timedelta

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.portfolio.intent import CASH_INTENT
    from src.universe.tournament import TournamentRules

    rules = TournamentRules(name="t", start_date=date(2026, 9, 21), end_date=date(2026, 11, 13), initial_capital=1_000_000_000, category="autonomous", leverage_allowed=True, inverse_allowed=True, max_weight=1.0, cash_allowed=True, sponsor_etf_only=True, manifest_path=None, issuer_whitelist=None, commission_bps=3.0, slippage_bps=5.0, max_order_to_adv=0.01, stress_grid=(0.01,))
    model = BASELINES["sticky.mom60_runner_reversal"]()
    def context(day: int, capital: float) -> DecisionContext:
        return DecisionContext(decision_date=date(2026, 9, 21) + timedelta(days=day), regime=None, capital=capital, held={"RUN": 0.95}, rules=rules, championship_sleeve="LOTTERY_ON")
    strong = pl.DataFrame({"ticker": ["RUN"], "name": ["KODEX 반도체레버리지"], "mom_60": [0.40], "mom_5": [0.02], "volume_expansion": [1.0], "drawdown_20": [0.0], "trading_value": [1.0e12]})
    for day, capital in enumerate((100.0, 102.0, 104.0, 106.0, 108.0)):
        assert getattr(model.score(strong, context(day, capital)), "kind", None) != CASH_INTENT.kind
    reversal = strong.with_columns(pl.lit(-0.01).alias("mom_5"))
    result = model.score(reversal, context(5, 106.0))
    assert getattr(result, "kind", None) == CASH_INTENT.kind


def test_p33_does_not_cash_before_confirmation_or_without_momentum_reversal() -> None:
    from datetime import date, timedelta

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.portfolio.intent import CASH_INTENT
    from src.universe.tournament import TournamentRules

    rules = TournamentRules(name="t", start_date=date(2026, 9, 21), end_date=date(2026, 11, 13), initial_capital=1_000_000_000, category="autonomous", leverage_allowed=True, inverse_allowed=True, max_weight=1.0, cash_allowed=True, sponsor_etf_only=True, manifest_path=None, issuer_whitelist=None, commission_bps=3.0, slippage_bps=5.0, max_order_to_adv=0.01, stress_grid=(0.01,))
    model = BASELINES["sticky.mom60_runner_reversal"]()
    snap = pl.DataFrame({"ticker": ["RUN"], "name": ["KODEX 반도체레버리지"], "mom_60": [0.40], "mom_5": [0.02], "volume_expansion": [1.0], "drawdown_20": [0.0], "trading_value": [1.0e12]})
    def call(day: int, capital: float) -> object:
        return model.score(snap, DecisionContext(decision_date=date(2026, 9, 21) + timedelta(days=day), regime=None, capital=capital, held={"RUN": 0.95}, rules=rules, championship_sleeve="LOTTERY_ON"))
    assert getattr(call(0, 100.0), "kind", None) != CASH_INTENT.kind
    assert getattr(call(1, 105.0), "kind", None) != CASH_INTENT.kind
    assert getattr(call(2, 103.0), "kind", None) != CASH_INTENT.kind
    call(3, 106.0)
    call(4, 108.0)
    assert getattr(call(5, 106.0), "kind", None) != CASH_INTENT.kind


def test_p33_reset_and_invalid_runner_column_fail_closed() -> None:
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.strategies.sticky.model import momentum_horizon

    model = BASELINES["sticky.mom60_runner_reversal"]()
    model._held = "RUN"
    model._hold_len = 5
    model._runner_entry_capital = 100.0
    model._runner_peak_capital = 120.0
    model._runner_held_sessions = 5
    model._runner_armed = True
    model.reset_trackers()
    assert model._held is None
    assert model._runner_entry_capital is None
    assert model._runner_peak_capital is None
    assert model._runner_held_sessions == 0
    assert model._runner_armed is False
    assert momentum_horizon("not_a_momentum_column") == 5
    assert momentum_horizon("mom_0") == 5


def test_p33_registry_cli_and_p27_exposure_wiring() -> None:
    from src.strategies.registry import STRATEGIES as BASELINES
    from src.cli.constants import STICKY_ADOPTION_MODELS
    from src.cli.constants import CHAMPION_STRATEGY
    from src.portfolio.constraints import load_p27_exposure_limits, resolve_exposure_limits_for_model
    from src.strategies.ids import STICKY_MOM60_RAW, STICKY_MOM60_RUNNER_REVERSAL
    from src.strategies.registry import resolve_strategy_id

    assert resolve_strategy_id("sticky.mom60_runner_reversal") == STICKY_MOM60_RUNNER_REVERSAL
    assert BASELINES["sticky.mom60_runner_reversal"]().name == "sticky.mom60_runner_reversal"
    assert getattr(BASELINES["sticky.mom60_runner_reversal"](), "path_dependent") is True
    assert "sticky.mom60_runner_reversal" in STICKY_ADOPTION_MODELS
    assert resolve_exposure_limits_for_model("sticky.mom60_runner_reversal", comparison_mode="full_strategy_own") == load_p27_exposure_limits()
    assert CHAMPION_STRATEGY == STICKY_MOM60_RAW
