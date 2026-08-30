# ruff: noqa
import polars as pl
from datetime import date

from src.alpha.baselines import BASELINES


def test_p15_keeps_p14_alpha_and_enables_selective_capacity_route() -> None:
    # P14 and P15 score dictionaries identical for same snapshot
    p14 = BASELINES["P14"]()
    p15 = BASELINES["P15"]()
    # Create minimal snapshot
    panel = pl.DataFrame(
        [
            {"date": date(2026, 1, 2), "ticker": "069500", "close": 100.0, "mom_20": 0.05, "mom_20_rs": 0.5, "name": "KODEX 200", "underlying_index_name": "KOSPI 200"},
            {"date": date(2026, 1, 2), "ticker": "114800", "close": 100.0, "mom_20": 0.04, "mom_20_rs": 0.4, "name": "KODEX 인버스", "underlying_index_name": "KOSPI 200"},
        ]
    )
    from src.alpha.base import DecisionContext
    from src.universe.tournament import TournamentRules

    rules = TournamentRules(name="test", start_date=date(2026, 1, 2), end_date=date(2026, 1, 3), initial_capital=1_000_000_000, category="auto", leverage_allowed=True, inverse_allowed=False, max_weight=1.0, cash_allowed=True, sponsor_etf_only=False, manifest_path=None, issuer_whitelist=None, commission_bps=3.0, slippage_bps=5.0, max_order_to_adv=0.01, stress_grid=(0.01,))
    ctx = DecisionContext(decision_date=date(2026, 1, 2), regime=None, capital=1_000_000_000.0, held={}, rules=rules)
    scores_p14 = p14.score(panel, ctx)
    scores_p15 = p15.score(panel, ctx)
    assert scores_p14 == scores_p15
    # Only P15 uses capacity-aware routing with suppress_vehicle_gate=false and suppress_trim=false
    assert getattr(p15.lottery_config, "suppress_vehicle_gate") is False
    assert getattr(p15.lottery_config, "suppress_trim") is False
    assert getattr(p14.lottery_config, "suppress_vehicle_gate") is True or getattr(p14.lottery_config, "suppress_vehicle_gate") is True  # P14 is true
    # P15 BASELINES registry exists
    assert "P15" in BASELINES
    assert callable(BASELINES["P15"])
