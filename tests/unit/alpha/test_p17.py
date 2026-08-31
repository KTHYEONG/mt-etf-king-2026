def test_p17_is_registered_as_p16_resize_deadband_ablation() -> None:
    from datetime import date
    from pathlib import Path

    import polars as pl
    import pytest

    from src.alpha.base import DecisionContext
    from src.alpha.baselines import BASELINES
    from src.universe.tournament import TournamentRules

    p16 = BASELINES["P16"]()
    p17 = BASELINES["P17"]()
    snapshot = pl.DataFrame({"ticker": ["069500", "122630"], "mom_20": [0.10, 0.22]})
    context = DecisionContext(
        decision_date=date(2024, 6, 1),
        regime=None,
        capital=1_000_000_000.0,
        held={},
        rules=TournamentRules.from_yaml(Path("configs/tournament.yaml")),
    )

    assert p17.name == "P17"
    assert p17.min_rebalance_delta == pytest.approx(0.10)
    assert p16.min_rebalance_delta == 0.0
    assert p17.lottery_config == p16.lottery_config
    assert p17.convexity_config == p16.convexity_config
    assert p17.score(snapshot, context) == p16.score(snapshot, context)
