def test_resolve_complement_active_book_lottery_on_is_attack() -> None:
    from src.tournament.championship_regime import ChampionshipSleeve
    from src.tournament.p27_complement_sleeve import resolve_complement_active_book

    assert resolve_complement_active_book(sleeve=ChampionshipSleeve.LOTTERY_ON) == "attack"
    assert resolve_complement_active_book(sleeve="LOTTERY_ON") == "attack"


def test_resolve_complement_active_book_non_lottery_is_complement() -> None:
    from src.tournament.championship_regime import ChampionshipSleeve
    from src.tournament.p27_complement_sleeve import resolve_complement_active_book

    for sleeve in (
        ChampionshipSleeve.INACTIVE,
        ChampionshipSleeve.CRASH_REBOUND,
        ChampionshipSleeve.UNCERTAIN,
        "INACTIVE",
        "CRASH_REBOUND",
        "UNCERTAIN",
        None,
        "",
        "UNKNOWN",
    ):
        assert resolve_complement_active_book(sleeve=sleeve) == "complement"


def test_reject_if_capital_split_raises_on_partial_share() -> None:
    import pytest

    from src.tournament.p27_complement_sleeve import reject_if_capital_split

    with pytest.raises(ValueError, match="capital split"):
        reject_if_capital_split(0.5)
    with pytest.raises(ValueError, match="capital split"):
        reject_if_capital_split(0.3)


def test_reject_if_capital_split_allows_exclusive_books() -> None:
    from src.tournament.p27_complement_sleeve import reject_if_capital_split

    reject_if_capital_split(0.0)
    reject_if_capital_split(1.0)


def test_evaluate_complement_tail_non_inferiority_fail_closed() -> None:
    from src.tournament.p27_complement_sleeve import evaluate_complement_tail_non_inferiority

    assert evaluate_complement_tail_non_inferiority(candidate_p50=0.0609, incumbent_p50=0.0508) is True
    assert evaluate_complement_tail_non_inferiority(candidate_p50=0.0508, incumbent_p50=0.0508) is True
    assert evaluate_complement_tail_non_inferiority(candidate_p50=0.0283, incumbent_p50=0.0508) is False
    assert evaluate_complement_tail_non_inferiority(candidate_p50=float("nan"), incumbent_p50=0.05) is False
    assert evaluate_complement_tail_non_inferiority(candidate_p50=0.06, incumbent_p50=float("inf")) is False


def test_p27_complement_production_gate_false_blocks_promotion() -> None:
    from src.tournament.p27_complement_sleeve import (
        P27_COMPLEMENT_IS_PRODUCTION_GATE,
        may_promote_complement_over_p27,
    )

    assert P27_COMPLEMENT_IS_PRODUCTION_GATE is False
    assert may_promote_complement_over_p27(candidate_p50=0.99, incumbent_p50=0.01) is False


def test_hard_switch_model_routes_lottery_to_attack_only() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.tournament.p27_complement_sleeve import P27ComplementHardSwitchModel
    from src.universe.tournament import TournamentRules

    class _Spy:
        def __init__(self, label: str) -> None:
            self.label = label
            self.calls = 0

        def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
            self.calls += 1
            return {self.label: 1.0}

    attack = _Spy("ATTACK")
    complement = _Spy("COMP")
    model = P27ComplementHardSwitchModel(attack=attack, complement=complement)
    assert model.path_dependent is True
    assert model.name == "sticky.p27_complement_switch"
    rules = TournamentRules(
        name="t",
        start_date=date(2026, 9, 21),
        end_date=date(2026, 11, 13),
        initial_capital=1_000_000_000,
        category="autonomous",
        leverage_allowed=True,
        inverse_allowed=True,
        max_weight=1.0,
        cash_allowed=True,
        sponsor_etf_only=True,
        manifest_path=None,
        issuer_whitelist=None,
        commission_bps=3.0,
        slippage_bps=5.0,
        max_order_to_adv=0.01,
        stress_grid=(0.01, 0.02, 0.05),
    )
    ctx = DecisionContext(
        decision_date=date(2025, 9, 22),
        regime=None,
        capital=1.0e9,
        held={},
        rules=rules,
        championship_sleeve="LOTTERY_ON",
    )
    out = model.score(pl.DataFrame({"ticker": ["X"], "mom_60": [0.2]}), ctx)
    assert out == {"ATTACK": 1.0}
    assert attack.calls == 1
    assert complement.calls == 0


def test_hard_switch_model_routes_inactive_to_complement_only() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.tournament.p27_complement_sleeve import P27ComplementHardSwitchModel
    from src.universe.tournament import TournamentRules

    class _Spy:
        def __init__(self, label: str) -> None:
            self.label = label
            self.calls = 0

        def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
            self.calls += 1
            return {self.label: 1.0}

    attack = _Spy("ATTACK")
    complement = _Spy("COMP")
    model = P27ComplementHardSwitchModel(attack=attack, complement=complement)
    rules = TournamentRules(
        name="t",
        start_date=date(2026, 9, 21),
        end_date=date(2026, 11, 13),
        initial_capital=1_000_000_000,
        category="autonomous",
        leverage_allowed=True,
        inverse_allowed=True,
        max_weight=1.0,
        cash_allowed=True,
        sponsor_etf_only=True,
        manifest_path=None,
        issuer_whitelist=None,
        commission_bps=3.0,
        slippage_bps=5.0,
        max_order_to_adv=0.01,
        stress_grid=(0.01, 0.02, 0.05),
    )
    ctx = DecisionContext(
        decision_date=date(2019, 6, 3),
        regime=None,
        capital=1.0e9,
        held={},
        rules=rules,
        championship_sleeve="INACTIVE",
    )
    out = model.score(pl.DataFrame({"ticker": ["Y"], "mom_20": [0.1]}), ctx)
    assert out == {"COMP": 1.0}
    assert attack.calls == 0
    assert complement.calls == 1


def test_hard_switch_model_reset_trackers_delegates_both() -> None:
    from src.tournament.p27_complement_sleeve import P27ComplementHardSwitchModel

    class _Tracked:
        def __init__(self) -> None:
            self.resets = 0

        def reset_trackers(self) -> None:
            self.resets += 1

        def score(self, snapshot: object, context: object) -> dict[str, float]:
            return {"noop": 0.0}

    attack = _Tracked()
    complement = _Tracked()
    model = P27ComplementHardSwitchModel(attack=attack, complement=complement)
    model.reset_trackers()
    assert attack.resets == 1
    assert complement.resets == 1


def test_make_p27_complement_switch_wires_p27_and_b1() -> None:
    from src.tournament.p27_complement_sleeve import make_p27_complement_switch

    model = make_p27_complement_switch()
    assert model.name == "sticky.p27_complement_switch"
    assert getattr(model.attack, "name", "") == "sticky.mom60_raw" or "mom60" in str(getattr(model.attack, "name", "")).lower()
    assert getattr(model.complement, "name", "") == "baseline.mom20_top1"


def test_may_promote_complement_over_p27_when_gate_enabled(monkeypatch: object) -> None:
    import pytest

    import src.tournament.p27_complement_sleeve as pcs

    monkeypatch.setattr(pcs, "P27_COMPLEMENT_IS_PRODUCTION_GATE", True)  # type: ignore[attr-defined]
    assert pcs.may_promote_complement_over_p27(candidate_p50=0.06, incumbent_p50=0.05) is True
    assert pcs.may_promote_complement_over_p27(candidate_p50=0.04, incumbent_p50=0.05) is False


def test_hard_switch_model_raises_type_error_on_non_callable_score() -> None:
    from datetime import date

    import polars as pl
    import pytest

    from src.alpha.base import DecisionContext
    from src.tournament.p27_complement_sleeve import P27ComplementHardSwitchModel
    from src.universe.tournament import TournamentRules

    model = P27ComplementHardSwitchModel(attack="not_a_model", complement="not_a_model")
    rules = TournamentRules(
        name="t",
        start_date=date(2026, 9, 21),
        end_date=date(2026, 11, 13),
        initial_capital=1_000_000_000,
        category="autonomous",
        leverage_allowed=True,
        inverse_allowed=True,
        max_weight=1.0,
        cash_allowed=True,
        sponsor_etf_only=True,
        manifest_path=None,
        issuer_whitelist=None,
        commission_bps=3.0,
        slippage_bps=5.0,
        max_order_to_adv=0.01,
        stress_grid=(0.01, 0.02, 0.05),
    )
    ctx = DecisionContext(
        decision_date=date(2019, 6, 3),
        regime=None,
        capital=1.0e9,
        held={},
        rules=rules,
        championship_sleeve="LOTTERY_ON",
    )
    with pytest.raises(TypeError, match="active book missing callable score"):
        model.score(pl.DataFrame(), ctx)

