from __future__ import annotations


def test_specialist_proposals_partition_direction_and_cap_gross() -> None:
    import polars as pl
    from src.tournament.adaptive_specialists import AdaptiveSpecialistConfig, build_specialist_proposals

    frame = pl.DataFrame({"ticker": ["L", "T", "I", "G"], "bucket": ["long_broad", "long_theme", "inverse", "defensive"], "source_multiple": [2, 2, -2, 1], "mom20": [0.1, 0.2, 0.3, 0.0], "mom60": [0.2, 0.3, 0.1, 0.0], "eligible": [True] * 4, "confidence": ["HIGH"] * 4})
    proposals = build_specialist_proposals(frame, AdaptiveSpecialistConfig())
    assert tuple(p.specialist for p in proposals) == ("long_broad", "long_theme", "inverse", "defensive")
    assert all(p.effective_gross <= 1.60 for p in proposals)
    assert proposals[0].target_weight == 0.80


def test_fixed_share_updates_only_from_observed_rewards() -> None:
    import pytest
    from src.tournament.adaptive_specialists import initialize_fixed_share, update_fixed_share

    state = initialize_fixed_share(("long", "inverse"))
    assert state.probabilities == (0.5, 0.5) and state.observations == 0
    updated = update_fixed_share(state, {"long": 0.02, "inverse": -0.02}, horizon=36, reward_bound=0.20)
    assert updated.probabilities[0] > updated.probabilities[1]
    assert updated.observations == 1
    with pytest.raises(ValueError, match="reward"):
        update_fixed_share(state, {"long": 0.02}, horizon=36, reward_bound=0.20)


def test_wealth_controller_rejects_ruin_and_secures_threshold() -> None:
    from src.tournament.adaptive_specialists import AdaptiveSpecialistConfig, TerminalActionDistribution, TournamentWealthState, choose_championship_action

    risky = TerminalActionDistribution("RISKY", tuple([0.70] * 28 + [-0.30] * 2), 1.60)
    cash = TerminalActionDistribution("CASH", tuple([0.41] * 30), 0.0)
    state = TournamentWealthState(1.41, 1.0, 8, 0.40)
    assert choose_championship_action(state, (risky, cash), AdaptiveSpecialistConfig()) == "CASH"


def test_specialist_proposals_fail_closed_for_empty_absent_invalid_rows() -> None:
    import polars as pl
    import pytest
    from src.tournament.adaptive_specialists import AdaptiveSpecialistConfig, build_specialist_proposals

    config = AdaptiveSpecialistConfig()
    with pytest.raises(ValueError, match="empty specialist"):
        build_specialist_proposals(pl.DataFrame({"ticker": []}), config)
    only_broad = pl.DataFrame({"ticker": ["B", "BAD"], "bucket": ["long_broad", "long_theme"], "source_multiple": [2, 2], "mom20": [0.1, float("nan")], "mom60": [0.2, 0.1], "eligible": [True, True], "confidence": ["HIGH", "HIGH"]})
    proposals = build_specialist_proposals(only_broad, config)
    assert proposals[0].ticker == "B"
    assert all(p.ticker is None and p.target_weight == 0.0 for p in proposals[1:])
    zero_multiple = only_broad.with_columns(pl.when(pl.col("ticker") == "B").then(0).otherwise(pl.col("source_multiple")).alias("source_multiple"))
    with pytest.raises(ValueError, match="non-zero"):
        build_specialist_proposals(zero_multiple, config)


def test_fixed_share_rejects_invalid_state_and_inputs() -> None:
    import pytest
    from src.tournament.adaptive_specialists import initialize_fixed_share, update_fixed_share

    with pytest.raises(ValueError, match="at least two"):
        initialize_fixed_share(("only",))
    with pytest.raises(ValueError, match="duplicate"):
        initialize_fixed_share(("same", "same"))
    state = initialize_fixed_share(("a", "b"))
    for horizon, bound, rewards, pattern in ((1, 0.2, {"a": 0.0, "b": 0.0}, "horizon"), (36, 0.0, {"a": 0.0, "b": 0.0}, "reward"), (36, 0.2, {"a": float("nan"), "b": 0.0}, "non-finite")):
        with pytest.raises(ValueError, match=pattern):
            update_fixed_share(state, rewards, horizon=horizon, reward_bound=bound)
    assert state.observations == 0


def test_wealth_controller_covers_insufficient_retention_and_nonfinite_paths() -> None:
    import pytest
    from src.tournament.adaptive_specialists import AdaptiveSpecialistConfig, TerminalActionDistribution, TournamentWealthState, choose_championship_action

    config = AdaptiveSpecialistConfig()
    short_incumbent = TerminalActionDistribution("INCUMBENT", tuple([0.0] * 29), 1.0)
    short_cash = TerminalActionDistribution("CASH", tuple([0.0] * 29), 0.0)
    assert choose_championship_action(TournamentWealthState(1.0, 1.0, 4, None), (short_incumbent, short_cash), config) == "INCUMBENT"
    assert choose_championship_action(TournamentWealthState(1.4, 1.0, 4, 0.40), (short_incumbent, short_cash), config) == "CASH"
    retention_fail = TerminalActionDistribution("RISK", tuple([0.39] * 30), 0.8)
    retention_safe = TerminalActionDistribution("CASH", tuple([0.40] * 30), 0.0)
    assert choose_championship_action(TournamentWealthState(1.4, 1.0, 4, 0.40), (retention_fail, retention_safe), config) == "CASH"
    nonfinite = TerminalActionDistribution("BAD", tuple([float("nan")] * 30), 0.0)
    with pytest.raises(ValueError, match="non-finite"):
        choose_championship_action(TournamentWealthState(1.0, 1.0, 4, None), (nonfinite,), config)


def test_adaptive_runtime_rejects_invalid_panel_contract() -> None:
    from datetime import date
    from types import SimpleNamespace
    import polars as pl
    import pytest
    from src.tournament.adaptive_specialists import run_adaptive_specialist_research

    with pytest.raises(ValueError, match="empty panel"):
        run_adaptive_specialist_research(SimpleNamespace(panel=pl.DataFrame()))
    duplicate = pl.DataFrame({"date": [date(2026, 1, 2), date(2026, 1, 2)]})
    with pytest.raises(ValueError, match="duplicate"):
        run_adaptive_specialist_research(SimpleNamespace(panel=duplicate, dataset_config=SimpleNamespace(label_horizon=36)))
    unique = pl.DataFrame({"date": [date(2026, 1, 2)]})
    with pytest.raises(ValueError, match="horizon"):
        run_adaptive_specialist_research(SimpleNamespace(panel=unique, dataset_config=SimpleNamespace(label_horizon=35)))
    with pytest.raises(ValueError, match="next-open execution"):
        run_adaptive_specialist_research(SimpleNamespace(panel=unique, dataset_config=SimpleNamespace(label_horizon=36), engine=SimpleNamespace()))


def test_adaptive_specialists_integrity_calls_population_counts_match() -> None:
    from pathlib import Path

    from src.research.feasibility_metrics import population_counts_match

    text = Path("src/tournament/adaptive_specialists.py").read_text(encoding="utf-8")
    assert "population_counts_match" in text
    assert "eligible_window_count == 2090" not in text
    assert population_counts_match(eligible=3, evaluated=3, router_reset=3, controller=3 * 36, shadow=3 * 36 * 4, horizon=36) is True
