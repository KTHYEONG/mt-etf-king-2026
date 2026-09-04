from __future__ import annotations

import pytest


@pytest.fixture
def champion_runtime():
    from datetime import date, timedelta

    import polars as pl

    from src.alpha.champion_dataset import ChampionDatasetConfig
    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.backtest.execution import NextOpenExecution
    from src.core.calendar import TradingCalendar
    from src.features.builder import FeatureBuilder, FeatureConfig
    from src.features.regime import RegimeConfig
    from src.portfolio.sizing import SizingScheme
    from src.strategies.champion_tail import ChampionPolicyConfig
    from src.tournament.champion_eval import ChampionResearchRuntime
    from src.tournament.objective_impl import ChampionshipObjectiveConfig
    from src.universe.instruments import Confidence, InstrumentAttributes, InstrumentMaster
    from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode

    n_sessions = 160
    start = date(2025, 1, 2)
    sessions = [start + timedelta(days=i) for i in range(n_sessions)]
    tickers = ["AAA", "BBB"]
    rows: list[dict[str, object]] = []
    for i, d in enumerate(sessions):
        for t in tickers:
            base = 100.0 + i * 0.5 + (1.0 if t == "AAA" else 0.0)
            rows.append(
                {
                    "date": d,
                    "ticker": t,
                    "open": base,
                    "high": base + 0.5,
                    "low": base - 0.5,
                    "close": base + 0.1,
                    "trading_value": 50_000_000_000.0,
                    "is_tradable": True,
                    "mom_60": 0.05 + i * 0.0001,
                    "mom_20": 0.02,
                }
            )
    panel = pl.DataFrame(rows)
    attrs = {
        t: InstrumentAttributes(
            ticker=t,
            name=t,
            issuer="삼성자산운용",
            leverage_multiple=1,
            leverage_family_key=t,
            is_synthetic=False,
            is_hedged=False,
            is_active=True,
            index_key="KOSPI 200",
            theme="ThemeA",
            first_seen=sessions[0],
            last_seen=sessions[-1],
            left_censored=True,
            confidence=Confidence.HIGH,
        )
        for t in tickers
    }
    master = InstrumentMaster(attributes=attrs, panel_start=sessions[0])
    cal = TradingCalendar()
    universe = PointInTimeUniverse(panel, master, cal, adv_window=5, brand_map={})
    filt = UniverseFilters(
        mode=UniverseMode.DEPLOYMENT,
        warmup_sessions=0,
        adv_window=5,
        capital=1_000_000_000,
        max_position_weight=1.0,
        max_order_to_adv=5.0,
    )
    fconfig = FeatureConfig(
        momentum_horizons=(20,),
        ma_windows=(20,),
        breakout_windows=(20,),
        volatility_windows=(20,),
        flow_windows=(5,),
        regime=RegimeConfig(weights={}, thresholds=(0.25, 0.45, 0.65, 0.85), breadth_floor=0.5, volatility_ceiling=0.025),
    )
    builder = FeatureBuilder(cal, fconfig)
    engine = BacktestEngine(cal, universe, builder, NextOpenExecution(cal))
    from src.tournament.simulator import TournamentSimulator

    simulator = TournamentSimulator(engine, cal)
    dataset_config = ChampionDatasetConfig(
        feature_columns=("mom_60", "mom_20"),
        label_horizon=5,
        entry_cost_rate=0.0005,
        exit_cost_rate=0.0005,
    )
    objective_config = ChampionshipObjectiveConfig(
        thresholds=(0.3, 0.4, 0.5, 0.6),
        scenario_weights={
            "weak_field": (0.55, 0.3, 0.1, 0.05),
            "championship": (0.1, 0.25, 0.45, 0.2),
            "hot_field": (0.0, 0.1, 0.35, 0.55),
        },
        primary_scenario="championship",
        ruin_threshold=-0.25,
        ruin_max=0.05,
        max_effective_gross=1.60,
        bootstrap_expected_block=36,
        bootstrap_resamples=10,
        seed=0,
        min_era_effective=0,
    )
    backtest_config = BacktestConfig(
        start=sessions[0],
        end=sessions[-1],
        capital=1_000_000_000.0,
        scheme=SizingScheme.TOP1,
        k=1,
        filters=filt,
        costs=CostConfig(),
    )
    from src.alpha.baselines import BASELINES

    return ChampionResearchRuntime(
        engine=engine,
        simulator=simulator,
        panel=panel,
        backtest_config=backtest_config,
        dataset_config=dataset_config,
        objective_config=objective_config,
        policy_config=ChampionPolicyConfig(),
        p27_factory=lambda: BASELINES["P27"](),
        min_train_sessions=20,
        candidate_mode="family_1x",
    )


def test_champion_promotion_requires_dual_scenario_loyo_and_integrity() -> None:
    from src.tournament.champion_eval import is_promotable

    assert is_promotable(aggressive_status='PASS', conservative_status='PASS', loyo_status='PASS', artifact_integrity=True) is True
    assert is_promotable(aggressive_status='PASS', conservative_status='FAIL', loyo_status='PASS', artifact_integrity=True) is False
    assert is_promotable(aggressive_status='PASS', conservative_status='PASS', loyo_status='FAIL', artifact_integrity=True) is False
    assert is_promotable(aggressive_status='PASS', conservative_status='PASS', loyo_status='PASS', artifact_integrity=False) is False


def test_champion_oos_model_uses_only_evaluation_scores() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.champion_ranker import OosScoreStore
    from src.portfolio.intent import HOLD_INTENT
    from src.strategies.champion_tail import ChampionTailPolicy
    from src.tournament.champion_eval import ChampionOosModel

    scored = date(2026, 3, 2)
    model = ChampionOosModel(scores=OosScoreStore(pl.DataFrame({"decision_date": [scored], "source_ticker": ["ONE"], "score": [0.75], "is_evaluation": [True]})), policy=ChampionTailPolicy())
    snapshot = pl.DataFrame({"ticker": ["ONE", "OTHER"], "mom_60": [0.10, 0.90]})
    context = DecisionContext(decision_date=scored, regime=None, capital=1_000_000_000.0, held={}, rules=None)

    assert model.score(snapshot, context) == {"ONE": 0.75}
    missing = DecisionContext(decision_date=date(2026, 3, 3), regime=None, capital=1_000_000_000.0, held={}, rules=None)
    assert model.score(snapshot, missing) == HOLD_INTENT


def test_build_champion_oos_scores_enforces_purge_and_lineage(champion_runtime) -> None:
    import polars as pl

    from src.tournament.champion_eval import build_champion_oos_scores

    scores, lineage = build_champion_oos_scores(champion_runtime)
    evaluated = scores.filter(pl.col("is_evaluation"))

    assert evaluated.height > 0
    assert evaluated.select((pl.col("decision_date") > pl.col("trained_through")).all()).item() is True
    assert evaluated.unique(subset=["decision_date", "source_ticker"]).height == evaluated.height
    assert all(int(row["test_count"]) > 0 for row in lineage)


def test_champion_walk_forward_requires_complete_runtime() -> None:
    from datetime import date

    from src.tournament.champion_eval import run_champion_walk_forward

    result = run_champion_walk_forward(start=date(2026, 1, 2), end=date(2026, 8, 27))

    assert result.status == "RESEARCH_ONLY"
    assert result.aggressive_status == "INSUFFICIENT_EVIDENCE"
    assert result.artifact_integrity is False
    assert result.extra["missing_runtime_inputs"]


def test_p27_matched_oos_model_has_no_private_allocation_or_vehicle_remap() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.champion_ranker import OosScoreStore
    from src.portfolio.intent import HOLD_INTENT
    from src.tournament.champion_eval import P27MatchedOosModel

    scored = date(2026, 3, 2)
    store = OosScoreStore(pl.DataFrame({'decision_date': [scored], 'source_ticker': ['TWO'], 'score': [0.75], 'fold_id': [0], 'trained_through': [date(2026, 1, 1)], 'is_evaluation': [True]}))
    model = P27MatchedOosModel(scores=store)
    context = DecisionContext(decision_date=scored, regime=None, capital=1_000_000_000.0, held={}, rules=None)

    assert model.name == 'P27'
    assert model.candidate_id == 'P35'
    assert not hasattr(model, 'allocate')
    assert model.score(pl.DataFrame({'ticker': ['TWO', 'ONE']}), context) == {'TWO': 0.75}
    missing = DecisionContext(decision_date=date(2026, 3, 3), regime=None, capital=1_000_000_000.0, held={}, rules=None)
    assert model.score(pl.DataFrame({'ticker': ['TWO']}), missing) == HOLD_INTENT


def test_p27_matched_comparison_profile_uses_identical_incumbent_limits() -> None:
    from src.portfolio.constraints import load_p27_exposure_limits
    from src.tournament.champion_eval import p27_matched_comparison_profile

    profile = p27_matched_comparison_profile()

    assert profile.candidate_model_name == 'P27'
    assert profile.incumbent_model_name == 'P27'
    assert profile.candidate_limits == load_p27_exposure_limits()
    assert profile.incumbent_limits == load_p27_exposure_limits()
    assert profile.candidate_limits == profile.incumbent_limits


def test_p35_result_remains_research_only_when_any_gate_fails() -> None:
    from src.tournament.champion_eval import ChampionEvaluation, is_promotable

    result = ChampionEvaluation(
        status='RESEARCH_ONLY',
        aggressive_status='FAIL',
        conservative_status='PASS',
        loyo_status='PASS',
        artifact_integrity=True,
        extra={'candidate_id': 'P35', 'gross_violation_count': 0},
    )

    assert result.status == 'RESEARCH_ONLY'
    assert result.extra['candidate_id'] == 'P35'
    assert is_promotable(
        aggressive_status=result.aggressive_status,
        conservative_status=result.conservative_status,
        loyo_status=result.loyo_status,
        artifact_integrity=result.artifact_integrity,
    ) is False
