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
    from src.strategies.registry import STRATEGIES as BASELINES

    return ChampionResearchRuntime(
        engine=engine,
        simulator=simulator,
        panel=panel,
        backtest_config=backtest_config,
        dataset_config=dataset_config,
        objective_config=objective_config,
        policy_config=ChampionPolicyConfig(),
        p27_factory=lambda: BASELINES["sticky.mom60_raw"](),
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


def test_champion_walk_forward_runtime_sets_power_fields(champion_runtime) -> None:
    from src.tournament.champion_eval import run_champion_walk_forward

    result = run_champion_walk_forward(runtime=champion_runtime)
    assert result.status in {"RESEARCH_ONLY", "INSUFFICIENT_POWER", "PROMOTE"}


def test_run_with_runtime_sets_discordant_power_fields(champion_runtime) -> None:
    import time
    from dataclasses import dataclass, field
    from types import SimpleNamespace
    from unittest.mock import patch

    import polars as pl

    from src.tournament.champion_eval import _run_with_runtime
    from src.tournament.loyo import PromotionRobustnessResult, SliceMetrics
    from src.tournament.objective_impl import ChampionshipAdoptionResult

    sessions = champion_runtime.panel["date"].unique().sort().to_list()
    scores = pl.DataFrame(
        {
            "decision_date": sessions,
            "source_ticker": ["AAA"] * len(sessions),
            "score": [0.5] * len(sessions),
            "is_evaluation": [True] * len(sessions),
            "trained_through": [sessions[0]] * len(sessions),
            "fold_id": [0] * len(sessions),
        }
    )
    lineage = ({"fold": 0, "test_count": 1},)
    starts = tuple(sessions[:80])
    rets = tuple(0.01 for _ in starts)

    @dataclass
    class _Rules:
        leverage_allowed: bool = True

    @dataclass
    class _Cache:
        rules: _Rules = field(default_factory=_Rules)

    diag = SimpleNamespace(gross_violation_count=0)
    trades = pl.DataFrame(
        {"decision_date": [sessions[0]], "ticker": ["AAA"], "weight_after": [0.95]}
    )
    roll = SimpleNamespace(
        starts=starts,
        returns=rets,
        diagnostics=diag,
        backtest=SimpleNamespace(trades=trades),
    )
    adoption = ChampionshipAdoptionResult(
        status="PASS",
        failures=(),
        candidate=None,
        incumbent=None,
        raw=None,
        scenario_delta_ci={},
        era_deltas={},
    )
    loyo = PromotionRobustnessResult(
        status="PASS",
        failures=(),
        full_candidate=SliceMetrics("full", 1, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        full_incumbent=None,
        year_metrics={},
        era_metrics={},
        loyo_pass_count=1,
        loyo_n_years=1,
        concentration_2025_2026=0.0,
        loyo_rows=(),
    )

    with (
        patch("src.tournament.champion.walkforward.build_champion_oos_scores", return_value=(scores, lineage)),
        patch("src.backtest.session_cache.build_session_cache", return_value=_Cache()),
        patch.object(champion_runtime.simulator, "run_rolling", return_value=roll),
        patch("src.tournament.objective_impl.evaluate_championship_adoption", return_value=adoption),
        patch("src.tournament.loyo.evaluate_promotion_robustness", return_value=loyo),
    ):
        result = _run_with_runtime(runtime=champion_runtime, t0=time.time())

    assert "n_effective_discordant" in result.extra
    assert result.status in {"INSUFFICIENT_POWER", "PROMOTE", "RESEARCH_ONLY"}


def test_p27_matched_oos_model_has_no_private_allocation_or_vehicle_remap() -> None:
    from datetime import date

    import polars as pl

    from src.alpha.base import DecisionContext
    from src.alpha.champion_ranker import OosScoreStore
    from src.portfolio.intent import HOLD_INTENT
    from src.tournament.champion_eval import Mom60RawMatchedOosModel

    scored = date(2026, 3, 2)
    store = OosScoreStore(pl.DataFrame({'decision_date': [scored], 'source_ticker': ['TWO'], 'score': [0.75], 'fold_id': [0], 'trained_through': [date(2026, 1, 1)], 'is_evaluation': [True]}))
    model = Mom60RawMatchedOosModel(scores=store)
    context = DecisionContext(decision_date=scored, regime=None, capital=1_000_000_000.0, held={}, rules=None)

    assert model.name == 'sticky.mom60_raw'
    assert model.candidate_id == 'champion.candidate'
    assert not hasattr(model, 'allocate')
    assert model.score(pl.DataFrame({'ticker': ['TWO', 'ONE']}), context) == {'TWO': 0.75}
    missing = DecisionContext(decision_date=date(2026, 3, 3), regime=None, capital=1_000_000_000.0, held={}, rules=None)
    assert model.score(pl.DataFrame({'ticker': ['TWO']}), missing) == HOLD_INTENT


def test_p27_matched_comparison_profile_uses_identical_incumbent_limits() -> None:
    from src.portfolio.constraints import load_p27_exposure_limits
    from src.tournament.champion_eval import mom60_raw_matched_comparison_profile

    profile = mom60_raw_matched_comparison_profile()

    assert profile.candidate_model_name == 'sticky.mom60_raw'
    assert profile.incumbent_model_name == 'sticky.mom60_raw'
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
        extra={'candidate_id': 'champion.candidate', 'gross_violation_count': 0},
    )

    assert result.status == 'RESEARCH_ONLY'
    assert result.extra['candidate_id'] == 'champion.candidate'
    assert is_promotable(
        aggressive_status=result.aggressive_status,
        conservative_status=result.conservative_status,
        loyo_status=result.loyo_status,
        artifact_integrity=result.artifact_integrity,
    ) is False


def _executable_panel(n_days: int = 480):
    from datetime import date, timedelta

    import polars as pl

    from src.universe.instruments import Confidence, InstrumentAttributes, InstrumentMaster

    start = date(2026, 1, 1)
    sessions = [start + timedelta(days=i) for i in range(n_days)]
    feature_cols = ['mom_3', 'mom_5', 'mom_10', 'mom_20', 'mom_40', 'mom_60', 'ma_ratio_20', 'drawdown_20', 'volume_expansion', 'rv_20', 'log_adv_20', 'fill_sessions', 'leverage_multiple']
    rows: list[dict[str, object]] = []
    for i, d in enumerate(sessions):
        for t in ('AAA', 'BBB'):
            base = 100.0 + i * 0.3 + (1.0 if t == 'AAA' else 0.0)
            row: dict[str, object] = {'date': d, 'ticker': t, 'open': base, 'high': base + 0.5, 'low': base - 0.5, 'close': base + 0.1, 'trading_value': 50_000_000_000.0, 'is_tradable': True}
            for j, col in enumerate(feature_cols):
                if col == 'leverage_multiple':
                    row[col] = 1.0 if t == 'AAA' else 2.0
                else:
                    row[col] = 0.05 + i * 0.0001 + j * 0.001
            rows.append(row)
    panel = pl.DataFrame(rows)
    attrs = {
        t: InstrumentAttributes(ticker=t, name=t, issuer='삼성자산운용', leverage_multiple=1, leverage_family_key=t, is_synthetic=False, is_hedged=False, is_active=True, index_key='KOSPI 200', theme='ThemeA', first_seen=sessions[0], last_seen=sessions[-1], left_censored=True, confidence=Confidence.HIGH)
        for t in ('AAA', 'BBB')
    }
    return panel, sessions, feature_cols, InstrumentMaster(attributes=attrs, panel_start=sessions[0])


def _executable_runtime(panel, master, start, end, **overrides):
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
    from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode
    from src.strategies.registry import STRATEGIES as BASELINES

    cal = TradingCalendar()
    universe = PointInTimeUniverse(panel, master, cal, adv_window=5, brand_map={})
    filt = UniverseFilters(mode=UniverseMode.DEPLOYMENT, warmup_sessions=0, adv_window=5, capital=1_000_000_000, max_position_weight=1.0, max_order_to_adv=5.0)
    fconfig = FeatureConfig(momentum_horizons=(20,), ma_windows=(20,), breakout_windows=(20,), volatility_windows=(20,), flow_windows=(5,), regime=RegimeConfig(weights={}, thresholds=(0.25, 0.45, 0.65, 0.85), breadth_floor=0.5, volatility_ceiling=0.025))
    builder = FeatureBuilder(cal, fconfig)
    engine = BacktestEngine(cal, universe, builder, NextOpenExecution(cal))
    from src.tournament.simulator import TournamentSimulator

    simulator = TournamentSimulator(engine, cal)
    kwargs = {
        'engine': engine, 'simulator': simulator, 'panel': panel,
        'backtest_config': BacktestConfig(start=start, end=end, capital=1_000_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=CostConfig()),
        'dataset_config': ChampionDatasetConfig(feature_columns=('mom_60', 'mom_20'), label_horizon=5, entry_cost_rate=0.0005, exit_cost_rate=0.0005),
        'objective_config': ChampionshipObjectiveConfig(thresholds=(0.3, 0.4, 0.5, 0.6), scenario_weights={'championship': (0.1, 0.25, 0.45, 0.2)}, primary_scenario='championship', ruin_threshold=-0.25, ruin_max=0.05, max_effective_gross=1.60, bootstrap_expected_block=36, bootstrap_resamples=10, seed=0, min_era_effective=0),
        'policy_config': ChampionPolicyConfig(), 'p27_factory': lambda: BASELINES['sticky.mom60_raw'](),
        'min_train_sessions': 10, 'n_folds': 2, 'candidate_mode': 'executable_hurdle',
    }
    kwargs.update(overrides)
    return ChampionResearchRuntime(**kwargs)


def test_build_executable_hurdle_oos_scores_produces_calibrated_scores() -> None:
    import polars as pl

    from src.tournament.champion_eval import build_executable_hurdle_oos_scores

    panel, sessions, _cols, master = _executable_panel()
    runtime = _executable_runtime(panel, master, sessions[0], sessions[-1])
    scores, lineage = build_executable_hurdle_oos_scores(runtime)
    evaluated = scores.filter(pl.col('is_evaluation'))

    assert scores.height > 0
    assert evaluated.height > 0
    assert 'activate' in scores.columns
    assert len(lineage) == 2
    assert all('activation_threshold' in dict(row) for row in lineage)


def test_build_executable_hurdle_oos_scores_fails_closed_on_runtime_gaps() -> None:
    from datetime import date
    from types import SimpleNamespace
    from unittest.mock import patch

    import polars as pl
    import pytest

    from src.core.calendar import TradingCalendar
    from src.tournament.champion_eval import build_executable_hurdle_oos_scores
    from src.universe.provider import UniverseFilters, UniverseMode

    cal = TradingCalendar()
    day = date(2026, 1, 5)
    good_panel = pl.DataFrame({'date': [day], 'ticker': ['ONE'], 'open': [100.0], 'close': [101.0], 'trading_value': [1e11], 'mom_5': [0.1]})
    master = SimpleNamespace(attributes={})
    universe = SimpleNamespace(master=master)
    engine = SimpleNamespace(universe=universe, calendar=cal)
    filt = UniverseFilters(mode=UniverseMode.DEPLOYMENT, warmup_sessions=0, adv_window=1, capital=1, max_position_weight=1.0, max_order_to_adv=1.0)
    config = SimpleNamespace(filters=filt, start=day, end=day)

    with pytest.raises(ValueError, match='empty panel'):
        build_executable_hurdle_oos_scores(SimpleNamespace(panel=pl.DataFrame(), engine=engine, backtest_config=config))
    with pytest.raises(ValueError, match='engine universe missing'):
        build_executable_hurdle_oos_scores(SimpleNamespace(panel=good_panel, engine=SimpleNamespace(calendar=cal), backtest_config=config))
    with pytest.raises(ValueError, match='universe master missing'):
        build_executable_hurdle_oos_scores(SimpleNamespace(panel=good_panel, engine=SimpleNamespace(universe=SimpleNamespace(), calendar=cal), backtest_config=config))
    with pytest.raises(ValueError, match='backtest filters missing'):
        build_executable_hurdle_oos_scores(SimpleNamespace(panel=good_panel, engine=engine, backtest_config=SimpleNamespace(start=day, end=day)))
    with pytest.raises(ValueError, match='panel date column'):
        build_executable_hurdle_oos_scores(SimpleNamespace(panel=pl.DataFrame({'ticker': ['ONE']}), engine=engine, backtest_config=config))
    saturday = date(2026, 1, 3)
    with pytest.raises(ValueError, match='no sessions'):
        build_executable_hurdle_oos_scores(SimpleNamespace(panel=pl.DataFrame({'date': [saturday], 'ticker': ['ONE']}), engine=engine, backtest_config=SimpleNamespace(filters=filt, start=day, end=day)))
    from datetime import timedelta as _td

    window = [date(2025, 1, 2) + _td(days=i) for i in range(10)]
    window_panel = pl.DataFrame({'date': window, 'ticker': ['ONE'] * 10, 'open': [100.0] * 10, 'close': [101.0] * 10, 'trading_value': [1e11] * 10, 'mom_5': [0.1] * 10})
    window_config = SimpleNamespace(filters=filt, start=window[0], end=window[-1])
    with patch('yaml.safe_load', return_value={}):  # noqa: SIM117
        with pytest.raises(ValueError, match='executable_feature_columns'):
            build_executable_hurdle_oos_scores(SimpleNamespace(panel=window_panel, engine=engine, backtest_config=window_config))


def test_build_executable_hurdle_oos_scores_rejects_empty_candidate_or_label_stages() -> None:
    from datetime import date, timedelta
    from types import SimpleNamespace

    import polars as pl
    import pytest

    from src.core.calendar import TradingCalendar
    from src.tournament.champion_eval import build_executable_hurdle_oos_scores
    from src.universe.instruments import Confidence
    from src.universe.provider import UniverseFilters, UniverseMode

    cal = TradingCalendar()
    start = date(2026, 1, 1)
    sessions = [start + timedelta(days=i) for i in range(40)]
    feature_cols = ['mom_3', 'mom_5', 'mom_10', 'mom_20', 'mom_40', 'mom_60', 'ma_ratio_20', 'drawdown_20', 'volume_expansion', 'rv_20', 'log_adv_20', 'fill_sessions', 'leverage_multiple']
    rows: list[dict[str, object]] = []
    for i, d in enumerate(sessions):
        row: dict[str, object] = {'date': d, 'ticker': 'ONE', 'open': 100.0, 'close': 101.0, 'trading_value': 1e11}
        for j, col in enumerate(feature_cols):
            row[col] = 0.05 + i * 0.0001 + j * 0.001
        rows.append(row)
    panel = pl.DataFrame(rows)
    filt = UniverseFilters(mode=UniverseMode.DEPLOYMENT, warmup_sessions=0, adv_window=1, capital=1, max_position_weight=1.0, max_order_to_adv=1.0)
    config = SimpleNamespace(filters=filt, start=sessions[0], end=sessions[-1])
    empty_universe = SimpleNamespace(get=lambda _d, _f: SimpleNamespace(tickers=()), master=SimpleNamespace(attributes={}))
    engine_empty = SimpleNamespace(universe=empty_universe, calendar=cal)
    with pytest.raises(ValueError, match='no direct candidates'):
        build_executable_hurdle_oos_scores(SimpleNamespace(panel=panel, engine=engine_empty, backtest_config=config, dataset_config=SimpleNamespace(), objective_config=SimpleNamespace(), policy_config=SimpleNamespace(), p27_factory=lambda: None, min_train_sessions=10, n_folds=2, embargo_sessions=36, purge_sessions=36, ranker_seed=7, ranker_num_leaves=8, ranker_max_depth=4, ranker_min_data_in_leaf=100, candidate_mode='executable_hurdle'))
    attrs = {'ONE': SimpleNamespace(is_synthetic=False, confidence=Confidence.HIGH, leverage_multiple=1, leverage_family_key='F1')}
    thin_universe = SimpleNamespace(get=lambda _d, _f: SimpleNamespace(tickers=('ONE',)), master=SimpleNamespace(attributes=attrs))
    engine_thin = SimpleNamespace(universe=thin_universe, calendar=cal)
    from src.alpha.champion_dataset import ChampionDatasetConfig as _CDC

    thin_dataset = _CDC(feature_columns=('mom_5',), label_horizon=2, entry_cost_rate=0.0, exit_cost_rate=0.0)
    with pytest.raises(ValueError, match='no executable labels'):
        build_executable_hurdle_oos_scores(SimpleNamespace(panel=panel, engine=engine_thin, backtest_config=config, dataset_config=thin_dataset, objective_config=SimpleNamespace(), policy_config=SimpleNamespace(), p27_factory=lambda: None, min_train_sessions=10, n_folds=2, embargo_sessions=36, purge_sessions=36, ranker_seed=7, ranker_num_leaves=8, ranker_max_depth=4, ranker_min_data_in_leaf=100, candidate_mode='executable_hurdle'))
