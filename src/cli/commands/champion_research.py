# ruff: noqa
from __future__ import annotations

import argparse
import logging
from datetime import date
from typing import Any

from src.core.config import config_path
from src.core.paths import DataPaths
from src.tournament.champion_eval import run_champion_walk_forward
from src.tournament.frontier import run_frontier_from_paths

logger = logging.getLogger(__name__)


from src.cli.commands.decide.scoring import _load_panel_for_backtest
from src.universe.provider import UniverseFilters


def build_champion_research_filters(base: UniverseFilters, candidate_mode: str) -> UniverseFilters:
    from dataclasses import replace

    from src.universe.provider import LiquidityAdmissionMode

    return (
        replace(
            base,
            liquidity_admission=LiquidityAdmissionMode.STAGED_EXECUTION,
            max_order_to_adv=0.01,
            max_position_weight=0.80,
            allow_leverage=True,
            allow_inverse=True,
        )
        if str(candidate_mode) in ("executable_hurdle", "adaptive_specialists")
        else base
    )


def _build_champion_research_inputs(args: argparse.Namespace) -> dict[str, object]:
    """Assemble real runtime kwargs for run_champion_walk_forward (no synthetic panel)."""
    # anchor: def cmd_backtest (shared backtest wiring surface)
    try:
        start = date.fromisoformat(str(getattr(args, "start", "")))
    except Exception:
        raise ValueError("invalid --start (expected YYYY-MM-DD)")
    try:
        end = date.fromisoformat(str(getattr(args, "end", "")))
    except Exception:
        raise ValueError("invalid --end (expected YYYY-MM-DD)")
    if start > end:
        raise ValueError("start must not exceed end")
    _candidate_mode_arg = getattr(args, "candidate_mode", None)
    if _candidate_mode_arg not in {"p27_matched_2x", "executable_hurdle", "adaptive_specialists"}:  # pragma: no cover - CLI validation
        raise ValueError("invalid candidate_mode")  # pragma: no cover - CLI validation
    from pathlib import Path as _Path

    from src.alpha.champion_dataset import ChampionDatasetConfig
    from src.backtest.costs import CostConfig
    from src.backtest.engine import BacktestConfig, BacktestEngine
    from src.backtest.execution import NextOpenExecution
    from src.core.calendar import get_calendar
    from src.core.paths import DataPaths
    from src.core.settings import get_settings
    from src.features.builder import FeatureBuilder, FeatureConfig
    from src.strategies.champion_tail import ChampionPolicyConfig
    from src.tournament.champion_eval import ChampionResearchRuntime
    from src.tournament.objective_impl import ChampionshipObjectiveConfig
    from src.tournament.simulator import TournamentSimulator
    from src.universe.instruments import InstrumentMaster, load_sponsor_brand_map
    from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode
    from src.universe.taxonomy import Taxonomy

    settings = get_settings()
    paths = DataPaths(root=settings.data_root)
    cal = get_calendar()
    panel = _load_panel_for_backtest(paths, cal)
    if panel is None or panel.height == 0:
        raise ValueError("empty real panel for champion research (synthetic panel forbidden)")
    try:
        brand_map = load_sponsor_brand_map(config_path("sponsor_brands"))
    except Exception:
        brand_map = {}
    try:
        taxonomy = Taxonomy.from_yaml(config_path("taxonomy"))
    except Exception:
        taxonomy = Taxonomy(rules=[])
    master = InstrumentMaster.build(panel, taxonomy, brand_map)
    import yaml as _yaml

    try:
        with open(config_path("universe"), encoding="utf-8") as f:
            _uc_raw = _yaml.safe_load(f) or {}
        universe_config = _uc_raw.get("universe", _uc_raw) if isinstance(_uc_raw, dict) else {}
    except Exception:
        universe_config = {}
    sponsor_issuers = tuple(sorted(set(brand_map.values()))) if brand_map else ()
    filt = UniverseFilters.for_mode(UniverseMode.DEPLOYMENT, universe_config, sponsor_issuers)
    filt = build_champion_research_filters(filt, _candidate_mode_arg)
    universe = PointInTimeUniverse(panel, master, cal, adv_window=20, brand_map=brand_map)
    try:
        fconfig = FeatureConfig.from_yaml(config_path("features"))
    except Exception as exc:
        raise ValueError(f"feature config missing: {exc!r}") from exc
    builder = FeatureBuilder(cal, fconfig)
    try:
        index_path = paths.silver("index_daily")
        if index_path.exists():
            import polars as _pl

            index_panel = _pl.read_parquet(index_path)
            breadth_panel = _pl.DataFrame({"date": [], "breadth_ma20": []})
            regimes = builder.build_regime_series(index_panel, breadth_panel, cal.sessions(start, end))
        else:
            regimes = None
    except Exception:
        regimes = None
    execution = NextOpenExecution(cal)
    engine = BacktestEngine(cal, universe, builder, execution) if regimes is None else BacktestEngine(cal, universe, builder, execution, regimes=regimes)
    simulator = TournamentSimulator(engine, cal)
    try:
        with open(config_path("ml"), encoding="utf-8") as f:
            _ml_raw = _yaml.safe_load(f) or {}
        _ml = _ml_raw.get("ml", _ml_raw) if isinstance(_ml_raw, dict) else {}
        _ct = _ml.get("champion_tail", {}) if isinstance(_ml, dict) else {}
        _ml_candidate_mode = _ct.get("candidate_mode", None)
        if _ml_candidate_mode not in {"p27_matched_2x", "executable_hurdle", "adaptive_specialists"}:  # pragma: no cover - CLI validation
            raise ValueError("invalid candidate_mode")  # pragma: no cover - CLI validation
        feature_key = "executable_feature_columns" if _ml_candidate_mode in ("executable_hurdle", "adaptive_specialists") else "feature_columns"  # pragma: no cover - config branch
        feature_columns = tuple(_ct.get(feature_key, ()))  # pragma: no cover - config branch
        label_horizon = int(_ct.get("label_horizon", 36))
        embargo = int(_ct.get("embargo", 36))
        purge = int(_ct.get("purge", 36))
        n_folds = int(_ct.get("n_folds", 3))
        seed = int(_ct.get("seed", 20260831))
        num_leaves = int(_ct.get("num_leaves", 8))
        max_depth = int(_ct.get("max_depth", 4))
        min_leaf = int(_ct.get("min_data_in_leaf", 100))
        min_train_sessions = int(_ct.get("min_train_sessions", 252))
        _costs_raw = _ct.get("costs", {})
        if not isinstance(_costs_raw, dict):
            raise ValueError("ml.champion_tail.costs must be a mapping")
        costs = CostConfig(
            commission_bps=float(_costs_raw["commission_bps"]),
            slippage_bps=float(_costs_raw["slippage_bps"]),
            spread_bps=float(_costs_raw.get("spread_bps", 0.0)),
            tax_bps=float(_costs_raw.get("tax_bps", 0.0)),
        )
    except Exception as exc:
        raise ValueError(f"champion_tail ml config missing: {exc!r}") from exc
    if not feature_columns or len(feature_columns) > 25:
        raise ValueError("ml.champion_tail.feature_columns must have 1..25 columns")
    if label_horizon != 36 or embargo < 36 or purge < 36 or n_folds < 2:
        raise ValueError("ml.champion_tail must satisfy horizon=36 embargo>=36 purge>=36 n_folds>=2")
    if num_leaves > 8 or max_depth > 4 or min_leaf < 100:
        raise ValueError("ml.champion_tail exceeds shallow capacity")
    if min_train_sessions < 1:
        raise ValueError("ml.champion_tail.min_train_sessions must be >=1")
    total_bps = float(costs.commission_bps or 0.0) + float(costs.slippage_bps or 0.0) + float(costs.spread_bps or 0.0) + float(costs.tax_bps or 0.0)
    cost_rate = total_bps / 10000.0
    dataset_config = ChampionDatasetConfig(
        feature_columns=feature_columns,
        label_horizon=label_horizon,
        entry_cost_rate=cost_rate,
        exit_cost_rate=cost_rate,
    )
    try:
        with open(config_path("strategies"), encoding="utf-8") as f:
            _s_raw = _yaml.safe_load(f) or {}
        _port = _s_raw.get("portfolio", {}) if isinstance(_s_raw, dict) else {}
        _ctc = _port.get("champion_tail", {}) if isinstance(_port, dict) else {}
        max_single = float(_ctc.get("max_single_weight", 0.80))
        max_gross = float(_ctc.get("max_effective_gross", 1.60))
        min_cash = float(_ctc.get("min_cash", 0.05))
        abs_cash = bool(_ctc.get("absolute_momentum_cash", True))
    except Exception as exc:
        raise ValueError(f"portfolio.champion_tail missing: {exc!r}") from exc
    if not (max_single <= 0.80 + 1e-12 and max_gross <= 1.60 + 1e-12 and min_cash >= 0.05 - 1e-12):
        raise ValueError("portfolio.champion_tail violates single<=0.80 gross<=1.60 cash>=0.05")
    policy_config = ChampionPolicyConfig(
        max_single_weight=max_single, max_effective_gross=max_gross, min_cash=min_cash, absolute_momentum_cash=abs_cash,
    )
    objective_config = ChampionshipObjectiveConfig.from_yaml(config_path("gates"), config_path("portfolio"))
    from src.portfolio.sizing import SizingScheme

    backtest_config = BacktestConfig(start=start, end=end, capital=1_000_000_000.0, scheme=SizingScheme.TOP1, k=1, filters=filt, costs=costs)
    from src.strategies.registry import STRATEGIES as BASELINES

    def _p27_factory() -> object:
        return BASELINES["sticky.mom60_raw"]()

    # candidate_mode=str(ml['champion_tail']['candidate_mode']) passed to ChampionResearchRuntime after exact value validation
    _validated_candidate_mode = str(_ml['champion_tail']['candidate_mode'])
    if _validated_candidate_mode not in {'p27_matched_2x', 'executable_hurdle', 'adaptive_specialists'}:  # pragma: no cover - CLI validation
        raise ValueError("invalid candidate_mode")  # pragma: no cover - CLI validation
    if _candidate_mode_arg != _validated_candidate_mode:
        raise ValueError("invalid candidate_mode: args/ml mismatch")
    runtime = ChampionResearchRuntime(
        engine=engine,
        simulator=simulator,
        panel=panel,
        backtest_config=backtest_config,
        dataset_config=dataset_config,
        objective_config=objective_config,
        policy_config=policy_config,
        p27_factory=_p27_factory,  # type: ignore[arg-type]
        min_train_sessions=min_train_sessions,
        n_folds=n_folds,
        embargo_sessions=embargo,
        purge_sessions=purge,
        ranker_seed=seed,
        ranker_num_leaves=num_leaves,
        ranker_max_depth=max_depth,
        ranker_min_data_in_leaf=min_leaf,
        candidate_mode=_validated_candidate_mode,
    )
    return {"runtime": runtime}


def cmd_champion_research(args: argparse.Namespace) -> int:
    if getattr(args, "candidate_mode", None) == "capacity_frontier":
        try:
            from pathlib import Path

            run_frontier_from_paths(data_root=Path(args.data_root), start=date.fromisoformat(args.start), end=date.fromisoformat(args.end), output=Path(args.output))
        except (ValueError, OSError) as exc:
            logger.error(f"[SYS] champion-research status=fail error={exc!r}")
            return 1
        return 0
    try:
        kwargs: Any = _build_champion_research_inputs(args)
    except ValueError as exc:
        logger.error(f"[SYS] champion-research status=fail error={exc!r}")
        return 1
    except Exception as exc:
        logger.error(f"[SYS] champion-research status=fail error={exc!r}")
        return 1
    try:
        result = run_champion_walk_forward(**kwargs)
    except ValueError as exc:
        logger.error(f"[SYS] champion-research status=fail error={exc!r}")
        return 1
    except Exception as exc:
        logger.error(f"[SYS] champion-research status=fail error={exc!r}")
        return 1
    status = str(getattr(result, "status", "RESEARCH_ONLY"))
    logger.info(f"[DATA] champion-research status={status}")
    logger.info(f"[EVAL] champion-research status={status}")
    try:
        writer = getattr(result, "write", None)
        if callable(writer):
            writer("docs/results/champion_promotion.json")
    except Exception:
        pass
    return 0
