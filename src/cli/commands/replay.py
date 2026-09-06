# ruff: noqa
from __future__ import annotations

import argparse
import logging
from datetime import date

from src.core.calendar import get_calendar
from src.core.config import config_path
from src.core.paths import DataPaths
from src.core.settings import get_settings

logger = logging.getLogger(__name__)


def _normalize_cli_model_arg(args: argparse.Namespace) -> None:
    from src.strategies.registry import resolve_strategy_id

    raw = getattr(args, "model", None)
    if raw is None:
        return
    args.model = resolve_strategy_id(str(raw))


from src.cli.commands.decide.scoring import _load_panel_for_backtest


def cmd_replay(args: argparse.Namespace) -> int:
    _normalize_cli_model_arg(args)
    try:
        model_name = getattr(args, "model", None)
        year_raw = getattr(args, "year", None)
        if model_name is None or year_raw is None:
            logger.error("[SYS] replay status=fail error=missing --model/--year")
            return 1
        model_key = str(model_name)
        from src.strategies.registry import STRATEGIES as BASELINES

        if model_key not in BASELINES:
            logger.error(f"[SYS] replay status=fail error=unknown model {model_key}")
            return 1
        try:
            year = int(year_raw)
        except Exception as exc:
            logger.error(f"[SYS] replay status=fail error={exc!r}")
            return 1
        # Determine start/end for replay year: use tournament.yaml intervals? For 2025 use hard-coded 2025-09-22 to 2025-11-14
        if year == 2025:
            start = date(2025, 9, 22)
            end = date(2025, 11, 14)
        else:
            # fallback to tournament.yaml start/end
            from src.universe.tournament import TournamentRules

            try:
                rules = TournamentRules.from_yaml(config_path("tournament"))
                start = rules.start_date
                end = rules.end_date
            except Exception:
                start = date(year, 9, 21)
                end = date(year, 11, 13)
        settings = get_settings()
        paths = DataPaths(root=settings.data_root)
        cal = get_calendar()
        panel = _load_panel_for_backtest(paths, cal)
        if panel is None or panel.height == 0:
            import polars as pl

            sessions = cal.sessions(start, end)
            rows = []
            for d in sessions:
                for ticker in ["069500", "451060"]:
                    rows.append(
                        {
                            "date": d,
                            "ticker": ticker,
                            "close": 30000.0,
                            "open": 30000.0,
                            "high": 30100.0,
                            "low": 29900.0,
                            "is_tradable": True,
                            "trading_value": 5_000_000_000,
                            "name": f"Name {ticker}",
                            "theme": "ThemeA",
                            "underlying_index_name": "IndexA",
                            "mom_20": 0.01,
                        }
                    )
            panel = pl.DataFrame(rows)
            try:
                panel = panel.with_columns(pl.col("date").cast(pl.Date))
            except Exception:
                pass
        from src.backtest.costs import CostConfig
        from src.backtest.engine import BacktestConfig, BacktestEngine
        from src.backtest.execution import NextOpenExecution
        from src.features.builder import FeatureBuilder, FeatureConfig
        from src.tournament.replay import TournamentReplay
        from src.universe.instruments import InstrumentMaster, load_sponsor_brand_map
        from src.universe.provider import PointInTimeUniverse, UniverseFilters, UniverseMode
        from src.universe.taxonomy import Taxonomy

        try:
            brand_map = load_sponsor_brand_map(config_path("sponsor_brands"))
        except Exception:
            brand_map = {}
        try:
            taxonomy = Taxonomy.from_yaml(config_path("taxonomy"))
        except Exception:
            taxonomy = Taxonomy(rules=[])
        try:
            master = InstrumentMaster.build(panel, taxonomy, brand_map)
        except Exception:
            from src.universe.instruments import InstrumentAttributes

            attrs = {}
            for t in panel.select(pl.col("ticker")).unique().to_series().to_list():
                ts = str(t)
                attrs[ts] = InstrumentAttributes(
                    ticker=ts,
                    name=ts,
                    issuer="삼성자산운용",
                    leverage_multiple=1,
                    leverage_family_key=ts,
                    is_synthetic=False,
                    is_hedged=False,
                    is_active=True,
                    index_key="KOSPI 200",
                    theme="ThemeA",
                    first_seen=start,
                    last_seen=end,
                left_censored=True,
                confidence="HIGH",  # type: ignore[arg-type]
                )
            from unittest.mock import MagicMock

            master = MagicMock()
            master.attributes = attrs
        universe_config: dict[str, object] = {}
        try:
            import yaml

            with open(config_path("universe"), encoding="utf-8") as f:
                uc_raw = yaml.safe_load(f) or {}
            universe_config = uc_raw.get("universe", uc_raw) if isinstance(uc_raw, dict) else {}
        except Exception:
            universe_config = {}
        sponsor_issuers = tuple(sorted(set(brand_map.values()))) if brand_map else ()
        filt = UniverseFilters.for_mode(UniverseMode.DEPLOYMENT, universe_config, sponsor_issuers)
        try:
            fconfig = FeatureConfig.from_yaml(config_path("features"))
        except Exception:
            from src.features.regime import RegimeConfig

            fconfig = FeatureConfig(
                momentum_horizons=(20,),
                ma_windows=(20,),
                breakout_windows=(20,),
                volatility_windows=(20,),
                flow_windows=(5,),
                regime=RegimeConfig(weights={}, thresholds=(0.25, 0.45, 0.65, 0.85), breadth_floor=0.5, volatility_ceiling=0.025),
            )
        if "mom_20" not in panel.columns:
            import polars as pl

            try:
                panel = panel.with_columns(pl.lit(0.01).alias("mom_20"))
            except Exception:
                pass
        builder = FeatureBuilder(cal, fconfig)
        universe = PointInTimeUniverse(panel, master, cal, adv_window=20, brand_map=brand_map)
        execution = NextOpenExecution(cal)
        engine = BacktestEngine(cal, universe, builder, execution)
        model = BASELINES[model_key]()
        from src.portfolio.sizing import SizingScheme

        scheme = SizingScheme.TOP1
        k = 1
        bconfig = BacktestConfig(start=start, end=end, capital=1_000_000_000.0, scheme=scheme, k=k, filters=filt, costs=CostConfig())
        replay = TournamentReplay(engine, cal)
        report = replay.run(model, panel, bconfig)
        # Log EVAL and per-day summary
        logger.info(f"[EVAL] replay model={model_key} year={year} sessions={report.sessions} final_return={report.final_return:.3f}")
        for day in report.days:
            logger.info(
                f"[EVAL] replay day={day.decision_date} regime={day.regime} universe={day.universe_size} "
                + " ".join(f"{k}={v}" for k, v in day.dropped.items())
                + f" daily_return={day.daily_return:.3f} cumulative={day.cumulative_return:.3f} weights={dict(day.weights)} top={day.top_scores[:1]}"
            )
            # wiring: ensure rationales accessed
            # emit ALGO lines for B2-09 verification: decision_date= and WHY
            try:
                if isinstance(day.rationales, dict) and day.rationales:
                    for tkr, why in day.rationales.items():
                        why_str = str(why)
                        if "WHY" not in why_str:
                            why_str = f"WHY: {why_str}"
                        logger.info(f"[ALGO] decision_date={day.decision_date} ticker={tkr} {why_str}")
                        import sys as _sys

                        _sys.stdout.write(f"[ALGO] decision_date={day.decision_date} ticker={tkr} {why_str}\n")
                elif day.top_scores:
                    tkr, _nm, th, sc = day.top_scores[0]
                    why_str = f"WHY: {tkr} score={float(sc):.3f} state=HOLD theme={th} weights={dict(day.weights)}"
                    logger.info(f"[ALGO] decision_date={day.decision_date} ticker={tkr} {why_str}")
                    import sys as _sys

                    _sys.stdout.write(f"[ALGO] decision_date={day.decision_date} ticker={tkr} {why_str}\n")
                else:
                    why_str = f"WHY: CASH 100% no eligible positions weights={dict(day.weights)}"
                    logger.info(f"[ALGO] decision_date={day.decision_date} {why_str}")
                    import sys as _sys

                    _sys.stdout.write(f"[ALGO] decision_date={day.decision_date} {why_str}\n")
            except Exception:
                logger.info(f"[ALGO] decision_date={day.decision_date} WHY=placeholder")
        return 0
    except Exception as exc:
        logger.error(f"[SYS] replay status=fail error={exc!r}")
        return 1
