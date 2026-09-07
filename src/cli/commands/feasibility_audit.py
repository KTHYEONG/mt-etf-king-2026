"""Feasibility-audit CLI (Phase 0 measurement, horizon locked to 36)."""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import polars as pl
import yaml

from src.core.calendar import get_calendar
from src.core.config import config_path
from src.core.paths import DataPaths
from src.research.decision_population import PopulationError
from src.research.executable_oracle import build_deployment_oracle_opens
from src.research.feasibility_audit import run_feasibility_audit
from src.research.feasibility_metrics import AlignError
from src.research.p38_promotion import load_p38_terminal_by_entry_date
from src.universe.instruments import load_sponsor_brand_map
from src.universe.manifest_validate import ManifestValidationError

logger = logging.getLogger(__name__)


def _load_strategy_windows(run: str | None) -> pl.DataFrame | None:
    return None if run is None else pl.read_parquet(Path(str(run)) / "windows.parquet")


def _load_sponsor_issuers() -> frozenset[str]:
    raw = yaml.safe_load(config_path("tournament").read_text(encoding="utf-8")) or {}
    tournament = raw.get("tournament", raw) if isinstance(raw, dict) else {}
    sponsors = tournament.get("sponsors", {}) if isinstance(tournament, dict) else {}
    asset_managers = sponsors.get("asset_managers", []) if isinstance(sponsors, dict) else []
    return frozenset(str(item) for item in asset_managers)


def cmd_feasibility_audit(args: argparse.Namespace) -> int:
    horizon = int(getattr(args, "horizon", 36) or 36)
    if horizon != 36:
        return 1
    try:
        start = date.fromisoformat(str(getattr(args, "start", "")))
        end = date.fromisoformat(str(getattr(args, "end", "")))
        paths = DataPaths(root=Path(str(getattr(args, "data_root", "data"))))
        panel = pl.read_parquet(paths.silver("etf_daily"))
        if panel.height == 0:
            raise PopulationError("fail-closed: empty panel")
        calendar_sessions = get_calendar().sessions(start, end)
        brand_map = load_sponsor_brand_map(config_path("sponsor_brands"))
        oracle_opens = build_deployment_oracle_opens(
            panel,
            sessions=resolve_session_grid_sessions(calendar_sessions, panel),
            sponsor_issuers=_load_sponsor_issuers(),
            brand_map=brand_map,
        )
        strategy_windows: dict[str, pl.DataFrame] = {}
        p27 = _load_strategy_windows(getattr(args, "p27_run", None))
        if p27 is not None:
            strategy_windows["sticky.mom60_raw"] = p27
        b1 = _load_strategy_windows(getattr(args, "b1_run", None))
        if b1 is not None:
            strategy_windows["baseline.mom20_top1"] = b1
        p38 = load_p38_terminal_by_entry_date(
            promotion_path=Path(str(p38_path)) if (p38_path := getattr(args, "p38_promotion", None)) else None,
            calendar_sessions=calendar_sessions,
            panel=panel,
            horizon=horizon,
        )
        output_dir = Path(str(getattr(args, "output", "docs/research") or "docs/research"))
        run_feasibility_audit(
            calendar_sessions=calendar_sessions,
            panel=panel,
            horizon=horizon,
            output_dir=output_dir,
            oracle_opens=oracle_opens,
            strategy_windows=strategy_windows,
            comparator_gross={
                "sticky.mom60_raw": {"effective_gross_max": 1.9, "gross_violation_count": 0},
                "baseline.mom20_top1": {"effective_gross_max": 2.0, "gross_violation_count": 1428},
            },
            champion_gross_max=1.9,
            p38_terminal_by_start=p38,
            manifest=None,
            enforce_horizon_36=True,
        )
    except (PopulationError, AlignError, ManifestValidationError) as exc:
        logger.error(f"[SYS] feasibility-audit status=fail error={exc!r}")
        return 1
    return 0


def resolve_session_grid_sessions(calendar_sessions: list[date], panel: pl.DataFrame) -> list[date]:
    from src.backtest.session_grid import resolve_session_grid

    return list(resolve_session_grid(calendar_sessions, panel).sessions)
