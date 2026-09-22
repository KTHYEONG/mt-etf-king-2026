"""Daily pipeline orchestration (ingest + normalize + features, optional decide)."""

from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any, Final
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    from src.core.calendar import TradingCalendar

logger = logging.getLogger(__name__)

_INDEX_INGEST_DATASET: Final[str] = "kospi_index"
_INDEX_NORMALIZE_DATASET: Final[str] = "index_daily"
_REGULAR_CLOSE: Final[time] = time(15, 30)


def resolve_pipeline_target_session(
    as_of: date | None = None,
    *,
    now: datetime | None = None,
    calendar: TradingCalendar | None = None,
) -> date | None:
    """Resolve the pipeline target session with KST time-awareness."""
    kst = ZoneInfo("Asia/Seoul")
    now_kst = datetime.now(kst) if now is None else now.astimezone(kst)
    today = now_kst.date()
    if calendar is None:
        from src.core.calendar import get_calendar

        calendar = get_calendar()
    requested = as_of if as_of is not None else today
    if requested > today:
        return None
    if requested < today:
        sessions = calendar.sessions(requested - timedelta(days=21), requested)
        if not sessions:
            return None
        return sessions[-1]
    if now_kst.time() >= _REGULAR_CLOSE and calendar.is_session(today):
        return today
    sessions = calendar.sessions(today - timedelta(days=21), today)
    return sessions[-1 - int(sessions[-1] == today)]


def cmd_daily_refresh(args: argparse.Namespace) -> int:
    """Run the daily batch: ingest, normalize, full-range features, optional decide."""
    from src.cli.commands.data import cmd_ingest, cmd_normalize
    from src.cli.commands.decide import cmd_decide
    from src.cli.commands.features import cmd_features
    from src.cli.constants import CHAMPION_STRATEGY
    from src.core.calendar import get_calendar
    from src.core.paths import DataPaths
    from src.core.settings import get_settings

    dataset = getattr(args, "dataset", None) or "etf_daily"
    as_of_raw = getattr(args, "as_of", None)
    lookback_days = getattr(args, "lookback_days", 10) or 10

    try:
        as_of = None if as_of_raw is None else date.fromisoformat(as_of_raw)
    except ValueError:
        return 1

    target = resolve_pipeline_target_session(as_of=as_of)
    if target is None:
        return 1
    sessions = get_calendar().sessions(target - timedelta(days=21), target)
    if not sessions:
        return 1
    end = sessions[-1]

    ingest_start = end - timedelta(days=lookback_days)
    rc = cmd_ingest(
        argparse.Namespace(
            dataset=dataset,
            start=ingest_start.isoformat(),
            end=end.isoformat(),
            dry_run=False,
        )
    )
    if rc != 0:
        return 1

    rc = cmd_normalize(argparse.Namespace(dataset=dataset, mode="incremental"))
    if rc != 0:
        return 1

    rc = cmd_ingest(argparse.Namespace(dataset=_INDEX_INGEST_DATASET, start=ingest_start.isoformat(), end=end.isoformat(), dry_run=False))
    if rc != 0:
        return 1
    rc = cmd_normalize(argparse.Namespace(dataset=_INDEX_NORMALIZE_DATASET, mode="incremental"))
    if rc != 0:
        return 1

    import polars as pl

    silver_path = DataPaths(root=get_settings().data_root).silver(dataset)
    panel = pl.read_parquet(silver_path) if silver_path.exists() else None
    if panel is None or panel.height == 0:
        return 1
    features_start_raw = panel["date"].min()
    if not isinstance(features_start_raw, date):
        return 1
    features_start = features_start_raw

    rc = cmd_features(
        argparse.Namespace(start=features_start.isoformat(), end=end.isoformat())
    )
    if rc != 0:
        return 1

    if not getattr(args, "decide", False):
        return 0

    output_dir = getattr(args, "output_dir", None) or "results/decide_daily"

    from src.cli.commands.decide.models import _LIVE_STICKY_STRATEGIES, _sticky_live_state_name

    if CHAMPION_STRATEGY in _LIVE_STICKY_STRATEGIES:
        from src.data.panel import BACKTEST_PANEL_COLUMNS, load_backtest_panel
        from src.tournament.live_decision import StateDiscontinuityError, pending_catchup_sessions

        catchup_paths = DataPaths(root=get_settings().data_root)
        state_path = catchup_paths.state(_sticky_live_state_name(CHAMPION_STRATEGY))
        catchup_panel: Any = load_backtest_panel(catchup_paths, columns=BACKTEST_PANEL_COLUMNS)
        try:
            pending = pending_catchup_sessions(state_path, catchup_panel, target_session=end)
        except StateDiscontinuityError as exc:
            logger.error(f"[SYS] catchup status=fail reason=state_discontinuity {exc}")
            return 1
        for k, session in enumerate(pending, start=1):
            rc = cmd_decide(
                argparse.Namespace(
                    model=CHAMPION_STRATEGY,
                    date=session.isoformat(),
                    panel=None,
                    capital=None,
                    output=f"{output_dir}/{session.isoformat()}.json",
                    trace=False,
                )
            )
            logger.info(f"[SYS] catchup model={CHAMPION_STRATEGY} session={session} k={k}/{len(pending)} rc={rc}")
            if rc != 0:
                logger.error(f"[SYS] catchup status=fail session={session}")
                return 1

    output_path = f"{output_dir}/{end.isoformat()}.json"
    rc = cmd_decide(
        argparse.Namespace(
            model=CHAMPION_STRATEGY,
            date=end.isoformat(),
            panel=None,
            capital=None,
            output=output_path,
            trace=False,
        )
    )
    if rc != 0:
        return 1
    return 0
