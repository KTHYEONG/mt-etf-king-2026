"""Contest segment CLI command (daily overnight-long + intraday-reversal card & equity ledger)."""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
from datetime import date
from pathlib import Path

from src.cli.commands.contest import _contest_config, _resolve_target_session
from src.core.atomic_io import atomic_write_text

logger = logging.getLogger(__name__)


def cmd_contest_segment(args: argparse.Namespace) -> int:
    """Build and persist the daily segment card for the signal session and advance our equity ledger.

    Target session: `--session` if given, else the last trading session on or before today (KST). `--our-return`
    (percent, as shown in the contest app at the close of the target session) anchors the ledger and implies a rebuild of
    an existing card.

    Quotes: read `<data_root>/<shadow.quotes_dir>/<YYYYMMDD>.json` for the target session; when absent, fetch the vote
    tickers plus the long and inverse tickers with `fetch_quotes`, in memory only. The quote archive is written only by
    contest-daily.

    Ledger: read `<data_root>/state/<segment.ledger_state_name>.json`, determine the side executed on the target session
    from the previous signal session's card in `segment.output_dir`, advance with `advance_ledger`, and write the state
    atomically only when it changed.

    Writes `<segment.output_dir>/<session>.json` and `<session>.md` atomically. An existing NO_DATA card is rebuilt on
    rerun; other existing cards are skipped unless `--force` or `--our-return` is given (the ledger still never compounds
    twice, by the core idempotency rule).

    Returns 0 on a written or skipped card (including NO_DATA) or when `segment.enabled` is false; 1 on unexpected
    exceptions or persistence failure.
    """
    from src.contest.daily import fetch_quotes, load_quotes
    from src.contest.leaderboard import latest_snapshot_on_or_before
    from src.contest.segment import (
        SegmentSettings,
        advance_ledger,
        build_segment_card,
        compute_standing,
        executed_side_from_card,
        leaderboard_equity,
        ledger_from_dict,
        ledger_to_dict,
        render_segment_markdown,
        segment_to_dict,
    )
    from src.core.calendar import get_calendar, kst_today
    from src.core.paths import DataPaths
    from src.core.settings import get_settings

    try:
        contest = _contest_config()
        seg_cfg = contest.get("segment", {})
        if not isinstance(seg_cfg, dict) or not seg_cfg.get("enabled", False):
            logger.info("[SYS] contest_segment status=disabled")
            return 0

        calendar = get_calendar()
        raw_session = getattr(args, "session", None)
        session: date | None = (
            date.fromisoformat(str(raw_session)) if raw_session else _resolve_target_session(calendar, kst_today())
        )
        if session is None:
            raise ValueError("no trading session on or before today")

        settings = SegmentSettings.load(contest)
        output_dir = settings.output_dir
        card_path = output_dir / f"{session.isoformat()}.json"

        our_return = getattr(args, "our_return", None)
        force = getattr(args, "force", False)

        if card_path.is_file() and not force and our_return is None:
            with contextlib.suppress(OSError, ValueError):
                existing_text = card_path.read_text(encoding="utf-8")
                if '"action": "NO_DATA"' not in existing_text:
                    logger.info(f"[PORTFOLIO] contest_segment session={session.isoformat()} status=exists_skip")
                    return 0

        data_root = Path(get_settings().data_root)
        shadow = contest.get("shadow", {})
        if not isinstance(shadow, dict):
            shadow = {}
        quotes_root = data_root / str(shadow.get("quotes_dir", "contest/quotes"))
        yahoo_suffix = str(shadow.get("yahoo_suffix", ".KS"))
        timeout_s = float(shadow.get("timeout_s", 20))

        leaderboard = contest.get("leaderboard", {})
        if not isinstance(leaderboard, dict):
            leaderboard = {}
        archive_root = data_root / str(leaderboard.get("archive_dir", "contest/leaderboard"))

        nickname = str(contest.get("nickname", ""))
        ledger_path = DataPaths(root=data_root).state(settings.ledger_state_name)

        bars = load_quotes(quotes_root, session)
        if bars is None:
            tickers = sorted({settings.long_ticker, settings.inverse_ticker, *(t for _, t in settings.vote)})
            try:
                bars = fetch_quotes(tickers, session, yahoo_suffix, timeout_s)
            except Exception as exc:
                logger.info(f"[DATA] contest_quotes session={session.isoformat()} status=fetch_fail error={exc!r}")
                bars = {}

        snapshot = latest_snapshot_on_or_before(archive_root, session)
        lb_eq = leaderboard_equity(snapshot, session, nickname)

        prev_ledger = None
        if ledger_path.is_file():
            with contextlib.suppress(OSError, ValueError):
                prev_ledger = ledger_from_dict(json.loads(ledger_path.read_text(encoding="utf-8")))
            if prev_ledger is None:
                logger.warning(f"[PORTFOLIO] contest_segment ledger_unreadable path={ledger_path}")

        try:
            prev_session = calendar.previous_session(session)
        except Exception:
            prev_session = None

        executed_side = None
        if prev_session is not None:
            prev_card_path = output_dir / f"{prev_session.isoformat()}.json"
            if prev_card_path.is_file():
                try:
                    prev_card_data = json.loads(prev_card_path.read_text(encoding="utf-8"))
                    executed_side = executed_side_from_card(prev_card_data)
                except Exception:
                    executed_side = None

        end_date_val = contest.get("end_date")
        end_date = date.fromisoformat(end_date_val) if isinstance(end_date_val, str) else end_date_val
        session_is_final = session == end_date

        user_equity = 1.0 + float(our_return) / 100.0 if our_return is not None else None

        new_ledger, ledger_warnings = advance_ledger(
            prev=prev_ledger,
            session=session,
            prev_session=prev_session if prev_session is not None else date.min,
            executed_side=executed_side,
            bars=bars,
            session_is_final=session_is_final,
            leaderboard_equity=lb_eq,
            user_equity=user_equity,
            settings=settings,
        )

        standing, standing_warnings = compute_standing(
            snapshot=snapshot,
            session=session,
            nickname=nickname,
            ledger=new_ledger,
            prize_rank=settings.prize_rank,
        )

        card = build_segment_card(
            session=session,
            calendar=calendar,
            bars=bars,
            config=contest,
            standing=standing,
            extra_warnings=(*ledger_warnings, *standing_warnings),
        )
    except Exception as exc:
        logger.error(f"[SYS] contest_segment status=fail error={exc!r}")
        return 1

    try:
        if new_ledger is not None and new_ledger != prev_ledger:
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_text(ledger_path, json.dumps(ledger_to_dict(new_ledger), indent=2) + "\n")
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(card_path, json.dumps(segment_to_dict(card), ensure_ascii=False, indent=2) + "\n")
        atomic_write_text(output_dir / f"{session.isoformat()}.md", render_segment_markdown(card, settings))
    except Exception as exc:
        logger.error(f"[SYS] contest_segment status=fail reason=persist error={exc!r}")
        return 1

    side_str = card.intraday_side.value if card.intraday_side is not None else "none"
    down_votes_str = f"{card.down_votes}/{len(card.vote_changes)}"
    equity_str = f"{card.standing.our_equity:.4f}" if card.standing.our_equity is not None else "none"
    source_str = card.standing.our_source.value if card.standing.our_source is not None else "none"
    req_str = f"{card.standing.required_multiple:.3f}" if card.standing.required_multiple is not None else "none"
    warn_str = ",".join(card.warnings) if card.warnings else "none"

    logger.info(
        f"[PORTFOLIO] contest_segment session={session.isoformat()} action={card.action.value} "
        f"side={side_str} down_votes={down_votes_str} equity={equity_str} "
        f"source={source_str} required={req_str} warnings={warn_str}"
    )
    return 0
