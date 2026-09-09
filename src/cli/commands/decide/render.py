"""Decide dashboard rendering (P4 decomposition of src/cli/_impl.py)."""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from src.reporting.dashboard import DailyDecision, build_rationale, render_dashboard, write_decision_artifact

logger = logging.getLogger(__name__)


def render_decision(
    *,
    weights: dict[str, float],
    decision_weights: object,
    scores: dict[str, float],
    decision_date: date,
    args: argparse.Namespace,
    peak_is_locked: bool,
    house_money_is_locked: bool,
    order_estimates: Mapping[str, object] | None = None,
) -> int:
    """Format rationales, render the dashboard, write artifacts. Returns the exit code."""
    # use rationales from policy if available
    rationales: dict[str, str] = {}
    try:
        rationale_src = getattr(decision_weights, "rationale", None)
        if rationale_src:
            rationales = dict(rationale_src)
    except Exception:
        rationales = {}
    if not rationales:
        for ticker, w in weights.items():
            pos = {"ticker": ticker, "weight": w, "state": "HOLD", "theme": "ThemeA"}
            rationales[ticker] = build_rationale(pos)
    # fail-closed: missing rationale or eligible 0 -> exit 1 already handled
    # handle peak lock cash case: inject CASH rationale if locked (P22 live 50%, keep 40% string for legacy wiring)
    if not weights and house_money_is_locked:
        rationales = {"CASH": "WHY: house_money late-lock remaining<=K state=CASH"}
    elif not weights and peak_is_locked:
        rationales = {"CASH": "WHY: peak_lock 50% triggered state=CASH"}
    # ensure state= present
    for ticker in list(rationales.keys()):
        if "state=" not in rationales[ticker]:
            rationales[ticker] = rationales[ticker] + " state=HOLD"
        if "WHY" not in rationales[ticker]:
            rationales[ticker] = f"WHY: {rationales[ticker]}"
    if (not weights and not peak_is_locked) or not rationales:
        logger.error("[SYS] decide status=fail error=eligible==0 weights empty")
        return 1
    if peak_is_locked:
        # enforce cash weights
        weights = {}
    daily = DailyDecision(decision_date=decision_date, weights=weights, rationales=rationales)
    out = render_dashboard(daily)
    if order_estimates:
        est_lines = ["추정 주문 수량 (decision_date 종가 기준, 실제 체결가와 다를 수 있음)"]
        for _tkr, _est in order_estimates.items():
            _shares = getattr(_est, "est_shares", None)
            _krw = getattr(_est, "est_krw", None)
            est_lines.append(f"{_tkr}: {_shares}주 {_krw}원")
        out = out + "\n" + "\n".join(est_lines)
    sys.stdout.write(out + "\n")
    logger.info(out)
    # also log ALGO style for uniformity
    for tkr, why in rationales.items():
        logger.info(f"[ALGO] decision_date={decision_date} ticker={tkr} WHY={why}")
        sys.stdout.write(f"[ALGO] decision_date={decision_date} ticker={tkr} WHY={why}\n")
    # write decision artifact
    try:
        art_name = f"{decision_date.strftime('%Y%m%d')}_decision.json"
        art_path = Path("data/state/decisions") / art_name
        out_p = getattr(args, "output", None)
        if out_p:
            art_path = Path(str(out_p))
        write_decision_artifact(daily, art_path, order_estimates=order_estimates)
    except Exception as e:
        logger.warning(f"[SYS] write_decision_artifact failed {e!r}")
    # trace artifacts for decide
    if getattr(args, "trace", False):
        try:
            from src.reporting.trace_store import write_trace_artifacts  # noqa: I001

            import polars as _pl_decide  # noqa: I001

            dest_decide = art_path.parent / (art_path.stem + "_trace")
            # minimal sessions/candidates for decide trace
            try:
                sess_df = _pl_decide.DataFrame(
                    {
                        "decision_date": [decision_date],
                        "n_universe": [len(scores)],
                        "n_scores": [len(scores)],
                        "n_selected": [len(weights)],
                        "n_fills": [0],
                        "n_unfilled": [0],
                        "n_candidates_written": [len(scores)],
                        "n_candidates_truncated": [0],
                        "dropped_existence": [0],
                        "dropped_price": [0],
                        "dropped_history": [0],
                        "dropped_sponsor": [0],
                        "dropped_liquidity": [0],
                        "dropped_eligibility": [0],
                        "regime": [""],
                        "equity": [0.0],
                    }
                )
                with contextlib.suppress(Exception):
                    sess_df = sess_df.with_columns(_pl_decide.col("decision_date").cast(_pl_decide.Date))
            except Exception:
                sess_df = _pl_decide.DataFrame({"decision_date": [], "n_universe": []})
            try:
                cand_rows = []
                sorted_sc = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
                for idx, (tkr, sc) in enumerate(sorted_sc, start=1):
                    cand_rows.append(
                        {
                            "decision_date": decision_date,
                            "ticker": tkr,
                            "score": float(sc),
                            "rank": idx,
                            "selected": tkr in weights,
                            "reject_reason": "" if tkr in weights else "TOPK_CUT",
                            "weight_raw": float(sc),
                            "weight_target": float(weights.get(tkr, 0.0)),
                            "weight_after_adv": float(weights.get(tkr, 0.0)),
                            "weight_fill": 0.0,
                        }
                    )
                    cand_df = _pl_decide.DataFrame(cand_rows) if cand_rows else _pl_decide.DataFrame(
                        {"decision_date": [], "ticker": [], "score": [], "rank": [], "selected": [],
                            "reject_reason": [], "weight_raw": [], "weight_target": [], "weight_after_adv": [],
                            "weight_fill": []}
                    )
                    with contextlib.suppress(Exception):
                        cand_df = cand_df.with_columns(_pl_decide.col("decision_date").cast(_pl_decide.Date))
            except Exception:
                cand_df = _pl_decide.DataFrame({"decision_date": [], "ticker": []})
            try:
                write_trace_artifacts(dest_decide, sessions=sess_df, candidates=cand_df, gates=[])
            except OSError as _oe_dec:
                logger.warning(f"[SYS] trace write failed {_oe_dec!r}")
            except Exception as _e_dec:
                logger.warning(f"[SYS] trace write failed {_e_dec!r}")
        except Exception as _e_outer_dec:
            logger.warning(f"[SYS] trace write failed {_e_outer_dec!r}")
    return 0


__all__ = ["render_decision"]
