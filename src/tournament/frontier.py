# ruff: noqa
# mypy: ignore-errors
from __future__ import annotations

import hashlib
import time
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import polars as pl

from src.backtest.costs import CostConfig, CostModel
from src.core.calendar import get_calendar
from src.execution.cash_accounting import cash_order_transition
from src.execution.ledger_state import PortfolioLedgerState
from src.portfolio.intent import CASH_INTENT, HOLD_INTENT, PortfolioIntent
from src.portfolio.opportunity_frontier import capacity_frontier_scores
from src.universe.instruments import resolve_leverage


def run_frontier_research(
    panel: pl.DataFrame,
    sessions: Sequence[date],
    *,
    capital: float = 1_000_000_000.0,
    participation: float = 0.01,
    cost_bps: float = 8.0,
    allow_leverage: bool = True,
) -> pl.DataFrame:
    grid = list(sessions)
    if grid != sorted(set(grid)):
        raise ValueError("sessions must be sorted unique")
    if panel.group_by(["date", "ticker"]).len().filter(pl.col("len") > 1).height > 0:
        raise ValueError("duplicate panel keys")
    panel_dates = set(panel.select("date").to_series().to_list())
    if panel_dates != set(grid):
        raise ValueError("panel dates must match sessions")
    idx_of = {d: i for i, d in enumerate(grid)}
    opens: list[dict[str, float]] = [{} for _ in grid]
    closes: list[dict[str, float]] = [{} for _ in grid]
    tvs: list[dict[str, float]] = [{} for _ in grid]
    names: list[dict[str, str]] = [{} for _ in grid]
    idx_names: list[dict[str, str]] = [{} for _ in grid]
    first_seen: dict[str, int] = {}
    for row in panel.iter_rows(named=True):
        d = row["date"]
        i = idx_of[d]
        t = str(row["ticker"])
        first_seen[t] = min(first_seen.get(t, i), i)
        o, c, tv = row.get("open"), row.get("close"), row.get("trading_value")
        if isinstance(o, (int, float)) and o == o and o > 0 and bool(row.get("is_tradable")):
            opens[i][t] = float(o)
        if isinstance(c, (int, float)) and c == c and c > 0 and bool(row.get("is_tradable")):
            closes[i][t] = float(c)
        if isinstance(tv, (int, float)) and tv == tv:
            tvs[i][t] = float(tv)
        names[i][t] = str(row.get("name") or "")
        idx_names[i][t] = str(row.get("underlying_index_name") or "")
    cost_model = CostModel(CostConfig(commission_bps=float(cost_bps)))
    limits = (0.8, 1.6, 0.05)
    n = len(grid)
    rows: list[dict[str, object]] = []
    for s in range(n - 35):
        window = grid[s : s + 36]
        state = PortfolioLedgerState(cash=float(capital), shares={})
        base = PortfolioLedgerState(cash=float(capital), shares={})
        peak = float(capital)
        mdd = 0.0
        giveback = 0.0
        cash_min = float(capital)
        missing = 0
        viols = 0
        first_fill = None
        for pos in range(36):
            idx = s + pos
            dec = idx - 1 if pos > 0 else (s - 1 if s > 0 else None)
            if dec is None or dec < 0:
                res = cash_order_transition(prior_state=state, intent=HOLD_INTENT, decision_date=window[0], prev_closes=closes[idx], opens=opens[idx], closes=closes[idx], cost_model=cost_model, adv_by_ticker={}, max_order_to_adv=float(participation), exposure_limits=limits, leverage_multiples=dict.fromkeys(state.shares, 1) or {"__none__": 1}, execution=None, panel=None, lot_size=1)
                state = res.state
                bres = cash_order_transition(prior_state=base, intent=HOLD_INTENT, decision_date=window[0], prev_closes=closes[idx], opens=opens[idx], closes=closes[idx], cost_model=cost_model, adv_by_ticker={}, max_order_to_adv=float(participation), exposure_limits=limits, leverage_multiples=dict.fromkeys(base.shares, 1) or {"__none__": 1}, execution=None, panel=None, lot_size=1)
                base = bres.state
                eq = state.equity_at_prices(closes[idx]) if closes[idx] else float(state.cash)
                peak = max(peak, eq)
                mdd = min(mdd, eq / peak - 1 if peak else 0.0)
                giveback = max(giveback, (peak - eq) / float(capital))
                cash_min = min(cash_min, float(state.cash))
                continue
            ddate = grid[dec]
            eq_now = state.equity_at_prices(closes[dec]) if closes[dec] else float(state.cash)
            beq_now = base.equity_at_prices(closes[dec]) if closes[dec] else float(base.cash)
            r = 36 - pos
            frame_rows: list[dict[str, object]] = []
            adv_map: dict[str, float] = {}
            mult_map: dict[str, int] = {}
            for t in set(closes[dec]) | set(opens[idx]):
                c0 = closes[dec].get(t)
                c60 = closes[dec - 60].get(t) if dec - 60 >= 0 else None
                mom = (float(c0) / float(c60) - 1.0) if c0 and c60 and c60 > 0 else None
                adv = sum(tvs[dec - k].get(t, 0.0) for k in range(20)) / 20.0 if dec >= 19 else None
                nm = names[dec].get(t, "")
                mult, conf = resolve_leverage(nm)
                if mult != 1 and not allow_leverage:
                    continue
                if adv is not None:
                    adv_map[t] = float(adv)
                    mult_map[t] = int(mult)
                frame_rows.append({"ticker": t, "mom_60": mom if mom is not None else float("nan"), "adv20": adv if adv is not None else float("nan"), "leverage_multiple": int(mult), "confidence": str(conf), "eligible": True})
            snapshot = pl.DataFrame(frame_rows, schema={"ticker": pl.String, "mom_60": pl.Float64, "adv20": pl.Float64, "leverage_multiple": pl.Int64, "confidence": pl.String, "eligible": pl.Boolean}) if frame_rows else pl.DataFrame(schema={"ticker": pl.String, "mom_60": pl.Float64, "adv20": pl.Float64, "leverage_multiple": pl.Int64, "confidence": pl.String, "eligible": pl.Boolean})
            scores = capacity_frontier_scores(snapshot, capital=float(eq_now), remaining_sessions=int(r), participation=float(participation), cost_bps=float(cost_bps)) if frame_rows else {}
            chall_intent = PortfolioIntent(kind="target", weights={sorted(scores)[0]: 0.8}) if scores else CASH_INTENT
            if scores:
                best = sorted(scores, key=lambda k: (-scores[k], k))[0]
                chall_intent = PortfolioIntent(kind="target", weights={best: 0.8})
            cand_mults = {t: mult_map.get(t, 1) for t in set(state.shares) | set(getattr(chall_intent, "weights", {}))}
            res = cash_order_transition(prior_state=state, intent=chall_intent, decision_date=ddate, prev_closes=closes[dec], opens=opens[idx], closes=closes[idx], cost_model=cost_model, adv_by_ticker=adv_map, max_order_to_adv=float(participation), exposure_limits=limits, leverage_multiples=cand_mults or {"__none__": 1}, execution=None, panel=None, lot_size=1)
            state = res.state
            if res.fills and first_fill is None:
                first_fill = ddate
            viols += int(bool(res.diagnostics.execution_gross_violation))
            eq = res.equity_close
            peak = max(peak, eq)
            mdd = min(mdd, eq / peak - 1 if peak else 0.0)
            giveback = max(giveback, (peak - eq) / float(capital))
            cash_min = min(cash_min, float(state.cash))
            baseline_pick = None
            baseline_mom = -1.0
            for fr in frame_rows:
                m = fr["mom_60"]
                a = fr["adv20"]
                if not isinstance(m, float) or m != m or m <= 0:
                    continue
                if m > baseline_mom or (m == baseline_mom and (baseline_pick is None or str(fr["ticker"]) < baseline_pick)):
                    baseline_mom = m
                    baseline_pick = str(fr["ticker"])
            base_intent = PortfolioIntent(kind="target", weights={baseline_pick: 0.8}) if baseline_pick else CASH_INTENT
            base_mults = {t: mult_map.get(t, 1) for t in set(base.shares) | set(getattr(base_intent, "weights", {}))}
            bres = cash_order_transition(prior_state=base, intent=base_intent, decision_date=ddate, prev_closes=closes[dec], opens=opens[idx], closes=closes[idx], cost_model=cost_model, adv_by_ticker=adv_map, max_order_to_adv=float(participation), exposure_limits=limits, leverage_multiples=base_mults or {"__none__": 1}, execution=None, panel=None, lot_size=1)
            base = bres.state
        term = state.equity_at_prices(closes[s + 35]) if closes[s + 35] else float(state.cash)
        bterm = base.equity_at_prices(closes[s + 35]) if closes[s + 35] else float(base.cash)
        rows.append({"window_start": window[0], "window_end": window[-1], "candidate_return": float(term) / float(capital) - 1.0, "baseline_return": float(bterm) / float(capital) - 1.0, "candidate_mdd": float(mdd), "candidate_giveback": float(giveback), "candidate_cash_min": float(cash_min), "candidate_first_fill": first_fill, "candidate_missing_mark_count": int(missing), "candidate_execution_violation_count": int(viols)})
    return pl.DataFrame(rows, schema={"window_start": pl.Date, "window_end": pl.Date, "candidate_return": pl.Float64, "baseline_return": pl.Float64, "candidate_mdd": pl.Float64, "candidate_giveback": pl.Float64, "candidate_cash_min": pl.Float64, "candidate_first_fill": pl.Date, "candidate_missing_mark_count": pl.Int64, "candidate_execution_violation_count": pl.Int64})


def run_frontier_from_paths(*, data_root: Path, start: date, end: date, output: Path) -> Path:
    t0 = time.perf_counter()
    silver_path = Path(data_root) / "normalized" / "etf_daily.parquet"
    panel_full = pl.read_parquet(str(silver_path))
    sessions = get_calendar().sessions(start, end)
    panel = panel_full.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    scenarios = [(p, b, a) for p in (0.01, 0.02) for b in (8.0, 23.0) for a in (True, False)]
    results: dict[tuple[float, float, bool], pl.DataFrame] = {}
    for part, bps, allow in scenarios:
        results[(part, bps, allow)] = run_frontier_research(panel, sessions, capital=1_000_000_000.0, participation=part, cost_bps=bps, allow_leverage=allow)
    primary = results[(0.01, 8.0, True)]
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    primary.write_parquet(str(out / "windows.parquet"))
    import resource

    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    sha = hashlib.sha256(str(panel.height).encode()).hexdigest()[:16]
    elapsed = time.perf_counter() - t0
    rets = primary["candidate_return"].to_list()
    report = {"status": "RESEARCH_ONLY", "session_count": len(sessions), "window_count": primary.height, "scenarios": len(scenarios), "median_return": float(sorted(rets)[len(rets) // 2]) if rets else 0.0, "ruin": float(sum(1 for v in rets if v < -0.25) / len(rets)) if rets else 0.0, "data_sha256": sha, "elapsed_seconds": float(elapsed), "peak_rss_mb": float(rss), "start": str(start), "end": str(end)}
    (out / "report.json").write_text(__import__("json").dumps(report, indent=2))
    return out / "windows.parquet"
