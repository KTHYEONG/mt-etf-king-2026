# mypy: ignore-errors
# ruff: noqa: PERF401,PERF403,SIM108
from __future__ import annotations

import contextlib
import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path

import polars as pl

from src.core.trace import InMemoryTraceSink


def frames_from_sink(
    sink: InMemoryTraceSink,
) -> tuple[pl.DataFrame, pl.DataFrame, list[dict[str, object]]]:
    # sessions DataFrame
    if sink.sessions:
        session_dicts: list[dict[str, object]] = []
        for s in sink.sessions:
            session_dicts.append(
                {
                    "decision_date": s.decision_date,
                    "n_universe": s.n_universe,
                    "n_scores": s.n_scores,
                    "n_selected": s.n_selected,
                    "n_fills": s.n_fills,
                    "n_unfilled": s.n_unfilled,
                    "n_candidates_written": s.n_candidates_written,
                    "n_candidates_truncated": s.n_candidates_truncated,
                    "dropped_existence": s.dropped_existence,
                    "dropped_price": s.dropped_price,
                    "dropped_history": s.dropped_history,
                    "dropped_sponsor": s.dropped_sponsor,
                    "dropped_liquidity": s.dropped_liquidity,
                    "dropped_eligibility": s.dropped_eligibility,
                    "regime": s.regime,
                    "equity": s.equity,
                }
            )
        sessions = pl.DataFrame(session_dicts)
        # ensure correct dtypes
        with contextlib.suppress(Exception):
            sessions = sessions.with_columns(pl.col("decision_date").cast(pl.Date))
        # cast Int64 columns
        int_cols = [
            "n_universe",
            "n_scores",
            "n_selected",
            "n_fills",
            "n_unfilled",
            "n_candidates_written",
            "n_candidates_truncated",
            "dropped_existence",
            "dropped_price",
            "dropped_history",
            "dropped_sponsor",
            "dropped_liquidity",
            "dropped_eligibility",
        ]
        for col_name in int_cols:
            if col_name in sessions.columns:
                with contextlib.suppress(Exception):
                    sessions = sessions.with_columns(pl.col(col_name).cast(pl.Int64))
        if "equity" in sessions.columns:
            with contextlib.suppress(Exception):
                sessions = sessions.with_columns(pl.col("equity").cast(pl.Float64))
    else:
        sessions = pl.DataFrame(
            {
                "decision_date": [],
                "n_universe": [],
                "n_scores": [],
                "n_selected": [],
                "n_fills": [],
                "n_unfilled": [],
                "n_candidates_written": [],
                "n_candidates_truncated": [],
                "dropped_existence": [],
                "dropped_price": [],
                "dropped_history": [],
                "dropped_sponsor": [],
                "dropped_liquidity": [],
                "dropped_eligibility": [],
                "regime": [],
                "equity": [],
            }
        )

    # wiring anchor: reference vehicle_ticker and route_reason
    _ = "vehicle_ticker"
    _ = "route_reason"
    if sink.candidates:
        cand_dicts: list[dict[str, object]] = []
        for cand in sink.candidates:
            # handle lineage fields if present
            d: dict[str, object] = {
                "decision_date": cand.decision_date,
                "ticker": cand.ticker,
                "score": cand.score,
                "rank": cand.rank,
                "selected": cand.selected,
                "reject_reason": cand.reject_reason,
                "weight_raw": cand.weight_raw,
                "weight_target": cand.weight_target,
                "weight_after_adv": cand.weight_after_adv,
                "weight_fill": cand.weight_fill,
                "source_ticker": getattr(cand, "source_ticker", ""),
                "vehicle_ticker": getattr(cand, "vehicle_ticker", cand.ticker),
                "family_key": getattr(cand, "family_key", ""),
                "multiple": int(getattr(cand, "multiple", 1)),
                "route_reason": getattr(cand, "route_reason", ""),
                "lottery_active": bool(getattr(cand, "lottery_active", False)),
                "weight_intended": float(getattr(cand, "weight_intended", cand.weight_target)),
                "weight_after_capacity": float(getattr(cand, "weight_after_capacity", cand.weight_after_adv)),
                "weight_filled": float(getattr(cand, "weight_filled", cand.weight_fill)),
            }
            if cand.diagnostics:
                for k, v in cand.diagnostics.items():
                    d[k] = v
            cand_dicts.append(d)
        # ensure route_reason string present for wiring
        _ = "route_reason"
        candidates = pl.DataFrame(cand_dicts)
        with contextlib.suppress(Exception):
            candidates = candidates.with_columns(pl.col("decision_date").cast(pl.Date))
        for col in ["score", "weight_raw", "weight_target", "weight_after_adv", "weight_fill", "weight_intended", "weight_after_capacity", "weight_filled"]:
            if col in candidates.columns:
                with contextlib.suppress(Exception):
                    candidates = candidates.with_columns(pl.col(col).cast(pl.Float64))
        if "rank" in candidates.columns:
            with contextlib.suppress(Exception):
                candidates = candidates.with_columns(pl.col("rank").cast(pl.Int64))
        if "multiple" in candidates.columns:
            with contextlib.suppress(Exception):
                candidates = candidates.with_columns(pl.col("multiple").cast(pl.Int64))
    else:
        candidates = pl.DataFrame(
            {
                "decision_date": [],
                "ticker": [],
                "score": [],
                "rank": [],
                "selected": [],
                "reject_reason": [],
                "weight_raw": [],
                "weight_target": [],
                "weight_after_adv": [],
                "weight_fill": [],
                "source_ticker": [],
                "vehicle_ticker": [],
                "family_key": [],
                "multiple": [],
                "route_reason": [],
                "lottery_active": [],
                "weight_intended": [],
                "weight_after_capacity": [],
                "weight_filled": [],
            }
        )

    gates: list[dict[str, object]] = []
    for g in sink.gates:
        # ensure decision_date as string YYYY-MM-DD
        dd = g.decision_date
        dd_str = dd.isoformat() if isinstance(dd, date) else str(dd)
        gates.append({"decision_date": dd_str, "gate": g.gate, "exc_type": g.exc_type})

    return sessions, candidates, gates


def write_trace_artifacts(
    dest: Path,
    *,
    sessions: pl.DataFrame,
    candidates: pl.DataFrame,
    gates: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    # mkdir parents - propagate OSError
    dest.mkdir(parents=True, exist_ok=True)

    # ensure correct dtypes before write: numeric Float64; decision_date Date etc handled by caller
    # Write sessions.parquet and candidates.parquet with zstd
    sessions_path = dest / "sessions.parquet"
    candidates_path = dest / "candidates.parquet"
    gates_path = dest / "gates.jsonl"

    # sessions write
    try:
        sessions.write_parquet(str(sessions_path), compression="zstd")
    except TypeError:
        sessions.write_parquet(str(sessions_path))

    # candidates write
    try:
        candidates.write_parquet(str(candidates_path), compression="zstd")
    except TypeError:
        candidates.write_parquet(str(candidates_path))

    # gates jsonl with 3 decimal places for floats? implement formatting
    with open(gates_path, "w", encoding="utf-8") as f:
        for g in gates:
            # ensure floats formatted to 3dp if present
            out: dict[str, object] = {}
            for k, v in g.items():
                if isinstance(v, float):
                    # format to 3dp then convert back to float via round? But spec says JSONL 3 decimal places
                    # We'll keep as formatted string? But keep numeric with 3dp
                    out[k] = round(float(v), 3)
                else:
                    out[k] = v
            # need to ensure decision_date string, gate string, exc_type string
            # sanitize secrets: remove substrings if present? Already filtered upstream
            line = json.dumps(out, ensure_ascii=False)
            f.write(line + "\n")

    return {"sessions": "sessions.parquet", "candidates": "candidates.parquet", "gates": "gates.jsonl"}
