"""Executable open-to-open oracle ceiling (vectorized polars, float64)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date

import polars as pl

from src.research.decision_population import DecisionWindow
from src.universe.instruments import resolve_issuer


def build_deployment_oracle_opens(
    panel: pl.DataFrame,
    *,
    sessions: Sequence[date],
    sponsor_issuers: frozenset[str],
    brand_map: Mapping[str, str],
    min_adv: float = 100_000_000.0,
    min_history_sessions: int = 60,
) -> pl.DataFrame:
    sessions_df = pl.DataFrame({"date": list(sessions), "session_idx": list(range(len(sessions)))}, schema={"date": pl.Date, "session_idx": pl.Int64})
    # first_seen을 티커별 filter 대신 기존 group_by 집계에 합침 (전체 패널 스캔 1383회 제거)
    names = panel.group_by("ticker").agg(
        pl.col("name").last().alias("name"),
        pl.col("date").min().alias("first_seen"),
    )
    attrs: list[dict[str, object]] = []
    for row in names.iter_rows(named=True):
        ticker = str(row["ticker"])
        name = str(row["name"])
        issuer = resolve_issuer(name, brand_map)
        attrs.append(
            {
                "ticker": ticker,
                "is_sponsor": issuer in sponsor_issuers,
                "is_synth": "합성" in name,
                "first_seen": row["first_seen"],
            }
        )
    attr_df = pl.DataFrame(attrs, schema={"ticker": pl.String, "is_sponsor": pl.Boolean, "is_synth": pl.Boolean, "first_seen": pl.Date})
    frame = (
        panel.sort(["ticker", "date"])
        .with_columns(pl.col("trading_value").cast(pl.Float64).rolling_mean(window_size=20, min_samples=5).over("ticker").alias("adv_20"))
        .join(attr_df, on="ticker", how="left")
        .join(sessions_df, on="date", how="left")
        .join(sessions_df.rename({"date": "first_seen", "session_idx": "first_seen_idx"}), on="first_seen", how="left")
        .with_columns(
            (
                pl.col("is_sponsor").fill_null(False)
                & (~pl.col("is_synth").fill_null(True))
                & pl.col("is_tradable").fill_null(False)
                & (pl.col("adv_20").fill_null(0.0) >= float(min_adv))
                & ((pl.col("session_idx") - pl.col("first_seen_idx").fill_null(0)) >= int(min_history_sessions))
                & (pl.col("open").cast(pl.Float64) > 0.0)
            ).alias("eligible")
        )
        .select(pl.col("date"), pl.col("ticker"), pl.col("open").cast(pl.Float64), pl.col("eligible"))
    )
    return frame


def market_wide_session_candidates(
    panel: pl.DataFrame,
    *,
    sessions: Sequence[date],
    sponsor_issuers: frozenset[str],
    brand_map: Mapping[str, str],
    min_adv: float = 100_000_000.0,
    min_history_sessions: int = 60,
) -> dict[date, tuple[str, ...]]:
    if len(sessions) == 0:
        return {}
    frame = build_deployment_oracle_opens(
        panel,
        sessions=sessions,
        sponsor_issuers=sponsor_issuers,
        brand_map=brand_map,
        min_adv=min_adv,
        min_history_sessions=min_history_sessions,
    )
    eligible = frame.filter(pl.col("eligible").eq(True)).select(
        pl.col("date").cast(pl.Date), pl.col("ticker").cast(pl.String)
    )
    out: dict[date, tuple[str, ...]] = dict.fromkeys(sessions, ())
    grouped = eligible.group_by("date").agg(pl.col("ticker").unique().sort().alias("tickers"))
    for row in grouped.iter_rows(named=True):
        out[row["date"]] = tuple(str(t) for t in row["tickers"])
    return out


def executable_open_to_open_ceiling(
    opens: pl.DataFrame, windows: Sequence[DecisionWindow], cost_rate: float = 0.001
) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "decision_date": [w.decision_date for w in windows],
            "entry_date": [w.entry_date for w in windows],
            "exit_date": [w.exit_date for w in windows],
        },
        schema_overrides={"decision_date": pl.Date, "entry_date": pl.Date, "exit_date": pl.Date},
        strict=False,
    )
    clean = opens.filter(pl.col("eligible").eq(True) & (pl.col("open") > 0)).select(
        pl.col("date").cast(pl.Date).alias("day"), pl.col("ticker"), pl.col("open").cast(pl.Float64).alias("price")
    )
    entries = (
        frame.select("decision_date", "entry_date")
        .join(clean, left_on="entry_date", right_on="day", how="inner")
        .rename({"price": "entry_open"})
        .drop("entry_date")
    )
    exits = (
        frame.select("decision_date", "exit_date")
        .join(clean, left_on="exit_date", right_on="day", how="inner")
        .rename({"price": "exit_open"})
        .drop("exit_date")
    )
    both = entries.join(exits, on=["decision_date", "ticker"], how="inner").with_columns(
        ((pl.col("exit_open") / pl.col("entry_open") - 1.0 - float(cost_rate)).cast(pl.Float64)).alias("net")
    )
    best = both.sort("net", descending=True).group_by("decision_date", maintain_order=True).first()
    out = (
        frame.select("decision_date")
        .join(best, on="decision_date", how="left")
        .with_columns(pl.col("net").fill_null(-1.0).alias("ceil_exec_raw"), pl.col("ticker").fill_null("").alias("best_ticker"))
    )
    return out.select(pl.col("decision_date"), pl.col("best_ticker"), pl.col("ceil_exec_raw")).sort("decision_date")
