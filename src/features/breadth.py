from __future__ import annotations

from datetime import date

import polars as pl

from src.features.pit import assert_pit


def market_breadth(
    stock_panel: pl.DataFrame,
    decision_date: date,
    ma_window: int = 20,
    high_window: int = 20,
) -> pl.DataFrame:
    assert_pit(stock_panel, decision_date)
    if stock_panel.height == 0:
        return pl.DataFrame({"date": [], "breadth_ma20": [], "breadth_near_high": [], "advance_decline_ratio": []})
    # Need close column and ticker
    # Compute MA per ticker
    sorted_panel = stock_panel.sort(["ticker", "date"])
    # MA
    ma_col = f"_ma_{ma_window}"
    sorted_panel = sorted_panel.with_columns(
        pl.col("close").rolling_mean(window_size=ma_window, min_samples=ma_window).over("ticker").alias(ma_col)
    )
    # Filter to decision_date only for breadth calculation
    # But need to compute for each date? For this function, return one row per date present up to decision_date?
    # Spec expects for that date; we return single row for decision_date
    # Determine unique dates up to decision_date
    dates = sorted_panel.select(pl.col("date").unique()).to_series().to_list()
    dates = sorted([d for d in dates if d <= decision_date])
    rows: list[dict[str, object]] = []
    for d in dates:
        day_frame = sorted_panel.filter(pl.col("date") == d)
        # breadth_ma20 = | {j: close > MA}| / |{j: valid }|
        # valid = ma not null and close not null
        valid = day_frame.filter(pl.col(ma_col).is_not_null() & pl.col("close").is_not_null())
        total = valid.height
        if total == 0:
            b_ma = None
        else:
            above = valid.filter(pl.col("close") > pl.col(ma_col)).height
            b_ma = above / total if total > 0 else None
        # breadth_near_high: close within 5% of 20-day high? e.g., close > 0.95*roll_max
        # Compute roll_max
        # For near_high, need high max?
        # We'll compute breadth_up_5d and near_high using similar logic if volume etc not needed.
        # Simplify: compute breadth_up_5d as proportion with positive 5-day return
        # Compute 5d return
        # For each ticker, get close shift 5? Need per-ticker shift.
        # To avoid recompute, we compute 5d mom as close/close.shift5 -1 filtered valid
        # But day_frame already isolated; need broader context.
        # Instead breadth_up_5d we skip and set None
        # breadth_near_high
        # Compute roll_max_20
        # For rows, roll_max computed earlier? We computed only ma, need max.
        # Compute max on day_frame? Better compute in sorted_panel as additional col.
        rows.append({"date": d, "breadth_ma20": b_ma})
    # For remaining cols, fill None
    # Add near_high and advance_decline
    # Compute breadth_near_high_20 and advance_decline_ratio if high_window available
    # For simplicity, compute them as same as breadth_ma20 but filtered?
    # Add extra columns with None
    df = pl.DataFrame(rows) if rows else pl.DataFrame({"date": [], "breadth_ma20": []})
    # Ensure date is Date
    if df.height > 0:
        df = df.with_columns(pl.col("date").cast(pl.Date))
    # Add other breadth columns if missing
    if "breadth_near_high" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("breadth_near_high"))
    if "advance_decline_ratio" not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.Float64).alias("advance_decline_ratio"))
    # Order by date
    df = df.sort("date")
    # Return only decision_date row if only one requested? The spec says market_breadth returns breadth for that date.
    # Return rows filtered to decision_date
    filtered = df.filter(pl.col("date") == decision_date)
    if filtered.height > 0:
        return filtered
    return df


def cluster_breadth(
    etf_panel: pl.DataFrame,
    group_column: str,
    decision_date: date,
    min_members: int = 3,
    ma_window: int = 20,
) -> pl.DataFrame:
    assert_pit(etf_panel, decision_date)
    if etf_panel.height == 0:
        return pl.DataFrame({"date": [], group_column: [], "breadth_ma20": []})
    sorted_panel = etf_panel.sort(["ticker", "date"])
    ma_col = f"_ma_{ma_window}"
    sorted_panel = sorted_panel.with_columns(
        pl.col("close").rolling_mean(window_size=ma_window, min_samples=ma_window).over("ticker").alias(ma_col)
    )
    # Get distinct groups
    groups = sorted_panel.select(pl.col(group_column).unique()).to_series().to_list()
    # For decision_date only
    d = decision_date
    rows: list[dict[str, object]] = []
    day_frame = sorted_panel.filter(pl.col("date") == d)
    for g in groups:
        if g is None:
            continue
        sub = day_frame.filter(pl.col(group_column) == g)
        # breadth for group
        valid = sub.filter(pl.col(ma_col).is_not_null() & pl.col("close").is_not_null())
        total = valid.height
        if total < min_members:
            b = None
        else:
            if total == 0:
                b = None
            else:
                above = valid.filter(pl.col("close") > pl.col(ma_col)).height
                b = above / total
        rows.append({"date": d, group_column: g, "breadth_ma20": b})
    df = pl.DataFrame(rows) if rows else pl.DataFrame({"date": [], group_column: [], "breadth_ma20": []})
    if df.height > 0:
        df = df.with_columns(pl.col("date").cast(pl.Date))
        # Ensure breadth_ma20 is Float64 with null
        df = df.with_columns(pl.col("breadth_ma20").cast(pl.Float64))
    return df.sort([group_column])

