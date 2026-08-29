from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

import polars as pl
import yaml

from src.core.calendar import TradingCalendar
from src.features.breadth import cluster_breadth as compute_cluster_breadth
from src.features.breadth import market_breadth as compute_market_breadth
from src.features.crosssec import add_momentum_acceleration, cross_sectional_zscore, percentile_rank
from src.features.flow import add_flow, decompose_aum_change
from src.features.momentum import add_momentum
from src.features.pit import align_session_grid, assert_pit
from src.features.regime import RegimeConfig, RegimeSnapshot, classify_regime
from src.features.trend import add_trend
from src.features.volatility import add_volatility
from src.universe.provider import UniverseSnapshot


@dataclass(frozen=True)
class FeatureConfig:
    momentum_horizons: tuple[int, ...]
    ma_windows: tuple[int, ...]
    breakout_windows: tuple[int, ...]
    volatility_windows: tuple[int, ...]
    flow_windows: tuple[int, ...]
    regime: RegimeConfig

    @classmethod
    def from_yaml(cls, path: Path) -> FeatureConfig:
        with open(path, encoding="utf-8") as f:
            raw_any: Any = yaml.safe_load(f)
        raw: dict[str, Any] = cast(dict[str, Any], raw_any) if isinstance(raw_any, dict) else {}
        # Support both top-level with 'features' key or direct
        data_any: Any = raw.get("features") if isinstance(raw, dict) and "features" in raw else raw
        data: dict[str, Any] = cast(dict[str, Any], data_any) if isinstance(data_any, dict) else {}
        # regime
        reg_any: Any = data.get("regime", {})
        reg_raw: dict[str, Any] = cast(dict[str, Any], reg_any) if isinstance(reg_any, dict) else {}
        weights = dict(reg_raw.get("weights", {}))
        thresholds_raw = reg_raw.get("thresholds", [0.25, 0.45, 0.65, 0.85])
        tmp_thresh = tuple(float(x) for x in thresholds_raw)
        if len(tmp_thresh) != 4:
            raise ValueError(f"regime thresholds must have 4 values, got {tmp_thresh}")
        thresholds: tuple[float, float, float, float] = (tmp_thresh[0], tmp_thresh[1], tmp_thresh[2], tmp_thresh[3])
        breadth_floor = float(reg_raw.get("breadth_floor", 0.5))
        volatility_ceiling = float(reg_raw.get("volatility_ceiling", 0.025))
        regime = RegimeConfig(
            weights=weights,
            thresholds=thresholds,
            breadth_floor=breadth_floor,
            volatility_ceiling=volatility_ceiling,
        )
        return cls(
            momentum_horizons=tuple(int(x) for x in data.get("momentum_horizons", [3, 5, 10, 20, 40, 60])),
            ma_windows=tuple(int(x) for x in data.get("ma_windows", [20, 60])),
            breakout_windows=tuple(int(x) for x in data.get("breakout_windows", [20, 60])),
            volatility_windows=tuple(int(x) for x in data.get("volatility_windows", [5, 20])),
            flow_windows=tuple(int(x) for x in data.get("flow_windows", [5, 20])),
            regime=regime,
        )

    def warmup_sessions(self, margin: int = 20) -> int:
        all_vals: list[int] = []
        all_vals.extend(self.momentum_horizons)
        all_vals.extend(self.ma_windows)
        all_vals.extend(self.breakout_windows)
        all_vals.extend(self.volatility_windows)
        all_vals.extend(self.flow_windows)
        if not all_vals:
            return int(margin)
        return int(max(all_vals) + margin)


class FeatureBuilder:
    def __init__(self, calendar: TradingCalendar, config: FeatureConfig) -> None:
        self.calendar = calendar
        self.config = config

    def build_panel(self, etf_panel: pl.DataFrame, decision_date: date) -> pl.DataFrame:
        assert_pit(etf_panel, decision_date)
        if etf_panel.height == 0:
            return etf_panel
        # Determine sessions range from panel
        # Use calendar to get full session list between min and max date present
        dates = etf_panel.select(pl.col("date").unique()).to_series().to_list()
        dates = [d for d in dates if d is not None]
        if not dates:
            sessions: list[date] = []
        else:
            min_d = min(dates)
            max_d = max(dates)
            try:
                sessions = self.calendar.sessions(min_d, max_d)
            except Exception:
                sessions = sorted(dates)
        # Align to session grid (once, not per-date loop)
        aligned = align_session_grid(etf_panel, sessions, key="ticker")
        # Apply rolling computations once over whole panel
        # Order: momentum, trend, volatility, flow (which includes decompose)
        panel = add_momentum(aligned, self.config.momentum_horizons, decision_date, price="close", key="ticker")
        panel = add_trend(panel, self.config.ma_windows, self.config.breakout_windows, decision_date, key="ticker")
        panel = add_volatility(panel, self.config.volatility_windows, decision_date, key="ticker")
        # decompose then flow
        panel = decompose_aum_change(panel, decision_date, key="ticker")
        panel = add_flow(panel, self.config.flow_windows, decision_date, key="ticker")
        # Ensure sorted for reproducibility
        sort_cols = []
        if "date" in panel.columns:
            sort_cols.append("date")
        if "ticker" in panel.columns:
            sort_cols.append("ticker")
        if sort_cols:
            panel = panel.sort(sort_cols)
        return panel

    def market_breadth_panel(self, stock_panel: pl.DataFrame, decision_date: date) -> pl.DataFrame:
        ma_window = self.config.ma_windows[0]
        high_window = self.config.breakout_windows[0]
        return compute_market_breadth(
            stock_panel,
            decision_date,
            ma_window=ma_window,
            high_window=high_window,
        )

    def cluster_breadth_panel(
        self,
        etf_panel: pl.DataFrame,
        group_column: str,
        decision_date: date,
        min_members: int,
    ) -> pl.DataFrame:
        ma_window = self.config.ma_windows[0]
        return compute_cluster_breadth(
            etf_panel,
            group_column,
            decision_date,
            min_members=min_members,
            ma_window=ma_window,
        )

    def regime_snapshot(
        self,
        index_panel: pl.DataFrame,
        breadth_panel: pl.DataFrame,
        decision_date: date,
    ) -> RegimeSnapshot:
        return classify_regime(index_panel, breadth_panel, decision_date, self.config.regime)

    def build_regime_series(
        self,
        index_panel: pl.DataFrame,
        breadth_panel: pl.DataFrame,
        sessions: Sequence[date],
    ) -> dict[date, RegimeSnapshot]:
        out: dict[date, RegimeSnapshot] = {}
        for sess in sessions:
            # Slice PIT: date <= session
            try:
                idx_slice = index_panel.filter(pl.col("date") <= sess) if "date" in index_panel.columns else index_panel  # noqa: SIM108
                br_slice = breadth_panel.filter(pl.col("date") <= sess) if "date" in breadth_panel.columns else breadth_panel  # noqa: SIM108
            except Exception:  # noqa: S112
                continue
            try:
                snap = classify_regime(idx_slice, br_slice, sess, self.config.regime)
            except Exception:  # noqa: S112
                continue
            out[sess] = snap
        return out

    def snapshot(self, feature_panel: pl.DataFrame, universe: UniverseSnapshot) -> pl.DataFrame:
        # Use universe.as_of as decision_date for PIT
        decision_date = universe.as_of
        if feature_panel.height == 0:
            return feature_panel
        # Filter to decision_date before PIT — full-history panels are valid input
        eligible = set(universe.tickers)
        if "date" in feature_panel.columns:
            snap = feature_panel.filter(pl.col("date") == decision_date)
        else:
            snap = feature_panel
        assert_pit(snap, decision_date)
        if eligible:
            snap = snap.filter(pl.col("ticker").is_in(list(eligible)))
        if snap.height == 0:
            return snap
        # Determine momentum columns for ranking
        mom_cols = [c for c in snap.columns if c.startswith("mom_") and not c.endswith("_rs") and not c.endswith("_z")]
        # Percentile rank over only rows present (eligible filtered)
        if mom_cols:
            snap = percentile_rank(snap, mom_cols, decision_date, by="date", suffix="_rs")
            snap = cross_sectional_zscore(snap, mom_cols, decision_date, by="date", suffix="_z")
            horizons = sorted(self.config.momentum_horizons)
            fast = horizons[1] if len(horizons) > 1 else horizons[0]
            slow = horizons[3] if len(horizons) > 3 else horizons[-1]
            snap = add_momentum_acceleration(snap, decision_date, fast=fast, slow=slow)
        # Ensure every row dated exactly on decision_date
        # Already filtered
        # Sort for reproducibility
        if "ticker" in snap.columns:
            snap = snap.sort("ticker")
        return snap
