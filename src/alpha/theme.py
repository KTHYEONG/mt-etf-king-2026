from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import polars as pl

from src.alpha.cluster import ClusterChoice, SnapshotIndex, build_snapshot_index

try:
    from src.alpha.leadership import SectorScoreWeights
except Exception:  # noqa: S110
    SectorScoreWeights = object  # type: ignore[assignment, misc]


@dataclass(frozen=True)
class ThemeMetrics:
    theme: str
    representative: str
    rs: float
    accel: float
    breadth: float
    ext: float
    dd: float


class ThemePanel:
    def __init__(self, metrics: Sequence[ThemeMetrics]) -> None:
        self._metrics: dict[str, ThemeMetrics] = {m.theme: m for m in metrics}
        self._themes: tuple[str, ...] = tuple(sorted(self._metrics.keys()))

    def metrics_for(self, theme: str) -> ThemeMetrics | None:
        return self._metrics.get(theme)

    def themes(self) -> tuple[str, ...]:
        return self._themes


def _percentile_rank_among_reps(values: list[float], target: float) -> float:
    if not values:
        return 0.5
    n = len(values)
    if n == 1:
        return 0.5
    less = sum(1 for v in values if v < target)
    rank_min = less + 1
    return float((rank_min - 1) / (n - 1))


def build_theme_panel_indexed(
    index: SnapshotIndex,
    choices: Sequence[ClusterChoice],
    decision_date: date,
    weights: object,
) -> ThemePanel:
    _ = decision_date
    _ = weights
    if not choices:
        return ThemePanel([])
    if not index:
        return ThemePanel([])
    mom_20_map: dict[str, float] = {}
    for c in choices:
        m = index.get(c.ticker)
        if m is None:
            continue
        v = m.get("mom_20")
        if v is not None:
            try:
                mom_20_map[c.ticker] = float(v)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    mom_vals = list(mom_20_map.values())
    metrics_list: list[ThemeMetrics] = []
    for c in choices:
        m = index.get(c.ticker)
        if m is None:
            continue
        mom20 = mom_20_map.get(c.ticker)
        rs = _percentile_rank_among_reps(mom_vals, mom20) if mom20 is not None and mom_vals else 0.5  # noqa: SIM108
        if rs < 0.0:
            rs = 0.0
        if rs > 1.0:
            rs = 1.0
        accel = 0.0
        try:
            av = m.get("mom_accel")
            if av is not None:
                accel = float(av)  # type: ignore[arg-type]
            elif m.get("mom_5") is not None and m.get("mom_20") is not None:
                v5 = m.get("mom_5")
                v20 = m.get("mom_20")
                if v5 is not None and v20 is not None:
                    accel = float(v5) - float(v20)  # type: ignore[arg-type]
        except Exception:
            accel = 0.0
        breadth = 0.0
        try:
            cv = m.get("close")
            mv = m.get("ma_20")
            if cv is not None and mv is not None:
                try:
                    breadth = 1.0 if float(cv) > float(mv) else 0.0  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    breadth = 0.0
            total = 0
            above = 0
            for ch in [x for x in choices if x.theme == c.theme]:
                m2 = index.get(ch.ticker)
                if m2 is None:
                    continue
                cv2 = m2.get("close")
                mv2 = m2.get("ma_20")
                if cv2 is None or mv2 is None:
                    continue
                total += 1
                try:
                    if float(cv2) > float(mv2):  # type: ignore[arg-type]
                        above += 1
                except (TypeError, ValueError):
                    continue
            if total > 0:
                breadth = above / total
        except Exception:
            breadth = 0.0
        ext = 1.0
        try:
            cv = m.get("close")
            mv = m.get("ma_20")
            if cv is not None and mv is not None:
                try:
                    mvf = float(mv)  # type: ignore[arg-type]
                    cvf = float(cv)  # type: ignore[arg-type]
                    ext = cvf / mvf if mvf != 0 else 1.0
                except (TypeError, ValueError):
                    ext = 1.0
            else:
                ext = 1.0
        except Exception:
            ext = 1.0
        dd = 0.0
        try:
            dv = m.get("drawdown_20")
            if dv is not None:
                dd = float(dv)  # type: ignore[arg-type]
                if dd < 0:
                    dd = -dd
            elif m.get("roll_max_20") is not None and m.get("close") is not None:
                rv = m.get("roll_max_20")
                cv = m.get("close")
                if rv is not None and cv is not None:
                    try:
                        rvf = float(rv)  # type: ignore[arg-type]
                        cvf = float(cv)  # type: ignore[arg-type]
                        if rvf != 0:
                            dd = (rvf - cvf) / rvf
                            if dd < 0:
                                dd = 0.0
                    except (TypeError, ValueError):
                        dd = 0.0
        except Exception:
            dd = 0.0
        if rs < 0.0:
            rs = 0.0
        if rs > 1.0:
            rs = 1.0
        if breadth < 0.0:
            breadth = 0.0
        if breadth > 1.0:
            breadth = 1.0
        if dd < 0.0:
            dd = 0.0
        metrics_list.append(
            ThemeMetrics(theme=c.theme, representative=c.ticker, rs=float(rs), accel=float(accel), breadth=float(breadth), ext=float(ext), dd=float(dd))
        )
    return ThemePanel(metrics_list)


def build_theme_panel(
    snapshot: pl.DataFrame,
    history: pl.DataFrame,
    choices: Sequence[ClusterChoice],
    decision_date: date,
    weights: object,
) -> ThemePanel:
    _ = history
    index: SnapshotIndex = build_snapshot_index(snapshot)
    # also ensure history not used; indexed version uses snapshot only
    return build_theme_panel_indexed(index, choices, decision_date, weights)
