from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import TypeAlias

import polars as pl

from src.universe.instruments import InstrumentAttributes, InstrumentMaster


@dataclass(frozen=True)
class ClusterChoice:
    ticker: str
    index_key: str
    theme: str


SnapshotIndex: TypeAlias = dict[str, Mapping[str, object]]

_EMPTY_TICKER_CANDIDATES = pl.DataFrame(schema={"ticker": pl.Utf8})


def build_snapshot_index(snapshot: pl.DataFrame) -> SnapshotIndex:
    if snapshot.height == 0 or "ticker" not in snapshot.columns:
        return {}
    index: dict[str, Mapping[str, object]] = {}
    for row in snapshot.iter_rows(named=True):
        t = row.get("ticker")
        if t is None:
            continue
        tk = str(t)
        # store only first occurrence for deterministic single-row snapshot
        if tk not in index:
            index[tk] = dict(row)
    return index


def select_representative_from_rows(
    members: Sequence[str],
    attrs: Mapping[str, InstrumentAttributes],
    index: SnapshotIndex,
    lookback: int = 20,
) -> str | None:
    _ = lookback
    if not members:
        return None
    eligible: list[str] = []
    for t in members:
        a = attrs.get(t)
        if a is not None and a.leverage_multiple != 1:
            continue
        eligible.append(t)
    seen: set[str] = set()
    uniq: list[str] = []
    for t in eligible:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return None
    scores: list[tuple[float, float, str]] = []
    for t in uniq:
        mapping = index.get(t)
        median_tv = 0.0
        te_val = 0.0
        if mapping is not None:
            tv_raw = mapping.get("trading_value")
            if tv_raw is not None:
                try:
                    median_tv = float(tv_raw)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    median_tv = 0.0
            te_raw = mapping.get("tracking_error")
            if te_raw is not None:
                try:
                    te_val = float(te_raw)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    te_val = 0.0
        scores.append((median_tv, te_val, t))
    scores.sort(key=lambda x: (-x[0], x[1], x[2]))
    return scores[0][2] if scores else None


def select_representative(
    candidates: pl.DataFrame,
    attrs: Mapping[str, InstrumentAttributes],
    lookback: int = 20,
    *,
    snapshot_index: SnapshotIndex | None = None,
    members: Sequence[str] | None = None,
) -> str | None:
    if snapshot_index is not None and members is not None:
        return select_representative_from_rows(members, attrs, snapshot_index, lookback=lookback)
    if snapshot_index is not None:
        if candidates.height == 0 or "ticker" not in candidates.columns:
            return None
        indexed_members = [str(t) for t in candidates.select(pl.col("ticker")).to_series().to_list()]
        return select_representative_from_rows(indexed_members, attrs, snapshot_index, lookback=lookback)
    if candidates.height == 0 or "ticker" not in candidates.columns:
        return None
    built = build_snapshot_index(candidates)
    members = list(built.keys())
    return select_representative_from_rows(members, attrs, built, lookback=lookback)


class ClusterResolver:
    def __init__(self, master: InstrumentMaster, max_per_theme: int = 2) -> None:
        self._master = master
        self._max_per_theme = max_per_theme

    def resolve_indexed(self, index: SnapshotIndex, decision_date: date) -> list[ClusterChoice]:
        _ = decision_date
        if not index:
            return []
        groups: dict[str, list[str]] = {}
        for ticker in index:
            attr = self._master.attributes.get(ticker)
            if attr is None:
                continue
            ik = attr.index_key
            groups.setdefault(ik, []).append(ticker)
        result: list[ClusterChoice] = []
        for ik, members in groups.items():
            selected = select_representative(
                _EMPTY_TICKER_CANDIDATES,
                self._master.attributes,
                lookback=20,
                snapshot_index=index,
                members=members,
            )
            if selected is None:
                continue
            attr = self._master.attributes.get(selected)
            if attr is None:
                continue
            result.append(ClusterChoice(ticker=selected, index_key=ik, theme=attr.theme))
        if self._max_per_theme is not None and self._max_per_theme > 0:
            theme_groups: dict[str, list[ClusterChoice]] = {}
            for c in result:
                theme_groups.setdefault(c.theme, []).append(c)
            trimmed: list[ClusterChoice] = []
            for choices in theme_groups.values():
                if len(choices) <= self._max_per_theme:
                    trimmed.extend(choices)
                else:
                    def _score(cc: ClusterChoice) -> float:
                        m = index.get(cc.ticker)
                        if m is None:
                            return 0.0
                        tv = m.get("trading_value")
                        if tv is not None:
                            try:
                                return float(tv)  # type: ignore[arg-type]
                            except (TypeError, ValueError):
                                pass
                        cv = m.get("close")
                        if cv is not None:
                            try:
                                return float(cv)  # type: ignore[arg-type]
                            except (TypeError, ValueError):
                                pass
                        return 0.0

                    sorted_choices = sorted(choices, key=_score, reverse=True)
                    trimmed.extend(sorted_choices[: self._max_per_theme])
            result = trimmed
        result = sorted(result, key=lambda c: (c.index_key, c.ticker))
        return result

    def resolve(self, snapshot: pl.DataFrame, decision_date: date) -> list[ClusterChoice]:
        index = build_snapshot_index(snapshot)
        return self.resolve_indexed(index, decision_date)
