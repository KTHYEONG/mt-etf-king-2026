# mypy: ignore-errors
from __future__ import annotations  # mypy: ignore-errors

from collections.abc import Mapping


class ClusterAwareSelection:
    def __init__(self, master, max_per_theme: int = 2, max_per_family: int = 1) -> None:
        self.master = master
        self.max_per_theme = int(max_per_theme)
        self.max_per_family = int(max_per_family)

    def select(self, scores: Mapping[str, float]) -> list[str]:
        return select_positions(scores, self.master, self.max_per_theme, self.max_per_family)


def select_positions(
    scores: Mapping[str, float],
    master,
    max_per_theme: int,
    max_per_family: int = 1,
) -> list[str]:
    if not scores:
        return []
    # deterministic sort descending score then ticker
    sorted_tickers = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    # Step 1: family dedup (leverage_family_key) precedes theme dedup
    # Need to keep top max_per_family per family
    family_counts: dict[str, int] = {}
    after_family: list[tuple[str, float]] = []
    for ticker, sc in sorted_tickers:
        try:
            attr = master.attributes.get(ticker) if hasattr(master, "attributes") else None
        except Exception:
            attr = None
        if attr is not None:
            fk = getattr(attr, "leverage_family_key", ticker)
            theme = getattr(attr, "theme", "UNKNOWN")
        else:
            fk = ticker
            theme = "UNKNOWN"
        cnt = family_counts.get(fk, 0)
        if cnt < max_per_family:
            after_family.append((ticker, sc))
            family_counts[fk] = cnt + 1
        else:
            continue
    # Step 2: theme dedup
    theme_counts: dict[str, int] = {}
    result: list[str] = []
    for ticker, _sc in after_family:
        try:
            attr = master.attributes.get(ticker) if hasattr(master, "attributes") else None
        except Exception:
            attr = None
        theme = getattr(attr, "theme", "UNKNOWN") if attr is not None else "UNKNOWN"
        cnt = theme_counts.get(theme, 0)
        if cnt < max_per_theme:
            result.append(ticker)
            theme_counts[theme] = cnt + 1
        else:
            continue
    return result
