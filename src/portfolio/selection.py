# mypy: ignore-errors
from __future__ import annotations  # mypy: ignore-errors

from collections.abc import Mapping
from typing import Any


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

def family_canonical_scores(scores: Mapping[str, float], master: object) -> dict[str, float]:
    if not scores:
        return {}
    # Group by leverage_family_key
    families: dict[str, list[tuple[str, float, Any]]] = {}
    for ticker, sc in scores.items():
        try:
            fv = float(sc)
        except Exception:
            continue
        try:
            attr = master.attributes.get(ticker) if hasattr(master, "attributes") else None  # type: ignore[union-attr]
        except Exception:
            attr = None
        if attr is not None:
            fk = str(getattr(attr, "leverage_family_key", ticker))
        else:
            fk = str(ticker)
        families.setdefault(fk, []).append((ticker, fv, attr))
    out: dict[str, float] = {}
    for fk, members in families.items():
        # Check if any +1 member exists (leverage_multiple==1)
        plus_one = [(t, s, a) for t, s, a in members if a is not None and int(getattr(a, "leverage_multiple", 1)) == 1]
        # If plus_one exists, choose highest scoring +1 (deterministic tie by ticker)
        if plus_one:
            # sort by score descending then ticker asc, pick first
            plus_one_sorted = sorted(plus_one, key=lambda x: (-x[1], x[0]))
            chosen_ticker, chosen_score, _ = plus_one_sorted[0]
            out[chosen_ticker] = float(chosen_score)
        else:
            # No +1: choose min |multiple| among non-synthetic, fallback to all if no non-synthetic
            # Filter non-synthetic
            non_synth = [(t, s, a) for t, s, a in members if a is None or not bool(getattr(a, "is_synthetic", False))]
            candidates = non_synth if non_synth else members

            def _abs_mult(item: tuple[str, float, Any]) -> int:
                _, _, a = item
                if a is None:
                    return 1
                try:
                    return abs(int(getattr(a, "leverage_multiple", 1)))
                except Exception:
                    return 1

            # Find minimal abs multiple
            min_abs = min(_abs_mult(c) for c in candidates)
            best = [c for c in candidates if _abs_mult(c) == min_abs]
            # Among ties, pick highest score then ticker
            best_sorted = sorted(best, key=lambda x: (-x[1], x[0]))
            chosen_ticker, chosen_score, _ = best_sorted[0]
            out[chosen_ticker] = float(chosen_score)
    return out
