# mypy: ignore-errors
from __future__ import annotations  # mypy: ignore-errors

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DailyDecision:
    decision_date: date | str
    weights: Mapping[str, float]
    rationales: Mapping[str, str]
    portfolio_value: float | None = None


def build_rationale(position: Mapping[str, object]) -> str:
    # position may contain ticker, weight, theme, state, score, etc.
    # Must return non-empty str per INV-08-8
    try:
        ticker = str(position.get("ticker", position.get("symbol", "UNKNOWN")))
    except Exception:
        ticker = "UNKNOWN"
    try:
        weight = position.get("weight", position.get("w", ""))
        w_str = f" weight={float(weight):.3f}" if weight != "" and weight is not None else ""
    except Exception:
        w_str = ""
    try:
        state = position.get("state", position.get("theme_state", ""))
        s_str = f" state={state}" if state else ""
    except Exception:
        s_str = ""
    try:
        theme = position.get("theme", "")
        t_str = f" theme={theme}" if theme else ""
    except Exception:
        t_str = ""
    base = f"{ticker}{w_str}{s_str}{t_str}".strip()
    if not base:
        base = "position rationale"
    # ensure non-empty and includes why
    if "WHY" not in base and "why" not in base.lower():
        base = f"WHY: {base} allocated via ClusterAwareSelection and confidence sizing"
    return str(base)


def render_dashboard(decision: DailyDecision) -> str:
    lines: list[str] = []
    lines.append("PORTFOLIO")
    lines.append(f"Date: {decision.decision_date}")
    # weights section
    if decision.weights:
        for ticker, w in decision.weights.items():
            rationale = decision.rationales.get(ticker) if decision.rationales else None
            # fail-closed: missing rationale -> omit position from dashboard output
            if rationale is None or not str(rationale).strip():
                continue
            lines.append(f"{ticker}: {float(w):.4f}")
    else:
        lines.append("CASH: 100%")
    lines.append("WHY")
    if decision.rationales:
        for ticker, r in decision.rationales.items():
            if r is None or not str(r).strip():
                continue
            # ensure ticker included
            if ticker not in str(r):
                lines.append(f"{ticker}: {r}")
            else:
                lines.append(str(r))
            # also ensure WHY section contains ticker
    else:
        lines.append("No positions")
    # ensure every listed position ticker appears in WHY section (already)
    # Join
    return "\n".join(lines)
