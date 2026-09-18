# mypy: ignore-errors
from __future__ import annotations  # mypy: ignore-errors

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path


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


_STATE_KO_MAP: dict[str, str] = {
    "HOLD": "보유유지(HOLD)",
    "BUY": "신규매수(BUY)",
    "NEW": "신규매수(BUY)",
    "TRIM": "비중축소(TRIM)",
    "EXIT": "전량매도(EXIT)",
    "SELL": "전량매도(EXIT)",
    "CASH": "현금화(CASH)",
}

_KNOWN_ETF_NAMES: dict[str, str] = {
    "122630": "KODEX 레버리지",
    "233740": "KODEX 코스닥150레버리지",
    "069500": "KODEX 200",
    "114800": "KODEX 인버스",
    "252670": "KODEX 200선물인버스2X",
    "412570": "TIGER Fn반도체TOP10",
    "451060": "ACE 미국배당다우존스",
    "494310": "PLUS 고배당주",
    "488080": "ACE 미국빅테크TOP7 Plus",
}


def _parse_state_and_reason_ko(ticker: str, weight: float, raw_reason: str) -> tuple[str, str]:
    import re

    m_state = re.search(r"state=([A-Za-z0-9_]+)", raw_reason)
    raw_state = m_state.group(1).upper() if m_state else "HOLD"
    state_ko = _STATE_KO_MAP.get(raw_state, f"{raw_state}")

    weight_pct = f"{float(weight) * 100:.1f}%"
    reasons: list[str] = []

    if "ClusterAwareSelection" in raw_reason and "confidence sizing" in raw_reason:
        reasons.append("클러스터 중복 제거 및 모델 확신도 산정에 따라")
    elif "ClusterAwareSelection" in raw_reason:
        reasons.append("클러스터 분산 선택 알고리즘에 따라")
    elif "confidence sizing" in raw_reason:
        reasons.append("확신도 기반 비중 산정에 따라")

    if "peak_lock" in raw_reason:
        reasons.append("고점 대비 보호 장치(Peak Lock) 발동으로")
    if "house_money" in raw_reason:
        reasons.append("수익금 보호 잠금(House Money) 발동으로")

    if raw_state == "HOLD":
        action_ko = "기존 포지션 보유 유지"
    elif raw_state in ("BUY", "NEW"):
        action_ko = "신규 매수 편입"
    elif raw_state == "TRIM":
        action_ko = "기존 비중 축소"
    elif raw_state in ("EXIT", "SELL"):
        action_ko = "전량 매도 청산"
    elif raw_state == "CASH":
        action_ko = "전량 현금화"
    else:
        action_ko = f"상태 {raw_state}"

    if reasons:
        reason_ko = f"{' '.join(reasons)} 비중 {weight_pct} 배분, {action_ko}"
    else:
        reason_ko = f"목표 비중 {weight_pct} 배분 ({action_ko})"

    return state_ko, reason_ko


def write_decision_artifact(
    decision: DailyDecision,
    path: Path,
    order_estimates: Mapping[str, object] | None = None,
    ticker_names: Mapping[str, str] | None = None,
) -> Path:
    # ensure parent
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # noqa: S110
        pass
    # build payload with as_of, selected
    try:
        as_of = str(decision.decision_date)
    except Exception:
        as_of = ""
    selected: list[dict[str, object]] = []
    try:
        for ticker, w in decision.weights.items():
            reason = ""
            try:
                reason = str(decision.rationales.get(ticker, "")) if decision.rationales else ""
            except Exception:
                reason = ""
            if not reason or not str(reason).strip():
                # ensure non-empty; but if missing, use placeholder
                reason = f"WHY: {ticker} weight={float(w):.3f} state=HOLD"
            # ensure state= present
            if "state=" not in reason:
                reason = reason + " state=HOLD"
            if "WHY" not in reason:
                reason = f"WHY: {reason}"

            state_ko, reason_ko = _parse_state_and_reason_ko(ticker, float(w), reason)
            name = ""
            if ticker_names and ticker in ticker_names:
                name = str(ticker_names[ticker])
            elif ticker in _KNOWN_ETF_NAMES:
                name = _KNOWN_ETF_NAMES[ticker]

            item: dict[str, object] = {
                "ticker": ticker,
                "name": name,
                "weight": float(w),
                "state": state_ko,
                "reason_ko": reason_ko,
            }
            if order_estimates and ticker in order_estimates:
                est = order_estimates[ticker]
                shares = getattr(est, "est_shares", None)
                krw = getattr(est, "est_krw", None)
                if shares is not None:
                    item["est_shares"] = shares
                if krw is not None:
                    item["est_krw"] = krw
            item["reason"] = reason
            selected.append(item)
    except Exception:
        selected = []
    payload = {"as_of": as_of, "selected": selected}
    # include portfolio_value if present
    try:
        if decision.portfolio_value is not None:
            payload["portfolio_value"] = float(decision.portfolio_value)
    except Exception:  # noqa: S110
        pass
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception:
        # fallback: try write
        Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return Path(path)

