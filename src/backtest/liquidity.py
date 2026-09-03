from __future__ import annotations

import math
from collections.abc import Mapping


def _sell_first_gross(weights: Mapping[str, float], multiples: Mapping[str, int]) -> float:
    total = 0.0
    for tkr, w in weights.items():
        try:
            wf = float(w)
        except Exception:  # noqa: S112
            continue
        m = multiples.get(tkr, 1) if isinstance(multiples, Mapping) else 1
        try:
            mf = int(m)
        except Exception:
            mf = 1
        total += abs(wf * float(mf))
    return float(total)


def constrain_target_weights_sell_first(
    target: Mapping[str, float],
    current: Mapping[str, float],
    equity: float,
    adv_by_ticker: Mapping[str, float],
    max_order_to_adv: float,
    *,
    leverage_multiples: Mapping[str, int],
    max_gross_exposure: float,
) -> dict[str, float]:
    try:
        max_gross = float(max_gross_exposure)
    except Exception:
        return cap_target_weights_by_adv(target, current, equity, adv_by_ticker, max_order_to_adv)
    if not max_gross > 0:
        return cap_target_weights_by_adv(target, current, equity, adv_by_ticker, max_order_to_adv)
    try:
        eq = float(equity)
        phi = float(max_order_to_adv)
    except Exception:
        return cap_target_weights_by_adv(target, current, equity, adv_by_ticker, max_order_to_adv)
    if not eq > 0 or not phi > 0:
        return cap_target_weights_by_adv(target, current, equity, adv_by_ticker, max_order_to_adv)
    try:
        tmap = {str(k): float(v) for k, v in dict(target).items()}
    except Exception:
        tmap = {}
    try:
        cmap = {str(k): float(v) for k, v in dict(current).items()}
    except Exception:
        cmap = {}
    try:
        adv = {str(k): float(v) for k, v in dict(adv_by_ticker).items() if v is not None}
    except Exception:
        adv = {}
    mults = leverage_multiples if isinstance(leverage_multiples, Mapping) else {}
    tickers = set(tmap.keys()) | set(cmap.keys())

    def _mult(tkr: str) -> float:
        try:
            return float(int(mults.get(tkr, 1)))
        except Exception:
            return 1.0

    def _max_delta(tkr: str) -> float | None:
        a = adv.get(tkr)
        if a is None:
            return None
        try:
            af = float(a)
        except Exception:
            return None
        if not af > 0:
            return None
        return float(af) * phi / eq

    def _is_sell(tkr: str) -> bool:
        tw = float(tmap.get(tkr, 0.0))
        cw = float(cmap.get(tkr, 0.0))
        return abs(tw * _mult(tkr)) < abs(cw * _mult(tkr)) - 1e-15

    after_sell: dict[str, float] = {}
    for tkr in sorted(tickers):
        tw = float(tmap.get(tkr, 0.0))
        cw = float(cmap.get(tkr, 0.0))
        if _is_sell(tkr):
            cap = _max_delta(tkr)
            if cap is None:
                after_sell[tkr] = cw
                continue
            delta = tw - cw
            if abs(delta) <= cap + 1e-15:
                after_sell[tkr] = tw
            else:
                import math as _math

                after_sell[tkr] = cw + _math.copysign(cap, delta)
        else:
            after_sell[tkr] = cw
    gross_after_sell = _sell_first_gross(after_sell, mults)
    residual = float(max_gross) - float(gross_after_sell)
    if not residual > 0:
        return {k: float(v) for k, v in after_sell.items()}
    buy_tickers = sorted(t for t in tickers if not _is_sell(t))
    capped_buys: dict[str, float] = {}
    for tkr in buy_tickers:
        tw = float(tmap.get(tkr, 0.0))
        cw = float(cmap.get(tkr, 0.0))
        cap = _max_delta(tkr)
        if cap is None:
            capped_buys[tkr] = cw
            continue
        delta = tw - cw
        if delta <= cap + 1e-15:
            capped_buys[tkr] = tw
        else:
            capped_buys[tkr] = cw + cap
    candidate = dict(after_sell)
    for tkr in buy_tickers:
        candidate[tkr] = float(capped_buys[tkr])
    if _sell_first_gross(candidate, mults) <= float(max_gross) + 1e-9:
        return {k: float(v) for k, v in candidate.items()}
    deltas = {t: float(capped_buys[t]) - float(cmap.get(t, 0.0)) for t in buy_tickers}
    lo, hi = 0.0, 1.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        trial = dict(after_sell)
        for tkr in buy_tickers:
            trial[tkr] = float(cmap.get(tkr, 0.0)) + float(deltas[tkr]) * mid
        if _sell_first_gross(trial, mults) <= float(max_gross) + 1e-9:
            lo = mid
        else:
            hi = mid
    out = dict(after_sell)
    for tkr in buy_tickers:
        out[tkr] = float(cmap.get(tkr, 0.0)) + float(deltas[tkr]) * lo
    return {k: float(v) for k, v in out.items()}


def cap_target_weights_by_adv(
    target: Mapping[str, float],
    current: Mapping[str, float],
    equity: float,
    adv_by_ticker: Mapping[str, float],
    max_order_to_adv: float,
) -> dict[str, float]:
    if equity <= 0 or max_order_to_adv <= 0:
        return {k: float(v) for k, v in target.items()}

    out: dict[str, float] = {}
    tickers = set(target.keys()) | set(current.keys())
    for ticker in sorted(tickers):
        tw = float(target.get(ticker, 0.0))
        cw = float(current.get(ticker, 0.0))
        delta = tw - cw
        if delta == 0.0:
            out[ticker] = cw
            continue
        adv = adv_by_ticker.get(ticker)
        if adv is None or adv <= 0:
            out[ticker] = cw
            continue
        max_notional = float(adv) * float(max_order_to_adv)
        max_delta_w = max_notional / float(equity)
        if abs(delta) <= max_delta_w + 1e-15:
            out[ticker] = tw
        else:
            out[ticker] = cw + math.copysign(max_delta_w, delta)
    return out
