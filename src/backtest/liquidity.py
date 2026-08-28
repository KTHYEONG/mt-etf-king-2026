from __future__ import annotations

import math
from collections.abc import Mapping


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
