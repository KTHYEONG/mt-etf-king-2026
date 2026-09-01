from __future__ import annotations

from collections.abc import Mapping


def compute_next_open_session_return(
    *,
    weights_before_open: Mapping[str, float],
    weights_after_open: Mapping[str, float],
    prev_closes: Mapping[str, float],
    opens: Mapping[str, float],
    closes: Mapping[str, float],
) -> tuple[float, float, float]:
    overnight = 0.0
    for tkr, w in weights_before_open.items():
        try:
            wf = float(w)
        except Exception:  # noqa: S112
            continue
        if abs(wf) < 1e-15:
            continue
        pc = prev_closes.get(tkr)
        op = opens.get(tkr)
        if pc is None or op is None:
            continue
        try:
            pcf = float(pc)
            opf = float(op)
        except Exception:  # noqa: S112
            continue
        if pcf == 0 or pcf != pcf or opf != opf:
            continue
        if not __import__("math").isfinite(pcf) or not __import__("math").isfinite(opf):
            continue
        overnight += wf * (opf / pcf - 1.0)
    intraday = 0.0
    for tkr, w in weights_after_open.items():
        try:
            wf = float(w)
        except Exception:  # noqa: S112
            continue
        if abs(wf) < 1e-15:
            continue
        op = opens.get(tkr)
        cl = closes.get(tkr)
        if op is None or cl is None:
            continue
        try:
            opf = float(op)
            clf = float(cl)
        except Exception:  # noqa: S112
            continue
        if opf == 0 or opf != opf or clf != clf:
            continue
        if not __import__("math").isfinite(opf) or not __import__("math").isfinite(clf):
            continue
        intraday += wf * (clf / opf - 1.0)
    effective = (1.0 + overnight) * (1.0 + intraday) - 1.0
    return float(overnight), float(intraday), float(effective)
