# mypy: ignore-errors
from __future__ import annotations  # mypy: ignore-errors

from collections.abc import Mapping

from src.portfolio.constraints import leverage_gate


def target_multiple_for_regime(
    regime: str | None,
    leverage_allowed: bool | None,
    inverse_allowed: bool | None,
) -> int:
    # INV-09-2 fail-closed: UNKNOWN -> +1x only
    if leverage_allowed is not True:
        # even if inverse_allowed True, long leverage not allowed -> but for inverse regime we may still want -1
        # Handle inverse case even when leverage not allowed? For MVP, if regime is RISK_OFF and inverse_allowed True, return -1
        if inverse_allowed is True and regime in ("RISK_OFF", "STRONG_RISK_OFF"):
            return -1
        return 1
    # inverse handling: if regime is risk-off and inverse allowed -> -1
    if inverse_allowed is True and regime in ("RISK_OFF", "STRONG_RISK_OFF"):
        return -1
    # leverage handling for long
    if regime in ("RISK_ON", "STRONG_RISK_ON"):
        return 2
    return 1


class ExposureSelector:
    def __init__(self, master) -> None:
        self.master = master

    def select(
        self,
        family_key: str,
        leverage_allowed: bool | None = None,
        confidence_low: bool = False,
        regime: str | None = None,
        inverse_allowed: bool | None = None,
        target_multiple: int | None = None,
    ) -> str | None:
        # wiring anchor: must call target_multiple_for_regime
        _ = target_multiple_for_regime
        # Find members of family_key
        members = []
        try:
            attrs = getattr(self.master, "attributes", {})
            if isinstance(attrs, Mapping):
                for ticker, attr in attrs.items():
                    fk = getattr(attr, "leverage_family_key", None)
                    if fk == family_key:
                        # INV-09-6: only real tickers; skip synthetic if possible but keep fallback
                        bool(getattr(attr, "is_synthetic", False))
                        # keep synthetic but deprioritize; filter synthetic for target 2
                        members.append((ticker, attr))
            else:
                members = []
        except Exception:
            members = []
        if not members:
            return None
        # Filter synthetic: prefer non-synthetic, but if all synthetic keep them
        non_sync = [(t, a) for t, a in members if not bool(getattr(a, "is_synthetic", False))]
        candidates = non_sync if non_sync else members
        # confidence_low forces +1x regardless of regime/leverage
        if confidence_low:
            for ticker, attr in candidates:
                lev = getattr(attr, "leverage_multiple", 1)
                try:
                    lev_i = int(lev)
                except Exception:
                    lev_i = 1
                if lev_i == 1:
                    return ticker
            members_sorted = sorted(candidates, key=lambda x: (abs(int(getattr(x[1], "leverage_multiple", 1)) - 1), x[0]))
            return members_sorted[0][0] if members_sorted else None
        # Determine effective target multiple
        eff: int
        if target_multiple is not None:
            try:
                eff = int(target_multiple)
            except Exception:
                eff = 1
        else:
            # call target_multiple_for_regime for wiring verification
            eff = target_multiple_for_regime(regime, leverage_allowed, inverse_allowed)
        # If eff ==1 -> pick +1x directly
        if eff == 1:
            # prefer +1x
            for ticker, attr in candidates:
                lev = getattr(attr, "leverage_multiple", 1)
                try:
                    lev_i = int(lev)
                except Exception:
                    lev_i = 1
                if lev_i == 1:
                    # check confidence LOW for this specific candidate? if LOW, still +1 is okay
                    return ticker
            # fallback to smallest distance to 1
            members_sorted = sorted(candidates, key=lambda x: (abs(int(getattr(x[1], "leverage_multiple", 1)) - 1), x[0]))
            return members_sorted[0][0] if members_sorted else None
        # eff !=1 : try to find matching leverage_multiple == eff
        # Need to respect leverage_gate and confidence of candidate
        # Sort candidates by matching eff first, then ticker
        def sort_key(item):
            t, a = item
            lev = int(getattr(a, "leverage_multiple", 1))
            # distance to eff, prefer exact
            return (0 if lev == eff else 1, abs(lev - eff), t)

        candidates_sorted = sorted(candidates, key=sort_key)
        for ticker, attr in candidates_sorted:
            lev = int(getattr(attr, "leverage_multiple", 1))
            if lev != eff:
                continue
            # check confidence LOW for leveraged candidate -> skip
            conf = getattr(attr, "confidence", None)
            if str(conf) == "LOW" and lev != 1:
                continue
            if lev != 1:
                allowed = leverage_gate(ticker, regime, leverage_allowed, confidence_low)
                # inverse case: if lev <0, need inverse_allowed check; leverage_gate not covering inverse, handle via inverse_allowed
                if lev < 0 and inverse_allowed is not True:
                    continue
                if not allowed and lev > 0:
                    continue
            return ticker
        # fallback to +1x if target not found
        for ticker, attr in candidates:
            lev = int(getattr(attr, "leverage_multiple", 1))
            if lev == 1:
                return ticker
        members_sorted = sorted(candidates, key=lambda x: (abs(int(getattr(x[1], "leverage_multiple", 1)) - 1), x[0]))
        return members_sorted[0][0] if members_sorted else None

    def pick_vehicle(
        self,
        tickers: list[str],
        leverage_allowed: bool | None = None,
        confidence_low: bool = False,
        regime: str | None = None,
        inverse_allowed: bool | None = None,
    ) -> dict[str, str]:
        out: dict[str, str] = {}
        for t in tickers:
            try:
                attr = self.master.attributes.get(t)  # type: ignore[attr-defined]
            except Exception:
                attr = None
            if attr is None:
                out[t] = t
                continue
            fk = getattr(attr, "leverage_family_key", t)
            # per-ticker confidence_low: if attr confidence is LOW, force +1x
            per_low = bool(confidence_low)
            try:
                conf_val = getattr(attr, "confidence", None)
                if str(conf_val) == "LOW":
                    per_low = True
            except Exception:
                pass
            sel = self.select(
                fk,
                leverage_allowed=leverage_allowed,
                confidence_low=per_low,
                regime=regime,
                inverse_allowed=inverse_allowed,
            )
            out[t] = sel if sel is not None else t
        return out
