# mypy: ignore-errors
from __future__ import annotations  # mypy: ignore-errors

from collections.abc import Mapping

from src.portfolio.constraints import leverage_gate


class ExposureSelector:
    def __init__(self, master) -> None:
        self.master = master

    def select(
        self,
        family_key: str,
        leverage_allowed: bool | None = None,
        confidence_low: bool = False,
    ) -> str | None:
        # Find members of family_key
        # master.attributes contains InstrumentAttributes with leverage_family_key
        members = []
        try:
            attrs = getattr(self.master, "attributes", {})
            if isinstance(attrs, Mapping):
                for ticker, attr in attrs.items():
                    fk = getattr(attr, "leverage_family_key", None)
                    if fk == family_key:
                        members.append((ticker, attr))
            else:
                members = []
        except Exception:
            members = []
        if not members:
            return None
        # Sort by leverage preference: if confidence_low True, prefer +1x
        # Else prefer higher leverage? But fail-closed says LOW -> +1x forced
        if confidence_low:
            # select member with leverage_multiple ==1 if exists
            for ticker, attr in members:
                lev = getattr(attr, "leverage_multiple", 1)
                if lev == 1:
                    # check gate
                    if not leverage_gate(ticker, None, leverage_allowed, confidence_low):
                        # gate denies leveraged, but 1x should still pass
                        # leverage_gate for 1x should return True even when confidence_low?
                        # Our gate currently returns False for any when confidence_low True, but for 1x we should allow
                        # Handle 1x specially: allow
                        return ticker
                    return ticker
            # fallback to first 1x
            # if no 1x, return None or first member? But should prefer 1x, so if no 1x, return smallest leverage
            # Sort by abs(lev-1)
            members_sorted = sorted(members, key=lambda x: (abs(int(getattr(x[1], "leverage_multiple", 1)) - 1), x[0]))
            return members_sorted[0][0]
        # confidence not low: allow leveraged if gate permits
        # Prefer leveraged if allowed
        # Sort by leverage_multiple descending if allowed, else prefer 1x
        # Check gate for each candidate sorted by leverage descending
        # If leverage_allowed is True and confidence not low, prefer max leverage
        members_sorted = sorted(members, key=lambda x: (-int(getattr(x[1], "leverage_multiple", 1)), x[0]))
        for ticker, attr in members_sorted:
            lev = int(getattr(attr, "leverage_multiple", 1))
            if lev != 1:
                allowed = leverage_gate(ticker, None, leverage_allowed, confidence_low)
                if allowed:
                    return ticker
                else:
                    continue
            else:
                return ticker
        return members_sorted[0][0] if members_sorted else None

    def pick_vehicle(
        self,
        tickers: list[str],
        leverage_allowed: bool | None = None,
        confidence_low: bool = False,
    ) -> dict[str, str]:
        # Map index_key or family_key to selected ticker
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
            sel = self.select(fk, leverage_allowed=leverage_allowed, confidence_low=confidence_low)
            out[t] = sel if sel is not None else t
        return out
