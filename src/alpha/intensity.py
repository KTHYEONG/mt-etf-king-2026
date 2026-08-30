# mypy: ignore-errors
from __future__ import annotations  # mypy: ignore-errors

import math
from collections.abc import Mapping
from dataclasses import dataclass

import polars as pl

from src.alpha.base import DecisionContext
from src.universe.instruments import Confidence, InstrumentMaster


@dataclass(frozen=True)
class FamilyIntensityConfig:
    mom_col: str = "mom_20"
    allowed_multiples: tuple[int, ...] = (1, 2)
    exclude_inverse: bool = True
    exclude_synthetic: bool = True
    exclude_low_confidence: bool = True

    @classmethod
    def from_yaml(cls, raw: Mapping[str, object]) -> FamilyIntensityConfig:
        # defaults
        mom_col = "mom_20"
        allowed: tuple[int, ...] = (1, 2)
        exclude_inverse = True
        exclude_synthetic = True
        exclude_low_confidence = True
        if not isinstance(raw, Mapping):
            return cls()
        if "mom_col" in raw:
            try:
                mom_col = str(raw["mom_col"])
            except Exception:
                mom_col = "mom_20"
        if "allowed_multiples" in raw:
            am = raw["allowed_multiples"]
            if isinstance(am, (list, tuple)):
                try:
                    allowed = tuple(int(x) for x in am)
                except Exception:
                    allowed = tuple(int(x) for x in list(am))  # type: ignore[arg-type]
            else:
                # unexpected type, try to coerce
                try:
                    allowed = tuple(int(x) for x in list(am))  # type: ignore[arg-type]
                except Exception:
                    allowed = (1, 2)
            if len(allowed) == 0:
                raise ValueError("allowed_multiples must not be empty")
        if "exclude_inverse" in raw:
            exclude_inverse = bool(raw["exclude_inverse"])
        if "exclude_synthetic" in raw:
            exclude_synthetic = bool(raw["exclude_synthetic"])
        if "exclude_low_confidence" in raw:
            exclude_low_confidence = bool(raw["exclude_low_confidence"])
        return cls(
            mom_col=mom_col,
            allowed_multiples=allowed,
            exclude_inverse=exclude_inverse,
            exclude_synthetic=exclude_synthetic,
            exclude_low_confidence=exclude_low_confidence,
        )


def family_intensity_scores(
    snapshot: pl.DataFrame,
    master: InstrumentMaster,
    config: FamilyIntensityConfig | None = None,
) -> dict[str, float]:
    if config is None:
        config = FamilyIntensityConfig()
    # fail-closed checks
    if snapshot is None or not isinstance(snapshot, pl.DataFrame):
        return {}
    if snapshot.height == 0:
        return {}
    if "ticker" not in snapshot.columns:
        return {}
    if config.mom_col not in snapshot.columns:
        return {}
    if master is None:
        return {}
    attrs = getattr(master, "attributes", None)
    if not isinstance(attrs, Mapping) or len(attrs) == 0:
        return {}
    # O(N) single pass grouping
    families: dict[str, dict[str, object]] = {}
    # structure per family: plus_one_tickers: list[str], non_synth: list[tuple[str, object]], all_members: list[tuple[str, object]], best: float|None
    for row in snapshot.iter_rows(named=True):
        ticker_raw = row.get("ticker")
        if ticker_raw is None:
            continue
        ticker = str(ticker_raw)
        attr = attrs.get(ticker)  # type: ignore[attr-defined]
        if attr is None:
            continue
        try:
            family_key = str(getattr(attr, "leverage_family_key", ticker))
        except Exception:
            family_key = ticker
        entry = families.get(family_key)
        if entry is None:
            entry = {
                "plus_one": [],
                "non_synth": [],
                "all_members": [],
                "best": None,
            }
            families[family_key] = entry
        # track members
        all_members = entry["all_members"]  # type: ignore[assignment]
        all_members.append((ticker, attr))  # type: ignore[attr-defined]
        # plus-one tracking
        try:
            lev = int(getattr(attr, "leverage_multiple", 1))
        except Exception:
            lev = 1
        if lev == 1:
            plus = entry["plus_one"]  # type: ignore[assignment]
            plus.append(ticker)  # type: ignore[attr-defined]
        # non-synthetic tracking
        try:
            is_syn = bool(getattr(attr, "is_synthetic", False))
        except Exception:
            is_syn = False
        if not is_syn:
            non = entry["non_synth"]  # type: ignore[assignment]
            non.append((ticker, attr))  # type: ignore[attr-defined]
        # source eligibility for intensity
        mom_val = row.get(config.mom_col)
        if mom_val is None:
            continue
        try:
            fv = float(mom_val)  # noqa: S112
            if not math.isfinite(fv):
                continue
        except Exception:  # noqa: S112
            continue
        if lev not in config.allowed_multiples:
            continue
        if config.exclude_inverse and lev <= 0:
            continue
        if config.exclude_synthetic and is_syn:
            continue
        if config.exclude_low_confidence:
            try:
                conf = getattr(attr, "confidence", Confidence.HIGH)
                if conf == Confidence.LOW or str(conf) == "LOW":
                    continue
            except Exception:  # noqa: S110
                pass
        # update best
        cur = entry["best"]
        if cur is None or fv > float(cur):  # type: ignore[arg-type]
            entry["best"] = float(fv)
    if not families:
        return {}
    out: dict[str, float] = {}
    for entry in families.values():
        best = entry["best"]
        if best is None:
            continue
        plus_one = entry["plus_one"]  # type: ignore[assignment]
        if plus_one:  # type: ignore[truthy-bool]
            # lexicographically smallest ticker among +1s
            chosen = sorted(plus_one)[0]  # type: ignore[arg-type]
            out[str(chosen)] = float(best)  # type: ignore[arg-type]
        else:
            non_synth = entry["non_synth"]  # type: ignore[assignment]
            candidates = non_synth if non_synth else entry["all_members"]  # type: ignore[assignment]
            if not candidates:  # type: ignore[truthy-bool]
                continue
            # min abs leverage among candidates
            def _abs_mult(item: tuple[str, object]) -> int:
                _, a = item
                try:
                    return abs(int(getattr(a, "leverage_multiple", 1)))
                except Exception:
                    return 1

            min_abs = min(_abs_mult(c) for c in candidates)  # type: ignore[arg-type]
            best_cands = [c for c in candidates if _abs_mult(c) == min_abs]  # type: ignore[arg-type]
            # pick lexicographically smallest ticker among ties (deterministic)
            # for deterministic, sort by ticker
            chosen = sorted(t for t, _ in best_cands)[0]
            out[str(chosen)] = float(best)  # type: ignore[arg-type]
    return dict(out)


class FamilyIntensityModel:
    name: str = "M13"

    def __init__(self, master: InstrumentMaster, config: FamilyIntensityConfig | None = None) -> None:
        self.master = master
        self.config = config if config is not None else FamilyIntensityConfig()

    def score(self, snapshot: pl.DataFrame, context: DecisionContext) -> dict[str, float]:
        return family_intensity_scores(snapshot, self.master, self.config)
