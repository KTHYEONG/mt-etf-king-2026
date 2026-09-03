# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from pathlib import Path

import polars as pl

from src.strategies.sticky.model import StickyLeaderConfig

logger = logging.getLogger(__name__)

def resolve_lock_level(value: object, *, default: float = 0.50) -> float:
    try:
        ll = float(value)  # type: ignore[arg-type]
        if not math.isfinite(ll) or ll < 0:
            return float(default)
        return float(ll)
    except Exception:
        return float(default)


def load_p22_lock_level(*, default: float = 0.50) -> float:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return float(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return float(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return float(default)
        p22 = port.get("p22")
        if not isinstance(p22, Mapping) or "lock_level" not in p22:
            return float(default)
        return resolve_lock_level(p22["lock_level"], default=default)
    except Exception:
        return float(default)


def load_p24_lock_level(*, default: float = 0.50) -> float:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return float(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return float(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return float(default)
        p24 = port.get("p24")
        if not isinstance(p24, Mapping) or "lock_level" not in p24:
            return float(default)
        return resolve_lock_level(p24["lock_level"], default=default)
    except Exception:
        return float(default)


def load_p24_trail(*, default: float = 0.0) -> float:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return float(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return float(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return float(default)
        p24 = port.get("p24")
        if not isinstance(p24, Mapping) or "trail" not in p24:
            # also support trail_level alias
            if isinstance(p24, Mapping) and "trail_level" in p24:
                v = p24["trail_level"]
                try:
                    fv = float(v)  # type: ignore[arg-type]
                    if not math.isfinite(fv) or fv < 0:
                        return float(default)
                    return float(fv)
                except Exception:
                    return float(default)
            return float(default)
        v = p24["trail"]
        try:
            fv = float(v)  # type: ignore[arg-type]
            if not math.isfinite(fv) or fv < 0:
                return float(default)
            return float(fv)
        except Exception:
            return float(default)
    except Exception:
        return float(default)


def load_p25_arm(*, default: float = 0.50) -> float:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return float(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return float(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return float(default)
        p25 = port.get("p25")
        if not isinstance(p25, Mapping) or "arm" not in p25:
            return float(default)
        v = p25["arm"]
        try:
            fv = float(v)  # type: ignore[arg-type]
            if not math.isfinite(fv) or fv <= 0:
                return float(default)
            return float(fv)
        except Exception:
            return float(default)
    except Exception:
        return float(default)


def load_p25_lock_remaining(*, default: int = 5) -> int:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return int(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return int(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return int(default)
        p25 = port.get("p25")
        if not isinstance(p25, Mapping) or "lock_remaining" not in p25:
            return int(default)
        v = p25["lock_remaining"]
        try:
            fv = float(v)  # type: ignore[arg-type]
            if not math.isfinite(fv) or fv < 0:
                return int(default)
            return int(fv)
        except Exception:
            return int(default)
    except Exception:
        return int(default)


def load_p26_arm(*, default: float = 0.50) -> float:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return float(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return float(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return float(default)
        p26 = port.get("p26")
        if not isinstance(p26, Mapping) or "arm" not in p26:
            return float(default)
        v = p26["arm"]
        try:
            fv = float(v)  # type: ignore[arg-type]
            if not math.isfinite(fv) or fv <= 0:
                return float(default)
            return float(fv)
        except Exception:
            return float(default)
    except Exception:
        return float(default)


def load_p26_lock_remaining(*, default: int = 5) -> int:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return int(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return int(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return int(default)
        p26 = port.get("p26")
        if not isinstance(p26, Mapping) or "lock_remaining" not in p26:
            return int(default)
        v = p26["lock_remaining"]
        try:
            fv = float(v)  # type: ignore[arg-type]
            if not math.isfinite(fv) or fv < 0:
                return int(default)
            return int(fv)
        except Exception:
            return int(default)
    except Exception:
        return int(default)


def load_p27_overlay_mode(*, default: str = "identity") -> str:
    from src.strategies.ids import STICKY_MOM60_RAW
    from src.strategies.sticky.config import load_overlay_mode

    return load_overlay_mode(strategy_key=STICKY_MOM60_RAW, default=default)


def load_p24_mom_col(*, default: str = "mom_60") -> str:
    try:
        from pathlib import Path

        import yaml

        fp = Path("configs/strategies.yaml")
        if not fp.exists():
            return str(default)
        with open(fp, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            return str(default)
        port = raw.get("portfolio")
        if not isinstance(port, dict):
            return str(default)
        p24 = port.get("p24")
        if not isinstance(p24, Mapping) or "mom_col" not in p24:
            return str(default)
        v = p24["mom_col"]
        if not isinstance(v, str) or not v.strip():
            return str(default)
        s = str(v).strip()
        # allow only mom_ prefixed? fallback for invalid names still allow but require non-empty
        if not s:
            return str(default)
        try:
            # if value is numeric-like string that is nan/inf, treat as invalid
            fv = float(s)  # type: ignore[arg-type]
            if math.isfinite(fv):
                # if s is numeric it is invalid for mom_col
                return str(default)
        except Exception:
            pass
        # validate mom_col pattern: startswith mom_
        if not s.startswith("mom_"):
            return str(default)
        return str(s)
    except Exception:
        return str(default)


def apply_impulse_switch(
    scores: Mapping[str, float], held: str | None, snapshot: pl.DataFrame, config: StickyLeaderConfig
) -> dict[str, float]:
    if not scores:
        return {}
    try:
        gap = float(config.impulse_gap)
        if not math.isfinite(gap) or gap <= 0:
            return dict(scores)
    except Exception:
        return dict(scores)
    if held is None or held not in scores:
        return dict(scores)
    if snapshot is None or not isinstance(snapshot, pl.DataFrame):
        return dict(scores)
    try:
        if snapshot.height == 0 or snapshot.width == 0:
            return dict(scores)
    except Exception:
        return dict(scores)
    if config.impulse_col not in snapshot.columns:
        return dict(scores)
    if "ticker" not in snapshot.columns:
        return dict(scores)
    # build row map for relevant tickers
    row_by_ticker: dict[str, dict] = {}
    try:
        for row in snapshot.iter_rows(named=True):
            try:
                t = str(row.get("ticker"))
            except Exception:
                continue
            if t in scores or t == held:
                row_by_ticker[t] = row
    except Exception:
        return dict(scores)
    if held not in row_by_ticker:
        return dict(scores)
    held_row = row_by_ticker[held]
    try:
        held_imp_raw = held_row.get(config.impulse_col)
        held_imp = float(held_imp_raw) if held_imp_raw is not None else float("nan")
    except Exception:
        return dict(scores)
    if not math.isfinite(held_imp):
        return dict(scores)
    candidates = [t for t in scores.keys() if t != held]
    if not candidates:
        return dict(scores)

    def _mom_val(ticker: str) -> float:
        row = row_by_ticker.get(ticker)
        if row is not None and config.mom_col in row and row.get(config.mom_col) is not None:
            try:
                fv = float(row.get(config.mom_col))
                if math.isfinite(fv):
                    return float(fv)
            except Exception:
                pass
        try:
            return float(scores[ticker])
        except Exception:
            return float("-inf")

    candidates_sorted = sorted(candidates, key=lambda t: (-_mom_val(t), str(t)))
    for chal in candidates_sorted:
        row = row_by_ticker.get(chal)
        if row is None:
            continue
        chal_raw = row.get(config.impulse_col)
        try:
            chal_imp = float(chal_raw) if chal_raw is not None else float("nan")
        except Exception:
            continue
        if not math.isfinite(chal_imp):
            continue
        # mom_5 >0 requirement
        mom5_ok = False
        if "mom_5" in snapshot.columns:
            mv = row.get("mom_5")
            try:
                mfv = float(mv) if mv is not None else float("nan")
                if math.isfinite(mfv) and mfv > 0:
                    mom5_ok = True
            except Exception:
                mom5_ok = False
        else:
            # fallback: impulse value >0
            if chal_imp > 0:
                mom5_ok = True
        if not mom5_ok:
            continue
        if config.impulse_require_volx:
            vol_raw = row.get("volume_expansion")
            try:
                vfv = float(vol_raw) if vol_raw is not None else 0.0
                if not math.isfinite(vfv) or vfv <= 0:
                    continue
            except Exception:
                continue
        if chal_imp - held_imp >= float(gap) - 1e-12:
            out = dict(scores)
            try:
                held_score = float(out[held])
            except Exception:
                try:
                    held_score = float(max(out.values()))
                except Exception:
                    held_score = 0.0
            out[chal] = float(held_score) + 1e-6
            return out
    return dict(scores)


def apply_crash_cash(
    scores: Mapping[str, float], held: str | None, snapshot: pl.DataFrame, config: StickyLeaderConfig
) -> dict[str, float] | object:
    if not scores:
        return {}
    try:
        cd = float(config.cash_drawdown)
        if not math.isfinite(cd) or cd >= 0:
            return dict(scores)
    except Exception:
        return dict(scores)
    if held is None or held not in scores:
        return dict(scores)
    if snapshot is None or not isinstance(snapshot, pl.DataFrame):
        return dict(scores)
    try:
        if snapshot.height == 0:
            return dict(scores)
    except Exception:
        return dict(scores)
    if "drawdown_20" not in snapshot.columns or "ticker" not in snapshot.columns:
        return dict(scores)
    held_dd = None
    try:
        for row in snapshot.iter_rows(named=True):
            if str(row.get("ticker")) == held:
                held_dd = row.get("drawdown_20")
                break
    except Exception:
        return dict(scores)
    if held_dd is None:
        return dict(scores)
    try:
        hf = float(held_dd)
    except Exception:
        return dict(scores)
    if not math.isfinite(hf):
        return dict(scores)
    if hf < float(cd) - 1e-12:
        try:
            from src.portfolio.intent import CASH_INTENT

            return CASH_INTENT
        except Exception:
            return {}
    return dict(scores)


def apply_abs_mom_cash(scores: Mapping[str, float] | object, config: StickyLeaderConfig) -> dict[str, float] | object:
    if not bool(getattr(config, "abs_mom_cash", False)): return scores
    try:
        from src.portfolio.intent import PortfolioIntent as _PI
        if isinstance(scores, _PI):
            return scores
    except Exception:
        pass
    try:
        if scores is not None and hasattr(scores, "kind"):
            k = getattr(scores, "kind", None)
            if k in ("cash", "hold", "target"):
                return scores
            try:
                from src.portfolio.intent import PortfolioIntent as _PI2
                if isinstance(scores, _PI2):
                    return scores
            except Exception:
                pass
    except Exception:
        pass
    if not isinstance(scores, Mapping):
        return scores
    try:
        if len(scores) == 0:  # type: ignore[arg-type]
            from src.portfolio.intent import CASH_INTENT

            return CASH_INTENT
    except Exception:
        try:
            from src.portfolio.intent import CASH_INTENT

            return CASH_INTENT
        except Exception:
            return scores
    max_finite: float | None = None
    has_finite = False
    for v in scores.values():  # type: ignore[union-attr]
        try:
            fv = float(v)  # type: ignore[arg-type]
            if math.isfinite(fv):
                has_finite = True
                if max_finite is None or fv > max_finite:
                    max_finite = float(fv)
        except Exception:
            continue
    if not has_finite:
        try:
            from src.portfolio.intent import CASH_INTENT

            return CASH_INTENT
        except Exception:
            return scores
    if max_finite is not None and float(max_finite) <= 0.0:
        try:
            from src.portfolio.intent import CASH_INTENT

            return CASH_INTENT
        except Exception:
            return scores
    try:
        return dict(scores)  # type: ignore[arg-type]
    except Exception:
        return scores


def apply_same_leader_hold(scores: Mapping[str, float] | object, held: str | None, enabled: bool) -> dict[str, float] | object:
    try:
        from src.portfolio.intent import HOLD_INTENT
    except Exception:
        HOLD_INTENT = None  # type: ignore[assignment]
    # FAIL-CLOSED: if scores is PortfolioIntent, return unchanged
    try:
        if scores is not None and hasattr(scores, "kind"):
            k = getattr(scores, "kind", None)
            if k in ("cash", "hold", "target"):
                return scores
            # also check instance of PortfolioIntent generically
            try:
                from src.portfolio.intent import PortfolioIntent as _PI
                if isinstance(scores, _PI):
                    return scores
            except Exception:
                pass
    except Exception:
        pass
    # Also if not Mapping, treat as intent
    if not isinstance(scores, Mapping):
        return scores
    if not bool(enabled):
        try:
            return dict(scores)  # type: ignore[arg-type]
        except Exception:
            return scores
    # empty mapping -> {}
    try:
        if len(scores) == 0:  # type: ignore[arg-type]
            return {}
    except Exception:
        try:
            return dict(scores)  # type: ignore[arg-type]
        except Exception:
            return scores
    # held None or blank -> return dict(scores)
    if held is None:
        try:
            return dict(scores)  # type: ignore[arg-type]
        except Exception:
            return scores
    try:
        hs = str(held).strip()
        if not hs:
            return dict(scores)  # type: ignore[arg-type]
    except Exception:
        try:
            return dict(scores)  # type: ignore[arg-type]
        except Exception:
            return scores
    # leader determination with non-finite skip
    try:
        finite_items: list[tuple[str, float]] = []
        for k, v in scores.items():  # type: ignore[union-attr]
            try:
                fv = float(v)  # type: ignore[arg-type]
                if math.isfinite(fv):
                    finite_items.append((str(k), float(fv)))
            except Exception:
                continue
        if not finite_items:
            return dict(scores)  # type: ignore[arg-type]
        sorted_items = sorted(finite_items, key=lambda kv: (-float(kv[1]), str(kv[0])))
        leader = str(sorted_items[0][0])
    except Exception:
        try:
            return dict(scores)  # type: ignore[arg-type]
        except Exception:
            return scores
    try:
        if str(leader) == str(held).strip() and bool(enabled):
            if HOLD_INTENT is not None:
                return HOLD_INTENT
            from src.portfolio.intent import HOLD_INTENT as _HI

            return _HI
    except Exception:
        pass
    try:
        return dict(scores)  # type: ignore[arg-type]
    except Exception:
        return scores

