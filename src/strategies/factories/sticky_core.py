# mypy: ignore-errors
# ruff: noqa
"""Core (leader-base / impulse-crash / family-peak-lock) sticky-leader strategy factories (P5 split of factories/sticky.py)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.sticky.model import StickyLeaderModel


def make_sticky_leader_base() -> StickyLeaderModel:
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.sticky import StickyLeaderConfig, StickyLeaderModel

    cfg = StickyLeaderConfig()
    try:
        fp = _P("configs/strategies.yaml")
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                raw = _yaml.safe_load(f) or {}
            sticky_raw = None
            if isinstance(raw, dict):
                sticky_raw = raw.get("sticky_leader")
                if sticky_raw is None and isinstance(raw.get("portfolio"), dict):
                    sticky_raw = raw["portfolio"].get("sticky_leader")  # type: ignore[index]
            if isinstance(sticky_raw, Mapping):
                cfg = StickyLeaderConfig.from_yaml(sticky_raw)  # type: ignore[arg-type]
    except Exception:
        cfg = StickyLeaderConfig()
    return StickyLeaderModel(name="sticky.leader_base", config=cfg)


def make_sticky_impulse_crash() -> StickyLeaderModel:
    import math as _math
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.sticky import StickyLeaderConfig, StickyLeaderModel

    cfg = StickyLeaderConfig()
    raw: dict = {}
    try:
        fp = _P("configs/strategies.yaml")
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                raw = _yaml.safe_load(f) or {}
            sticky_raw = None
            if isinstance(raw, dict):
                sticky_raw = raw.get("sticky_leader")
                if sticky_raw is None and isinstance(raw.get("portfolio"), dict):
                    sticky_raw = raw["portfolio"].get("sticky_leader")  # type: ignore[index]
            if isinstance(sticky_raw, Mapping):
                cfg = StickyLeaderConfig.from_yaml(sticky_raw)  # type: ignore[arg-type]
    except Exception:
        cfg = StickyLeaderConfig()
        raw = {}
    # overlay portfolio.p21
    try:
        p21_raw = None
        if isinstance(raw, dict):
            port = raw.get("portfolio")
            if isinstance(port, dict):
                p21_raw = port.get("p21")
        if isinstance(p21_raw, Mapping):
            if "impulse_gap" in p21_raw:
                try:
                    ig = float(p21_raw["impulse_gap"])  # type: ignore[arg-type]
                    if not _math.isfinite(ig) or ig < 0:
                        cfg.impulse_gap = 0.0
                    else:
                        cfg.impulse_gap = float(ig)
                except Exception:
                    cfg.impulse_gap = 0.0
            else:
                if cfg.impulse_gap == 0.0:
                    cfg.impulse_gap = 0.04
            if "impulse_require_volx" in p21_raw:
                try:
                    cfg.impulse_require_volx = bool(p21_raw["impulse_require_volx"])
                except Exception:
                    cfg.impulse_require_volx = True
            else:
                cfg.impulse_require_volx = True
            if "impulse_col" in p21_raw:
                try:
                    v = p21_raw["impulse_col"]
                    if isinstance(v, str) and v.strip():
                        cfg.impulse_col = str(v).strip()
                    else:
                        cfg.impulse_col = "mom_5"
                except Exception:
                    cfg.impulse_col = "mom_5"
            if "cash_drawdown" in p21_raw:
                try:
                    cd = float(p21_raw["cash_drawdown"])  # type: ignore[arg-type]
                    if not _math.isfinite(cd) or cd > 0:
                        cfg.cash_drawdown = 0.0
                    else:
                        cfg.cash_drawdown = float(cd)
                except Exception:
                    cfg.cash_drawdown = 0.0
            else:
                if cfg.cash_drawdown == 0.0:
                    cfg.cash_drawdown = -0.12
        else:
            # no p21 block: default P21 overlay
            if cfg.impulse_gap == 0.0:
                cfg.impulse_gap = 0.04
            cfg.impulse_require_volx = True
            if cfg.cash_drawdown == 0.0:
                cfg.cash_drawdown = -0.12
            if not cfg.impulse_col or cfg.impulse_col.strip() == "":
                cfg.impulse_col = "mom_5"
    except Exception:
        if cfg.impulse_gap == 0.0:
            cfg.impulse_gap = 0.04
        cfg.impulse_require_volx = True
        if cfg.cash_drawdown == 0.0:
            cfg.cash_drawdown = -0.12
    return StickyLeaderModel(name="sticky.impulse_crash", config=cfg)


def make_sticky_family_peak_lock() -> StickyLeaderModel:
    import math as _math
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.sticky import StickyLeaderConfig, StickyLeaderModel, resolve_lock_level

    cfg = StickyLeaderConfig()
    cfg.lock_level = 0.50
    raw: dict = {}
    try:
        fp = _P("configs/strategies.yaml")
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                raw = _yaml.safe_load(f) or {}
            sticky_raw = None
            if isinstance(raw, dict):
                sticky_raw = raw.get("sticky_leader")
                if sticky_raw is None and isinstance(raw.get("portfolio"), dict):
                    sticky_raw = raw["portfolio"].get("sticky_leader")  # type: ignore[index]
            if isinstance(sticky_raw, Mapping):
                cfg = StickyLeaderConfig.from_yaml(sticky_raw)  # type: ignore[arg-type]
    except Exception:
        cfg = StickyLeaderConfig()
        raw = {}
    # Enforce P22 defaults
    cfg.min_gap = 0.0
    cfg.min_hold = 0
    cfg.collapse_family = True
    cfg.impulse_gap = 0.0
    cfg.cash_drawdown = 0.0
    cfg.only_plus_2 = True
    cfg.no_inverse = True
    # overlay portfolio.p22 fail-closed
    try:
        p22_raw = None
        if isinstance(raw, dict):
            port = raw.get("portfolio")
            if isinstance(port, dict):
                p22_raw = port.get("p22")
        if isinstance(p22_raw, Mapping):
            if "min_gap" in p22_raw:
                try:
                    mg = float(p22_raw["min_gap"])  # type: ignore[arg-type]
                    if not _math.isfinite(mg) or mg < 0:
                        cfg.min_gap = 0.0
                    else:
                        cfg.min_gap = float(mg)
                except Exception:
                    cfg.min_gap = 0.0
            else:
                cfg.min_gap = 0.0
            if "min_hold" in p22_raw:
                try:
                    mh_raw = p22_raw["min_hold"]
                    mh = int(mh_raw)  # type: ignore[arg-type]
                    try:
                        f = float(mh_raw)  # type: ignore[arg-type]
                        if not _math.isfinite(f):
                            raise ValueError
                    except Exception:
                        raise
                    if mh < 0:
                        cfg.min_hold = 0
                    else:
                        cfg.min_hold = int(mh)
                except Exception:
                    cfg.min_hold = 0
            else:
                cfg.min_hold = 0
            if "collapse_family" in p22_raw:
                try:
                    cfg.collapse_family = bool(p22_raw["collapse_family"])
                except Exception:
                    cfg.collapse_family = True
            else:
                cfg.collapse_family = True
            # impulse and cash must remain disabled
            cfg.impulse_gap = 0.0
            cfg.cash_drawdown = 0.0
            if "lock_level" in p22_raw:
                cfg.lock_level = resolve_lock_level(p22_raw["lock_level"], default=0.50)
            else:
                cfg.lock_level = 0.50
        else:
            # missing p22 block, keep defaults
            cfg.min_gap = 0.0
            cfg.min_hold = 0
            cfg.collapse_family = True
            cfg.impulse_gap = 0.0
            cfg.cash_drawdown = 0.0
            cfg.lock_level = 0.50
    except Exception:
        cfg.min_gap = 0.0
        cfg.min_hold = 0
        cfg.collapse_family = True
        cfg.impulse_gap = 0.0
        cfg.cash_drawdown = 0.0
        cfg.lock_level = 0.50
    # final fail-closed ensure
    if not _math.isfinite(float(cfg.min_gap)) or float(cfg.min_gap) < 0:
        cfg.min_gap = 0.0
    if not _math.isfinite(float(cfg.min_hold)) or int(cfg.min_hold) < 0:
        cfg.min_hold = 0
    cfg.collapse_family = bool(cfg.collapse_family)
    cfg.impulse_gap = 0.0
    cfg.cash_drawdown = 0.0
    cfg.only_plus_2 = True
    cfg.no_inverse = True
    cfg.lock_level = resolve_lock_level(cfg.lock_level, default=0.50)
    return StickyLeaderModel(name="sticky.family_peak_lock", config=cfg)
