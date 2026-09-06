# mypy: ignore-errors
# ruff: noqa
"""Lock-overlay (split-fill / mom60-peak / house-money) sticky-leader strategy factories (P5 split of factories/sticky.py)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.sticky.model import StickyLeaderModel


def make_sticky_split_fill_lock() -> object:
    import math as _math
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.sticky import StickyLeaderConfig
    from src.strategies.sticky.split_fill_model import SplitFillStickyModel

    # start from P21 config
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
    # overlay P21
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
                    cfg.impulse_gap = 0.0 if not _math.isfinite(ig) or ig < 0 else float(ig)
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
                    cfg.impulse_col = str(v).strip() if isinstance(v, str) and v.strip() else "mom_5"
                except Exception:
                    cfg.impulse_col = "mom_5"
            if "cash_drawdown" in p21_raw:
                try:
                    cd = float(p21_raw["cash_drawdown"])  # type: ignore[arg-type]
                    cfg.cash_drawdown = 0.0 if not _math.isfinite(cd) or cd > 0 else float(cd)
                except Exception:
                    cfg.cash_drawdown = 0.0
            else:
                if cfg.cash_drawdown == 0.0:
                    cfg.cash_drawdown = -0.12
        else:
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
    # overlay portfolio.p23
    try:
        p23_raw = None
        if isinstance(raw, dict):
            port = raw.get("portfolio")
            if isinstance(port, dict):
                p23_raw = port.get("p23")
        if isinstance(p23_raw, Mapping):
            if "collapse_family" in p23_raw:
                try:
                    cfg.collapse_family = bool(p23_raw["collapse_family"])
                except Exception:
                    cfg.collapse_family = True
            else:
                cfg.collapse_family = True
            if "lock_level" in p23_raw:
                from src.alpha.sticky import resolve_lock_level

                cfg.lock_level = resolve_lock_level(p23_raw["lock_level"], default=0.40)
            else:
                cfg.lock_level = 0.40
            # score_max_order_to_adv is universe filter, not sticky, wiring via yaml anchor score_max_order_to_adv
        else:
            cfg.collapse_family = True
            cfg.lock_level = 0.40
    except Exception:
        cfg.collapse_family = True
        cfg.lock_level = 0.40
    # enforce P23 invariants
    cfg.min_gap = 0.08
    cfg.min_hold = 3
    if not _math.isfinite(float(cfg.impulse_gap)) or float(cfg.impulse_gap) < 0:
        cfg.impulse_gap = 0.04
    if cfg.cash_drawdown == 0.0:
        cfg.cash_drawdown = -0.12
    cfg.collapse_family = True
    from src.alpha.sticky import resolve_lock_level as _rl

    cfg.lock_level = _rl(cfg.lock_level, default=0.40)
    # ensure lock_level exactly 0.40 if not explicitly set differently? Contract says 0.40
    # keep 0.40 unless yaml overrides but wiring expects 0.40
    if abs(float(cfg.lock_level) - 0.40) > 1e-9:
        # still enforce 0.40 per spec (P23 lock_level 0.40)
        cfg.lock_level = 0.40
    return SplitFillStickyModel(name="sticky.split_fill_lock", config=cfg)


def make_sticky_mom60_peak_lock() -> StickyLeaderModel:
    import math as _math
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.sticky import StickyLeaderConfig, StickyLeaderModel, overlay_param
    from src.strategies.ids import STICKY_MOM60_PEAK_LOCK

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
    # overlay portfolio.p21 baseline (impulse/crash/sticky) as P24 shares P21 dynamics
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
    # enforce P24 invariants: P21 impulse/crash/sticky
    cfg.min_gap = 0.08
    cfg.min_hold = 3
    cfg.only_plus_2 = True
    cfg.no_inverse = True
    cfg.collapse_family = False
    if cfg.impulse_gap == 0.0 or not _math.isfinite(float(cfg.impulse_gap)) or float(cfg.impulse_gap) < 0:
        cfg.impulse_gap = 0.04
    cfg.impulse_require_volx = True
    if cfg.cash_drawdown == 0.0 or not _math.isfinite(float(cfg.cash_drawdown)):
        cfg.cash_drawdown = -0.12
    # mom_col from sticky.mom60_peak_lock.mom_col with fail-closed to mom_60
    try:
        mom = overlay_param(STICKY_MOM60_PEAK_LOCK, "mom_col", default="mom_60")
        if not isinstance(mom, str) or not mom.strip() or not mom.startswith("mom_"):
            cfg.mom_col = "mom_60"
        else:
            cfg.mom_col = str(mom)
    except Exception:
        cfg.mom_col = "mom_60"
    # lock_level default 0.50 with NaN fail-closed
    try:
        lk = overlay_param(STICKY_MOM60_PEAK_LOCK, "lock_level", default=0.50)
        cfg.lock_level = float(lk)
    except Exception:
        cfg.lock_level = 0.50
    # final fail-closed ensure
    if not _math.isfinite(float(cfg.lock_level)) or float(cfg.lock_level) < 0:
        cfg.lock_level = 0.50
    return StickyLeaderModel(name="sticky.mom60_peak_lock", config=cfg)


def make_sticky_house_money() -> StickyLeaderModel:
    import math as _math
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.sticky import StickyLeaderConfig, StickyLeaderModel, overlay_param
    from src.strategies.ids import STICKY_MOM60_PEAK_LOCK

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
    # overlay portfolio.p21 baseline (impulse/crash/sticky) as P25 shares P21 dynamics via P24
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
    # enforce P25 invariants identical to P24
    cfg.min_gap = 0.08
    cfg.min_hold = 3
    cfg.only_plus_2 = True
    cfg.no_inverse = True
    cfg.collapse_family = False
    if cfg.impulse_gap == 0.0 or not _math.isfinite(float(cfg.impulse_gap)) or float(cfg.impulse_gap) < 0:
        cfg.impulse_gap = 0.04
    cfg.impulse_require_volx = True
    if cfg.cash_drawdown == 0.0 or not _math.isfinite(float(cfg.cash_drawdown)):
        cfg.cash_drawdown = -0.12
    try:
        mom = overlay_param(STICKY_MOM60_PEAK_LOCK, "mom_col", default="mom_60")
        if not isinstance(mom, str) or not mom.strip() or not mom.startswith("mom_"):
            cfg.mom_col = "mom_60"
        else:
            cfg.mom_col = str(mom)
    except Exception:
        cfg.mom_col = "mom_60"
    try:
        lk = overlay_param(STICKY_MOM60_PEAK_LOCK, "lock_level", default=0.50)
        cfg.lock_level = float(lk)
    except Exception:
        cfg.lock_level = 0.50
    if not _math.isfinite(float(cfg.lock_level)) or float(cfg.lock_level) < 0:
        cfg.lock_level = 0.50
    return StickyLeaderModel(name="sticky.house_money", config=cfg)
