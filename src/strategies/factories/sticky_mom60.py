# mypy: ignore-errors
# ruff: noqa
"""Mom60 variant (concentrated / raw / hold / abs-cash / equity / fillable / runner-reversal) sticky-leader strategy factories (P5 split of factories/sticky.py)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.sticky.model import StickyLeaderModel


def make_sticky_mom60_concentrated() -> StickyLeaderModel:
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
    # enforce P26 invariants after yaml load so missing p26 block cannot inherit P21 crash-cash=-0.12
    cfg.mom_col = "mom_60"
    cfg.cash_drawdown = 0.0
    cfg.min_gap = 0.04
    cfg.min_hold = 2
    cfg.impulse_gap = 0.0
    cfg.impulse_require_volx = True
    cfg.only_plus_2 = True
    cfg.no_inverse = True
    cfg.collapse_family = False
    # overlay portfolio.p26 if present (fail-closed, but invariants remain forced)
    try:
        p26_raw = None
        if isinstance(raw, dict):
            port = raw.get("portfolio")
            if isinstance(port, dict):
                p26_raw = port.get("p26")
        if isinstance(p26_raw, Mapping):
            if "mom_col" in p26_raw:
                try:
                    v = p26_raw["mom_col"]
                    if isinstance(v, str) and v.strip() and v.strip().startswith("mom_"):
                        cfg.mom_col = str(v).strip()
                    else:
                        cfg.mom_col = "mom_60"
                except Exception:
                    cfg.mom_col = "mom_60"
            # cash_drawdown must stay 0.0 regardless
            cfg.cash_drawdown = 0.0
            # min_gap
            if "min_gap" in p26_raw:
                try:
                    mg = float(p26_raw["min_gap"])  # type: ignore[arg-type]
                    if not _math.isfinite(mg) or mg < 0:
                        cfg.min_gap = 0.04
                    else:
                        cfg.min_gap = 0.04
                except Exception:
                    cfg.min_gap = 0.04
            else:
                cfg.min_gap = 0.04
            if "min_hold" in p26_raw:
                try:
                    mh = int(p26_raw["min_hold"])  # type: ignore[arg-type]
                    fv = float(p26_raw["min_hold"])  # type: ignore[arg-type]
                    if not _math.isfinite(fv) or mh < 0:
                        cfg.min_hold = 2
                    else:
                        cfg.min_hold = 2
                except Exception:
                    cfg.min_hold = 2
            else:
                cfg.min_hold = 2
            if "impulse_gap" in p26_raw:
                try:
                    ig = float(p26_raw["impulse_gap"])  # type: ignore[arg-type]
                    if not _math.isfinite(ig) or ig < 0:
                        cfg.impulse_gap = 0.0
                    else:
                        cfg.impulse_gap = 0.0
                except Exception:
                    cfg.impulse_gap = 0.0
            else:
                cfg.impulse_gap = 0.0
    except Exception:
        cfg.mom_col = "mom_60"
        cfg.cash_drawdown = 0.0
        cfg.min_gap = 0.04
        cfg.min_hold = 2
        cfg.impulse_gap = 0.0
    # final fail-closed ensure
    cfg.mom_col = "mom_60"
    cfg.cash_drawdown = 0.0
    cfg.min_gap = 0.04
    cfg.min_hold = 2
    cfg.impulse_gap = 0.0
    cfg.impulse_require_volx = True
    cfg.only_plus_2 = True
    cfg.no_inverse = True
    cfg.collapse_family = False
    return StickyLeaderModel(name="sticky.mom60_concentrated", config=cfg)


def make_sticky_mom60_raw() -> StickyLeaderModel:
    import math as _math
    from pathlib import Path as _P

    import yaml as _yaml

    from src.alpha.sticky import StickyLeaderConfig, StickyLeaderModel

    cfg = StickyLeaderConfig(
        mom_col="mom_60",
        min_gap=0.04,
        min_hold=2,
        abs_mom_cash=True,
        exclude_synthetic=True,
        min_fill_ratio=0.25,
    )
    try:
        fp = _P("configs/strategies.yaml")
        raw: dict = {}
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                raw = _yaml.safe_load(f) or {}
        champ_raw = None
        if isinstance(raw, dict):
            port = raw.get("portfolio")
            if isinstance(port, dict):
                sticky = port.get("sticky")
                if isinstance(sticky, Mapping) and isinstance(sticky.get("mom60_raw"), Mapping):
                    champ_raw = sticky.get("mom60_raw")
                elif isinstance(port.get("mom60_raw"), Mapping):
                    champ_raw = port.get("mom60_raw")
                elif isinstance(port.get("p27"), Mapping):
                    champ_raw = port.get("p27")
        if isinstance(champ_raw, Mapping):
            if "mom_col" in champ_raw:
                try:
                    v = champ_raw["mom_col"]
                    if isinstance(v, str) and v.strip().startswith("mom_"):
                        cfg.mom_col = str(v).strip()
                    else:
                        cfg.mom_col = "mom_60"
                except Exception:
                    cfg.mom_col = "mom_60"
            if "min_gap" in champ_raw:
                try:
                    mg = float(champ_raw["min_gap"])  # type: ignore[arg-type]
                    if _math.isfinite(mg) and mg >= 0:
                        cfg.min_gap = float(mg)
                except Exception:
                    pass
            if "min_hold" in champ_raw:
                try:
                    mh = int(champ_raw["min_hold"])  # type: ignore[arg-type]
                    fv = float(champ_raw["min_hold"])  # type: ignore[arg-type]
                    if _math.isfinite(fv) and mh >= 0:
                        cfg.min_hold = int(mh)
                except Exception:
                    pass
            if "min_fill_ratio" in champ_raw:
                try:
                    mfr = float(champ_raw["min_fill_ratio"])  # type: ignore[arg-type]
                    if _math.isfinite(mfr) and mfr >= 0:
                        cfg.min_fill_ratio = float(mfr)
                except Exception:
                    pass
            if "exclude_synthetic" in champ_raw:
                try:
                    cfg.exclude_synthetic = bool(champ_raw["exclude_synthetic"])
                except Exception:
                    pass
            if "abs_mom_cash" in champ_raw:
                try:
                    cfg.abs_mom_cash = bool(champ_raw["abs_mom_cash"])
                except Exception:
                    pass
    except Exception:
        pass
    if not isinstance(cfg.mom_col, str) or not cfg.mom_col.startswith("mom_"):
        cfg.mom_col = "mom_60"
    cfg.cash_drawdown = 0.0
    cfg.impulse_gap = 0.0
    cfg.only_plus_2 = True
    cfg.no_inverse = True
    cfg.collapse_family = False
    cfg.same_leader_hold = False
    cfg.exclude_name_tokens = ()
    return StickyLeaderModel(name="sticky.mom60_raw", config=cfg)


def make_sticky_mom60_hold() -> StickyLeaderModel:

    model = make_sticky_mom60_raw()
    model.name = "sticky.mom60_hold"
    model.config.same_leader_hold = True
    model.config.abs_mom_cash = False
    return model


def make_sticky_mom60_abs_cash() -> StickyLeaderModel:

    model = make_sticky_mom60_raw()
    model.name = "sticky.mom60_abs_cash"
    model.config.same_leader_hold = True
    model.config.abs_mom_cash = True
    return model


def make_sticky_equity_mom60() -> StickyLeaderModel:
    from src.strategies.sticky.model import DEFAULT_EXCLUDE_NAME_TOKENS

    model = make_sticky_mom60_raw()
    model.name = "sticky.equity_mom60"
    model.config.exclude_name_tokens = tuple(DEFAULT_EXCLUDE_NAME_TOKENS)
    model.config.mom_col = "mom_60"
    model.config.min_gap = 0.04
    model.config.score_aux_col = None
    model.config.score_aux_weight = 0.0
    return model


def make_sticky_equity_mom60_vol() -> StickyLeaderModel:

    model = make_sticky_equity_mom60()
    model.name = "sticky.equity_mom60_vol"
    model.config.score_aux_col = "volume_expansion"
    model.config.score_aux_weight = 0.3
    model.config.min_gap = 0.10
    return model


def make_sticky_fillable_mom60() -> StickyLeaderModel:
    from src.strategies.sticky.model import DEFAULT_EXCLUDE_NAME_TOKENS

    model = make_sticky_equity_mom60()
    model.name = "sticky.fillable_mom60"
    model.config.exclude_name_tokens = tuple(DEFAULT_EXCLUDE_NAME_TOKENS)
    model.config.exclude_synthetic = True
    model.config.min_fill_ratio = 0.25
    model.config.mom_col = "mom_60"
    model.config.min_gap = 0.04
    model.config.min_hold = 2
    model.config.impulse_gap = 0.0
    model.config.score_aux_col = None
    model.config.score_aux_weight = 0.0
    return model


def make_sticky_mom60_inactive_participate() -> StickyLeaderModel:
    import yaml

    from src.alpha.sticky import StickyLeaderModel
    from src.core.config import config_path as _resolve_config_path

    base = make_sticky_mom60_raw()
    raw_config: Mapping[str, object] = {}
    try:
        strategies_config_path = _resolve_config_path("strategies")
        if strategies_config_path.exists():
            with strategies_config_path.open(encoding="utf-8") as handle:
                document = yaml.safe_load(handle) or {}
            portfolio = document.get("portfolio") if isinstance(document, dict) else None
            sticky = portfolio.get("sticky") if isinstance(portfolio, dict) else None
            candidate = sticky.get("mom60_inactive_participate") if isinstance(sticky, Mapping) else None
            if isinstance(candidate, Mapping):
                raw_config = candidate
    except Exception:
        raw_config = {}
    config = base.config.from_yaml(raw_config) if raw_config else base.config
    model = StickyLeaderModel(name="sticky.mom60_inactive_participate", config=config)
    model.name = "sticky.mom60_inactive_participate"
    model.config.mom_col = "mom_60"
    model.config.min_gap = 0.04
    model.config.min_hold = 2
    model.config.only_plus_2 = True
    model.config.no_inverse = True
    model.config.abs_mom_cash = True
    model.config.exclude_synthetic = True
    model.config.min_fill_ratio = 0.25
    model.config.collapse_family = False
    model.config.inactive_participation = True
    return model


def make_sticky_mom60_runner_reversal() -> StickyLeaderModel:
    import yaml

    from src.alpha.sticky import StickyLeaderModel
    from src.core.config import config_path as _resolve_config_path

    base = make_sticky_mom60_raw()
    raw_config: Mapping[str, object] = {}
    try:
        strategies_config_path = _resolve_config_path("strategies")
        if strategies_config_path.exists():
            with strategies_config_path.open(encoding="utf-8") as handle:
                document = yaml.safe_load(handle) or {}
            portfolio = document.get("portfolio") if isinstance(document, dict) else None
            sticky = portfolio.get("sticky") if isinstance(portfolio, dict) else None
            candidate = sticky.get("mom60_runner_reversal") if isinstance(sticky, Mapping) else None
            if isinstance(candidate, Mapping):
                raw_config = candidate
    except Exception:
        raw_config = {}
    config = base.config.from_yaml(raw_config) if raw_config else base.config
    model = StickyLeaderModel(name="sticky.mom60_runner_reversal", config=config)
    model.name = "sticky.mom60_runner_reversal"
    model.config.mom_col = "mom_60"
    model.config.min_gap = 0.04
    model.config.min_hold = 2
    model.config.only_plus_2 = True
    model.config.no_inverse = True
    model.config.abs_mom_cash = True
    model.config.exclude_synthetic = True
    model.config.min_fill_ratio = 0.25
    model.config.collapse_family = False
    model.config.same_leader_hold = False
    model.config.runner_reversal_exit = True
    model.config.runner_mom_col = "mom_5"
    return model
