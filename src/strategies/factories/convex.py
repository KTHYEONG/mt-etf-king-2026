# mypy: ignore-errors
# ruff: noqa
"""Convex lottery-impulse strategy factories (P3 split of src/alpha/baselines.py)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.strategies.convex_impulse import ConvexImpulseModel


def make_convex_lottery_impulse() -> ConvexImpulseModel:
    from pathlib import Path as _P

    import yaml as _yaml

    from src.strategies.convex_impulse import ConvexImpulseConfig, ConvexImpulseModel

    cfg = ConvexImpulseConfig()
    try:
        fp = _P("configs/strategies.yaml")
        if fp.exists():
            with open(fp, encoding="utf-8") as f:
                raw = _yaml.safe_load(f) or {}
            port = raw.get("portfolio") if isinstance(raw, dict) else None
            block = None
            if isinstance(port, dict):
                convex = port.get("convex")
                if isinstance(convex, dict) and isinstance(convex.get("lottery_impulse"), dict):
                    block = convex.get("lottery_impulse")
                elif isinstance(port.get("p31"), dict):
                    block = port.get("p31")
            if isinstance(block, Mapping):
                cfg = ConvexImpulseConfig.from_yaml(block)  # type: ignore[arg-type]
    except Exception:
        cfg = ConvexImpulseConfig()
    return ConvexImpulseModel(name="convex.lottery_impulse", config=cfg)
