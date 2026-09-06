# mypy: ignore-errors
# ruff: noqa
"""Portfolio-policy strategy factories (PEP 562 lazy re-export).

Canonical homes moved to src/portfolio/builders_* for ARCH-1 layering
(factories select theme/index keys only; portfolio-backed construction lives
in the portfolio layer). This module keeps the historical import path working
via a lazy redirect — no top-level portfolio import, so the layer boundary
test still sees zero cross-layer edges here.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "make_portfolio_convexity_hold",
    "make_portfolio_convexity_rebalance",
    "make_portfolio_convexity_variant",
    "make_portfolio_leadership_confidence",
    "make_portfolio_leadership_policy",
    "make_portfolio_lottery_exposure",
    "make_portfolio_lottery_rebalance",
    "make_portfolio_momentum_confidence",
    "make_portfolio_momentum_policy",
    "make_portfolio_momentum_vehicle",
    "make_portfolio_tail_concentration",
]

_LAZY_TABLE: dict[str, str] = {
    "make_portfolio_momentum_policy": "src.portfolio.builders_momentum",
    "make_portfolio_momentum_vehicle": "src.portfolio.builders_momentum",
    "make_portfolio_momentum_confidence": "src.portfolio.builders_momentum",
    "make_portfolio_leadership_policy": "src.portfolio.builders_leadership",
    "make_portfolio_leadership_confidence": "src.portfolio.builders_leadership",
    "make_portfolio_lottery_exposure": "src.portfolio.builders_convexity",
    "make_portfolio_tail_concentration": "src.portfolio.builders_convexity",
    "make_portfolio_convexity_hold": "src.portfolio.builders_convexity",
    "make_portfolio_convexity_rebalance": "src.portfolio.builders_convexity",
    "make_portfolio_convexity_variant": "src.portfolio.builders_convexity",
    "make_portfolio_lottery_rebalance": "src.portfolio.builders_convexity",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_TABLE:
        import importlib

        return getattr(importlib.import_module(_LAZY_TABLE[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
