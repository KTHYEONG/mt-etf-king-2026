from __future__ import annotations

from datetime import date

from src.alpha.state import TransitionConfig
from src.universe.instruments import Confidence, InstrumentAttributes, InstrumentMaster


def make_attr(
    ticker: str,
    *,
    index_key: str = "kospi 200",
    theme: str = "EQUITY",
    leverage: int = 1,
) -> InstrumentAttributes:
    d = date(2024, 1, 2)
    return InstrumentAttributes(
        ticker=ticker,
        name=ticker,
        issuer="삼성자산운용",
        leverage_multiple=leverage,
        leverage_family_key=ticker,
        is_synthetic=False,
        is_hedged=False,
        is_active=True,
        index_key=index_key,
        theme=theme,
        first_seen=d,
        last_seen=d,
        left_censored=False,
        confidence=Confidence.HIGH,
    )


def make_master(attrs: dict[str, InstrumentAttributes]) -> InstrumentMaster:
    return InstrumentMaster(attributes=attrs, panel_start=date(2024, 1, 2))


def transition_config() -> TransitionConfig:
    return TransitionConfig(
        rs_in=0.55,
        rs_out=0.35,
        rs_hi=0.75,
        accel_in=-0.08,
        accel_out=-0.20,
        breadth_in=0.65,
        breadth_out=0.45,
        ext_in=2.5,
        ext_out=1.5,
        dd_in=0.08,
        dd_out=0.05,
        patience=3,
    )
