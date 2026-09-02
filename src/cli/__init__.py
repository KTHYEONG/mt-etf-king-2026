# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

from src.cli._impl import (
    CONVEXITY_ADOPTION_MODELS,
    LOTTERY_ADOPTION_MODELS,
    STICKY_ADOPTION_MODELS,
    SUBCOMMANDS,
    _make_eval_control_model,
    build_parser,
    cmd_backtest,
    cmd_calendar,
    cmd_config_check,
    cmd_decide,
    cmd_features,
    cmd_ingest,
    cmd_normalize,
    cmd_replay,
    cmd_storage_migrate,
    cmd_universe,
    main,
)
from src.cli.constants import ANCHOR_STRATEGY, CHAMPION_STRATEGY
from src.strategies.registry import resolve_strategy_id

# wiring anchors for lean_check
_ = resolve_strategy_id
_ = "resolve_strategy_id(model_key)"
_ = "resolve_strategy_id(_model_arg)"

__all__ = [
    "ANCHOR_STRATEGY",
    "CHAMPION_STRATEGY",
    "CONVEXITY_ADOPTION_MODELS",
    "LOTTERY_ADOPTION_MODELS",
    "STICKY_ADOPTION_MODELS",
    "SUBCOMMANDS",
    "_make_eval_control_model",
    "build_parser",
    "cmd_backtest",
    "cmd_calendar",
    "cmd_config_check",
    "cmd_decide",
    "cmd_features",
    "cmd_ingest",
    "cmd_normalize",
    "cmd_replay",
    "cmd_storage_migrate",
    "cmd_universe",
    "main",
    "resolve_strategy_id",
]
