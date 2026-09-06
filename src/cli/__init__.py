"""CLI package (P4 decomposition; re-exports the new module layout)."""

from __future__ import annotations

from src.cli.commands.backtest import cmd_backtest
from src.cli.commands.champion_research import cmd_champion_research
from src.cli.commands.config import cmd_calendar, cmd_config_check
from src.cli.commands.data import cmd_ingest, cmd_normalize
from src.cli.commands.decide import cmd_decide
from src.cli.commands.features import cmd_features
from src.cli.commands.forensics import cmd_forensics
from src.cli.commands.loyo import cmd_loyo
from src.cli.commands.replay import cmd_replay
from src.cli.commands.storage import cmd_storage_migrate
from src.cli.commands.universe import cmd_universe
from src.cli.constants import (
    ANCHOR_STRATEGY,
    CHAMPION_STRATEGY,
    CONVEXITY_ADOPTION_MODELS,
    LOTTERY_ADOPTION_MODELS,
    STICKY_ADOPTION_MODELS,
)
from src.cli.context import _make_eval_control_model
from src.cli.main import SUBCOMMANDS, main
from src.cli.parser import build_parser
from src.strategies.registry import resolve_strategy_id

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
    "cmd_champion_research",
    "cmd_config_check",
    "cmd_decide",
    "cmd_features",
    "cmd_forensics",
    "cmd_ingest",
    "cmd_loyo",
    "cmd_normalize",
    "cmd_replay",
    "cmd_storage_migrate",
    "cmd_universe",
    "main",
    "resolve_strategy_id",
]
