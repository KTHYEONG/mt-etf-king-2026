"""CLI argument parser (P4 decomposition of src/cli/_impl.py)."""

from __future__ import annotations

import argparse
from typing import Final

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

SUBCOMMANDS: Final[tuple[str, ...]] = (
    "config-check",
    "calendar",
    "ingest",
    "normalize",
    "universe",
    "features",
    "backtest",
    "forensics",
    "loyo",
    "replay",
    "decide",
    "storage-migrate",
    "champion-research",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mt-etf")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", help="log level")
    parser.add_argument("--trace", action="store_true", default=False, help="enable trace")
    sub = parser.add_subparsers(dest="subcommand")
    # config-check
    p_cfg = sub.add_parser("config-check", help="validate settings")
    p_cfg.set_defaults(func=cmd_config_check)
    # calendar
    p_cal = sub.add_parser("calendar", help="report session count")
    p_cal.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    p_cal.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    p_cal.set_defaults(func=cmd_calendar)
    # ingest
    p_ingest = sub.add_parser("ingest", help="ingest KRX data")
    p_ingest.add_argument("--dataset", required=True, help="dataset alias")
    p_ingest.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    p_ingest.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    p_ingest.add_argument("--dry-run", action="store_true", dest="dry_run", help="print plan without fetching")
    p_ingest.set_defaults(func=cmd_ingest)
    # normalize
    p_norm = sub.add_parser("normalize", help="build silver panel")
    p_norm.add_argument("--dataset", required=True, help="dataset alias")
    p_norm.add_argument("--mode", choices=["full", "incremental"], default="incremental", help="build mode")
    p_norm.set_defaults(func=cmd_normalize)
    # universe
    p_uni = sub.add_parser("universe", help="query PIT universe")
    p_uni.add_argument("--date", required=True, help="as_of date YYYY-MM-DD")
    p_uni.add_argument("--mode", choices=["structural", "deployment"], default="deployment", help="universe mode")
    p_uni.add_argument("--max-order-to-adv", type=float, default=0.05, dest="max_order_to_adv", help="max order to ADV ratio")
    p_uni.set_defaults(func=cmd_universe)
    # features
    p_feat = sub.add_parser("features", help="build feature panel")
    p_feat.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    p_feat.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    p_feat.set_defaults(func=cmd_features)
    # backtest
    p_bt = sub.add_parser("backtest", help="run backtest and rolling distribution")
    p_bt.add_argument("--model", required=True, help="semantic strategy id, e.g. sticky.mom60_raw")
    p_bt.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    p_bt.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    p_bt.add_argument("--leverage-scenario", choices=["aggressive", "conservative", "rules"], default="aggressive", dest="leverage_scenario", help="leverage scenario")
    p_bt.add_argument("--eval-mode", choices=["adoption", "operational"], default="adoption", dest="eval_mode", help="eval mode")
    p_bt.add_argument("--protocol", choices=["single", "grid"], default="single", help="cost x participation protocol")
    p_bt.add_argument("--stress-grid", action="store_true", dest="stress_grid", help="alias for --protocol grid")
    p_bt.add_argument("--commission-bps", type=float, default=None, dest="commission_bps", help="commission bps for single protocol")
    p_bt.add_argument("--slippage-bps", type=float, default=None, dest="slippage_bps", help="slippage bps for single protocol")
    p_bt.add_argument("--participation", type=float, default=None, help="participation rate for single protocol")
    p_bt.add_argument("--forensics", action="store_true", default=False, dest="forensics", help="emit tail forensics report")
    p_bt.set_defaults(func=cmd_backtest)
    # forensics
    p_for = sub.add_parser("forensics", help="tail attribution forensics report")
    p_for.add_argument("--run-id", required=True, dest="run_id", help="results run id")
    p_for.add_argument("--top-q", type=float, default=0.95, dest="top_q", help="top quantile threshold")
    p_for.add_argument("--near-miss-lo", type=float, default=0.20, dest="near_miss_lo", help="near-miss lower bound")
    p_for.add_argument("--near-miss-hi", type=float, default=0.50, dest="near_miss_hi", help="near-miss upper bound")
    p_for.set_defaults(func=cmd_forensics)
    # loyo
    p_loyo = sub.add_parser("loyo", help="LOYO era robustness evaluation")
    p_loyo.add_argument("--run-id", required=True, dest="run_id", help="results run id")
    p_loyo.add_argument("--incumbent-run-id", required=False, default=None, dest="incumbent_run_id", help="incumbent results run id")
    p_loyo.set_defaults(func=cmd_loyo)
    # replay
    p_rp = sub.add_parser("replay", help="run tournament replay")
    p_rp.add_argument("--model", required=True, help="model key")
    p_rp.add_argument("--year", required=True, help="tournament year")
    p_rp.set_defaults(func=cmd_replay)
    # decide
    p_dec = sub.add_parser("decide", help="portfolio decision dashboard")
    p_dec.add_argument("--date", required=False, default="2026-10-07", help="decision date YYYY-MM-DD")
    p_dec.add_argument("--panel", required=False, help="panel path")
    p_dec.add_argument("--model", required=False, default=None, help="model key")
    p_dec.add_argument("--capital", type=float, default=None)
    p_dec.set_defaults(func=cmd_decide)
    # storage-migrate
    p_mig = sub.add_parser("storage-migrate", help="migrate bronze plain JSON to gzip")
    p_mig.add_argument("--endpoint", required=False, default="etp/etf_bydd_trd", help="KRX endpoint to migrate")
    p_mig.add_argument("--no-delete", action="store_true", dest="no_delete", help="do not delete plain after migrate")
    p_mig.set_defaults(func=cmd_storage_migrate)
    # champion-research
    p_champ = sub.add_parser("champion-research", help="P34 family-tail research walk-forward")
    p_champ.add_argument("--start", required=True, help="start date YYYY-MM-DD")
    p_champ.add_argument("--end", required=True, help="end date YYYY-MM-DD")
    p_champ.add_argument(
        "--candidate-mode",
        dest="candidate_mode",
        choices=("p27_matched_2x", "executable_hurdle"),
        default="executable_hurdle",
        help="research-only challenger mode (P27 execution-matched +2x)",
    )
    p_champ.set_defaults(func=cmd_champion_research)
    return parser
