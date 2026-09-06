#!/usr/bin/env python3
"""One-time migration of legacy P/B/M model keys to semantic strategy ids (D1).

Rewrites every legacy model key in a runs-registry JSONL file (or, for a
directory, in runs_registry.jsonl files and results/*/meta.json files) to its
semantic id. Writes a .bak alongside before rewriting. Supports --dry-run.
Run exactly once, in P2. This file embeds the legacy mapping table — the last
place LEGACY_ALIASES exists — and it is not importable from src/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

LEGACY_TO_SEMANTIC: Final[dict[str, str]] = {
    "B0": "baseline.buy_hold",
    "B1": "baseline.mom20_top1",
    "B2": "baseline.mom20_equal3",
    "B3": "baseline.mom20_ma_gate",
    "B4": "baseline.theme_momentum",
    "B5": "baseline.regime_gated_theme",
    "M07": "alpha.sector_leadership",
    "M13": "alpha.family_intensity",
    "P08": "portfolio.momentum_policy",
    "P10": "portfolio.momentum_vehicle",
    "P11": "portfolio.momentum_confidence",
    "P12": "portfolio.leadership_policy",
    "P13": "portfolio.leadership_confidence",
    "P14": "portfolio.lottery_exposure",
    "P15": "portfolio.tail_concentration",
    "P16": "portfolio.convexity_hold",
    "P17": "portfolio.convexity_rebalance",
    "P18": "portfolio.convexity_variant",
    "P19": "portfolio.lottery_rebalance",
    "P20": "sticky.leader_base",
    "P21": "sticky.impulse_crash",
    "P22": "sticky.family_peak_lock",
    "P23": "sticky.split_fill_lock",
    "P24": "sticky.mom60_peak_lock",
    "P25": "sticky.house_money",
    "P26": "sticky.mom60_concentrated",
    "P27": "sticky.mom60_raw",
    "P28A": "sticky.mom60_hold",
    "P28B": "sticky.mom60_abs_cash",
    "P29": "sticky.equity_mom60",
    "P29V": "sticky.equity_mom60_vol",
    "P30": "sticky.fillable_mom60",
    "P31": "convex.lottery_impulse",
    "P33": "sticky.mom60_runner_reversal",
}

_MODEL_KEYS: Final[tuple[str, ...]] = ("model", "model_key", "legacy_model_id")


def _migrate_file(path: Path, *, dry_run: bool) -> dict[str, int]:
    counts: dict[str, int] = {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    changed = False
    out_lines: list[str] = []
    for line in lines:
        if not line.strip():
            out_lines.append(line)
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            out_lines.append(line)
            continue
        if isinstance(row, dict):
            for field in _MODEL_KEYS:
                value = row.get(field)
                if isinstance(value, str) and value in LEGACY_TO_SEMANTIC:
                    counts[value] = counts.get(value, 0) + 1
                    if field != "legacy_model_id":
                        row[field] = LEGACY_TO_SEMANTIC[value]
                        changed = True
            out_lines.append(json.dumps(row))
        else:
            out_lines.append(line)
    if changed and not dry_run:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(text, encoding="utf-8")
        path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return counts


def _target_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    targets: list[Path] = sorted(path.rglob("runs_registry.jsonl"))
    targets.extend(sorted(path.glob("results/*/meta.json")))
    return targets


def migrate_runs_registry_ids(path: Path, *, dry_run: bool = False) -> dict[str, int]:
    """Rewrite legacy model keys under `path`; return per-key counts."""
    total: dict[str, int] = {}
    for target in _target_files(Path(path)):
        for legacy, count in _migrate_file(target, dry_run=dry_run).items():
            total[legacy] = total.get(legacy, 0) + count
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy run-registry ids to semantic ids.")
    parser.add_argument("path", type=Path, help="Registry file or repository root.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    counts = migrate_runs_registry_ids(args.path, dry_run=args.dry_run)
    print(json.dumps(counts, indent=2, sort_keys=True))  # noqa: T201
    return 0


if __name__ == "__main__":
    sys.exit(main())
