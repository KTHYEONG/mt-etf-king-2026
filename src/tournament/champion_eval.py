# mypy: ignore-errors
# ruff: noqa: S101
"""P34 walk-forward promotion evaluation (identical fold-local OOS comparison)."""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.alpha.champion_dataset import (
    ChampionDatasetConfig as _ChampionDatasetConfigRef,
)
from src.alpha.champion_dataset import (
    build_family_tail_dataset as _build_family_tail_dataset_ref,
)
from src.alpha.champion_dataset import (
    collect_family_candidates as _collect_family_candidates_ref,
)
from src.alpha.champion_ranker import ChampionTailRanker as _ChampionTailRankerRef
from src.alpha.champion_ranker import OosScoreStore as _OosScoreStoreRef
from src.alpha.champion_ranker import PurgedDateWalkForward as _PurgedDateWalkForwardRef
from src.strategies.champion_tail import ChampionTailPolicy as _ChampionTailPolicyRef

_ = _ChampionDatasetConfigRef
_ = _collect_family_candidates_ref
_ = _build_family_tail_dataset_ref
_ = _PurgedDateWalkForwardRef
_ = _ChampionTailRankerRef
_ = _OosScoreStoreRef
_ = _ChampionTailPolicyRef


@dataclass
class ChampionEvaluation:
    status: str = "RESEARCH_ONLY"
    aggressive_status: str = "FAIL"
    conservative_status: str = "FAIL"
    loyo_status: str = "FAIL"
    artifact_integrity: bool = False
    elapsed_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def write(self, path: str | Path) -> Path:
        import json

        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": self.status,
            "aggressive_status": self.aggressive_status,
            "conservative_status": self.conservative_status,
            "loyo_status": self.loyo_status,
            "artifact_integrity": self.artifact_integrity,
            "elapsed_seconds": self.elapsed_seconds,
            "peak_memory_mb": self.peak_memory_mb,
            **self.extra,
        }
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(dest)
        return dest


def is_promotable(
    *,
    aggressive_status: str,
    conservative_status: str,
    loyo_status: str,
    artifact_integrity: bool,
) -> bool:
    """Promotion requires dual-scenario PASS, LOYO PASS, and artifact integrity."""
    return bool(
        aggressive_status == "PASS"
        and conservative_status == "PASS"
        and loyo_status == "PASS"
        and artifact_integrity is True
    )


def run_champion_walk_forward(
    *,
    start: Any = None,
    end: Any = None,
    engine: Any = None,
    simulator: Any = None,
    panel: Any = None,
    config: Any = None,
    model_config: Any = None,
    dataset_config: Any = None,
    p27_factory: Callable[[], Any] | None = None,
) -> ChampionEvaluation:
    """Execute P34-raw/P34/P27/conservative on identical fold-local OOS sessions."""
    t0 = time.time()
    required = {
        "engine": engine,
        "simulator": simulator,
        "panel": panel,
        "config": config,
        "model_config": model_config,
        "dataset_config": dataset_config,
        "p27_factory": p27_factory,
    }
    missing = tuple(name for name, value in required.items() if value is None)
    integrity = not missing
    # CLI smoke/research path: use the production incumbent engine when the
    # heavyweight injected research components are not assembled yet.  This
    # keeps execution real (fills/costs/artifacts) and never promotes P27.
    if missing and start is not None and end is not None:
        from argparse import Namespace

        from src.cli._impl import cmd_backtest

        rc = cmd_backtest(
            Namespace(
                model="P27",
                start=start,
                end=end,
                leverage_scenario="aggressive",
                eval_mode="adoption",
                protocol="single",
                stress_grid=False,
                commission_bps=None,
                slippage_bps=None,
                participation=None,
                forensics=False,
                log_level="INFO",
                trace=False,
            )
        )
        return ChampionEvaluation(
            status="RESEARCH_ONLY",
            aggressive_status="PASS" if rc == 0 else "FAIL",
            conservative_status="FAIL",
            loyo_status="FAIL",
            artifact_integrity=False,
            elapsed_seconds=time.time() - t0,
            peak_memory_mb=0.0,
            extra={"incumbent_model": "P27", "backtest_exit_code": int(rc)},
        )
    # Keep orchestration in the existing engine: this function owns the
    # promotion contract, while the engine owns fills, costs and diagnostics.
    runner = getattr(engine, "run_champion_walk_forward", None) if engine is not None else None
    if callable(runner) and not missing:
        result = runner(
            simulator=simulator,
            panel=panel,
            config=config,
            model_config=model_config,
            dataset_config=dataset_config,
            p27_factory=p27_factory,
        )
        if not isinstance(result, ChampionEvaluation):
            raise TypeError("champion engine returned an invalid evaluation")
        result.elapsed_seconds = time.time() - t0
        return result
    evaluation = ChampionEvaluation(
        status="RESEARCH_ONLY",
        aggressive_status="FAIL",
        conservative_status="FAIL",
        loyo_status="FAIL",
        artifact_integrity=integrity,
        elapsed_seconds=time.time() - t0,
        peak_memory_mb=0.0,
        extra={"missing_runtime_inputs": missing} if missing else {},
    )
    return evaluation


__all__ = ["ChampionEvaluation", "is_promotable", "run_champion_walk_forward"]
