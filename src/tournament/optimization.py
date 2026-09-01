# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import hashlib
import json
import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardFold:
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    purge: int
    embargo: int


@dataclass(frozen=True)
class P25OverlaySelection:
    selections: tuple[dict, ...]
    oos_returns: tuple[float, ...]
    raw_oos_returns: tuple[float, ...]
    trials: tuple[dict, ...]
    config_hash: str


def purged_walk_forward_indices(n_samples: int, horizon: int, n_folds: int) -> tuple[WalkForwardFold, ...]:
    if n_samples <= 0:
        raise ValueError("n_samples must be >0")
    if horizon <= 0:
        raise ValueError("horizon must be >0")
    if n_folds <= 0:
        raise ValueError("n_folds must be >0")
    purge = int(horizon)
    embargo = int(horizon)
    # allocate test sizes
    # Use block size = n_samples // (n_folds + 1)
    block = n_samples // (n_folds + 1)
    if block <= 0:
        block = n_samples // n_folds
        if block <= 0:
            block = 1
    folds: list[WalkForwardFold] = []
    for i in range(n_folds):
        train_end = (i + 1) * block
        # ensure train_end leaves room for purge+embargo+test
        test_start = train_end + purge + embargo
        test_end = test_start + block
        if test_start >= n_samples:
            # no room, adjust: place test at tail
            test_start = max(train_end + purge + embargo, n_samples - block)
            test_end = n_samples
        if test_end > n_samples:
            test_end = n_samples
        if test_start >= test_end:
            # fallback minimal test size 1
            test_start = n_samples - 1
            test_end = n_samples
            if test_start < train_end + purge + embargo:
                # if still overlap, shrink train
                train_end = max(0, test_start - purge - embargo)
        train_indices = tuple(range(0, train_end))
        test_indices = tuple(range(test_start, test_end))
        # ensure disjoint
        if set(train_indices).intersection(test_indices):
            # adjust
            test_indices = tuple(x for x in test_indices if x not in set(train_indices))
        if not train_indices or not test_indices:
            continue
        # verify purge/embargo gap
        if max(train_indices) + purge + embargo >= min(test_indices):
            # extend gap by shrinking train
            allowed_train_max = min(test_indices) - purge - embargo - 1
            if allowed_train_max < 0:
                continue
            train_indices = tuple(x for x in train_indices if x <= allowed_train_max)
            if not train_indices:
                continue
        folds.append(WalkForwardFold(train_indices=train_indices, test_indices=test_indices, purge=purge, embargo=embargo))
    # ensure we have n_folds; if fewer, pad with last?
    if len(folds) != n_folds:
        # if not enough, generate alternative simple expanding with fixed gap
        # fallback: sequential folds with equal test sizes
        folds = []
        test_size = max(1, (n_samples - n_folds * (purge + embargo)) // (n_folds + 1))
        for i in range(n_folds):
            train_end = (i + 1) * test_size + i * (purge + embargo)
            test_start = train_end + purge + embargo
            test_end = test_start + test_size
            if test_start >= n_samples:
                break
            if test_end > n_samples:
                test_end = n_samples
            folds.append(WalkForwardFold(train_indices=tuple(range(0, train_end)), test_indices=tuple(range(test_start, test_end)), purge=purge, embargo=embargo))
    # final guarantee: trim to n_folds
    folds = folds[:n_folds]
    if len(folds) != n_folds:
        raise ValueError(f"cannot generate {n_folds} folds with n_samples={n_samples} horizon={horizon}")
    # validate no overlap and purge/embargo
    for f in folds:
        if set(f.train_indices).intersection(f.test_indices):
            raise ValueError("overlapping indices")
        if max(f.train_indices) + f.purge + f.embargo >= min(f.test_indices):
            raise ValueError("purge/embargo violation")
    return tuple(folds)


def optimize_p25_overlay(
    daily_rets: Sequence[float],
    sessions: Sequence[date],
    horizon: int,
    config,
    arms: Sequence[float],
    lock_remaining_values: Sequence[int],
    *,
    n_folds: int = 3,
) -> P25OverlaySelection:
    import math as _math

    if not daily_rets or not sessions:
        raise ValueError("daily_rets/sessions empty")
    if len(daily_rets) != len(sessions):
        # sessions length should match daily_rets; fail closed if mismatch?
        # allow but require same n
        if len(daily_rets) != len(sessions):
            raise ValueError("daily_rets and sessions length mismatch")
    if horizon <= 0:
        raise ValueError("horizon must be >0")
    if not arms or not lock_remaining_values:
        raise ValueError("arms/lock_remaining_values empty")
    # validate arms
    for a in arms:
        fv = float(a)  # type: ignore[arg-type]
        if not _math.isfinite(fv) or fv <= 0:
            raise ValueError(f"arm invalid {a}")
    for lr in lock_remaining_values:
        fv = float(lr)  # type: ignore[arg-type]
        if not _math.isfinite(fv) or fv < 0:
            raise ValueError(f"lock_remaining invalid {lr}")
    n = len(daily_rets)
    folds = purged_walk_forward_indices(n, horizon, n_folds)
    from src.tournament.distribution import execution_faithful_late_lock_returns

    # for config hash
    try:
        cfg_dict = {
            "thresholds": list(config.thresholds),
            "scenario_weights": {k: list(v) for k, v in config.scenario_weights.items()},
            "primary_scenario": config.primary_scenario,
            "bootstrap_expected_block": int(config.bootstrap_expected_block),
            "bootstrap_resamples": int(config.bootstrap_resamples),
            "seed": int(config.seed),
            "horizon": int(horizon),
            "arms": list(float(x) for x in arms),  # type: ignore[arg-type]
            "lock_remaining_values": list(int(x) for x in lock_remaining_values),  # type: ignore[arg-type]
        }
        config_hash = hashlib.sha256(json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()[:16]
    except Exception:
        config_hash = "unknown"

    selections: list[dict] = []
    oos_returns: list[float] = []
    raw_oos_returns: list[float] = []
    trials: list[dict] = []
    trial_n = 0

    for fold_idx, fold in enumerate(folds):
        train_daily = [float(daily_rets[i]) for i in fold.train_indices]  # type: ignore[index]
        # also need raw oos returns for test fold: use raw daily without overlay?
        # raw returns are window returns without lock (simple window compound)
        # but for selection we use execution faithful returns
        best_score = -float("inf")
        best_arm = None
        best_lr = None
        # deterministic iteration order: arms sorted ascending, lock values ascending?
        # But tie-breaking prefers larger arm and smaller lock -> we need to handle tie after scoring
        candidates: list[tuple[float, float, int, float]] = []  # score, arm, lr, maximin?
        for arm in arms:
            for lr in lock_remaining_values:
                trial_n += 1
                # compute train window returns with this overlay
                try:
                    train_windows = execution_faithful_late_lock_returns(train_daily, horizon, float(arm), int(lr))
                except Exception:
                    train_windows = []
                if not train_windows:
                    score = -float("inf")
                else:
                    # compute scenario maximin: min across scenarios of weighted exceedance
                    # Use config thresholds and scenario_weights
                    # compute exceedance per threshold
                    scores_by_scenario: list[float] = []
                    for scen, weights in config.scenario_weights.items():
                        sc = 0.0
                        for thr, w in zip(config.thresholds, weights):
                            p = sum(1 for r in train_windows if float(r) > float(thr)) / len(train_windows) if train_windows else 0.0
                            sc += float(w) * p
                        scores_by_scenario.append(sc)
                    # maximin
                    score = min(scores_by_scenario) if scores_by_scenario else -float("inf")
                # log trial
                logger.debug(f"[EVAL] trial={trial_n} arm={float(arm):.3f} lock_remaining={int(lr)} fold={fold_idx} score={float(score):.3f}")
                trials.append({"trial": int(trial_n), "arm": float(arm), "lock_remaining": int(lr), "fold": int(fold_idx), "score": float(score)})
                candidates.append((float(score), float(arm), int(lr), float(score)))
                # selection logic: keep best by maximin, tie break larger arm then smaller lock
                if best_score is None or score > best_score + 1e-12:
                    best_score = float(score)
                    best_arm = float(arm)
                    best_lr = int(lr)
                elif abs(score - best_score) <= 1e-12:
                    # tie: larger arm preferred, if equal arm then smaller lock
                    if best_arm is not None and best_lr is not None:
                        if float(arm) > best_arm + 1e-12:
                            best_arm = float(arm)
                            best_lr = int(lr)
                        elif abs(float(arm) - best_arm) <= 1e-12 and int(lr) < best_lr:
                            best_arm = float(arm)
                            best_lr = int(lr)
        # if best not found, fallback to first
        if best_arm is None or best_lr is None:
            best_arm = float(arms[0])
            best_lr = int(lock_remaining_values[0])
            best_score = -float("inf")
        selections.append({"fold": int(fold_idx), "arm": float(best_arm), "lock_remaining": int(best_lr), "train_score": float(best_score), "train_indices": list(fold.train_indices), "test_indices": list(fold.test_indices)})
        # apply to test
        test_daily = [float(daily_rets[i]) for i in fold.test_indices]
        # Note: test indices are not contiguous daily_rets slice for window returns? But we use them as daily_rets for test window returns
        # Need to reconstruct test windows: daily_rets segment for test period plus horizon-1 overlap? For simplicity we use execution_faithful on test_daily with horizon
        # However to align with whole series, we should compute window returns over full daily_rets then slice to test window positions
        # Simpler: compute full series window returns and average over test positions that are start indices within test_indices
        try:
            full_windows = execution_faithful_late_lock_returns(list(float(x) for x in daily_rets), horizon, float(best_arm), int(best_lr))
        except Exception:
            full_windows = []
        # raw oos (no lock): simple window returns via compounding daily_rets without overlay
        # compute raw window returns: compound of daily_rets horizon
        raw_windows: list[float] = []
        daily_list = list(float(x) for x in daily_rets)
        if len(daily_list) >= horizon:
            for i in range(len(daily_list) - horizon + 1):
                eq = 1.0
                for k in range(horizon):
                    eq *= 1.0 + float(daily_list[i + k])
                raw_windows.append(float(eq - 1.0))
        # map test window returns: test start indices are day indices, window start indices are 0..n-horizon
        # We consider OOS returns as mean of window returns whose start index is inside test_indices (and window fits)
        test_oos: list[float] = []
        raw_test_oos: list[float] = []
        max_start = n - horizon
        for idx in fold.test_indices:
            if 0 <= idx <= max_start:
                if idx < len(full_windows):
                    test_oos.append(float(full_windows[idx]))
                if idx < len(raw_windows):
                    raw_test_oos.append(float(raw_windows[idx]))
        # aggregate to single OOS return? Use mean
        oos_val = float(sum(test_oos) / len(test_oos)) if test_oos else 0.0
        raw_oos_val = float(sum(raw_test_oos) / len(raw_test_oos)) if raw_test_oos else 0.0
        oos_returns.append(float(oos_val))
        raw_oos_returns.append(float(raw_oos_val))

    return P25OverlaySelection(
        selections=tuple(selections),
        oos_returns=tuple(float(x) for x in oos_returns),
        raw_oos_returns=tuple(float(x) for x in raw_oos_returns),
        trials=tuple(trials),
        config_hash=str(config_hash),
    )
