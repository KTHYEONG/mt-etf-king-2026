from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EvalMode(StrEnum):
    ADOPTION = "adoption"
    OPERATIONAL = "operational"


@dataclass(frozen=True)
class EvalFlags:
    path_dependent: bool
    state_enabled: bool
    mode: EvalMode


def resolve_path_dependent_mode(model: object, *, mode: EvalMode | str = EvalMode.ADOPTION) -> str:
    # INV-PERF-1: StickyLeader adoption eval MUST use fast path
    # All adoption evaluations use fast (simulate_window_from_cache + live model.score)
    if isinstance(mode, str):
        low = mode.strip().lower()
        if low == "adoption":
            eval_mode = EvalMode.ADOPTION
        elif low == "operational":
            eval_mode = EvalMode.OPERATIONAL
        else:
            try:
                eval_mode = EvalMode(low)
            except Exception:  # noqa: S110
                eval_mode = EvalMode.ADOPTION
    else:
        eval_mode = mode
    _ = model
    _ = eval_mode
    return "fast"


def resolve_eval_flags(model: object, mode: EvalMode | str) -> EvalFlags:
    # normalize mode
    if isinstance(mode, str):
        low = mode.strip().lower()
        if low == "adoption":
            eval_mode = EvalMode.ADOPTION
        elif low == "operational":
            eval_mode = EvalMode.OPERATIONAL
        else:
            try:
                eval_mode = EvalMode(low)
            except Exception:  # noqa: S110
                eval_mode = EvalMode.ADOPTION
    else:
        eval_mode = mode
    if eval_mode == EvalMode.ADOPTION:
        # INV-12-1: adoption eval MUST use path_dependent=False OR state_enabled=False
        # Must inspect scores_path_independent BEFORE mutating path_dependent
        scores_pi = getattr(model, "scores_path_independent", True)
        if scores_pi is False:
            try:  # noqa: SIM105
                model.path_dependent = True  # type: ignore[attr-defined]
            except Exception:  # noqa: S110
                pass
            try:  # noqa: SIM105
                model.state_enabled = False  # type: ignore[attr-defined]
            except Exception:  # noqa: S110
                pass
            return EvalFlags(path_dependent=True, state_enabled=False, mode=eval_mode)
        try:  # noqa: SIM105
            model.path_dependent = False  # type: ignore[attr-defined]
        except Exception:  # noqa: S110
            pass
        try:  # noqa: SIM105
            model.state_enabled = False  # type: ignore[attr-defined]
        except Exception:  # noqa: S110
            pass
        return EvalFlags(path_dependent=False, state_enabled=False, mode=eval_mode)
    else:
        # INV-12-2: operational MUST use path_dependent=True with state_enabled=True
        try:  # noqa: SIM105
            model.path_dependent = True  # type: ignore[attr-defined]
        except Exception:  # noqa: S110
            pass
        try:  # noqa: SIM105
            model.state_enabled = True  # type: ignore[attr-defined]
        except Exception:  # noqa: S110
            pass
        return EvalFlags(path_dependent=True, state_enabled=True, mode=eval_mode)
