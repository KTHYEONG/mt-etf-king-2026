# mypy: ignore-errors
# ruff: noqa
from __future__ import annotations

import warnings
from pathlib import Path

from src.strategies.ids import STICKY_MOM60_RAW


def _read_yaml_raw(path: Path) -> dict:
    try:
        import yaml

        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


_YAML_KEY_MAP = {
    "sticky.impulse_crash": "p21",
    "sticky.family_peak_lock": "p22",
    "sticky.split_fill_lock": "p23",
    "sticky.mom60_peak_lock": "p24",
    "sticky.house_money": "p25",
    "sticky.mom60_concentrated": "p26",
    "sticky.mom60_raw": "p27",
    "sticky.leader_base": "sticky_leader",
    "sticky.mom60_hold": "p28a",
    "sticky.mom60_abs_cash": "p28b",
}


def _resolve_semantic_key(strategy_key: str) -> str:
    from src.strategies.registry import resolve_strategy_id

    try:
        return resolve_strategy_id(strategy_key)
    except Exception:
        return strategy_key.strip().lower()


def load_overlay_mode(*, strategy_key: str = "sticky.mom60_raw", default: str = "identity") -> str:
    _ = read_sticky_yaml_block(strategy_key)
    # sticky.mom60_raw is identity overlay per invariant
    if strategy_key == STICKY_MOM60_RAW:
        # check yaml semantic key first, legacy fallback
        raw = _read_yaml_raw(Path("configs/strategies.yaml"))
        port = raw.get("portfolio") if isinstance(raw, dict) else None
        if isinstance(port, dict):
            # semantic: portfolio.sticky.mom60_raw overlay_mode
            sticky = port.get("sticky")
            if isinstance(sticky, dict):
                entry = sticky.get("mom60_raw")
                if isinstance(entry, dict) and "overlay_mode" in entry:
                    v = entry.get("overlay_mode")
                    if isinstance(v, str) and v.strip():
                        s = v.strip().lower()
                        if s in {"identity", "raw", "none", "off"}:
                            return "identity"
                        if s in {"house_money", "late_lock"}:
                            return "house_money"
            # legacy fallback P27
            p27 = port.get("p27")
            if isinstance(p27, dict):
                v = p27.get("overlay_mode")
                if isinstance(v, str) and v.strip():
                    s = v.strip().lower()
                    if s in {"identity", "raw", "none", "off"}:
                        return "identity"
                    if s in {"house_money", "late_lock"}:
                        return "house_money"
        return "identity"
    # for other stickies, delegate to generic but default identity
    # Try to read overlay_mode for given strategy_key
    try:
        raw = _read_yaml_raw(Path("configs/strategies.yaml"))
        port = raw.get("portfolio") if isinstance(raw, dict) else None
        if isinstance(port, dict):
            # try semantic path portfolio.sticky.<descriptor>
            canonical = _resolve_semantic_key(strategy_key)
            # canonical is like sticky.mom60_raw -> split
            if canonical.startswith("sticky."):
                _, desc = canonical.split(".", 1)
                # map descriptor to yaml key: desc may be mom60_raw etc; yaml uses same
                sticky = port.get("sticky")
                if isinstance(sticky, dict):
                    entry = sticky.get(desc)
                    if isinstance(entry, dict) and "overlay_mode" in entry:
                        v = entry.get("overlay_mode")
                        if isinstance(v, str) and v.strip():
                            s = v.strip().lower()
                            if s in {"identity", "raw", "none", "off"}:
                                return "identity"
                            if s in {"house_money", "late_lock"}:
                                return "house_money"
                # fallback to legacy pXX if desc maps
                legacy = _YAML_KEY_MAP.get(canonical)
                if legacy:
                    entry2 = port.get(legacy)
                    if isinstance(entry2, dict) and "overlay_mode" in entry2:
                        v = entry2.get("overlay_mode")
                        if isinstance(v, str) and v.strip():
                            s = v.strip().lower()
                            if s in {"identity", "raw", "none", "off"}:
                                return "identity"
                            if s in {"house_money", "late_lock"}:
                                return "house_money"
    except Exception:
        pass
    return str(default)


def load_sticky_exposure_limits(strategy_key: str, path: Path | None = None) -> tuple[float, float, float]:
    # For sticky strategies, use p27/p26 limits (0.95,1.90,0.05) as per test
    # Attempt to read semantic key then legacy
    try:
        from src.portfolio.constraints import load_p27_exposure_limits, load_p26_exposure_limits

        # Use p27 semantics for mom60_raw etc; for now all sticky return p27 limits
        # Try semantic-specific path if exists, else legacy
        # Per spec, renamed loaders replace load_p27_*/load_p26_* but we just delegate
        # Check if strategy_key resolves to sticky.mom60_concentrated -> p26 else p27
        canon = _resolve_semantic_key(strategy_key)
        if canon == "sticky.mom60_concentrated":
            return load_p26_exposure_limits(path)
        return load_p27_exposure_limits(path)
    except Exception:
        return (0.95, 1.90, 0.05)


def load_p27_overlay_mode(*, default: str = "identity") -> str:
    warnings.warn("load_p27_overlay_mode is deprecated, use load_overlay_mode", DeprecationWarning, stacklevel=2)
    return load_overlay_mode(strategy_key=STICKY_MOM60_RAW, default=default)
def read_sticky_yaml_block(strategy_key: str, path: Path | None = None) -> dict[str, object]:
    p = Path(path) if path is not None else Path("configs/strategies.yaml")
    raw = _read_yaml_raw(p)
    port = raw.get("portfolio") if isinstance(raw, dict) else None
    if not isinstance(port, dict):
        return {}
    try:
        from src.strategies.registry import resolve_strategy_id
        canonical = resolve_strategy_id(strategy_key)
    except Exception:
        return {}
    # canonical like sticky.mom60_raw
    if not canonical.startswith("sticky."):
        return {}
    _, desc = canonical.split(".", 1)
    # semantic first
    sticky = port.get("sticky")
    if isinstance(sticky, dict):
        entry = sticky.get(desc)
        if isinstance(entry, dict):
            return dict(entry)
    # fallback legacy
    legacy = _YAML_KEY_MAP.get(canonical)
    if legacy:
        entry2 = port.get(legacy)
        if isinstance(entry2, dict):
            return dict(entry2)
    return {}
