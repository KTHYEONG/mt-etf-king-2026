"""Fail-closed deployment manifest validation (Phase 0)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass


class ManifestValidationError(ValueError):
    """Raised when a non-null manifest fails membership or issuer checks."""


@dataclass(frozen=True, slots=True)
class ManifestValidationResult:
    status: str


def validate_deployment_manifest(
    *,
    manifest: frozenset[str] | None,
    present_tickers: frozenset[str],
    issuer_by_ticker: Mapping[str, str],
    allowed_issuers: frozenset[str],
) -> ManifestValidationResult:
    if manifest is None:
        return ManifestValidationResult(status="PROVISIONAL")
    if len(manifest) == 0:
        raise ManifestValidationError("empty deployment manifest")
    for code in sorted(manifest):
        if code not in present_tickers:
            raise ManifestValidationError(f"manifest ticker missing from panel: {code}")
        issuer = issuer_by_ticker.get(code, "UNKNOWN")
        if issuer == "UNKNOWN" or issuer not in allowed_issuers:
            raise ManifestValidationError(f"manifest ticker issuer not allowed: {code} issuer={issuer}")
    return ManifestValidationResult(status="ENFORCED")
