"""Guard that repo deploy assets never push state to offsite Drive backup.

Covers the gdrive backup removal decision (2026-09-23): mt-etf-king-2026
is a short-lived (~8 weeks) deployment whose state lives only on or-vps.
No file under ``deploy/`` or ``scripts/`` may reference rclone/Drive.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCOPES = (REPO_ROOT / "deploy", REPO_ROOT / "scripts")
FORBIDDEN_TOKENS = ("rclone", "gdrive:", "quant-lake")


def _iter_scope_files() -> list[Path]:
    files: list[Path] = []
    for scope in SCOPES:
        if not scope.is_dir():
            continue
        files.extend(p for p in sorted(scope.rglob("*")) if p.is_file())
    return files


def test_deploy_assets_never_push_to_drive() -> None:
    """Deploy/script assets must not reference offsite Drive backup tooling."""
    files = _iter_scope_files()
    assert files, "expected deploy/script assets under deploy/ and scripts/"
    texts = {path: path.read_text(encoding="utf-8", errors="ignore") for path in files}
    violations = [
        f"{path.relative_to(REPO_ROOT)} contains {token!r}"
        for path, text in texts.items()
        for token in FORBIDDEN_TOKENS
        if token in text
    ]
    assert not violations, "offsite backup references found:\n" + "\n".join(violations)
