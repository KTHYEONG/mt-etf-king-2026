from __future__ import annotations

from src.data.providers.base import PermanentProviderError, ProviderError, RawRow, TransientProviderError


def test_base_types_exist() -> None:
    """Smoke test for base symbols."""
    assert issubclass(PermanentProviderError, ProviderError)
    assert issubclass(TransientProviderError, ProviderError)
    row: RawRow = {"a": "b"}
    assert row["a"] == "b"
