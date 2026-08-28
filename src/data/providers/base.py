from __future__ import annotations

from datetime import date
from typing import Protocol, TypeAlias

RawRow: TypeAlias = dict[str, str]


class ProviderError(RuntimeError):
    pass


class PermanentProviderError(ProviderError):
    pass


class TransientProviderError(ProviderError):
    pass


class MarketDataProvider(Protocol):
    async def fetch_session(self, endpoint: str, bas_dd: date) -> list[RawRow]:
        ...
