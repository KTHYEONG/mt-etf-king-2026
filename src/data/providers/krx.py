from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import date
from typing import Final

import httpx
import tenacity

from src.data.providers.base import PermanentProviderError, RawRow, TransientProviderError

logger = logging.getLogger(__name__)

KRX_ENDPOINTS: Final[Mapping[str, str]] = {
    "etf_daily": "etp/etf_bydd_trd",
    "kospi_stock": "sto/stk_bydd_trd",
    "kosdaq_stock": "sto/ksq_bydd_trd",
    "kospi_index": "idx/kospi_dd_trd",
    "kosdaq_index": "idx/kosdaq_dd_trd",
    "krx_index": "idx/krx_dd_trd",
    "bond_index": "idx/bon_dd_trd",
    "kospi_stock_info": "sto/stk_isu_base_info",
    "kosdaq_stock_info": "sto/ksq_isu_base_info",
    "futures": "drv/fut_bydd_trd",
}


def resolve_endpoint(dataset: str) -> str:
    if dataset not in KRX_ENDPOINTS:
        raise KeyError(dataset)
    return KRX_ENDPOINTS[dataset]


class KRXOpenAPIProvider:
    def __init__(
        self,
        client: httpx.AsyncClient,
        base_url: str,
        limiter: object | None = None,
        max_attempts: int = 4,
    ) -> None:
        self._client = client
        self._base_url = base_url.rstrip("/")
        self._limiter = limiter
        self._max_attempts = max_attempts
        # Retrieve AUTH_KEY from Settings without logging
        auth_key = ""
        try:
            from src.core.settings import get_settings

            s = get_settings()
            # SecretStr
            auth_key = s.krx_openapi_key.get_secret_value()
        except Exception:
            auth_key = ""
        self._auth_key = auth_key
        # Ensure client has AUTH_KEY header (httpx headers are case-insensitive)
        if auth_key:
            # httpx.AsyncClient.headers is case-insensitive; check presence
            try:
                if "AUTH_KEY" not in self._client.headers:
                    self._client.headers["AUTH_KEY"] = auth_key
            except Exception:  # noqa: S110
                pass  # noqa: S110

    async def fetch_session(self, endpoint: str, bas_dd: date) -> list[RawRow]:
        bas_str = bas_dd.strftime("%Y%m%d")
        url = f"{self._base_url}/{endpoint.lstrip('/')}"

        # Define inner function for tenacity
        async def _do_request() -> list[RawRow]:
            # Rate limiter acquire per attempt
            if self._limiter is not None:
                # limiter has async acquire
                await self._limiter.acquire()  # type: ignore[attr-defined]

            headers = {}
            if self._auth_key:
                headers["AUTH_KEY"] = self._auth_key

            try:
                resp = await self._client.get(url, params={"basDd": bas_str}, headers=headers)
            except httpx.TimeoutException as exc:
                raise TransientProviderError(str(exc)) from exc
            except httpx.ConnectError as exc:
                raise TransientProviderError(str(exc)) from exc

            status = resp.status_code
            if status in (401, 404):
                raise PermanentProviderError(f"HTTP {status} for {endpoint} {bas_str}")
            if status == 429 or 500 <= status < 600:
                raise TransientProviderError(f"HTTP {status} for {endpoint} {bas_str}")
            if status != 200:
                # Treat other 4xx as permanent? But spec only defines 401/404 as permanent, so treat others as transient
                if 400 <= status < 500:
                    raise PermanentProviderError(f"HTTP {status} for {endpoint} {bas_str}")
                raise TransientProviderError(f"HTTP {status} for {endpoint} {bas_str}")

            try:
                payload = resp.json()
            except Exception as exc:
                raise TransientProviderError(f"invalid json: {exc}") from exc

            out = payload.get("OutBlock_1", [])
            if out is None:
                return []
            if not isinstance(out, list):
                return []
            # Return verbatim - no transformation
            return out

        # Use tenacity retry for transient errors
        retrying = tenacity.AsyncRetrying(
            stop=tenacity.stop_after_attempt(self._max_attempts),
            wait=tenacity.wait_exponential(multiplier=0.1, min=0.1, max=2),
            retry=tenacity.retry_if_exception_type(TransientProviderError),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                result = await _do_request()
                return result
        # Should not reach here; tenacity will re-raise
        raise TransientProviderError("max retries exceeded")
