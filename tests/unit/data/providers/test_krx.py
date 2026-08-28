from __future__ import annotations

import os
from datetime import date

import httpx
import pytest

from src.core.settings import clear_settings_caches
from src.data.providers.base import PermanentProviderError, TransientProviderError
from src.data.providers.krx import KRX_ENDPOINTS, KRXOpenAPIProvider, resolve_endpoint


@pytest.mark.asyncio
async def test_scenario_02_01_fetch_session_single_param_and_verbatim() -> None:
    """SCENARIO-02-01."""
    os.environ["KRX_OPENAPI_KEY"] = "TESTKEY123"
    clear_settings_caches()
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["params"] = dict(request.url.params)
        captured["headers"] = dict(request.headers)
        assert request.url.params.get("basDd") == "20260827"
        assert len(request.url.params) == 1
        hdr_lower = {k.lower(): v for k, v in request.headers.items()}
        assert "auth_key" in hdr_lower
        return httpx.Response(
            200, json={"OutBlock_1": [{"BAS_DD": "20260827", "ISU_CD": "451060", "TDD_CLSPRC": "34970", "NAV": ""}]}
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = KRXOpenAPIProvider(client=client, base_url="https://data-dbg.krx.co.kr/svc/apis")
        rows = await provider.fetch_session("etp/etf_bydd_trd", date(2026, 8, 27))
        assert rows == [{"BAS_DD": "20260827", "ISU_CD": "451060", "TDD_CLSPRC": "34970", "NAV": ""}]
        assert rows[0]["NAV"] == ""
        assert rows[0]["TDD_CLSPRC"] == "34970"
        assert captured["params"] == {"basDd": "20260827"}
    clear_settings_caches()


@pytest.mark.asyncio
async def test_scenario_02_02_permanent_error_no_retry() -> None:
    """SCENARIO-02-02."""
    os.environ["KRX_OPENAPI_KEY"] = "TESTKEY123"
    clear_settings_caches()

    count = [0]

    def handler401(request: httpx.Request) -> httpx.Response:
        count[0] += 1
        return httpx.Response(401, json={"respMsg": "Unauthorized API Call", "respCode": "401"})

    transport = httpx.MockTransport(handler401)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = KRXOpenAPIProvider(client=client, base_url="https://data-dbg.krx.co.kr/svc/apis", max_attempts=4)
        with pytest.raises(PermanentProviderError):
            await provider.fetch_session("etp/etf_bydd_trd", date(2026, 8, 27))
        assert count[0] == 1

    count2 = [0]

    def handler404(request: httpx.Request) -> httpx.Response:
        count2[0] += 1
        return httpx.Response(404, json={"respMsg": "does not exist", "respCode": "404"})

    transport2 = httpx.MockTransport(handler404)
    async with httpx.AsyncClient(transport=transport2) as client:
        provider = KRXOpenAPIProvider(client=client, base_url="https://data-dbg.krx.co.kr/svc/apis")
        with pytest.raises(PermanentProviderError):
            await provider.fetch_session("etp/etf_bydd_trd", date(2026, 8, 27))
        assert count2[0] == 1
    clear_settings_caches()


@pytest.mark.asyncio
async def test_scenario_02_03_transient_retry() -> None:
    """SCENARIO-02-03."""
    os.environ["KRX_OPENAPI_KEY"] = "TESTKEY123"
    clear_settings_caches()

    cnt = [0]

    def handler_retry(request: httpx.Request) -> httpx.Response:
        cnt[0] += 1
        if cnt[0] <= 2:
            return httpx.Response(503, json={})
        return httpx.Response(200, json={"OutBlock_1": [{"BAS_DD": "20260827"}]})

    transport = httpx.MockTransport(handler_retry)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = KRXOpenAPIProvider(client=client, base_url="https://data-dbg.krx.co.kr/svc/apis", max_attempts=4)
        rows = await provider.fetch_session("etp/etf_bydd_trd", date(2026, 8, 27))
        assert rows == [{"BAS_DD": "20260827"}]
        assert cnt[0] == 3

    cnt2 = [0]

    def handler_always(request: httpx.Request) -> httpx.Response:
        cnt2[0] += 1
        return httpx.Response(503, json={})

    transport2 = httpx.MockTransport(handler_always)
    async with httpx.AsyncClient(transport=transport2) as client:
        provider = KRXOpenAPIProvider(client=client, base_url="https://data-dbg.krx.co.kr/svc/apis", max_attempts=4)
        with pytest.raises(TransientProviderError):
            await provider.fetch_session("etp/etf_bydd_trd", date(2026, 8, 27))
        assert cnt2[0] == 4
    clear_settings_caches()


@pytest.mark.asyncio
async def test_scenario_02_04_empty_and_resolve() -> None:
    """SCENARIO-02-04."""
    os.environ["KRX_OPENAPI_KEY"] = "TESTKEY123"
    clear_settings_caches()

    def handler_empty(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"OutBlock_1": []})

    transport = httpx.MockTransport(handler_empty)
    async with httpx.AsyncClient(transport=transport) as client:
        provider = KRXOpenAPIProvider(client=client, base_url="https://data-dbg.krx.co.kr/svc/apis")
        rows = await provider.fetch_session("etp/etf_bydd_trd", date(2026, 8, 27))
        assert rows == []

    assert resolve_endpoint("etf_daily") == "etp/etf_bydd_trd"
    with pytest.raises(KeyError):
        resolve_endpoint("etn_daily")

    # forbidden endpoints not in registry
    for forbidden in [
        "etp/etn_bydd_trd",
        "etp/elw_bydd_trd",
        "idx/drvprod_dd_trd",
        "gen/oil_bydd_trd",
        "etp/etf_isu_base_info",
        "etp/etf_pdf",
    ]:
        assert forbidden not in KRX_ENDPOINTS.values()
    clear_settings_caches()
