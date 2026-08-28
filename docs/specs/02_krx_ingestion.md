# 02. KRX Ingestion — provider · rate limit · bronze · resumable backfill

**선행**: [01_core_spine](01_core_spine.md)
**상위**: [00_architecture.md](00_architecture.md)

---

## 1. Diagnosis & Invariants

### 1.1 문제의 본질

KRX Open API 는 **에러를 상태 코드로 알려주지 않는다.** 실측 확인:

| 입력 | HTTP | 본문 |
| --- | --- | --- |
| 정상 거래일 `20260827` | 200 | `OutBlock_1` 1,163행 (가격 채워짐) |
| **휴장일 `20260815`** | 200 | `OutBlock_1` **1,163행** (가격 전부 `""`) |
| 미래일 `20261231` | 200 | `OutBlock_1: []` |
| 2010 이전 `20090101` | 200 | `OutBlock_1: []` |
| 문법 오류 `"bad"` | 200 | `OutBlock_1: []` |
| 미구독 endpoint | 401 | `{"respMsg": "Unauthorized API Call", "respCode": "401"}` |
| 미존재 endpoint | 404 | `{"respMsg": "... does not exist.", "respCode": "404"}` |

따라서 **수집 계층은 "성공/실패"를 판정할 수 없다.** 판정 가능한 것은 오직 "응답을 받았다"이며, 의미론적 유효성(거래일이었는가, 결측이 정상인가)은 spec 03 의 validation 계층 책임이다.

**INV-02-1 (책임 분리)**: 수집 계층은 payload 를 **해석하지 않는다**. `""` 를 그대로 보존하고, 행이 0개여도 예외를 던지지 않는다. 판단은 silver/validation 이 한다.

### 1.2 재시도해야 할 실패 vs 절대 재시도하면 안 되는 실패

| 분류 | 조건 | 처리 |
| --- | --- | --- |
| **영구(permanent)** | 401 (미구독), 404 (미존재 경로) | 즉시 예외. **재시도 금지** — 쿼터만 소모하고 결코 성공하지 않는다 |
| **일시(transient)** | 5xx, `httpx.TimeoutException`, `httpx.ConnectError`, 429 | 지수 백오프 재시도 (tenacity), 최대 4회 |
| **의미론적(semantic)** | 200 + 빈 배열, 200 + 공백 가격 | **예외 아님**. 그대로 저장하고 통과 |

401 을 재시도하는 것은 이 시스템에서 가장 비용이 큰 버그다 — 일일 쿼터가 유한하고(R-9) 401은 구독 상태가 바뀌기 전엔 영원히 401이다.

### 1.3 파라미터 위생

**TRAP-3 실측**: `{"basDd": "20260827", "isuCd": "069500"}` 를 보내면 서버가 `isuCd` 를 **조용히 무시**하고 1,163행 전체를 반환한다. `{"strtDd","endDd"}` 는 0행.

**INV-02-2 (파라미터 화이트리스트)**: 요청 쿼리는 `basDd` **단 하나**만 포함한다. 다른 키를 넣는 것은 "필터가 걸렸다"는 거짓 확신을 만들고 하위 계층에서 조용한 오류로 번진다.

### 1.4 수집 예산 산술

$$
\text{calls} = |\text{endpoints}| \times \texttt{session\_count}(\text{start}, \text{end})
$$

| 범위 | ETF만 | 6 endpoint 전량 |
| --- | --- | --- |
| 2018-01-01 ~ 2026-08-27 (2,125 세션) | 2,125 | 12,750 |
| 2010-01-04 ~ 2026-08-27 (4,101 세션) | 4,101 | 24,606 |

측정 처리량 **0.62 s/call**(순차). 일일 쿼터는 **미확인**(비공식 10,000/day). 따라서:

**INV-02-3 (재개 가능성)**: backfill 은 언제 중단되어도 **이미 받은 세션을 다시 받지 않고** 재개 가능해야 한다. 이는 bronze 파일 존재 여부로 판정한다 — 별도 상태 파일에 의존하면 파일과 상태가 어긋난다.

**INV-02-4 (쿼터 원장)**: 하루 누적 호출 수를 `data/state/krx_quota.json` 에 KST 날짜 기준으로 기록하고, 소진 시 예외가 아니라 **정상 종료**(부분 완료 리포트)로 끝낸다.

### 1.5 불변 원본 (INV-3)

`data/raw/**` 는 write-once 다. 이미 존재하는 세션 파일은 **덮어쓰지 않는다.** 이유는 감사가능성이 아니라 **정확성**이다 — KRX 가 과거 데이터를 사후 정정하면, 덮어쓰기는 이미 그 데이터로 산출한 백테스트 결과를 조용히 무효화한다. 재수집이 필요하면 리비전 파일로 나란히 남긴다.

---

## 2. Architecture & Mitigation

```
              ┌───────────────────────────────┐
 cli.py       │ cmd_ingest                    │
              └───────────────┬───────────────┘
                              ▼
              ┌───────────────────────────────┐
 backfill.py  │ BackfillPlanner → BackfillPlan│  세션 diff · 쿼터 절단
              │ run_backfill (async, bounded) │
              └───────────────┬───────────────┘
                    ┌─────────┴─────────┐
                    ▼                   ▼
        ┌────────────────────┐  ┌──────────────────┐
 krx.py │ KRXOpenAPIProvider │  │ BronzeStore      │ bronze.py
        │  · AUTH_KEY header │  │  · write-once    │
        │  · basDd only      │  │  · envelope+meta │
        │  · 401/404 = fatal │  └──────────────────┘
        └─────────┬──────────┘
                  ▼
        ┌────────────────────┐
        │ RateLimiter        │ ratelimit.py
        │ QuotaLedger        │  (주입 가능한 clock)
        └────────────────────┘
```

### 2.1 Endpoint 레지스트리

구독 상태가 실측으로 확인된 것만 등록한다. 미구독(401)/미존재(404)는 **레지스트리에 넣지 않는다** — 존재하지 않는 데이터를 코드가 약속하지 않게 한다.

```python
KRX_ENDPOINTS = {
    "etf_daily":       "etp/etf_bydd_trd",     # 1,163 rows — 핵심
    "kospi_stock":     "sto/stk_bydd_trd",     #   944 rows
    "kosdaq_stock":    "sto/ksq_bydd_trd",     # 1,823 rows
    "kospi_index":     "idx/kospi_dd_trd",     #    51 rows
    "kosdaq_index":    "idx/kosdaq_dd_trd",    #    40 rows
    "krx_index":       "idx/krx_dd_trd",       #    40 rows
    "bond_index":      "idx/bon_dd_trd",       #     3 rows
    "kospi_stock_info":"sto/stk_isu_base_info",
    "kosdaq_stock_info":"sto/ksq_isu_base_info",
    "futures":         "drv/fut_bydd_trd",     #   385 rows
}
```

### 2.2 Bronze envelope

원본 행을 그대로 담되, 재현에 필요한 메타를 함께 남긴다.

```json
{
  "endpoint": "etp/etf_bydd_trd",
  "bas_dd": "20260827",
  "fetched_at": "2026-08-28T00:57:12+00:00",
  "http_status": 200,
  "row_count": 1163,
  "rows": [ { "BAS_DD": "20260827", "...": "..." } ]
}
```

`rows` 는 **API 응답 원문 그대로**다 — 키 리네이밍, 타입 변환, 공백 제거 일절 없음.

### 2.3 동시성

`httpx.AsyncClient` + `asyncio.Semaphore(settings.max_concurrency)`. 세션 간 의존성이 없으므로 순수 fan-out. `RateLimiter` 가 전역 토큰 버킷으로 초당 요청 수를 제한한다.

기본값: `max_concurrency=6`, `requests_per_second=5.0`. 2,125 세션 ETF backfill ≈ **7분**.

### 2.4 결정론적 테스트 전략

네트워크는 `httpx.MockTransport` 로 대체한다(testing.md §2 "외부 네트워크 경계에서만 mocking"). `RateLimiter`/`QuotaLedger` 는 `monotonic`/`sleep`/`today` 를 주입받아 **시간 의존 테스트를 제거**한다 — `time.sleep` 기반 타이밍 단언은 CI 에서 반드시 flaky 해진다.

---

## 3. Assumptions

- **A-1**: 일일 쿼터 기본값 8,000 (비공식 10,000 보고에 20% 안전마진). `Settings.daily_call_quota` 로 조정.
- **A-2**: 쿼터 리셋 기준은 KST 자정. 근거 없음 → 보수적으로 KST 날짜 키 사용하고, 리셋 시점이 다르면 조기 중단될 뿐 초과 호출은 발생하지 않는다(fail-closed 방향).
- **A-3**: KRX 는 `basDd` 를 2010-01-04 부터 제공(실측). 그 이전 요청은 빈 배열이므로 계획 단계에서 잘라낸다.

---

## 4. Execution Target

```bash
uv run pytest tests/unit/data -q --tb=short
uv run python tools/agent_skills/lean_check.py --spec docs/specs/02_krx_ingestion_contract.json

# 실제 수집 (1순위 데이터)
uv run mt-etf ingest --dataset etf_daily --start 2018-01-01 --end 2026-08-27
```
