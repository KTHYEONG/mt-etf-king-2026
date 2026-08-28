# 04. Data Pipeline — L1 계층

KRX Open API 에서 raw JSON 을 수집하고, 검증된 silver Parquet 패널을 만드는 파이프라인입니다.

---

## 1. KRX API 실측 사실 (설계 전제)

아래는 `scratch/probe_*.py` 로 검증된 동작입니다. **가정이 아니라 관측 결과**입니다.

| 사실 | 영향 |
| --- | --- |
| Base URL: `https://data-dbg.krx.co.kr/svc/apis` | provider 고정 |
| 인증: `AUTH_KEY` 헤더 | settings 에서 로드 |
| 파라미터: `basDd` 만 동작 | 1 call = 1 session 전체 ETF 단면 |
| `isuCd`, `strtDd/endDd` 무시 | 날짜 범위 조회 불가 → 일별 backfill |
| 휴장일: 1,163행, 가격 전부 `""` | 행 수 ≠ 거래일 |
| 미래/불량 날짜: `200 + []` | 에러 아님 — 빈 응답 처리 |
| `/etp/etf_pdf`, `/etp/etf_isu_base_info` → 404 | PDF·상장일 API 없음 |
| ETF 상장일 API 없음 | 패널 first_seen/last_seen 으로 유도 |

### TRAP 요약

```
TRAP-1: 휴장일도 1,163행 반환, 가격 ""
TRAP-2: 미래 날짜 200 + [] (에러 아님)
TRAP-3: 미지원 파라미터 조용히 무시
```

---

## 2. Provider 계층

```
MarketDataProvider (Protocol)
    └── KRXOpenAPIProvider
            ├── RateLimiter (token bucket)
            └── QuotaLedger (일일 호출 한도 추적)
```

### 2.1 엔드포인트

| Dataset | Path | 용도 |
| --- | --- | --- |
| ETF 일별 | `/etp/etf_bydd_trd` | OHLCV, NAV, LIST_SHRS |
| 지수 일별 | `/idx/krx_dd_trd` | regime, breadth |
| 주식 일별 | `/sto/stk_bydd_trd` | (보조, 필요 시) |

### 2.2 Rate Limit & Quota

- `rate_limit_rps` 초과 시 sleep
- `QuotaLedger` → `data/state/krx_quota.json` 에 일별 호출 수 기록
- quota 소진 시 **당일 ingest 중단** (다음 거래일 재시도)

---

## 3. Bronze Layer

### 3.1 원칙

- **write-once**: 동일 `{dataset}/{date}` 재수집 시 revision suffix (`_r2.json`)
- API 응답을 **해석 없이** envelope 으로 저장
- bronze 는 "법적 원본" — silver 오류 시 bronze 에서 재빌드

### 3.2 Envelope 구조

```json
{
  "meta": {
    "dataset": "etp/etf_bydd_trd",
    "bas_dd": "20260827",
    "fetched_at": "2026-08-27T15:30:00+09:00",
    "row_count": 1163,
    "http_status": 200
  },
  "payload": [ /* raw API rows */ ]
}
```

---

## 4. Silver Layer

### 4.1 디코딩 규칙 (INV-1)

| KRX 값 | Python |
| --- | --- |
| `""` | `None` |
| `"0"` | `0` (명시적 0) |
| `"1,234.56"` | `1234.56` (콤마 제거) |

**`""` → 0 변환은 절대 금지.** 휴장일 빈 가격이 0으로 들어가면 수익률·모멘텀이 오염됩니다.

### 4.2 ETF Daily 스키마 (핵심 컬럼)

| 컬럼 | KRX 필드 | 용도 |
| --- | --- | --- |
| `date` | `BAS_DD` | 세션 |
| `isu_cd` | `ISU_CD` | 종목코드 |
| `isu_nm` | `ISU_NM` | 종목명 |
| `close` | `TDD_CLSPRC` | 종가 |
| `open` | `TDD_OPNPRC` | 시가 (체결) |
| `volume` | `ACC_TRDVOL` | 거래량 |
| `trading_value` | `ACC_TRDVAL` | 거래대금 |
| `nav` | `NAV` | 순자산가치 |
| `list_shrs` | `LIST_SHRS` | 설정/해지 단위 |
| `mktcap` | `MKTCAP` | 시가총액 |

### 4.3 Creation/Redemption Flow

```
ΔLIST_SHRS × NAV ≈ 자금 유입/유출
```

`LIST_SHRS` 변화와 `NAV` 를 결합하면 ETF 설정·환매 흐름을 정확히 분해할 수 있습니다. 이는 `features/flow.py` 의 입력입니다.

---

## 5. Validation Gates

`PanelValidator` 는 silver 쓰기 **전에** 실행됩니다. CRITICAL 실패 시 Parquet 갱신 중단.

| Gate | 조건 | 심각도 |
| --- | --- | --- |
| `valid_price_ratio` | 거래일에 유효 close 비율 ≥ 90% | CRITICAL |
| `row_count_stability` | 전일 대비 행 수 ±20% 이내 | WARNING |
| `nav_positive` | NAV > 0 인 행 비율 ≥ 95% | CRITICAL |
| `no_future_dates` | date ≤ today | CRITICAL |
| `duplicate_keys` | (date, isu_cd) 유일 | CRITICAL |

휴장일 판정: **행 수가 아니라 `valid_price_ratio < threshold`** 로 판단합니다.

---

## 6. Backfill 전략

```
BackfillPlanner
  ├── missing_sessions(calendar, bronze_index)
  ├── prioritize: 최근 → 과거 (대회 임박 시 최신 우선)
  └── run_backfill(dates, provider, bronze_store)
```

- 날짜 범위 API 없음 → **세션 단위 순차 호출**
- 2018-01-01 ~ 현재 ≈ 2,000+ calls → quota·rate limit 고려한 배치
- 이미 bronze 에 있는 날짜는 skip (write-once)

---

## 7. Silver Panel (03 spec)

일별 단면 JSON 을 **종목×날짜 패널**로 pivot 합니다.

```
etf_daily.parquet
  index: (date, isu_cd)
  columns: open, close, volume, trading_value, nav, list_shrs, ...
```

### 7.1 패널 불변식

- `(date, isu_cd)` 유일
- `date` 는 XKRX session 만
- 결측은 `null` (forward-fill 금지 — feature 계층에서 PIT 윈도우로 처리)
- `first_seen` / `last_seen` 은 패널 등장·소멸로 유도 (상장일 API 없음)

---

## 8. 유동성 스냅샷

10억 자본, ADV 1% 참여율 기준:

| 지표 | 값 |
| --- | --- |
| 전체 ETF | ~1,163 |
| ADV ≥ 1억 (거래 가능 후보) | ~200 |
| 10억 × 1% = 1,000만 이하 ADV | **26개** |

유동성은 universe·portfolio 계층의 hard constraint 입니다. 데이터 파이프라인은 `trading_value` 를 정확히 보존해 downstream 이 ADV 를 계산하게 합니다.

---

## 9. CLI

| Command | 동작 |
| --- | --- |
| `mt-etf ingest --from 2018-01-01 --to today` | bronze backfill |
| `mt-etf normalize --dataset etf` | bronze → silver + validation |
| `mt-etf normalize --validate-only` | silver 재검증만 |
