# Data Flow & Temporal Architecture

본 문서는 외부 데이터 소스(KRX Open API)에서 수집된 로우 데이터가 정규화, Point-in-Time 유니버스 선별, 피처 엔지니어링, 전략 스코어링을 거쳐 최종 포트폴리오 주문으로 확정되기까지의 전체 데이터 흐름과 시계열 정합성 보장 메커니즘을 설명합니다.

---

## 1. End-to-End Data Pipeline Stages

| Stage | Input | Processing | Output | Main Module |
| :--- | :--- | :--- | :--- | :--- |
| **1. Bronze Ingestion** | KRX OpenAPI JSON (`etp/etf_bydd_trd`, `idx/kospi_dd_trd`) | 토큰 버킷 Rate Limiter(2~5 rps), 일일 쿼터 확인(`krx_quota.json`), 원형 응답 압축 저장 | `data/raw/krx/.../{YYYY}/{YYYYMMDD}.json.gz` | [`src/data/providers/krx.py`](../../src/data/providers/krx.py)<br>[`src/data/bronze.py`](../../src/data/bronze.py) |
| **2. Silver Normalization** | Bronze JSON envelopes | 휴장일 더미 레코드 필터링, 정적 타입 캐스팅(`""` $\\to$ `None`), 일별 증분 병합 | `data/normalized/etf_daily.parquet`<br>`data/normalized/index_daily.parquet` | [`src/data/silver.py`](../../src/data/silver.py)<br>[`src/data/schema.py`](../../src/data/schema.py)<br>[`src/data/validation.py`](../../src/data/validation.py) |
| **3. PIT Universe Selection** | Silver Parquet + 운용사 브랜드 설정 | 생존자 편향 없는 상장기간 판정, 거래정지 제외, 후원사 매칭, ADV 유동성 게이트 | Eligible Ticker Set at $t$ (`UniverseSnapshot`) | [`src/universe/provider.py`](../../src/universe/provider.py)<br>[`src/universe/instruments.py`](../../src/universe/instruments.py) |
| **4. Gold Feature Pipeline** | Silver Panel + PIT Universe + XKRX Calendar | `assert_pit` 가드, 팬텀 세션 선별 제외, 모멘텀·변동성·자금유입·브레드스·시장국면 벡터 연산 | `data/features/etf_features.parquet` | [`src/features/builder.py`](../../src/features/builder.py)<br>[`src/features/pit.py`](../../src/features/pit.py)<br>[`src/features/regime.py`](../../src/features/regime.py) |
| **5. Alpha Signal Generation** | Gold Feature Panel | 단면 백분위수 랭킹, 지수 모멘텀 필터, 급락 후 반등 앵커 룰 평가 | Ticker Scores at $t$ | [`src/strategies/sticky/model.py`](../../src/strategies/sticky/model.py)<br>[`src/alpha/base.py`](../../src/alpha/base.py) |
| **6. Portfolio & Overlay** | Alpha Scores + 이전 상태(`PositionState`) | LeverageFamily 중복 제거, Top-1 집중 비중 산출, ADV 5% 참여율 한도, 최소 보유 세션 가드 | Target Weights & Orders | [`src/portfolio/selection.py`](../../src/portfolio/selection.py)<br>[`src/portfolio/sizing.py`](../../src/portfolio/sizing.py)<br>[`src/portfolio/state.py`](../../src/portfolio/state.py) |
| **7. Execution / Decision** | Target Weights + Market Open Prices | $t+1$ 시가 체결 시뮬레이션(백테스트) 또는 HTS 입력용 수량/금액 렌더링(라이브) | Backtest PnL Distribution / `results/decide_daily/*.json` | [`src/backtest/execution.py`](../../src/backtest/execution.py)<br>[`src/cli/commands/decide/render.py`](../../src/cli/commands/decide/render.py) |

---

## 2. Temporal Integrity & Timestamp Architecture

금융 시계열 파이프라인에서 가장 중요한 품질 기준은 **미래 참조 편향(Look-Ahead Bias)**과 **생존자 편향(Survivorship Bias)**의 배제입니다. 본 시스템은 아키텍처 레벨에서 이를 엄격히 관리합니다.

### 2.1 시간 축 정의 (Event vs Processing vs Execution Time)
모든 시계열 기준은 한국 표준시(KST, UTC+9) 및 한국거래소 개장일 기준(XKRX)으로 정렬됩니다.

```text
Session t Close (15:30 KST)
      │
      ▼  [Event Time: t]
16:00 KST Batch Processing
      │  - Ingest KRX data up to t
      │  - Normalize & validate
      │  - Build features strictly using data <= t
      │  - Compute target weights for session t+1
      ▼  [Processing Time: t 16:00 KST]
State Persisted (data/state/*.json)
      │
      ▼  [Execution Time: t+1 09:00 KST]
Session t+1 Open (09:00 KST)
      └─ NextOpenExecution fills orders at t+1 Open price * (1 + slippage)
```

1. **Event Time ($t$)**:
   * 당일 장 마감(15:30 KST) 시점에 확정된 시장 데이터(종가, 거래량, 거래대금, NAV 등).
2. **Processing Time ($t$ 16:00 KST)**:
   * 장 마감 후 실행되는 일일 배치 파이프라인([`scripts/daily_pipeline.sh`](../../scripts/daily_pipeline.sh)).
   * 결정 시점 이전($\\le t$)의 데이터만을 엄격히 입력으로 허용.
3. **Execution Time ($t+1$ 09:00 KST)**:
   * 체결은 익일 개장 시가(Next Open)에 이루어집니다.
   * 당일 종가 시그널을 당일 종가에 즉시 체결시키는 비현실적인 Same-bar Fill 가정(INV-10)은 코드 레벨에서 배제됩니다.

### 2.2 Point-in-Time (PIT) 검증 메커니즘
* **`assert_pit(frame, decision_date)`**:
  * 모든 피처 연산 및 유니버스 필터 함수 진입 시점에 호출됩니다.
  * 프레임 내에 `date > decision_date`인 데이터가 포함된 경우 즉시 `PitViolationError`를 발생시키고 실행을 중단(Fail-closed)합니다.
* **XKRX 캘린더 세션 정렬 (`align_session_grid`)**:
  * 단순 일자 나열이 아닌 `exchange-calendars` XKRX를 기준으로 유효 세션 목록을 확정합니다.
  * 종목별 최초 관측일(`first_seen`)과 최종 관측일(`last_seen`) 사이의 개장 세션을 결합하여 거래 결측일을 투명하게 노출합니다.
* **팬텀 세션(Phantom Session) 선별 처리 (`restrict_to_traded_sessions`)**:
  * 전산 장애나 임시 휴장 등으로 인해 시장 전체의 가격이 결측인 세션이 존재할 경우, Polars의 `rolling_*` 연산은 물리적 행 개수를 카운트하므로 이후 윈도우 전체로 NaN이 전파되는 결함이 발생합니다.
  * 피처 엔진은 롤링 연산 전 실제 거래된 세션만을 슬라이싱하여 피처를 산출한 뒤 전체 그리드에 Left Join합니다.

### 2.3 가격 데이터 특성 (Adjusted vs Unadjusted)
* KRX Open API는 **수정주가가 아닌 당일 실제 체결가(Unadjusted Price)**를 제공합니다.
* 레버리지/인버스 ETF의 36세션 수익률은 기초지수 수익률의 2배가 아니며, 일별 리밸런싱에 따른 변동성 감쇠(Volatility Drag)가 복합적으로 작용합니다.
* 시스템은 지수 수익률에 승수를 곱하는 **임의의 합성 가격 생성을 엄격히 배제**하며, KRX에서 실제 체결된 ETF의 시장 가격만을 사용합니다 (INV-14).

### 2.4 결측치 및 KRX 비정상 응답 처리 (Fail-Closed)
* **공백 문자열 디코딩**: KRX API는 결측치를 빈 문자열(`""`)로 반환합니다. 이를 `0.0`으로 자동 변환하면 수익률이 -100%로 왜곡되므로, `DatasetSchema`에서 강제로 `None`으로 변환합니다 (INV-1).
* **휴장일 더미 레코드 필터**: KRX는 휴장일에도 1,163행의 레코드를 반환하지만 모든 가격 필드가 비어 있습니다. 행 수가 아닌 `valid_price_ratio < threshold`를 기준으로 휴장일 응답을 식별하여 폐기합니다 (INV-2).

---

## 3. Storage Layer & Schema Specification

```text
data/
├── raw/krx/                          # [Bronze] 불변 원본 압축 JSON (.json.gz)
│   ├── etp/etf_bydd_trd/2026/20260910.json.gz
│   └── idx/kospi_dd_trd/2026/20260910.json.gz
├── normalized/                       # [Silver] 정규화된 시계열 패널 (Parquet)
│   ├── etf_daily.parquet             # ETF 일별 OHLCV, NAV, 상장주식수 등
│   └── index_daily.parquet           # 지수 일별 OHLCV, 시가총액 등
├── features/                         # [Gold] 피처 엔지니어링 패널 (Parquet)
│   └── etf_features.parquet          # 모멘텀, 변동성, 브레드스, 레짐 결합 패널
└── state/                            # [State] 런타임 연속성 및 라이브 상태 (JSON)
    ├── krx_quota.json                # API 일일 호출량 추적
    └── sticky_mom60_post_crash_anchor_position.json
```

### Silver: `etf_daily.parquet` 핵심 스키마
| 필드명 | 물리 타입 | 설명 | 제약 조건 |
| :--- | :--- | :--- | :--- |
| `date` | `Date` | 기준 거래일 (XKRX) | Non-null, Key |
| `ticker` | `String` | 단축 종목코드 (6자리) | Non-null, Key (`^[0-9A-Z]{6}$`) |
| `name` | `String` | 종목명 | Non-null |
| `close` | `Float64` | 당일 종가 (원) | Nullable |
| `open` | `Float64` | 당일 시가 (원) | Nullable |
| `high` | `Float64` | 당일 고가 (원) | Nullable |
| `low` | `Float64` | 당일 저가 (원) | Nullable |
| `volume` | `Int64` | 누적 거래량 (주) | Nullable |
| `trading_value` | `Int64` | 누적 거래대금 (원) | Nullable |
| `nav` | `Float64` | 순자산가치 (원) | Nullable |
| `market_cap` | `Int64` | 시가총액 (원) | Nullable |
| `shares_outstanding` | `Int64` | 상장주식수 (주) | Nullable |
| `underlying_index_name` | `String` | 기초지수명 | Nullable |

### Gold: `etf_features.parquet` 핵심 피처 그룹
* **Momentum**: `mom_3`, `mom_5`, `mom_10`, `mom_20`, `mom_40`, `mom_60` (수익률)
* **Trend & Structure**: `ma_ratio_20`, `ma_ratio_60`, `drawdown_20`, `breakout_20`
* **Volatility**: `rv_5`, `rv_20` (실현 변동성), `downside_vol_20`
* **Liquidity & Flow**: `adv_20` (20일 평균 거래대금), `volume_expansion`, `creation_flow_20`
* **Cross-Sectional**: `cs_rank_mom60`, `cs_percentile_rank`, `mom_accel_20` (가속도)
* **Market Regime**: `regime` (STRONG_RISK_OFF, RISK_OFF, NEUTRAL, RISK_ON, STRONG_RISK_ON)
