# Component Breakdown

본 문서는 시스템을 구성하는 핵심 모듈들의 책임(Responsibility), 입출력 인터페이스, 의존성 관계 및 실제 구현체를 상세히 기술합니다.

---

### 1. KRXOpenAPIProvider & BronzeStore

**Responsibility**
* KRX Open API와의 HTTP 통신을 담당하며, 토큰 버킷 기반 Rate Limiter(기본 2~5 rps)와 일일 쿼터 추적기(`QuotaLedger`)를 통해 API 호출 제한을 준수합니다.
* 외부 API에서 수집된 원본 JSON 페이로드를 원형 그대로 일자별 gzip 압축 봉투(`.json.gz`)로 영속화(Write-Once)합니다.

**Input**
* 환경변수/SOPS 복호화된 KRX API 인증키 (`SecretStr`)
* 데이터셋 엔드포인트(`etp/etf_bydd_trd`, `idx/kospi_dd_trd`) 및 조회 일자 (`date`)

**Output**
* 불변 Bronze 압축 파일 (`data/raw/krx/{endpoint}/{year}/{YYYYMMDD}.json.gz`)
* `QuotaLedger` 업데이트 (`data/state/krx_quota.json`)

**Dependencies**
* `httpx`, `tenacity`, [`src/core/settings.py`](../../src/core/settings.py), [`src/core/paths.py`](../../src/core/paths.py)

**Key Implementation**
* [`src/data/providers/krx.py`](../../src/data/providers/krx.py): `KRXOpenAPIProvider`, `fetch_daily_trading()`
* [`src/data/providers/ratelimit.py`](../../src/data/providers/ratelimit.py): `RateLimiter`, `QuotaLedger`
* [`src/data/bronze.py`](../../src/data/bronze.py): `BronzeStore.write_envelope()`

---

### 2. SilverBuilder & DatasetSchema

**Responsibility**
* Bronze 단계의 원시 JSON 데이터를 파싱하고, 엄격한 스키마 검증 및 정규화(Normalization)를 수행하여 고성능 Parquet 포맷으로 변환합니다.
* KRX 결측치 공백 문자열(`""`)을 `None`으로 디코딩하여 0.0 왜곡을 방지하고, 휴장일 더미 레코드(1,163행 공백 가격)를 자동으로 식별 및 제외(Fail-closed)합니다.

**Input**
* Bronze 압축 JSON 파일들 (`*.json.gz`)
* `DatasetSchema` 정의 (`ETF_DAILY_SCHEMA`, `INDEX_DAILY_SCHEMA`)

**Output**
* Silver Parquet 테이블 (`data/normalized/etf_daily.parquet`, `index_daily.parquet`)

**Dependencies**
* `polars`, `pyarrow`, [`src/data/schema.py`](../../src/data/schema.py), [`src/data/validation.py`](../../src/data/validation.py)

**Key Implementation**
* [`src/data/schema.py`](../../src/data/schema.py): `DatasetSchema`, `decode_optional_float()`, `decode_rows()`
* [`src/data/silver.py`](../../src/data/silver.py): `SilverBuilder.build_panel()`
* [`src/data/validation.py`](../../src/data/validation.py): `PanelValidator.validate_silver()`

---

### 3. PointInTimeUniverse & InstrumentMaster

**Responsibility**
* 특정 결정 시점($t$)에 생존해 있고 거래 가능한 ETF 목록을 확정합니다.
* **Structural 모드**(전체 과거 상장 종목, 순수 팩터 연구용)와 **Deployment 모드**(대회 후원 10개 운용사 브랜드, 유동성 ADV $\\ge$ 1억 원, 최소 60일 거래 이력)를 엄격히 분리하여 전략 채택 시 생존자 편향과 실전 괴리를 차단합니다.
* 동일 기초지수를 추종하는 다중 배수 종목들(1X, 2X, -1X, -2X)을 `LeverageFamily`로 그룹화하여 포트폴리오의 중복 베팅을 제어합니다.

**Input**
* Silver ETF 패널 (`etf_daily.parquet`)
* 유니버스 설정 (`configs/universe.yaml`, `configs/sponsor_brands.yaml`)
* 기준 일자 (`as_of`)

**Output**
* `UniverseSnapshot`: $t$일 기준 적격 종목 리스트 및 메타데이터

**Dependencies**
* [`src/core/calendar.py`](../../src/core/calendar.py), [`src/universe/taxonomy.py`](../../src/universe/taxonomy.py)

**Key Implementation**
* [`src/universe/provider.py`](../../src/universe/provider.py): `PointInTimeUniverse.eligible_tickers()`
* [`src/universe/instruments.py`](../../src/universe/instruments.py): `InstrumentMaster.build()`
* [`src/universe/families.py`](../../src/universe/families.py): `LeverageFamily`

---

### 4. FeatureBuilder & PitGuard

**Responsibility**
* 일별 시계열 데이터로부터 모멘텀, 추세 왜곡, 변동성, 거래대금 팽창, 자금유입, 마켓 브레드스 및 5단계 시장 국면(Regime) 피처를 벡터화 연산합니다.
* 모든 피처 함수 진입 시 `assert_pit`를 실행하여 미래 데이터 누출을 차단하고, 팬텀 세션의 결측 전파를 방어합니다.

**Input**
* Silver ETF 및 Index 패널
* 피처 설정 (`configs/features.yaml`)

**Output**
* Gold Feature Parquet (`data/features/etf_features.parquet`)

**Dependencies**
* `polars`, [`src/core/calendar.py`](../../src/core/calendar.py), [`src/features/pit.py`](../../src/features/pit.py)

**Key Implementation**
* [`src/features/builder.py`](../../src/features/builder.py): `FeatureBuilder.build_panel()`
* [`src/features/pit.py`](../../src/features/pit.py): `assert_pit()`, `align_session_grid()`, `restrict_to_traded_sessions()`
* [`src/features/momentum.py`](../../src/features/momentum.py): `add_momentum()`
* [`src/features/regime.py`](../../src/features/regime.py): `classify_regime()`

---

### 5. AlphaModel & Sticky Strategy Engine

**Responsibility**
* 시그널을 생성하는 Alpha 모델 인터페이스(`AlphaModel` Protocol)와 대회 챔피언 전략의 핵심 룰을 실행합니다.
* **챔피언 전략(`sticky.mom60_post_crash_anchor`)**:
  * 정상 국면(LOTTERY_ON): 장기 60일 모멘텀 최선호 종목 선별.
  * 급락-반등 국면(CRASH_REBOUND): 지수 급락 후 반등 구간에서 P27(mom60)이 현금 100%로 철수하는 문제를 해결하기 위해 대표 지수 +2X 레버리지(코스닥150레버리지 233740, KODEX 레버리지 122630)를 20일 모멘텀으로 앵커링하고 20일 드로우다운 15% 손절 가드로 방어.

**Input**
* Gold Feature Dataframe
* PIT 적격 종목 집합

**Output**
* 종목별 상대강도 스코어 (`score`) 및 랭킹 (`rank`)

**Dependencies**
* [`src/strategies/protocol.py`](../../src/strategies/protocol.py), [`src/alpha/base.py`](../../src/alpha/base.py), [`src/strategies/sticky/config.py`](../../src/strategies/sticky/config.py)

**Key Implementation**
* [`src/strategies/sticky/model.py`](../../src/strategies/sticky/model.py): `StickyModel`
* [`src/strategies/sticky/model_runner.py`](../../src/strategies/sticky/model_runner.py): 국면별 슬리브 라우팅 (LOTTERY_ON vs CRASH_REBOUND)
* [`src/strategies/registry.py`](../../src/strategies/registry.py): 전략 팩토리 및 식별자 매핑

---

### 6. ClusterAwareSelection & Portfolio Policy

**Responsibility**
* Alpha 스코어를 바탕으로 실제 포트폴리오 목표 비중(Target Weights)을 결정합니다.
* **Family Deduplication**: 동일 `leverage_family` 내에서 복수 종목을 매수하지 않도록 최상위 1개만 통과시킵니다 (예: KODEX 200과 KODEX 레버리지의 중복 매수 차단).
* **Concentrated Sizing**: 대회 특성에 맞춰 Top-1 종목에 최대 95% 집중 배분하고 5% 현금 버퍼를 유지합니다.
* **Capacity & ADV Limit**: 개별 주문 규모가 20일 평균 거래대금의 5%(기본 0.05)를 초과하지 못하도록 제한합니다.

**Input**
* Alpha 점수표
* LeverageFamily 매핑 및 ADV 유동성 통계

**Output**
* 목표 비중 딕셔너리 (`Mapping[str, float]`)

**Dependencies**
* [`src/portfolio/constraints.py`](../../src/portfolio/constraints.py), [`src/portfolio/sizing.py`](../../src/portfolio/sizing.py)

**Key Implementation**
* [`src/portfolio/selection.py`](../../src/portfolio/selection.py): `ClusterAwareSelection`
* [`src/portfolio/constraints.py`](../../src/portfolio/constraints.py): `cap_target_weights_by_adv()`, `apply_portfolio_exposure_limits()`
* [`src/portfolio/sizing.py`](../../src/portfolio/sizing.py): `weights_from_scores()`

---

### 7. NextOpenExecution & BacktestEngine

**Responsibility**
* $t$일 장 마감 후 생성된 목표 비중을 $t+1$일 개장 시가(Next Open)에 체결시키는 시뮬레이션을 수행합니다.
* 거래정지, 결측 시가 등에 대한 체결 불가(Unfillable) 판정 및 수수료/슬리피지 비용을 즉시 반영합니다.
* `--protocol grid` 실행 시 비용축과 유동성 축(36개 셀)을 효율적으로 평가하기 위한 `SessionCacheRegistry`를 내장합니다.

**Input**
* 포트폴리오 목표 비중 시계열
* 전 세션 OHLCV 패널
* 비용 설정 (`CostConfig`: 수수료 1.5~3 bps, 슬리피지 3~10 bps)

**Output**
* 백테스트 세션별 체결 내역(`Fill`), 일별 NAV, 포지션 내역, 실현 노출 통계

**Dependencies**
* [`src/core/calendar.py`](../../src/core/calendar.py), [`src/backtest/costs.py`](../../src/backtest/costs.py), [`src/backtest/session_cache.py`](../../src/backtest/session_cache.py)

**Key Implementation**
* [`src/backtest/execution.py`](../../src/backtest/execution.py): `NextOpenExecution.resolve()`, `is_open_fillable()`
* [`src/backtest/engine.py`](../../src/backtest/engine.py): `BacktestEngine.run()`
* [`src/backtest/costs.py`](../../src/backtest/costs.py): `CostModel.apply()`

---

### 8. TournamentSimulator & ObjectiveGates

**Responsibility**
* 2018년부터 현재까지의 전체 역사적 기간에 대해 36거래일 롤링 윈도우(2,090+개)를 생성하여 36일 수익률 전체 분포(Quantiles, CVaR 5%, Peak Giveback)를 도출합니다.
* 하드 게이트 판정:
  * **G1**: 벤치마크(B0) 대비 $P(R_{36d} > 30\%) \ge +2\%p$
  * **G2a**: 파산 제약 $P(R_{36d} < -25\%) \le 5\%$
* **LOYO (Leave-One-Year-Out)**: 연도별 윈도우를 분리 평가하여 특정 강세장에만 기댄 과적합 전략을 판별합니다.

**Input**
* 전략 인스턴스 및 롤링 백테스트 패널
* 게이트 설정 (`configs/gates.yaml`)

**Output**
* `ReturnDistribution` 및 게이트 통과 여부 (`objective_gate_status`: PASS/FAIL)
* 실행 레지스트리 기록 (`docs/results/runs_registry.jsonl`)

**Dependencies**
* [`src/tournament/distribution/`](../../src/tournament/distribution), [`src/tournament/objective/`](../../src/tournament/objective)

**Key Implementation**
* [`src/tournament/simulator.py`](../../src/tournament/simulator.py): `TournamentSimulator.run()`
* [`src/tournament/objective_core.py`](../../src/tournament/objective_core.py): `evaluate_objective_gates()`, 게이트 상수 기준 정의
* [`src/tournament/loyo.py`](../../src/tournament/loyo.py): `LoyoReport`

---

### 9. PositionStateManager & Execution Ledger

**Responsibility**
* 라이브 의사결정 시 프로세스 재시작 간 상태를 유지하기 위해 현재 보유 종목, 보유 기간(`hold_len`), 진입가를 `data/state/{model}_position.json`에 영속화합니다.
* 세션 연속성이 훼손되거나 패널이 최신이 아닐 경우 주문 생성을 즉시 중단(Fail-closed)합니다.

**Input**
* 당일 확정 포지션 및 결정일

**Output**
* `data/state/{model}_position.json`
* 실행 원장 감사 레코드

**Dependencies**
* [`src/execution/ledger.py`](../../src/execution/ledger.py), [`src/portfolio/state.py`](../../src/portfolio/state.py)

**Key Implementation**
* [`src/strategies/sticky/factories.py`](../../src/strategies/sticky/factories.py): `persist_sticky_state()`, `load_sticky_state()`
* [`src/portfolio/state.py`](../../src/portfolio/state.py): `PositionState`
* [`src/execution/cash_accounting.py`](../../src/execution/cash_accounting.py): `CashAccountingLedger`

---

### 10. CLI & DailyRefresh Orchestrator

**Responsibility**
* 엔지니어 및 운영자를 위한 단일 커맨드라인 인터페이스(`mt-etf`)를 제공하며 15개 서브커맨드를 라우팅합니다.
* 장 마감 후 단일 명령(`mt-etf daily-refresh --decide`)으로 수집 $\to$ 정규화 $\to$ 피처 빌드 $\to$ 의사결정 추천 저장을 일괄 수행합니다.

**Input**
* CLI 옵션 (`--as-of`, `--model`, `--protocol`, `--eval-mode` 등)

**Output**
* CLI 터미널 서머리 렌더링 및 `results/decide_daily/{date}.json` 아티팩트

**Dependencies**
* `argparse`, [`src/cli/parser.py`](../../src/cli/parser.py), [`deploy/systemd/`](../../deploy/systemd)

**Key Implementation**
* [`src/cli/parser.py`](../../src/cli/parser.py): `build_parser()`, `SUBCOMMANDS`
* [`src/cli/commands/pipeline.py`](../../src/cli/commands/pipeline.py): `cmd_daily_refresh()`
* [`src/cli/commands/decide/render.py`](../../src/cli/commands/decide/render.py): `cmd_decide()`
