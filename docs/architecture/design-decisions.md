# Architecture Decision Records (ADRs)

본 문서는 시스템 설계 및 구현 과정에서 결정된 핵심 아키텍처 의사결정을 ADR 형식으로 기록합니다.

---

## ADR-01: Tournament Objective Function (Distribution Tail Optimization vs Sharpe Maximization)

### Decision
포트폴리오 평가 및 전략 채택의 주 목적함수를 샤프 지수(Sharpe Ratio)나 평균-분산 최적화(MVO)가 아닌, **36거래일 롤링 수익률 우측 꼬리 확률 $P(R_{36d} > \\theta)$ ($\\theta \\in \\{30\\%, 40\\%, 50\\%, 60\\%\\}$) 및 파산 제약($P(R < -25\\%) \\le 5\\%$)**으로 정의한다.

### Context
머니투데이 모의투자 대회는 36거래일(약 8주) 동안 진행되며, 상금은 1위(1,000만 원)와 2위(500만 원)에만 집중된 계단형 보상 구조를 가집니다. 3위부터 400위까지의 기대 상금은 0원으로 동일합니다.

### Alternatives
1. **전통적 샤프 지수 극대화**: 변동성을 최소화하고 다변화된 자산 배분을 유지.
2. **단일 2025년 대회 리플레이 수익률 단독 채택**: 직전 연도 대회 기간에 가장 높은 수익률을 낸 파라미터 선정.
3. **롤링 36D 분포 우측 꼬리 확률 최적화 (선택)**.

### Selected Approach
대안 3을 채택. 2018~2026년 전체 역사적 데이터에서 36세션 윈도우 2,090+개를 생성하여 수익률 분포 전체를 도출하고, B0 벤치마크 대비 $P(R > 30\\%) \\ge +2\\%p$ 개선(G1) 및 극단 손실 제약(G2a)을 모두 충족하는 전략만 승격.

### Rationale
* 대안 1(샤프 극대화)은 연율 8~12% 수준의 낮은 변동성 포트폴리오를 만들어 대회 1~2위 도달 확률이 0에 수렴합니다.
* 대안 2는 단 1회의 표본 경로(Sample Path)에 과적합되어 다음 대회 환경에서 심각한 언더퍼폼으로 이어질 위험이 큽니다.
* 우측 꼬리 확률을 목적함수로 설정하되, 회복 불가능한 손실 구간(-25% 이하)에 하드 캡을 씌움으로써 대회 특성에 정확히 부합하는 효용 함수를 구현했습니다.

### Trade-offs
* 개별 윈도우의 일별 변동성 및 최대 낙폭(MDD)이 전통적 펀드 대비 커질 수 있습니다.
* 장기 복리 자산 배분 전략으로는 부적합하며, 단기 토너먼트 우승 최적화에 특화된 시스템입니다.

---

## ADR-02: Next-Open Execution Protocol (Elimination of Same-Bar Look-Ahead Bias)

### Decision
모든 전략 시그널은 $t$일 장 마감(15:30 KST) 종가를 기준으로 산출하며, 체결은 **익일 개장 시가($t+1$ 09:00 KST Open)**에 체결 수수료 및 슬리피지 모델을 가산하여 반영한다.

### Context
통상적인 백테스터들이 $t$일 종가에 시그널을 계산하면서 동시에 $t$일 종가에 체결되는(Same-bar Fill) 가정을 두는 경우가 많습니다. 이는 마감 동시호가 10분 전에 최종 종가를 미리 알고 주문을 완료해야 하는 비현실적인 가정(Look-ahead Bias)입니다.

### Alternatives
1. **Same-bar Close Fill**: $t$일 종가 시그널 $\\to$ $t$일 종가 체결 (비현실적).
2. **Next-Close Fill**: $t$일 종가 시그널 $\\to$ $t+1$일 종가 체결 (불필요한 1일 지연).
3. **Next-Open Fill (선택)**: $t$일 장 마감 후 시그널 확정 $\\to$ $t+1$일 동시호가/시가 체결.

### Selected Approach
대안 3을 채택. `NextOpenExecution` 모듈을 구축하고 장 마감 후 16:00 KST에 배치가 돌아 익일 장전 HTS 주문 가이드를 완성하도록 설계.

### Rationale
* 운영자가 장 마감 후 여유 있게 모델의 추천을 검토하고 익일 개장 전 HTS에 주문을 안전하게 제출할 수 있는 실제 운영 환경과 정확히 일치합니다.
* 백테스트와 실전 간의 괴리를 최소화하며, 시가 갭 변동과 슬리피지 리스크를 백테스트 결과에 충실히 반영합니다.

### Trade-offs
* 야간 글로벌 시장 변동에 따른 오버나이트 갭(Overnight Gap) 슬리피지 위험에 노출됩니다.
* 개장 직후 거래정지나 이상 시가가 발생할 경우 체결 불가(Unfillable) 처리가 수반됩니다.

---

## ADR-03: Dual-Mode Universe Separation (Structural vs Deployment)

### Decision
종목 유니버스를 연구용인 **`structural` 모드**와 실전 채택용인 **`deployment` 모드**로 분리하고, 전략 승격 판정은 오직 `deployment` 모드 결과로만 한정한다(`require_universe_mode: deployment`).

### Context
현재 상장된 1,100여 개 ETF 중 대회 규정에 의해 매매 가능한 종목은 후원 10개 운용사가 발행하고 유동성(ADV $\\ge$ 1억 원)을 충족하는 종목군으로 한정됩니다. 반면 과거 데이터 백테스트 시 현재 시점의 후원사 목록을 과거에 소급 적용하면 생존 편향이 발생하며, 반대로 전 종목을 허용하면 실전에서 매수할 수 없는 유령 알파가 발생합니다.

### Alternatives
1. **전 종목 단일 유니버스**: 상장된 모든 ETF를 자유롭게 매수 백테스트.
2. **현재 후원사 고정 유니버스**: 과거 8년 전 구간에 현재 후원사 필터 강제 적용.
3. **Dual-Mode 분리 운용 (선택)**: 아이디어의 구조적 타당성은 `structural`로 검증하고, 실전 채택 및 ML 학습은 `deployment`로 검증.

### Selected Approach
대안 3을 채택. [`src/universe/provider.py`](../../src/universe/provider.py)에 모드 스위치를 두고, `configs/universe.yaml`의 적격성 파이프라인(`existence $\\to$ price $\\to$ history $\\to$ sponsor $\\to$ liquidity $\\to$ rules`)을 일관되게 집행.

### Rationale
* 팩터의 통계적 유의성은 시장 전체(`structural`)에서 확인하여 과적합을 방지합니다.
* 최종 전략 채택 및 데일리 주문 산출은 실제 모의투자 계좌에서 주문 가능한 종목(`deployment`)으로만 한정하여 실전 주문 집행 가능성을 안정적으로 확보합니다.

### Trade-offs
* 두 유니버스 패널을 각각 빌드하고 검증해야 하므로 데이터 파이프라인 관리 복잡도가 다소 증가합니다.

---

## ADR-04: Post-Crash Anchor Sleeve for Tournament Regime Adaptation

### Decision
기존 P27 챔피언 전략(`sticky.mom60_raw`)에 지수 급락 후 반등 구간 전용 앵커 슬리브를 결합한 **`sticky.mom60_post_crash_anchor`를 최종 챔피언 전략으로 승격**한다.

### Context
P27은 60일 모멘텀 기반으로 우수한 우측 꼬리 수익률을 보였으나, 2024~2026년과 같이 코스피 지수가 급락한 직후 단기 급반등하는 구간(CRASH_REBOUND 슬리브)에서는 `mom60 < 0` 조건으로 인해 100% 현금(CASH)으로 철수하여 $P(R>30\\%) = 0\\%$로 기회를 완전히 방기하는 구조적 한계가 있었습니다. 2026년 대회 시작 시점(2026-09-21)이 바로 이 급락-반등 국면에 인접해 있습니다.

### Alternatives
1. **P27 단독 유지**: 현금 보존을 최우선으로 하고 장기 모멘텀이 회복될 때까지 대기.
2. **모멘텀 윈도우 일괄 단기화(mom20/mom10)**: 전 구간에서 단기 모멘텀 추종 (전체 윈도우에서 잦은 매매와 휩소 손실로 우측 꼬리 파괴 확인).
3. **상태 머신 기반 조건부 앵커 슬리브 신설 (선택)**.

### Selected Approach
대안 3을 채택. 평시(LOTTERY_ON)에는 P27의 60일 모멘텀 랭킹을 유지하되, 급락 후 반등 국면(CRASH_REBOUND) 및 앵커 보유 중인 비활성 국면에서는 대표 지수 레버리지(코스닥150레버리지 233740, KODEX 레버리지 122630)를 20일 모멘텀으로 앵커링하고 20일 드로우다운 15% 손절 가드로 방어.

### Rationale
* 실측 결과 2,097개 롤링 윈도우에서 $P(R>30\\%)$가 6.78% $\\to$ **7.77%**로 +0.99%p 향상.
* 극단 꼬리 위험 CVaR(5%) 역시 -23.16% $\\to$ **-21.56%**로 개선되어 파산 게이트(G2a)를 여유 있게 통과.

### Trade-offs
* 앵커 ETF에 대한 포지션 의존성이 존재하며, 반등 실패 후 2차 급락 시 15% 손절 라인까지의 낙폭을 감수해야 합니다.

---

## ADR-05: Parquet / Polars Columnar Storage vs Relational Database

### Decision
데이터 저장소로 PostgreSQL이나 SQLite 같은 RDBMS 대신, **Bronze 원본 gzip JSON (`.json.gz`)과 Silver/Gold In-Process Parquet + Polars 구조**를 채택한다.

### Context
일별 1,100여 개 ETF의 8년치 시계열은 약 수백만 행 수준으로, 빅데이터 클러스터가 필요한 규모는 아니지만 일별 단면 랭킹 및 롤링 윈도우 시뮬레이션 시 높은 디스크 I/O 처리량과 빠른 벡터화 연산이 요구됩니다.

### Alternatives
1. **RDBMS (PostgreSQL / TimescaleDB)**: 정규화된 테이블 및 네트워크 질의.
2. **SQLite**: 로컬 파일 기반 관계형 DB.
3. **Parquet 파일 시스템 + Polars (선택)**: 컬럼 기반 파티셔닝 파일 저장소.

### Selected Approach
대안 3을 채택. `DataPaths` 불변 경로 계층 하에 `data/normalized/`와 `data/features/`를 Parquet으로 관리하고 Polars 지연 평가(LazyFrame)로 분석.

### Rationale
* 외부 DB 데몬 설치나 네트워크 포트 설정 없이 `git clone` 및 `uv sync`만으로 즉시 단일 머신에서 재현 가능합니다.
* Polars의 멀티스레드 컬럼너 엔진을 통해 수백만 행의 단면 랭킹과 모멘텀 연산을 수 초 이내에 완료합니다.
* Zero-copy Arrow 메모리 구조를 통해 머신러닝(LightGBM) 파이프라인과 직접 연동됩니다.

### Trade-offs
* 개별 행(Row-level) 단위의 ACID 트랜잭션 업데이트가 어려우며, 일별 배치 시 증분 Append 또는 파티션 재생성이 필요합니다.

---

## ADR-06: Purged Walk-Forward GBDT Capacity Bound vs Deep Learning

### Decision
머신러닝 적용 범위를 **단면 랭킹 전용 얕은 GBDT (LightGBM Ranker)**로 한정하고, 엄격한 용량 상한(`max_depth=4`, `num_leaves=8`, `min_data_in_leaf=100`)과 Purged Walk-Forward CV를 의무화하며, 딥러닝/강화학습은 범위 외(Out-of-scope)로 배제한다.

### Context
8년간의 일별 ETF 데이터는 총 행 수로는 수십만 건에 달하지만, 36일 전방 수익률 라벨의 시계열 중첩과 국내 ETF 간의 높은 단면 상관(코스피200/S&P500 복제 ETF 다수)으로 인해 **유효 독립 표본 수는 약 2,400개에 불과**합니다.

### Alternatives
1. **딥러닝 시계열 모델 (LSTM, Transformer)**: 고차원 피처와 복잡한 신경망 적용.
2. **강화학습 (RL)**: 동적 포트폴리오 에이전트 학습.
3. **용량 제약형 GBDT LambdaRank + Purged CV (선택)**.

### Selected Approach
대안 3을 채택. `PurgedWalkForward` CV(Purge 10세션, Embargo 10세션)를 적용하고, 룰 기반 베이스라인(B1~B5) 대비 통계적으로 우월할 때만 채택되는 조건부 후보 모델로 배치.

### Rationale
* 2,400개의 유효 표본으로 수만~수백만 개의 신경망 가중치를 학습시키면 필연적으로 시장 노이즈를 암기하여 과적합됩니다.
* 절대 수익률 회귀는 시장 베타(지수 등락)에 매몰되므로, 동일 시점 단면 내 상대 우위를 학습하는 LambdaRank가 종목 선택력을 분리하는 데 적합합니다.

### Trade-offs
* 비선형 복합 패턴이나 텍스트/뉴스 등 비정형 데이터의 직접 결합이 제한됩니다.

---

## ADR-07: In-Memory Secret Decryption via SOPS + Age (Zero Plaintext Secrets on Disk)

### Decision
KRX API 키 등 민감한 인증 정보를 평문 `.env` 파일에 저장하지 않고, **저장소에 커밋된 `.env.enc` 파일을 SOPS와 Age 비대칭 키를 통해 프로세스 기동 시 메모리 상에서만 복호화**하여 사용한다.

### Context
Git 저장소 관리 시 실수로 평문 `.env` 파일이 커밋되거나 서버 디스크에 방치되어 API 키가 탈취되는 보안 사고가 빈번합니다.

### Alternatives
1. **평문 `.env` 및 `.gitignore` 의존**: 개발자 로컬에 평문 보관.
2. **클라우드 KMS 연동 (AWS/GCP Secrets Manager)**: 외부 클라우드 벤더 종속성 발생.
3. **SOPS + Age 로컬 파일 암호화 (선택)**: 저장소 내 암호화 파일 보관 및 프로세스 메모리 직접 복호화.

### Selected Approach
대안 3을 채택. [`src/core/sops_env.py`](../../src/core/sops_env.py)에서 `sops -d` 명령을 subprocess로 호출하여 stdout 스트림만 pydantic-settings에 주입.

### Rationale
* 디스크에 단 1바이트의 평문 시크릿도 기록되지 않아 안전한 보안 체계를 유지합니다.
* 클라우드 벤더 종속성 없이 로컬 키(`~/.config/sops/age/keys.txt`) 하나로 일관되게 형상 관리가 가능합니다.

### Trade-offs
* 환경에 `sops` 바이너리 설치 및 Age 개인키 배포가 선행되어야 합니다 (키 누락 시 fail-closed).
