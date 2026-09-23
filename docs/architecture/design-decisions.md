# Architecture Decision Records (ADRs)

본 문서는 시스템 설계 및 구현 과정에서 결정된 핵심 아키텍처 의사결정과 엔지니어링 트레이드오프를 ADR(Architecture Decision Record) 형식으로 기록합니다.

---

## 1. Executive Decision Matrix

기술 면접관이 1분 내에 주요 설계 트레이드오프를 조망할 수 있는 핵심 요약표입니다.

| ADR | 의사결정 주제 | 기각된 대안 | 채택된 솔루션 | 핵심 엔지니어링 근거 및 트레이드오프 |
| :--- | :--- | :--- | :--- | :--- |
| **ADR-01** | **토너먼트 목적함수** | • 전통적 샤프 지수 극대화<br>• 단일 대회 리플레이 수익률 | **36거래일 롤링 우측 꼬리 확률($P(R_{36d} > 30\%)$) 최적화** + 파산 제약(G2a) | 1~2위에만 집중된 계단형 상금 구조 부합. 일별 변동성 증가는 감수하되 우승 기대값 극대화 |
| **ADR-02** | **체결 시뮬레이션 모델** | • 당일 종가 동시체결 (Same-bar)<br>• 익일 종가 체결 (Next-Close) | **익일 개장 시가 체결 (Next-Open Fill)** + 슬리피지 모델 | 장 마감(15:30) 후 시그널 확정 $\to$ 익일 09:00 체결. 미래 참조 편향(Look-Ahead) 원천 제거 |
| **ADR-03** | **유니버스 분리 정책** | • 전 종목 단일 유니버스<br>• 현재 후원사 과거 소급 유니버스 | **Dual-Mode 분리** (`structural` 연구용 vs `deployment` 실전용) | 팩터 통계 검증 시 생존 편향 방지 + 대회 규정 종목 주문 실행 가능성 확보 |
| **ADR-04** | **시장 국면 적응 전략** | • 장기 모멘텀(P27) 단독 유지<br>• 모멘텀 윈도우 일괄 단기화 | **급락 후 반등 앵커 슬리브** 결합 (`sticky.mom60_post_crash_anchor`) | 지수 급락 후 장기 모멘텀이 현금으로 과도하게 철수하는 결함 해결 ($P(R>30\%)$: 6.78% $\to$ 7.77%) |
| **ADR-05** | **데이터 저장소 & 엔진** | • RDBMS (PostgreSQL)<br>• SQLite 파일 DB | **In-Memory Polars + Parquet 파일 시스템** | 외부 DB 데몬 없이 재현 가능. 멀티스레드 컬럼너 엔진으로 수백만 행 수 초 내 벡터 연산 |
| **ADR-06** | **머신러닝 적용 범위** | • 심층 신경망 (LSTM, Transformer)<br>• 강화학습 (RL) 에이전트 | **용량 제약형 GBDT LambdaRank** + Purged Walk-Forward CV | 금융 시계열의 유효 독립 표본($n_{\text{eff}} \approx 2,400$) 한계 극복 및 시장 노이즈 과적합 방어 |
| **ADR-07** | **시크릿 인증키 관리** | • SOPS + Age 인메모리 복호화 | **표준 `.env` (git 비추적) 환경변수 주입** | 실보안 이득 대비 과도한 도구 체인 오버헤드(바이너리, 키 관리)를 식별하고 실용적 회귀 |

---

## 2. Detailed Architecture Decision Records

### ADR-01: Tournament Objective Function (Tail Optimization vs Sharpe Maximization)

* **의사결정 (Decision)**:
  포트폴리오 평가 및 전략 채택의 주 목적함수를 샤프 지수(Sharpe Ratio)나 평균-분산 최적화(MVO)가 아닌, **36거래일 롤링 수익률 우측 꼬리 확률 $P(R_{36d} > \theta)$ ($\theta \in \{30\%, 40\%, 50\%, 60\%\}$) 및 파산 제약($P(R_{36d} < -25\%) \le 5\%$)**으로 정의합니다.

* **배경 및 맥락 (Context)**:
  대회 상금 체계는 순위에 대한 계단 함수(Step Function)입니다. 1위(1,000만 원)와 2위(500만 원)를 제외하면 3위나 최하위나 기대 상금은 0원으로 동일합니다. 따라서 포트폴리오 변동성을 낮추고 샤프 지수를 높이는 전통적 분산 포트폴리오는 단기 대회에서 요구되는 높은 수익률에 도달할 수 없습니다.

* **선택 근거 (Rationale)**:
  * 2018~2026년 전체 역사적 데이터에서 36세션 윈도우 2,090+개를 추출하여 수익률 전체 분포를 분석했습니다.
  * 벤치마크(B0) 대비 $P(R > 30\%) \ge +2\%p$ 개선(G1)과 극단 손실 제약(G2a)을 결합하여, 공격적인 우승권 진입 확률과 파산 방지를 동시에 달성했습니다.

* **엔지니어링 트레이드오프 (Trade-offs)**:
  * 일별 변동성과 MDD가 전통 펀드 대비 커지므로 일반 복리 자산배분에는 부적합하며, 단기 대회 목적에 고도로 특화된 구조입니다.

---

### ADR-02: Next-Open Execution Protocol (Elimination of Look-Ahead Bias)

* **의사결정 (Decision)**:
  모든 전략 시그널은 $t$일 장 마감(15:30 KST) 종가를 기준으로 산출하며, 체결 시뮬레이션은 **익일 개장 시가($t+1$ 09:00 KST Open)**에 체결 수수료 및 슬리피지를 가산하여 집행합니다.

* **배경 및 맥락 (Context)**:
  통상적인 백테스트 엔진들이 당일 종가에 시그널을 계산하고 동시에 당일 종가에 체결시키는(Same-bar Fill) 가정을 두는 경우가 많습니다. 이는 마감 10분 전 호가와 종가를 미리 알고 주문을 완료해야 하는 비현실적인 미래 참조 편향(Look-Ahead Bias)입니다.

* **선택 근거 (Rationale)**:
  * 장 마감 후 16:00 KST에 자동 배치가 돌아 익일 장전 HTS 주문 가이드를 완성하는 실제 운영 환경과 완벽히 일치합니다.
  * 야간 글로벌 시장 변동에 따른 오버나이트 갭과 개장 시가 슬리피지(3~10 bps)를 백테스트 결과에 충실히 반영합니다.

* **엔지니어링 트레이드오프 (Trade-offs)**:
  * 개장 직후 거래정지나 이상 시가가 발생할 경우 체결 불가(Unfillable) 예외 처리가 수반됩니다.

---

### ADR-03: Dual-Mode Universe Separation (Structural vs Deployment)

* **의사결정 (Decision)**:
  종목 유니버스를 연구용인 **`structural` 모드**와 실전 채택용인 **`deployment` 모드**로 분리하고, 전략 승격 판정은 오직 `deployment` 모드 결과로만 한정합니다.

* **배경 및 맥락 (Context)**:
  현재 상장된 1,100여 개 ETF 중 대회 규정에 의해 매매 가능한 종목은 후원 10개 운용사가 발행하고 유동성(ADV $\ge$ 1억 원)을 충족하는 종목군으로 한정됩니다. 반면 과거 데이터 백테스트 시 현재 시점의 후원사 목록을 과거에 소급 적용하면 생존 편향이 발생하며, 반대로 전 종목을 허용하면 실전에서 매수할 수 없는 유령 알파가 발생합니다.

* **선택 근거 (Rationale)**:
  * 팩터의 통계적 유의성은 시장 전체(`structural`)에서 확인하여 과적합을 방지합니다.
  * 최종 전략 승격 및 데일리 주문 산출은 실제 모의투자 계좌에서 주문 가능한 종목(`deployment`)으로만 한정하여 실전 주문 집행 가능성을 안정적으로 확보합니다.

* **엔지니어링 트레이드오프 (Trade-offs)**:
  * 두 유니버스 패널을 각각 빌드하고 검증해야 하므로 파이프라인 관리 복잡도가 다소 증가합니다.

---

### ADR-04: Post-Crash Anchor Sleeve for Tournament Regime Adaptation

* **의사결정 (Decision)**:
  기존 P27 챔피언 전략(`sticky.mom60_raw`)에 지수 급락 후 반등 구간 전용 앵커 슬리브를 결합한 **`sticky.mom60_post_crash_anchor`를 최종 챔피언 전략으로 승격**합니다.

* **배경 및 맥락 (Context)**:
  P27은 60일 모멘텀 기반으로 우수한 꼬리 수익률을 보였으나, 코스피 지수가 급락한 직후 단기 급반등하는 구간에서는 `mom60 < 0` 조건으로 인해 100% 현금으로 철수하여 반등 기회를 완전히 방기하는 구조적 한계가 있었습니다.

* **선택 근거 (Rationale)**:
  * 평시(LOTTERY_ON)에는 60일 모멘텀 랭킹을 유지하되, 급락 후 반등 국면(CRASH_REBOUND) 진입 시 대표 지수 레버리지(코스닥150레버리지, KODEX 레버리지)를 20일 모멘텀으로 앵커링하고 20일 드로우다운 15% 손절 가드로 방어합니다.
  * 실측 결과 2,097개 롤링 윈도우에서 $P(R>30\%)$가 6.78% $\to$ **7.77%**로 +0.99%p 향상되었으며, CVaR(5%)도 -23.16% $\to$ **-21.56%**로 개선되었습니다.

* **엔지니어링 트레이드오프 (Trade-offs)**:
  * 반등 실패 후 2차 급락 시 15% 손절 라인까지의 드로우다운을 감수해야 합니다.

---

### ADR-05: Parquet / Polars Columnar Storage vs Relational Database

* **의사결정 (Decision)**:
  데이터 저장소로 RDBMS(PostgreSQL) 대신, **Bronze 원본 gzip JSON (`.json.gz`)과 Silver/Gold In-Process Parquet + Polars 구조**를 채택합니다.

* **배경 및 맥락 (Context)**:
  일별 1,100여 개 ETF의 8년치 시계열은 수백만 행 수준으로, 빅데이터 분산 클러스터가 필요한 규모는 아니지만 일별 단면 랭킹 및 롤링 백테스트 시 높은 I/O 처리량과 빠른 벡터화 연산이 요구됩니다.

* **선택 근거 (Rationale)**:
  * 외부 DB 데몬 설치나 네트워크 설정 없이 `git clone` 및 `uv sync`만으로 단일 머신에서 즉시 재현 가능합니다.
  * Polars의 멀티스레드 컬럼너 엔진을 통해 수백만 행의 단면 랭킹과 모멘텀 연산을 수 초 이내에 완료합니다.
  * Zero-copy Arrow 메모리 구조를 통해 머신러닝 파이프라인과 직접 연동됩니다.

* **엔지니어링 트레이드오프 (Trade-offs)**:
  * 행 단위의 실시간 트랜잭션 업데이트(OLTP)가 어려우므로, 일별 배치 시 증분 Append 및 파티션 재구축 방식을 사용합니다.

---

### ADR-06: Purged Walk-Forward GBDT Capacity Bound vs Deep Learning

* **의사결정 (Decision)**:
  머신러닝 적용 범위를 **단면 랭킹 전용 얕은 GBDT (LightGBM Ranker)**로 한정하고, 엄격한 용량 상한(`max_depth=4`, `num_leaves=8`, `min_data_in_leaf=100`)과 Purged Walk-Forward CV를 의무화하며, 딥러닝/강화학습은 배제합니다.

* **배경 및 맥락 (Context)**:
  8년간의 일별 ETF 데이터는 총 행 수로는 수십만 건에 달하지만, 36일 전방 수익률 라벨의 시계열 중첩과 국내 ETF 간의 높은 단면 상관(지수 복제 ETF 다수)으로 인해 **유효 독립 표본 수는 약 2,400개 수준**에 불과합니다.

* **선택 근거 (Rationale)**:
  * 2,400개의 유효 표본으로 고차원 신경망을 학습시키면 필연적으로 시장 노이즈를 암기하여 과적합됩니다.
  * 절대 수익률 예측 대신 단면 내 상대 순위를 학습하는 LambdaRank가 종목 선택력 분리에 가장 적합합니다.

* **엔지니어링 트레이드오프 (Trade-offs)**:
  * 비선형 복합 패턴이나 텍스트 등 비정형 데이터의 직접 결합이 제한됩니다.

---

### ADR-07: Credential Management Pragmatism (Deprecation of SOPS/Age)

> [!NOTE]
> **Status: Superseded**
> 초기 프로토타입 단계에서 검토했던 SOPS + Age 비대칭 인메모리 복호화 방식은 실보안 이득 대비 툴체인 운영 오버헤드가 과도하여 공식 폐기되었습니다. 현재는 `.gitignore` 기반의 비추적 `.env` 파일과 `pydantic-settings` 표준 환경변수 주입 체계를 유지합니다.

* **선택 근거 (Rationale)**:
  * 단일 운영자 모의투자 시스템 환경에서 외부 바이너리 의존성(`sops`)과 개인키 배포 절차는 유지보수 비용을 불필요하게 가중시킵니다.
  * 저장소에 비밀키가 유출되지 않는 표준 `.env` 관리만으로 보안 요구사항을 충분히 충족할 수 있어 실용주의적 관점에서 간소화했습니다.

---

## 3. Deployment Operations

* **Offsite Backup**: no offsite backup by design (short-lived deployment, ~8 weeks); state lives only on or-vps (Drive backup removed 2026-09-23).
