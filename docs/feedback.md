# MT ETF King 2026 — Next Development Directive

## 0. 목적

이 문서는 `mt-etf-king-2026` 프로젝트의 **다음 개발 단계에 대한 실행 지시서**다.

현재 단계에서 목표는 새로운 전략 아이디어를 계속 추가하는 것이 아니다.

최근 feasibility audit 결과를 기준으로 다음 4개 문제를 순차적으로 해결한다.

1. **측정/평가 신뢰성 확정**
2. **대회 시작 시점에 사용할 causal regime gate 구축**
3. **P27의 selection / entry timing 병목 개선**
4. **우승권 수익 도달 후 giveback 방어**

새로운 P39/P40 전략을 무작정 만드는 방식은 금지한다.

---

# 1. 현재까지 확정된 핵심 데이터

## 1.1 Market opportunity

2018-01-02 ~ 2026-08-27, 36-session rolling window 기준:

| Metric | Value |
|---|---:|
| 전체 분석 window | 2,086 |
| effective independent blocks | 약 57 |
| Structural Oracle P>50 | 24.64% |
| Executable Oracle P>45 | 20.28% |
| Executable Oracle P>50 | 17.69% |
| Executable Oracle P>60 | 11.98% |

핵심 해석:

> 36거래일 동안 +45~50% 우승권 수익을 낼 수 있는 ETF 시장 기회 자체는 충분히 존재한다.

따라서 프로젝트의 핵심 실패 원인은 **market opportunity 부족이 아니다.**

---

## 1.2 P27 성능

현재 champion:

`sticky.mom60_raw` / P27

전체 기간 기준:

| Metric | Value |
|---|---:|
| P>30 | 8.77% |
| P>40 | 6.52% |
| P>45 | 5.37% |
| P>50 | 5.03% |
| P>60 | 4.03% |
| Capture@45 | 24.35% |
| Capture@50 | 25.75% |
| Ruin < -25% | 2.44% |
| q95 | 약 +50.17% |
| q99 | 약 +137.15% |

즉 실행 가능한 +50% 기회가 있었을 때 P27은 약 4번 중 1번을 실제 +50% 수익으로 연결한다.

그러나 이 수치는 전체 평균만으로 평가하면 안 된다.

---

# 2. P27의 실제 정체

P27의 +50% window는 총 105개다.

그 중:

- 2018~2024: 4개
- 2025: 48개
- 2026 YTD: 53개

즉:

`101 / 105 = 96.19%`

가 2025~2026에 집중되어 있다.

연도별 주요 수치:

| Period | P27 P>50 |
|---|---:|
| 2018~2024 | 0.23% |
| 2025 | 19.83% |
| 2026 YTD | 43.09% |
| Full | 5.03% |

따라서 P27을 다음과 같이 정의한다.

> **P27은 general-purpose champion strategy가 아니다.**
>
> **P27은 strong directional / high-volatility upside regime에서 championship tail을 노리는 aggressive specialist다.**

이 정의를 이후 architecture에 반영한다.

---

# 3. Regime 분석 결과

현재 audit 기준:

| Regime | 비중 | Exec Oracle P>50 | P27 P>50 |
|---|---:|---:|---:|
| Strong Risk-On | 19.0% | 35.4% | 13.4% |
| Weak Risk-On | 30.2% | 9.7% | 0.6% |
| Sideways | 31.3% | 6.1% | 0.0% |
| Risk-Off | 11.1% | 10.8% | 0.0% |
| High-Vol Reversal | 8.4% | 58.5% | 27.3% |

P27의 +50% tail은 사실상:

- `Strong Risk-On`
- `High-Vol Reversal`

두 regime에서 발생한다.

Weak Risk-On / Sideways / Risk-Off에서는 championship tail이 거의 없다.

따라서 앞으로의 핵심 질문은:

> "P27이 좋은 전략인가?"

가 아니라:

> **"2026-09-21 시점의 시장이 P27을 활성화할 regime인가?"**

다.

---

# 4. P38 평가

P38 adaptive specialists의 현재 결과:

| Metric | P27 | P38 |
|---|---:|---:|
| P>50 | 5.03% | 0.96% |
| Ruin < -25% | 2.44% | 0.34% |

P38은 위험을 크게 줄였으나 championship right-tail을 지나치게 제거한다.

현재 P38 router selection count:

- long_broad: 16,690
- long_theme: 17,864
- inverse: 11,389
- defensive: 29,225

defensive 사용 비중이 매우 높다.

따라서:

> **P38을 P27 replacement champion으로 승격하지 않는다.**

대신 이후 architecture에서는:

- P27 = aggressive specialist
- P38 또는 일부 defensive component = unfavorable regime fallback

으로 제한적으로 재사용할 수 있다.

---

# 5. Oracle Gap — 실제 개발 병목

현재 분석에서 executable oracle과 P27 사이 gap:

| Rank | Loss Component | Mean | Median | q90 |
|---:|---|---:|---:|---:|
| 1 | Selection & Entry Timing | 35.00% | 31.78% | 59.08% |
| 2 | Giveback | 10.39% | 2.96% | 31.74% |
| 3 | Capacity / Fillability | 6.98% | 0.10% | 22.60% |
| 4 | Universe Restriction | 0.00% | 0.00% | 0.00% |

따라서 다음 alpha research는 반드시 다음 2개 문제에 집중한다.

### 병목 A — Selection / Entry

현재 `mom_60` 기반 선택은 우승급 ETF의 상승 초기에 반응이 늦다.

개발 질문:

> "이미 가장 많이 오른 ETF가 무엇인가?"

가 아니라:

> **"향후 championship move로 이어질 초기 가속/확산 신호를 causal하게 더 빨리 감지할 수 있는가?"**

### 병목 B — Giveback

일부 window는 중간에 +40~70%까지 도달했으나 종료 시점까지 수익을 크게 반납한다.

개발 질문:

> **"우승권 수익을 만든 이후 expected upside보다 wealth loss risk가 커지는 시점을 causal하게 감지할 수 있는가?"**

---

# 6. 2025 결과 사용 규칙

2025 exact contest replay:

| Start offset | P27 Return |
|---:|---:|
| -10 sessions | +84.31% |
| -5 | +61.55% |
| Exact | +53.37% |
| +5 | +22.65% |
| +10 | +13.32% |

실제 2025 우승 수익률: 약 +47.82%

하지만 이 결과는:

`HEAVILY_REUSED`

로 분류한다.

이유:

- P20~P38 연구 중 반복적으로 2025 구간 참조
- 124회 이상의 run
- 64개 research task
- 25종 이상의 strategy family
- 100종 이상의 parameter/variant 탐색

따라서:

> **2025 +53.37%는 OOS validation evidence로 사용 금지.**

허용되는 해석:

> P27 구조가 우승급 return path를 실제 historical market에서 만들 수 있었다는 existence proof.

금지되는 해석:

> P27이 실제 대회에서 우승할 확률이 높다는 증거.

---

# 7. 개발 우선순위

이후 개발은 아래 순서를 반드시 따른다.

---

# Phase 0 — Measurement Hardening

## 목적

alpha 연구 전에 현재 평가 framework 자체를 신뢰 가능한 상태로 만든다.

## TASK M0-1 — 2086 / 2088 window mismatch 제거

현재:

- feasibility audit: 2,086 windows
- P38 promotion artifact: 2,088 windows

차이가 존재한다.

### 해야 할 일

다음 source별 eligible window generation path를 추적한다.

- feasibility audit
- P27 rolling evaluation
- P38 promotion
- B1 evaluation

그리고 각각:

- raw session count
- feature warmup
- 36-session horizon
- next-open requirement
- terminal liquidation session requirement
- missing session filtering
- PIT universe eligibility

를 비교한다.

### 산출물

`docs/research/window_population_reconciliation.md`

필수 표:

| Pipeline | Raw Sessions | Eligible Windows | Excluded Windows | Exact Reason |
|---|---:|---:|---:|---|

### 완료 조건

모든 전략 비교에 같은 36-session decision population을 사용할 수 있거나,
다른 population을 사용하는 경우 그 이유가 명시적으로 정당화되어 있어야 한다.

silent mismatch 금지.

---

## TASK M0-2 — Feasibility audit 완전 재현 가능화

현재 다음 artifact가 핵심 분석에 사용되었다.

`scratch/final_merged_windows.parquet`

하지만 연구 결과를 재생성하는 canonical committed pipeline이 명확해야 한다.

### 구현 요구

다음 중 하나를 생성한다.

`tools/research/championship_feasibility_audit.py`

또는

`src/research/championship_feasibility.py`

반드시 한 command로:

1. rolling windows 생성
2. oracle 계산
3. P27/P38/B1 결과 merge
4. yearly metrics
5. regime metrics
6. capture funnel
7. oracle gap
8. risk metrics
9. CSV 생성
10. JSON 생성
11. MD report 생성

이 재현되어야 한다.

### 산출물

- `docs/research/2026_championship_feasibility_audit.md`
- `docs/research/2026_championship_feasibility_metrics.json`
- `docs/research/2026_championship_feasibility_tables.csv`

### 완료 조건

동일 raw data + config에서 deterministic output hash 또는 주요 metrics parity가 확인되어야 한다.

---

## TASK M0-3 — B1 execution compliance 정리

현재 B1 full-history run 중:

- effective_gross_max = 2.0
- gross_violation_count = 1428

인 기록이 존재한다.

### 해야 할 일

B1이 P27/P38과 동일한 execution policy로 비교 가능한지 검증한다.

### 선택지

A. execution-compliant B1 재실행

또는

B. 기존 B1을:

`NON_COMPLIANT_REFERENCE`

로 명시하고 causal benchmark 비교에서 제외

### 금지

gross invariant가 다른 B1 metric을 동일 조건 benchmark처럼 표시하지 않는다.

---

## TASK M0-4 — tournament rule state 명시화

현재 config에:

- leverage_allowed: unknown
- inverse_allowed: unknown

이 남아 있다.

### 해야 할 일

공개 대회 규정상 자율형 leverage/inverse 허용 여부를 config에 명시적으로 반영한다.

단:

> 공개 규정상의 허용 여부와 실제 HTS tradable manifest는 별개다.

따라서 exact deployment universe는 manifest가 제공되기 전까지 provisional 상태로 둔다.

---

## TASK M0-5 — exact deployment universe hook 준비

현재:

`configs/universe.yaml -> manifest: null`

이다.

### 해야 할 일

대회 HTS 종목 리스트가 제공되는 즉시:

`configs/universe_manifest.yaml`

등으로 연결할 수 있도록 loader / validation / fail-closed path를 준비한다.

### 완료 조건

manifest가 설정된 deployment mode에서는:

- manifest 외 ETF 진입 금지
- missing ticker fail closed
- issuer mismatch detection
- leverage/inverse classification validation
- tradability validation

을 보장한다.

---

# Phase 1 — Causal Regime Gate

## 목적

대회 시작 시점에 P27을 공격적으로 사용할 조건을 판단하는 causal gate를 만든다.

이 단계가 현재 가장 우선순위가 높은 전략적 개발이다.

## TASK R1-1 — Regime label causality 검증

현재 regime 정의에 사용된 feature:

- KOSPI mom20
- KOSPI mom60
- realised vol20
- drawdown60
- 필요 시 KOSDAQ equivalents

모든 feature에 대해:

`feature_timestamp <= decision_timestamp`

를 증명한다.

### 금지

window 종료 이후 가격을 사용한 ex-post regime classification을 deployment decision gate로 재사용하는 것.

### 산출물

`docs/research/regime_feature_causality_audit.md`

---

## TASK R1-2 — Pre-start regime diagnostic panel

대회 시작일 2026-09-21 직전 사용 가능한 데이터만으로 diagnostic panel을 생성한다.

후보 feature:

### Broad market

- KOSPI mom20
- KOSPI mom60
- KOSPI mom120
- KOSDAQ mom20
- KOSDAQ mom60
- drawdown20/60
- realised vol20

### Breadth

- sponsor ETF positive mom20 ratio
- sponsor ETF positive mom60 ratio
- +2x ETF positive mom20 ratio
- +2x ETF positive mom60 ratio
- cross-sectional return dispersion
- sector/theme breadth

### Acceleration

- mom20 - mom60 normalized
- 5/20-day acceleration
- volume expansion
- high-breakout breadth

### Trend persistence

- positive days ratio
- market index above moving average
- sector leadership persistence

---

## TASK R1-3 — Fixed buckets only

threshold mining을 막기 위해 무제한 threshold search 금지.

허용:

- fixed economic thresholds
- tercile / quartile
- predeclared quantile bins

금지:

> P27 P>50가 최대가 되는 threshold를 전체 history에서 brute-force search

---

## TASK R1-4 — Regime conditional table 생성

각 regime / bucket에 대해:

- n_windows
- n_effective
- P27 P>30
- P27 P>40
- P27 P>45
- P27 P>50
- P27 ruin<-25
- executable oracle P>50
- capture@50

를 계산한다.

### 핵심 출력

`docs/research/p27_regime_activation_table.csv`

---

## TASK R1-5 — Gate candidate 정의

최종적으로 최소 3개 state를 정의한다.

### `AGGRESSIVE_ON`

P27 aggressive deployment를 허용할 정도로 historical evidence가 있는 상태.

### `UNCERTAIN`

P27 full exposure를 정당화하기 어려운 상태.

### `AGGRESSIVE_OFF`

P27 championship tail evidence가 거의 없는 상태.

중요:

이 단계에서는 아직 gate를 production에 바로 연결하지 않는다.

먼저 research-only rule로 평가한다.

---

# Phase 2 — Selection / Early Entry Research

## 목적

Oracle gap 1위인 `Selection & Entry Timing Loss`를 줄인다.

기존 P27을 대체하는 새로운 대규모 전략을 만드는 것이 아니다.

P27의 ranking/entry layer만 연구한다.

## TASK S2-1 — P27 winning/losing opportunity dataset

모든 executable oracle > +30/+40/+50 window에 대해 다음을 저장한다.

- window start
- oracle ticker
- oracle return
- P27 selected ticker
- P27 selected return
- entry delay
- rank of oracle ticker at t0
- rank at t+1 ... t+10
- mom3/mom5/mom10/mom20/mom60
- volume expansion
- rv20
- cross-sectional percentile
- breadth context
- leverage class
- sector/theme

### 목적

P27이 놓친 championship move가:

- 너무 늦게 rank 상승
- early acceleration 미탐지
- wrong-theme persistence
- incumbent stickiness
- universe/fill issue

중 어디에 해당하는지 구분한다.

---

## TASK S2-2 — Detectability decomposition

Oracle opportunity를 다음으로 분류한다.

### A. Detectable-at-start

t0 causal feature만으로 이미 상위권 signal.

### B. Detectable-early

t+1 ~ t+5 안에 causal signal이 나타남.

### C. Late-detectable

t+6 이후에야 signal 발생.

### D. Fundamentally unpredictable

초기 causal ranking에서는 식별 근거가 거의 없음.

최종 목표:

> P27 개선으로 실제 줄일 수 있는 selection gap의 상한이 얼마인지 계산한다.

---

## TASK S2-3 — Minimal candidate signals

새 ML 모델 금지.

먼저 기존 feature에서만 비교한다.

예:

- mom20
- mom10
- mom5
- momentum acceleration
- volume expansion
- volatility-adjusted momentum
- cross-sectional breakout percentile
- breadth-confirmed momentum

### 원칙

하나씩 독립적으로 검증하고
P27 대비:

- entry lead time
- P>50
- capture@50
- ruin
- LOYO
- regime concentration

을 비교한다.

---

## TASK S2-4 — No aggregate-only promotion

다음은 promotion 근거로 금지:

- 전체 P>50가 조금 상승
- 2025 replay 상승
- q99 상승

반드시 같이 본다.

- 2018~2024
- 2025
- 2026
- LOYO
- strong risk-on
- high-vol reversal
- weak/sideways/risk-off
- ruin
- discordant windows

---

# Phase 3 — Giveback / Wealth Retention

## 목적

championship path를 만든 이후 terminal wealth를 지킨다.

새 entry alpha와 동시에 튜닝하지 않는다.

Selection 연구와 분리한다.

## TASK G3-1 — Giveback event dataset

P27 각 window에서:

- terminal return
- peak return
- peak session
- terminal giveback
- peak 이후 1/3/5/10-day returns
- remaining sessions
- current regime
- holding ticker
- holding mom5/mom20
- market breadth
- volatility
- drawdown from strategy peak

를 저장한다.

---

## TASK G3-2 — Wealth milestone 분석

다음 milestone 이후 terminal outcome을 분석한다.

- +20%
- +30%
- +40%
- +45%
- +50%
- +60%

각 milestone 도달 후:

- 추가 상승 확률
- terminal > milestone 유지 확률
- -5/-10/-15/-20% giveback 확률
- expected terminal return
- ruin probability

를 계산한다.

---

## TASK G3-3 — Exit family 연구

무작정 trailing stop grid search 금지.

최소한 서로 다른 경제적 가설로 분리한다.

### A. Peak drawdown lock

strategy wealth peak 대비 일정 drawdown.

### B. Momentum deterioration

보유 ETF 단기 momentum 붕괴.

### C. Market regime deterioration

breadth / index trend 붕괴.

### D. Tournament threshold lock

+40/+50 등 milestone 확보 이후 risk budget 축소.

각 family는 동일 평가 framework에서 비교한다.

---

# Phase 4 — P27 / Defensive Routing

## 목적

P38 전체를 champion으로 사용하지 않고,
regime fallback role만 검증한다.

## TASK D4-1 — P38 component attribution

P38 전체가 아니라 specialist별 기여도를 계산한다.

- long_broad
- long_theme
- inverse
- defensive

각 specialist가:

- 어떤 regime에서
- P27 대비
- return / ruin / capture

를 개선하는지 분석한다.

---

## TASK D4-2 — Minimal fallback test

다음 단순 architecture부터 평가한다.

```text
if AGGRESSIVE_ON:
    use P27
else:
    use defensive/fallback policy
```

처음부터 복잡한 router를 만들지 않는다.

비교:

1. always P27
2. always P38
3. causal gate + P27/fallback
4. baseline

---

# 8. 금지 사항

다음은 명시적으로 금지한다.

## 전략 탐색

- P39/P40 신규 전략 family 즉시 생성
- feature 20~50개 조합 brute-force
- 전체 2018~2026에 대한 threshold optimizer
- 2025 contest return을 objective로 직접 최적화

## 평가

- overlapping 2,086 windows를 독립 표본처럼 해석
- top2_rate를 실제 대회 top2 probability로 해석
- internal win_rate를 실제 competition win probability로 해석
- 2025 +53.37%를 OOS 성과로 사용
- artifact_integrity=false 상태의 promotion을 production-ready로 해석

## 코드

- research 결과 때문에 champion 자동 교체
- measurement bug와 alpha change를 한 commit에 혼합
- silent fallback
- missing manifest를 full deployment universe로 간주
- causality가 확인되지 않은 regime feature를 production gate로 사용

---

# 9. Commit 정책

각 phase는 분리 commit한다.

권장 예:

```text
research: reconcile championship rolling-window populations
research: make feasibility audit fully reproducible
fix: enforce comparator gross-exposure parity
research: validate causal regime features
research: build P27 activation diagnostics
research: decompose championship selection misses
research: quantify P27 giveback after wealth milestones
```

alpha 로직 변경과 measurement 변경은 같은 commit에 넣지 않는다.

---

# 10. 반드시 생성할 문서

최소 다음 문서를 생성한다.

```text
docs/research/window_population_reconciliation.md
docs/research/regime_feature_causality_audit.md
docs/research/p27_regime_activation_report.md
docs/research/p27_regime_activation_table.csv
docs/research/p27_selection_gap_analysis.md
docs/research/p27_giveback_analysis.md
docs/research/p38_component_attribution.md
```

machine-readable artifact:

```text
docs/research/p27_regime_activation_metrics.json
docs/research/p27_selection_gap_metrics.json
docs/research/p27_giveback_metrics.json
```

---

# 11. Phase별 Stop / Go 조건

## Phase 0 완료 조건

- 2086/2088 mismatch 설명 또는 제거
- audit deterministic reproduction 가능
- B1 comparison compliance 정리
- exact universe manifest hook 준비
- rule semantics 명확화

완료 전 alpha 연구 금지.

---

## Phase 1 GO 조건

causal regime features만으로 P27 championship tail의 조건부 차이를 설명할 수 있어야 한다.

예:

```text
AGGRESSIVE_ON:
P27 P>50 materially higher
capture@50 materially higher
ruin acceptable
```

이 증거가 없으면 복잡한 regime router 개발 금지.

---

## Phase 2 GO 조건

새 selection layer가 다음 중 최소 하나를 만족해야 한다.

- causal entry lead time 개선
- capture@50 개선
- strong-risk-on/high-vol P>50 개선

동시에:

- ruin 악화 제한
- LOYO concentration 악화 제한
- 2025-only improvement 금지

---

## Phase 3 GO 조건

giveback control이:

- q90 giveback 감소
- terminal P>45/P>50 유지 또는 개선
- championship tail 파괴 없음

을 만족해야 한다.

단순히 drawdown만 줄이고 P>50가 크게 감소하면 실패다.

---

# 12. 최종 architecture 방향

현재 목표 architecture:

```text
                    causal market diagnostics
                              |
                              v
                     Regime Activation Gate
                       /               \
                      /                 \
           AGGRESSIVE_ON             OFF/UNCERTAIN
                 |                        |
                 v                        v
          P27 aggressive            fallback policy
                 |
                 v
        early selection / entry
                 |
                 v
       championship tail capture
                 |
                 v
        wealth milestone monitor
                 |
                 v
          giveback protection
```

---

# 13. 최종 연구 질문

향후 모든 개발은 아래 3개 질문 중 하나에 직접 답해야 한다.

### Q1

> 현재 시장이 P27 championship tail이 historically 존재했던 causal regime인가?

### Q2

> 우승급 ETF move가 시작될 때 P27보다 더 빠르고 causal하게 포착할 수 있는가?

### Q3

> 이미 만든 +40~60% wealth를 tail upside를 크게 희생하지 않고 지킬 수 있는가?

이 세 질문에 직접 연결되지 않는 전략 개발은 우선순위에서 제외한다.

---

# 14. 최종 지시

현재 project status를 다음과 같이 해석한다.

- **목표 자체:** FEASIBLE
- **현재 범용 전략:** UNPROVEN
- **P27:** REGIME-DEPENDENT AGGRESSIVE SPECIALIST
- **P38:** CHAMPION REPLACEMENT로는 부적합
- **시장 opportunity:** 충분함
- **가장 큰 병목:** Selection / Entry Timing
- **두 번째 병목:** Giveback
- **핵심 next step:** Causal Regime Gate

따라서 앞으로의 개발 목표는:

> **"더 많은 전략을 만드는 것"이 아니라, P27이 작동할 때를 정확히 식별하고, 우승급 종목을 더 빨리 포착하며, 이미 만든 우승 수익을 지키는 시스템을 만드는 것"**

이다.

모든 새로운 변경은 반드시 데이터 기반 hypothesis → isolated measurement → causal validation → promotion gate 순서로 진행한다.
