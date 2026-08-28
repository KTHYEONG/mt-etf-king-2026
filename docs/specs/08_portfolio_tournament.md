# 08. Portfolio Policy & Tournament Overlay — concentration · exit/re-entry · aggression · live decision

**선행**: [06_research_harness](06_research_harness.md), [07_leadership_engine](07_leadership_engine.md)
**상위**: [00_architecture.md](00_architecture.md)
**상태**: **Blueprint only — contract 유예** (§7 참조)

---

## 1. Diagnosis

### 1.1 Signal ≠ Portfolio

점수 `A=95, B=91, C=72` 가 자동으로 특정 비중을 함의하지 않는다. 그런데 이 계층에 **대회 우승 확률의 상당 부분**이 걸려 있다 — 같은 알파라도 Top1 100% 와 Top3 균등은 36세션 수익률 분포가 완전히 다르다.

### 1.2 집중도는 확신도의 함수여야 한다

$$
\texttt{conf}(t) = s_{(1)}(t) - s_{(2)}(t)
$$

`95 / 62 / 58` 은 leadership 이 명확하고, `82 / 81 / 80` 은 사실상 무차별이다. 후자에서 Top1 에 100% 를 넣는 것은 알파가 아니라 **랭킹 잡음에 베팅**하는 것이다.

**INV-08-1 (단조성)**: `conf` 가 증가하면 최상위 비중 $w_{(1)}$ 은 감소하지 않는다. 파라미터를 바꿔도 유지되어야 하는 구조적 성질이며 property test 대상이다.

매핑은 계단이 아니라 연속이어야 한다 — 계단이면 임계 근처에서 매일 리밸런싱이 발생한다.

$$
w_{(1)} = w_{\min} + (w_{\max} - w_{\min}) \cdot \sigma\!\left(\frac{\texttt{conf} - c_0}{\tau}\right)
$$

### 1.3 집중도의 진짜 제약은 유동성이다

§2.6 실측: $\phi = 1\%$ 일 때 거래 가능 종목 **26개**. Top1 100% 집중은 그 1종목이 ADV 1,000억 이상이어야 한다는 뜻이다.

$$
w_{(1)} \le \frac{\phi \cdot \text{ADV}_{20}(i)}{C}
$$

**INV-08-2**: sizing 은 확신도가 아무리 높아도 이 상한을 넘지 않는다. 유동성 상한이 목표 비중보다 낮으면 잔여는 차순위 또는 현금으로 간다.

### 1.4 Exit / Re-entry — 대회의 승부처

2025 우승자가 손절보다 **재진입 기준**을 강조했다는 점(next.md §50)은 구조적으로 타당하다. 36세션은 짧고, 한 번 이탈한 뒤 돌아오지 못하면 남은 기간에 만회가 불가능하다.

07 의 테마 state machine 이 이미 재진입 경로(`BREAKDOWN → RECOVERY → LEADING`)를 정의한다. 08 은 이를 **포지션 레벨**로 옮긴다.

| 포지션 상태 | 진입 조건 | 행동 |
| --- | --- | --- |
| `HOLD` | 테마 `LEADING` 유지 | 유지 |
| `TRIM` | 테마 `OVERHEATED` | 비중 축소 (전량 청산 아님) |
| `EXIT` | 테마 `BREAKDOWN` 또는 개별 손절 | 청산 |
| `WATCH` | `EXIT` 이후 | 재진입 후보로 추적 |
| `RE_ENTER` | `WATCH` 상태에서 테마가 `RECOVERY → LEADING` | 재진입 |

**INV-08-3 (재진입 쿨다운)**: `EXIT → RE_ENTER` 는 최소 $n_{\text{cool}}$ 세션이 지나야 가능하다. 없으면 임계 근처에서 청산/재매수를 반복하며 비용만 발생한다.

**INV-08-4 (경로 의존성 선언)**: 이 정책은 경로 의존적이다. 따라서 06 의 $O(T)$ 창 수익률 단축이 성립하지 않는다 → `run_rolling(path_dependent=True)` 로만 평가한다. 이 사실을 코드가 아니라 **모델 메타데이터로 선언**하고, 시뮬레이터가 잘못된 모드를 쓰면 예외를 던진다.

### 1.5 Tournament Aggression Overlay — 알파와 분리

**전략과 대회 정책은 다른 계층이다**(next.md §52). 이유는 명확하다 — 알파는 "무엇이 오를 것인가"를 답하고, aggression 은 "지금 내 순위에서 얼마나 위험을 져야 상금 기대값이 최대인가"를 답한다.

$n$ 세션 남았고 선두와의 격차가 $\Delta$ 일 때, 따라잡으려면 필요한 초과수익:

$$
\mu_{\text{req}} = \frac{\Delta}{n}\ \text{(세션당)}
$$

정규 근사에서 추월 확률은

$$
P(\text{overtake}) \approx \Phi\!\left(\frac{\mu_{\text{excess}} \cdot n - \Delta}{\sigma_{\text{excess}} \sqrt{n}}\right)
$$

여기서 나오는 **비대칭 구조**가 핵심이다.

- **뒤처져 있을 때** ($\Delta > 0$): 분자가 음수이므로, $\sigma$ 를 **키우면** 확률이 올라간다. 변동성이 유일한 희망이다.
- **앞서 있을 때** ($\Delta < 0$): 분자가 양수이므로, $\sigma$ 를 **줄이면** 확률이 올라간다. 변동성은 순수한 위험이다.
- **$n$ 이 줄수록** 두 효과 모두 강해진다 ($\sqrt{n}$ 분모).

**INV-08-5 (부호 규칙)**: `risk_multiplier` 는 $\Delta$ 에 대해 단조증가, 그리고 $\Delta > 0$ 일 때 $n$ 에 대해 단조감소, $\Delta < 0$ 일 때 $n$ 에 대해 단조증가여야 한다. 이 부호 조건은 property test 로 검증한다. 임계값이 아니라 **부호**가 계약이다.

| 상황 | 처방 |
| --- | --- |
| 선두 + 종반 | risk ↓ (분산 최소화) |
| 중위 + 초반 | normal (알파에 맡김) |
| 중위 + 종반 | risk ↑ |
| 큰 격차 + 종반 | extreme tail (레버리지 허용 시) |

**INV-08-6 (마지막에 붙인다)**: 알파가 검증되기 전에 overlay 를 만들면 잡음만 키운다. 06 게이트 통과 후에만 활성화한다. 그리고 **기본값은 비활성**이다.

### 1.6 경쟁자 Monte Carlo — 과신 금지

2025 대회 참가자 수는 약 1,000명 수준으로 보도되었고, 우리는 수익률 분포 $F$ 를 **모른다**. 따라서 이 모듈의 출력은 "우승 확률 4.72%" 같은 정밀 수치가 **아니다**.

세 시나리오의 **stress test** 로만 사용한다.

| 시나리오 | 설정 |
| --- | --- |
| `aggressive` | 경쟁자 상위 tail 두꺼움 (2025년 5주차 선두 +72% 수준을 상위 0.1% 앵커로) |
| `normal` | 2025년 최종 우승 +47.8% 를 상위 0.1% 앵커로 |
| `weak` | 상위 0.1% 를 +30% 수준으로 |

**INV-08-7 (보고 형식 강제)**: 출력은 단일 확률이 아니라 **세 시나리오의 구간**으로만 표기한다. 단일 스칼라 우승 확률을 반환하는 API 를 만들지 않는다 — 만들면 반드시 누군가(사람이든 LLM이든) 그것을 진실로 취급한다.

### 1.7 Live Decision — 매일 볼 화면

```
==================================================
ETF TOURNAMENT ALPHA          2026-10-07 (D+12/36)
==================================================
MARKET REGIME     RISK_ON            score 0.82
UNIVERSE          38 / 1163   drop: liq 1089, hist 24, elig 12

THEME LEADERS
1. SEMICONDUCTOR   93.2   LEADING     breadth 0.84
2. ROBOTICS        87.3   EMERGING    breadth 0.71
3. NUCLEAR         76.8   LEADING     breadth 0.66
4. DEFENSE         66.1   OVERHEATED  breadth 0.48

CHANGES           ROBOTICS  5 -> 2      DEFENSE  2 -> 4

PORTFOLIO         conf 0.31  (moderate)
  091160 KODEX 반도체        55%   HOLD    ADV cap 62%
  0142D0 TIGER AI데이터센터  25%   RE_ENTER
  CASH                       20%

WHY
  regime RISK_ON: kospi>ma20, breadth 0.63, rv20 0.011
  SEMICONDUCTOR: rs 0.97, accel +0.21, flow +1.2%/5d
  cash 20%: liquidity cap on rank-1, conf below full-concentration

TOURNAMENT        rank n/a   aggression NORMAL (overlay off)
==================================================
```

**INV-08-8 (근거 필수)**: 모든 포지션에는 기계 생성 근거가 붙는다. 근거를 만들 수 없는 결정은 출력하지 않는다. 36세션 동안 매일 이걸 보고 사람이 판단해야 하므로, 설명 불가능한 추천은 실용적 가치가 없다.

---

## 2. Architecture

```
 SectorLeadershipModel (07)  또는  baseline (06)
              │  scores
              ▼
 portfolio/selection.py   ClusterAwareSelection  (L1 유일 · L2 상한 m)
              ▼
 portfolio/sizing.py      confidence_weights  (INV-08-1)
              ▼
 portfolio/constraints.py liquidity cap (INV-08-2) · leverage gate · normalize (INV-7)
              ▼
 portfolio/state.py       PositionState machine  HOLD/TRIM/EXIT/WATCH/RE_ENTER
              ▼
 tournament/policy.py     AggressionPolicy  (INV-08-5, 기본 비활성)
              ▼
 tournament/montecarlo.py CompetitorField  (3 시나리오 구간 보고)
              ▼
 reporting/dashboard.py   DailyDecision + rationale
 cli:  mt-etf decide --date 2026-10-07
```

### 2.1 레버리지 게이트

레버리지 ETF 편입은 **세 조건이 모두 참**일 때만 허용한다.

1. `rules.leverage_allowed is True` — `UNKNOWN` 이면 허용하지 않는다 (R-2, INV-8)
2. `regime in {STRONG_RISK_ON, RISK_ON}`
3. 해당 종목의 `Confidence` 가 `LOW` 가 아니다 (INV-04-6)

세 조건 모두 fail-closed 다. 그리고 `UNKNOWN` 상태에서는 연구 단계에서 레버리지 허용/비허용 **양쪽 분포를 모두 산출**한다.

---

## 3. Acceptance Gates

| 게이트 | 기준 |
| --- | --- |
| B-1 | confidence sizing 이 고정 Top-k 대비 `P(R>30%)` 비열위이면서 worst-5% 개선 |
| B-2 | 재진입 정책이 있는 쪽의 36세션 분포가, 청산 후 미복귀 정책보다 median 및 `P(R>30%)` 모두 우위 |
| B-3 | 쿨다운 제거 시 턴오버가 유의하게 증가함을 보여 INV-08-3 의 필요성 입증 |
| B-4 | aggression overlay on/off 비교에서 on 이 `E[Prize]` 대리지표를 개선 (3 시나리오 모두) |
| B-5 | 전체 파이프라인이 2025 replay 에서 35세션 완주하고 일별 근거 로그를 남김 |

B-4 가 실패하면 **overlay 를 끄고 대회에 나간다.** 기본값이 비활성인 이유다.

---

## 4. Assumptions

- **A-1**: 리밸런싱 주기는 일간(매 세션 재평가). 단, 목표 비중 변화가 `min_rebalance_delta` 미만이면 주문하지 않는다(밴드 방식, quant.md §2).
- **A-2**: 대회 중 실제 순위·선두 수익률은 대회 랭킹 페이지에서 **수동 입력**한다. 자동 스크래핑은 만들지 않는다(범위 밖, 그리고 실패 시 대회 중 장애가 된다).
- **A-3**: 현금 보유는 허용 가정. R-8 확정 시 조정.

---

## 5. 예상 심볼 (contract 확정 시)

```
src/portfolio/selection.py   ClusterAwareSelection · select_positions
src/portfolio/sizing.py      confidence_weights · apply_liquidity_cap
src/portfolio/constraints.py leverage_gate · rebalance_band
src/portfolio/state.py       PositionState · PositionTracker · transition
src/tournament/policy.py     AggressionInput · AggressionPolicy · risk_multiplier
src/tournament/montecarlo.py CompetitorScenario · CompetitorField · rank_interval
src/reporting/dashboard.py   DailyDecision · render_dashboard · build_rationale
configs/strategies.yaml      sizing · state · aggression 파라미터
```

---

## 6. 대회 운영 절차 (2026-09-21 ~ 11-13)

```
매 거래일 08:30 (KST)
  1. mt-etf ingest --dataset etf_daily --start <T-1> --end <T-1>
  2. mt-etf normalize --mode incremental
  3. mt-etf features --end <T-1>
  4. mt-etf decide --date <T-1>          # 체결은 T 시가
  5. 대시보드 확인 → HTS 주문
  6. (주 1회) 랭킹 페이지에서 순위/선두 수익률 입력 → overlay 갱신
```

전일 종가 데이터가 당일 장 시작 전에 조회 가능함을 실측 확인했다(2026-08-28 09:57 KST 에 2026-08-27 데이터 정상 반환). 따라서 **당일 시가 체결이 성립**한다.

**운영 리스크**: KRX API 장애 시 결정 불가. 대응은 전일 포트폴리오 유지(fail-safe 기본값)이며, 데이터 없이 추정하지 않는다.

---

## 7. Contract 유예 사유

- sizing 곡선 파라미터($w_{\min}, w_{\max}, c_0, \tau$)는 06 이 산출한 **실제 confidence 분포**에서 유도해야 한다. 분포를 보기 전에 정하면 임의의 숫자다.
- 상태 전이 임계와 쿨다운 길이는 07 의 테마 상태 통계에 의존한다.
- aggression 곡선은 06 의 36세션 수익률 분포 $\sigma$ 추정치가 있어야 정규 근사를 쓸 수 있다.

**해제 조건**: 06 W3 게이트 통과 **and** 07 acceptance gate A-1~A-5 판정 완료. 그 시점에 `/spec 08_portfolio_tournament` 재실행.

07 이 기각되면 08 은 baseline(B1~B3) 위에 sizing·state·overlay 만 얹는 축소 버전으로 재작성한다 — 이 경우에도 §1.2~1.5 의 불변식은 그대로 유효하다.
