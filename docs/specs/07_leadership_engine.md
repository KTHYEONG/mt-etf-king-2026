# 07. Sector Leadership Engine — cluster dedup · theme score · 6-state machine

**선행**: [06_research_harness](06_research_harness.md) + **07_preflight PASS** + B5≠B4 재측정
**상위**: [00_architecture.md](00_architecture.md)
**상태**: **Blueprint only — contract 유예** (§6 참조, preflight 전 해제 금지)

---

## 1. Diagnosis

### 1.1 왜 개별 ETF 랭킹만으로는 부족한가

실측: 1,163 ETF, **880개 distinct 기초지수, 89.4%가 singleton**. 그러나 유동성 필터 통과 종목은 26~121개.

이 구조에서 개별 ETF 를 그대로 랭킹하면 두 가지가 동시에 깨진다.

**(a) 중복 베팅.** 반도체 ETF 3개가 1~3위를 차지하면 "3종목 분산"이 아니라 **반도체 300% 집중**이다. 상위 지수별 ETF 수 실측: 코스피200 25개, S&P500 24개, NASDAQ100 16개, 코스피200선물 16개. 대형 테마일수록 상위권을 독식할 구조적 소지가 크다.

**(b) 신호 희석.** 같은 테마 안에서 ETF 간 미세한 수익률 차이(운용보수·추적오차·괴리율)가 랭킹을 흔든다. 이 차이는 **알파가 아니라 잡음**이다. 테마 레벨로 집계하면 잡음이 상쇄된다.

### 1.2 중복제거는 상관계수 추정 문제가 아니다

일부 설계안은 `underlying index / sector / holdings overlap / return correlation` 을 제안한다. 그러나 KRX 패널은 매일 `IDX_IND_NM` 을 **공식적으로, point-in-time 하게** 제공한다.

$$
\texttt{index\_key}(i) = \texttt{index\_key}(j) \;\Longrightarrow\; i,j \text{ 는 동일 베팅}
$$

이건 추정이 아니라 **정의**다. 상관계수 추정(표본오차 있음, lookback 선택 필요, 레짐 의존)보다 엄격하게 우월하다.

2단계 중복제거:

| 단계 | 키 | 성격 |
| --- | --- | --- |
| L1 | `index_key` | 결정론적. 동일 지수 → 반드시 1개만 |
| L2 | `theme` | 규칙기반. 동일 테마 → 최대 $m$ 개 |

### 1.3 클러스터 대표 선정 — 알파가 아닌 실행 품질로

같은 `index_key` 안에서 어떤 ETF를 살지는 **알파 문제가 아니라 실행 문제**다. 판정 기준(사전순 우선순위, 알파 점수 사용 금지):

1. **유동성**: ADV20 최대 (10억 집행 가능성이 1순위 — §2.6 실측상 이게 제일 희소하다)
2. **추적 충실도**: $|R_{ETF} - R_{IDX}|$ 의 20세션 평균 최소 (`underlying_index_close` 로 직접 계산 가능)
3. **괴리 안정성**: $|disparity|$ 의 20세션 median 최소 (실측 정상 범위 p50 34bp)
4. **규모**: `net_assets` 최대 (동률 시)

알파 점수로 대표를 뽑으면 안 된다 — 그건 같은 베팅 안에서 **잡음 최댓값**을 고르는 것이고, 정의상 과적합이다.

### 1.4 Theme Score — 가설이지 정답이 아니다 (preflight: rs/accel/breadth only)

$$
\text{SectorScore}(g,t) = \sum_k w_k \cdot z_k(g,t)
$$

| 성분 $z_k$ | 정의 | 기본 가중 | 근거 |
| --- | --- | --- | --- |
| `rs` | 테마 20세션 수익률의 테마 간 percentile | **>0** | 상대강도 |
| `accel` | $\texttt{rs}_5 - \texttt{rs}_{20}$ | **>0** | leadership 부상 감지 |
| `breadth` | cluster breadth (테마 내 ETF 중 MA20 상회 비율) | **>0** | 상승의 질 |
| `breakout` | 20/40세션 신고가 근접도 | **0 (B3 실패)** | 추세 확정 |
| `flow` | 5세션 누적 $\Delta L \cdot N$ / net_assets | **0 (V5 신뢰도)** | 실제 자금 유입 |

초기 가중치 `(0.30, 0.20, 0.15, 0.15, 0.10, 0.10)` 는 **hypothesis 이며 그대로 production 에 박지 않는다**(INV-11). preflight 권고: **rs/accel/breadth만>0**.

**검증 절차**: 각 성분을 **단독 알파 모델**로 06 harness 에 태워 $P(R>\theta)$ 곡선을 얻는다. 기여가 없는 성분은 가중치를 주는 게 아니라 **제거**한다.

### 1.5 6-State Machine — 점수보다 상태가 유용한 이유 (OVERHEATED = CROWDED+EXHAUSTING)

```
DISCOVERY → EMERGING → LEADING → OVERHEATED → BREAKDOWN → RECOVERY ─┐
     ▲                                                                │
     └────────────────────────────────────────────────────────────────┘
```
6-state 유지, `OVERHEATED`는 feedback CROWDED+EXHAUSTING 통합.

스칼라 점수는 "지금 강하다"만 말하지만, 상태는 **"어떻게 강해졌고 다음에 무엇을 해야 하는가"** 를 말한다. **재진입 기준**은 점수로 표현할 수 없다 — 같은 점수 60점이라도 `EMERGING` 의 60점과 `BREAKDOWN` 의 60점은 정반대 행동을 요구한다.

**전이 술어** (임계값은 전부 `configs/strategies.yaml`):

| 전이 | 조건 |
| --- | --- |
| `DISCOVERY → EMERGING` | $\texttt{accel} > a_{\text{in}}$ **and** $\texttt{breadth} > b_{\text{in}}$ |
| `EMERGING → LEADING` | $\texttt{rs} > r_{\text{in}}$ **and** $\texttt{breadth} > b_{\text{in}}$ **and** $\texttt{flow} > 0$ |
| `EMERGING → DISCOVERY` | $\texttt{accel} < a_{\text{out}}$ ($n_{\text{patience}}$ 세션 연속) |
| `LEADING → OVERHEATED` | $\texttt{ext} > e_{\text{in}}$ **or** ($\texttt{rs} > r_{\text{hi}}$ **and** $\Delta\texttt{breadth} < 0$) |
| `LEADING → BREAKDOWN` | $\texttt{dd} < -d_{\text{out}}$ **or** $\texttt{breadth} < b_{\text{out}}$ |
| `OVERHEATED → BREAKDOWN` | $\texttt{dd} < -d_{\text{out}}$ |
| `OVERHEATED → LEADING` | $\texttt{ext} < e_{\text{out}}$ **and** $\texttt{breadth} > b_{\text{in}}$ |
| `BREAKDOWN → RECOVERY` | $\texttt{accel} > a_{\text{in}}$ **and** $\texttt{dd} > -d_{\text{in}}$ |
| `RECOVERY → LEADING` | $\texttt{rs} > r_{\text{in}}$ **and** $\texttt{breadth} > b_{\text{in}}$ ← **재진입 지점** |
| `RECOVERY → BREAKDOWN` | $\texttt{dd} < -d_{\text{out}}$ |

여기서

$$
\texttt{ext}(g,t) = \frac{P_g(t) - \text{MA}_{20}(g,t)}{\text{ATR}_{20}(g,t)}, \qquad
\texttt{dd}(g,t) = \frac{P_g(t)}{\max_{s \le t,\, t-s < 20} P_g(s)} - 1
$$

**INV-07-1 (히스테리시스 필수)**: 진입/이탈 임계는 반드시 다르다 — $b_{\text{in}} > b_{\text{out}}$, $a_{\text{in}} > a_{\text{out}}$, $e_{\text{in}} > e_{\text{out}}$. 같으면 임계 근처에서 상태가 매 세션 진동하고 턴오버가 폭발한다. 이는 파라미터 선택이 아니라 **구조적 제약**이며, 설정 로더가 위반 시 예외를 던진다.

**INV-07-2 (순수 함수)**: `transition(state, metrics, config) -> ThemeState` 는 순수 함수다. 내부 상태·시각·난수를 참조하지 않으므로 전이표 전체를 망라 테스트할 수 있다.

**INV-07-3 (도달 가능성)**: `BREAKDOWN` 에서 `LEADING` 으로 가는 유일한 경로는 `RECOVERY` 를 경유한다. 직접 전이가 존재하면 "떨어지는 칼날"을 즉시 재매수하게 된다. 그래프 도달성 테스트로 강제한다.

---

## 2. Architecture

```
 gold feature panel + InstrumentMaster(theme, index_key)
              │
              ├─► cluster.py    ClusterResolver
              │     L1 index_key 유일화 → L2 theme 상한 m
              │     대표 선정: 유동성 → 추적 → 괴리 → 규모
              │
              ├─► theme.py      ThemePanel  (테마 레벨 시계열 집계)
              │     테마 수익률 = 대표 ETF 수익률 (동일가중 평균 아님 — 실제 매수 가능 대상과 일치시킴)
              │
              ├─► state.py      ThemeState · transition()  (INV-07-1..3)
              │
              └─► leadership.py SectorLeadershipModel(AlphaModel)
                    score = f(SectorScore, ThemeState)
```

`src/alpha/leadership.py` 는 06 의 `AlphaModel` Protocol 을 그대로 만족하므로, **기존 harness 를 한 줄도 고치지 않고** B0~B5 와 동일 프로토콜로 비교된다(INV-10).

### 2.1 테마 수익률 정의

테마 시계열은 **대표 ETF 의 실제 가격**으로 만든다. 테마 내 ETF 동일가중 평균이 아니다 — 우리가 실제로 살 수 있는 것은 대표 ETF 하나이므로, 평균을 쓰면 실행 불가능한 수익률을 최적화하게 된다.

레버리지 ETF 는 대표 선정에서 **기본 제외**하고(INV-9: 배수 합성 금지, 그리고 R-2 미확정), 레버리지 사용 여부는 08 의 포트폴리오 정책에서 별도 결정한다.

---

## 3. Acceptance Gates

07 이 채택되려면 06 harness 에서 다음을 **모두** 만족해야 한다.

| 게이트 | 기준 |
| --- | --- |
| A-1 | 동일 프로토콜에서 `P(R>30%)` **>0.091** (B2, 06 결과) |
| A-2 | 클러스터 중복제거를 껐을 때보다 worst-5% 가 개선 (분산 효과 실재 확인) |
| A-3 | state machine 을 끄고 점수만 썼을 때 대비 턴오버 감소 **and** `P(R>30%)` 비열위 |
| A-4 | 성분별 ablation 에서 기여 없는 성분이 제거된 뒤에도 A-1 유지 |
| A-5 | 파라미터 ±30% 섭동에서 `P(R>30%)` 붕괴 50% 미만 (G-2) |

하나라도 실패하면 **07 을 폐기하고 baseline 으로 대회에 나간다.** 이것이 정상적인 결과이며 실패가 아니다.

---

## 4. Assumptions

- **A-1**: 테마 최소 멤버 3개(cluster breadth 유의성 하한). 미만이면 그 테마는 개별 ETF로만 취급.
- **A-2**: 테마 간 percentile 은 그날 유동성 필터를 통과한 대표 ETF 를 가진 테마들 사이에서만 계산.
- **A-3**: `configs/taxonomy.yaml` 커버리지 목표는 유동 universe 기준 90% (INV-04-7).

---

## 5. 예상 심볼 (contract 확정 시)

```
src/alpha/cluster.py     ClusterResolver · ClusterChoice · select_representative
src/alpha/theme.py       ThemePanel · build_theme_panel
src/alpha/state.py       ThemeState · TransitionConfig · transition · run_state_machine
src/alpha/leadership.py  SectorScoreWeights · SectorLeadershipModel
configs/strategies.yaml  weights · thresholds (히스테리시스 검증 포함)
```

---

## 6. Contract 유예 사유

이 spec 의 파라미터는 전부 **06 의 실측 분포에 의존**한다.

- 성분 가중치 $w_k$: ablation 결과 없이 정하면 근거 없는 숫자다
- 전이 임계 $a, b, r, e, d$: 실제 테마 수익률 분포의 분위수에서 유도해야 한다
- 테마당 상한 $m$: 집중도-분산 트레이드오프 실측 결과에 의존

지금 contract 로 고정하면 `/implement` 가 근거 없는 상수를 코드에 박고, 그것이 INV-11 과 AGENTS.md 의 *Invariant Logic Over Magic Numbers* 를 정면 위반한다.

**해제 조건**: 06 의 W3 게이트 4항목 충족 + B0~B5 분포표 + **preflight PASS 및 B5≠B4 재측정** 시에 `/spec 07_leadership_engine` 재실행.
