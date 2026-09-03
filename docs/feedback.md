최신 코드와 `latest.md`를 다시 기준점으로 잡으면, 이전 답변과 우선순위가 달라집니다.

## 핵심 판단

**TASK_41의 sell-first 수정은 실제 개선입니다.** P27의 gross violation이 `1,958 → 0`, ruin이 `4.21% → 2.54%`로 낮아지고, 동시에 `P>50% 4.69% → 4.26%`, q99 `86.9% → 80.8%`로 떨어졌습니다. 즉 과도한 노출이 제거되면서 성과도 함께 정상화됐습니다. 현재 P27은 처음으로 “백테스트 엔진 오류 때문에 좋아 보이는 전략”에서 벗어났다고 봐도 됩니다.

하지만 이제 `latest.md`가 제안하는

> timing controller → selection ensemble

순서를 그대로 구현하는 것은 권하지 않습니다.

현재 Tail Forensics가 **방향 탐색용으로는 유용하지만 전략 개발의 정량적 근거로 쓰기에는 attribution 설계가 아직 부정확하기 때문입니다.**

제가 지금 코드를 맡는다면 아래 순서로 갑니다.

---

# 1. 가장 먼저 TASK_43: Tail Forensics 2.0

현재 가장 큰 문제입니다.

## 문제 1. Oracle universe가 실제 대회 universe가 아님

현재 `_plus2_tickers(master)`는 master에 있는 모든 +2x ETF를 대상으로 oracle을 찾습니다. 그리고 각 window에서 그 ETF가 당시 실제로

* 존재했는지
* sponsor universe였는지
* history 조건을 충족했는지
* liquidity 조건을 충족했는지
* deployment universe에 포함되었는지

를 확인하지 않습니다.

즉 다음과 같은 일이 가능합니다.

```text
2020 window
↓
2024년에 상장된 ETF가 현재 master에 존재
↓
_close_on(2020) = None
```

이면 자연히 제외되기는 합니다.

그러나 더 미묘한 문제는:

```text
당시 존재 O
당시 대회 deployment eligibility X
당시 ADV 부족
당시 sponsor 조건 X
```

인 ETF도 oracle candidate가 될 수 있다는 점입니다.

### 수정

Oracle도 반드시:

```python
universe.get(window_start, deployment_filters)
```

또는 날짜별 PIT universe를 사용해야 합니다.

더 정확하게는 각 entry 날짜마다:

```text
PIT eligible family set(t)
```

을 만들어야 합니다.

---

# 2. 현재 timing attribution은 실전 execution과 맞지 않음

P27의 실제 실행은:

```text
decision close
→ next session open fill
→ open-to-close PnL
```

입니다.

그런데 forensics는:

```python
compound_close_return(
    ticker,
    window_start,
    window_end
)
```

즉 **close-to-close**입니다.

그리고 first entry 역시 `decision_date`를 entry로 사용해 close 가격부터 계산합니다.

따라서 현재:

> entry_timing_loss = 0.216

이라는 숫자는 실제 P27 execution의 entry timing opportunity cost가 아닙니다.

### 반드시 바꿔야 합니다

oracle도 실제 engine semantics와 동일하게:

```text
signal(t close)
→ trade(t+1 open)
→ fees
→ ADV constraint
→ gross constraint
```

로 평가해야 합니다.

즉 `compound_close_return()` 대신 최소한:

```python
next_open_to_close_path_return(...)
```

형태가 필요합니다.

이걸 하지 않으면 앞으로 entry controller를 개발하면서 **잘못된 objective를 최적화할 위험**이 큽니다.

---

# 3. `exit_timing_loss`와 `giveback_loss`가 중복됨

이건 코드상 명확한 문제입니다.

현재 exit timing은:

```python
peak = entry_bh

for e in sessions:
    r = return(first_entry, e)
    peak = max(peak, r)

exit_timing_loss = peak - entry_bh
```

입니다. 여기서 `entry_bh`는 이미:

```text
first_entry → window_end
```

수익률입니다.

따라서 `exit_timing_loss`는 사실상:

> actual-family ETF의 peak-to-final giveback

입니다.

그런데 동시에:

```python
giveback_loss = window.giveback
```

도 따로 넣습니다.

결과적으로 같은 현상을

```text
exit_timing
+
giveback
```

에 중복 배분하고 있을 수 있습니다.

그래서 현재 latest의:

```text
entry timing 29.8%
exit timing 16.5%
giveback 17.1%
timing total 46.3%
```

를 그대로 신뢰하면 안 됩니다.

### 수정

손실 decomposition을 mutually exclusive하게 만들어야 합니다.

제가 권하는 방식은:

$$
R_{oracle}
-
R_{actual}
$$

을 순차 counterfactual로 분해하는 겁니다.

```text
A = actual strategy
B = actual selection + oracle entry
C = actual selection + oracle entry + oracle exit
D = oracle selection + oracle entry + oracle exit
```

그럼:

```text
entry loss     = B - A
exit loss      = C - B
selection loss = D - C
```

가 됩니다.

이렇게 해야 합이 정확히:

```text
oracle gap = selection + entry + exit
```

이 됩니다.

giveback은 별도 원인 bucket으로 더하지 말고 **진단 metric**으로만 두는 편이 좋습니다.

---

# 4. 그래서 현재 `primary_gap=timing` 판정은 보류

현재 결과에서는:

* 평균 loss 기준 timing > selection
* window dominant 기준 selection 53.6%

으로 서로 다른 메시지가 나오고 있습니다.

저는 이것을

> “timing이 더 중요하다”

라고 해석하지 않습니다.

오히려:

> **몇 개의 극단적인 window가 timing 평균을 크게 올리고 있을 가능성이 높다**

고 봅니다.

실제로 worst entry timing window가:

```text
+315.9%
+286.9%
+284.8%
```

수준의 gap을 만들고 있습니다.

이런 2026 Q2 초대형 상승 구간 몇 개가 mean attribution을 지배할 가능성이 매우 큽니다.

따라서 다음에는 mean만 보지 말고:

```text
median
trimmed mean
q75
q90
share of total
effective independent window
era별 attribution
```

을 모두 출력해야 합니다.

특히 overlapping 36D windows라서 265개의 attribution row가 265개의 독립 사건이 아닙니다.

---

# 5. Forensics 수정 후 첫 전략 실험은 “exit controller”가 아니라 Momentum Ensemble

여기가 지난 답변과 달라지는 부분입니다.

현재 P27은 여전히 사실상:

```python
score = mom_60
```

입니다.

반면 이미 feature에는:

```yaml
momentum_horizons:
  [3, 5, 10, 20, 40, 60]
```

이 준비되어 있습니다.

36-session 대회에서 `mom60` 하나만 쓰는 것은 지나치게 단일 horizon입니다.

### 제가 가장 먼저 실험할 P29

복잡한 ML이 아니라 **rank ensemble**입니다.

예:

$$
S_i =
0.15R(mom10_i)
+0.25R(mom20_i)
+0.25R(mom40_i)
+0.35R(mom60_i)
$$

여기서 `R()`은 cross-sectional percentile/rank입니다.

중요한 건 저 weight를 optimize하면 안 됩니다.

딱 3개 정도만 사전 정의합니다.

### Candidate A — slow

```text
mom20  0.15
mom40  0.30
mom60  0.55
```

### Candidate B — balanced

```text
mom10  0.15
mom20  0.25
mom40  0.30
mom60  0.30
```

### Candidate C — acceleration

```text
mom20 rank
+
(mom20 - mom60 normalized)
```

정도만 테스트합니다.

---

# 6. 왜 ensemble을 먼저 보느냐

2025 actual oneshot에서 P27은 이미 +41.4%입니다.

따라서 P27의 문제는:

> 승자가 발생했을 때 전혀 못 따라가는 것

만은 아닙니다.

더 큰 문제는 2018~2024 competition-anchor:

```text
2018 -11.3
2019 -4.1
2020 +4.0
2021  0
2022  0
2023 -2.1
2024 -0.4
2025 +41.4
```

입니다.

즉 현재 전략은 **강한 60일 trend가 이미 존재하는 해에만 제대로 작동하는 경향**이 강합니다.

이건 exit 문제 이전에 **signal responsiveness 문제**일 가능성이 큽니다.

그래서:

```text
mom60 → mom10/20/40/60 ensemble
```

을 먼저 검증할 가치가 큽니다.

목적은 평균 수익률 상승이 아니라:

```text
P30/P40/P50 유지 또는 상승
+
ruin <= P27
+
anchor non-2025 개선
+
year stability 개선
```

입니다.

---

# 7. 두 번째 실험: Relative Strength + Acceleration, breadth는 아직 넣지 말 것

현재 `strategies.yaml`에는 leadership score에:

```text
RS        45%
accel     30%
breadth   25%
```

가 존재합니다.

하지만 저는 이것을 그대로 P29에 넣지는 않겠습니다.

이유는 factor를 세 개 동시에 추가하면 **무엇이 개선을 만들었는지 식별하기 어려워지기 때문**입니다.

먼저:

$$
score = momentum\ ensemble + acceleration
$$

만 봅니다.

예를 들어:

$$
Accel =
mom_{20} - mom_{60}
$$

또는 더 안정적으로:

$$
Accel =
rank(mom_{20}) - rank(mom_{60})
$$

입니다.

이건 현재 P27이 놓치는:

> 최근 2~4주에 급격히 leadership으로 진입한 theme

을 잡는 데 직접 대응합니다.

---

# 8. Regime controller는 현재 구현을 그대로 사용하면 안 됨

현재 `regime.py`는 생각보다 단순합니다.

5개 component가 전부 사실상 binary입니다.

```text
KOSPI > MA20
KOSPI MA20 slope > 0
KOSDAQ > MA20
breadth > 0.5
20d vol < 2.5%
```

그리고 이를 weighted sum해서 5단계 regime으로 나눕니다.

이 정도로는:

```text
STRONG_RISK_ON
RISK_ON
NEUTRAL
RISK_OFF
STRONG_RISK_OFF
```

이라는 5개 상태가 주는 정밀도가 실제보다 높아 보일 가능성이 있습니다.

특히:

```python
MA window = 20
```

도 함수 내부에 hard-code되어 있습니다.

### 저는 regime부터 전략 switching에 사용하지 않겠습니다.

우선 regime classifier 자체를 검증합니다.

각 regime별로 P27 next-5/10/20-session return을 출력합니다.

예:

| regime     |  n | mean | P>20 | P<-10 |
| ---------- | -: | ---: | ---: | ----: |
| strong on  |    |      |      |       |
| on         |    |      |      |       |
| neutral    |    |      |      |       |
| off        |    |      |      |       |
| strong off |    |      |      |       |

만약 monotonicity가:

```text
STRONG_ON > ON > NEUTRAL > OFF
```

형태로 나오지 않으면 **regime 이름은 있지만 예측력은 없는 것**입니다.

이 검증 전에는 inverse switching을 붙이지 않는 게 좋습니다.

---

# 9. 이후 P30: absolute trend gate가 아니라 “relative deterioration gate”

P28B 결과가 이미 알려준 것이 있습니다.

Absolute momentum cash gate는:

```text
ruin 2.58%
```

까지 낮췄지만 tail을 크게 훼손했습니다. 이전 결과에서도 P28B가 P27보다 tail이 낮았습니다. 최신 문서 역시 해당 결과를 reference로 유지합니다.

즉:

```text
mom < 0 → cash
```

는 너무 둔합니다.

다음에는 absolute level 대신 **momentum deterioration**을 보는 게 좋습니다.

예:

```text
held leader:
mom20 rank drops > X percentile
AND
mom5 < mom20 trend
```

또는:

```text
leader score percentile
90 → 85 → 72
```

처럼 cross-sectional leadership deterioration을 감지합니다.

핵심은 시장이 떨어졌다는 이유로 나가는 것이 아니라:

> **내가 들고 있는 winner가 leadership을 잃었기 때문에 나가는 것**

입니다.

---

# 10. 현재 `min_gap=0.04`도 scale 문제가 있음

P27의 sticky switching은:

```python
held_score + 0.04 >= top_score
```

이면 기존 종목을 유지합니다.

여기서 score가 raw `mom60`이라 0.04는 **4%p momentum difference**라는 경제적 의미가 있습니다.

그런데 ensemble/rank score로 바꾸면 이 threshold는 의미가 완전히 달라집니다.

따라서 P29에서는 sticky logic도:

```text
absolute raw difference
```

가 아니라:

```text
rank gap
```

또는

```text
score z-gap
```

기준으로 바꿔야 합니다.

예:

```text
new leader percentile - held percentile > 0.10
```

처럼 합니다.

이렇게 해야 feature scale 변화에도 안정적입니다.

---

# 11. sell-first에도 아직 한 단계 더 고칠 부분이 있음

현재 구현은 개선됐지만 완성형은 아닙니다.

`constrain_target_weights_sell_first()`는:

1. ADV 기준으로 sell 가능한 양 계산
2. sell 이후 gross 계산
3. remaining buys 계산
4. gross 초과 시 **모든 buy delta를 동일 비율로 binary scaling**

합니다.

현재 P27처럼 거의 Top1이면 크게 문제되지 않습니다.

하지만 향후 2~3개 candidate를 사용하면 문제가 생깁니다.

예:

```text
A alpha = 매우 강함
B alpha = 약함

gross budget 부족
```

이어도 A와 B buy를 같은 비율로 줄입니다.

### 개선

향후 multi-position 전략에서는 gross budget을:

$$
priority =
\frac{alpha\ confidence}
{gross\ consumption}
$$

또는 간단히 score rank 순으로 배분해야 합니다.

```text
best candidate fill first
→ residual gross
→ second candidate
```

가 대회 목적에는 더 맞습니다.

단 **P27 자체에서는 지금 수정 우선순위가 낮습니다.**

---

# 12. 더 중요한 execution edge case도 하나 있음

현재 sell-first는 **target을 계산할 때만** sell-first입니다.

실제 execution은 이후:

```python
execution.resolve(target, ...)
```

로 target 전체를 한 번에 넘깁니다.

따라서 이론적으로:

```text
old ETF sell
new ETF buy
```

중 old ETF가 실제 next-open에서 미체결되고 new ETF만 체결될 수 있습니다.

그 경우 target 단계에서는 sell이 된다고 가정했지만 실제 ledger에서는 sell이 안 된 상태가 됩니다.

현재 역사 run에서 `execution_gross_violation=0`이므로 데이터상 발생하지 않았거나 문제가 드러나지 않은 것으로 보입니다.

하지만 실전용 엔진은 더 강하게 만들어야 합니다.

### 진짜 sell-first execution

```text
Phase 1
SELL orders resolve

↓ actual fills 확인

실제 post-sell state 계산

↓ residual gross 재계산

Phase 2
BUY orders resolve
```

로 바꾸는 것이 맞습니다.

**지금 구현은 sell-first target constraint이고, 제가 권하는 것은 sell-first actual execution입니다.**

이건 대회 전 반드시 넣는 것이 좋습니다.

---

# 13. `carry_gross_drift_count=19,430`은 현재 거의 쓸모 없는 metric

latest에도 이미:

> rolling session diagnostics 합산, calendar-unique 아님

이라고 되어 있습니다.

2090 overlapping windows 안에서 반복 집계된 숫자라

```text
19,430번 발생했다
```

는 실제 의미가 거의 없습니다.

다음에는:

```text
unique calendar day drift rate
mean excess gross
q95 excess gross
max excess gross
next-session delever success rate
delever turnover cost
```

를 내야 합니다.

예:

```text
gross limit        1.90
close realized     1.91
excess             0.01
```

같은 drift는 사실 크게 신경 쓸 필요 없습니다.

반면:

```text
close realized     2.15
```

가 반복된다면 다른 문제입니다.

count보다 magnitude가 중요합니다.

---

# 14. 평가 framework를 다음처럼 바꾸면 좋음

현재 단일 headline:

```text
P>50%
```

에 너무 쉽게 시선이 갑니다.

앞으로 candidate promotion은 4축으로 두십시오.

### A. Championship tail

```text
P>30
P>40
P>50
P>60
q95
q99
```

### B. Robustness

```text
leave-one-year-out
era split
Sep-Nov anchor
parameter neighborhood
```

### C. Downside

```text
P<-15
P<-25
CVaR5
max giveback
```

### D. Execution

```text
execution violation
unfilled rate
ADV utilization
turnover
cost
gross drift magnitude
```

그리고 저는 promotion 조건을:

```text
P50 개선
AND
ruin 비열화 없음
AND
LOYO 6/8년 이상 non-inferior
AND
parameter neighborhood 안정
```

정도로 잡겠습니다.

---

# 15. 특히 Leave-One-Year-Out을 지금 당장 추가해야 함

이게 현재 가장 중요한 과최적화 방지 장치입니다.

2018~2026 데이터를 계속 보고 P20부터 P28까지 개발했기 때문에 **현재 전체 기간은 이미 사실상 training set**입니다.

따라서 단순 full-period bootstrap만으로는 부족합니다.

예:

```text
train idea without 2018 → evaluate 2018
train idea without 2019 → evaluate 2019
...
```

라는 엄밀한 ML 방식까지는 어렵더라도, 최소한 candidate 결과를:

```text
2018–19
2020–21
2022–23
2024–26
```

또는 연도별로 분해해야 합니다.

특히 candidate가:

```text
full P50 4.26 → 5.10
```

인데 개선분의 90%가 2025~26에서 나오면 채택하지 않는 편이 좋습니다.

---

# 16. 실제 개발 순서

지금이라면 이렇게 갑니다.

| Priority | TASK                              | 목적                          |
| -------- | --------------------------------- | --------------------------- |
| **P0**   | TASK_43 Attribution v2            | 지금의 timing/selection 결론 검증  |
| **P0**   | TASK_44 LOYO / era evaluation     | 과최적화 차단                     |
| **P1**   | TASK_45 `mom_ensemble` P29        | selection responsiveness 개선 |
| **P1**   | TASK_46 acceleration A/B          | 신규 leadership 포착            |
| **P1**   | TASK_47 actual sell→buy execution | 실전 무결성                      |
| **P2**   | TASK_48 deterioration exit P30    | giveback 감소                 |
| **P2**   | TASK_49 regime predictive audit   | regime의 실제 가치 검증            |
| **P3**   | TASK_50 +1/+2/inverse router      | 검증된 regime일 때만 적용           |
| **P3**   | leaderboard controller            | 대회 시작 이후                    |

---

# 17. 제가 가장 기대하는 후보

현재 코드와 결과를 고려하면 다음 구조가 가장 유력합니다.

```text
             Multi-horizon Family Momentum
                    10 / 20 / 40 / 60
                           │
                           ▼
                Relative Acceleration
                           │
                           ▼
                 Cross-sectional Rank
                           │
                           ▼
                   Sticky Leader
                    rank hysteresis
                           │
                           ▼
                 +2x vehicle (우선)
                           │
                           ▼
            Leadership deterioration exit
                           │
                           ▼
                sell-first execution
```

처음부터:

```text
regime + inverse + gold + breadth + breakout + flow
```

를 모두 넣으면 안 됩니다.

지금 가장 필요한 것은 **더 복잡한 전략이 아니라 한 번에 하나의 hypothesis를 검증하는 것**입니다.

---

# 최종 판단

현재 상태를 제가 점수화하면:

| 영역                            | 평가                                  |
| ----------------------------- | ----------------------------------- |
| PIT universe                  | 좋음                                  |
| next-open execution           | 좋음                                  |
| gross correctness             | **크게 개선됨**                          |
| rolling tournament evaluation | 좋음                                  |
| statistical overfit control   | 부족                                  |
| tail forensics                | **아이디어는 좋지만 현재 attribution은 수정 필요** |
| P27 alpha                     | 지나치게 단순                             |
| regime model                  | 아직 전략 switching에 쓰기엔 약함             |
| 실전 execution                  | 한 단계 더 필요                           |
| 우승권 가능성                       | tail은 존재하지만 아직 낮음                   |

따라서 **지금 P31, P32처럼 전략을 계속 늘리지 말고 먼저 forensics를 바로잡는 게 맞습니다.**

그리고 그 다음 실험 하나를 고르라면 저는 **`P29 = multi-horizon rank momentum ensemble`**을 고르겠습니다. 현재 코드에서 가장 적은 복잡도 증가로 P27의 가장 명확한 구조적 약점인 `mom60 단일 horizon 의존`을 해결할 수 있기 때문입니다.

목표도 `P>50% 4.26 → 무조건 6%`처럼 잡으면 안 됩니다. 첫 성공 기준은 오히려:

```text
P>50 ≥ 4.26%
P>40 > 6.22%
ruin ≤ 2.54%
CVaR5 ≥ -30.1%
2025 +41.4% 크게 훼손하지 않음
2018~24 anchor aggregate 개선
gross violation = 0
LOYO/era 안정성 개선
```

으로 잡는 것이 신뢰성 있는 방향입니다.

원하면 다음 답변에서 바로 **TASK_43~47을 실제 파일/클래스/함수 수준의 구현 명세로 작성해서 Codex에 그대로 넣을 수 있는 작업 지시서 형태**로 만들어드리겠습니다.
