## 1. 최신 push 기준 결론

이번 수정은 방향이 맞습니다. 특히 `TASK_35_P27_EXECUTION_REBASELINE`에서 제가 직전에 지적했던 네 가지 핵심 문제인 **next-open 수익 귀속, HOLD/CASH 의미 분리, P21/P27 exposure 비교 오염, gross metric 미확인 상태의 PASS**를 직접 수정했습니다. 다만 결정 로그에 명시되어 있듯 **수정된 실행엔진으로 2018~2026 전체 rebaseline은 아직 끝나지 않았습니다.** 따라서 이전의 `P>50≈4.8%`, `ruin≈3.1%`, `P21 대비 46.7%`는 이제 최종 전략 판단에 쓰면 안 됩니다.

제 판단은 현재 이렇습니다.

| 영역                    | 현재 평가 | 판단                        |
| --------------------- | ----: | ------------------------- |
| P27 알파 아이디어           |    B+ | 계속 기준전략으로 유지              |
| path-dependent 검증     |    A- | 이전 핵심 오류 해결               |
| next-open 방향성         |    B+ | 큰 오류 수정, 잔여 정합성 있음        |
| intent 설계             |    A- | HOLD/CASH/TARGET 분리는 올바름  |
| fast simulator parity |    C+ | **아직 핵심 불일치 존재**          |
| 통계적 검증 설계             |    B- | 구조는 있으나 selection bias가 큼 |
| 목적함수                  |    B+ | 지금은 변경보다 freeze가 낫다       |
| 실전 최종전략 신뢰도           | 아직 미달 | rebaseline 전에 판단 금지       |

그리고 이번 재검토에서 결론이 하나 더 명확해졌습니다.

> **앞으로의 핵심은 P28, P29처럼 알파를 계속 붙이는 것이 아니라, “한 번의 36일 대회를 하나의 독립 실험으로 정확하게 재현하는 시스템”을 먼저 완성하는 것입니다.**

그 위에서 아주 제한된 후보만 비교해야 합니다.

---

# 2. 최신 수정에서 제대로 해결된 부분

`compute_next_open_session_return()`이 새 포지션에 overnight gap을 귀속하지 않고,

$$
\text{old position: close}_{t-1}\rightarrow open_t
$$

와

$$
\text{new position: open}_t\rightarrow close_t
$$

를 분리합니다. 이전의 가장 큰 인과 오류는 제거됐습니다.

또 `PortfolioIntent`가 생기면서

```text
HOLD
TARGET(weights)
CASH
```

가 구분됐습니다. 빈 dict가 오류인지, 유지인지, 전량매도인지 알 수 없었던 과거 구조보다 훨씬 낫습니다.

모델별 exposure도 이제 `alpha_equal`과 `full_strategy_own`을 분리할 수 있습니다. P21의 순수 알파와 P27을 비교할 때 동일하게 P27 exposure budget을 적용하고, 실제 전체전략 비교 때는 각자의 원래 exposure를 적용할 수 있습니다.

마지막으로 gross diagnostic이 없는데 0으로 간주하는 것도 막았습니다. `gross_violation_count=None`이면 championship gate가 `INSUFFICIENT_EVIDENCE`가 되도록 한 것은 정확한 설계입니다.

여기까지는 상당히 좋은 개선입니다.

---

# 3. 하지만 아직 남은 가장 큰 문제: fast simulator가 `PortfolioIntent`를 이해하지 못한다

이건 다음 수정에서 반드시 해결해야 합니다.

현재 `StickyLeaderModel.score()`는 crash 조건에서 실제로 `CASH_INTENT`를 반환할 수 있습니다.

full `BacktestEngine`은 이제 그것을 처리합니다.

그런데 fast `simulate_window_from_cache()`에서는 여전히:

```python
sc = model.score(...)
scores = {str(k): float(v) for k, v in dict(sc).items()}
```

형태입니다.

`sc == CASH_INTENT`이면 `dict(sc)`가 실패하고 `scores={}`가 됩니다. 그 이후 generic sizing이 empty target을 만들며 실제 `CASH` 의미가 사라집니다.

이건 현재 P27 자체에는 `cash_drawdown=0`이라 영향이 제한적입니다.

하지만 **P21은 crash-cash를 쓰고**, 제가 이후 가장 중요한 후보라고 보는 `P27 + absolute momentum cash`도 CASH를 쓰게 됩니다.

즉 지금 상태로 P28을 테스트하면:

> full engine에서는 cash
> fast 2090-window에서는 hold

가 될 수 있습니다.

### 따라서 원칙을 바꿔야 합니다

`BacktestEngine`과 `TournamentSimulator`에 각각 execution logic을 복사하지 마십시오.

둘 다 동일한 함수 하나를 사용해야 합니다.

예를 들면:

```text
transition_portfolio_state(
    previous_state,
    market_open,
    target_intent,
    execution_constraints
) -> new_state
```

를 canonical implementation으로 두고,

* full backtest
* fast rolling
* one-shot
* live decide/replay

가 전부 이것을 호출하게 하는 것이 맞습니다.

이렇게 해야 앞으로 새로운 cash/inverse/partial-fill 로직이 생겨도 simulator와 engine이 갈라지지 않습니다.

---

# 4. 더 심각한 잔여 look-ahead: 다음 날 거래량을 시가 주문에 사용한다

이건 최신 코드에서도 반드시 수정해야 합니다.

`PointInTimeUniverse`의 ADV는 trailing trading value 평균인데 **조회 날짜 자체의 거래대금을 포함**합니다.

그런데 engine/session cache에서 다음 날 시가에 주문할 때 capacity를 계산하기 위해:

```text
adv(ticker, execution_date)
```

를 사용합니다.

예를 들어:

```text
9/10 종가에 signal 계산
9/11 시가에 매수
```

라면 9/11 시가 시점에는 9/11 하루 전체 거래대금을 알 수 없습니다.

그런데 현재 방식은 실질적으로:

```text
9/11 종가까지의 거래량을 알고
9/11 시가에 얼마 살지 결정
```

하는 셈입니다.

### 수정 기준

다음 날 시가 주문의 ADV는:

$$
ADV_t = ADV \text{ known at decision date } t-1
$$

이어야 합니다.

즉:

```python
adv(ticker, decision_date)
```

또는 엄격하게는

```python
trailing ADV ending decision_date
```

를 써야 합니다.

이건 단순 execution approximation이 아니라 **실제 look-ahead**이므로 전체 rebaseline 전에 반드시 수정해야 합니다.

---

# 5. next-open P&L도 아직 완전히 정확하지 않다

새 helper 자체의 수익률 분리는 맞습니다. 문제는 engine에서 사용하는 순서입니다.

현재 엔진은 대략:

```text
prev close equity
↓
transaction cost 차감
↓
overnight return
↓
intraday return
```

순서입니다.

그런데 trade는 다음 날 **open**에서 일어납니다.

정확한 순서는:

```text
prev close equity
↓
기존 포지션 overnight return
↓
open equity
↓
open에서 매매
↓
transaction cost
↓
새 포지션 intraday return
↓
close equity
```

입니다.

수식으로는:

$$
E_{\text{open,before}}
=
E_{t-1}(1+r_{overnight})
$$

그다음:

$$
Cost
=
Turnover_{\text{open}}\times E_{\text{open,before}}\times c
$$

$$
E_{\text{open,after}}
=
E_{\text{open,before}}-Cost
$$

그리고

$$
E_t
=
E_{\text{open,after}}(1+r_{intraday})
$$

이어야 합니다.

현재는 transaction cost가 overnight 전에 빠집니다.

3+5bps 정도에서는 큰 차이가 아닐 수 있습니다. 하지만 목표가 **실전 신뢰성**이라면 이런 근사를 남겨놓고 tail 0.5~1%p 차이를 논하면 안 됩니다.

---

# 6. 더 근본적으로 `weight`를 실제 보유 상태로 쓰는 문제

현재 `current_weights`는 사실상 목표 비중입니다.

예를 들어:

```text
ETF 95%
cash 5%
```

를 들고 있는데 ETF가 하루 +10% 올랐다고 하겠습니다.

실제 종가 비중은 더 이상 95%가 아닙니다.

$$
\frac{0.95\times1.10}
{0.95\times1.10+0.05}
\approx95.43\%
$$

가 됩니다.

그런데 현재 engine/simulator는 다음 날에도 계속 `0.95`를 보유비중으로 사용합니다.

이 문제는 다음에 영향을 줍니다.

| 영향                      | 결과               |
| ----------------------- | ---------------- |
| 다음 날 overnight P&L      | 실제 exposure와 다름  |
| 리밸런싱 turnover           | 실제보다 크거나 작음      |
| commission/slippage     | 잘못 계산            |
| ADV position delta      | 잘못 계산            |
| effective gross         | 실제와 차이           |
| current holding context | 전략 state와 조금씩 차이 |

P27은 대부분 한 종목 95%라 차이가 작아 보일 수 있지만, **2x ETF + 높은 변동성 + 36일 누적**에서는 무시하는 게 좋지 않습니다.

---

# 7. 그래서 execution engine은 weight simulator가 아니라 `cash + units` ledger가 되어야 한다

최종 전략을 신뢰하려면 여기까지 가는 걸 권합니다.

상태를:

```text
cash
ticker -> shares
```

로 표현합니다.

하루 transition은:

```text
전일 보유수량
→ 오늘 open mark-to-market
→ 주문 계산
→ partial fill
→ 수수료/슬리피지
→ 새로운 보유수량
→ close mark-to-market
→ equity 계산
```

입니다.

그러면:

$$
Equity_t
=
Cash_t+\sum_i Shares_{i,t}\times Price_{i,t}
$$

라는 불변식이 매일 정확하게 성립합니다.

그리고 weight는 계산 결과일 뿐입니다.

$$
w_{i,t}=
\frac{Shares_{i,t}P_{i,t}}{Equity_t}
$$

이 구조를 만들면:

* overnight gap
* position drift
* partial fills
* cash liquidation
* turnover
* cost
* gross
* integer share rounding

이 전부 자연스럽게 해결됩니다.

### 성능 걱정은 크지 않습니다

2090 windows × 36 sessions ≈ 75,000 day transitions입니다.

매일 실제 보유 종목이 1~2개라면 **가격/feature/universe를 미리 cache해 놓은 상태에서 ledger 계산은 매우 가볍습니다.**

지금 7분이 걸리는 이유는 ledger 연산 자체가 아니라 feature/universe/panel access 쪽이 더 큽니다.

따라서:

> **cache는 유지하고, execution state만 정확한 ledger로 변경**

하는 게 최선입니다.

---

# 8. 현재 `RollingDiagnostics`는 구현이 아니라 scaffold 상태다

`RollingDiagnostics`가 생긴 것은 좋은데 현재 fast rolling에서는:

```python
gross_violation_count=None
effective_gross_max=None
turnover_mean=None
fill_count=None
unfilled_count=None
```

로 생성됩니다.

그리고 championship gate는 이제 이 경우 제대로 `INSUFFICIENT_EVIDENCE`를 반환합니다.

즉 현재 코드상 정상적인 상황은:

> **P27 최신 엔진 성과는 아직 championship PASS라고 부를 수 없음**

입니다.

Task35의 "full rebaseline pending"과 정확히 일치합니다.

따라서 먼저 rolling simulator에서 실제로 다음을 집계해야 합니다.

```text
effective_gross_max
gross_violation_count
turnover
fill_count
unfilled_count
cash_sessions
active_weight
```

특히 violation은 post-hoc reconstructed trade가 아니라 **실제 portfolio state에서 직접 계산**하는 게 맞습니다.

---

# 9. 또 하나 검사해야 할 데이터 문제: ETF 가격이 raw OHLC다

현재 KRX provider는 값을 거의 verbatim으로 가져옵니다.

ETF schema도:

```text
TDD_CLSPRC → close
TDD_OPNPRC → open
NAV
...
```

등 raw 가격입니다.

제가 확인한 코드 경로에서는 **분배금/분할/기타 corporate action을 total-return으로 조정하는 계층이 보이지 않습니다.**

이건 반드시 audit해야 합니다.

왜냐하면 mom60이:

$$
\frac{P_t}{P_{t-60}}-1
$$

이므로 중간에 ETF 분배락 등이 있으면 실제 투자 총수익과 가격 momentum이 달라질 수 있기 때문입니다.

실제 대회가 특정 계절에 열린다는 것과 별개로, 지금의 2018~2026 rolling sample은 **연중 전체**를 사용합니다.

따라서 historical distribution에 이런 이벤트가 섞이면:

* mom60 signal
* 36일 P&L
* ruin event
* tail event

가 왜곡될 수 있습니다.

최소한 `corporate_action_audit`를 만들어:

```text
대규모 overnight gap
NAV divergence
underlying index divergence
known distribution/reset event
```

를 탐지하고 그 window가 tail/ruin에 얼마나 영향을 줬는지 확인해야 합니다.

---

# 10. 여기서부터가 진짜 중요한 부분: “한 번의 실전” 문제는 통계적으로 완전히 다르다

일반적인 퀀트 전략이면:

$$
E[R]
$$

이나 Sharpe를 높이는 게 중요합니다.

지금 목적은:

$$
\boxed{
P(\text{36일 후 최종 1위})
}
$$

입니다.

그러면 다음은 별로 중요하지 않습니다.

```text
평균 월수익
평균 MDD
Sharpe
승률 55%
```

P27이 P21보다 36일 window에서 46%밖에 못 이기더라도,

P27이 이기는 window가:

```text
+50%
+70%
+100%
```

이고 P21이 이기는 window가:

```text
+5%
+10%
+15%
```

라면 대회용으로는 P27이 훨씬 좋을 수 있습니다.

그래서 **field win_rate를 주목적으로 만들면 안 됩니다.**

---

# 11. 현재 championship objective는 당분간 그대로 두는 게 낫다

현재 championship score는:

$$
C_s
=
\sum_{k} w_{s,k}P(R>T_k)
$$

이고 threshold가 30/40/50/60%입니다.

이건 사실 꽤 합리적입니다.

우승에 필요한 최종수익률 \(T\)가 정확히 얼마인지 모른다고 하고,

$$
T\in
\{30,40,50,60\%\}
$$

에 어떤 사전확률을 준다면,

$$
E[1(R>T)]
$$

가 바로 현재 score입니다.

즉 현재 score는 **불확실한 우승 threshold에 대한 win utility surrogate**로 볼 수 있습니다.

그래서 지금 objective를 다시 바꾸면 안 됩니다.

이미 P20→P27까지 같은 데이터로 상당한 연구가 이루어졌습니다.

objective까지 다시 결과에 맞춰 움직이면 researcher degrees of freedom이 더 늘어납니다.

### 바꿔야 하는 것은 objective 자체가 아니라 “선택 방법”입니다

point estimate:

$$
C(P28)>C(P27)
$$

만으로 선택하지 말고,

$$
\boxed{
LCB_{95\%}
[
C(P28)-C(P27)
]
\ge0
}
$$

가 핵심이어야 합니다.

그리고 weak/championship/hot 세 scenario에서 최소한 심각한 열화가 없어야 합니다.

---

# 12. 지금 가장 큰 적은 모델이 아니라 `researcher overfitting`이다

P20부터 P27까지 동일한 2018~2026 데이터를 여러 번 봤습니다.

또 일부 실제 대회 경로도 이미 알고 있습니다.

그러면 전통적인 의미의 clean OOS는 사실상 없습니다.

이 상태에서 P28, P29, P30을 계속 만들면 결국:

> 2018~2026의 우연에 가장 잘 맞는 전략

을 찾을 가능성이 높아집니다.

### 여기서 연구 방식을 바꿔야 합니다

앞으로는 **후보 개수 자체를 제한**하십시오.

제가 권하는 연구 budget은 딱 세 가지입니다.

| 후보    | 내용                           |
| ----- | ---------------------------- |
| P27-R | execution-corrected P27      |
| P28-A | P27 + absolute momentum cash |
| P28-B | signed ±2x + cash sticky     |

그 이상은 이 세 후보의 결과를 보기 전에는 만들지 않는 것이 좋습니다.

---

# 13. P27의 구조적 약점은 명확하다

P27은 현재:

```text
long +2x only
mom60 top1
95%
min_hold 2
min_gap 4%
cash 없음
inverse 없음
```

입니다.

그리고 +2x 후보가 모두 음의 60일 momentum이어도 **가장 덜 나쁜 하나를 매수합니다.**

이건 `filter_plus2_scores()`가 음수 momentum을 제거하지 않기 때문입니다.

경제적으로 보면:

> “현재 모든 long 2x에 edge가 없다”

는 상태가 존재하지 않습니다.

이 구조는 상승장이면 강력합니다.

반대로 36일 대회가 하락 추세에 걸리면 출발부터 구조적으로 불리합니다.

---

# 14. 그래서 첫 번째 알파 개선은 `absolute momentum cash`가 맞다

execution을 완성한 후 P28-A는 이것만 추가하십시오.

```python
top_score = max(mom60 among +2x)

if top_score <= 0:
    CASH
else:
    current P27
```

다른 파라미터는 P27과 100% 동일해야 합니다.

왜 `0`이 좋냐면 3%, 7%, 11% 같은 threshold를 데이터에 맞춰 튜닝한 것이 아니라:

$$
60d\ total\ return >0
$$

이라는 경제적으로 해석 가능한 조건이기 때문입니다.

### 중요한 질문은 평균수익이 아닙니다

P28-A가:

```text
ruin 3% → 1%
CVaR 개선
```

했다고 해서 채택하면 안 됩니다.

만약 동시에:

```text
P>50 5% → 3%
P>60 4% → 2%
```

가 되면 대회용으로 나빠진 겁니다.

채택 조건은:

> **right tail 거의 유지 + catastrophic/no-edge path 제거**

여야 합니다.

---

# 15. 그 다음 가장 중요한 후보는 `signed ±2x + cash`

저는 이것이 P27 이후 가장 중요한 구조적 확장이라고 봅니다.

복잡한 regime classifier부터 넣을 필요가 없습니다.

아주 단순하게:

```text
eligible:
  +2x
  -2x

score:
  actual ETF mom60

select:
  highest positive mom60

if no positive:
  cash
```

로 시작합니다.

즉:

$$
a_t =
\begin{cases}
\arg\max_{i\in \{\pm2x\}} mom60_i,&\max mom60_i>0\\
cash,&otherwise
\end{cases}
$$

입니다.

### 이 구조가 중요한 이유

현재 P27은 우승 가능 경로가:

> 강한 상승 theme 발생

으로 제한됩니다.

signed 전략이면:

> 강한 상승 theme
> 또는 강한 하락 direction

둘 다 convex tail source가 됩니다.

36일 한 번만 투자해야 한다면 **가능한 우승 regime의 영역 자체를 넓히는 것**이 중요합니다.

---

# 16. 하지만 signed 전략도 그냥 채택하면 안 된다

inverse를 넣으면 whipsaw가 생길 수 있습니다.

예:

```text
상승 → 하락 → 상승
```

에서:

```text
+2 → -2 → +2
```

를 반복하면 수익이 무너질 수 있습니다.

따라서 기존 P27의 sticky가 오히려 중요합니다.

처음에는:

```text
mom60
min_gap=.04
min_hold=2
```

를 그대로 유지하십시오.

direction별 별도 parameter를 만들지 마십시오.

```text
long gap
inverse gap
direction-flip gap
inverse hold
```

처럼 늘리는 순간 과적합 공간이 급격히 커집니다.

---

# 17. 성과 분석을 “return distribution”에서 “실패 원인 decomposition”으로 바꿔야 한다

이게 다음 research에서 가장 중요합니다.

각 36일 window를 단순히:

```text
P27 = +20%
```

로 끝내지 마십시오.

그 window에서:

```text
실제로 우승권 수익을 만들 기회가 있었는가?
```

를 먼저 분석해야 합니다.

예를 들어 완벽한 hindsight oracle을 diagnostic으로만 만듭니다.

$$
Oracle_w
=
\max_{i\in eligible} R_{i,w}
$$

그리고 P27과 비교합니다.

다음 네 유형으로 분류할 수 있습니다.

| 유형                               | 의미              | 다음 연구        |
| -------------------------------- | --------------- | ------------ |
| Oracle도 낮음                       | 시장 자체에 우승 기회 없음 | 알파 문제가 아님    |
| Oracle 높음, P27 낮음                | 종목선택 실패         | scoring 개선   |
| P27 중간, peak 높음                  | giveback        | exit/overlay |
| long oracle 낮고 inverse oracle 높음 | 방향성 문제          | inverse 필요   |

이 분석이 없으면:

> momentum을 60에서 50으로 바꿀까?
> gap을 4%에서 3%로 바꿀까?

같은 미세 튜닝만 반복하게 됩니다.

그런 튜닝은 우승 가능성을 구조적으로 높이지 못합니다.

---

# 18. Tail을 window 개수로 세면 안 된다

2090개 36일 rolling window는 실제로 2090개의 독립 대회가 아닙니다.

서로 35일씩 겹칩니다.

따라서:

```text
P>50 = 5%
```

라면 100개 이상 window가 +50처럼 보여도 실제로는:

```text
강한 상승 episode A
강한 상승 episode B
강한 상승 episode C
```

세 번일 수도 있습니다.

### `TailEpisodeReport`를 꼭 만드십시오

각 threshold에 대해:

```text
raw exceedance windows
independent episodes
years represented
families represented
largest episode share
largest year share
largest family share
```

를 계산하십시오.

특히 다음 상황은 위험합니다.

```text
P>50 = 5%

하지만
70%가 2025 한 episode
80%가 semiconductor family
```

라면 P50 5%라는 숫자는 신뢰도가 매우 낮습니다.

---

# 19. `n_effective=58`이라면 5% tail은 사실 몇 번 안 나온 것이다

execution 수정 후 다시 계산해야 하지만, 이전 수준처럼:

```text
n_effective ≈ 58
P50 ≈ 5%
```

라면 독립 관점에서는 대략:

$$
58\times0.05\approx3
$$

번입니다.

단순 binomial Wilson interval만 계산해도 3/58의 95% 범위는 대략:

$$
1.8\%\sim14.1\%
$$

입니다.

그리고 실제로는:

* 시계열 의존성
* 모델 선택
* 반복 튜닝
* 시장구조 변화

가 있으므로 이것보다 불확실합니다.

즉 앞으로 절대로:

> P50=4.8%니까 우승권 수익이 나올 확률이 약 5%

라고 해석하면 안 됩니다.

---

# 20. 그래서 non-overlapping 검증을 추가해야 한다

rolling 2090개는 distribution visualization에는 좋습니다.

**모델 selection에는 별도 dataset view가 필요합니다.**

제가 추천하는 방식은 `phase robustness`입니다.

horizon=36이면 시작 offset을 0~35로 바꿔가며:

```text
offset 0:
window 0,36,72,...

offset 1:
window 1,37,73,...

...

offset 35
```

를 만듭니다.

각 phase 안에서는 window가 겹치지 않습니다.

그리고 각 parameter/model의 championship score를 36개 phase에서 계산합니다.

보고 싶은 것은:

```text
best phase
median phase
q25 phase
worst phase
```

입니다.

P28이 전체 rolling에서는 좋아졌는데 36 phase 중 몇 개에서만 압도적으로 좋다면 과적합 가능성이 큽니다.

---

# 21. parameter의 “최적점”이 아니라 “평원”을 찾아야 한다

현재 P27의:

```text
mom60
gap4%
hold2
weight95%
```

가 정말 견고한지 확인해야 합니다.

결과를 보고 최고점을 찾지 말고 **주변만 조사**하십시오.

예:

| parameter | neighborhood           |
| --------- | ---------------------- |
| momentum  | 40 / 50 / 60 / 70 / 80 |
| min_gap   | 2 / 4 / 6%             |
| min_hold  | 1 / 2 / 3              |
| exposure  | 0.85 / 0.90 / 0.95     |

좋은 전략은:

```text
mom50 0.060
mom60 0.064
mom70 0.061
```

처럼 나와야 합니다.

나쁜 전략은:

```text
mom50 0.040
mom60 0.064
mom70 0.039
```

입니다.

후자는 P27이 아니라 **mom60이라는 historical accident**를 선택했을 가능성이 높습니다.

---

# 22. 연도별 검증보다 `leave-one-tail-episode-out`가 더 중요할 수 있다

일반 평균 전략에서는 leave-one-year-out이 좋습니다.

Tail 전략에서는 아예 가장 좋은 episode 하나를 빼보는 게 중요합니다.

예:

```text
전체:
champ score 0.06

최대 tail episode 제거:
0.058
```

이면 상당히 강합니다.

반대로:

```text
0.06 → 0.025
```

면 모델이 사실상 한 번의 역사적 rally를 학습한 것입니다.

`leave-one-year-out`, `leave-one-tail-episode-out`, `leave-one-family-out` 세 가지는 반드시 넣는 것이 좋습니다.

---

# 23. 현재 historical opportunity set 문제도 반드시 진단해야 한다

정확한 HTS 종목 얘기가 아니라 **전략 통계의 문제**입니다.

2018의 +2x 선택지와 2026의 +2x 선택지는 다릅니다.

따라서 P27의 성과가:

```text
n_candidates=2
```

시대와

```text
n_candidates=10
```

시대를 그냥 섞으면 distribution이 이상해질 수 있습니다.

각 window 시작 시:

```text
eligible 2x count
number of families
top mom60
top-second mom gap
positive-momentum breadth
cross-sectional dispersion
```

을 저장하십시오.

그리고 championship performance를 조건부로 봅니다.

이 결과가 상당히 중요합니다.

예를 들어:

$$
P(\text{tail}|\ top\ mom60>20\%)
$$

가 급격히 높고

$$
P(\text{tail}|\ top\ mom60<0)
$$

가 거의 0이라면 absolute momentum cash의 경제적 근거가 강해집니다.

---

# 24. 이 분석을 한 뒤에만 상태변수를 전략에 넣는다

중요한 순서는:

```text
조건부 분석
→ 반복되는 failure pattern 확인
→ 하나의 state variable 추가
```

입니다.

반대 순서:

```text
regime
breadth
volatility
flow
RS
MACD
RSI
...
```

를 한꺼번에 넣으면 안 됩니다.

P14~P19 계열에서 이미 비슷한 방향의 복잡도가 실패한 기록이 있습니다.

P27의 장점은 **단순하고 convex**하다는 것입니다.

이 장점을 버리면 안 됩니다.

---

# 25. 실제 대회에서는 static strategy만으로는 최종적으로 부족하다

알파는 static해도 됩니다.

하지만 **대회 마지막 단계 risk policy**는 static일 이유가 없습니다.

우승 목적이라면 동일한 +40%도 상황에 따라 의미가 다릅니다.

```text
나는 +40%
1위 +60%
남은 3일
```

이면 공격해야 합니다.

반대로:

```text
나는 +40%
2위 +25%
남은 2일
```

이면 리스크를 줄이는 게 합리적일 수 있습니다.

따라서 최종 architecture는:

```text
Alpha Layer
+
Tournament Risk Layer
```

로 분리하는 게 맞습니다.

---

# 26. Tournament Risk Layer는 고정 +50 lock이 아니라 “gap + remaining” 기반이어야 한다

P26이 이미 보여준 핵심 교훈은:

```text
수익률 +50%
```

그 자체가 lock 조건이 되면 안 된다는 것입니다.

실전에서는:

$$
state_t
=
(
R_{\text{self}},
R_{\text{leader}},
rank,
remaining,
market\ opportunity
)
$$

여야 합니다.

### 특히 좋은 기준은 `remaining opportunity frontier`입니다

남은 \(d\)일 동안 과거 시장에서 가능한 최대 convex move를 계산합니다.

예:

$$
O_d =
\max_{ETF}R_{ETF,d}
$$

의 historical distribution.

내가 1등이고 2등과의 gap이 \(G\)라면:

$$
G > Q_{99}(O_d)
$$

처럼 **남은 기간 역사적으로 따라잡기 매우 어려운 gap**이 되었을 때만 defensive lock을 검토할 수 있습니다.

이건:

```text
+50% 도달하면 cash
```

보다 훨씬 논리적입니다.

---

# 27. 대회 초중반에는 leaderboard를 무시하는 게 오히려 낫다

너무 일찍 rank에 반응하면 noise를 추적하게 됩니다.

36일 중 초반 20~25일은:

> **market alpha만 실행**

하는 게 좋습니다.

대회 risk overlay는 마지막 7~10일 정도부터만 의미가 생길 가능성이 큽니다.

그리고 이 값도 7인지 8인지 10인지 backtest 최고점을 찾으면 안 됩니다.

넓은 범위에서 결과가 비슷한지를 봐야 합니다.

---

# 28. 최종 채택 기준은 이런 형태가 되어야 한다

| Gate                 | 제가 요구할 조건                                    |
| -------------------- | -------------------------------------------- |
| Engine parity        | full/fast random windows 결과 일치               |
| Intent parity        | HOLD/CASH/TARGET 모두 full/fast 동일             |
| No lookahead         | 모든 signal/capacity data timestamp ≤ decision |
| Ledger invariant     | 매일 cash + positions = equity                 |
| Diagnostics          | gross/turnover/fill 모두 실제 값 존재               |
| Primary objective    | paired bootstrap LCB vs P27 ≥ 0              |
| Hot-field            | P50/P60 tail 비열화                             |
| Ruin                 | hard constraint 통과                           |
| Episode robustness   | 특정 한 episode 의존 아님                           |
| Era robustness       | 특정 era에서 catastrophic collapse 없음            |
| Family robustness    | 특정 ETF family 하나에만 의존하지 않음                   |
| Parameter robustness | 주변 parameter에서도 유사 성능                        |
| Cost stress          | 비용 증가해도 ranking 유지                           |
| Delay stress         | 1 session delay에 완전히 붕괴하지 않음                 |
| Selection aware      | nested/outer validation에서도 승격                |

이 gate를 모두 통과하지 못하면:

> backtest 최고 모델

이지

> 실전에서 믿을 모델

은 아닙니다.

---

# 29. 앞으로 제가 추천하는 개발 순서

1. **P27 알파를 완전히 freeze**합니다. mom60/gap4/hold2/95를 건드리지 않습니다.

2. `ExecutionStateTransition` 또는 `PortfolioLedger`를 하나 만들고 full engine과 fast simulator가 동일 코드를 호출하게 합니다.

3. execution 시점 ADV를 `decision_date`까지만 사용하도록 수정합니다.

4. 거래비용을 `overnight → open trade → intraday` 순서에 맞춰 계산하고 weight drift 또는 share ledger를 정확히 처리합니다.

5. fast simulator에 `PortfolioIntent`를 직접 지원시켜 P21 crash cash와 future P28 cash가 full engine과 완전히 동일하게 동작하도록 합니다.

6. `RollingDiagnostics`를 실제 state transition에서 산출하고 `None` scaffold를 제거합니다.

7. raw ETF 가격의 distribution/corporate-action 영향을 audit합니다.

8. 그 뒤 **P21 / P26 / P27 전체를 완전히 새로 rebaseline**합니다. 이전 task log 성능 수치는 전부 legacy로 표시합니다.

9. rebaseline 결과에서 `tail episode / family / era / candidate-count / oracle gap` forensic을 수행합니다.

10. 그 분석에서 no-edge long exposure가 실제 ruin/low-performance와 연결되면 **P28-A = positive absolute momentum + cash**만 테스트합니다.

11. 이후 **P28-B = signed ±2x + cash + 동일 sticky**를 테스트합니다.

12. P27/P28-A/P28-B 중 하나를 nested/phase/episode/parameter/stress validation으로 선택합니다.

13. 알파가 freeze된 뒤에만 **rank + remaining + gap 기반 tournament overlay**를 별도 레이어로 개발합니다.

14. 최종적으로 `alpha model`과 `tournament overlay`를 각각 freeze하고 이후에는 코드 버그 외 parameter 변경을 금지합니다.

---

# 30. 제가 보는 가장 유력한 최종 형태

지금 단계에서 장기적으로 가장 가능성이 높은 구조는 다음입니다.

```text
[Directional Opportunity]
     ↓
signed +2x / -2x candidates
     ↓
actual mom60
     ↓
positive absolute momentum gate
     ↓
sticky leader
(min_hold, min_gap)
     ↓
high concentration
     ↓
accurate next-open execution
     ↓
rank-independent core trading
     ↓
last-phase tournament risk overlay
(rank + gap + remaining)
```

즉 복잡한 ML ensemble이 아닙니다.

오히려:

$$
\boxed{
\text{Trend direction}
+
\text{Convex instrument}
+
\text{Concentration}
+
\text{Hysteresis}
+
\text{Cash when no edge}
+
\text{Late tournament adaptation}
}
$$

형태가 이 대회 목적에는 가장 자연스럽습니다.

---

# 최종 판단

지금까지의 P20→P27 과정에서 가장 의미 있는 발견은 **`mom60 + 집중 + sticky`가 유력한 right-tail 구조라는 것**입니다.

하지만 지금부터 중요한 것은 그 backtest score를 조금 더 높이는 게 아닙니다.

한 번뿐인 실전에서 믿을 수 있게 만들려면:

$$
\boxed{
\text{simulation error}
<
\text{statistical uncertainty}
<
\text{strategy edge}
}
$$

가 되어야 합니다.

현재는 아직 execution/validation uncertainty가 strategy edge와 비슷한 크기일 가능성이 있습니다.

그래서 **지금 P28 알파 개발로 넘어가는 것은 아직 빠릅니다.**

우선순위는 명확합니다.

> **canonical execution ledger → ADV look-ahead 제거 → full/fast intent parity → 실제 diagnostics → 완전 rebaseline → tail opportunity decomposition**

까지 먼저 끝내십시오.

그 이후 결과를 제가 다시 본다면, 그때는 단순히 “P28을 어떻게 만들까”가 아니라 **P27을 유지할지, cash만 추가할지, signed ±2x로 확장할지를 실제 데이터 근거에 따라 결정할 수 있습니다.**

그 지점부터가 실제 **우승 전략 개발 단계**라고 봅니다.
