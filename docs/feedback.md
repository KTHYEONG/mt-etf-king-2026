## 재평가 결론

**P27의 방향 자체는 P26보다 명확히 좋아졌습니다.** `house-money overlay`를 제거한 것은 맞는 결정입니다. 다만 코드를 다시 따라가 보니, 지금은 새로운 알파를 추가할 단계보다 **P27의 평가 엔진을 먼저 바로잡아야 할 가능성이 높습니다.**

특히 중요한 문제가 하나 있습니다.

> **현재 P27의 rolling 36일과 annual one-shot이 실제로 “각 시작일마다 현금에서 새로 시작한 36일 대회”를 정확하게 시뮬레이션하지 않을 가능성이 큽니다.**

이게 맞다면 현재 `P>50%=5.3%`, `ruin=4.5%`, one-shot 연도별 성과, P21 대비 win rate 55%까지 다시 계산해야 합니다.

---

# 1. 가장 먼저 수정해야 할 문제: P27은 경로의존적인데 평가기가 경로독립으로 취급한다

`StickyLeaderModel`은 명백하게 상태를 가집니다.

```python
self._held
self._hold_len
```

그리고 오늘 점수는 현재 보유 종목과 보유 기간에 따라 달라집니다. `min_hold=2`, `min_gap=0.04`가 바로 이 상태를 이용합니다.

그런데 `model_requires_path_dependent()`는

```python
if hasattr(model, "path_dependent"):
    ...
return isinstance(model, PortfolioPolicy)
```

만 검사합니다. `StickyLeaderModel`에는 현재 `path_dependent=True`가 없습니다. 따라서 경로독립으로 판정됩니다.

그리고 평가기는 경로독립 모델이면

```python
engine.run(model, 전체기간)
→ 전체 daily return 생성
→ window_returns(daily_rets, 36)
```

으로 처리합니다. 즉 2090개의 각 36일 window를 새로 실행하는 것이 아니라 **2018~2026 하나의 긴 투자 경로를 실행한 다음 36일씩 잘라냅니다.**

### 이 차이는 P27에서 중요합니다

예를 들어 어떤 36일 window가 2023-09-01에 시작한다고 해도 현재 방식에서는:

* 보유종목이 이미 존재할 수 있고
* `_hold_len`도 이전 기간에서 이어지고
* 전날 전략 상태도 이어지고
* 시작일부터 새로운 TOP1을 선택하는 것이 아닐 수 있습니다.

실제 대회 모델은:

```text
t = 0
capital = initial capital
holdings = {}
_hold_len = 0
```

이어야 합니다.

따라서 제가 이전 답변에서 rolling 구조를 높게 평가했던 부분은 **P27 코드까지 다시 추적한 결과 수정해야 합니다.**

---

# 2. Annual one-shot도 같은 문제가 있다

P27 CLI는 one-shot 계산에서 전체기간 backtest의 daily return을 가져온 후,

```python
oneshot_window_returns(
    _daily_p27,
    sessions,
    starts,
    horizon
)
```

으로 특정 날짜의 36일을 잘라냅니다.

`oneshot_window_returns()` 자체는 단순 복리 계산 함수라 문제 없습니다.

문제는 입력되는 `_daily_p27`이 **그 해 9월 21일에 포트폴리오를 초기화해 만든 daily return이 아니라 2018년부터 이어진 긴 backtest 경로**라는 것입니다.

따라서 지금 제시한:

| 연도 | 현재 수치 |
| ---- | --------: |
| 2018 |    -33.4% |
| 2019 |     -4.5% |
| ...  |       ... |
| 2025 |    +43.1% |

은 지금 단계에서는 **진짜 annual one-shot이라고 단정하면 안 됩니다.**

참고로 표 자체에서도 작은 오류가 있습니다. 양수는 2개가 아니라 **2020, 2022, 2025의 3/8개**입니다. 제공한 숫자로 단순 계산하면 평균 약 -1.46%, 중앙값 -6.45%입니다. 하지만 위 시뮬레이션 문제를 수정하기 전에는 이 숫자에도 큰 의미를 부여하지 않는 게 맞습니다.

---

# 3. 수정 방법

P27 자체의 알파는 건드리지 말고 우선 평가기를 고치는 **P27-validation fix**를 먼저 하는 것을 권합니다.

`StickyLeaderModel`에 최소한 다음 의미가 들어가야 합니다.

```python
path_dependent = True
scores_path_independent = False

def reset_trackers(self):
    self._held = None
    self._hold_len = 0
```

`engine.run()`은 시작할 때 `reset_trackers()`를 호출할 준비가 이미 되어 있습니다.

그리고 `TournamentSimulator`도 `scores_path_independent=False`이면 fast cache 대신 slow per-window rerun으로 전환하는 구조가 이미 있습니다.

즉 아키텍처를 새로 만들 필요는 없습니다.

### 검증해야 할 invariant

가장 중요한 테스트는 이것입니다.

```text
rolling window start = X
```

일 때

```text
simulator rolling[X]
```

와

```text
engine.run(
    fresh P27(),
    start=X,
    end=X+35 sessions
)
```

의 terminal return이 완전히 같아야 합니다.

랜덤하게 20~50개 window를 골라:

```text
abs(rolling_ret - independent_ret) < 1e-10
```

을 보장하십시오.

이게 통과하지 않으면 championship 통계는 사용하면 안 됩니다.

---

# 4. P21 field 비교도 현재 같은 문제가 있다

P27 field report에서 P21을 이렇게 돌립니다.

```python
simulator.run_rolling(
    b21_m,
    ...,
    path_dependent=False,
)
```

즉 P21도 명시적으로 경로독립 처리됩니다.

따라서 현재

> P27 vs P21 win_rate = 55%

도 재계산해야 합니다.

다행히 비교 양쪽이 비슷한 방식으로 왜곡되어 있어서 55%가 완전히 무의미하다고 볼 수는 없습니다. 하지만 **채택 근거로 쓰기에는 부족합니다.**

---

# 5. P27 championship PASS에도 구조적 함정이 하나 있다

현재 코드:

```python
candidate_p27 = list(rolling.returns)
raw_p27 = list(rolling.returns)
```

입니다.

그리고:

```python
evaluate_championship_adoption(
    candidate_returns=candidate_p27,
    incumbent_returns=P21,
    raw_returns=raw_p27,
)
```

를 실행합니다.

따라서

$$
R_{candidate}=R_{raw}
$$

입니다.

그러므로 다음 검사는 구조적으로 무조건 통과합니다.

$$
score(candidate)\ge score(raw)
$$

그리고 paired CI도 정확히 0을 중심으로 하는 동일 비교입니다.

### 즉 P27 PASS의 정확한 의미

현재 PASS는:

> **P27 raw가 P21에 비해 championship scenario에서 열등하지 않고, 자기 자신(raw)을 훼손하지 않았다.**

입니다.

다음 뜻은 아닙니다.

> P27이 P26보다 통계적으로 유의하게 우수하다.

P26→P27 승격을 검증하려면 incumbent를 바로 전 전략으로 잡는 게 논리적으로 맞습니다.

```text
candidate = P27 identity
incumbent = P26 executable house-money
anchor = P21
raw reference = P26/P27 raw
```

즉:

### Promotion gate

$$
P27 > P26
$$

### Long-term anchor gate

$$
P27 \not< P21
$$

를 분리하는 것이 더 깔끔합니다.

---

# 6. `field win_rate=55%, top2=100%`의 해석

현재 field에는 rival이 사실상 P21 하나뿐입니다.

`field_relative_report()`는

```python
n_agents = 1 + len(rivals)
```

이고 rank≤2이면 top2로 계산합니다.

지금은:

```text
P27
P21
```

두 명뿐이므로

$$
P(top2)=100\%
$$

는 **정의상 무조건 100%**입니다.

따라서 현재 field 지표에서:

* `win_rate=55%` → 정보 있음
* `top2_rate=100%` → 정보 없음
* `median_rank_percentile=0.50` → 정보 거의 없음

입니다.

### 그래서 field system을 크게 만드는 것도 아직 권하지 않습니다

저라면 synthetic 참가자 100명을 만드는 것보다 먼저 내부 anchor set만 고정합니다.

예:

```text
B1
P21
P24 raw
P26 executable
P27
```

그리고 각 동일 window에서 pairwise comparison을 합니다.

단 이것을 `P(win)`이라고 부르면 안 되고:

```text
internal_field_win_rate
```

정도로만 사용합니다.

---

# 7. 알파 자체에서 제가 지금 가장 집중해서 볼 부분

검증 문제를 제외하면 P27에서 가장 눈에 띄는 약점은 **inverse가 없는 것보다 먼저 이것**입니다.

## P27은 모든 +2x ETF의 momentum이 음수여도 반드시 하나를 산다

`filter_plus2_scores()`는 `mom_60`이 음수라고 제거하지 않습니다.

```python
out[ticker] = float(fv)
```

입니다.

그리고 P27은 `allocate()`가 없으므로 generic TOP1 sizing으로 갑니다.

`weights_from_scores(TOP1)`은 점수의 부호와 관계없이 가장 높은 종목 하나를 100% target으로 만듭니다.

예를 들면:

```text
ETF A mom60 = -12%
ETF B mom60 = -18%
ETF C mom60 = -25%
```

여도 A를 매수합니다.

P27 exposure rule에 의해 사실상:

```text
95% × +2x
```

노출을 갖습니다.

### 이건 P27의 중요한 구조적 약점입니다

상승 추세에서는 매우 좋은 convexity를 만들어냅니다.

반대로 시장 전체가 하락/약세인 상황에서는:

> **“좋은 기회가 없음”이라는 상태가 모델에 존재하지 않습니다.**

이건 inverse보다 먼저 실험할 가치가 있습니다.

---

# 8. 제가 P28에서 가장 먼저 실험할 것은 inverse가 아니다

아주 단순한 **absolute momentum gate**입니다.

### P28-A

```python
if top_mom60 <= 0:
    return {}
```

나머지는 P27 그대로:

```text
mom60
+2x only
min_gap=.04
min_hold=2
95% concentration
identity overlay
```

입니다.

이렇게 해야 무엇이 좋아졌는지 명확합니다.

후보는 처음에는 딱 세 개면 충분합니다.

| 후보  | 조건            |
| ----- | --------------- |
| P27   | 현재 그대로     |
| P28-A | top mom60 > 0   |
| P28-B | top mom60 > +5% |

그리고 **평균수익률을 보고 고르면 안 됩니다.**

P>50/P>60과 tail objective가 유지되면서 ruin/CVaR가 얼마나 줄어드는지를 봐야 합니다.

개인적으로는 이 실험의 정보가 inverse branch보다 훨씬 클 가능성이 있다고 봅니다.

---

# 9. 목적함수는 P27에서 아직 개선할 필요가 있다

현재 championship score는:

$$
0.1P(R>30)
+0.25P(R>40)
+0.45P(R>50)
+0.2P(R>60)
$$

형태입니다.

나쁘지 않은 진단값이지만 **hard threshold가 너무 많습니다.**

예를 들어:

```text
+49.9%
+50.1%
```

가 꽤 다르게 취급되고,

```text
+60%
+110%
```

은 마지막 threshold 관점에서는 차이가 없습니다.

## 저는 바로 synthetic P(win)까지 가지 않겠습니다

경쟁자 분포 자체가 불확실하기 때문에 잘못 만든 field model은 오히려 목적함수를 더 과적합시킬 수 있습니다.

그 대신 **bounded continuous championship utility**를 추천합니다.

예를 들면:

$$
u(R)=
\mathrm{clip}
\left(
\frac{R-0.30}{0.50},
0,
1
\right)
$$

입니다.

그러면:

```text
R <= 30%   → 0
40%        → 0.2
50%        → 0.4
60%        → 0.6
70%        → 0.8
>=80%      → 1
```

이 됩니다.

그리고

$$
J=E[u(R)]
$$

를 계산합니다.

이건 수학적으로 30~80% 구간의 **survival curve 면적**과 같은 형태입니다.

### 장점

* +60%보다 +80%를 제대로 높게 평가
* +49.9/+50.1 같은 threshold artifact 감소
* q99 하나에 모델이 끌려가지 않음
* bounded라 극단적인 단일 사례의 영향 제한
* 36일 우승권 수익률이라는 목적과 직접 정렬

---

# 10. 그런데 point estimate를 최대화하면 또 과적합한다

그래서 실제 최적화 목적은

$$
\boxed{
J_{\rm robust}
=
LCB_{95\%}(E[u(R)])
}
$$

를 추천합니다.

즉 평균 championship utility가 아니라 **block bootstrap의 하단 신뢰한계**를 최대화합니다.

그리고 ruin은 목적함수에 섞지 않고 hard constraint로 유지합니다.

예:

$$
P(R<-25\%) \le 5\%
$$

또는 최소한

$$
P_{candidate}(R<-25)
\le
P_{incumbent}(R<-25)+\epsilon
$$

을 사용합니다.

결국:

```text
maximize:
    bootstrap_LCB(championship_utility)

subject to:
    ruin constraint
    execution invariant
    gross invariant
    tail non-inferiority
```

가 됩니다.

이 방식이 현재 championship score보다 **견고성 측면에서 한 단계 낫습니다.**

---

# 11. n_effective=58보다 더 중요한 진단을 하나 추가해야 한다

현재:

```text
2090 rolling windows
P>50 = 5.3%
```

면 raw count 기준으로 약 110개 정도의 +50% window가 존재합니다.

그런데 36일 rolling은 35일이 겹칩니다.

즉 이 110개가 정말 110개의 성공 사례인지,

아니면

```text
강한 상승장 1
강한 상승장 2
강한 상승장 3
```

에서 연속된 수십 개 window가 발생한 것인지가 매우 중요합니다.

## `tail_episode_report`를 추가하는 것을 강하게 권합니다

예:

```text
threshold = 50%

raw exceedance windows: 111
independent tail episodes: ?
years represented: ?
max episode concentration: ?
top-1 episode share: ?
top-3 episode share: ?
```

연속된 threshold exceedance window를 하나의 episode로 묶습니다.

더 엄격하게는 start date 간격이 36 sessions 이내면 같은 episode로 묶어도 됩니다.

### 이상적인 모습

```text
P>50 = 5%
tail episodes = 10
여러 시기에 분산
```

### 위험한 모습

```text
P>50 = 5%
tail episodes = 3
그중 70%가 2025 한 번
```

후자라면 5.3%라는 숫자는 실제보다 훨씬 강해 보이는 것입니다.

**다음 분석에서 저는 이 지표를 매우 중요하게 보겠습니다.**

---

# 12. parameter stability도 지금부터는 필수다

P27은 이미 P20→P27까지 동일 데이터에서 반복적으로 발전했습니다.

따라서 단순히 P28이 0.064 → 0.068이 됐다고 채택하면 안 됩니다.

P27 주변을 확인해야 합니다.

예:

```text
mom horizon:
40 50 60 70 80

min_gap:
0.00 0.02 0.04 0.06 0.08

min_hold:
0 1 2 3 5

weight:
0.80 0.85 0.90 0.95
```

다만 전체 Cartesian grid에서 최고 하나를 선택하면 또 과적합합니다.

찾아야 하는 건 **peak가 아니라 plateau**입니다.

예를 들어:

```text
mom50 = .061
mom60 = .064
mom70 = .062
```

면 좋습니다.

반대로:

```text
mom50 = .043
mom60 = .064
mom70 = .039
```

면 P27은 매우 취약합니다.

### 따라서 새 지표

```text
parameter_neighborhood_median
parameter_neighborhood_q25
parameter_neighborhood_worst
```

를 추가하십시오.

P27 자체 성능보다 **P27 근처에서도 비슷하게 작동하는가**가 훨씬 중요해졌습니다.

---

# 13. Era gate도 이미 있는데 P27에서 사용하지 않고 있다

`evaluate_championship_adoption()`에는 era별 비교 기능이 이미 있습니다.

하지만 P27에서는:

```python
era_pairs=None
```

으로 호출합니다.

따라서 현재 PASS에는 temporal robustness가 들어가지 않습니다.

고정된 구간을 사전에 정해서 넣는 것이 좋습니다.

예:

```text
2018-2020
2021-2023
2024-2026
```

여기서 매번 이기라는 뜻은 아닙니다.

최소한:

```text
global PASS
+
어느 한 era에서 catastrophic deterioration 없음
```

정도를 확인해야 합니다.

구간을 결과를 보고 바꾸면 안 됩니다.

---

# 14. 우선순위를 다시 매기면

지금은 다음 순서가 맞습니다.

| 순위        | 작업                                                     |            중요도 |
| ----------- | -------------------------------------------------------- | ----------------: |
| **1** | StickyLeader path-dependent rolling 수정                 |  **최우선** |
| **2** | annual one-shot을 fresh portfolio run으로 수정           |  **최우선** |
| **3** | P21/P26/P27 모두 동일 independent-window 방식으로 재평가 |  **최우선** |
| **4** | P27→P26 직접 paired promotion gate                      |              높음 |
| **5** | tail episode concentration 분석                          |              높음 |
| **6** | continuous championship utility + bootstrap LCB          |              높음 |
| **7** | parameter plateau 검사                                   |              높음 |
| **8** | `top_mom60 > 0` cash 후보 실험                         |              높음 |
| 9           | era robustness wiring                                    |              중상 |
| 10          | inverse / 복잡한 dynamic policy                          | **그 이후** |

---

# 최종 평가

P26 때보다 평가는 올라갔습니다.

### 제가 보는 현재 상태

**P27 설계 방향:** `A-`

* P26 late-lock 제거: 맞음
* raw right-tail 유지: 맞음
* gross 1.90 제한: 합리적
* 별도 field/oneshot diagnostic 추가: 방향 좋음
* 복잡한 새 알파를 무작정 추가하지 않은 것: 좋음

하지만

**P27 성과 검증 신뢰도:** 현재 `C+`

입니다.

가장 큰 이유는 **StickyLeader의 상태의존성을 rolling/one-shot 평가가 정확히 반영하지 못할 가능성**입니다.

이것부터 고친 후에도

```text
P>50 ≈ 5%
P>60 ≈ 4%
ruin ≈ 4~5%
q99 ≈ 90%
```

가 비슷하게 유지된다면 그때는 P27을 상당히 강한 base champion으로 평가하겠습니다.

그리고 그 다음 로직 개선은 복잡한 inverse나 ML이 아니라:

$$
\boxed{
\text{P27} + \text{simple absolute-momentum no-edge gate}
}
$$

를 제일 먼저 검증하겠습니다.

특히 코드상 현재 **모든 +2x 후보가 음수 momentum이어도 95%를 투자하는 구조**는 P27에서 제가 가장 먼저 공격해볼 알파 레벨 약점입니다.

그리고 목적함수는 `championship_score`를 없애기보다 **진단지표로 유지하고, 실제 모델 선택은 bounded continuous tail utility의 block-bootstrap LCB로 옮기는 것**이 지금 단계에서 가장 견고한 방향이라고 판단합니다.
