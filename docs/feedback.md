
코드와 최근 P20~P26 의사결정 로그, 2026 대회 공식 규칙, 2025 실제 대회 결과까지 맞춰봤습니다.

## 결론

**P26을 현재 상태 그대로 2026 실전에 투입해서 “우승 가능성이 충분히 검증됐다”고 판단하면 안 됩니다.**

반대로, P26이 실패작이라는 뜻도 아닙니다. 현재까지 만든 모델 중에서는 **실제로 우승권 수익률을 낼 수 있는 우측 꼬리를 만든 상당히 유의미한 후보**입니다. 문제는 지금 검증하는 목적함수가 **“내가 1등 할 확률”을 직접 최적화하지 않고 있다는 것**입니다.

저라면 현 상태를 이렇게 평가합니다.

| 항목                                 |           평가 | 판단                                             |
| ------------------------------------ | -------------: | ------------------------------------------------ |
| 우승권 수익률 생성 능력              |   **B+** | P26에서 확실히 개선됨                            |
| 36일 단발성 대회 시뮬레이션          |   **B+** | 경로의존 상태를 매 윈도우 초기화하는 구조가 좋음 |
| 거래/유동성 현실성                   |    **B** | ADV·비용·next-open 등을 상당히 고려            |
| 목적함수의 실제 대회 정렬            |    **C** | 현재 가장 큰 문제                                |
| 통계적 OOS 신뢰도                    |   **C-** | P20→P26 반복 선택으로 연구자 과적합 위험 큼     |
| 하락장 대응                          |   **C-** | P26은 사실상 +2x long-only                       |
| 실시간 순위 대응                     |   **D+** | 구조는 있으나 현재 정책에서 거의 활용하지 않음   |
| **현재 상태로 우승 전략 확정** | **보류** | repo 자체 championship gate도 FAIL               |

가장 중요한 사실은 저장소의 P26 최종 기록입니다.

> KRX 2018~2026 기준 **P(36일 수익률 > 50%) = 5.3%**, championship score = **0.064**, ruin = **4.5%**, gross violation = 0. P25 대비 우측 꼬리는 크게 개선됐지만 최종 `championship` 판정은 **FAIL**, 이유는 `hot_field vs raw`, `primary CI vs raw`입니다.

즉 **코드베이스 자신의 가장 엄격한 검증도 아직 P26을 챔피언으로 승인하지 않았습니다.**

---

# 1. P26에서 실제로 잘 된 부분

P26은 그냥 흔한 ETF 모멘텀 전략이 아닙니다. 현재 핵심은 대략 다음과 같습니다.

`mom_60`으로 +2배 레버리지 ETF 중 강한 리더를 찾고, 2거래일 minimum hold와 0.04 switching gap을 두면서, 단일 ETF 비중을 최대 95%까지 허용합니다. 2배 ETF이므로 실질 gross exposure는 약 1.90까지 갑니다. 현금은 최소 5%입니다.

P20의 `P>30%=9.28%, P>40%=6.75%`에서 시작해 P21, P24 등을 거쳐 P26에서는 `P>30%≈11.2%, P>50%=5.3%`까지 올라왔습니다. 단순 평균수익률을 올린 게 아니라 **대회에 필요한 우측 꼬리를 실제로 밀어올렸다는 점은 좋습니다.**

그리고 롤링 백테스트 구조도 좋은 부분이 있습니다. 경로의존 모델은 각각의 36일 window마다 tracker를 reset하고, 현금 상태에서 포트폴리오를 다시 구성합니다. 따라서 하나의 장기 백테스트 수익률을 단순히 36일씩 잘라낸 것보다 실제 대회 단발 실행에 훨씬 가깝습니다.

이 부분은 유지해야 합니다.

---

# 2. 가장 큰 문제: `championship score`는 우승 확률이 아니다

현재 championship 목적함수는 사실상

$$
C =
0.10P(R>30\%)
+0.25P(R>40\%)
+0.45P(R>50\%)
+0.20P(R>60\%)
$$

형태입니다. 설정은 `thresholds=[0.3,0.4,0.5,0.6]`, championship weights는 `[0.1,0.25,0.45,0.2]`입니다.  코드에서도 각 threshold exceedance를 가중합해서 scenario score를 계산합니다.

따라서 **P26의 championship score 0.064는 “우승 확률 6.4%”가 절대 아닙니다.**

이 목적함수에는 중요한 정보가 빠져 있습니다.

실제 2026 대상 선정 기준은 단 하나입니다.

**전체 참가자 중 최종 수익률 1위.** ([머니투데이][1])

그러므로 궁극적인 목적함수는

$$
\boxed{
P(R_{\text{P26}} > \max(R_1,\dots,R_N))
}
$$

이어야 합니다.

현재 함수는 예를 들어 +61%와 +120%를 똑같이 `R>60%=1`로 처리합니다. 반대로 +59.9%와 +60.1%에는 불연속적인 차이를 줍니다.

순위 경쟁에서는 이상적이지 않습니다.

---

# 3. 목적함수를 `P(win)`으로 바꾸는 것이 다음 개발의 1순위

상대 참가자의 최종 수익률 CDF를 \(F(r)\), 경쟁자가 N명이라고 단순화하면 내 전략이 수익률 \(r\)을 냈을 때 우승할 확률은 대략

$$
P(\text{win}\mid R=r)=F(r)^N
$$

이고,

$$
\boxed{
J(\theta)
=
E[F(R_\theta)^N]
}
$$

를 최대화하는 것이 실제 대회 목적과 맞습니다.

다만 참가자들이 IID는 아니므로 코드에서는 더 현실적으로 **rival-agent field**를 만드는 쪽을 추천합니다.

예를 들어 동일한 역사적 36일 window에서:

* +2x 섹터 모멘텀
* +2x 지수 모멘텀
* 단기 breakout
* 20/60일 momentum
* -2x inverse trend
* 금/원자재 trend
* contrarian
* P20~P26 변형
* 단순 집중매매형

등을 동시에 돌립니다.

그리고 각 window마다 가상의 참가자 집단을 생성해

$$
M_w = \max_j R_{j,w}
$$

를 만들고,

$$
\boxed{
P_{\rm win}
=
\frac{1}{W}
\sum_w
I(R_{\theta,w}>M_w)
}
$$

를 직접 계산하십시오.

최적화가 불안정하면 indicator 대신

$$
J_{\rm soft}
=
E\left[
\sigma\left(
\frac{R_\theta-M}{\tau}
\right)
\right]
$$

같은 soft rank objective를 쓰면 됩니다.

### 최종적으로 보고 싶은 지표

`championship_score`를 없앨 필요는 없습니다. 하지만 **진단지표로 강등**해야 합니다.

Primary metric은 `win_rate`.

Secondary는 `top2_rate`, `median_rank_percentile`, `win_margin`, `P(R<-25%)`.

이렇게 바꾸는 게 맞습니다.

---

# 4. 현재 P26의 더 큰 문제는 알파보다 `+50% late lock`일 가능성이 높다

P26에는

```text
arm = 0.50
lock_remaining = 5
```

가 들어가 있습니다. 그리고 실제 live predicate는 사실상

```text
return >= 50%
AND
remaining_sessions <= 5
=> cash
```

입니다.

이 규칙은 **자산관리 목적이라면 합리적**입니다.

하지만 1등만 노리는 대회에서는 그렇지 않습니다.

더구나 P26의 championship 실패 원인이 바로 **hot-field에서 raw 전략보다 열세이고 primary CI도 raw보다 열세**라고 저장소에 기록돼 있습니다. 결정 로그도 다음 레버로 `overlay or hot_field scenario alignment`를 명시하고 있습니다.

저라면 다음 실험은 새로운 알파를 추가하지 않고 바로 이렇게 합니다.

| Candidate | 알파 | 종료 정책                     |
| --------- | ---- | ----------------------------- |
| A         | P26  | **아무 lock 없음**      |
| B         | P26  | 현재 +50%, remaining≤5       |
| C         | P26  | rank-aware dynamic lock       |
| D         | P26  | estimated P(win)-optimal lock |

여기서 **A를 반드시 기준점**으로 놓으십시오.

현재 결과만 보면 P26에서 제일 좋은 부분은 **95% 집중 + mom60 alpha**, 제일 의심스러운 부분은 **house-money overlay**입니다.

---

# 5. 고정 +50%가 특히 위험한 이유

2025년은 정확히 8주 대회였고 최종 1위가 +47.82%, 2위가 +44.64%였습니다. 약 1000명이 참가했습니다. ([머니투데이][2])

그런데 중간 경로를 보면 완전히 다릅니다.

5주차에는 1위가 **+72.28%**, 4위도 +49.17%였습니다. ([머니투데이][3])

6주차에도 상위 5명이 각각 **65.63%, 59.73%, 55.01%, 52.50%, 50.96%**였습니다. ([머니투데이][4])

그런데 7주차 조정으로 1위가 다시 **46.99%**까지 내려갔습니다. ([머니투데이][5])

즉 실제 대회에서는

**“50%면 충분하다”도 틀리고, “현재 1등 65%를 무조건 넘어야 한다”도 틀립니다.**

경쟁자들의 향후 손실 가능성까지 같이 봐야 합니다.

따라서 lock은 절대 수익률이 아니라

$$
P(\text{win}\mid
R_t,\ rank_t,\ leaderboard_t,\ remaining_t,\ regime_t)
$$

으로 결정하는 게 맞습니다.

---

# 6. 오히려 repo에 이걸 구현하기 위한 흔적은 이미 있다

`AggressionInput`에는 이미

```python
delta
n
remaining
rank
```

가 있습니다.

그런데 현재 `risk_multiplier()`는 실제로 `remaining`과 `rank`를 사용하지 않습니다. 사실상

$$
1 + 2\delta - 0.001n
$$

이라는 단순식입니다.

여기가 **P27의 핵심 개발 포인트**가 되어야 합니다.

2026 대회는 대회 시작 후 ranking page도 운영한다고 공식 안내했습니다. ([머니투데이][6])

따라서 매일 상태를

```text
내 수익률
현재 순위
1위와 gap
TOP5 수익률 분포
남은 거래일
현재 market regime
현재 leader ETF의 momentum strength
```

로 잡고,

각 행동

```text
cash
현재 +2x 유지
다른 +2x leader로 switch
inverse
defensive / gold
```

에 대해 남은 기간 Monte Carlo rollout을 돌린 후

```text
action = argmax P(final_rank == 1)
```

로 선택하는 구조가 **대회 목적에 가장 정확하게 맞습니다.**

이건 현재 P26보다 한 단계 큰 개선입니다.

---

# 7. P26의 두 번째 구조적 약점: 사실상 long +2x 전용

P26은 강제로

```python
only_plus_2 = True
no_inverse = True
```

입니다.

더 정확히 보면 ETF 필터에서

```python
if config.only_plus_2 and lev != 2:
    continue
```

가 먼저 실행되므로 `no_inverse=False`만 바꿔도 -2x inverse는 들어오지 않습니다. 구조 자체를 바꿔야 합니다.

또 P26의

```text
cash_drawdown = 0
impulse_gap = 0
```

이므로 crash cash도 사실상 비활성, impulse switch도 비활성입니다. Sticky model의 실제 score는 `mom_60` + sticky가 중심입니다.

따라서 P26을 단순화하면:

> **최근 60일 동안 가장 강한 +2x ETF를 거의 올인해서 일정 기간 붙잡는 전략**

에 가깝습니다.

상승 추세가 이어지는 장에서는 강력합니다.

하지만 **대회 중간에 추세가 risk-off로 바뀌면 사용할 무기가 없습니다.**

2026 공식 규칙상 자율형은 레버리지·인버스 사용이 가능합니다. 현재 repo의 `tournament.yaml`에는 두 값이 아직 `unknown`으로 남아 있습니다.  공식 규칙은 자율형 외 부문에서만 레버리지·인버스를 제외한다고 명시합니다. ([머니투데이][1])

그리고 이건 단순 이론 문제가 아닙니다.

2025 최종 우승자는 실제로 레버리지와 인버스 ETF를 모두 거래했고 금 ETF도 활용했습니다. ([머니투데이][2])

조정장이 온 7주차에는 상위 참가자들이 `KODEX 200선물인버스2X` 한 종목에 집중해 순위를 끌어올리는 현상까지 나타났습니다. ([머니투데이][5])

따라서 **inverse branch는 반드시 실험할 가치가 있습니다.**

단, P26에 이것저것 섞지는 마십시오.

`P27-L`: 현재 P26 long-only
`P27-B`: ±2x signed momentum
`P27-R`: regime-gated long/inverse/cash

세 후보를 분리해서 비교하는 것이 좋습니다.

---

# 8. 통계적으로는 P26 숫자를 아직 많이 믿으면 안 된다

여기가 상당히 중요합니다.

코드의 `effective_sample_size()`는

```python
n_effective = n_windows // horizon
```

입니다. 36일 rolling window를 사용하므로 수천 개 rolling window가 있어도 실제 독립 정보량은 대략 수십 개 수준입니다.

그러므로 `P>50%=5.3%`라는 숫자는 실제로는 **독립적인 +50% 사건이 몇 건 안 되는 tail estimate**일 가능성이 높습니다.

stationary bootstrap을 쓰는 건 올바른 방향이지만, 더 큰 문제가 있습니다.

P20 → P21 → P22 → … → P26을 모두 **동일한 KRX 2018–2026 데이터의 결과를 보면서 계속 수정**했습니다.

결정 로그를 보면 실제로:

* P22에서 2025 우승수익률 47.82%를 참고해 lock level 변경
* P23에서도 `tour_20250922` 결과와 47.82% acceptance 기준 사용
* P24에서 mom60 선택
* P25에서 +50% lock
* P26에서 95%/1.90 exposure 변경

등이 반복됐습니다.

이건 execution look-ahead bug는 아닙니다.

하지만 **연구자 과적합 / multiple-testing bias**입니다.

따라서 2025 대회 구간에서 P26이 잘 나오는 것은 이제 OOS 증거로 취급하면 안 됩니다. 이미 그 결과를 보면서 모델을 만든 셈이기 때문입니다.

---

# 9. “실전 딱 한 번”이라는 질문을 검증하는 별도 테스트가 필요하다

현재 rolling 36일 분포는 유용하지만, 사용자가 걱정하는 것과 정확히 같은 질문은 아닙니다.

추가로 **Annual One-Shot Tournament Test**를 만드십시오.

예를 들면 매년:

```text
2018: 9월 셋째 주 → 36 sessions
2019: 9월 셋째 주 → 36 sessions
...
2025: 실제 9/22 → 11/14
```

딱 **1년에 1회**만 실행합니다.

각 해에:

| Year | Return | MDD | Peak | Giveback | #Trades | Main ETF | >30 | >40 | >50 |
| ---- | -----: | --: | ---: | -------: | ------: | -------- | --- | --- | --- |

를 생성하십시오.

샘플이 8개 남짓이라 통계검정용으로는 부족합니다.

하지만 사용자가 묻는

> “실제로 어느 날 시작해서 딱 36일 한 번 수행했을 때 어떤 일이 벌어지는가?”

를 rolling distribution보다 훨씬 직관적으로 보여줍니다.

**rolling distribution + annual one-shot**을 같이 봐야 합니다.

---

# 10. P27 개발 우선순위

지금부터는 알파 feature를 계속 추가하기보다는 아래 순서가 낫습니다.

1. **P26 RAW를 별도 champion baseline으로 고정.** 현재 +50/5 house-money overlay를 제거한 결과를 먼저 다시 계산합니다. P26의 championship FAIL이 raw 대비 overlay 열세에서 발생했으므로 최우선입니다.
2. **현재 threshold championship score 위에 실제 `P(win)` field objective를 구현.** 과거 TOP5 자료 + 경쟁 전략 agent pool로 `field_max_return`을 생성하고 `win_rate`, `top2_rate`, `rank percentile`을 계산합니다.
3. **실시간 rank-aware policy를 구현.** 현재 사용하지 않는 `rank`, `remaining`을 정책 state에 넣고, 고정 +50% lock을 Monte Carlo continuation 기반 행동 선택으로 교체합니다.
4. **inverse/risk-off branch를 독립 후보로 추가.** 단순히 `no_inverse=False`가 아니라 +2/-2 후보군을 구분한 signed/regime model로 구현합니다. 2025 실제 대회에서도 이 기능의 실전 가치가 확인됐습니다.
5. **1.90 gross도 상수가 아니라 최적화 대상 또는 action으로 전환.** 신호가 약한데 무조건 95% +2배를 들 이유는 없습니다. `score_gap × regime × rank × remaining`에 따라 1.2~1.9 사이를 선택하게 할 수 있습니다. 단 평균 Sharpe가 아니라 `P(win)` 기준으로 결정해야 합니다.
6. **selection-aware nested walk-forward를 다시 구성.** outer fold 안에서는 모델/파라미터 선택을 절대 하지 말고, inner fold에서 P26/P27 후보 선택까지 전부 수행한 뒤 outer에 딱 한 번 적용해야 합니다. 현재의 bootstrap CI만으로 P20~P26 반복 선택 bias가 없어지지는 않습니다.
7. **9월 17~18일 실제 HTS manifest를 받은 뒤 마지막 execution parity test.** 현재 `universe_manifest=null`이고 exact HTS 종목 리스트가 아직 확정되지 않았습니다.  이때만 최종 universe와 주문 체결 특성을 동결해야 합니다.

---

## 최종 판단

현재 저는 **P26 alpha 자체는 버리지 않겠습니다.**

오히려 다음 개발의 anchor로 두겠습니다.

다만 **`P26 = 최종 실전전략`으로 승격하지는 않겠습니다.**

특히 지금은:

$$
\boxed{
\text{새 알파 탐색}
<
\text{목적함수 수정}
+
\text{fixed lock 제거/개선}
+
\text{rank-aware control}
+
\text{inverse branch}
+
\text{selection-aware validation}
}
$$

순으로 중요합니다.

한 문장으로 정리하면:

> **현재 P26은 “가끔 우승권 수익률을 만드는 전략”까지는 왔지만, “1000명 안팎의 경쟁자 중 단 한 번의 36일 실전에서 1등 할 확률을 최대화한 전략”은 아직 아닙니다.**

특히 repo에 기록된 `P>50%=5.3%`는 분명 긍정적인 신호지만, **우승확률 5.3%가 아니며**, 현재 최종 championship gate가 FAIL이라는 점을 무시하면 안 됩니다. 다음 P27은 알파를 한두 개 더 붙이는 버전이 아니라 **`True Win-Probability Objective + Dynamic Tournament Policy`**로 잡는 것을 권합니다.

원한다면 다음 답변에서는 이 분석을 바탕으로 **P27 개발 명세를 코드 수준으로 설계해서, AI 코딩 에이전트에게 그대로 전달할 수 있는 구현 지시서**까지 만들어드릴 수 있습니다.

대회 직전에는 HTS 종목 목록과 규칙 확정 여부를 다시 확인하는 것이 필요합니다.

[1]: https://www.mt.co.kr/etf/join/index.html?utm_source=chatgpt.com
[2]: https://www.mt.co.kr/stock/2025/11/19/2025111814504029132?utm_source=chatgpt.com
[3]: https://www.mt.co.kr/stock/2025/10/25/2025102418201958573?utm_source=chatgpt.com
[4]: https://www.mt.co.kr/stock/2025/11/02/2025110210422182221?utm_source=chatgpt.com
[5]: https://www.mt.co.kr/stock/2025/11/09/2025110718053119988?utm_source=chatgpt.com
[6]: https://www.mt.co.kr/amp/stock/2026/08/14/2026081416135978575?utm_source=chatgpt.com
