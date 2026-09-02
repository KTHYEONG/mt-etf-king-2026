최신 push까지 다시 확인했습니다. 현재 HEAD 근처는 2026년 9월 2일 13:09 KST경 올라온 `be8e72c`(엔진 개선), `2b43e33`(테스트 보강), `7e87361`(문서 갱신)입니다. 이번 수정으로 이전에 지적했던 execution 쪽 P0 문제는 대부분 제대로 닫혔습니다.

## 1. 현재 판정: 이제 알파 개선 단계로 넘어가도 된다

`TASK_37_EXECUTION_CORRECTNESS` 기준으로 next-open causal fill, 대회 첫날 시가 진입을 위한 pre-start signal, open 기준 turnover/ADV, post-fill gross와 가격상승에 의한 realized gross 분리, ledger 단일 실행경로까지 반영됐습니다. 새 P27-R rebaseline에서도 `execution_parity=True`가 확인됐습니다.

현재 제가 보는 상태는 이렇습니다.

| 영역                          |        현재 상태 | 판단                         |
| --------------------------- | -----------: | -------------------------- |
| Next-open P&L               |           좋음 | 치명적 오류 해결                  |
| 첫날 진입                       |           좋음 | 직전일 signal→첫날 open 구현      |
| ADV causality               |           좋음 | decision-date 기준           |
| HOLD/CASH/TARGET            |           좋음 | ledger 통합                  |
| Weight drift                |           좋음 | shares+cash ledger         |
| Full/Fast parity            |           좋음 | 구조적으로 동일 transition        |
| Open fill look-ahead        |           해결 | `is_tradable` 제거, open만 판정 |
| Gross violation             |           해결 | post-fill 기준               |
| Trade source                |           해결 | transition fills 기준        |
| **P27 championship wiring** | **잔여 버그 있음** | 즉시 수정                      |
| 통계적 신뢰도                     |          미완성 | 이제 만들어야 함                  |
| P27 우승 전략성                  |        아직 부족 | 구조적 확장 필요                  |

`NextOpenExecution`도 이제 execution-day의 `is_tradable`이나 종가 정보를 보지 않고 실제 open 가격의 존재/유효성만으로 체결 가능 여부를 판단합니다.

ledger 역시 `weights_before_close`와 `weights_before_open`을 분리하고 ADV cap·turnover를 open 비중으로 계산하며, gross도 `target_gross / post_fill_gross / close_realized_gross`로 분리했습니다.

따라서 **이제 버그를 끝없이 찾는 단계에서 빠져나와도 됩니다.**

---

# 2. 다만 바로 하나 고칠 것: `GROSS_METRIC_UNAVAILABLE`

이건 전략 문제가 아니라 CLI wiring 문제입니다.

fast rolling 자체에서는 이미 각 window의 실제 `SessionTransitionDiagnostics`를 수집한 다음 `aggregate_session_diagnostics()`로 `gross_violation_count`, `effective_gross_max`, turnover, fills, unfilled를 계산합니다.

그런데 P27 CLI에는:

```python
_diag_p27 = getattr(rolling, "diagnostics", None)
_ = _diag_p27
_ = diagnostics
```

가 있습니다.

`diagnostics`라는 이름이 이 scope에서 정의되지 않은 상태라면 여기서 예외가 나고 바로:

```python
except Exception:
    gross_viol_p27 = None
```

으로 떨어집니다. 실제로 Task37 로그의 `championship gate still GROSS_METRIC_UNAVAILABLE`와 정확히 일치합니다.

즉 다음 push에서 이 부분을 고친 다음 **P27-R을 한 번 완전히 rebaseline**하십시오.

그 이후 과거의 4.8%, 5.3% 같은 P50 숫자는 모두 폐기하고 새로운 숫자만 사용해야 합니다.

---

# 3. 최신 P27-R 성과에서 지금 읽을 수 있는 것은 많지 않다

Task37에 현재 확인된 건:

$$
median\ 36d\ return \approx +0.12\%
$$

$$
P(R>30\%)\approx11.1\%
$$

그리고 `execution_parity=True`입니다.

여기서 median +0.12%는 문제라고 볼 필요가 없습니다.

이 전략의 목적은 median을 높이는 것이 아니라:

$$
P(R>40\%),\quad P(R>50\%),\quad P(R>60\%)
$$

입니다.

전체 수익률 1위가 대상이라는 2026 공식 기준도 그대로입니다. 자율형은 투자자산 제한이 없고, 레버리지·인버스 제외 규정은 자율형 외 부문에 적용됩니다. ([머니투데이][1])

따라서 P27을 일반적인 Sharpe 전략으로 바꾸면 오히려 목표에서 멀어집니다.

---

# 4. 현재 P27의 진짜 구조적 약점은 두 개다

현재 P27은 여전히 사실상:

$$
\boxed{
+2x\ long\ only
+
mom60\ top1
+
95\%
+
sticky(4\%,2d)
}
$$

입니다. 설정에도 cash와 inverse는 꺼져 있습니다.

그리고 `filter_plus2_scores()`는 `mom60 > 0` 조건을 요구하지 않습니다. 유효한 +2 ETF라면 momentum이 -30%여도 score에 들어갑니다.

즉 현재 P27은 두 가지 행동을 합니다.

**첫째, 모든 long 2x가 하락 추세여도 반드시 하나를 삽니다.**

**둘째, 같은 leader가 계속 유지되면 매일 다시 95% target을 만들 가능성이 높습니다.**

두 번째가 생각보다 중요합니다.

---

# 5. 성과개선 1순위: `same leader = HOLD`

현재 가장 먼저 테스트할 후보입니다.

예를 들어 처음 95%를 샀다고 합시다.

ETF가 상승하면 실제 비중은:

$$
95\%\rightarrow95.5\%\rightarrow96\%
$$

처럼 올라갑니다.

지금 P27이 다음 날 다시 95%를 target으로 내면 계속 일부를 팔게 됩니다.

반대로 ETF가 하락해서:

$$
95\%\rightarrow94\%
$$

가 되면 다시 95%로 올리면서 추가매수합니다.

이건 position-management 관점에서 사실상:

> **winner는 조금씩 팔고 loser는 조금씩 더 사는 constant-weight 정책**

입니다.

momentum 전략의 철학과 반대입니다.

### P28-A는 이것만 바꿉니다

$$
leader_t=held_t
\Rightarrow HOLD
$$

새 leader가 등장해 sticky switch 조건을 만족할 때만:

$$
TARGET(95\%,new\ leader)
$$

합니다.

따라서 주문은 실질적으로:

```text
최초 진입
leader switch
cash exit/reentry
방향 전환
```

시에만 발생합니다.

이 후보에는 **새 파라미터가 하나도 없습니다.**

제가 보기에는 현재 가장 가치가 높은 첫 A/B test입니다.

특히 이것이 좋아지면:

$$
turnover\downarrow
$$

$$
cost\downarrow
$$

$$
winner\ convexity\uparrow
$$

가 동시에 가능합니다.

---

# 6. 성과개선 2순위: `absolute momentum → cash`

그다음은 forced-long 문제를 제거합니다.

P28-B를:

$$
M_t=\max_{i\in +2x} mom60_i
$$

라고 할 때,

$$
M_t\le0
\Rightarrow CASH
$$

$$
M_t>0
\Rightarrow 기존\ P27
$$

로 만드십시오.

threshold를 3%, 5%, 7%, 10% 식으로 최적화하지 마십시오.

**0 하나만 primary candidate로 사전 고정**하는 것이 좋습니다.

경제적 의미도 명확합니다.

> 지난 60일 동안 상승한 long leveraged ETF가 하나도 없다면 long 2x에 우승 edge가 있다고 가정하지 않는다.

이 실험의 목적은 평균수익 개선이 아닙니다.

좋은 결과는:

$$
ruin\downarrow,\ CVaR\ 개선
$$

하면서도

$$
P50,\ P60
$$

이 거의 손상되지 않는 것입니다.

만약 downside는 좋아지지만 P60이 크게 줄면 대회용으로는 버려야 합니다.

---

# 7. 성과개선 3순위가 가장 큰 구조적 확장: signed ±2x

공식 2026 규정상 자율형은 투자자산 제한이 없고, 자율형 외 부문에 레버리지·인버스 제외가 명시돼 있습니다. 따라서 실제 HTS manifest에서도 해당 상품이 제공되는지만 확인되면, inverse 연구는 규정 방향과 맞습니다. ([머니투데이][1])

현재 repo의:

```yaml
leverage_allowed: unknown
inverse_allowed: unknown
```

은 공개 규정 기준으로는 이제 너무 보수적인 상태입니다.

연구 config에서는 최소한:

```text
autonomous:
leverage_allowed = true
inverse_allowed = true
```

로 명시하고, 실제 대회 계정이 9월 17~18일경 배포되면 **HTS manifest를 최종 authority**로 덮는 구조가 좋습니다. 공식 안내도 후원 운용사 ETF로 거래 대상을 제한한다고 밝히고 있습니다. ([머니투데이][1])

### P28-C

후보군을:

$$
leverage\ multiple\in\{+2,-2\}
$$

로 만듭니다.

그리고:

$$
i^*=\arg\max_i mom60_i
$$

$$
mom60_{i^*}>0
\Rightarrow i^*\ 매수
$$

$$
mom60_{i^*}\le0
\Rightarrow CASH
$$

입니다.

sticky는 그대로:

$$
min\_gap=4\%,\quad min\_hold=2
$$

를 유지합니다.

별도의:

```text
inverse gap
long gap
direction-switch gap
inverse hold
```

같은 파라미터는 만들지 마십시오.

그 순간부터 다시 과적합 영역으로 들어갑니다.

---

# 8. 이 signed 전략이 중요한 이유

P27의 winning regime은 사실상 하나입니다.

$$
강한\ 상승장
$$

P28-B는:

$$
상승장\rightarrow +2x
$$

$$
하락장\rightarrow cash
$$

입니다.

P28-C는:

$$
상승장\rightarrow+2x
$$

$$
하락장\rightarrow-2x
$$

$$
애매함\rightarrow cash
$$

가 됩니다.

즉 **우승 가능한 시장 상태의 면적 자체가 넓어집니다.**

한 번뿐인 대회에는 이게 매우 중요합니다.

평균수익을 조금 올리는 것보다:

> 어떤 방향으로 큰 trend가 발생해도 그것을 2x vehicle로 잡을 수 있음

이 훨씬 우승 목적에 맞습니다.

---

# 9. 2026에는 한 가지 더 중요하다: 단일종목 레버리지 ETF

2026년에는 삼성전자·SK하이닉스 등을 기초로 한 단일종목 레버리지·인버스 ETF 시장이 새로 생겼고, 최근 규제 강화 후 거래대금은 크게 감소했지만 관련 16종의 하루 거래대금이 8월 26일 약 7110억원 수준으로 보도됐습니다. ([머니투데이][2])

이건 P27 historical backtest에 매우 중요한 구조변화입니다.

2018~2025의 universe에는 이런 종류의 convex vehicle이 없었습니다.

따라서:

$$
Historical\ P50(P27)
$$

가 2026 실제 P27의 potential을 완전히 나타낸다고 볼 수 없습니다.

특히 삼성전자/하이닉스 같은 단일종목 +2x가 실제 후원사·HTS universe에 포함된다면 36일 대회의 right tail은 크게 달라질 수 있습니다.

### 하지만 실제 ETF의 짧은 history로 parameter를 맞추면 안 된다

좋은 방법은 별도의 **structural proxy study**입니다.

예를 들어 과거 삼성전자/하이닉스 일간 수익률로:

$$
r^{synthetic}_{2x,t}\approx2r_{stock,t}
$$

를 만들어,

> 단일종목 +2x가 historical opportunity set에 있었다면 mom60 selection의 우승 tail이 얼마나 늘어났는가?

만 diagnostic으로 봅니다.

이를 P27의 공식 historical backtest에 섞지는 않습니다.

즉:

**deployment evidence**와 **structural opportunity evidence**를 분리합니다.

---

# 10. 그런데 전략을 더 만들기 전에 먼저 해야 할 분석이 있다

바로 **Oracle Gap Decomposition**입니다.

각 historical 36-session window에 대해 hindsight로 다음을 계산합니다.

$$
O_L(w)=\max_{i\in+2x}R_{i,w}
$$

$$
O_S(w)=\max_{i\in-2x}R_{i,w}
$$

$$
O(w)=\max(O_L,O_S,0)
$$

그리고:

$$
Gap(w)=O(w)-R_{P27}(w)
$$

입니다.

그러면 P27이 왜 우승권을 놓치는지 정확히 분해할 수 있습니다.

| 상황                    | 원인                  | 다음 개선                 |
| --------------------- | ------------------- | --------------------- |
| Long oracle도 낮음       | 시장에 기회 없음           | 전략 건드리지 않음            |
| Long oracle 높고 P27 낮음 | selection timing 문제 | mom horizon/leader 개선 |
| P27 peak 높고 final 낮음  | giveback            | exit 연구               |
| Inverse oracle만 높음    | direction 문제        | signed ±2x            |
| 모든 momentum ≤0        | forced-long 문제      | cash gate             |
| P27≈oracle            | 이미 충분               | 복잡화 금지                |

이 결과가 **P29 방향을 결정해야 합니다.**

미리 RSI, volatility, breadth, flow 등을 붙이는 것은 하지 않는 게 좋습니다.

---

# 11. P27을 믿으려면 tail 숫자보다 tail의 “출처”를 검증해야 한다

2090개의 36일 rolling window가 있어도 대부분 서로 35일씩 겹칩니다.

따라서 P50이 5%라고 해도 실제론 하나의 강한 rally가 수십 개 window를 만든 것일 수 있습니다.

최종 P27-R report에는 반드시:

$$
P30,P40,P50,P60,P80
$$

뿐 아니라 다음도 있어야 합니다.

```text
독립 tail episode 수
연도별 episode 분산
top-1 episode contribution
top-3 episode contribution
dominant ticker/family share
long vs inverse opportunity
```

특히:

> +50% window의 70%가 동일한 한 rally에서 생성

이라면 P50=5%라는 숫자를 실전 probability로 믿으면 안 됩니다.

---

# 12. 이제 P20→P27식 개발 방법도 끝내야 한다

이미 동일 데이터에서 많은 iteration을 했습니다.

앞으로는 후보를 계속 만들수록 backtest winner는 좋아지지만 실제 우승확률은 오히려 불확실해집니다.

그래서 다음 후보까지만 미리 고정하는 것을 권합니다.

| 모델        | 변경점                 | 새 자유도 |
| --------- | ------------------- | ----: |
| **P27-R** | corrected baseline  |     0 |
| **P28-A** | same leader → HOLD  |     0 |
| **P28-B** | HOLD + mom60≤0 cash |     0 |
| **P28-C** | HOLD + ±2x + cash   |  거의 0 |

**이 네 개의 결과를 보기 전에는 P29를 만들지 마십시오.**

이것이 지금 과적합을 줄이는 가장 효과적인 방법입니다.

---

# 13. 그리고 P28의 promotion comparator는 P21이 아니라 P27-R이어야 한다

현재 P27 championship code는:

```python
candidate_p27 = rolling.returns
raw_p27 = rolling.returns
```

입니다. 즉 candidate vs raw는 동일합니다.

P27 자체를 identity baseline으로 인증할 때는 괜찮지만, 앞으로 P28을 선택할 때는 아무 의미가 없습니다.

앞으로는 명시적으로:

$$
candidate=P28
$$

$$
champion=P27\text{-}R
$$

$$
anchor=P21
$$

로 분리해야 합니다.

즉 P21은 역사적 anchor일 뿐이고 **실제 promotion opponent는 바로 직전 champion**이어야 합니다.

---

# 14. 제가 권하는 P28 채택 기준

단순히:

$$
ChampScore(P28)>ChampScore(P27)
$$

이면 안 됩니다.

가장 중요한 것은 같은 시작 window에서 pairwise 차이를 보는 것입니다.

$$
\Delta_w=
U(R_{P28,w})-U(R_{P27,w})
$$

그리고 stationary/block bootstrap으로:

$$
LCB_{95\%}
(E[\Delta])
$$

를 봅니다.

**LCB가 0 이상**이면 강한 승격 근거입니다.

추가로 반드시:

| 검증                     | 요구                      |
| ---------------------- | ----------------------- |
| Ruin                   | ≤5%                     |
| P50/P60                | P27-R 대비 심각한 열화 없음      |
| Tail episodes          | 특정 1 episode 의존 금지      |
| Phase robustness       | 특정 시작일 offset만 잘하지 않음   |
| Era                    | 특정 연도/국면 collapse 없음    |
| Parameter neighborhood | mom60만 needle peak이면 탈락 |
| Cost stress            | 비용 증가해도 상대우위 유지         |
| 1-day delay stress     | 완전히 붕괴하지 않음             |
| Remove-top-family      | 특정 ETF 하나 제거해도 전략 존속    |

이 정도를 통과해야 **“backtest에서 좋은 P28”**이 아니라 **“한 번의 실전에 맡길 수 있는 P28”**이 됩니다.

---

# 15. momentum horizon은 나중에 이렇게만 검증한다

P28-A/B/C가 정해진 뒤:

$$
40,50,60,70,80
$$

정도만 검사합니다.

목적은 optimum을 찾는 게 아니라 plateau를 확인하는 겁니다.

좋음:

$$
J_{50}=0.058,\quad
J_{60}=0.061,\quad
J_{70}=0.059
$$

나쁨:

$$
J_{50}=0.032,\quad
J_{60}=0.061,\quad
J_{70}=0.030
$$

후자는 mom60이라는 특정 historical accident에 맞았을 가능성이 큽니다.

**60이 최고인지보다 60 주변도 살아있는지가 훨씬 중요합니다.**

---

# 16. 실전 우승에는 core alpha 이후 별도 단계가 하나 더 필요하다

최종적으로 가장 강한 core가 P28-C라고 가정해도 대회 마지막에는 일반 퀀트와 목표가 달라집니다.

예를 들어 남은 3일:

$$
내\ 수익=+35\%
$$

$$
1위=+50\%
$$

이면 정상적인 risk management보다 더 공격적이어야 합니다.

반대로:

$$
내\ 수익=+55\%
$$

$$
2위=+35\%
$$

이면 굳이 full convexity를 유지할 이유가 줄어듭니다.

따라서 최종 architecture는:

$$
\boxed{
Core\ Market\ Strategy
+
Late\ Tournament\ Policy
}
$$

여야 합니다.

하지만 이건 **P28 core가 확정된 뒤** 하십시오.

P26처럼 단순히 "+50%면 cash"로 만들면 right tail을 잘라버린다는 것을 이미 경험했습니다.

---

# 17. Late Tournament Policy는 `수익률`이 아니라 `gap × remaining`으로 만든다

상태를:

$$
S_t=
(
R_{self},
R_{leader},
rank,
remaining
)
$$

으로 둡니다.

그리고 역사적으로 남은 \(d\)일 동안 achievable return의 분포:

$$
F_d(r)
$$

를 구합니다.

내가 1위이고 2위와의 gap이 역사적으로 남은 기간의 극단적 추격 가능성보다 충분히 크다면 방어합니다.

반대로 뒤지고 있으면 required return:

$$
R_{needed}
=
\frac{1+R_{leader}}{1+R_{self}}-1
$$

와 남은 기간을 비교해 convexity를 유지하거나 확대합니다.

이게 우승 목적에 가장 자연스럽습니다.

---

# 18. 현재 config에서 바로 수정해야 할 부분도 하나 있다

공식 2026 안내는 자율형을 **투자자산 제한 없음**으로 설명하고, 자율형 외 부문에 레버리지·인버스 제외를 명시합니다. ([머니투데이][1])

그런데 현재 repo는 여전히:

```yaml
category:
  name: autonomous
  leverage_allowed: unknown
  inverse_allowed: unknown
```

입니다.

연구 단계에서는 공식 규정 근거로 `true/true`를 사용하고,

최종적으로 9월 17~18일경 대회 계정과 프로그램이 오면:

```text
actual HTS ETF list
actual order unit
actual fill mechanics
```

를 `universe_manifest.yaml`로 고정하는 게 맞습니다.

**실전 freeze의 마지막 authority는 실제 HTS입니다.**

---

# 19. 앞으로의 정확한 개발 순서

1. **TASK_38 — P27-R Final Certification**
   `gross_viol_p27` CLI wiring을 고치고 P27-R 전체 rebaseline. 기존 성과는 모두 stale 처리합니다.

2. **TASK_39 — Tail Forensics**
   P30/P40/P50/P60/P80, bootstrap CI, independent tail episodes, phase robustness, oracle long/inverse gap을 생성합니다.

3. **TASK_40 — P28-A Same-Leader HOLD**
   leader가 같으면 `HOLD_INTENT`. P27과 leader selection 자체는 동일해야 합니다. turnover/cost/right-tail 차이만 측정합니다.

4. **TASK_41 — P28-B Absolute Momentum Cash**
   `max(+2 mom60)<=0 → CASH`. 다른 모든 parameter 동결.

5. **TASK_42 — P28-C Signed Convexity**
   `abs(leverage_multiple)==2`, 최고 mom60>0 선택, 아니면 cash. same leader HOLD, gap4%, hold2 유지.

6. **TASK_43 — Selection-Aware Tournament**
   P27-R/P28-A/B/C만 동일 protocol에서 비교. paired championship utility, tail episodes, phase/era/cost/delay stress로 champion 하나를 선택합니다.

7. **TASK_44 — 2026 Universe Expansion Audit**
   실제 HTS manifest와 신규 단일종목 leveraged/inverse ETF를 검증하고, 짧은 history는 별도 structural proxy 연구로만 평가합니다.

8. **TASK_45 — Late Tournament Controller**
   최종 core는 손대지 않고 마지막 구간에만 leaderboard gap × remaining-session 정책을 연구합니다.

이 순서를 권합니다.

---

# 최종 방향

현재까지의 연구에서 P27의 핵심 아이디어 자체를 버릴 이유는 없습니다.

오히려 가장 중요한 구성은 남겨야 합니다.

$$
\boxed{
medium\ trend
+
high\ convexity
+
concentration
+
hysteresis
}
$$

문제는 지금 P27이:

$$
\boxed{
long\ only
+
forced\ exposure
+
daily\ 95\%\ retarget
}
$$

라는 세 가지 제약을 갖고 있다는 점입니다.

제가 보는 가장 유력한 최종 전략은 다음 형태입니다.

$$
\boxed{
\text{60d directional momentum}
\rightarrow
\text{best } \pm2x
\rightarrow
\text{positive edge gate}
\rightarrow
\text{same leader HOLD}
\rightarrow
\text{95\% initial concentration}
\rightarrow
\text{sticky switch}
\rightarrow
\text{late tournament controller}
}
$$

이 방향은 P27을 복잡하게 만드는 것이 아니라 **P27의 우승 논리를 더 순수하게 만드는 방향**입니다.

현재 기준으로는 **P28-A `same leader → HOLD`가 첫 번째 성과 실험**, **P28-C signed ±2x가 가장 큰 구조적 우승확률 확장**이라고 판단합니다. 다만 그 전에 `GROSS_METRIC_UNAVAILABLE` wiring을 고친 **최종 P27-R baseline 숫자**를 한 번 확보해야 합니다. 그 결과가 나오면 그다음부터는 실제 수치를 기준으로 P28-A/B/C 중 어떤 구조가 가장 유망한지 훨씬 강하게 판단할 수 있습니다.

원하면 이후 push마다 이 기준으로 P27/P28의 승격 여부를 다시 검증할 수 있습니다.

[1]: https://www.mt.co.kr/etf/join/index.html?utm_source=chatgpt.com "대회 참여 신청 | ETF 투자왕대회 - 머니투데이"
[2]: https://www.mt.co.kr/stock/2026/08/27/2026082713291660323?utm_source=chatgpt.com "12조→0.7조, '단일레버리지' 광풍 어떻게 꺼졌나 - 머니투데이"
