현재 저장소의 `latest.md`, 전략/실행/토너먼트 코드, 최근 작업 이력, 그리고 2026년 대회 공식 규칙과 2025년 실제 우승 패턴까지 같이 보면 결론은 꽤 명확합니다.

## 결론

**현재 P27은 “우승권 전략”이라기보다 “우측 꼬리가 존재하는 고변동성 후보”입니다. 아직 실전 champion으로 확정하면 안 됩니다.**

가장 중요한 이유는 다음입니다.

| 항목                |                   P27 현재 상태 | 해석                         |
| ----------------- | --------------------------: | -------------------------- |
| `P(R>50%)`        |                       4.69% | 우승권 proxy는 존재하지만 낮음        |
| q50               |                      +0.12% | 평범한 36일 구간은 사실상 수익 없음      |
| q95               |                      +48.6% | 상위 5%에서만 우승권               |
| q99               |                      +86.9% | right-tail은 강함             |
| `P(R<-25%)`       |                       4.21% | 우승권 발생확률과 파산성 tail이 비슷한 수준 |
| CVaR 5%           |                      -37.2% | 실패 시 손실 매우 큼               |
| gross violation   |                  **1,958회** | 현재 가장 큰 blocker            |
| championship gate |                    **FAIL** | 아직 strict champion 아님      |
| 9월 anchor         | 2018~24 대부분 부진, 2025 +41.4% | 2025 regime 의존 가능성 큼       |

특히 2018~2025의 실제 9월 시작 36-session replay에서 2025만 +41.4%로 크게 튀고, 2018 -31.6%, 2019 -3.4%, 2021 -1.2%, 2023 -6.1%, 2024 -12.3%입니다. 즉 **현재 성과는 “일관된 대회형 알파”보다는 특정 강세장 regime에서 폭발하는 구조에 가깝습니다.**

그리고 P26은 `P>50=5.31%`, `P>60=4.55%`, gross violation=0으로 raw right-tail만 보면 오히려 더 좋습니다. 다만 기존 평가에서 hot-field/raw 비교와 CI 조건을 통과하지 못했기 때문에 역시 champion으로 볼 수 없습니다.

따라서 지금부터는 **P27 파라미터 미세조정을 계속하는 것보다 아래 순서로 개발하는 것이 맞습니다.**

---

# 1. P0: 새 전략 개발보다 gross/execution correctness를 먼저 끝내야 함

제가 코드에서 가장 우선적으로 고칠 부분입니다.

`transition_portfolio_state()`의 흐름은 대략

`exposure cap → ADV cap → execution → post-fill gross 진단`

순서입니다. 문제는 ADV cap이 기존 포지션의 청산을 일부 제한하면서 신규 포지션 진입은 허용할 수 있다는 점입니다. 그러면 원래 target은 gross 1.9 이하였더라도 **실제 체결 후 old + new가 겹쳐 gross가 1.9를 넘을 수 있습니다.** 현재 코드는 이걸 사후 `gross_violation=True`로 발견하지만 다시 강제 수정하지는 않습니다.

`cap_target_weights_by_adv()` 역시 `target ∪ current`에 대해 각각 delta를 제한하기 때문에 이런 switch-overlap이 발생 가능한 구조입니다.

이것이 P27의 1,958건 전부의 원인이라고 현재 artifact만으로 단정할 수는 없습니다. 문제는 **P27 최신 adoption run에는 `windows.parquet`가 저장되어 있지 않아 실제 violation session을 forensic 분석할 수도 없다는 것**입니다.

따라서 다음 순서가 최우선입니다.

1. P27을 `windows + daily + trades + diagnostics` 전부 저장해서 다시 실행.
2. gross violation마다 `current → requested target → ADV constrained target → fill/unfilled → post-fill`을 저장.
3. **SELL/UNWIND 우선 → 실제 남은 gross budget 계산 → BUY**의 2단계 execution으로 변경.
4. 기존 종목 청산이 안 되면 신규 종목 buy를 축소하거나 취소.
5. `post_fill_gross <= max_gross`를 invariant로 강제.
6. 자연스러운 가격 상승에 의한 `close_realized_gross` drift와 실제 잘못된 신규 체결에 의한 violation을 별도 metric으로 분리.

현재 HOLD에서는 가격 drift에 의한 gross 초과를 violation에서 제외하도록 수정해 놓았는데, 이 방향 자체는 타당합니다. 다만 앞으로는

* `execution_gross_violation`
* `carry_gross_drift`
* `delever_required_next_session`

세 가지를 분리하는 편이 좋습니다.

**이게 0이 되기 전에는 P29 같은 새 alpha를 만드는 것을 중단하는 것을 권합니다.**

최근 TASK_33~39에서도 실행 시점, 첫 session 진입, HOLD 원장 등의 correction이 들어갈 때마다 P27 수치가 상당히 바뀌었습니다. 이는 현재 가장 큰 model risk가 alpha가 아니라 simulator였다는 의미입니다.

---

# 2. 현재 P27의 핵심 한계: 60일 모멘텀 + 2x long에 지나치게 고정되어 있음

P27 factory를 보면 실질적으로 다음이 강제됩니다.

```text
mom_col       = mom_60
min_gap       = 0.04
min_hold      = 2
cash_drawdown = 0
impulse_gap   = 0
only_plus_2   = True
no_inverse    = True
```

게다가 config를 읽더라도 마지막에 다시 같은 값을 강제로 덮어씁니다.

그리고 `filter_plus2_scores()`가 **2배 레버리지 ETF의 `mom_60` 자체를 직접 비교**합니다. inverse는 완전히 제거합니다.

이 방식은 2025년 강한 상승 추세에는 매우 잘 맞았습니다. 실제 2025년 우승자는 상승장에서 레버리지 ETF에 집중했고, 강한 반도체 sector에 집중 투자했습니다. 하지만 시장이 꺾이자 인버스와 금 ETF로 전환해 최종 +47.82%를 기록했습니다. ([머니투데이][1])

2025년 첫 주 상위권도 레버리지·인버스를 적극 이용했고, 5주차에는 전체 1위가 한때 **+72.28%**까지 올라갔습니다. 이후 조정으로 최종 우승 수익률은 +47.82%까지 내려왔습니다. ([머니투데이][2])

즉 실제 대회가 요구하는 것은 단순한:

> `best 2x long momentum ETF`

보다는

> **`best theme selection + regime direction + concentration + exit/re-entry`**

에 가깝습니다.

---

# 3. 다음 champion은 P27 개선판이 아니라 3-layer 구조로 만드는 것이 좋음

제가 다음 전략의 구조를 설계한다면 이렇게 합니다.

### Layer A — Alpha: “어느 테마가 강한가?”

여기서는 **레버리지 ETF 가격 자체가 아니라 underlying/family 수준을 score**하는 것이 좋습니다.

현재 프로젝트에는 이미 이를 위한 상당한 인프라가 있습니다. `strategies.yaml`에도 leadership score가

* RS 0.45
* acceleration 0.30
* breadth 0.25

로 정의되어 있습니다.

P27은 이 정보를 거의 사용하지 않고 `mom_60` 한 가지에 집중합니다.

다음 candidate는 예를 들어:

$$
Score =
w_1 Rank(Mom60)+
w_2 Rank(Mom20)+
w_3 Acceleration+
w_4 Breadth+
w_5 TrendQuality
$$

정도의 **소수 factor ensemble**이 더 적합합니다.

핵심은 최적의 `mom_43` 같은 lookback을 찾는 것이 아닙니다.

`20 / 40 / 60` 같은 몇 개 horizon을 rank ensemble하여 **lookback parameter sensitivity를 줄이는 것**이 중요합니다.

특히 36-session 대회인데 60일 momentum 하나만 보면 신규 leadership 전환에 늦을 가능성이 있습니다.

---

### Layer B — Vehicle: “그 view를 어떤 ETF로 표현할 것인가?”

이건 alpha와 분리해야 합니다.

현재 `src/portfolio/exposure.py`에는 이미 regime에 따라 `+2 / +1 / -1` vehicle을 선택하는 기반 코드가 있습니다.

그런데 champion P27은 처음부터 `+2 ETF`만 score하기 때문에 이 좋은 architecture를 활용하지 못합니다.

제가 바꿀 구조는:

```text
Family/Underlying Alpha
        ↓
Top Theme
        ↓
Regime Controller
 ├─ STRONG_RISK_ON → +2x
 ├─ RISK_ON        → +2x / +1x
 ├─ NEUTRAL        → +1x / cash
 ├─ RISK_OFF       → inverse / gold
 └─ STRONG_RISK_OFF→ inverse
```

입니다.

이렇게 하면 **“반도체가 강하다”와 “반도체 2배 ETF를 사야 한다”가 분리**됩니다.

이는 robustness 측면에서도 중요합니다. 2배 ETF의 60일 수익률은 underlying direction 외에도 daily compounding과 volatility drag가 섞이기 때문입니다.

---

### Layer C — Tournament controller: “현재 순위에서 얼마나 위험을 져야 하는가?”

이 부분이 현재 프로젝트에서 가장 큰 추가 alpha가 될 가능성이 있습니다.

2026년 공식 규정상 대회는 9월 21일~11월 13일 8주이고, **전체 최종 수익률 1위가 대상**입니다. 자율형은 투자자산 제한이 없고, 레버리지·인버스 제외 조건은 자율형 이외 부문에 적용됩니다. 단 실제 거래 가능 종목은 후원 운용사 ETF로 제한됩니다. ([머니투데이][3])

따라서 최적 목적함수는 사실

$$
\max P(R > 50\%)
$$

도 아니고

$$
\max E[R]
$$

도 아닙니다.

가능하다면 최종적으로는

$$
\boxed{\max P(rank=1)}
$$

이어야 합니다.

현재 `championship_score`의 30/40/50/60% exceedance weighting은 좋은 proxy지만, **실제 field distribution을 모델링하지 않습니다.**

특히 latest의 P27 `field_win_rate=56.8%`는 사실상 P21 incumbent와의 비교입니다. 실제 1,000명 수준의 참가자를 상대로 한 56.8% 승률이라는 뜻이 아닙니다. 2025년에는 약 1,000명이 참가했습니다.  ([머니투데이][1])

그래서 대회가 시작되면 `rank-aware controller`를 넣는 것이 좋습니다.

예:

```text
현재 수익률
현재 전체 1위/5위/10위 수익률
남은 session
현재 peak에서 giveback
market regime
```

를 state로 받아서,

```text
뒤처짐 + 많이 남음       → normal aggressive
뒤처짐 + 적게 남음       → maximum convexity
선두권 + 많이 남음       → trend continuation
선두권 + 적게 남음       → giveback control
압도적 선두              → capital preservation
```

처럼 risk budget을 변경합니다.

**고정 `lock@50%`보다 이게 합리적입니다.**

2025년에 5주차 선두가 +72.28%였지만 최종 우승은 +47.82%였다는 것 자체가 왜 fixed absolute threshold보다 leaderboard-relative control이 필요한지 보여줍니다. ([머니투데이][4])

---

# 4. 전략 탐색 방식도 바꿔야 함

현재 가장 위험한 것은 **P20 → P21 → ... → P28 식으로 같은 2018~2026 데이터에서 반복 개선하는 과정 자체**입니다.

이 과정을 충분히 오래 반복하면 실제 alpha가 없어도 backtest tail이 좋아지는 candidate를 결국 찾게 됩니다.

특히 `n_windows=2,090`처럼 보여도 overlapping 36-day windows 때문에 프로젝트 자체가 계산한 `n_effective`는 겨우 **58**입니다.

따라서 `P>50=4.69% vs 5.31%` 차이는 생각보다 훨씬 불확실합니다.

앞으로는 candidate를 다음 방식으로 검증하는 것이 좋습니다.

| 검증           | 필요한 조건                                     |
| ------------ | ------------------------------------------ |
| Execution    | post-fill invariant 0 violation            |
| Full vs fast | 완전 parity                                  |
| Tail         | P30/P40/P50/P60 모두 기록                      |
| Statistical  | paired stationary-bootstrap CI             |
| Regime       | bull / sideways / crash 별 결과               |
| Year         | leave-one-year-out                         |
| Parameters   | 최적점 주변 ±10~20%에서도 성과 유지                    |
| Liquidity    | φ=1%, 2%, 5% stress                        |
| Costs        | 비용 stress에서도 순위 유지                         |
| Universe     | exact HTS manifest                         |
| Artifact     | every champion run windows/daily/trades 저장 |

기존 `championship` gate가 scenario별 non-inferiority와 paired bootstrap CI를 요구하는 것은 좋은 방향입니다. 이 부분은 제거할 게 아니라 강화하는 편이 낫습니다.

그리고 **새 candidate 수 자체를 제한**하는 것이 좋습니다.

예를 들어 앞으로 competition 전까지 30개 parameter variant를 시도하기보다, 서로 다른 가설 3개만:

* `family_momentum_ensemble`
* `regime_switch`
* `rank_aware_controller`

를 연구하는 식입니다.

---

# 5. Oracle 결과를 제대로 활용해야 함

현재 oracle의 `P>50 ≈20.1%`와 inverse oracle `≈0.4%`는 중요한 단서입니다.

이걸 “20%까지 올릴 수 있다”고 해석하면 안 됩니다. 미래를 보는 oracle이기 때문에 실현 불가능합니다.

대신 의미는:

> **현재 남은 edge의 대부분은 overlay 미세조정보다 selection + timing에 있다.**

입니다.

따라서 `lock_level 0.45 vs 0.50`, `min_hold 2 vs 3` 같은 실험을 계속하기보다는 `tail-forensics`를 이용해 **상위 5% 역사 window에서 왜 P27이 승자를 못 잡았는지** 분해하는 편이 훨씬 생산적입니다.

각 window마다 최소한 다음 attribution을 만들 것을 권합니다.

```text
best ex-post eligible family
actual selected family
family selection loss

best entry date
actual entry date
entry timing loss

best exit/switch date
actual exit date
exit timing loss

execution/liquidity loss
giveback loss
wrong-regime loss
```

그리고

```text
selection loss > timing loss
```

인지,

```text
timing loss > selection loss
```

인지 수치로 확인한 다음 다음 feature를 개발해야 합니다.

이 과정이 현재 프로젝트에서 **가장 중요한 연구 단계**라고 봅니다.

---

# 6. 코드 구조도 한 번 더 정리하는 것이 좋음

Task 40에서 semantic strategy ID로 리팩터링을 시작한 방향은 맞습니다. 하지만 아직 실제 P27 source-of-truth는 legacy `_make_p27()`입니다. `factories.py`도 결국 `src.alpha.baselines._make_p27`을 부릅니다.

그리고 더 중요한 것은:

`configs/strategies.yaml`

```yaml
portfolio:
  sticky:
    mom60_raw:
```

가 존재하는데,

`_make_p27()`은 `portfolio.p27`을 찾은 뒤 상당수 값을 다시 hard-code합니다.

이 상태에서는 실험 config와 실제 실행 코드가 미묘하게 달라질 위험이 있습니다.

대회 전까지는 다음처럼 만드는 것이 좋습니다.

```text
semantic strategy ID
      ↓
one typed StrategyConfig
      ↓
signal config
vehicle config
risk config
execution config
      ↓
hash(config + commit + data snapshot + universe manifest)
      ↓
run_id
```

**“실제로 어떤 설정으로 이 5.31%가 나왔는가?”를 run 하나만 보고 완벽하게 복구할 수 있어야 합니다.**

---

# 7. HTS manifest와 execution calibration은 반드시 대회 직전 다시 해야 함

현재 `tournament.yaml`에는

```yaml
leverage_allowed: unknown
inverse_allowed: unknown
manifest: null
```

입니다.

현재 공개된 2026 공식 안내에는 자율형이 `투자자산 제한없음`이고, 비자율형만 레버리지·인버스를 제외한다고 명시되어 있으므로 **자율형에서 leverage/inverse 사용 자체는 공개 규정상 허용되는 것으로 판단할 수 있습니다.** ([머니투데이][3])

다만 정확한 종목 리스트는 별개입니다.

공식 안내상 HTS 계정/프로그램 정보가 **9월 17~18일경** 전달될 예정입니다. ([머니투데이][3])

그 시점에 바로:

* 실제 매매 가능 ETF 전체 추출
* `universe_manifest.yaml` 고정
* leverage/inverse/gold 등 분류 검증
* 주문 단위
* 시장가/지정가 처리
* 미체결 처리
* 수수료
* HTS 체결가격

을 확인해야 합니다.

현재 프로젝트의 PIT universe 구현은 exact-date existence/history/sponsor/liquidity를 순차 적용하기 때문에 기본 구조는 상당히 잘 되어 있습니다.

다만 **manifest=null 상태의 backtest는 어디까지나 proxy universe**입니다.

또 현재 φ=1% ADV와 3bps+5bps 비용도 실제 대회 HTS 규정이 아니라 modelling assumption입니다. φ sensitivity grid는 이미 있으니 1/2/5% 전부에서 candidate ranking이 유지되는지를 보는 것이 좋습니다.

---

# 8. 제가 지금부터 9월 21일까지 개발한다면

| 기간            | 해야 할 일                                                              | 신규 alpha         |
| ------------- | ------------------------------------------------------------------- | ---------------- |
| **9/3~9/5**   | P27 gross 1,958건 forensic → ledger/ADV transition 수정 → 0 violation  | 금지               |
| **9/5~9/7**   | P26/P27 artifact-complete 재baseline, 데이터 8/27→최신 갱신                 | 금지               |
| **9/7~9/10**  | tail-forensics로 selection/timing/giveback attribution               | 최소               |
| **9/10~9/14** | family momentum ensemble + regime vehicle controller                | 핵심               |
| **9/14~9/16** | nested/leave-year-out + parameter stability + liquidity/cost stress | candidate freeze |
| **9/17~18**   | HTS manifest/체결 규칙 실제 확인 후 exact universe 재baseline                 | 전략 변경 최소화        |
| **9/19~20**   | 최종 champion / fallback 확정, dry-run                                  | freeze           |
| **9/21~**     | predeclared rank-aware controller + daily decide                    | 즉흥 튜닝 금지         |

특히 `latest.md`의 data end가 아직 **2026-08-27**이므로, 현재 9월 3일 기준으로는 data freshness도 바로 해결해야 합니다.

---

## 제가 보는 최종 목표 architecture

```text
              ┌─────────────────────────┐
              │ Family Alpha            │
              │ mom20/40/60 + accel     │
              │ + breadth + RS          │
              └────────────┬────────────┘
                           ↓
                  Top 1~2 Theme
                           ↓
              ┌─────────────────────────┐
              │ Regime Controller       │
              │                         │
              │ strong risk-on → +2x   │
              │ risk-on        → +2/+1 │
              │ neutral        → +1/cash
              │ risk-off       → inverse│
              │ crisis         → inv/gold
              └────────────┬────────────┘
                           ↓
              ┌─────────────────────────┐
              │ Execution / Liquidity   │
              │ SELL-first              │
              │ residual gross budget   │
              │ exact HTS manifest      │
              └────────────┬────────────┘
                           ↓
              ┌─────────────────────────┐
              │ Tournament Controller   │
              │ current rank / gap      │
              │ remaining sessions      │
              │ peak / giveback         │
              └─────────────────────────┘
```

### 우선순위를 한 문장으로 압축하면

**`P27 파라미터 튜닝 → 추가 overlay` 방향은 이제 멈추고, `execution correctness → tail-forensics → underlying/family selection → long/inverse regime switching → leaderboard-aware risk control` 순서로 가는 것이 가장 합리적입니다.**

그리고 현재 기준으로는 **P27을 최종 champion이라 부르기보다 P27=Risk-on benchmark, P26=right-tail benchmark, P21=risk anchor**로 두고, 이 세 가지를 확실하게 이기는 새 semantic strategy만 promotion시키는 것이 좋습니다.

특히 남은 기간에 RL/딥러닝 같은 복잡한 모델을 추가하는 것은 추천하지 않습니다. `n_effective≈58`이고 이미 상당한 strategy-selection이 이루어진 상태라 모델 복잡도를 높이면 과최적화 가능성이 더 커집니다. 지금은 **단순하지만 causal하고, regime-aware하며, execution-correct한 전략**이 우승확률을 높일 가능성이 더 큽니다.

원한다면 다음 단계에서는 제가 이 분석을 바탕으로 **실제 저장소 기준 `P0 → P1 → P2` 개발 TASK를 파일·함수 단위로 쪼개서, “어떤 코드를 어떤 순서로 수정/추가해야 하는지” 구현 로드맵까지 설계**해드릴 수 있습니다.

[1]: https://www.mt.co.kr/stock/2025/11/25/2025112516260834930 "https://www.mt.co.kr/stock/2025/11/25/2025112516260834930"
[2]: https://www.mt.co.kr/stock/2025/09/27/2025092618035044343 "https://www.mt.co.kr/stock/2025/09/27/2025092618035044343"
[3]: https://www.mt.co.kr/etf/join/index.html "https://www.mt.co.kr/etf/join/index.html"
[4]: https://www.mt.co.kr/stock/2025/10/25/2025102418201958573 "https://www.mt.co.kr/stock/2025/10/25/2025102418201958573"
