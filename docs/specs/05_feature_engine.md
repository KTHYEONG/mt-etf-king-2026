# 05. Feature Engine — PIT guard · momentum · trend · volatility · flow · breadth · regime

**선행**: [04_pit_universe](04_pit_universe.md)
**상위**: [00_architecture.md](00_architecture.md)

---

## 1. Diagnosis & Invariants

### 1.1 Look-ahead 는 "미래 데이터를 읽는 것"이 아니라 "정렬 실수"로 들어온다

명시적으로 미래 행을 읽는 코드는 리뷰에서 잡힌다. 실제로 통과하는 것은 이런 것들이다.

| 유형 | 증상 |
| --- | --- |
| 전체 패널에 `rolling_mean` 후 날짜 슬라이스 | 각 날짜 값은 맞지만, **정규화·랭킹 단계에서 전체 기간 통계가 섞임** |
| `shift(k)` 를 관측 개수로 적용 | 거래정지로 행이 빠진 종목은 $k$ 세션이 아니라 $k$ **관측** 전을 참조 |
| cross-sectional rank 를 전체 종목에 적용 | 그날 매수 불가능한 종목까지 분모에 들어가 percentile 왜곡 |
| feature 계산 후 universe 필터 | 위와 동일. **순서가 결과를 바꾼다** |

**INV-05-1 (구조적 방어)**: 모든 feature 함수는 `decision_date` 를 필수 인자로 받고, 첫 줄에서 `assert_pit(frame, decision_date)` 를 호출한다. 규율이 아니라 코드 계약으로 강제한다.

**INV-05-2 (Shift-invariance — 가장 강한 검증)**: 임의의 $t$ 에 대해

$$
f\big(\text{panel}|_{\le t},\; t\big) \;=\; f\big(\text{panel}_{\text{full}},\; t\big)
$$

전체 패널로 계산한 뒤 $t$ 를 슬라이스한 결과와, $t$ 까지 잘라낸 패널로 계산한 결과가 **완전히 동일**해야 한다. 이 등식이 깨지면 그 feature 는 미래를 본 것이다. 이것은 property test 로 검증한다(`hypothesis` 는 이미 dev 의존성).

### 1.2 세션 그리드 정렬 — 조용한 lookback 단축

ETF 는 거래정지·유동성공급 중단으로 특정일 행이 빠질 수 있다. Polars `shift(20)` 을 ticker 그룹에 적용하면 **20번째 이전 관측**을 참조하는데, 그 종목이 3일 결측이면 실제로는 23 세션 전 가격이다.

**INV-05-3**: 계산 전에 패널을 `(ticker × sessions[first_seen..last_seen])` **완전 그리드**로 reindex 한다. 결측은 `None` 으로 남기고(전방 채움 금지 — 거래정지 중 가격을 "유지"하면 변동성이 0으로 왜곡된다), rolling 은 `min_periods` 로 유효성을 요구한다.

### 1.3 Cross-sectional 정규화의 모집단

$$
\texttt{rs}_k(i, t) = \frac{\big|\{ j \in \mathcal{U}(t) : \texttt{mom}_k(j,t) < \texttt{mom}_k(i,t) \}\big|}{|\mathcal{U}(t)| - 1}
$$

분모는 **그날의 적격 universe $\mathcal{U}(t)$** 이지 전체 상장 ETF가 아니다. 실측상 이 차이는 1,163 vs 26~121 — 분모가 10배 다르면 percentile 은 완전히 다른 값이 된다.

**INV-05-4**: cross-sectional 통계(rank, z-score)는 반드시 `UniverseSnapshot.tickers` 로 마스킹한 뒤 계산한다.

### 1.4 Flow — 이 시스템의 고유 우위

next.md §25 는 `AUM change ≠ capital inflow` 를 경고하지만 해법은 제시하지 않았다. KRX ETF 패널은 `LIST_SHRS`(설정좌수)와 `NAV` 를 **동시에** 주므로 정확히 분해된다.

$$
A_t = L_t N_t \;\Longrightarrow\;
\Delta A_t = \underbrace{\Delta L_t \cdot N_t}_{\texttt{creation\_flow}} + \underbrace{L_{t-1} \cdot \Delta N_t}_{\text{성과효과}}
$$

이 항등식은 **대수적으로 정확**하다(근사가 아니다). 따라서 테스트에서 잔차 0 을 요구할 수 있다.

실측(2026-08-26→27): 1,163종목 중 **289종목**에서 $\Delta L \neq 0$, 총 설정 1.744조 / 환매 0.671조. 가격 상승으로 인한 AUM 증가와 완전히 분리된 실제 자금 유입 신호다.

파생 feature:

| feature | 정의 | 해석 |
| --- | --- | --- |
| `creation_flow_krw` | $\Delta L_t \cdot N_t$ | 당일 순설정액 |
| `flow_ratio` | $\Delta L_t / L_{t-1}$ | 규모 정규화 |
| `flow_5d`, `flow_20d` | 누적 $\sum \Delta L \cdot N$ | 지속적 자금 유입 |
| `turnover` | $\texttt{trading\_value} / \texttt{net\_assets}$ | 회전율 |
| `volume_expansion` | $\text{ADV}_5 / \text{ADV}_{20}$ | 거래 급증 |
| `disparity` | $(P - N)/N$ | 괴리 — 과열/체결비용 |

### 1.5 Breadth — 가능한 것과 불가능한 것

`/etp/etf_pdf` 는 **404, 존재하지 않는다**(실측). 따라서 ETF 구성종목 breadth 는 primary source 로 불가능하다. 대신 실제로 가능한 두 계층을 쓴다.

**(a) Market breadth** — `stock_daily`(KOSPI 944 + KOSDAQ 1,823 = 2,767 종목)에서 계산. **완전 가능.**

$$
\texttt{breadth\_ma20}(t) = \frac{|\{ j : P_j(t) > \text{MA}_{20}(j,t) \}|}{|\{ j : \text{유효}(j,t) \}|}
$$

+ `breadth_up_5d`, `breadth_near_high_20`, `advance_decline_ratio`. → **regime 판정용**.

**(b) Cluster breadth** — 같은 `theme` 에 속한 ETF 집합 내부에서 동일 지표를 계산. 구성종목 없이도 "이 테마가 한두 종목이 끌고 가는가, 전체가 같이 오르는가"를 답한다. → **sector leadership 품질 판정용**.

**(c) Constituent breadth** — PDF 필요. **pykrx 어댑터로 유예**(Stage 3 이후, optional).

### 1.6 Regime — rule-based 먼저

5단계 이산 상태. ML 없이 시작한다(next.md §29-30).

$$
S(t) = w^\top c(t), \qquad c(t) \in \{0,1\}^m
$$

$c$ 는 이진 조건 벡터(KOSPI > MA20, MA20 기울기 > 0, KOSDAQ > MA20, market breadth > $\beta$, 실현변동성 < $\sigma^\*$), $w$ 는 `configs/features.yaml` 의 가중치. $S$ 를 4개 임계로 5구간에 매핑한다.

**INV-05-5 (단조성)**: 조건 하나가 0→1 로 바뀌면 $S$ 는 감소할 수 없고, regime 상태도 risk-off 방향으로 이동할 수 없다. 이는 property test 로 검증 가능한 구조적 성질이며, 가중치를 바꿔도 유지되어야 한다.

### 1.7 레버리지 ETF (INV-9)

레버리지 ETF 수익률을 `기초지수 수익률 × 2` 로 합성하지 않는다. 일간 리셋과 복리 효과 때문에 다기간 수익률이 배수와 크게 달라진다. **항상 실제 ETF 가격 시계열을 쓴다.** `underlying_index_close` 는 tracking difference 계산에만 쓰고 가격 합성에 쓰지 않는다.

### 1.8 복잡도 예산

- 패널 $N \approx 2.3\text{M}$ 행. 전체 feature 빌드는 Polars lazy 표현식으로 **단일 패스 그룹 연산**들의 합성이어야 한다.
- 목표: 전체 feature 패널 생성 < 30초, 단일 날짜 슬라이스 < 50ms.
- **금지**: 날짜 루프 안에서 rolling 재계산. rolling 은 패널 전체에 한 번만 적용하고, 백테스트는 사전 계산된 gold 패널을 날짜로 슬라이스한다.

---

## 2. Architecture & Mitigation

```
 normalized/etf_daily.parquet ─┐
 normalized/stock_daily.parquet├─► pit.py (assert_pit · session grid)
 normalized/index_daily.parquet┘         │
                                          ├─► momentum.py  mom_{3,5,10,20,40,60}
                                          ├─► trend.py     ma ratio · slope · breakout · drawdown
                                          ├─► volatility.py rv_{5,20} · atr · downside · gap
                                          ├─► flow.py      creation flow · turnover · disparity
                                          ├─► crosssec.py  rs_k · zscore · acceleration
                                          ├─► breadth.py   market · cluster
                                          └─► regime.py    5-state
                                                  │
                                          builder.py ──► features/etf_features.parquet (gold)
```

### 2.1 `assert_pit` 계약

```python
def assert_pit(frame: pl.DataFrame, decision_date: date, date_column: str = "date") -> pl.DataFrame
```

`date > decision_date` 인 행이 하나라도 있으면 `PitViolationError` 를 던지고 위반 날짜 샘플(최대 5개)을 메시지에 담는다. 통과하면 프레임을 그대로 반환해 체이닝이 가능하다.

### 2.2 Momentum 계산

$$
\texttt{mom}_k(i,t) = \frac{P_i(t)}{P_i(t-k)} - 1
$$

세션 그리드 상의 `shift(k)` 로 정확히 계산한다(로그 변환 불필요 — 단일 비율은 정확하다). 반면 **누적 수익률 합성**(백테스트 자산곡선)에서는 `log1p`/`expm1` 을 쓴다(quant.md §4).

$P_i(t-k)$ 가 `None` 이거나 0 이면 결과는 `None` (0 나눗셈 방지, INV-1 연장).

### 2.3 Momentum acceleration

$$
\texttt{accel}(i,t) = \texttt{rs}_5(i,t) - \texttt{rs}_{20}(i,t)
$$

percentile rank 차이를 쓴다(z-score 차이보다 이상치에 강건). 값이 양수 = "최근 5일 상대강도가 20일 상대강도보다 높다" = **leadership 부상 중**.

### 2.4 설정 파일

`configs/features.yaml` 에 horizon 목록, rolling window, regime 조건·가중치·임계값을 둔다. **어떤 숫자도 Python 리터럴로 두지 않는다**(INV-11).

---

## 3. Assumptions

- **A-1**: momentum horizon 은 `{3,5,10,20,40,60}`. 대회가 36세션이므로 40/60 은 대회 길이를 초과하지만, "대회 시작 시점의 중기 추세"를 판단하는 데 필요하다.
- **A-2**: 실현변동성은 연율화하지 않는다. 36세션 단일 구간 문제라 연율화는 정보를 더하지 않고 임계값 해석만 흐린다.
- **A-3**: cluster breadth 의 최소 멤버 수는 3. 그 미만이면 breadth 는 `None` (2종목의 "50%"는 의미 없음).
- **A-4**: 결측은 전방 채움(forward-fill) 하지 않는다. 거래정지 기간 가격 유지는 변동성을 0으로, momentum 을 평평하게 왜곡한다.

---

## 4. Execution Target

```bash
uv run pytest tests/unit/features -q --tb=short
uv run python tools/agent_skills/lean_check.py --spec docs/specs/05_feature_engine_contract.json

uv run mt-etf features --start 2018-01-01 --end 2026-08-27
```
