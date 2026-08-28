# 07. Alpha & Portfolio — L4~L5 계층

Signal 생성(alpha)과 포지션 결정(portfolio)을 분리합니다. **Signal ≠ Portfolio ≠ Tournament Policy.**

---

## 1. AlphaModel Protocol

```python
class AlphaModel(Protocol):
    def score(
        self,
        features: pl.DataFrame,
        universe: pl.DataFrame,
        as_of: date,
    ) -> pl.DataFrame:
        """returns: isu_cd, score, rank, cluster, ..."""
```

- 입력: PIT feature + universe
- 출력: 종목별 score (높을수록 매수 선호)
- alpha 는 **비중을 결정하지 않음**

---

## 2. Baseline 전략 (B0~B5)

모든 신규 아이디어는 동일 프로토콜로 baseline 대비 검증합니다.

| ID | 전략 | 핵심 로직 |
| --- | --- | --- |
| B0 | Buy & Hold KODEX 200 | 단일 ETF 보유 |
| B1 | Equal Weight Top-5 Momentum | `mom_20` 상위 5개 동일 비중 |
| B2 | Risk Parity Top-5 | `rv_20` 역가중 |
| B3 | Sector Rotation | theme breadth 상위 1 cluster |
| B4 | Creation Flow | `creation_flow` z-score 상위 |
| B5 | Regime Switch | RISK_ON → momentum, RISK_OFF → cash |

B0 은 **floor** — 어떤 전략도 B0 대비 $P(R>θ)$ 하위 꼬리를 악화시키면 기각.

---

## 3. Sector Leadership Model (Stage 4)

핵심 가설: **개별 ETF ranking 보다 주도 섹터 선정이 tournament 에 유리하다.**

```
Regime (market state)
  → Cluster Selection (어떤 theme 이 강한가)
    → Within-cluster Ranking (그 theme 내 최고 ETF)
      → Concentrated Pick (Top 1~3)
```

### 3.1 Cluster Score

```
cluster_score(theme, t) =
    w1 × cluster_breadth(theme, t)
  + w2 × mean(rank_mom_20 | theme, t)
  + w3 × mean(creation_flow_z | theme, t)
```

가중치 `w1, w2, w3` 는 `configs/leadership.yaml`.

### 3.2 Within-cluster Pick

선정된 theme 내에서 `rank_mom_20` + `rank_accel` 복합 순위.

---

## 4. Portfolio Layer

### 4.1 Selection (`selection.py`)

```python
ClusterAwareSelection:
  1. alpha.score() → index_key ranked list
  2. family dedup (동일 leverage_family 최대 1개)
  3. cluster dedup (동일 theme 최대 N개)
  4. liquidity filter (ADV 참여율)
  5. tournament_rules filter
```

**family dedup 이 cluster dedup 보다 먼저**입니다. `KODEX 200` 과 `KODEX 레버리지` 는 동일 베팅이므로 theme 수준에서 걸러지기 전에 제거되어야 합니다.

### 4.2 Sizing (`sizing.py`)

| Mode | 설명 |
| --- | --- |
| `equal` | 선택 종목 동일 비중 |
| `score_weighted` | score 비례 |
| `confidence` | score × regime confidence |
| `concentrated` | Top-1 에 60~80% (tournament 최적화) |

Tournament 연구 질문 #8: **Top-1 집중 vs Top-3 분산** — rolling-36D 분포로 비교.

### 4.3 Constraints (`constraints.py`)

```
normalize_weights()     — 합 = 1.0
leverage_gate()         — Unknown 시나리오별 처리
max_single_weight()     — 단일 종목 상한 (config)
min_cash()              — 최소 현금 비율
```

### 4.4 Position State Machine (`state.py`)

```
CASH ──(entry signal)──→ LONG
LONG ──(exit signal)───→ CASH
LONG ──(rebalance)─────→ LONG (adjusted weights)
```

- Entry: alpha score > threshold + regime 허용
- Exit: score 하락 / drawdown limit / regime 전환
- Re-entry: cooldown 기간 후 (연구 질문 #10)

상태 전이는 **next-open 체결** (INV-3).

---

## 5. Tournament Overlay (L7 연계)

Portfolio 가 산출한 target weights 에 대회 상황에 따른 조정을 적용합니다. `AggressionPolicy` 는 portfolio 와 분리된 모듈 (`tournament/policy.py`).

### 5.1 Aggression 입력

| 입력 | 조정 |
| --- | --- |
| 잔여 거래일 ≤ 5 | aggression ↑ (집중도·배수 증가) |
| 현재 수익률 < 0 | risk ↓ (cash 비율 ↑) |
| 현재 수익률 > 30% | aggression ↑ (lead 확보) |
| regime = CRISIS | 전량 현금 또는 inverse |

### 5.2 ExposureSelector — 배수 선택

레버리지 허용에 따라 추가된 계층입니다. **alpha 가 고른 `index_key` 에 대해 어느 배수를 실제로 살지** 결정합니다.

```
alpha.score()        → index_key 랭킹 (배수 무관)
       ▼
ExposureSelector     → LeverageFamily 에서 배수 1개 선택
       ▼
target position      → 구체적 isu_cd
```

선택 규칙:

| 조건 | 배수 |
| --- | --- |
| regime RISK_ON + 잔여일수 충분 + family 에 +2x 존재 | `+2` |
| regime NEUTRAL / +2x 유동성 부족 | `+1` |
| regime RISK_OFF (인버스 허용 시) | `-1` |
| regime CRISIS + 명확한 하락추세 | `-2` 또는 cash |
| `leverage_multiple` Confidence = LOW | **`+1` 강제** (fail-closed) |

배수 선택은 **family 내 유동성 검사를 다시 통과**해야 합니다. `+2x` 가 ADV 조건에 미달하면 `+1x` 로 강등합니다.

### 5.3 실효 노출 계산

포지션 비중이 아니라 **실효 노출**로 제약을 겁니다.

```
gross_exposure = Σ |weight_i × leverage_multiple_i|
net_exposure   = Σ (weight_i × leverage_multiple_i)
```

`max_gross_exposure` 는 config 값입니다. Top-1 에 80% 를 `+2x` 로 넣으면 gross = 1.6 이므로, 비중 상한만 보면 실제 위험을 놓칩니다.

### 5.4 레버리지 감쇠는 실제 가격으로만

일별 리밸런싱 레버리지 ETF 의 36세션 수익률은 기초지수 36일 수익률의 2배가 **아닙니다**. 변동성이 클수록 경로 의존적 감쇠가 커집니다.

**합성 금지**: 지수 수익률에 배수를 곱해 레버리지 ETF 를 만들어내면 안 됩니다. KRX 에서 실제 레버리지 ETF 의 일별 가격을 받고 있으므로 **그 가격만** 사용합니다 (INV-14).

---

## 6. 체결 모델 (backtest 연계)

```
signal at close(t)
  → order at open(t+1)
  → fill price = open(t+1) × (1 + slippage)
```

- same-bar 체결 금지
- 유동성 제약: `order_value ≤ ADV × participation_rate`
- 초과분은 partial fill 또는 skip (config)

---

## 7. 비용 모델

| 항목 | 기본값 | 범위 (robustness grid) |
| --- | --- | --- |
| Commission | 0.015% | 0.01% ~ 0.03% |
| Slippage | 5 bps | 3 ~ 10 bps |
| Tax | 0% (모의투자) | — |

`CostConfig.grid()` 로 robustness sweep.

---

## 8. ML AlphaModel (범위 내)

LightGBM Ranker 는 `AlphaModel` Protocol 을 구현하는 **또 하나의 후보 모델**입니다.

```python
class LightGBMRankerAlpha:                  # AlphaModel 구현체
    def score(self, features, universe, as_of) -> pl.DataFrame: ...
```

portfolio·backtest·tournament 계층은 **변경 없이** 그대로 동작합니다. rule model 과 ML model 은 동일한 `score` 계약을 만족하므로 게이트 비교도 동일 프로토콜입니다.

| 원칙 | 내용 |
| --- | --- |
| 위치 | L4 (alpha) 내부. 새 계층 아님 |
| 출력 | index_key 단위 score — **배수 선택은 여전히 overlay** |
| 채택 조건 | purged walk-forward 에서 best rule baseline 초과 |
| 기본값 | **기각**. 이기지 못하면 버린다 |
| 범위 외 | deep learning, RL |

설계 상세(표본 수 산정, purged/embargo CV, 모델 용량 상한, 라벨 정의)는 [12-ml-layer.md](12-ml-layer.md) 를 참조하십시오.
