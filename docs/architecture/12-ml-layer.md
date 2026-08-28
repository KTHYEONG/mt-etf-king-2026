# 12. ML Layer — LightGBM Ranker

ML 은 **별도 계층이 아니라 `AlphaModel` 구현체 하나**입니다. 이 문서는 왜 shallow GBDT 까지만 허용되는지, 그리고 라벨 중첩으로 인한 누출을 어떻게 막는지를 다룹니다.

---

## 1. 범위 판단: 일정이 아니라 표본

ML 배제/포함의 기준은 개발 기간이 아니라 **유효 독립 표본 수**입니다.

### 1.1 Raw row 수는 지표가 아니다

| 항목 | 값 | 출처 |
| --- | --- | --- |
| XKRX 세션 (2010-01 ~ 2026-08-27) | 4,101 | 실측 |
| 현재 상장 ETF | 1,163 | 실측 |
| ADV ≥ 100억 (오늘) | 121 | 실측 |
| ADV ≥ 1,000억 (오늘) | 26 | 실측 |

세션 × 종목으로 세면 수십만 행이 나오지만, 이 숫자로 모델 용량을 정하면 과적합합니다.

### 1.2 유효 표본 산정 (추정)

세 가지가 표본을 깎습니다.

```
1. 라벨 중첩       forward-h 수익률은 h 세션 동안 겹친다
                   독립 시점 ≈ sessions / h

2. 단면 상관       국내 ETF 는 기초지수가 겹친다
                   (코스피200 추종 25종, S&P500 24종, 나스닥100 16종 — 실측)
                   시점당 독립 베팅 ≪ 종목 수

3. 유니버스 성장   2010년대 초반은 단면이 얇다
                   초기 구간의 정보량이 낮다
```

`h = 10`, 2018년 이후 약 2,000 세션, 시점당 유효 독립 베팅을 10~15 로 보면:

$$
n_{\text{effective}} \approx \frac{2{,}000}{10} \times 12 \approx 2{,}400
$$

**이 값은 추정입니다.** 실제 값은 06 harness 구현 후 단면 상관행렬의 고유값 분포로 재산정하고, 그 결과로 아래 용량 상한을 갱신합니다.

### 1.3 결론

| 모델 | 판정 | 근거 |
| --- | --- | --- |
| LightGBM / XGBoost (shallow) | **채택 후보** | 수천 표본에서 강한 정칙화로 학습 가능 |
| LightGBM Ranker (LambdaRank) | **주 후보** | 단면 순위 목적과 직접 정합 |
| MLP / LSTM / Transformer | 범위 외 | 파라미터 수가 유효 표본을 초과 |
| 강화학습 | 범위 외 | 표본 요구량이 한 자릿수 배 더 큼 |

---

## 2. 왜 Ranker 인가

절대 수익률 회귀는 **시장 베타에 지배**됩니다. 상승장에서는 전부 오르고 하락장에서는 전부 내리므로, 모델이 "시장 방향 예측"에 표본을 소진하고 종목 선택력을 학습하지 못합니다.

우리가 필요한 것은 시장 방향이 아니라 **그날 무엇이 상대적으로 강한가**입니다. 시장 방향은 regime classifier 와 overlay 가 담당합니다.

```
target = percentile_rank(forward_return_h | 그날 적격 유니버스)
group  = date
objective = lambdarank
```

`group = date` 가 핵심입니다. 손실 함수가 **같은 날 종목들 사이의 순서**만 평가하므로 시장 공통 요인이 자동으로 제거됩니다.

---

## 3. 라벨 누출 차단

가장 큰 위험이자, 이 문서가 존재하는 이유입니다.

### 3.1 문제

`t` 시점의 라벨은 `t+h` 까지의 가격을 사용합니다. 단순 시계열 분할(train `≤ T`, test `> T`)은 경계 부근에서 누출됩니다.

```
train 마지막 샘플: t = T
그 라벨의 실현 구간: T ~ T+h        ← test 구간과 겹침
```

### 3.2 해결: Purged + Embargoed Walk-Forward

```
|─────── train ───────|  purge  |  embargo  |─── test ───|
                        ← h →     ← e →
```

| 요소 | 크기 | 목적 |
| --- | --- | --- |
| **purge** | `h` 세션 | 라벨 실현 구간이 test 를 침범하는 train 샘플 제거 |
| **embargo** | `e` 세션 (기본 `h`) | 자기상관으로 인한 잔여 누출 차단 |

**INV-13**: 모든 ML 검증은 purge ≥ label horizon, embargo ≥ label horizon 을 만족해야 한다. 단순 `train_test_split` 또는 무작위 KFold 는 금지.

### 3.3 Walk-Forward 구조

```
fold 1: train 2018-2021 | purge+embargo | test 2022H1
fold 2: train 2018-2022H1 | purge+embargo | test 2022H2
fold 3: train 2018-2022 | purge+embargo | test 2023H1
...
```

expanding window 를 기본으로 합니다. rolling window 는 표본이 이미 부족하므로 보조 확인용입니다.

### 3.4 하이퍼파라미터 선택도 fold 안에서

early stopping 라운드 수, learning rate 등을 **전체 데이터로 고르면 그 자체가 누출**입니다. 각 fold 의 train 구간 내부에서 다시 purged split 을 만들어 선택합니다 (nested).

---

## 4. 모델 용량 상한

`n_effective ≈ 2,400` 에서 도출한 제약입니다. `configs/ml.yaml` 에 두고, §1.2 재산정 결과에 따라 갱신합니다.

| 파라미터 | 상한 | 이유 |
| --- | --- | --- |
| `num_leaves` | ≤ 8 | 유효 표본 대비 분할 수 |
| `max_depth` | ≤ 4 | 동일 |
| `min_data_in_leaf` | ≥ 100 | 잎당 최소 표본 |
| feature 수 | ≤ 25 | 차원 대비 표본 |
| `feature_fraction` | ~0.6 | 상관 feature 분산 |
| `bagging_fraction` | ~0.8 | 분산 감소 |
| `lambda_l2` | > 0 필수 | 정칙화 |
| 부스팅 라운드 | early stopping | purged validation 기준 |

**정칙화 파라미터를 성능 튜닝 대상으로 삼지 않습니다.** 상한은 표본 구조에서 나온 제약이지 탐색 공간이 아닙니다.

---

## 5. Feature 입력

L3 feature engine 이 생성한 것만 사용합니다. ML 전용 feature 를 따로 만들지 않습니다 — rule model 과 **동일한 입력**이어야 비교가 공정합니다.

**Universe**: 학습·검증·채택은 `ml.yaml` → `universe_mode: deployment` (후원 운용사 ETF). structural 패널로 학습한 모델은 채택할 수 없습니다 (INV-21). 비후원 ETF 는 유동성 상위에 없어 full-universe 학습과 결과가 거의 같을 수 있으나, 설계상 deployment 를 강제합니다.

```
momentum      mom_3, mom_5, mom_10, mom_20, mom_40
relative      rank_mom_5, rank_mom_20, mom_accel
volume        volume_ratio, turnover_ratio
volatility    rv_5, rv_20, atr_20
trend         breakout_20, drawdown_20, ma_ratio_20
flow          creation_flow_z, nav_disparity
cluster       theme_rs, theme_breadth
market        regime (categorical), mkt_breadth
```

feature 수가 §4 상한(25)에 근접하므로 **추가는 교체를 동반**합니다.

---

## 6. 모듈 구성

```
src/alpha/ml/
├── dataset.py     # (X, y, group) 구성, 라벨 = 단면 percentile rank
├── splits.py      # PurgedWalkForward(n_folds, horizon, embargo)
├── ranker.py      # LightGBMRankerAlpha — AlphaModel 구현
├── train.py       # fold 루프, nested early stopping
└── registry.py    # 모델 아티팩트 버전 관리
```

### 6.1 Registry

모델 아티팩트는 재현 가능해야 합니다.

```
data/models/
  └── ranker_{config_hash}_{data_range}_{trained_at}.txt
```

`config_hash` 는 feature config + ml config 의 해시입니다. 백테스트 결과는 항상 어떤 모델 아티팩트로 산출됐는지 기록합니다.

---

## 7. 채택 게이트

**기본값은 기각입니다.** ML 은 아래를 모두 만족할 때만 채택됩니다.

| Gate | 조건 |
| --- | --- |
| G6 | purged walk-forward 전 fold 에서 IC > 0, 부호 일관 |
| G7 | best rule baseline 대비 rolling-36D `P(R>30%)` 개선 |
| G8 | fold 간 성능 편차가 baseline 대비 크게 확대되지 않음 |
| G1~G5 | rule model 과 **동일 게이트** 통과 |

in-sample 개선만으로는 절대 채택하지 않습니다.

### 7.1 목적 불일치 주의

Ranker 는 **평균 순위 정확도(IC/NDCG)** 를 최적화합니다. 대회 목적은 **우측 꼬리 `P(R>θ)`** 입니다. 둘은 같지 않습니다.

따라서 ML 의 역할은 **selection 품질 개선**에 한정하고, 꼬리는 여전히 집중도·배수(overlay)가 만듭니다. IC 가 올랐는데 `P(R>30%)` 가 안 오르면 **G7 FAIL 로 기각**합니다.

---

## 8. 일정상 위치

Stage 5 이므로 **rule baseline 검증(W3) 이후**입니다. W3 가 끝나지 않으면 ML 은 시작하지 않습니다.

ML 이 W4 안에 끝나지 않아도 대회 참가에는 지장이 없습니다. `AlphaModel` drop-in 구조이므로 **대회 기간 중에도 게이트를 통과하는 시점에 교체**할 수 있습니다. 이것이 ML 을 별도 계층으로 만들지 않는 실질적 이유입니다.
