# 01. Overview — 목적과 설계 원칙

## 1. 프로젝트 정의

이 시스템은 **"ETF 추천 AI"** 가 아니라, 약 36거래일 모의투자 대회에서 **우승·최우수상 확률을 높이는 포지션**을 연구·검증·실행하기 위한 **Tournament Quant Research System** 입니다.

```
Primary Objective: 8주 대회에서 전체 1위 / 2위 확률 극대화
```

일반 퀀트 목표(`maximize Sharpe`, `minimize MDD`, `maximize CAGR`)와 다릅니다.

---

## 2. 목적 함수

참가자 분포를 모르므로 $P(\text{rank}=1)$ 을 직접 추정하지 않습니다. 대신 **구조적으로 등가인 대리 목적**을 사용합니다.

$$
\text{maximize} \quad P(R_{36d} > R_{\text{competitors}})
$$

상금을 고려하면:

$$
\mathbb{E}[\text{Prize}] = 1000 \cdot P(\text{rank}{=}1) + 500 \cdot P(\text{rank}{=}2) + 100 \cdot P(\text{category top}=1)
$$

### 2.1 평가 단위는 스칼라가 아니라 분포

모든 전략 평가는 **36거래일 수익률 분포 전체**를 산출해야 합니다. mean·Sharpe 같은 요약 지표는 분포에서 파생될 뿐, 분포를 대체할 수 없습니다.

핵심 산출물:

- $P(R > \theta)$ for $\theta \in \{10, 20, 30, 40, 50\}\%$
- 분위수 $q_{05}, q_{25}, \ldots, q_{99}$
- CVaR(5%), MDD 분포, **giveback median·q90** (INV-25)
- $n_{\text{effective}}$ (겹치는 rolling window 보정)

### 2.1.1 데이터 역할 (INV-PRE-3)

| 데이터 | 역할 | 금지 |
| --- | --- | --- |
| 제1회(2024, ~4개월) | Hypothesis generator | parameter calibration, 07 채택 근거 |
| 제2회(2025, 35세션) | Case-study replay | 단독 accept/reject, 파라미터 최적화 |
| Rolling 36D (다년) | 전략 채택의 유일한 정량 근거 | 2025 창을 표본에서 삭제하지는 않되 별도 라벨 |

### 2.2 의사결정 관련 구간

2025년 제2회 대회 관측값을 **참조 앵커**(분포 추정치 아님)로 사용합니다.

| 구간 | 의미 |
| --- | --- |
| $\theta \in [30\%, 60\%]$ | 대상(1위) 관련 |
| $\theta \in [20\%, 40\%]$ | 최우수상(2위) 관련 |
| $\theta < 15\%$ | 상금 기대값에 거의 기여 없음 |

**평균 수익률 1%p 개선**보다 **$P(R > 40\%)$ 1%p 개선**이 우선입니다. 단, 하위 꼬리(worst 5%) 악화는 허용하지 않습니다.

---

## 3. 대회 맥락 (2026)

| 항목 | 값 |
| --- | --- |
| 기간 | 2026-09-21 ~ 2026-11-13 |
| 거래일 수 | **36 sessions** (XKRX) |
| 초기자금 | 10억원 |
| 부문 | 국내주식형 / 연금형 / 글로벌형 / **자율형** |
| 상금 | 대상 1,000만 / 최우수 500만 / 부문별 우수 100만 |
| 체결 환경 | 코스콤 모의투자 HTS |

### 3.1 레버리지·인버스 (자율형)

자율형은 투자자산 제한이 없고, **레버리지·인버스가 허용될 것으로 본다**는 것이 현재 작업 가정입니다. 다만 보도자료에는 명시적 허용 문구가 없고 "건전한 투자 전략을 유도"라는 취지만 있으므로, 규칙 상태는 계속 `Unknown` 으로 표현하되 **primary scenario 를 허용(aggressive) 쪽으로 둡니다.**

이 전환은 취향이 아니라 유동성 실측에 근거합니다 (2026-08-27, 10억 전액을 ADV 1% 이내로 소화 기준).

| 참여율 | 필요 ADV | 거래가능 | 일반 | 레버리지·인버스 |
| --- | --- | --- | --- | --- |
| 1% | 1,000억 | 26 | **15** | **11** |
| 2% | 500억 | 41 | 28 | 13 |
| 5% | 200억 | 65 | 50 | 15 |

레버리지·인버스를 배제하면 실질 유니버스가 **26 → 15 종목**으로 줄어듭니다. 이는 위험 회피가 아니라 **집중 리스크 증가**입니다. 거래대금 상위권(`KODEX 레버리지` 2위, `KODEX 인버스` 4위, `KODEX 200선물인버스2X` 11위)이 통째로 빠지기 때문입니다.

보수(deny) 시나리오는 폐기하지 않고 **fallback 으로 유지**합니다. 대회 직전 HTS 종목 리스트가 공개되면 그때 `Unknown` 을 확정값으로 대체합니다 (R-4).

### 3.2 후원사 ETF 제약

**후원 운용사 ETF만 매매 가능**합니다. 인프라 후원(금융투자협회·한국거래소·코스콤)은 ETF를 발행하지 않습니다.

| 운용사 | 브랜드 |
| --- | --- |
| 삼성·미래에셋·KB·한국투자·신한·한화·타임폴리오·NH아문디·키움·하나 | KODEX, TIGER, KoAct, RISE, ACE, SOL, PLUS, TIME, HANARO, KIWOOM, 1Q |

- **Structural 연구**: 전체 패널로 아이디어 탐색 가능
- **Deployment / ML / decide**: 후원사 ETF만 (`configs/sponsor_brands.yaml`)
- **최종**: HTS 부문별 허용 리스트 (`configs/universe_manifest.yaml`)

유동성 상위(26종·Top 20)는 후원사만으로도 동일하지만, deployment 필터는 설계상 필수입니다 ([05 §6](05-universe-and-instruments.md)).

---

## 4. 핵심 설계 원칙

### 4.1 Signal ≠ Portfolio ≠ Tournament Policy

세 계층을 절대 섞지 않습니다.

```
Alpha (무엇이 강한가)
  → Portfolio (얼마나, 어떻게 배분)
    → Tournament Overlay (대회 순위·잔여일수에 따른 risk 조정)
```

### 4.2 Point-in-Time 우선

모든 데이터·universe·feature·체결은 **그 시점에 관측 가능했던 정보만** 사용합니다.

- ETF 상장일 API 없음 → 패널 등장/소멸로 유도
- signal at `close(t)` → fill at `open(t+1)`
- cross-sectional rank → 그날의 적격 universe 만 대상

### 4.3 Fail-Closed

불확실한 것을 추측하지 않습니다.

- KRX `""` → `None` (0 변환 금지)
- 휴장일 → 행 수가 아니라 유효 가격 비율로 판정
- 미확정 대회 규칙 → `Unknown` + 시나리오 스윕
- validation CRITICAL → Parquet 쓰기 중단

### 4.4 Baseline First, ML Gated

모든 신규 아이디어는 B0~B5 baseline 대비 **동일 프로토콜**로 검증됩니다. ML 은 이 프로토콜을 통과해야만 채택되는 **후보 중 하나**이지 특별 대우 대상이 아닙니다.

범위 판단 기준은 개발 일정이 아니라 **표본 구조**입니다.

| 모델 | 판정 | 근거 |
| --- | --- | --- |
| Shallow GBDT (LightGBM Ranker) | **범위 내** | 유효 표본 수천 규모에서 강한 정칙화로 학습 가능 |
| Deep learning / RL | 범위 외 | 유효 표본이 파라미터 수를 지탱하지 못함 |

핵심은 raw row 수(수십만)가 아니라 **유효 독립 표본 수**입니다. 라벨이 forward-h 수익률이라 시점 간 중첩되고, 국내 ETF 는 기초지수가 겹쳐 단면 상관이 높습니다. 상세 산정과 그로부터 도출된 모델 용량 상한은 [12-ml-layer.md](12-ml-layer.md) 를 참조하십시오.

ML 의 위치도 고정입니다. LightGBM Ranker 는 `AlphaModel` Protocol 을 구현하는 **drop-in 교체 대상**이며, portfolio·tournament 계층은 ML 존재를 알지 못합니다. ARCH-1(Signal ≠ Portfolio)이 그대로 유지됩니다.

### 4.5 파라미터는 코드가 아니라 config

가중치·임계값·horizon 은 `configs/*.yaml` 에만 존재합니다. 코드에 magic number 를 박지 않습니다.

---

## 5. 연구 질문 10개

프로젝트의 본체는 아래 질문에 데이터로 답하는 것입니다.

| # | 질문 | 검증 대상 |
| --- | --- | --- |
| 1 | 5/10/20일 중 어느 momentum horizon 이 좋은가? | Momentum |
| 2 | 절대 모멘텀보다 상대 모멘텀이 강한가? | Ranking |
| 3 | B4 실패 후 대표 ETF + lifecycle 로만 재시도 시 주도섹터가 유효한가? | Sector |
| 4 | momentum acceleration 이 미래수익을 설명하는가? | Early leadership |
| 5 | breadth 가 추세 지속 가능성을 설명하는가? | Breadth |
| 6 | 거래대금/자금유입이 추가 alpha 인가? | Flow |
| 7 | risk-on 에서 레버리지 기대효과는? | Regime |
| 8 | Top1 집중이 Top3 보다 tournament 에서 나은가? | Sizing |
| 9 | 언제 빠져야 하는가? | Exit |
| 10 | 언제 다시 진입해야 하는가? | Re-entry |
| 11 | 같은 기초지수에서 배수(1x/2x/-1x/-2x)를 언제 바꿔야 하는가? | Exposure |
| 12 | LightGBM Ranker 가 purged walk-forward 에서 rule baseline 을 이기는가? | ML |

11번은 레버리지 허용을 전제로 새로 생긴 질문입니다. 12번은 **기각이 기본값**입니다 — ML 은 rule baseline 을 이기지 못하면 버립니다.

---

## 6. 검증 파이프라인

```
Hypothesis
  → Feature
  → Simple Strategy
  → Structural Backtest
  → Deployment Backtest
  → Rolling-36D 분포
  → Robustness grid (유동성·비용)
  → 2025 Tournament Replay
  → Accept / Reject
```

**결과가 안 좋으면 버립니다.** 프로젝트 목적은 많은 알고리즘 구현이 아니라 **작동하지 않는 아이디어를 빠르게 제거**하는 것입니다.
