가능합니다. 다만 이 프로젝트를 처음부터 **“ETF 추천 AI”**로 정의하면 방향이 흐려집니다. 더 정확한 정의는 다음입니다.

> **40거래일 안팎의 ETF 모의투자대회에서, 특정 시점의 시장·섹터 상태를 데이터로 판단하고 우승 확률을 높이는 포지션을 제안하는 Tournament Quant Research System**

2025년 우승자는 사이클 방향 판단, 강한 섹터 집중, 레버리지 활용, 재진입 기준을 중시했고, 대회 중 상당 기간 전체 상위권을 자율형이 독식했습니다. 따라서 반도체를 하드코딩하는 게 아니라 **“그때그때 강한 섹터를 찾아 집중하는 과정” 자체를 시스템화**하는 것이 설계의 중심이어야 합니다. ([머니투데이][1])

---

# 1. 프로젝트의 목적부터 명확하게 고정

### Primary Objective

```text
8주 대회에서 전체 1위 / 2위가 될 확률 극대화
```

일반적인

```text
maximize CAGR
maximize Sharpe Ratio
minimize MDD
```

가 아닙니다.

연구 목표는 다음에 더 가깝습니다.

$$
P(R_{40d} > R_{competitors})
$$

또는 실제 상금을 고려해서 나중에는

$$
E[Prize]
=
1000P(rank=1)
+
500P(rank=2)
+
100P(category=1)
$$

같은 Tournament utility를 정의할 수도 있습니다.

단, 참가자 분포를 정확히 모르므로 `P(rank=1)`은 처음부터 정확한 통계량으로 간주하면 안 됩니다.

**먼저 전략 자체의 40일 return distribution을 만드는 게 우선**입니다.

---

# 2. 프로젝트의 최종 구조

전체 시스템은 다음 8계층으로 나누는 것을 권합니다.

```text
┌─────────────────────────────────────────────┐
│  1. External Data Sources                  │
│ KRX / ETF / Stock / Index / Macro / Flow   │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  2. Data Lake                              │
│ raw → normalized → point-in-time           │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  3. Universe Engine                        │
│ "오늘 실제로 매매 가능한 ETF는 무엇인가?" │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  4. Feature Engine                         │
│ Momentum / Flow / Breadth / Vol / Regime   │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  5. Alpha / Ranking Engine                 │
│ Rule / Factor / ML ranking                 │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  6. Portfolio Policy                       │
│ ETF selection / leverage / concentration   │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  7. Tournament Simulator                   │
│ Rolling 40D / walk-forward / stress test   │
└─────────────────────┬───────────────────────┘
                      ▼
┌─────────────────────────────────────────────┐
│  8. Live Decision Engine                   │
│ regime / rankings / portfolio / rationale  │
└─────────────────────────────────────────────┘
```

여기서 **5번 Alpha 모델만 만드는 프로젝트가 되어서는 안 됩니다.**

1~7이 먼저 제대로 되어 있어야 5번을 믿을 수 있습니다.

---

# 3. 대회 규칙 자체를 Config로 만들어야 함

현재 공식적으로 확정된 것은:

* 2026-09-21 ~ 2026-11-13
* 초기자금 10억원
* 후원 운용사 ETF만 가능
* 자율형은 투자자산 제한 없음
* 자율형 이외는 레버리지·인버스 제외
* 전체 수익률 1·2위 대상/최우수상
* HTS에 부문별 허용 ETF가 사전 설정될 예정

입니다. ([mt.co.kr][2])

[2026 ETF 투자왕 공식 페이지](https://www.mt.co.kr/etf?utm_source=chatgpt.com)

이를 코드에 박아 넣지 말고:

```yaml
# configs/tournament_2026.yaml

tournament:
  start_date: 2026-09-21
  end_date: 2026-11-13
  initial_capital: 1_000_000_000

category:
  name: autonomous
  leverage_allowed: true
  inverse_allowed: true

execution:
  commission_bps: null
  slippage_bps: null

universe:
  source: official_manifest
```

처럼 둡니다.

`null`이 중요한 이유는 **아직 정확히 모르는 것을 임의로 가정하지 않기 위해서**입니다.

---

# 5. 지금 반드시 확인해야 할 미확정 규칙

프로젝트가 실제 HTS를 받은 이후 확인해야 합니다.

```text
□ 거래 수수료 적용 여부
□ 세금 적용 방식
□ ETF 분배금 처리
□ 평가수익률 정확한 계산 방식
□ 미실현손익 포함 여부
□ 주문 종류
□ 지정가 / 시장가
□ 일일 거래 제한
□ ETF 최대 보유 비중
□ 현금 100% 가능 여부
□ 거래정지 ETF 처리
□ LP/호가 체결 시뮬레이션 방식
□ 레버리지 ETF 제한 여부
```

따라서 simulator가 이 값을 parameter로 받아야 합니다.

---

# 6. 데이터 소스 — 최우선은 KRX

## A. KRX Open API

여기가 Core Source입니다.

KRX 공식 Open API는 현재 ETF 일별매매정보를 **2010년 1월 4일부터** 제공합니다. 지수, KOSPI/KOSDAQ 주식, ETF, 선물 등의 API도 제공됩니다. 인증키와 개별 서비스 이용 승인이 필요합니다. ([Open API][3])

[KRX Open API](https://openapi.krx.co.kr?utm_source=chatgpt.com)

[KRX Open API 서비스 목록](https://openapi.krx.co.kr/contents/OPP/INFO/service/OPPINFO004.cmd?utm_source=chatgpt.com)

--> .env.enc 에 api 키 존재함.

---

# 7. KRX Data Marketplace도 같이 사용

Open API보다 더 다양한 ETF 정보가 있습니다.

KRX Data Marketplace에는 공식적으로:

* ETF 전종목 시세
* ETF 기본정보
* 개별종목 정보
* 투자자별 거래실적
* PDF(Portfolio Deposit File)
* 액티브 ETF 편입자산
* 추적오차
* 괴리율
* 지수 구성종목

등이 있습니다. ([한국거래소][4])

[KRX Data Marketplace](https://data.krx.co.kr?utm_source=chatgpt.com)

특히 **PDF(Portfolio Deposit File)**가 유용합니다.

ETF 자체만 보는 게 아니라:

```text
ETF
 ↓
실제 구성종목
 ↓
삼성전자
SK하이닉스
한미반도체
...
```

까지 내려가서 주도 섹터의 내부 breadth를 계산할 수 있기 때문입니다.

---

# 8. pykrx는 Secondary Adapter

Python에서는 `pykrx`가 굉장히 편리합니다.

ETF OHLCV뿐 아니라 KRX ETF PDF, 투자자별 거래실적, 괴리율 등의 wrapper가 구현되어 있습니다. ([GitHub][5])

다만 구조상 KRX 웹페이지를 scraping/wrapping하는 부분이 존재하므로:

```text
Primary   = KRX 공식 API
Fallback  = pykrx
```

구조로 하는 편을 권합니다.

provider interface를 둡니다.

```python
class ETFDataProvider(Protocol):

    def get_prices(self, ...): ...
    def get_universe(self, ...): ...
    def get_holdings(self, ...): ...
```

그리고

```text
KRXOpenAPIProvider
KRXWebProvider
PykrxProvider
```

등을 교체 가능하게 만듭니다.

---

# 9. ETF CHECK는 탐색용으로 사용

코스콤 ETF CHECK는 꽤 좋은 참고자료입니다.

현재도 ETF:

* 수익률
* 거래량
* 자금유입
* 순자산
* 투자자
* iNAV

등을 비교할 수 있습니다. ([ETF CHECK][6])

[코스콤 ETF CHECK](https://www.etfcheck.co.kr?utm_source=chatgpt.com)

다만 ETF CHECK는 사이트에서 **사전 허가 없는 DB화에 대한 제한을 명시하고 있습니다.** 따라서 대량 scraper를 프로젝트 Core datasource로 삼는 건 권하지 않습니다. ([ETF CHECK][6])

용도는:

```text
아이디어 탐색
데이터 cross-check
ETF 분류 확인
시장 모니터링
```

정도로 두는 편이 안전합니다.

---

# 10. Macro source

한국:

[한국은행 ECOS Open API](https://ecos.bok.or.kr/api/?utm_source=chatgpt.com)

미국 및 글로벌 macro:

[Federal Reserve FRED API](https://fred.stlouisfed.org/docs/api/fred/overview.html?utm_source=chatgpt.com)

FRED에는 특히 **ALFRED / vintage data** 개념이 있어서 당시 발표되어 있던 데이터만 재구성할 수 있습니다. Macro feature에서 revised data를 그대로 과거 백테스트에 사용해 발생하는 look-ahead를 줄이는 데 유용합니다. ([FRED][7])

--> ECOS, FRED api 키 존재함.

---

# 11. 실시간 데이터가 필요해진다면

초기 프로젝트에서는 필요 없습니다.

EOD 기반으로 먼저 검증합니다.

나중에:

```text
signal = 전일 종가 이후 계산
execution = 다음날
```

대신 intraday까지 내려가고 싶으면 한국투자증권 Open API 같은 broker API를 붙일 수 있습니다.

REST와 WebSocket 실시간 체결/호가 API가 제공됩니다. ([Korea Investment API Portal][8])

[한국투자 Open API 개발자센터](https://apiportal.koreainvestment.com?utm_source=chatgpt.com)

다만 **이번 프로젝트 MVP에는 넣지 않는 게 낫습니다.**

---

# 12. 필요한 데이터 구조

## `instrument_master`

ETF에 대한 정적/준정적 정보입니다.

```text
ticker
name
issuer
listing_date
delisting_date

asset_class
geography
sector
theme

underlying_index

leverage_multiple
is_inverse
is_synthetic
is_currency_hedged
is_active

expense_ratio

tournament_category
tournament_eligible
```

중요한 컬럼:

```text
eligible_from
eligible_to
```

도 고려하십시오.

---

# 13. `etf_daily`

최소:

```text
date
ticker

open
high
low
close

volume
trading_value

nav
aum
shares_outstanding

premium_discount
```

가능한 것만 채웁니다.

없는 데이터는 억지로 만들지 않습니다.

---

# 14. `holdings_daily`

가능하면 강력합니다.

```text
date
etf_ticker
constituent_ticker
weight
shares
```

이 데이터가 있으면:

```text
ETF momentum
```

뿐만 아니라

```text
ETF 내부 구성종목 breadth
```

를 계산할 수 있습니다.

예:

```text
ETF 수익률: +6%

구성종목:
20일 신고가 비율       78%
MA20 이상 비율         84%
5일 상승 종목 비율     91%
```

이라면 상승추세의 질이 상당히 강하다고 볼 수 있습니다.

---

# 15. 시장 데이터

```text
KOSPI
KOSDAQ
KOSPI200
KOSDAQ150

sector indices

USD/KRW

gold
oil

US index
US tech index
semiconductor proxy

interest rates
```

전부 처음부터 필요하지는 않습니다.

---

# 16. 데이터 저장 구조

개인 퀀트 프로젝트라면 DB 서버를 크게 만들 필요 없습니다.

나는:

```text
Parquet + DuckDB
```

를 권합니다.

```text
data/
├── raw/
│   ├── krx/
│   ├── fred/
│   └── ecos/
│
├── normalized/
│   ├── instruments/
│   ├── prices/
│   ├── holdings/
│   └── macro/
│
└── features/
```

Raw data는 절대로 수정하지 않습니다.

```text
Bronze = 원본
Silver = 정제
Gold   = features
```

형태가 적당합니다.

---

# 17. 매우 중요한 Universe Engine

이 프로젝트의 핵심 중 하나입니다.

```python
universe = universe_provider.get(
    date="2026-10-01",
    category="autonomous"
)
```

결과:

```python
[
    "A ETF",
    "B ETF",
    "C ETF",
    ...
]
```

### 왜 분리하나?

2026 실제 허용 ETF 목록은 아직 정확히 공개되지 않았기 때문입니다.

또한 과거에는 존재하지 않던 ETF를 과거 백테스트에서 선택해서도 안 됩니다.

```python
listing_date <= t < delisting_date
```

를 반드시 만족해야 합니다.

---

# 18. 여기서 Survivorship Bias가 가장 큰 문제

예를 들어 지금 존재하는

```text
반도체 레버리지
원자력
방산
로봇
AI전력
```

ETF만 가지고 2015년부터 백테스트하면 사실상 잘못된 실험입니다.

따라서 두 종류 백테스트를 분리하십시오.

### Deployment backtest

```text
현재 실제 거래 가능한 ETF들만 분석
```

목적:

> 이번 대회에 실제 사용할 전략 검증

### Structural backtest

```text
각 역사적 시점에 존재했던 ETF universe 사용
```

목적:

> 전략 아이디어 자체가 장기적으로 존재했는가?

둘을 섞으면 안 됩니다.

---

# 19. Feature Engine

이제 핵심입니다.

Feature를 크게 7개 그룹으로 나눕니다.

```text
PRICE
TREND
MOMENTUM
VOLATILITY
FLOW
BREADTH
REGIME
```

---

# 20. Momentum

기본:

$$
mom_k(t)=\frac{P_t}{P_{t-k}}-1
$$

다음 horizon을 테스트합니다.

```text
3D
5D
10D
20D
40D
60D
```

40일 대회라는 이유로 40일만 보면 안 됩니다.

실제로는 **5~20일 leadership rotation**이 중요할 가능성이 있습니다.

---

# 21. Cross-sectional Relative Strength

절대수익률보다 중요할 수 있습니다.

같은 날 모든 ETF를 비교합니다.

```text
ETF A 20D return = +14%
ETF B            = +8%
ETF C            = +2%
...
```

이를 percentile화:

```text
A = 98%
B = 85%
C = 61%
```

즉:

```python
rs_20 = percentile_rank(momentum_20)
```

---

# 22. Momentum acceleration

아주 중요한 후보 feature입니다.

단순히 많이 오른 것과 **최근 갑자기 강해지는 것**을 구분합니다.

예:

```python
momentum_accel =
    zscore(mom_5)
    - zscore(mom_20)
```

또는

```text
5D rank ↑
10D rank ↑
20D rank →
```

같은 leadership emergence를 탐지합니다.

---

# 23. Trend

기본:

```text
close / MA20
close / MA60

MA5 slope
MA20 slope
```

그리고:

```text
20D high breakout
40D high breakout
```

---

# 24. Volatility

```text
ATR
realized volatility 5D
realized volatility 20D
downside volatility
gap volatility
```

여기서 중요한 건

> 저변동 ETF를 찾는 것

이 아닙니다.

오히려

> **수익률을 크게 낼 수 있을 만큼 변동성이 있지만 추세가 있는 ETF**

를 찾는 용도입니다.

---

# 25. Flow

가능하다면 상당히 유용합니다.

예:

```text
ETF 거래대금 증가
ETF 순자산 변화
설정좌수 변화
외국인 순매수
기관 순매수
```

단순 AUM 변화는 가격 상승 때문에 증가할 수 있으므로:

```text
AUM change ≠ capital inflow
```

라는 점에 주의해야 합니다.

가능하면 설정좌수 또는 creation/redemption을 이용합니다.

---

# 26. Breadth

주도 섹터 판정에서 매우 중요하게 연구할 가치가 있습니다.

ETF 구성종목을 이용해:

```text
breadth_ma20
= 구성종목 중 Price > MA20 비율

breadth_positive_5d
= 구성종목 중 5D return > 0 비율

breadth_high_20
= 20D 신고가 근접 종목 비율
```

등을 만듭니다.

---

# 27. Sector Leadership Engine

ETF를 단순 ticker로 보면 안 됩니다.

```text
반도체
AI
로봇
자동차
방산
조선
원전
전력
바이오
2차전지
금
은
중국
NASDAQ
...
```

등의 **Theme/Sector graph**를 구축합니다.

예:

```text
Semiconductor
 ├─ ETF A
 ├─ ETF B
 ├─ ETF C
 └─ ETF D
```

---

# 28. Sector Score

초기 연구모델은 단순하게 시작합니다.

```text
SectorScore =

0.30 × Relative Strength
+0.20 × Momentum Acceleration
+0.15 × Breakout
+0.15 × Breadth
+0.10 × Volume Expansion
+0.10 × Flow
```

단, 이 숫자를 정답으로 보지 마십시오.

**초기 hypothesis일 뿐입니다.**

AI tool이 이 가중치를 그대로 production에 박아 넣으면 안 됩니다.

---

# 29. Regime Engine

시장 전체 환경을 별도로 판단합니다.

처음에는 ML보다 rule-based가 좋습니다.

예:

```text
KOSPI > MA20
KOSPI MA20 slope > 0

KOSDAQ > MA20

market breadth > 60%

realized volatility < threshold
```

등으로

```text
STRONG_RISK_ON
RISK_ON
NEUTRAL
RISK_OFF
STRONG_RISK_OFF
```

5단계 정도로 만듭니다.

---

# 30. 이후 ML Regime Detector

rule baseline이 만들어지고 나면:

```text
Logistic Regression
Random Forest
XGBoost
LightGBM
HMM
```

등을 비교할 수 있습니다.

처음부터 deep learning은 권하지 않습니다.

40일 투자대회용 ETF 전략에서 LSTM/Transformer가 복잡도만 키우고 실제 out-of-sample 성능은 더 나쁠 가능성도 충분합니다.

---

# 31. Alpha Engine은 “가격 예측”보다 Ranking

나는 이걸 강하게 권합니다.

문제 정의를:

```text
내일 수익률은 몇 %인가?
```

가 아니라

> **현재 ETF 중 앞으로 5~10일 동안 가장 강할 ETF는 무엇인가?**

로 잡습니다.

---

# 32. Label

예:

```python
future_return_5d
future_return_10d
future_return_20d
```

그리고 cross-sectional rank:

```python
target_10d_rank =
    percentile_rank(future_return_10d)
```

---

# 33. ML을 붙인다면

가장 적합한 시작점은:

```text
LightGBM Ranker
```

후보입니다.

입력:

```text
mom_3
mom_5
mom_10
mom_20
mom_40

rs_5
rs_20

momentum_acceleration

volume_ratio
turnover_ratio

rv_5
rv_20
atr

breakout_20
drawdown_20

sector_rs
sector_breadth

market_regime
USD/KRW
```

출력:

```text
ETF ranking score
```

---

# 34. 반드시 Rule Model을 Baseline으로 둬야 함

ML 결과가

```text
+22%
```

를 기록해도 의미 없습니다.

Rule baseline이:

```text
+27%
```

이면 ML을 버리는 게 맞습니다.

반드시 비교:

```text
Baseline 0: Buy & Hold KOSPI
Baseline 1: Top-1 20D Momentum
Baseline 2: Top-3 Momentum
Baseline 3: Momentum + Trend
Baseline 4: Sector Momentum
Baseline 5: Sector + Regime

Model 1: LightGBM
Model 2: Ranker
Model 3: Ensemble
```

---

# 35. Portfolio Policy

Alpha score를 position으로 바꾸는 별도 계층이 필요합니다.

```text
Signal ≠ Portfolio
```

입니다.

예:

```text
ETF A 95
ETF B 91
ETF C 72
```

라고 해서 자동으로 동일가중할 이유는 없습니다.

실험:

```text
Top1          100%

Top2          70 / 30

Top3          50 / 30 / 20

Top3 Equal    33 / 33 / 33
```

를 모두 비교해야 합니다.

---

# 36. Confidence 기반 concentration

예를 들어:

```text
rank1 = 95
rank2 = 62
rank3 = 58
```

이면 leadership이 명확합니다.

반면:

```text
rank1 = 82
rank2 = 81
rank3 = 80
```

이면 확신이 낮습니다.

따라서

```python
confidence =
    score_rank1 - score_rank2
```

같은 simple dispersion부터 연구할 수 있습니다.

---

# 37. 레버리지 ETF 처리

중요합니다.

레버리지 ETF는 별도 상품으로 취급합니다.

```text
ETF A
ETF A leverage
```

를 같은 asset으로 간주하면 안 됩니다.

또한 장기 수익률을

```python
underlying_return * 2
```

로 만들면 잘못됩니다.

daily reset과 compounding 때문에 실제 ETF 가격 시계열을 우선 사용합니다.

---

# 38. 비슷한 ETF 중복 제거

예:

```text
반도체 ETF A
반도체 ETF B
반도체 ETF C
```

가 모두 ranking 1~3위를 차지할 수 있습니다.

실질적으로 같은 bet입니다.

따라서:

```text
underlying index
sector
holdings overlap
return correlation
```

등을 이용해 Cluster를 만들 필요가 있습니다.

예:

```text
Semiconductor Cluster
    A
    B
    C
```

그중 실제 선택은:

```text
alpha
liquidity
tracking
spread
```

기준으로 합니다.

---

# 39. 유동성 필터도 필수

초기자금이 **10억원**입니다.

매우 작은 ETF에 10억원을 몰아넣는 백테스트는 허상일 수 있습니다.

따라서:

```text
daily trading value
ADV20
```

기반으로 filter합니다.

예를 들어 parameter:

```yaml
liquidity:
  min_adv_20: ...
  max_order_to_adv: 0.01
```

`1%`가 정답이라는 뜻이 아닙니다.

```text
1%
2%
5%
10%
```

stress test를 해야 합니다.

---

# 40. 백테스트의 가장 중요한 체결 규칙

Daily feature를 종가 기준으로 만들었다면:

```text
2026-10-01 종가 데이터
        ↓
Signal calculation
        ↓
2026-10-02 매매
```

해야 합니다.

```text
10/1 종가를 보고
10/1 종가 체결
```

하면 look-ahead입니다.

처음에는 가장 보수적으로:

```text
Signal at close(t)
Execute at open(t+1)
```

을 추천합니다.

---

# 41. Tournament Backtester

일반 백테스터에 이 기능을 추가하는 게 아니라 별도 클래스로 만드는 게 좋습니다.

```python
TournamentSimulator(
    duration=40,
    initial_capital=1_000_000_000
)
```

---

# 42. Rolling Tournament 방식

예:

```text
2018-01-02 → +40D
2018-01-03 → +40D
2018-01-04 → +40D
...
```

수천 번의 가상 대회를 진행합니다.

그런데 window가 겹치므로 서로 독립 표본이라고 착각하면 안 됩니다.

---

# 43. 평가 Metric

CAGR보다 다음을 출력합니다.

```text
Mean 40D Return
Median 40D Return

75th percentile
90th percentile
95th percentile
99th percentile

P(Return > 10%)
P(Return > 20%)
P(Return > 30%)
P(Return > 40%)
P(Return > 50%)

MDD
Worst 5%
```

특히:

```text
95th percentile
P(R > 30%)
P(R > 40%)
```

를 중요하게 봅니다.

---

# 44. 여기서 새로운 핵심 Metric

나는 `Right Tail Score` 같은 내부 metric도 만들겠습니다.

예:

```python
tail_score =
    0.2 * q75
  + 0.3 * q90
  + 0.3 * q95
  + 0.2 * q99
```

역시 가중치는 연구용입니다.

목적은:

> 평균 수익률은 낮지만 대박이 자주 발생하는 전략

을 놓치지 않는 것입니다.

---

# 45. 하지만 Tail만 최적화하면 위험

여기에 중요한 반대 검증이 있습니다.

예:

```text
전략 A

5% 확률 +100%
95% 확률 -30%
```

이런 전략은 tail이 엄청납니다.

하지만 실제 우승 가능성이 높은지는 별개입니다.

따라서 반드시:

```text
return distribution 전체
```

를 같이 봐야 합니다.

---

# 46. Tournament Monte Carlo

나중에는 경쟁자를 simulation합니다.

예:

```text
N = 500
N = 1000
N = 2000
```

참가자를 생성하고 내 전략을 한 명 넣습니다.

```python
for simulation in range(10000):

    competitors = sample_competitor_returns()

    my_return = strategy_return()

    rank = calculate_rank(...)
```

출력:

```text
P(rank == 1)
P(rank <= 2)
P(rank <= 10)
```

---

# 47. 단, 이 부분은 과신하면 안 됨

2025년 대회는 약 1000명이 참가했습니다. ([머니투데이][9])

하지만 참가자 return distribution을 제대로 알고 있는 게 아닙니다.

따라서 이 결과는:

```text
"우승 확률 4.72%"
```

같은 정밀한 예측으로 표현하면 안 됩니다.

대신:

```text
aggressive competitor scenario
normal scenario
weak competitor scenario
```

의 stress testing으로 사용합니다.

---

# 48. 2025 실제 데이터를 Benchmark로 이용

2025년은 굉장히 좋은 테스트 케이스입니다.

5주차:

* 1위 +72.28%
* 상위 5명 전부 자율형

최종 우승:

* +47.82%

대회 후반 조정으로 순위와 수익률이 크게 변했습니다. ([머니투데이][1])

따라서 중요한 검증 질문:

> **우리 모델을 2025년 대회 시작일 전 데이터만 가지고 실행했다면 반도체/주도 섹터를 잡았는가?**

입니다.

이걸 별도의 historical case study로 만드십시오.

---

# 49. 아주 좋은 Integration Test

```text
2025-09-22
```

시점에서 코드가 미래를 전혀 모르는 상태로 시작합니다.

하루씩 데이터를 공개합니다.

```text
09/22
09/23
09/24
...
11/14
```

그리고 모델이 실제로 어떻게 행동했는지 기록합니다.

이 방식은 단순 백테스트보다 훨씬 디버깅하기 좋습니다.

---

# 50. Re-entry 모델도 중요

2025년 우승자가 특히 강조한 부분 중 하나가 **손절 기준보다 재진입 기준**입니다. ([머니투데이][10])

따라서 단순:

```text
trend broken → sell
```

만 만들면 부족합니다.

```text
exit
 ↓
watch state
 ↓
momentum recovery
 ↓
breadth recovery
 ↓
re-entry
```

상태 머신을 연구할 가치가 있습니다.

---

# 51. 전략을 State Machine으로 만드는 방법

```text
DISCOVERY
   ↓
EMERGING
   ↓
LEADING
   ↓
OVERHEATED
   ↓
BREAKDOWN
   ↓
RECOVERY
```

ETF 또는 sector마다 상태를 가집니다.

예:

```text
Semiconductor = LEADING
Robot         = EMERGING
Defense       = BREAKDOWN
Gold          = RECOVERY
```

이 방식이 단순 점수보다 사람이 분석할 때도 훨씬 유용할 수 있습니다.

---

# 52. 대회용 Tournament Overlay

전략 그 자체와 분리하십시오.

```text
Alpha Strategy
       +
Tournament Policy
```

입니다.

예:

```text
현재 4주차
내 수익률 +25%
1위 +27%
```

→ 굳이 극단적인 risk를 쓸 이유가 적습니다.

반면:

```text
현재 7주차
내 수익률 +20%
1위 +45%
```

→ 방어적 포트폴리오는 대상을 받을 가능성이 거의 없습니다.

---

# 53. Tournament aggressiveness

입력:

```text
days_remaining
current_rank
return_gap_to_1st
return_gap_to_2nd
strategy_confidence
```

출력:

```text
risk_multiplier
concentration
```

예:

```text
Leader + 마지막 주
→ risk ↓

Middle + 초반
→ normal

Middle + 마지막 주
→ risk ↑

Far behind + 마지막 주
→ extreme tail strategy
```

하지만 이건 **마지막 단계**입니다.

Alpha 자체가 검증되기 전에 만들면 오히려 잡음입니다.

---

# 54. 추천 Repository 구조

```text
etf-tournament-alpha/
│
├── AGENTS.md
├── README.md
├── pyproject.toml
│
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   ├── strategies.yaml
│   └── tournament_2026.yaml
│
├── docs/
│   ├── PROJECT_OBJECTIVE.md
│   ├── TOURNAMENT_RULES.md
│   ├── DATA_CONTRACT.md
│   ├── UNIVERSE_SPEC.md
│   ├── FEATURE_SPEC.md
│   ├── BACKTEST_PROTOCOL.md
│   └── STRATEGY_RESEARCH.md
│
├── src/
│   ├── data/
│   │   ├── providers/
│   │   │   ├── krx.py
│   │   │   ├── ecos.py
│   │   │   └── fred.py
│   │   ├── normalize.py
│   │   └── validation.py
│   │
│   ├── universe/
│   │   ├── provider.py
│   │   └── tournament.py
│   │
│   ├── features/
│   │   ├── momentum.py
│   │   ├── trend.py
│   │   ├── volatility.py
│   │   ├── breadth.py
│   │   ├── flow.py
│   │   └── regime.py
│   │
│   ├── alpha/
│   │   ├── factor_ranker.py
│   │   ├── sector_ranker.py
│   │   └── ml_ranker.py
│   │
│   ├── portfolio/
│   │   ├── selection.py
│   │   ├── sizing.py
│   │   └── constraints.py
│   │
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── execution.py
│   │   ├── costs.py
│   │   └── metrics.py
│   │
│   ├── tournament/
│   │   ├── simulator.py
│   │   ├── leaderboard.py
│   │   └── policy.py
│   │
│   └── reporting/
│       ├── dashboard.py
│       └── research_report.py
│
├── tests/
│
├── notebooks/
│   └── exploratory/
│
└── data/
```

---

# 55. AI Coding Agent에게 특히 강제할 원칙

`AGENTS.md`에는 최소한 다음을 넣는 게 좋습니다.

```text
PROJECT GOAL
Build a research system for an approximately 40-trading-day
ETF tournament. Do not optimize for long-term CAGR by default.

CORE RULES

1. Never use future information in features.
2. All datasets must have explicit timestamps.
3. Universe must be point-in-time.
4. Newly listed ETFs cannot exist before listing.
5. Generate EOD signals only using information available at that time.
6. Default execution is the next tradable session.
7. Separate signal generation from portfolio construction.
8. Separate alpha strategy from tournament aggression policy.
9. Every new strategy must be compared against simple baselines.
10. No ML model is accepted solely because in-sample performance improves.
11. Every experiment must record parameters, dataset version, and result.
12. Unknown tournament rules must remain configurable, never guessed.
```

이 정도는 AI tool에게 계속 보이게 해두는 것을 권합니다.

---

# 56. 구현 순서

처음부터 모든 걸 만들지 마십시오.

## Stage 1 — Research Foundation

먼저:

```text
KRX ingestion
ETF master
OHLCV
point-in-time universe
feature calculator
simple backtester
```

까지만 완성합니다.

**성공 조건**

```text
어떤 날짜를 주면
그 시점에 존재하던 ETF 목록을 만들고
과거 데이터만으로 signal을 생성하여
다음날 거래할 수 있음
```

---

# 57. Stage 2 — Baseline Tournament Research

전략:

```text
Top1 Momentum
Top3 Momentum
Momentum + Trend
Momentum + Breakout
Sector Momentum
```

만 테스트합니다.

아직 ML 금지.

---

# 58. Stage 3 — Leadership Engine

추가:

```text
Sector taxonomy
Breadth
Momentum acceleration
Volume
Flow
```

그리고

```text
Leading Sector Detector
```

를 완성합니다.

나는 **여기까지가 가장 중요한 MVP**라고 봅니다.

---

# 59. Stage 4 — Regime

```text
Market Regime
+
Sector Leadership
```

을 합칩니다.

```text
Risk-on → leverage momentum
Neutral → reduced concentration
Risk-off → cash / defensive / inverse 후보
```

---

# 60. Stage 5 — ML

여기서 처음:

```text
LightGBM
LightGBM Ranker
XGBoost
```

등을 추가합니다.

Rule system보다 실제 walk-forward에서 개선되는지 봅니다.

---

# 61. Stage 6 — Tournament Layer

마지막:

```text
remaining days
leaderboard
rank gap
```

을 이용한 aggression control.

---

# 62. Stage 7 — 실전 Dashboard

최종적으로 매일 보고 싶은 화면은 복잡할 필요 없습니다.

```text
==================================================
ETF TOURNAMENT ALPHA
2026-10-07
==================================================

MARKET REGIME
RISK_ON                     0.82 confidence

SECTOR LEADERS
1. Semiconductor            93.2
2. Robotics                 87.3
3. Nuclear                  76.8
4. Defense                  66.1

LEADERSHIP CHANGES
Robotics      ↑  5 → 2
Defense       ↓  2 → 4

ETF RANKING
1. ETF_A                    94.2
2. ETF_B                    89.8
3. ETF_C                    81.5

RECOMMENDED PORTFOLIO
ETF_A                       60%
ETF_B                       30%
Cash                        10%

SIGNAL
Momentum                    STRONG
Breadth                     STRONG
Flow                        MODERATE
Volatility                  HIGH

TOURNAMENT
Current Rank                13
Days Remaining              24
Aggression                  NORMAL
==================================================
```

그 옆에:

```text
Why?
```

를 반드시 제공합니다.

---

# 63. LLM/AI는 어디에 사용해야 하는가

이 프로젝트에서 **LLM이 Alpha predictor일 필요는 없습니다.**

LLM은 오히려:

```text
연구 orchestration
실험 비교
결과 설명
데이터 오류 탐지
전략 hypothesis 생성
리포트 작성
```

에 쓰는 편이 좋습니다.

Quant engine은 deterministic Python 코드로 유지하는 걸 권합니다.

---

# 64. 나중에 AI를 활용할 수 있는 영역

후순위로는:

### 뉴스/Event NLP

```text
ETF 구성종목 뉴스
산업 뉴스
정책
실적
FOMC
산업 이벤트
```

를 받아

```text
sector catalyst score
```

를 만들 수 있습니다.

하지만 이것 역시 **가격/모멘텀 모델이 작동한 뒤 추가**하는 게 맞습니다.

뉴스 NLP부터 만들면 검증이 매우 어렵습니다.

---

# 65. 이 프로젝트에서 하지 말아야 할 것

초기에는 다음을 피하는 게 좋습니다.

```text
❌ LSTM 가격 예측
❌ Transformer 가격 예측
❌ 강화학습 portfolio agent
❌ 초단타 HFT
❌ 자동 주문 시스템
❌ 복잡한 NLP 뉴스 시스템
❌ 수백 개 indicator
❌ Bayesian hyperparameter optimization부터 시작
```

이것들은 기술적으로는 재미있지만 **우승 확률을 올리는 핵심 작업보다 먼저 할 이유가 없습니다.**

---

# 66. 내가 생각하는 최종 Research Stack

최종적으로는 아래 정도가 가장 균형이 좋습니다.

```text
                     ┌─────────────┐
                     │ KRX / Macro │
                     └──────┬──────┘
                            │
                      Point-in-Time
                            │
                            ▼
                ┌──────────────────────┐
                │ Market Regime       │
                └──────────┬───────────┘
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
     Momentum         Breadth/Flow       Volatility
          │                │                │
          └────────────────┼────────────────┘
                           ▼
                  Sector Leadership
                           │
                           ▼
                 Cross-sectional Rank
                           │
                     ┌─────┴─────┐
                     ▼           ▼
                  Factor       ML Ranker
                     └─────┬─────┘
                           ▼
                       Ensemble
                           │
                           ▼
                Portfolio Construction
                           │
                           ▼
                  Tournament Overlay
                           │
                           ▼
                    Daily Decision
```

---

# 67. 가장 중요한 연구 질문 10개

프로젝트에서는 결국 이것들을 데이터로 답하면 됩니다.

| 질문                                      | 검증 대상            |
| --------------------------------------- | ---------------- |
| 최근 5/10/20일 중 어느 momentum horizon이 좋은가? | Momentum         |
| 절대 모멘텀보다 상대 모멘텀이 강한가?                   | Ranking          |
| 주도섹터를 따라가는 전략이 ETF 개별 ranking보다 좋은가?    | Sector           |
| momentum acceleration이 미래수익을 설명하는가?     | Early leadership |
| breadth가 추세 지속 가능성을 설명하는가?              | Breadth          |
| 거래대금/자금유입이 추가 alpha인가?                  | Flow             |
| risk-on에서 레버리지의 기대효과가 얼마나 커지는가?         | Regime           |
| Top1 집중이 Top3보다 tournament에서는 나은가?      | Sizing           |
| 언제 빠져야 하는가?                             | Exit             |
| 언제 다시 진입해야 하는가?                         | Re-entry         |

**이 10개에 답하는 것이 프로젝트의 본체**라고 봐도 됩니다.

---

# 68. 특히 중요한 검증 순서

AI tool이 프로젝트를 만들면서 다음 논리를 유지하면 됩니다.

```text
Hypothesis
    ↓
Feature
    ↓
Simple Strategy
    ↓
Historical Test
    ↓
Walk-forward Test
    ↓
Robustness Test
    ↓
2025 Tournament Replay
    ↓
Accept / Reject
```

그리고

```text
결과가 안 좋으면 버린다.
```

가 중요합니다.

프로젝트 목적은 많은 알고리즘을 구현하는 게 아니라 **작동하지 않는 아이디어를 빠르게 제거하는 것**입니다.

---

## 최종 방향

현재 단계에서 설계를 한 문장으로 압축하면:

> **KRX 기반 point-in-time ETF 데이터베이스 위에 `Market Regime → Sector Leadership → ETF Cross-sectional Ranking → Concentrated Portfolio → Tournament Risk Overlay`를 얹고, 모든 판단을 rolling 40-day tournament와 walk-forward 방식으로 검증하는 연구 시스템을 만든다.**

그리고 프로젝트 안에서 가장 먼저 검증할 전략은:

```text
Regime-Adaptive
Sector Leadership
+
Cross-Sectional Momentum
+
Concentrated Leveraged Rotation
```

입니다.

다만 이것을 **정답 전략으로 확정하지 않는 것이 중요합니다.** 아키텍처는 이 전략이 실패해도 다른 `AlphaModel`, `PortfolioPolicy`, `RegimeModel`로 쉽게 갈아끼울 수 있도록 만들어야 합니다.

개발 관점에서는 다음 작업을 **1차 구현 단위**로 잡는 것이 가장 적절합니다.

```text
1. KRX API key 준비
2. repo skeleton + AGENTS.md
3. ETF master / OHLCV ingestion
4. point-in-time UniverseProvider
5. 3/5/10/20/40D momentum feature
6. cross-sectional rank
7. next-open backtester
8. rolling 40D tournament evaluator
9. Top1 / Top3 momentum baseline
10. 2025-09-22 대회 replay
```

여기까지 만든 뒤 나온 실제 결과를 보고 **Sector/Breadth/Regime/ML을 어느 순서로 추가할지 결정하는 것**이 가장 합리적입니다. 이 정도면 AI coding tool에게 프로젝트를 맡기기 위한 설계 기준도 충분히 명확해집니다.

[1]: https://www.mt.co.kr/stock/2025/10/25/2025102418201958573?utm_source=chatgpt.com "1등 수익률 무려 72%…TOP5 모두 '이 ETF' 담았다 - 머니투데이"
[2]: https://www.mt.co.kr/etf/join/index.html?utm_source=chatgpt.com "대회 참여 신청 | ETF 투자왕대회 - 머니투데이"
[3]: https://openapi.krx.co.kr/contents/OPP/INFO/service/OPPINFO004.cmd?utm_source=chatgpt.com "서비스 목록 - KRX | Data Marketplace OPEN API"
[4]: https://data.krx.co.kr/contents/MDC/MAIN/main/index.cmd?utm_source=chatgpt.com "KRX | KRX Data Marketplace"
[5]: https://github.com/sharebook-kr/pykrx/blob/master/.github/copilot-instructions.md?utm_source=chatgpt.com "pykrx/.github/copilot-instructions.md at master · sharebook-kr/pykrx · GitHub"
[6]: https://www.etfcheck.co.kr/?redirect=%2Fmobile%2Fkrx%2Fetpctg%2F0101%3FetpType%3DETF&utm_source=chatgpt.com "ETF CHECK"
[7]: https://fred.stlouisfed.org/docs/api/fred/licenses/fred.html?utm_source=chatgpt.com "St. Louis Fed Web Services: FRED® API"
[8]: https://apiportal.koreainvestment.com/docs?utm_source=chatgpt.com "KIS Developers : 한국투자증권 오픈API 개발자센터"
[9]: https://www.mt.co.kr/stock/2025/11/19/2025111814504029132?utm_source=chatgpt.com "두 달만에 48% 수익...투자왕이 선택한 ETF포트폴리오는 - 머니투데이"
[10]: https://www.mt.co.kr/stock/2025/11/25/2025112414165129357?utm_source=chatgpt.com "\"시장 흐름과 주도섹터 분석한 포트폴리오 적중\" ETF투자왕의 비결 - 머니투데이"
