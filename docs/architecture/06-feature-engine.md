# 06. Feature Engine — L3 계층

Silver 패널 위에 **point-in-time 보장** feature 를 생성합니다. 모든 alpha·backtest 의 입력입니다.

---

## 1. 설계 원칙

1. **Feature ≠ Signal** — feature 는 관측값, signal 은 alpha 모델이 생성
2. **PIT guard 강제** — `assert_pit(df, as_of)` 로 미래 누출 차단
3. **Session grid 정렬** — XKRX session 만, 비거래일 forward-fill 금지
4. **결측은 null 유지** — 0 대체 금지 (INV-1 연장)
5. **Config-driven window** — `mom_20` 의 20은 `configs/features.yaml`

---

## 2. PIT Guard

```python
# features/pit.py
def assert_pit(df: pl.DataFrame, as_of: date) -> None:
    """df.date.max() > as_of 이면 RuntimeError"""

def align_session_grid(
    df: pl.DataFrame,
    calendar: TradingCalendar,
    as_of: date,
    lookback: int,
) -> pl.DataFrame:
    """as_of 이전 lookback sessions 만 반환"""
```

모든 feature 함수는 입력 전 `assert_pit` 호출. 테스트에서도 동일.

---

## 3. Feature 그룹

### 3.1 Momentum (`momentum.py`)

| Feature | 정의 |
| --- | --- |
| `mom_{w}` | `close(t) / close(t-w) - 1` |
| `mom_accel` | `mom_5 - mom_20` |
| `rel_mom` | cross-sectional rank of `mom_20` |

Window: 3, 5, 10, 20, 40, 60 (config).

### 3.2 Trend (`trend.py`)

| Feature | 정의 |
| --- | --- |
| `ma_ratio_{w}` | `close / SMA(close, w)` |
| `breakout_{w}` | `close == max(close, w)` |
| `drawdown_{w}` | `(close - max(close,w)) / max(close,w)` |

### 3.3 Volatility (`volatility.py`)

| Feature | 정의 |
| --- | --- |
| `rv_{w}` | realized vol (daily returns std) |
| `downside_{w}` | downside deviation |
| `atr_{w}` | average true range proxy |

### 3.4 Flow (`flow.py`)

| Feature | 정의 |
| --- | --- |
| `creation_flow` | `ΔLIST_SHRS × NAV` |
| `flow_zscore` | creation_flow 의 z-score (PIT window) |
| `nav_disparity` | `(close - nav) / nav` |

`LIST_SHRS` + `NAV` 조합은 KRX 실측으로 정확한 설정/환매 분해가 가능합니다.

### 3.5 Breadth (`breadth.py`)

| Feature | 정의 |
| --- | --- |
| `mkt_breadth` | universe 내 `mom_20 > 0` 비율 |
| `cluster_breadth` | theme 내 positive momentum 비율 |
| `new_high_ratio` | `breakout_20` 비율 |

### 3.6 Cross-sectional (`crosssec.py`)

| Feature | 정의 |
| --- | --- |
| `rank_{feature}` | universe 내 percentile rank |
| `zscore_{feature}` | cross-sectional z-score |
| `rank_accel` | rank 변화율 |

### 3.7 Regime (`regime.py`)

5-state classifier:

| State | 조건 (개략) |
| --- | --- |
| `RISK_ON` | index mom > 0, breadth > 60% |
| `RISK_OFF` | index mom < 0, breadth < 40% |
| `ROTATION` | sector dispersion ↑, breadth 중립 |
| `CRISIS` | index drawdown > threshold |
| `NEUTRAL` | 그 외 |

입력: KRX 지수 daily + market breadth. regime 은 **portfolio overlay** 에만 사용 (alpha score 에 직접 섞지 않음).

---

## 4. FeatureBuilder

```python
builder = FeatureBuilder(config=load_features_config())
features = builder.build(
    silver=etf_daily,
    universe=universe_snapshot,
    as_of=date(2026, 8, 27),
)
# → etf_features.parquet (gold)
```

### 4.1 빌드 파이프라인

```
silver panel
  → filter universe (as_of)
  → align_session_grid
  → compute groups (parallel-safe)
  → join on (date, isu_cd)
  → assert no future dates
  → write gold
```

### 4.2 멱등성

동일 `(silver_hash, config_hash, as_of)` → 동일 출력. 재빌드 안전.

---

## 5. Feature Store

```
data/features/
  └── etf_features.parquet
      index: (date, isu_cd)
      columns: mom_5, mom_20, rv_20, creation_flow, rank_mom_20, ...
```

- Parquet, zstd compression
- 날짜별 partition 은 선택 (파일 크기 보고 결정)
- feature 추가 시 **기존 컬럼 불변** (append-only columns)

---

## 6. Look-ahead 방어 체크리스트

| 위험 | 방어 |
| --- | --- |
| 미래 가격 in window | `align_session_grid(as_of)` |
| 미래 universe | `universe.get_universe(as_of)` |
| 전체 샘플 mean/std | crosssec 는 **그날 단면** 만 |
| 휴장일 0 가격 | silver 에서 `""` → null, feature skip |
| 상장 전 데이터 | `first_seen` 필터 |

---

## 7. CLI

| Command | 동작 |
| --- | --- |
| `mt-etf features --as-of 2026-08-27` | gold feature 빌드 |
| `mt-etf features --validate-pit` | PIT guard 자기검증 |
