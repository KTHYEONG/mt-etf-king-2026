# 04. Point-in-Time Universe — instrument master · taxonomy · eligibility

**선행**: [03_silver_panel](03_silver_panel.md)
**상위**: [00_architecture.md](00_architecture.md)

---

## 1. Diagnosis & Invariants

### 1.1 "오늘 실제로 살 수 있는 ETF는 무엇인가"

이 질문의 답이 틀리면 그 위의 모든 alpha 연구가 무의미하다. 제약은 **모드에 따라 달라진다**.

$$
\mathcal{U}_{\text{struct}}(t) = \mathcal{E}(t) \cap \mathcal{P}(t) \cap \mathcal{H}(t) \cap \mathcal{L}(t)
$$

$$
\mathcal{U}_{\text{deploy}}(t) = \mathcal{E}(t) \cap \mathcal{P}(t) \cap \mathcal{H}(t) \cap \mathcal{S}(t) \cap \mathcal{L}(t) \cap \mathcal{R}(t)
$$

| 기호 | 의미 | structural | deployment |
| --- | --- | --- | --- |
| $\mathcal{E}$ | 존재 (패널에 있음) | ✓ | ✓ |
| $\mathcal{P}$ | 유효 가격 | ✓ | ✓ |
| $\mathcal{H}$ | 이력 충분 (warmup) | ✓ | ✓ |
| $\mathcal{S}$ | **후원 운용사 ETF** | skip | ✓ |
| $\mathcal{L}$ | 유동성 | ✓ | ✓ |
| $\mathcal{R}$ | 대회 규칙 (레버리지·manifest) | skip | ✓ |

**INV-04-1 (순서 고정)**: 필터는 반드시 `existence → price → history → sponsor → liquidity → eligibility` 순서로 적용하고, 각 단계의 탈락 수를 기록한다. `sponsor` 단계는 `mode=structural` 일 때 skip 한다.

**INV-04-8 (모드 분리)**: structural 결과만으로 전략 채택·ML·decide 를 하지 않는다. deployment 모드가 대회용 진실 공급원이다.

### 1.2 $\mathcal{E}$ — 존재: 상장일 API 가 없다는 사실의 귀결

**실측 확인**: `/etp/etf_isu_base_info` 는 HTTP 404 — **ETF 기본정보 API 자체가 존재하지 않는다.**

$$
\texttt{first\_seen}(i) = \min\{t : i \in \text{panel}(t)\}, \qquad
\texttt{last\_seen}(i) = \max\{t : i \in \text{panel}(t)\}
$$

**INV-04-2 (좌측 절단 표시)**: `first_seen == panel_start` 인 종목은 `left_censored=True`. warmup 필터에서 예외 처리한다.

### 1.3 $\mathcal{H}$ — 이력: warmup 없는 종목은 조용히 NaN 을 만든다

**INV-04-3**: `session_count(first_seen, t) ≥ warmup_sessions` 를 universe 단계에서 강제한다.

### 1.4 $\mathcal{S}$ — 후원사: deployment 전용

대회 규칙: **후원 운용사 ETF만 매매 가능**.

- 인프라 후원(금융투자협회·한국거래소·코스콤)은 ETF 발행 주체가 **아님**
- 운용사 10곳 ↔ ETF 브랜드: `configs/sponsor_brands.yaml`
- KRX API에 운용사 필드 없음 → `ISU_NM` 첫 토큰(브랜드)으로 `resolve_issuer()` 유도

**INV-04-9 (sponsor fail-closed)**: `issuer_whitelist` 활성 시 `issuer=UNKNOWN` 인 종목은 제외한다. 추측으로 포함하지 않는다.

**INV-04-10 (manifest 우선)**: `configs/universe_manifest.yaml` 에 HTS `isu_cd` 리스트가 있으면 브랜드 매핑보다 **manifest 가 우선**한다 (더 좁은 집합).

실측(2026-08-27): 후원사 1,077종 / 비후원 86종. 유동성 상위(ADV≥1,000억 26종, Top 20)는 **전부 후원사**이나, deployment 필터는 설계상 필수이다.

### 1.5 $\mathcal{L}$ — 유동성

| $\phi$ | 요구 ADV | 통과 종목 수 (전체=후원사, 실측) |
| --- | --- | --- |
| 1% | 1,000억 | **26** |
| 2% | 500억 | 41 |
| 5% | 200억 | 65 |
| 10% | 100억 | 121 |

**INV-04-4**: $\phi \in \{1, 2, 5, 10\}\%$ 그리드로 산출 가능해야 한다.

### 1.6 $\mathcal{R}$ — 대회 적격성

확정: 기간 2026-09-21 ~ 11-13, 36 거래일, 초기자금 10억, **후원 운용사 ETF만**, HTS 부문별 허용 리스트 사전 설정.

미확정: 자율형 레버리지·인버스 허용 여부.

**INV-04-5 (Unknown ≠ 기본값)**: `leverage_allowed: unknown` 은 양쪽 시나리오를 모두 산출한다. primary scenario 는 `allow` (유동성 실측 근거, architecture §5.1).

### 1.7 LeverageFamily — 배수 변형을 하나의 베팅으로

동일 `leverage_family_key` 의 +1/+2/-1/-2 변형은 **하나의 경제적 베팅**이다.

```
leverage_family_key := normalize_index_key(IDX_IND_NM)   # 정답
leverage_family_key := strip_tokens(ISU_NM)            # 금지 (KODEX 레버리지 → "")
```

**INV-04-11**: `leverage_family_key` 는 `IDX_IND_NM` 에서만 유도한다. `ISU_NM` 토큰 제거로 family 를 만들지 않는다.

`configs/leverage_family_overrides.yaml` (선택): 서로 다른 `index_key` 를 하나의 `family_id` 로 병합 (예: 현물 지수 vs 선물 지수 변형). override 없으면 family = index_key 1:1.

alpha 는 `index_key` / family 를 고르고, overlay(Stage 8)가 배수를 고른다. universe 계층은 family membership 만 제공한다.

### 1.8 속성 유도: 이름 파싱은 휴리스틱이다

레버리지 배수 파싱 순서 (이름에서만):

```
1. 인버스2X | -2X | 곱버스        → -2
2. 인버스   | -1X                 → -1
3. 레버리지 | 2X | 2배            → +2
4. (없음)                          → +1
```

**INV-04-6 (모호성 fail-closed)**: `Confidence.LOW` 인 종목은 레버리지 조건부 정책에서 자동 제외.

### 1.9 Taxonomy: 3단계

| 레벨 | 키 | 유도 | 용도 |
| --- | --- | --- | --- |
| L1 | `index_key` | `normalize(IDX_IND_NM)` | 중복 베팅 제거 |
| L1.5 | `leverage_family_key` | `index_key` 또는 override | **LeverageFamily** |
| L2 | `theme` | `configs/taxonomy.yaml` | 섹터 leadership |

**INV-04-7 (커버리지)**: 유동성 통과 종목 기준 `theme != OTHER` 비율 ≥ 90%.

### 1.10 복잡도

- `InstrumentMaster.build`: $O(N)$
- `LeverageFamilyIndex.build`: $O(N)$
- `PointInTimeUniverse.get(t)`: $O(|\text{panel}(t)|)$ — ADV 는 생성 시 1회 계산

---

## 2. Architecture & Mitigation

```
 normalized/etf_daily.parquet
        │
        ├──► instruments.py ──► InstrumentMaster
        │      issuer · leverage_multiple · leverage_family_key
        │                    │
        ├──► families.py ────┤  LeverageFamilyIndex
        │                    │
        ├──► taxonomy.py ────┤  index_key + theme
        │                    ▼
        ├──► provider.py ──► PointInTimeUniverse.get(date, filters)
        │      mode: structural | deployment
        │                       └► UniverseSnapshot(tickers, drop_counts)
        │                    ▲
        └──► tournament.py ──┘  TournamentRules
              configs/tournament.yaml
              configs/sponsor_brands.yaml
              configs/universe_manifest.yaml
```

### 2.1 Config 연동

| 파일 | 역할 |
| --- | --- |
| `configs/universe.yaml` | `modes.structural` / `modes.deployment` |
| `configs/sponsor_brands.yaml` | 운용사 ↔ 브랜드 매핑 |
| `configs/tournament.yaml` | 대회 기간·후원사 목록·규칙 |
| `configs/universe_manifest.yaml` | HTS 허용 `isu_cd` (확정 전 `null`) |

```yaml
# configs/universe.yaml (발췌)
universe:
  modes:
    structural:
      issuer_whitelist: null
    deployment:
      issuer_whitelist: sponsor_asset_managers
      manifest: null
```

### 2.2 UniverseSnapshot 감사가능성

```python
UniverseSnapshot(
    as_of=date(2026, 10, 7),
    mode=UniverseMode.DEPLOYMENT,
    tickers=("069500", "233740", ...),
    dropped={
        "existence": 12, "price": 3, "history": 108,
        "sponsor": 86, "liquidity": 897, "eligibility": 0,
    },
    filters=UniverseFilters(...),
)
```

---

## 3. Assumptions

- **A-1**: 브랜드 접두어로 운용사 식별. 미매핑 → `issuer=UNKNOWN`, deployment 에서 제외.
- **A-2**: `warmup_sessions` = 최장 lookback(60) + 20 = **80**.
- **A-3**: ADV 는 `is_tradable=False` 날을 결측 처리.
- **A-4**: 후원사 필터 후 유동성 숫자는 전체 패널과 동일 (비후원 86종은 유동성 상위 미진입, 실측).

---

## 4. Execution Target

```bash
uv run pytest tests/unit/universe -q --tb=short
uv run python tools/agent_skills/lean_check.py --pre-impl docs/specs/04_pit_universe_contract.json

uv run mt-etf universe --date 2026-08-27 --mode deployment --max-order-to-adv 0.05
uv run mt-etf universe --date 2026-08-27 --mode structural --max-order-to-adv 0.05
```
