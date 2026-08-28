# 05. Universe & Instruments — L2 계층

"오늘 이 ETF를 실제로 살 수 있는가?" 와 "대회 규칙상 허용되는가?" 를 **그 시점 기준**으로 판정합니다.

---

## 1. 두 가지 Universe

혼동하면 survivorship bias 와 deployment gap 이 생깁니다.

| Universe | 정의 | 용도 |
| --- | --- | --- |
| **Structural** | 각 `t` 에 실제 존재·거래된 ETF (후원사 필터 **없음**) | 구조적 alpha 검증 — "이 신호가 원래 시장에 있었나?" |
| **Deployment** | 후원 운용사 ETF + 유동성·대회 규칙 충족 | **대회 전략 확정·ML·decide** — "실제로 살 수 있는 걸로 되나?" |

백테스트 리포트에서 **항상 어떤 universe 인지 명시**합니다. Structural 결과만으로 전략을 채택하지 않습니다 (`gates.yaml` → `require_universe_mode: deployment`).

---

## 2. InstrumentMaster

ETF 메타데이터의 단일 진실 공급원입니다.

### 2.1 유도 필드 (API 없음 → 패널에서 계산)

| 필드 | 유도 방법 |
| --- | --- |
| `first_seen` | silver 패널에서 `isu_cd` 최초 등장일 |
| `last_seen` | silver 패널에서 `isu_cd` 최종 등장일 |
| `is_active` | `last_seen ≥ t - grace_period` |
| `leverage_factor` | 종목명·ISU_NM 휴리스틱 + manual override |

### 2.2 Leverage 분류

KRX 에 leverage flag API 가 없습니다. 2단계로 처리합니다.

```
1. resolve_leverage(isu_nm) → {1, 2, -1, -2, Unknown}
2. configs/instruments_override.yaml 에 수동 보정
```

`Unknown` 은 fail-closed: 해당 종목은 leverage 시나리오 양쪽 모두에서 제외하거나 별도 표기.

---

## 3. Taxonomy (3단계)

```
index_key (KRX 지수)  →  leverage_family  →  theme (투자 주제)
```

| 단계 | 예시 | 용도 |
| --- | --- | --- |
| `index_key` | `KOSPI200`, `SP500`, `NASDAQ100` | cluster, breadth |
| `leverage_family` | `{KOSPI200: [+1, +2, -1, -2]}` | **exposure 선택** |
| `theme` | `semiconductor`, `dividend`, `gold` | sector leadership |

- `index_key` 는 KRX `IDX_IND_NM` 필드에서 매핑
- `theme` 은 `configs/taxonomy.yaml` 에 정의
- 미매핑 종목 → `theme = "unclassified"` (ranking 에서 별도 처리)

### 3.1 LeverageFamily

동일 기초지수의 배수 변형을 **하나의 베팅**으로 묶습니다.

```
LeverageFamily(index_key="KOSPI200") = {
   +2: KODEX 레버리지            (1.9조)
   +1: KODEX 200                 (2.0조)
   -1: KODEX 인버스              (5,865억)
   -2: KODEX 200선물인버스2X     (4,492억)
}
```

실측(2026-08-27, ADV ≥ 200억 65종목)에서 **다중 배수 패밀리 8개**가 확인됩니다: 코스피200, 코스닥150, 반도체, 반도체TOP10, SK하이닉스단일종목, 2차전지산업, 200IT, 미국달러선물.

**왜 필요한가**: `KODEX 200` 과 `KODEX 레버리지` 를 alpha 가 별개 종목으로 랭킹하면 동일 베팅을 중복 계산하고 cluster dedup 이 무력화됩니다. 둘 다 Top-3 에 들면 실효 노출이 의도의 1.5배가 됩니다.

**책임 분리**:

```
alpha    → index_key 를 고른다   ("코스피200이 강하다")
overlay  → 배수를 고른다          ("regime + 잔여일수 → +2x")
```

### 3.2 Family key 유도 (구현 주의)

**종목명 파싱으로 family key 를 만들면 안 됩니다.** `KODEX 레버리지` / `KODEX 인버스` 는 이름에 기초지수 토큰이 없어 브랜드·배수 토큰을 제거하면 **빈 문자열**이 됩니다.

```
family_key := normalize(IDX_IND_NM)      # 정답
family_key := strip_tokens(ISU_NM)       # 실패 (KODEX 레버리지 → "")
```

배수(`leverage_multiple`)는 이름 파싱 휴리스틱 + override 로 유도하되(§2.2), **family 소속은 지수 필드로** 판정합니다.

---

## 4. PointInTimeUniverse

```python
universe = provider.get_universe(as_of=date(2026, 8, 27))
# → DataFrame[isu_cd, is_eligible, reason, adv_20d, ...]
```

### 4.1 Eligibility 체크 (순서 고정 — INV-5)

```
1. exists_at(t)        — first_seen ≤ t ≤ last_seen
2. has_valid_price(t)  — close not null, > 0
3. min_history(t)      — 상장 후 N일 경과
4. sponsor(t)          — deployment 모드: 후원 운용사 ETF 만 (§6)
5. liquidity(t)        — ADV_20d ≥ threshold
6. tournament_rules(t) — 레버리지·인버스·manifest ...
```

앞 단계 실패 시 뒤 단계 평가하지 않음 (short-circuit). `sponsor` 단계는 **deployment 모드에서만** 적용됩니다. structural 모드는 4번을 skip 합니다.

### 4.2 ADV 계산

```
ADV_20d(t) = mean(trading_value[t-19:t])  — PIT: t 이전 20 session 만
```

캐시는 `(as_of, isu_cd)` 키로 저장. **미래 데이터로 ADV 를 계산하면 INV 위반.**

거래정지·휴장으로 `is_tradable=False` 인 날은 0 이 아니라 **결측**으로 처리하고 유효 관측만 평균합니다. 0 으로 채우면 ADV 가 인위적으로 낮아져 멀쩡한 종목이 유동성 필터에서 탈락합니다.

### 4.3 유동성 실측과 레버리지 (2026-08-27)

10억 전액을 참여율 `p` 이내로 소화하려면 `ADV ≥ capital / p` 여야 합니다.

| 참여율 | 필요 ADV | 거래가능 (전체) | 일반 | 레버리지·인버스 |
| --- | --- | --- | --- | --- |
| 1% | 1,000억 | 26 | 15 | **11** |
| 2% | 500억 | 41 | 28 | 13 |
| 5% | 200억 | 65 | 50 | 15 |
| 10% | 100억 | 121 | 100 | 21 |

**후원사 필터 적용 시(1,077종) 위 숫자는 전부 동일**합니다. 비후원 86종은 ADV 200억 이상이 0개이고 거래대금 상위권에 없습니다. 유동성 관련 결론은 우연히 맞은 것이 아니라 **비후원이 상위 tier 에 진입하지 못하기 때문**입니다. 다만 deployment 백테스트·ML·decide 에는 여전히 필터를 켜야 합니다.

거래대금 상위 20 중 7개가 레버리지·인버스입니다(`KODEX 레버리지` 2위, `KODEX 인버스` 4위, `KODEX 코스닥150레버리지` 9위, `KODEX 200선물인버스2X` 11위).

**해석**: 레버리지 배제는 리스크 축소가 아니라 **유니버스를 26 → 15 로 줄이는 집중 리스크 증가**입니다. 이 관측이 §5.1 primary scenario 전환의 근거입니다.

`participation_rate` 는 단일 값이 아니라 robustness grid 축입니다(1% / 2% / 5%). 유니버스 크기가 참여율에 급격히 반응하므로, 하나의 값으로 결론을 내면 안 됩니다.

---

## 5. TournamentRules

대회 규칙 중 **아직 확정되지 않은 항목**은 `Unknown` sentinel 으로 표현합니다.

| 규칙 | 상태 | primary | 처리 |
| --- | --- | --- | --- |
| 레버리지 ETF 허용 | Unknown | **allow** | 양쪽 산출, allow 가 기본 |
| 인버스 ETF 허용 | Unknown | **allow** | 동일 |
| 파생상품 ETF | Unknown | allow | 동일 |
| 부문별 제한 (자율형) | 확정: 제한 없음 | — | — |
| 최소 거래 단위 | 확정: 1주 | — | — |

```python
class TournamentRules:
  allow_leverage: bool | Unknown
  allow_inverse: bool | Unknown
  ...
```

### 5.1 Primary scenario = allow

`Unknown` 의 **의미는 바뀌지 않습니다** — 여전히 양쪽 시나리오를 산출하고 단일 값으로 가정하지 않습니다(INV-04-5). 바뀐 것은 **어느 쪽을 primary 로 리포트하고 파라미터를 튜닝하느냐**입니다.

근거는 §4.2 유동성 실측입니다. deny 시나리오의 거래가능 종목은 15개뿐이고, 그중 상당수가 채권·금리형입니다. deny 를 primary 로 두면 **존재하지 않을 가능성이 높은 제약에 맞춰 전략을 과최적화**하게 됩니다.

| 시나리오 | 역할 | 산출 |
| --- | --- | --- |
| `aggressive` (allow) | **primary** — 파라미터 튜닝 기준 | 전 게이트 통과 필수 |
| `conservative` (deny) | fallback — 규칙 확정 시 즉시 전환 | 게이트 통과 필수 |

두 시나리오 모두 accept 게이트를 통과해야 합니다. deny 로 확정되더라도 **대회 첫날에 쓸 전략이 준비되어 있어야** 하기 때문입니다.

### 5.2 규칙 확정 시 절차

대회 직전 부문별 투자 가능 ETF 리스트가 HTS 에 설정됩니다(보도자료 확정 사실). 그 시점에:

1. `tournament.yaml` 의 `Unknown` → 확정 boolean
2. `configs/universe_manifest.yaml` 에 HTS `isu_cd` 리스트 반영
3. 해당 시나리오 백테스트 재산출 후 게이트 재확인

---

## 6. 후원사 ETF 제약

대회 규칙: **후원 운용사 ETF만 매매 가능**. 설정은 `configs/sponsor_brands.yaml`, `configs/tournament.yaml`.

### 6.1 후원사 구분

| 구분 | 후원 주체 | ETF 발행 |
| --- | --- | --- |
| 인프라 | 금융투자협회, 한국거래소, 코스콤 | **없음** (주최·인프라) |
| 운용사 | 삼성·미래에셋·KB·한국투자·신한·한화·타임폴리오·NH아문디·키움·하나 | **있음** — 브랜드로 식별 |

### 6.2 브랜드 → 운용사 매핑

KRX `etf_bydd_trd` 에 운용사 필드가 없으므로 `ISU_NM` 첫 토큰(브랜드)으로 유도합니다.

| 운용사 | 브랜드 |
| --- | --- |
| 삼성자산운용 | KODEX |
| 미래에셋자산운용 | TIGER, KoAct |
| KB자산운용 | RISE |
| 한국투자신탁운용 | ACE |
| 신한자산운용 | SOL |
| 한화자산운용 | PLUS |
| 타임폴리오자산운용 | TIME |
| NH아문디자산운용 | HANARO |
| 키움투자자산운용 | KIWOOM |
| 하나자산운용 | 1Q |

`resolve_issuer(name, brand_map)` → 운용사명. 미매핑 브랜드는 `UNKNOWN` 이며 deployment 에서 **fail-closed 제외** (INV-20).

### 6.3 3단계 진실 공급원

```
1. sponsor_brands.yaml     — 지금 사용 (브랜드 휴리스틱)
2. universe_manifest.yaml  — HTS 부문별 허용 isu_cd (대회 직전 확정)
3. HTS 실주문             — 최종 (시스템은 2번까지 반영)
```

manifest 가 채워지면 브랜드 매핑보다 **manifest 가 우선**합니다 (더 좁은 집합).

### 6.4 어디에 적용하나

| 단계 | 후원사 필터 |
| --- | --- |
| Structural 백테스트 | **OFF** — 아이디어 구조 검증 |
| Deployment 백테스트 | **ON** — 전략 확정 |
| ML 학습·CV·채택 | **ON** (`ml.yaml` → `universe_mode: deployment`) |
| `decide` / 실매매 | **ON** + manifest (확정 시) |

연구 탐색은 넓게 볼 수 있지만, **"이번 대회에 쓸 전략"이라고 말할 때는 deployment 기준**입니다.

### 6.5 실측 (2026-08-27)

| | 전체 | 후원사만 | 비후원 |
| --- | --- | --- | --- |
| 종목 수 | 1,163 | 1,077 (92.6%) | 86 |
| ADV≥1,000억 (1% 참여) | 26 | **26** | 0 |
| 거래대금 Top 20 | 20 | **20** | 0 |

---

## 7. 대회 부문 매핑

| 부문 | 필터 (개략) |
| --- | --- |
| 국내주식형 | 국내 주식 지수 추종 |
| 연금형 | 연금 투자 적격 ETF |
| 글로벌형 | 해외 지수 추종 |
| 자율형 | **필터 없음** (우리 전략의 주 타겟) |

자율형은 제약이 가장 적으므로 **primary deployment universe** 입니다. 부문별 우수상(100만)은 secondary objective.

---

## 8. PIT Universe 스냅샷

일별 universe 를 materialize 해 감사·재현에 사용합니다.

```
data/gold/universe/
  └── universe_20260827.parquet
```

컬럼: `isu_cd`, `is_eligible`, `reject_reason`, `issuer`, `adv_20d`, `theme`, `leverage_factor`, `index_key`, `universe_mode`

---

## 9. CLI

| Command | 동작 |
| --- | --- |
| `mt-etf universe --as-of 2026-08-27` | deployment PIT universe (기본) |
| `mt-etf universe --mode structural` | 후원사 필터 없음 |
| `mt-etf universe --scenario conservative` | Unknown 규칙 보수 적용 |
