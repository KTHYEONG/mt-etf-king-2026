# 10. Invariants & Gates — 불변식과 검증 게이트

시스템 전체에 걸친 **절대 불변식(invariants)** 과 **채택/기각 게이트**입니다. 위반 시 fail-closed.

---

## 1. Global Invariants (INV-1 ~ INV-12)

### 데이터 (L1)

| ID | 불변식 | 위반 시 |
| --- | --- | --- |
| INV-1 | KRX `""` → `None`. 0 변환 금지 | 수익률·모멘텀 오염 |
| INV-2 | 휴장일 = `valid_price_ratio < threshold`. 행 수로 판정 금지 | phantom trading day |
| INV-3 | Bronze write-once. 재수집은 revision suffix | 원본 훼손 |

### Universe (L2)

| ID | 불변식 | 위반 시 |
| --- | --- | --- |
| INV-4 | `first_seen`/`last_seen` 은 패널 유도. 상장일 API 가정 금지 | survivorship bias |
| INV-5 | Eligibility 체크 순서 고정 (exists → price → history → **sponsor** → liquidity → rules) | inconsistent filter |
| INV-6 | ADV 는 `as_of` 이전 window 만 사용 | look-ahead |
| **INV-20** | **deployment 모드에서 `issuer=UNKNOWN` 또는 비후원 ETF 는 제외 (fail-closed)** | 매매 불가 종목 포함 |
| **INV-21** | **전략 채택·ML·decide 는 deployment universe 만 사용. structural 단독 채택 금지** | 백테스트-실전 괴리 |

### Features (L3)

| ID | 불변식 | 위반 시 |
| --- | --- | --- |
| INV-7 | `assert_pit(df, as_of)` 모든 feature 함수 진입 시 | future leak |
| INV-8 | Session grid = XKRX only. `bdate_range` 금지 | wrong calendar |
| INV-9 | Cross-sectional rank/zscore 는 **그날 단면** 만 | full-sample stats leak |

### Execution (L5~L6)

| ID | 불변식 | 위반 시 |
| --- | --- | --- |
| INV-10 | Signal at `close(t)` → fill at `open(t+1)` | same-bar cheat |
| INV-11 | `order_value ≤ ADV × participation_rate` | unrealistic fill |
| INV-12 | Cost 는 fill 시점에 적용. 사후 조정 금지 | understated drag |

### ML (L4)

| ID | 불변식 | 위반 시 |
| --- | --- | --- |
| INV-13 | ML 검증은 purge ≥ label horizon, embargo ≥ label horizon 필수. 무작위 KFold·`train_test_split` 금지 | 라벨 중첩 누출 |
| INV-15 | 하이퍼파라미터·early stopping 은 fold **내부**에서 선택 | selection 누출 |
| INV-16 | ML 은 rule model 과 **동일 feature 집합** 사용 | 불공정 비교 |

### Leverage (L2·L5)

| ID | 불변식 | 위반 시 |
| --- | --- | --- |
| INV-14 | 레버리지 ETF 수익률은 **실제 ETF 가격**에서만. 지수 수익률 × 배수 합성 금지 | 변동성 감쇠 누락 |
| INV-17 | 동일 `leverage_family` 에서 동시 보유 종목은 최대 1개 | 실효 노출 중복 |
| INV-18 | 제약은 비중이 아니라 **실효 노출**(`Σ\|w × mult\|`)에 적용 | 위험 과소평가 |
| INV-19 | `leverage_multiple` Confidence = LOW 인 종목은 배수 선택에서 `+1` 강제 | 부호 오판 (`인버스2X` → `+2`) |

---

## 2. 아키텍처 불변식

| ID | 규칙 |
| --- | --- |
| ARCH-1 | Signal ≠ Portfolio ≠ Tournament Policy |
| ARCH-2 | Deployment / Structural 백테스트 혼합 금지 |
| ARCH-3 | ML 은 `AlphaModel` 구현체. 별도 계층·특별 게이트 금지. deep learning·RL 은 범위 외 |
| ARCH-4 | 파라미터는 config only (코드 magic number 금지) |
| ARCH-5 | Unknown 규칙 → scenario 양쪽 산출, 단일 값 가정 금지 |
| ARCH-6 | alpha 는 `index_key` 를 고르고, overlay 가 배수를 고른다 |

> **ARCH-3 변경 이력**: 초기 설계는 "ML 범위 외"였습니다. 근거로 든 24일 일정은 잘못된 기준이었고, 실제 제약은 **유효 표본 수**입니다. shallow GBDT 는 이 제약 안에 들어오므로 범위에 포함하되, 용량 상한과 누출 방지(INV-13/15/16)를 강제합니다. 상세는 [12-ml-layer.md](12-ml-layer.md).

---

## 3. Validation Gates (Data)

`PanelValidator` — silver 쓰기 전 실행:

| Gate | Severity | 조건 |
| --- | --- | --- |
| `valid_price_ratio` | CRITICAL | ≥ 90% on trading day |
| `nav_positive` | CRITICAL | ≥ 95% |
| `duplicate_keys` | CRITICAL | 0 duplicates |
| `no_future_dates` | CRITICAL | max(date) ≤ today |
| `row_count_stability` | WARNING | ±20% vs prior day |

CRITICAL → **Parquet 쓰기 중단** + exit code ≠ 0.

---

## 4. Strategy Accept/Reject Gates

| Gate | 조건 | FAIL 시 |
| --- | --- | --- |
| G1 | $P(R>30\%)$ ≥ B0 + 2%p | 기각 |
| G2a | $P(R_{36d} < -25\%) \le 5\%$ (절대 기준) | 기각 |
| G2b | $\mathbb{E}[\text{Prize}]$ ≥ B0 수준 | 기각 |
| G3 | Robustness grid 전 조합 PASS | 기각 |
| G4 | 2025 replay > median anchor | 경고 (단독 기각 아님) |
| G5 | Structural + Deploy median R > 0 | 기각 |
| G6~G8 | ML 전용 (purged fold IC·안정성·baseline 초과) | 기각 |

> **G2 변경 이력**: 초기 `CVaR(5%) 악화 ≤ 3%p vs B0` 는 레버리지 허용과 충돌해 성능과 무관하게 `+2x` 전략을 자동 기각시켰습니다. 대회 보상이 순위의 계단 함수(3위와 400위 보상 동일)이므로 대칭 위험 척도는 목적함수와 정합하지 않습니다. 회복 불가 구간만 절대 기준(G2a)으로 막고 보상 정합은 G2b 로 봅니다. 근거는 [08 §8.1](08-research-harness.md).

CVaR(5%)·MDD 분포는 **진단 지표로 항상 리포트**하되 기각 사유는 아닙니다.

---

## 5. Risk Register

| Risk | 확률 | 영향 | 완화 |
| --- | --- | --- | --- |
| R1: KRX API 장애 | 중 | ingest 중단 | bronze cache, retry |
| R2: Quota 소진 | 중 | 당일 수집 불가 | QuotaLedger, 우선순위 |
| R3: 대회 규칙 변경 | 중 | universe 변동 | Unknown + scenario |
| **R4: 레버리지가 deny 로 확정** | 중 | 유니버스 26 → 15, primary 전략 무효 | conservative 시나리오를 게이트 통과 상태로 상시 유지 |
| R5: 유동성 병목 (1% 참여 시 26종목) | 높 | alpha 다양성 제한 | 레버리지 포함으로 완화, 참여율 grid |
| R6: Overfitting (36D) | 높 | 대회 실패 | 분포 평가, bootstrap, purged CV |
| R7: 2025→2026 레짐 변화 | 중 | replay 무의미 | structural backtest 병행 |
| R8: 휴장일 TRAP-1 | 확정 | 데이터 오염 | valid_price_ratio gate |
| R9: 구현 일정 | 중 | 미완성 | spec 순차, ML 은 W3 완료 후 착수 |
| R10: 경쟁자 unknown | 확정 | 순위 예측 불가 | $P(R>θ)$ 분포 최적화 |
| **R11: 레버리지 변동성 감쇠** | 높 | 2x 수익률 과대추정 | 실제 ETF 가격만 사용 (INV-14) |
| **R12: 배수 파싱 오류** | 중 | `인버스2X` → `+2` 오판 | 순서 의존 규칙 + Confidence, LOW 는 `+1` 강제 (INV-19) |
| **R13: ML 라벨 누출** | 높 | 검증 결과 전면 무효 | purge + embargo (INV-13), nested 선택 (INV-15) |
| **R15: 브랜드→운용사 매핑 오류** | 중 | 비후원 종목 통과 또는 후원 종목 제외 | manifest 확정 시 override, UNKNOWN fail-closed |

---

## 6. Pre-Implementation Gate

각 spec contract 는 구현 전 `lean_check --pre-impl` PASS 필수:

```
docs/specs/01_core_spine_contract.json  → PASS
docs/specs/02_krx_ingestion_contract.json → PASS
...
```

Contract 항목:

- `modules`: 생성할 파일 경로
- `cli_subcommands`: 등록할 CLI
- `invariants`: 해당 spec 이 보장해야 할 INV
- `tests`: 최소 테스트 케이스

---

## 7. Daily Operations Gate

매 거래일 `decide` 실행 전:

```
1. ingest (전일 데이터)     — bronze 존재 확인
2. normalize               — validation PASS
3. features (as_of)        — PIT guard PASS
4. universe (as_of)        — eligible > 0
5. decide                  — 근거 포함 출력
```

하나라도 FAIL → **전일 포지션 유지** (fail-closed, 무리한 매매 금지).

---

## 8. 감사 추적

모든 의사결정 artifact 보존:

```
data/state/decisions/
  └── 20260827_decision.json
      {
        "as_of": "2026-08-27",
        "regime": "RISK_ON",
        "selected": [{"isu_cd": "...", "weight": 0.7, "reason": "..."}],
        "scenario": "conservative",
        "gates_passed": ["G1", "G2", "G3"]
      }
```

재현 가능성 = 대회 기간 내내 **왜 이 포지션인지 설명 가능**해야 함.
