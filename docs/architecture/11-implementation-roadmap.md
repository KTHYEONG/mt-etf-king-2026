# 11. Implementation Roadmap — 구현 로드맵

Spec 순서대로 구현하고, 각 단계마다 contract 검증 후 다음으로 진행합니다.

---

## 1. Spec → 구현 매핑

| 순서 | Spec | 산출물 | Architecture |
| --- | --- | --- | --- |
| 01 | core_spine | settings, calendar, paths, logging, cli | [03](03-core-infrastructure.md) |
| 02 | krx_ingestion | provider, bronze, backfill, quota | [04](04-data-pipeline.md) |
| 03 | silver_panel | schema, silver, validation | [04](04-data-pipeline.md) |
| 04 | pit_universe | instruments, taxonomy, universe | [05](05-universe-and-instruments.md) |
| 05 | feature_engine | pit, feature groups, builder | [06](06-feature-engine.md) |
| 06 | research_harness | backtest, simulator, replay | [08](08-research-harness.md) |
| 07 | leadership_engine | baselines, leadership (blueprint) | [07](07-alpha-and-portfolio.md) |
| 08 | portfolio_tournament | portfolio, exposure, policy, reporting (blueprint) | [07](07-alpha-and-portfolio.md) |
| 09 | ml_ranker (예정) | dataset, splits, ranker, registry | [12](12-ml-layer.md) |

07·08 은 contract 유예 — 06 결과(백테스트 인프라) 확보 후 작성. 09 는 W3 완료 후 착수.

### 1.1 레버리지 편입에 따른 spec 영향

| Spec | 영향 |
| --- | --- |
| `04_pit_universe` | `LeverageFamily` 추가, family key 는 `IDX_IND_NM` 기반 (contract 갱신 필요) |
| `08_portfolio_tournament` | `ExposureSelector`, gross exposure 제약 (blueprint 단계라 반영 용이) |
| `06_research_harness` | robustness grid 참여율 축 1/2/5%, 게이트 G2a/G2b |

`04_pit_universe_contract.json` 은 **갱신 완료** (LeverageFamily, sponsor_brands, deployment 모드). `lean_check --pre-impl` 은 01~03 구현 후 순차 PASS.

---

## 2. 주차별 일정 (W1~W4)

### W1: Data Foundation

| 일 | 작업 | 완료 기준 |
| --- | --- | --- |
| D1-2 | spec 01 implement | `config-check`, `calendar` 동작 |
| D3-4 | spec 02 implement | `ingest` → bronze 1일치 |
| D5-7 | spec 03 implement | `normalize` → silver + validation PASS |

**마일스톤**: 2018~현재 ETF daily silver 패널 완성.

### W2: Universe + Features

| 일 | 작업 | 완료 기준 |
| --- | --- | --- |
| D8-9 | spec 04 implement | `universe --as-of` 출력 |
| D10-12 | spec 05 implement | `features` gold 빌드 |
| D13-14 | B0~B2 baseline wiring | score → weights 파이프 확인 |

**마일스톤**: PIT feature + universe end-to-end.

### W3: Research Harness

| 일 | 작업 | 완료 기준 |
| --- | --- | --- |
| D15-17 | spec 06 implement | `backtest --strategy B0` |
| D18-19 | rolling-36D simulator | `simulate` 분포 출력 |
| D20-21 | 2025 replay + robustness grid | `replay --year 2025` |

**마일스톤**: B0~B5 전부 rolling-36D 분포 산출.

### W4: Alpha + Tournament + Operations

| 일 | 작업 | 완료 기준 |
| --- | --- | --- |
| D22-23 | leadership model | sector rotation vs B1 비교 |
| D24-25 | portfolio + ExposureSelector | concentrated sizing, 배수 선택 |
| D26-27 | `decide` + reporting | 일일 의사결정 + 근거 |
| D28 | integration test | full pipeline 1일 end-to-end |

**마일스톤**: 대회 전 운영 가능 상태.

### W5(선택): ML — Stage 5

**W3 가 끝나지 않으면 착수하지 않습니다.** rule baseline 검증이 선행 조건입니다.

| 일 | 작업 | 완료 기준 |
| --- | --- | --- |
| D29-30 | dataset + PurgedWalkForward | 누출 없는 fold 생성 검증 |
| D31-32 | LightGBM Ranker 학습 | fold별 IC 산출 |
| D33 | G6~G8 게이트 판정 | 채택 또는 **기각** |

ML 이 W5 안에 안 끝나도 대회 참가에는 지장이 없습니다. `AlphaModel` drop-in 이므로 **대회 기간 중 게이트를 통과하는 시점에 교체**할 수 있습니다. 반대로 게이트를 통과하지 못하면 그대로 버립니다.

---

## 3. 구현 명령

```bash
# 각 spec 순서대로
uv run python tools/agent_skills/lean_check.py --pre-impl docs/specs/01_core_spine_contract.json
# → PASS 확인 후
/implement docs/specs/01_core_spine_contract.json
```

---

## 4. 현재 상태 (2026-08-28)

| 항목 | 상태 |
| --- | --- |
| Spec 00~06 | ✅ **구현 완료** |
| Spec 07_preflight | ✅ 구현 완료 (본 contract) |
| Contract 07·08 | ⏳ blueprint-only (preflight 후 `/spec 07` 재실행) |
| `src/` | ✅ 엔진·시뮬레이터·리플레이·지표 구현 |
| KRX 실측 검증 | ✅ probe_*.py + 06 실데이터 baseline |
| Silver 데이터 | 2024~ panel 확보 (06 결과) |

---

## 5. 다음 즉시 작업

```
1. preflight PASS 후 B5 재측정 + giveback 열 추가
2. /spec 07_leadership_engine (파라미터 분위수·ablation)
3. /implement 07 (A-1 P(R>30%)>0.091)
4. /spec 08_portfolio_tournament (giveback 보고 필수)
```
ML은 W3+preflight+07 A-1 이전 착수 금지.

---

## 6. 일일 운영 절차 (대회 기간)

```
09:00  전일 데이터 ingest 확인
09:30  normalize + validation
10:00  features + universe 빌드
10:30  alpha score + portfolio weights
11:00  decide → HTS 수동 입력
15:30  장 마감 후 당일 데이터 ingest (익일 준비)
```

모의투자 HTS 는 수동 체결이므로, 시스템은 **추천 + 근거** 를 제공하고 최종 실행은 사람이 합니다.

---

## 7. 성공 기준

| 시점 | 기준 |
| --- | --- |
| W1 종료 | silver 패널 2018~현재, validation 100% PASS |
| W2 종료 | 6개 feature group + B0 동작 |
| W3 종료 | B0~B5 rolling-36D 분포 비교표 |
| W4 종료 | leadership vs best baseline 우위 확인 |
| 대회 D-1 | `decide` 파이프라인 5일 연속 무장애 |

---

## 8. 의도적 제외 (이번 대회)

- **Deep learning / RL** (shallow GBDT 는 범위 내 — [12-ml-layer.md](12-ml-layer.md))
- 실시간 streaming (일별 batch 충분)
- PDF holdings 파싱 (API 404)
- 다중 브로커 연동 (코스콤 HTS 수동)
- Cloud 배포 (로컬 WSL2 충분)
