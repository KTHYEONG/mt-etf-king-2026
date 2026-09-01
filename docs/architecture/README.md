# Architecture Documentation

머니투데이 제3회 ETF 투자왕 대회(2026) 우승을 목표로 하는 **Tournament Quant Research System** 의 구조 설명 문서입니다.

구현 계약(contract)은 [`docs/specs/`](../specs/)에 있고, 본 디렉터리는 **왜 이렇게 설계했는지·어떻게 연결되는지**를 설명합니다.

---

## 문서 목록

| 문서 | 내용 |
| --- | --- |
| [01-overview.md](01-overview.md) | 목적 함수, 설계 원칙, 대회 맥락 |
| [02-system-layers.md](02-system-layers.md) | 8계층 아키텍처, 모듈 맵, 데이터 흐름 |
| [03-core-infrastructure.md](03-core-infrastructure.md) | settings · calendar · paths · logging · CLI |
| [04-data-pipeline.md](04-data-pipeline.md) | KRX 수집 · bronze/silver · 검증 게이트 |
| [05-universe-and-instruments.md](05-universe-and-instruments.md) | PIT universe · taxonomy · 대회 규칙 |
| [06-feature-engine.md](06-feature-engine.md) | feature 그룹 · PIT guard · regime/breadth |
| [07-alpha-and-portfolio.md](07-alpha-and-portfolio.md) | baseline · leadership · sizing · state machine |
| [08-research-harness.md](08-research-harness.md) | backtest · rolling-36D · replay · 평가 지표 |
| [09-configuration.md](09-configuration.md) | YAML 설정 · 미확정 규칙 처리 |
| [10-invariants-and-gates.md](10-invariants-and-gates.md) | 불변식 · 리스크 레지스터 · 채택/기각 게이트 |
| [11-implementation-roadmap.md](11-implementation-roadmap.md) | spec 의존 순서 · 주차별 일정 · 운영 절차 |
| [12-ml-layer.md](12-ml-layer.md) | LightGBM Ranker · purged walk-forward · 용량 상한 |

---

## Spec ↔ Architecture 매핑

| Spec | Architecture 문서 |
| --- | --- |
| `00_architecture.md` | 01, 02, 10 |
| `01_core_spine` | 03 |
| `02_krx_ingestion` | 04 |
| `03_silver_panel` | 04 |
| `04_pit_universe` | 05, 09 |
| `05_feature_engine` | 06 |
| `06_research_harness` | 08 |
| `07_preflight` (완료) | 01, 07, 08, 10, 11 |
| `07_leadership_engine` | 07 |
| `08_portfolio_tournament` | 07, 11 |
| `09_ml_ranker` (예정) | 12 |
| `p27_identity_overlay_field_report` | 01, 07, 08, 10, 11 |

---

## 한 줄 요약

> **KRX 기반 point-in-time ETF 데이터 위에 `Regime → Sector Leadership → Cross-sectional Ranking → Concentrated Portfolio → Exposure/Tournament Overlay`를 얹고, 모든 판단을 rolling 36-day tournament 방식으로 검증하는 연구 시스템**

---

## 주요 설계 변경 이력

| 변경 | 내용 | 근거 |
| --- | --- | --- |
| ML 재편입 | "범위 외" → shallow GBDT 는 범위 내, DL·RL 만 제외 | 제약은 일정이 아니라 유효 표본 수 ([12](12-ml-layer.md)) |
| 레버리지 primary | Unknown 의 primary scenario 를 deny → **allow** | 배제 시 유동성 유니버스 26 → 15 ([05 §4.3](05-universe-and-instruments.md)) |
| LeverageFamily 도입 | alpha 는 지수를, overlay 는 배수를 선택 | 동일 기초지수 다중 배수 패밀리 8개 실측 |
| G2 게이트 재정의 | CVaR 대칭 게이트 → 파산 제약(G2a) + 보상 정합(G2b) | 대회 보상은 순위 계단 함수 ([08 §8.1](08-research-harness.md)) |
| P27 identity overlay | live 정책에서 house-money late-lock 제거. $F(r)^N$·MC rank policy 는 채택 게이트가 아님 | P26 championship FAIL = overlay < raw ([11](11-implementation-roadmap.md)) |
| 후원사 universe | deployment = 후원 운용사 ETF, structural = 전체 | `sponsor_brands.yaml` + HTS manifest ([05 §6](05-universe-and-instruments.md)) |
