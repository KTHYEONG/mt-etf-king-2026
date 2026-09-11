# Architecture Documentation Index

머니투데이 제3회 ETF 투자왕 대회(2026) 출전을 위해 개발된 **Tournament Quant Research & Execution System**의 기술 아키텍처 문서군입니다.

본 문서군은 채용 담당자, 기술 면접관 및 시스템 엔지니어가 1~2분 내에 시스템의 설계 의도와 핵심 엔지니어링 결정을 파악할 수 있도록 4개의 핵심 심층 문서로 체계화되어 있습니다.

---

## 핵심 아키텍처 문서 체계

| 문서 | 핵심 내용 | 주요 대상 |
| :--- | :--- | :--- |
| **[`overview.md`](overview.md)** | 시스템 목표, 비대칭 토너먼트 효용 함수, 8계층 시스템 토폴로지, 엔드투엔드 런타임 흐름, 외부 의존성 | 전체 시스템 구조 및 대회 경제적 목적 파악 |
| **[`data-flow.md`](data-flow.md)** | KRX Open API Ingestion $\\to$ Bronze $\\to$ Silver $\\to$ PIT Universe $\\to$ Gold Features 전 파이프라인 흐름, 이벤트/처리/체결 시간 축 및 시계열 무결성 규칙 | 데이터 엔지니어링 및 시계열 무결성 검증 |
| **[`components.md`](components.md)** | 10개 핵심 컴포넌트별 책임(Responsibility), 입력/출력 인터페이스, 의존성 및 실제 구현 클래스/모듈 상세 | 소프트웨어 아키텍처 및 모듈 간 결합도 평가 |
| **[`design-decisions.md`](design-decisions.md)** | 토너먼트 목적함수, Next-Open 체결, 듀얼 유니버스, 급락 반등 앵커, Parquet/Polars, GBDT 용량 제약 등 핵심 ADR | 기술 면접관 대상 아키텍처 트레이드오프 심층 검토 |

---

## 도메인 참조 문서

* **[`mt-data-report.md`](../knowledge/mt-data-report.md)**: 대회 공식 규정 온톨로지, 역대 대회 통계 분석 및 제약조건 지식베이스 (Knowledge Base)
