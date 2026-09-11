---
doc_id: KB-MT-ETF-2026
version: 2.0.0
as_of_date: "2026-09-05"
target_competition: "제3회 머니투데이 ETF 투자왕 (2026-09-21 ~ 2026-11-13, 8주)"
purpose: "대회 대응 퀀트 트레이딩 시스템 및 AI 분석 에이전트의 지식 탐색, 백테스트 가설 생성, 가드레일 준수를 위한 표준 온톨로지 지식베이스"
ontology_version: "1.0"
---

# 2026 머니투데이 ETF 투자왕 — AI 지식베이스 (Knowledge Base)

> **문서 사용 원칙**  
> 본 지식베이스는 과거 우승자 전략의 단순 복제용이 아니며, **검증해야 할 가설을 생성하고 대회 구조에 최적화된 데이터·모델·리스크 관리 요구사항을 정의하기 위한 정형화된 지식 엔진 문서**다.  
> 원문의 모든 사실(FACT), 관찰(OBSERVATION), 추론(INFERENCE), 가설(HYPOTHESIS), 출처 충돌(CONFLICT) 및 미확인(UNKNOWN) 상태를 무결하게 보존한다.

---

## 목차 (Table of Contents)

- [PART 1. 시스템 개요 및 에이전트 가이드](#part-1-시스템-개요-및-에이전트-가이드)
  - [1.1 핵심 결론 (Executive Summary)](#11-핵심-결론-executive-summary)
  - [1.2 지식 계층 및 우선순위 원칙](#12-지식-계층-및-우선순위-원칙)
  - [1.3 AI System Prompt 가드레일 (10대 원칙)](#13-ai-system-prompt-가드레일-10대-원칙)
- [PART 2. 온톨로지 및 데이터 스키마](#part-2-온톨로지-및-데이터-스키마)
  - [2.1 지식 분류 체계 (Claim Type)](#21-지식-분류-체계-claim-type)
  - [2.2 출처 신뢰도 등급 (Source Confidence)](#22-출처-신뢰도-등급-source-confidence)
  - [2.3 필수 메타데이터 스키마](#23-필수-메타데이터-스키마)
  - [2.4 AI/RAG 권장 교환 데이터 포맷](#24-airag-권장-교환-데이터-포맷)
- [PART 3. 2026 제3회 대회 공식 규정 및 필수 요건](#part-3-2026-제3회-대회-공식-규정-및-필수-요건)
  - [3.1 공식 대회 규약 (Official Specification)](#31-공식-대회-규약-official-specification)
  - [3.2 핵심 구조적 불변 법칙 (Structural Invariants)](#32-핵심-구조적-불변-법칙-structural-invariants)
  - [3.3 대회 개시 전 필수 확보 데이터 (Data Readiness: P0/P1/P2)](#33-대회-개시-전-필수-확보-데이터-data-readiness-p0p1p2)
- [PART 4. 과거 대회 검증 데이터 및 케이스 스터디](#part-4-과거-대회-검증-데이터-및-케이스-스터디)
  - [4.1 기존 문서 팩트체크 및 데이터 정정 매트릭스](#41-기존-문서-팩트체크-및-데이터-정정-매트릭스)
  - [4.2 제1회(2024) 대회 검증 지식](#42-제1회2024-대회-검증-지식)
  - [4.3 제2회(2025) 대회 심층 타임라인 및 행동 분석](#43-제2회2025-대회-심층-타임라인-및-행동-분석)
  - [4.4 과거 데이터의 한계 및 활용 역할 규정](#44-과거-데이터의-한계-및-활용-역할-규정)
- [PART 5. 퀀트 시스템 아키텍처 및 전략 가설](#part-5-퀀트-시스템-아키텍처-및-전략-가설)
  - [5.1 퀀트 시스템 계층 아키텍처](#51-퀀트-시스템-계층-아키텍처)
  - [5.2 전략 가설 우선순위 매트릭스](#52-전략-가설-우선순위-매트릭스)
  - [5.3 피처 엔지니어링 명세 (Feature Engineering)](#53-피처-엔지니어링-명세-feature-engineering)
  - [5.4 상태 전이 모델 및 리스크 컨트롤러](#54-상태-전이-모델-및-리스크-컨트롤러)
  - [5.5 목적 함수 및 평가 메트릭 체계](#55-목적-함수-및-평가-메트릭-체계)
- [PART 6. 백테스트 및 검증 데이터셋 프로토콜](#part-6-백테스트-및-검증-데이터셋-프로토콜)
  - [6.1 Point-in-Time Universe 관리 규약](#61-point-in-time-universe-관리-규약)
  - [6.2 레버리지 ETF 모델링 및 시뮬레이션 지침](#62-레버리지-etf-모델링-및-시뮬레이션-지침)
  - [6.3 역사적 검증 데이터셋(Validation Dataset) 설계](#63-역사적-검증-데이터셋validation-dataset-설계)
  - [6.4 백테스트 엔진 최소 요구사항](#64-백테스트-엔진-최소-요구사항)
- [PART 7. 소스 레지스트리 및 운영 컴플라이언스](#part-7-소스-레지스트리-및-운영-컴플라이언스)
  - [7.1 저작권 및 운영 컴플라이언스 주의사항](#71-저작권-및-운영-컴플라이언스-주의사항)
  - [7.2 출처 URL 레지스트리 (Source Registry)](#72-출처-url-레지스트리-source-registry)

---

# PART 1. 시스템 개요 및 에이전트 가이드

## 1.1 핵심 결론 (Executive Summary)

1. **2026 제3회 대회는 제2회와 구조적으로 매우 유사하다.**
   - 기간: 8주
   - 초기 모의투자금: 10억원
   - 부문: 국내주식형 / 연금형 / 글로벌형 / 자율형
   - 대상: 전체 수익률 1위 / 최우수상: 전체 수익률 2위
   - 자율형 외 레버리지·인버스 제외
   - 후원 운용사 ETF로 거래 대상 제한
2. **제2회 상위권에서 반복적으로 관찰된 패턴은 주도 섹터/테마 + 고베타·레버리지 표현이다.**
   - 그러나 이것은 인과관계가 아니라 **상위권 사례에서 반복 관찰된 공통점**이다.
   - 레버리지 자체가 알파를 만든다는 증거는 없다.
   - 제2회 전체 2위는 레버리지 사용이 제한된 국내주식형 참가자였다.
3. **제2회에서 가장 강한 리스크 신호는 대회 후반 수익률 반납이다.**
   - 6주차 상위권은 +50~65% 수준이었으나 7주차에는 +36~47%대로 하락했다.
   - 5주차 1위의 +72.28%는 7주차 +41.64%까지 하락했다.
   - 따라서 대회 목적함수는 `기간 중 최대수익률`이 아니라 `종료일 수익률`이다.
4. **우승자 인터뷰에서 확인되는 전략 원칙과 데이터에서 검증된 원칙을 구분해야 한다.**
   - 우승자가 “사이클·주도 섹터·재진입”을 중요하게 봤다는 것은 사실(A3)이다.
   - 이것이 통계적으로 우월하다는 것은 별도의 백테스트가 필요하다.
5. **2024 제1회 데이터는 2026 전략 파라미터 최적화용으로 쓰기 어렵다.**
   - 기간이 훨씬 길고 시상 구조도 다르다.
   - 제1회는 `Hypothesis Generator`, 제2회는 `Tournament Case Study`로 취급하는 것이 적절하다.
6. **원문 문서의 가장 중요한 수정 사항**
   - 2024 대상의 “상대수익률” **정확한 계산식은 공개 기사만으로 확인되지 않는다.**
   - 2025 2주차 글로벌형 최고수익률은 기사 내부에 **8.64%와 8.56%가 동시에 존재하는 원문 모순(`CONFLICT`)**이 있다.
   - “레버리지 = 알파가 아니다”는 방향은 맞지만, 더 정확히는 **레버리지는 우수 성과의 필요조건이 아니며 인과적 기여도는 공개 자료만으로 추정할 수 없다.**
   - “재진입/레짐/집중/크라우딩이 효과적”이라는 문장은 사실이 아니라 **검증 대상 가설(`HYPOTHESIS`)**로 내려야 한다.

## 1.2 지식 계층 및 우선순위 원칙

AI 추론 및 신호 생성 시 적용되는 정보 우선순위는 다음과 같다.

```text
[우선순위 1] 공식 2026 규칙 (FACT)
    >
[우선순위 2] 2025 직접 관측 데이터 (FACT/OBSERVATION)
    >
[우선순위 3] 2024 직접 관측 데이터 (FACT/OBSERVATION)
    >
[우선순위 4] 수상자 자기보고 전략 (A3 FACT - 주관적 진술)
    >
[우선순위 5] 분석적 관찰 (OBSERVATION / INFERENCE)
    >
[우선순위 6] 전략 가설 (HYPOTHESIS)
```

> **지식베이스 대원칙:**  
> 과거 대회 데이터는 **"무엇을 사야 하는가"**를 알려주는 데이터가 아니라, **"무엇을 검증해야 하는가"**를 알려주는 데이터다.  
> 2026 실제 운용 시에는 `현재 시장 데이터 + 실제 허용 ETF universe + 실시간 leadership + regime + execution constraints + terminal tournament state`가 과거 우승 사례보다 절대적으로 우선한다.

## 1.3 AI System Prompt 가드레일 (10대 원칙)

AI 에이전트 시스템 프롬프트에 직접 주입할 가드레일 규칙 목록:

```text
1. FACT와 HYPOTHESIS를 절대 동일한 신뢰도로 취급하지 않는다.
2. 기사에 공개되지 않은 포트폴리오 비중·진입시각을 추정하지 않는다.
3. 과거 우승 ETF 이름을 현재 추천의 직접 근거로 사용하지 않는다.
4. 한 명의 우승자 인터뷰를 통계적 인과관계로 일반화하지 않는다.
5. 레버리지 2배를 다기간 기초지수 수익률 ×2로 계산하지 않는다.
6. 대회 목표는 종료일 평가수익률임을 항상 고려한다.
7. 허용 ETF universe 밖 상품을 추천하지 않는다.
8. 실시간 데이터가 없으면 현재 regime/leader를 단정하지 않는다.
9. 소스 간 숫자가 충돌하면 CONFLICT로 표시한다.
10. 데이터가 없는 필드는 UNKNOWN으로 유지한다.
```

---

# PART 2. 온톨로지 및 데이터 스키마

## 2.1 지식 분류 체계 (Claim Type)

AI가 기사 사실과 분석자의 해석을 혼동하지 않도록 모든 지식 항목에 아래 타입을 부여한다.

| 타입 | 의미 | 모델 사용 |
|---|---|---|
| `FACT` | 기사/공식 페이지에 숫자·규칙·행동·발언이 직접 명시 | 직접 참조 가능 |
| `OBSERVATION` | 여러 FACT를 요약한 기술적 관찰 | 설명 변수/가설 생성에 사용 |
| `INFERENCE` | 공개 사실을 기반으로 한 해석 | 반드시 “추론”으로 표시 |
| `HYPOTHESIS` | 향후 데이터로 검증해야 하는 전략 아이디어 | 백테스트 전 매매 규칙으로 승격 금지 |
| `UNKNOWN` | 공개 자료로 확인 불가 | 추측 금지 |
| `CONFLICT` | 동일 출처 또는 복수 출처가 서로 충돌 | 보수적으로 처리 |

## 2.2 출처 신뢰도 등급 (Source Confidence)

| 등급 | 기준 | 설명 |
|---|---|---|
| `A1` | 대회 공식 안내 페이지/공식 규정 | 최고 신뢰도 공식 문서 |
| `A2` | 머니투데이 최종 결과·주간 순위 기사에 직접 기재 | 공식 언론 보도 수치 |
| `A3` | 수상자 인터뷰의 본인 발언 | 자기보고(Self-reported) 진술 |
| `B1` | 기사에 보유/매매 종목은 나오나 비중·전체 거래내역이 없음 | 부분 공개 관측치 |
| `C1` | 기사 사실을 묶어 만든 분석적 분류 | 분석자 가공 데이터 |
| `U` | 확인 불가 | 출처 부재 |

## 2.3 필수 메타데이터 스키마

각 지식 항목은 최소 다음 필드를 가져야 한다.

```yaml
claim_id: string          # 고유 식별자 (예: 2025_W05_TOP5_2BATTERY)
claim_type: enum          # [FACT, OBSERVATION, INFERENCE, HYPOTHESIS, UNKNOWN, CONFLICT]
competition_year: integer # [2024, 2025, 2026]
as_of_date: string        # YYYY-MM-DD
participant: string       # 참가자명 또는 닉네임 (nullable)
division: string          # 부문 (국내주식형, 연금형, 글로벌형, 자율형)
metric: string            # 측정 지표명
value: any                # 값
unit: string              # 단위 (%, KRW, count 등)
instrument: string        # 관련 ETF 종목명 또는 티커 (nullable)
action: string            # 행동 유형 (매수, 매도, 보유 등)
source_id: string         # 소스 레지스트리 키
source_confidence: enum   # [A1, A2, A3, B1, C1, U]
source_conflict: boolean  # 충돌 여부
caveat: string            # 해석 시 주의점
```

## 2.4 AI/RAG 권장 교환 데이터 포맷

### 2.4.1 Event Record (주간 관측/행동 이벤트)

```json
{
  "id": "2025_W05_TOP5_2BATTERY",
  "claim_type": "FACT",
  "competition_year": 2025,
  "as_of_date": "2025-10-24",
  "entity_type": "leaderboard_group",
  "participants": ["노환준", "정훈", "간절함", "남준", "범고래"],
  "event": "all_top5_held_same_etf",
  "instrument": "KODEX 2차전지산업레버리지",
  "evidence_level": "A2",
  "source_id": "MT_2025_W05",
  "limitations": [
    "exact weights unknown",
    "entry times differ",
    "full transaction histories unknown"
  ]
}
```

### 2.4.2 Strategy Statement (참가자 자기보고 원칙)

```json
{
  "id": "2025_WINNER_REENTRY",
  "claim_type": "FACT",
  "evidence_subtype": "SELF_REPORTED_METHOD",
  "participant": "박남준",
  "statement_summary": "손절 기준보다 재진입 기준을 더 중요하게 설정",
  "causal_status": "NOT_ESTABLISHED",
  "research_hypothesis": "explicit re-entry state may improve terminal return",
  "source_id": "MT_2025_WINNER_INTERVIEW"
}
```

### 2.4.3 Conflict Record (원문 데이터 불일치)

```json
{
  "id": "2025_W02_GLOBAL_MAX",
  "claim_type": "CONFLICT",
  "source_id": "MT_2025_W02",
  "values": [
    {"location": "ranking paragraph", "value_pct": 8.64},
    {"location": "division summary paragraph", "value_pct": 8.56}
  ],
  "preferred_value_pct": 8.64,
  "reason": "named global leader in ranking paragraph is explicitly 8.64",
  "machine_action": "do_not_treat_as_clean_ground_truth"
}
```

---

# PART 3. 2026 제3회 대회 공식 규정 및 필수 요건

## 3.1 공식 대회 규약 (Official Specification)

```yaml
competition:
  name: "제3회 ETF 투자왕"
  year: 2026
  organizer: "머니투데이"
  simulation_provider: "코스콤"
  start_date: "2026-09-21"
  end_date: "2026-11-13"
  duration_label: "8주"
  initial_capital_krw: 1000000000
  divisions:
    - 국내주식형
    - 연금형
    - 글로벌형
    - 자율형
  eligible_universe:
    restriction: "대회 후원 자산운용사 ETF"
  leverage_inverse_rule:
    autonomous: "허용"
    other_divisions: "제외"
  awards:
    grand_prize:
      rule: "전체 수익률 1위"
      prize_krw: 10000000
    second_prize:
      rule: "전체 수익률 2위"
      prize_krw: 5000000
    division_prize:
      rule: "대상·최우수상 제외 부문별 수익률 1위"
      prize_krw: 1000000
```

## 3.2 핵심 구조적 불변 법칙 (Structural Invariants)

### FACT-2026-001 — Terminal Ranking (종료 시점 1위)
대상은 **종료 시점 전체 수익률 1위**다.  
따라서 최적화 대상은 원칙적으로:

```text
maximize R(T)
```

이지:

```text
maximize max_t R(t)
```

가 아니다.

### FACT-2026-002 — Vehicle Asymmetry (상품 비대칭성)
레버리지·인버스는 자율형에서만 사용할 수 있다.  
따라서 모델 구조에서 다음을 분리한다.

```text
Alpha / Direction / Theme Selection
        ↓
Vehicle Selection
        ↓
1x / 2x / inverse / defensive
```

### FACT-2026-003 — Restricted Universe (비시장 전체 유니버스)
거래 대상은 **후원 운용사 ETF로 제한**된다.  
따라서 백테스트도 반드시:

```text
전체 ETF universe
```

가 아니라:

```text
2026 대회 실제 허용 ETF universe
```

로 다시 실행해야 한다.

## 3.3 대회 개시 전 필수 확보 데이터 (Data Readiness: P0/P1/P2)

### P0 — 없으면 전략 검증이 왜곡될 수 있음
- 2026 실제 허용 ETF 전체 리스트
- 각 ETF 티커
- 운용사
- 자율형 거래 가능 여부
- 레버리지/인버스 구분
- HTS 체결 규칙
- 수수료 여부
- 지정가/시장가 지원
- 거래정지/상한가/하한가 처리
- 순위 산식
- 평가손익률 산식
- 현금 포함 방식
- 당일 체결 후 순위 반영 시점

### P1 — 매우 중요
- 일별 leaderboard snapshot
- TOP participant return
- TOP5 return
- 매수집중 종목
- 보유계좌 상위 ETF
- 허용 universe 변경 여부

### P2 — 모델 강화
- ETF NAV/괴리율
- 구성종목
- 해외시장 전일 데이터
- macro/catalyst
- 뉴스량

---

# PART 4. 과거 대회 검증 데이터 및 케이스 스터디

## 4.1 기존 문서 팩트체크 및 데이터 정정 매트릭스

| 기존 주장 | 판정 | 수정 |
|---|---|---|
| 제1회 2024-08-26~12-31 | 맞음 | FACT |
| 제1회 초기자금 10억원 | 맞음 | FACT |
| 제1회 약 1,500명 | 맞음 | FACT |
| 제1회 5개 부문 | 맞음 | FACT |
| 제1회 대상 54.05% | 맞음 | FACT |
| 제1회 대상이 “부문 평균 대비 상대성과”로 선정 | **불완전** | “상대적 성과로 선정”은 확인되나 정확 계산식 UNKNOWN |
| 제1회는 약 4개월 | 실질적으로 맞음 | 공식 기사는 3개월여/4개월 표현 혼재. 날짜를 우선 사용 |
| 9월 김병수 38.68%, 월 35% | 맞음 | FACT |
| 스튜아리 11월 22.18% | 맞음 | FACT |
| 구기 방산 16.91%, 조선 28.43% | 맞음 | FACT |
| Ytm Contrarian | 맞음 | 본인 인터뷰 FACT |
| 제2회 2025-09-22~11-14, 8주 | 맞음 | FACT |
| 제2회 초기자금 10억원, 약 1,000명 | 맞음 | FACT |
| 제2회 대상 전체 수익률 1위 | 맞음 | FACT |
| 2주차 글로벌형 최고 8.56% | **원문 충돌** | 순위 문단 8.64%, 요약 문단 8.56%. `CONFLICT` |
| 쉬었음청년 회전율 409.18% | 맞음 | FACT |
| 4주차 쉬었음청년 19.41% 12위 | 맞음 | FACT |
| 고려대통계학과 8.73% 55위 | 맞음 | FACT |
| 5주차 TOP5 모두 KODEX 2차전지산업레버리지 보유 | 맞음 | FACT |
| 6주차 TOP5 모두 TIGER 반도체TOP10레버리지 보유 | 맞음 | FACT |
| 6주차 TOP5 중 4명 SOL 조선TOP3플러스레버리지 | 맞음 | FACT |
| 7주차 상위권 수익률 급락 | 맞음 | FACT |
| 남준 인버스 거래 -4.48% | 맞음 | FACT |
| 남준 최종 47.82%, 비서실장 44.64% | 맞음 | FACT |
| 남준: 사이클·주도섹터·재진입 중시 | 맞음 | 인터뷰 FACT |
| 남준: 초반 레버리지, 이후 인버스·금 활용 | 맞음 | 시상식 발언 FACT |
| “레버리지가 알파가 아니다” | 방향 타당하나 인과 표현 주의 | “레버리지는 top performance의 필요조건이 아님”으로 수정 |
| “집중이 상위권 핵심” | 과도함 | 일부 사례 관찰. 전체 정확 비중 미공개 |
| “재진입 신뢰도 높음” | 과도함 | 우승자 1인의 원칙. 통계적 우위는 미검증 |
| “crowding × lifecycle” | 전략 아이디어 | HYPOTHESIS |
| “Regime + Vehicle이 직접 지지됨” | 표현 과도 | 사례와 정합적인 HYPOTHESIS |

## 4.2 제1회(2024) 대회 검증 지식

### 4.2.1 대회 구조

| 항목 | 검증 값 | 타입 | 비고 |
|---|---|---|---|
| 기간 | 2024-08-26 ~ 2024-12-31 | FACT | 공식 기사 |
| 초기자금 | 10억원 | FACT | 코스콤 모의투자 |
| 참가자 | 약 1,500명 | FACT | 최종 기사 |
| 부문 수 | 5개 | FACT | 밸류업/연금/국내/글로벌/자율 |
| 대상 | Ytm / 김대현 | FACT | 글로벌 |
| 대상 수익률 | 54.05% | FACT | 최종 결과 |
| 대상 선정 | 부문별 상대적 성과 | FACT | 정확 계산식 UNKNOWN |
| 최고수익률상 | 스튜아리 43.83% | FACT | 대상 수상자 제외 구조 |
| 2026 직접 비교 가능성 | 제한적 | INFERENCE | 기간·시상구조 차이 |

#### 중요 보정 (대상 산식)
제1회 기사에는 “상대 수익률” 또는 “부문별로 상대적으로 가장 우수한 성과”라는 표현이 확인되지만, 다음 수식은 공개 자료에서 확인되지 않는다.

```text
participant_return - division_average_return
```

따라서 **“부문 평균 대비 초과수익률이 대상 산식”이라고 단정하면 안 된다.**

지식베이스에는:

```yaml
target_scoring_formula:
  status: UNKNOWN
  known: "부문별 상대적 성과를 고려"
  unknown:
    - 평균 대비 초과수익률인지
    - 표준화 점수인지
    - 다른 보정값이 있었는지
```

로 저장한다.

### 4.2.2 최종 수상 결과

| 구분 | 참가자 | 부문 | 최종 수익률 | 타입 |
|---|---|---|---:|---|
| 대상 | Ytm / 김대현 | 글로벌 | 54.05% | FACT |
| 베스트포트폴리오 | 스튜아리 / 고성호 | 글로벌 | 43.83% | FACT |
| 자율 최우수 | etf_king / 임병주 | 자율 | 42.59% | FACT |
| 연금 최우수 | 혜화동스캘퍼 / 정진서 | 연금 | 36.55% | FACT |
| 글로벌 최우수 | Hereum / 정웅 | 글로벌 | 36.34% | FACT |
| 밸류업 최우수 | 구기 / 장민국 | 밸류업 | 35.78% | FACT |
| 국내 최우수 | 생거진천 / 임봉규 | 국내 | 26.69% | FACT |

#### 관찰
상위 수상자들의 공개 인터뷰에서는 서로 다른 접근이 등장한다.
- Contrarian
- 주도주 변화 추적
- 정책/대선 catalyst
- 산업 집중
- 레버리지
- 비교적 낮은 회전

이것은 **하나의 전략만이 성공했다는 증거가 없다는 것**을 의미한다.

### 4.2.3 2024년 9월 — 김병수

```yaml
claim_id: 2024_SEP_KBS_001
claim_type: FACT
participant: 김병수
as_of_date: 2024-09-30
cumulative_return_pct: 38.68
monthly_return_pct: 35.0
initial_capital_krw: 1000000000
profit_krw: 386760000
known_positions:
  - TIGER 미국필라델피아반도체레버리지(합성)
  - TIGER 차이나항셍테크레버리지(합성 H)
```

기사에서 확인되는 추가 정보:
- TIGER 차이나항셍테크레버리지(합성 H)를 9월 30일 약 4.5억원 매수
- 당시 해당 포지션 수익률 82.88%
- TIGER 미국필라델피아반도체레버리지(합성) 약 2억원 매수
- 해당 포지션 평가수익률 6.49%

- **올바른 해석:** `글로벌 이벤트 → 테마 → 레버리지`라는 **사례가 존재했다**.
- **잘못된 해석:** “글로벌 이벤트 기반 레버리지 전략의 기대수익률이 높다.” (공개 사례 1개로 기대수익률 추정 불가).

### 4.2.4 2024년 11월 — 스튜아리 / 고성호

```yaml
participant: 스튜아리
monthly_return_pct: 22.18
known_traded_etfs:
  - ACE 테슬라밸류체인액티브
  - KODEX 테슬라밸류체인FactSet
  - ACE 미국달러SOFR금리(합성)
  - SOL 미국AI전력인프라
  - KODEX 미국반도체MV
self_reported_method:
  - 주도주 변화 관찰
  - 엘리어트 파동이론 기반 차트 분석
```

- **프로젝트에 남길 것:** 특정 종목명을 하드코딩하지 않고 아래 국면을 측정하는 구조:
  ```text
  leadership emergence
  leadership acceleration
  relative-strength persistence
  leadership decay
  ```
- **주의:** 엘리어트 파동 자체가 유효하다는 증거는 없다. 이는 참가자의 **자기보고 방법론**이다.

### 4.2.5 방산 → 조선 전환 — 구기 / 장민국

```yaml
participant: 구기
initial_theme: PLUS K방산
initial_position_description: "자산 대부분"
return_on_first_theme_pct: 16.91
rotation_time: "2024-11 초"
next_theme: SOL 조선TOP3플러스
return_on_second_theme_pct: 28.43
self_reported_catalysts:
  - 미국 대선
  - 트럼프 LNG 공약
  - 원화 약세
  - 수출기업 마진 기대
```

- **FACT:** 실제 해당 참가자가 이 논리로 포지션을 전환했다고 말했다.
- **HYPOTHESIS:** 가격·수급 모델에 catalyst 정보를 추가하면 성과가 개선될 수 있다. (백테스트 필요).

### 4.2.6 대상 Ytm / 김대현

#### 확인된 자기보고 원칙

```yaml
philosophy: Contrarian
rules:
  - 극단적 낙관 시 보수적
  - 극단적 공포 시 낙관적
  - 판단 오류를 인정하면 빠르게 리스크 축소
```

대회 초반에는 금리 하락을 예상해 성장주와 장기채를 섞었으나, 경제지표와 장기금리 움직임이 예상과 달라 손실을 보고 전략을 바꿨다고 인터뷰했다.  
ACE 테슬라밸류체인액티브에 큰 비중을 실었다고도 밝혔다.

#### 프로젝트 해석
`Contrarian = 우승 전략`으로 저장하지 않는다. 대신 다음과 같이 분해한다:

```yaml
hypotheses:
  - extreme_sentiment_reversal
  - failed_thesis_fast_exit
  - contrarian_inside_structural_uptrend
```

---

## 4.3 제2회(2025) 대회 심층 타임라인 및 행동 분석

### 4.3.1 기본 구조

```yaml
competition:
  start_date: 2025-09-22
  end_date: 2025-11-14
  duration: 8주
  initial_capital_krw: 1000000000
  participants_approx: 1000
  divisions:
    - 국내주식형
    - 연금투자형
    - 글로벌형
    - 자율형
  award_rule:
    grand_prize: 전체 수익률 1위
    second_prize: 전체 수익률 2위
```

제3회와 가장 비교 가능한 역사 사례다.

### 4.3.2 1주차

| 순위 | 참가자 | 누적수익률 |
|---:|---|---:|
| 1 | YSK | 7.85% |
| 2 | 쉬었음청년 | 7.37% |
| 3 | 고려대통계학과 | 6.50% |
| 4 | 중위1223 | 5.63% |
| 5 | 타고나다 | 5.37% |

TOP5 모두 자율형.

YSK의 기사상 행동:
```text
코스피 상승 → KODEX 레버리지
미국 반도체 강세 → KODEX 반도체레버리지
국내시장 숨고르기 → KODEX 200선물인버스2X
```

- 행동 자체: `FACT`
- 이것이 최적의 `Regime + Vehicle` 구조라는 주장: `HYPOTHESIS`

### 4.3.3 2주차

| 순위 | 참가자 | 누적수익률 |
|---:|---|---:|
| 1 | 쉬었음청년 | 16.55% |
| 2 | 고려대통계학과 | 14.20% |
| 3 | 남준 | 12.86% |
| 4 | 범고래 | 9.76% |
| 5 | 정훈 | 9.35% |

TOP5 모두 자율형.

#### 부문별 평균·최저

| 부문 | 평균 | 최저 |
|---|---:|---:|
| 자율형 | 0.24% | -19.89% |
| 국내주식형 | 0.31% | -5.09% |
| 글로벌형 | 0.50% | -3.31% |
| 연금투자형 | 0.54% | -4.87% |

#### 부문별 최고 — 원문 충돌 (`CONFLICT`)

기사 앞부분:
- 국내주식형: 9.01%
- 글로벌형: **8.64%**
- 연금형: 8.56%

동일 기사 뒤 요약:
- 국내주식형: 9.01%
- 글로벌형: **8.56%**
- 연금형: 8.56%

따라서:

```yaml
global_division_max_week2:
  preferred_value_pct: 8.64
  status: CONFLICT
  reason: "동일 기사 순위 문단과 요약 문단 불일치"
```

데이터베이스에서 자동으로 `A`급 확정 숫자로 사용하지 않는다.

### 4.3.4 고회전 사례

쉬었음청년:
```yaml
week2_return_pct: 16.55
week2_rank: 1
turnover_pct: 409.18
week4_return_pct: 19.41
week4_rank: 12
```

고려대통계학과:
```yaml
week2_return_pct: 14.20
week2_rank: 2
week4_return_pct: 8.73
week4_rank: 55
```

- **OBSERVATION:** 초기 고수익·고회전이 최종 상위권 지속성을 보장하지 않았다.
- **금지된 결론:** `turnover가 낮을수록 좋다`도 증명되지 않았다.
- **검증 대상:**
  ```text
  turnover
  × signal quality
  × regime persistence
  × transaction/friction assumptions
  × whipsaw frequency
  ```

### 4.3.5 4주차

| 순위 | 참가자 | 누적수익률 |
|---:|---|---:|
| 1 | 범고래 | 40.60% |
| 2 | 노환준 | 39.48% |
| 3 | 남준 | 32.64% |
| 4 | 간절함 | 32.09% |
| 5 | 동탄짝귀 | 30.06% |

확인된 반도체 관련 사례:
- 범고래: TIGER 반도체TOP10레버리지, KODEX 반도체레버리지
- 노환준: 미국필라델피아반도체레버리지, KODEX 반도체레버리지
- 남준: 대회 첫날 KODEX 반도체레버리지 진입, 10월 17일 정리, 약 33.85% 실현손익

- **OBSERVATION:** 이 구간의 상위권 성과와 반도체 랠리가 강하게 겹쳤다.
- **HYPOTHESIS:** `Sector Leadership Strength`가 ETF 선택의 핵심 설명변수일 수 있다.

### 4.3.6 5주차

| 순위 | 참가자 | 누적수익률 |
|---:|---|---:|
| 1 | 노환준 | 72.28% |
| 2 | 정훈 | 50.03% |
| 3 | 간절함 | 49.61% |
| 4 | 남준 | 49.17% |
| 5 | 범고래 | 42.84% |

#### 매우 강한 FACT
TOP5 전원이 `KODEX 2차전지산업레버리지`를 보유한 것으로 기사에 기재됐다.  
그 주 해당 ETF의 최근 일주일 수익률은 기사 기준 29.31%였다.

#### 해석 경계
이 사실은 `crowding caused future reversal`을 의미하지 않는다. 가능한 설명은 최소 세 가지다:
1. 강한 추세를 여러 상위 참가자가 동시에 포착
2. 공개 랭킹/매수 정보로 인해 추종 매매가 발생
3. 우연히 동일한 매우 강한 테마에 노출

공개 데이터만으로 구분 불가.

### 4.3.7 6주차

| 순위 | 참가자 | 누적수익률 |
|---:|---|---:|
| 1 | 노환준 | 65.63% |
| 2 | 남준 | 59.73% |
| 3 | 정훈 | 55.01% |
| 4 | 범고래 | 52.50% |
| 5 | 항해 | 50.96% |

- **FACT:**
  - TOP5 전원이 `TIGER 반도체TOP10레버리지` 보유
  - TOP5 중 4명이 `SOL 조선TOP3플러스레버리지` 보유
  - TOP5 중 남준을 제외한 4명의 회전율이 100% 이상
  - 남준 회전율은 기사 기준 18.25%
- **중요한 관찰:** 상위권이라는 동일 결과가 높은 회전과 상대적으로 낮은 회전 양쪽에서 모두 나타났다. 따라서 “고회전” 자체를 목적함수로 두면 안 된다.

### 4.3.8 7주차 — Tournament Risk의 핵심 데이터

#### 순위
| 순위 | 참가자 | 누적수익률 |
|---:|---|---:|
| 1 | 남준 | 46.99% |
| 2 | 노환준 | 41.64% |
| 3 | 정훈 | 38.90% |
| 4 | 비서실장 | 38.16% |
| 5 | 범고래 | 36.53% |

#### 6주차 → 7주차 변화
| 참가자 | 6주차 | 7주차 | 변화 |
|---|---:|---:|---:|
| 노환준 | 65.63% | 41.64% | -23.99%p |
| 남준 | 59.73% | 46.99% | -12.74%p |
| 정훈 | 55.01% | 38.90% | -16.11%p |
| 범고래 | 52.50% | 36.53% | -15.97%p |

노환준:
```text
5주차 72.28% → 6주차 65.63% → 7주차 41.64%
5주차 고점 대비 7주차 누적수익률 감소: 72.28 - 41.64 = 30.64%p
```

#### Peak-to-Current Giveback
추천 정의:
```text
PeakReturn_t = max(R_0 ... R_t)

Giveback_t = PeakReturn_t - R_t
```

상대값으로는:
```text
NormalizedGiveback_t = (PeakEquity_t - Equity_t) / PeakEquity_t
```

둘을 구분한다:
- `%p giveback`: 대회 수익률 관점
- `equity drawdown %`: 자산곡선 관점

- **FACT:** 큰 수익률 반납이 실제 상위권에서 발생했다.
- **HYPOTHESIS:** `giveback-aware risk reduction`이 최종 순위를 개선할 수 있다. (두 문장을 혼동하지 않는다).

### 4.3.9 인버스 전환의 함정

7주차 한 주간:
- 일부 참가자들은 `KODEX 200선물인버스2X` 단일 포트폴리오로 약 7.12%를 기록
- 전체 누적 1위 남준은 11월 5일 매수 후 6일 매도한 같은 ETF에서 약 -4.48% 손실

- **FACT:** 같은 인버스 ETF라도 진입 시점에 따라 결과가 크게 달랐다.
- **프로젝트 해석:** `Regime classification`과 `turning-point timing`을 별도 문제로 모델링한다:

```yaml
regime_model:
  output:
    - bullish
    - neutral
    - bearish

transition_model:
  output:
    - probability_of_breakdown
    - probability_of_recovery
    - transition_confidence

execution_model:
  output:
    - entry_trigger
    - exit_trigger
    - reentry_trigger
```

### 4.3.10 제2회 최종 결과

| 구분 | 참가자 | 부문 | 최종수익률 |
|---|---|---|---:|
| 대상 | 남준 / 박남준 | 자율형 | 47.82% |
| 최우수상 | 비서실장 / 전진우 | 국내주식형 | 44.64% |
| 부문 우수 | 마이더스6 | 국내주식형 | 공개 기사에서 수치 미확인 |
| 부문 우수 | Lobe / 조영준 | 연금투자형 | 공개 기사에서 수치 미확인 |
| 부문 우수 | 깐부통닭 / 주원중 | 글로벌형 | 공개 기사에서 수치 미확인 |
| 부문 우수 | 노환준 | 자율형 | 공개 기사에서 최종 수치 미확인 |

#### 레버리지 해석 수정
- 잘못된 단순화: `Leverage = Alpha`
- 더 정확한 문장:
  ```text
  레버리지는 상위권 성과의 필요조건이 아니다.
  공개 사례만으로 레버리지가 알파의 원인인지 추정할 수 없다.
  ```
- 비서실장은 레버리지·인버스가 제한된 국내주식형에서 44.64%로 전체 2위를 기록했다.
- 따라서 모델은 `Signal Quality + Asset/Theme Selection + Timing + Exposure Sizing + Vehicle`을 분리해 평가한다.

### 4.3.11 남준 / 박남준 — 직접 확인된 원칙

수상자 인터뷰와 시상식 발언에서 확인되는 내용:

```yaml
participant: 박남준
nickname: 남준
final_return_pct: 47.82
self_reported_principles:
  - 사이클 기반 방향성 판단
  - 추세가 강한 섹터 ETF 활용
  - 손절 기준보다 재진입 기준을 중요하게 설정
  - 변동성 구조 이해
  - 리스크 통제
  - 추세 추종
competition_path:
  early_phase: 레버리지 ETF 중심
  later_phase:
    - 인버스 ETF
    - 금 ETF
```

- **중요:** 위 내용은 `A3: 우승자 자기보고`다.
- **금지 문장:** `재진입 기준이 손절 기준보다 통계적으로 더 중요하다.` (아직 검증되지 않음).
- **대체 저장:**
  ```text
  우승자는 재진입 기준을 손절 기준보다 더 중요하게 설정했다고 밝혔다.
  → re-entry policy를 별도 연구축으로 검증한다.
  ```

---

## 4.4 과거 데이터의 한계 및 활용 역할 규정

### 4.4.1 강한 OBSERVATION
- **O1. 상위권은 반복적으로 강한 섹터/테마와 겹쳤다:** 2025 중반 반도체, 2차전지, 조선, 방산.
- **O2. 자율형이 초중반 순위의 오른쪽 꼬리를 지배했다:** 1~6주차 TOP5는 자율형 참가자가 지속적으로 강했다.
- **O3. 자율형의 하방도 컸다:** 2주차 자율형 최저 -19.89%, 평균 0.24%.
- **O4. 높은 중간 수익률은 최종 수익률과 동일하지 않았다:** 5~7주차 큰 giveback이 존재했다.
- **O5. 동일한 성공 경로 안에서도 turnover가 크게 달랐다:** 남준은 6주차 기준 상위권이면서 기사상 회전율 18.25%, 다른 TOP5 네 명은 100% 이상.

### 4.4.2 HYPOTHESIS로만 유지

| 가설 | 이유 |
|---|---|
| 주도섹터 상대강도 모델 | 사례와 정합적이나 통계 검증 필요 |
| Regime + Transition | 우승자 발언/인버스 사례와 정합적 |
| Re-entry model | 우승자가 직접 중요성을 언급 |
| Crowding lifecycle | 5·6주차 동종 ETF 쏠림에서 아이디어 도출 |
| Concentration | 일부 사례 존재하지만 전체 포트폴리오 비중 미공개 |
| Event catalyst overlay | 2024 대선·정책 사례 존재 |
| Cross-market signal | 미국 반도체/중국/환율 등 사례 존재 |
| Pullback timing | Contrarian 사례를 단기화한 연구 아이디어 |
| Tournament risk budget | 대회 구조와 후반 giveback에서 도출 |

### 4.4.3 공개 기사 데이터의 한계 (확인 불가)
- 전체 참가자의 일별 equity curve
- 참가자별 정확한 ETF 비중
- 모든 주문의 체결시각
- 전체 주문내역
- 일별 현금 비중
- 모든 날짜의 완전한 leaderboard
- HTS의 정확한 체결 로직
- 주문 종류별 처리
- 슬리피지/수수료 설정
- 거래정지/상하한가 상황의 모의체결 처리
- 2024 대상 상대성과의 정확한 계산식
- 공개되지 않은 최종 부문별 우수상 수익률

따라서 기사 데이터는 `Strategy validation dataset`이 아니라 `Tournament history / case-study / hypothesis dataset`이다.

### 4.4.4 과거 대회 데이터를 사용하는 정확한 역할

```text
2024 제1회 = 아이디어 생성
2025 제2회 = 8주 tournament 구조의 사례 연구
장기간 point-in-time ETF 데이터 = 통계적 검증
2026 실시간 데이터 = 실제 signal generation / model updating
```

과거 대회 기사만으로 전략을 선택하지 않는다.

---

# PART 5. 퀀트 시스템 아키텍처 및 전략 가설

## 5.1 퀀트 시스템 계층 아키텍처

```text
[Data Layer]
    ↓
[Market State / Regime]
    ↓
[Leadership Ranking]
    ↓
[Signal Confidence]
    ↓
[Vehicle Selection]
    ↓
[Exposure / Concentration]
    ↓
[Execution]
    ↓
[Exit / Watch / Re-entry]
    ↓
[Tournament Risk Controller]
```

### 5.1.1 Data Layer

**필수:**
- OHLCV, 거래대금
- ETF 순자산/NAV 가능 시, 괴리율 가능 시
- ETF 구성종목, 기초지수
- 레버리지 배수, ETF 상장일, 운용사, 대회 허용 여부
- 테마/섹터 taxonomy
- 해외 기초자산의 전일 수익률, 환율, 금리, 시장 지수, 변동성 지표
- 뉴스/catalyst 메타데이터

**선택:**
- 외국인/기관 수급, 선물 베이시스, 옵션 변동성
- 검색량/뉴스량
- 대회 랭킹 및 보유/매수 집중 정보

## 5.2 전략 가설 우선순위 매트릭스

### Tier 1 — 가장 먼저 검증
1. `Sector/Theme Relative Strength`
2. `Leadership Persistence`
3. `Market Regime`
4. `Regime Transition`
5. `Vehicle Selection`
6. `Exit → Watch → Re-entry`
7. `Terminal Giveback Control`  
*(이유: 제2회 사례와 2026 대회 구조에 가장 직접적으로 연결됨)*

### Tier 2
1. Cross-market lead/lag
2. Breadth
3. Volatility-aware sizing
4. Leadership × Crowding
5. Concentration optimization

### Tier 3
1. News catalyst
2. 정책/대선 이벤트
3. short-horizon reversal
4. intraday high-turnover

### Core에서 제외
- 특정 섹터 하드코딩
- 2025 우승자 거래 그대로 복제
- 장기 contrarian 단독 엔진
- 장기 fundamental valuation 단독 엔진
- 무조건 100% 레버리지
- 무조건 높은 turnover
- 무조건 후반 de-risk

## 5.3 피처 엔지니어링 명세 (Feature Engineering)

### 5.3.1 Leadership Feature 설계

1. **절대 모멘텀:** `ret_1d`, `ret_3d`, `ret_5d`, `ret_10d`, `ret_20d`
2. **상대강도:**
   ```text
   RS_i(t) = Return_i(t, lookback) - MedianReturn_universe(t, lookback)
   ```
   또는 rank percentile:
   ```text
   RS_rank_i(t) = percentile_rank(Return_i)
   ```
3. **가속도:**
   ```text
   MomentumAcceleration = RS_5d - RS_20d
   ```
4. **Breadth (테마 내부 구성종목 중):**
   - 5일 상승 종목 비율
   - 20일 이동평균 상회 종목 비율
   - 신고가 종목 비율
   - 거래대금 증가 종목 비율
5. **Persistence:**
   ```text
   top_decile_days_5
   top_quintile_days_10
   rank_autocorrelation
   ```

### 5.3.2 Regime Feature 설계

1. **시장 방향:**
   - KOSPI 5/20/60일 추세, KOSDAQ 추세
   - 미국 Nasdaq/SOX 전일 추세, 해외 선물, USD/KRW
2. **변동성:**
   - realized vol 5/10/20, intraday range, downside semivol, gap frequency
3. **Breadth:**
   - 상승/하락 종목 수, 20일 고가 돌파 비율, 대형주 vs 중소형주, 섹터별 확산도
4. **Regime Example:**
   ```yaml
   regime:
     trend:
       bullish:
       neutral:
       bearish:
     volatility:
       low:
       normal:
       high:
     breadth:
       narrow:
       normal:
       broad:
   ```
   최종 상태 예: `Bull + Broad + NormalVol`, `Bull + Narrow + HighVol`, `Bear + BroadDown + HighVol`, `Transition` 등.

### 5.3.3 Crowding Feature — 반드시 분리해서 해석

대회 랭킹 페이지에서 매수집중·보유계좌 수가 제공된다면:

```text
crowding_score = z(buy_amount_rank) + z(holder_count) + z(leaderboard_holder_count)
```

하지만 crowding은 단독 매도 신호가 아니다.

- **가능한 상호작용:**
  - `Strong Leadership + Rising Crowding` → 추세 확인 가능성
  - `Weakening Leadership + Extreme Crowding` → late-stage 위험 후보
- **검증식:**
  ```text
  future_return ~ leadership + crowding + leadership * crowding
  ```

## 5.4 상태 전이 모델 및 리스크 컨트롤러

### 5.4.1 Re-entry를 별도 상태로 관리

단순한 `BUY → SELL`이 아니라 아래 구조를 둔다:

```text
HOLD → REDUCE/EXIT → WATCH → RE-ENTRY
```

**Watch 상태 필드:**
```yaml
watch:
  previous_leader: true
  exit_reason:
  days_since_exit:
  price_from_exit:
  rs_recovery:
  volume_recovery:
  market_regime_recovery:
  reentry_score:
```

**검증 지표:**
- exit 후 최대 추가 하락
- re-entry까지 경과 일수
- 재진입 후 1/3/5/10일 수익률
- false re-entry rate
- missed continuation return
- terminal return contribution

### 5.4.2 Tournament Risk Controller

일반 장기투자와 달리 대회에는 종료일이 고정되어 있다.

**상태 변수:**
```yaml
tournament_state:
  days_remaining:
  current_return:
  peak_return:
  giveback:
  current_rank:
  leader_return:
  gap_to_leader:
  gap_to_second:
  realized_vol:
  exposure:
```

**연구할 정책:**
- **Early phase:** 우승권 진입을 위해 더 높은 risk budget이 필요한지 검증.
- **Middle phase:** 리더십 유지 시 추세를 따라가되 whipsaw 방지.
- **Late phase:** 현재 순위·leader gap·giveback·남은 일수를 사용해 risk budget을 동적으로 조정.
- **절대 금지:** “마지막 주에는 무조건 보수적으로” 같은 하드코딩. (순위가 100위이고 leader와 +40%p 차이가 난다면 보수화가 부적합할 수 있음).

즉:
```text
RiskBudget = f(
  days_remaining,
  rank,
  gap_to_leader,
  current_return,
  giveback,
  signal_strength,
  volatility
)
```
형태로 검증한다.

## 5.5 목적 함수 및 평가 메트릭 체계

1. **기본:**
   ```text
   TerminalReturn = R(T)
   ```
2. **리스크 보조지표:**
   ```text
   MaxDrawdown
   PeakToFinalGiveback
   Worst5DayReturn
   Turnover
   WhipsawCount
   ReentrySuccessRate
   ```
3. **대회 특화:**
   ```text
   Top1ThresholdHitRate
   Top5ThresholdHitRate
   TerminalRankProxy
   Probability(Return > historical_top1)
   ```
   *(단, 과거 대회 2개만으로 historical_top1 분포를 추정하면 표본이 지나치게 작으므로 rolling historical windows 사용)*

---

# PART 6. 백테스트 및 검증 데이터셋 프로토콜

## 6.1 Point-in-Time Universe 관리 규약

가장 큰 백테스트 오류 중 하나는 **현재 존재하는 ETF를 과거에도 존재했던 것처럼 넣는 것**이다.

각 ETF에 메타데이터를 둔다:
```yaml
ticker:
  listing_date:
  delisting_date:
  manager:
  category:
  leverage_multiple:
  inverse:
  competition_eligible_2026:
  eligibility_start:
  eligibility_end:
```

백테스트 시 필터 조건:
```text
date >= listing_date
AND (date <= delisting_date OR delisting_date is null)
AND competition_eligible(date) == true
```
만 사용한다.

## 6.2 레버리지 ETF 모델링 및 시뮬레이션 지침

레버리지/인버스 ETF는 일반적으로 **기초지수의 일간 수익률 배수**를 목표로 한다. 따라서:

```text
2x ETF 10일 수익률 ≠ 기초지수 10일 수익률 × 2
```

일 수 있으며 변동성이 클수록 복리효과로 괴리가 커질 수 있다.

- **백테스트 원칙:** 가능하면 **실제 ETF 가격 시계열**을 직접 사용한다.
- **피해야 할 방식:**
  ```python
  leveraged_return = underlying_20d_return * 2
  ```
- **더 나은 방식:**
  ```python
  # 실제 ETF가 과거에 존재한다면 실제 ETF 수익률 사용
  # 존재하지 않았던 기간은 임의 합성하지 않거나, 별도 synthetic 실험으로 명확히 구분
  ```

## 6.3 역사적 검증 데이터셋(Validation Dataset) 설계

`2010s~2026 역사적 40거래일 구간`을 그대로 사용하면 위험하므로 반드시 보완한다:

1. **기간 길이 민감도:** 2026 실제 대회는 달력상 8주이며 거래일 수는 휴일에 따라 달라지므로, 고정 40일 외에 **38~42 trading days** 민감도 분석을 병행한다.
2. **Point-in-Time ETF Universe:** 미래 상장 ETF를 과거에 삽입하지 않는다.
3. **Market Regime Stratification:** 전체 rolling window 결과만 평균내지 않고 bull, bear, high vol, low vol, narrow leadership, broad rally 등으로 층화 분석한다.
4. **Walk-forward:** 파라미터는 미래 window를 보지 않고 결정한다.
5. **Selection bias 방지:** 2024·2025 우승 섹터를 보고 만든 feature가 과거에도 잘 됐다고 주장하려면 별도 out-of-sample이 필요하다.

## 6.4 백테스트 엔진 최소 요구사항

```yaml
backtest:
  price:
    adjusted: true
    point_in_time: true

  universe:
    actual_listing_dates: true
    delisted_products: include_if_possible
    competition_filter: true

  execution:
    order_delay: explicit
    fill_price: explicit
    slippage: scenario_test
    commissions: scenario_test
    liquidity_constraint: scenario_test

  validation:
    walk_forward: true
    out_of_sample: true
    parameter_stability: true
    multiple_testing_control: recommended

  metrics:
    - terminal_return
    - max_drawdown
    - peak_to_final_giveback
    - turnover
    - worst_5d
    - win_rate
    - tail_loss
```

---

# PART 7. 소스 레지스트리 및 운영 컴플라이언스

## 7.1 저작권 및 운영 컴플라이언스 주의사항

- 머니투데이 기사 페이지에는 기사 콘텐츠의 무단 전재·복사·배포 및 AI 학습을 금지한다는 표시가 있다.
- 따라서 프로젝트 지식베이스에는 기사 전문을 복제하지 않고:
  - 수치·날짜·종목·순위 등 사실을 구조화한 요약
  - 참가자 발언의 짧은 요약
  - 원문 URL
  - 출처 ID
  - 검증일
  을 저장하는 방향이 안전하다.
- 기사 전문을 대량 크롤링해 모델 학습 데이터로 사용하는 것은 별도의 이용조건·권리 검토가 필요하다.

## 7.2 출처 URL 레지스트리 (Source Registry)

### 2026 제3회 공식 소스
- **MT_2026_OFFICIAL**: https://www.mt.co.kr/etf/join/index.html
- **MT_2026_ANNOUNCEMENT**: https://www.mt.co.kr/amp/stock/2026/08/14/2026081416135978575

### 2025 제2회 주간 및 결과 기사
- **MT_2025_RULES**: https://www.mt.co.kr/stock/2025/08/18/2025081505332252684
- **MT_2025_W01**: https://www.mt.co.kr/stock/2025/09/27/2025092618035044343
- **MT_2025_W02**: https://www.mt.co.kr/stock/2025/10/03/2025100219480951844
- **MT_2025_W04**: https://www.mt.co.kr/stock/2025/10/19/2025101913260159704
- **MT_2025_W05**: https://www.mt.co.kr/stock/2025/10/25/2025102418201958573
- **MT_2025_W06**: https://www.mt.co.kr/stock/2025/11/02/2025110210422182221
- **MT_2025_W07**: https://www.mt.co.kr/stock/2025/11/09/2025110718053119988
- **MT_2025_FINAL**: https://www.mt.co.kr/stock/2025/11/19/2025111814504029132
- **MT_2025_WINNER_INTERVIEW**: https://www.mt.co.kr/stock/2025/11/25/2025112414165129357
- **MT_2025_AWARDS**: https://www.mt.co.kr/stock/2025/11/25/2025112516260834930

### 2024 제1회 주간 및 결과 기사
- **MT_2024_RULES**: https://www.mt.co.kr/stock/2024/08/01/2024073110504417965
- **MT_2024_STRUCTURE**: https://www.mt.co.kr/stock/2024/08/01/2024073116180688061
- **MT_2024_SEP**: https://www.mt.co.kr/stock/2024/10/02/2024093017512784826
- **MT_2024_NOV**: https://www.mt.co.kr/stock/2024/12/05/2024120514443448617
- **MT_2024_DEC**: https://www.mt.co.kr/stock/2025/01/11/2025011016442896090
- **MT_2024_FINAL**: https://www.mt.co.kr/amp/stock/2025/01/03/2025010314165324996
- **MT_2024_AWARDS**: https://www.mt.co.kr/stock/2025/01/21/2025012114302323396
- **MT_2024_WINNER_INTERVIEW**: https://www.mt.co.kr/stock/2025/01/21/2025012016181576680
- **MT_2024_YTM_DEEP**: https://www.mt.co.kr/stock/2025/02/15/2025021316393922718

### 레버리지 ETF 구조 참고
- **KRX / KIND 레버리지 ETF 유의사항**: https://kind.krx.co.kr/disclosure/etfisudetail.do?method=searchEtfIsuSummary&strIsurCd=12263
- **SEC Investor Bulletin**: https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-alerts/sec
