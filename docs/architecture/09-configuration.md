# 09. Configuration — 설정 체계

모든 파라미터·규칙·매핑은 **코드가 아닌 YAML config** 에 존재합니다.

---

## 1. Config 파일 구조

```
configs/
├── base.yaml
├── tournament.yaml            # 대회 기간·후원사·규칙
├── sponsor_brands.yaml        # 운용사 ↔ ETF 브랜드 매핑
├── universe.yaml              # structural / deployment 모드
├── universe_manifest.yaml     # HTS 허용 isu_cd (확정 전 null)
├── features.yaml
├── portfolio.yaml
├── costs.yaml
├── gates.yaml
├── ml.yaml
├── taxonomy.yaml
└── instruments_override.yaml
```

---

## 2. 로드 우선순위

### 2.1 Secrets (API 키)

```
프로세스 환경변수 (KRX_OPENAPI_KEY 등)   ← 최우선 (CI·테스트·임시 override)
  → .env.enc (sops 복호화, 메모리만)     ← 로컬 기본
```

- **평문 `.env` 는 사용하지 않음.** `.env.enc` 만 저장소에 커밋.
- sops+age: `.sops.yaml` 에 age public key, private key 는 `~/.config/sops/age/keys.txt`.
- 경로 변경: `MT_ETF_ENV_ENC`.
- `config-check` / 로그는 credential **존재 여부(boolean)** 만 출력.

### 2.2 비시크릿 파라미터 (YAML)

```
base.yaml
  → {env}.yaml (dev/prod)
  → CLI --config override (선택)
```

`Settings` 객체가 secrets + 비시크릿 config 를 merge 한 단일 진실 공급원 (YAML merge 는 단계별 연결 예정).

---

## 3. 주요 Config 섹션

### 3.1 `base.yaml`

```yaml
data:
  root: "data"
calendar:
  exchange: "XKRX"
krx:
  base_url: "https://data-dbg.krx.co.kr/svc/apis"
  rate_limit_rps: 2
logging:
  level: "INFO"
```

### 3.2 `tournament.yaml`

```yaml
tournament:
  year: 2026
  start_date: "2026-09-21"
  end_date: "2026-11-13"
  capital: 1_000_000_000
  sessions: 36
  rules:
    sponsor_etf_only: true
    min_trade_unit: 1
  sponsors:
    infra: [금융투자협회, 한국거래소, 코스콤]
    asset_managers: [삼성자산운용, 미래에셋자산운용, ...]
  brand_map: "configs/sponsor_brands.yaml"
  manifest: null   # → universe_manifest.yaml
```

### 3.3 `sponsor_brands.yaml`

후원 **운용사** ↔ ETF 브랜드 매핑. 인프라 후원사(금투협·거래소·코스콤)는 ETF를 발행하지 않습니다.

```yaml
asset_managers:
  삼성자산운용:
    brands: ["KODEX"]
  미래에셋자산운용:
    brands: ["TIGER", "KoAct"]
  # ... KB, 한국투자, 신한, 한화, 타임폴리오, NH아문디, 키움, 하나
unknown_brand_policy: exclude   # deployment 시 UNKNOWN 브랜드 제외
```

### 3.4 `universe.yaml`

```yaml
universe:
  modes:
    structural:
      issuer_whitelist: null
    deployment:
      issuer_whitelist: "sponsor_asset_managers"
      manifest: null
```

- **structural**: 후원사 필터 없음 — 아이디어 탐색용
- **deployment**: 후원사 필터 ON — 전략 확정·ML·decide

### 3.5 `universe_manifest.yaml`

HTS 부문별 허용 `isu_cd` 리스트. 대회 직전 확정 전까지 `manifest: null`. 채워지면 `sponsor_brands` 보다 **우선**(교집합의 상한).

### 3.6 `features.yaml`

```yaml
features:
  momentum_windows: [3, 5, 10, 20, 40, 60]
  volatility_windows: [10, 20, 60]
  trend_windows: [20, 60, 120]
  flow_window: 20
  regime:
    risk_on_breadth: 0.60
    risk_off_breadth: 0.40
    crisis_drawdown: -0.10
```

### 3.7 `portfolio.yaml`

```yaml
portfolio:
  sizing_mode: "concentrated"   # equal | score_weighted | concentrated
  max_positions: 3
  max_single_weight: 0.80
  max_gross_exposure: 1.60      # Σ|w × mult| — 비중이 아닌 실효 노출 (INV-18)
  max_per_family: 1             # 동일 leverage_family 최대 1종목 (INV-17)
  min_cash: 0.05
  rebalance_threshold: 0.10
  exit:
    score_drop_pct: 0.30
    max_drawdown: -0.15
  reentry:
    cooldown_days: 5
  universe_mode: deployment     # decide 시 후원사 필터
```

### 3.8 `gates.yaml`

```yaml
gates:
  g1_prob_threshold: 0.30
  g1_min_improvement: 0.02
  g2a_ruin_threshold: -0.25
  g2a_max_prob: 0.05
  require_universe_mode: deployment
  diagnostics:
  - cvar_05
  - mdd_distribution
```

### 3.9 `ml.yaml`

```yaml
ml:
  universe_mode: deployment   # 학습·CV·채택 모두 후원사 ETF 기준
  label:
    horizon: 10                 # forward sessions
    target: "cs_percentile_rank"
  cv:
    scheme: "purged_walk_forward"
    purge: 10                   # ≥ horizon (INV-13)
    embargo: 10                 # ≥ horizon
    n_folds: 6
    window: "expanding"
  capacity:                     # §1.2 유효 표본 재산정 시 갱신
    num_leaves: 8
    max_depth: 4
    min_data_in_leaf: 100
    max_features: 25
    feature_fraction: 0.6
    bagging_fraction: 0.8
    lambda_l2: 1.0
```

`capacity` 는 표본 구조에서 도출된 **상한**입니다. 성능 개선을 위한 탐색 공간이 아닙니다 ([12 §4](12-ml-layer.md)).

### 3.10 `costs.yaml`

```yaml
costs:
  default:
    commission_bps: 1.5
    slippage_bps: 5.0
  grid:
  - { commission_bps: 1.0, slippage_bps: 3.0 }
  - { commission_bps: 1.5, slippage_bps: 5.0 }
  - { commission_bps: 3.0, slippage_bps: 10.0 }
```

---

## 4. Unknown Sentinel 처리

Config 에서 `Unknown` 은 Python `Unknown` sentinel 과 1:1 대응:

```yaml
allow_leverage: Unknown
```

런타임:

```python
if rules.allow_leverage is Unknown:
    for scenario in ["aggressive", "conservative"]:   # primary 먼저
        run_backtest(scenario=scenario)
```

결과 리포트에 **시나리오별 분리 표기** 필수. `primary: aggressive` 는 "conservative 를 안 돌린다"는 뜻이 아니라 **파라미터 튜닝 기준이 aggressive** 라는 뜻입니다. 두 시나리오 모두 게이트를 통과해야 합니다.

---

## 5. Taxonomy 매핑

`taxonomy.yaml` 예시:

```yaml
themes:
  semiconductor:
    index_keys: ["KRX반도체", "SOXX"]
    keywords: ["반도체", "SEMICON"]
  battery:
    index_keys: ["2차전지"]
    keywords: ["2차전지", "배터리"]
  ...
unclassified:
  fallback: true
```

매핑 실패 종목 → `unclassified`. leadership 에서 별도 bucket.

---

## 6. Instruments Override

KRX API 로 leverage 를 알 수 없는 종목:

```yaml
overrides:
  "122630":  # KODEX 레버리지
    leverage_factor: 2
  "114800":  # KODEX 인버스
    leverage_factor: -1
```

`resolve_leverage()` 는 override → 휴리스틱 → `Unknown` 순으로 조회.

---

## 7. Config 검증

`mt-etf config-check` 가 확인:

- 필수 키 존재
- 날짜 형식·순서 (`start < end`)
- `sessions` == calendar 실제 계산값
- `capital > 0`
- `Unknown` 필드가 scenario handler 에 등록됨
- `sponsor_brands.yaml` 의 asset_managers 가 tournament.sponsors 와 일치
- deployment 모드에서 `issuer_whitelist` 활성
- taxonomy fallback 존재

---

## 8. 원칙

1. **코드에 magic number 금지** — window, threshold, weight 는 전부 config
2. **Config 변경 = 실험** — git 으로 추적, 결과와 함께 기록
3. **Secret 은 YAML·평문 파일에 넣지 않음** — `.env.enc`(sops) 또는 프로세스 환경변수만. 평문 `.env` 생성 금지.
4. **대회 규칙 확정 시** — `Unknown` → `true/false` 로 업데이트 + scenario 결과 재산출
