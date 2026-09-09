#!/usr/bin/env bash
# 세션 마감 후(기본 16:00 KST) 실행되는 데이터 파이프라인 배치.
# ingest -> normalize -> features(전체범위 재빌드) -> decide(챔피언 전략 추천 저장)
# 까지만 수행하며, 실제 매매 체결(HTS 수동 입력)은 여기서 하지 않는다.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/daily_refresh_$(date +%F).log"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

echo "[$(date -Iseconds)] daily_pipeline.sh start" >> "$LOG_FILE"
"$UV_BIN" run mt-etf daily-refresh --decide --as-of "$(date +%F)" >> "$LOG_FILE" 2>&1
STATUS=$?
echo "[$(date -Iseconds)] daily_pipeline.sh end status=$STATUS" >> "$LOG_FILE"

exit "$STATUS"
