#!/usr/bin/env bash
# 매일 16:00 KST 이후 재생성되는 MT ETF 투자왕 순위표 JSON 스냅샷을 보관한다.
# systemd 유닛 스펙(%Y-%m-%d 등)은 strftime 치환을 지원하지 않으므로,
# 날짜별 로그 파일명은 이 래퍼 스크립트에서 직접 만든다 (daily_pipeline.sh와 동일한 패턴).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/contest_archive_$(TZ=Asia/Seoul date +%F).log"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

echo "[$(date -Iseconds)] contest_archive.sh start" >> "$LOG_FILE"
"$UV_BIN" run mt-etf contest-archive >> "$LOG_FILE" 2>&1
STATUS=$?
echo "[$(date -Iseconds)] contest_archive.sh end status=$STATUS" >> "$LOG_FILE"

exit "$STATUS"
