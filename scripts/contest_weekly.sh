#!/usr/bin/env bash
# Weekly contest decision runner: retries are safe (idempotent), but a retry
# after a NO_DATA card needs --force, so pass --force only when the latest
# card for the target session decided NO_DATA.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/contest_weekly_$(TZ=Asia/Seoul date +%F).log"

UV_BIN="${UV_BIN:-$HOME/.local/bin/uv}"

echo "[$(date -Iseconds)] contest_weekly.sh start" >> "$LOG_FILE"
EXTRA=()
LATEST_JSON=$(ls -t results/contest_weekly/*.json 2>/dev/null | head -n 1 || true)
if [ -n "${LATEST_JSON:-}" ] && grep -q '"action": "NO_DATA"' "$LATEST_JSON"; then
  EXTRA=(--force)
fi
set +e
"$UV_BIN" run mt-etf contest-weekly "${EXTRA[@]}" >> "$LOG_FILE" 2>&1
STATUS=$?
set -e
echo "[$(date -Iseconds)] contest_weekly.sh end status=$STATUS" >> "$LOG_FILE"

exit "$STATUS"
