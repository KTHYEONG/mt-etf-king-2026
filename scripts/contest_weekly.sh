#!/usr/bin/env bash
# Weekly contest decision runner: retries are safe (idempotent), but a retry
# after a NO_DATA card needs --force, so pass --force only when the latest
# card for the target session decided NO_DATA.
set -u
cd "$HOME/mt-etf-king-2026" || exit 1
exec >> "logs/contest_weekly_$(date +%F).log" 2>&1

EXTRA=()
LATEST_JSON=$(ls -t results/contest_weekly/*.json 2>/dev/null | head -n 1)
if [ -n "${LATEST_JSON:-}" ] && grep -q '"action": "NO_DATA"' "$LATEST_JSON"; then
  EXTRA=(--force)
fi
exec "$HOME/.local/bin/uv" run mt-etf contest-weekly "${EXTRA[@]}"
