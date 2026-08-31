#!/usr/bin/env bash
set -Eeuo pipefail
HOST="${1:-lnx}"
PORT="${2:-5000}"
curl --fail --silent --show-error --max-time 10 "http://${HOST}:${PORT}/api/health"
printf '\n'
curl --fail --silent --show-error --max-time 10 "http://${HOST}:${PORT}/api/state" | head -c 800
printf '\n'
