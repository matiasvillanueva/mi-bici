#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8888}"
ROOT="${1:-/workspace}"

if [ ! -d "${ROOT}/puntos" ]; then
  ROOT="$(cd "$(dirname "$0")/.." && pwd)"
fi

echo "Jupyter Lab → http://127.0.0.1:${PORT}/lab/tree/puntos"
echo ""

exec jupyter lab \
  --ip=0.0.0.0 \
  --port="${PORT}" \
  --no-browser \
  --allow-root \
  --ServerApp.token='' \
  --ServerApp.password='' \
  --ServerApp.allow_origin='*' \
  --ServerApp.root_dir="${ROOT}" \
  --ServerApp.default_url=/lab/tree/puntos
