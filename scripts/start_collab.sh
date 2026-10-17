#!/usr/bin/env bash
# Wrapper: Jupyter se lanza con Python para que un checkout Windows (CRLF) no rompa bash.
set -eu
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "${DIR}/start_collab.py" "$@"
