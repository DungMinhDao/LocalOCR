#!/bin/zsh
# Launch LocalOCR (opens http://127.0.0.1:8765). Extra args: --port N --no-browser
cd "$(dirname "$0")"
exec uv run --quiet python -m localocr.server "$@"
