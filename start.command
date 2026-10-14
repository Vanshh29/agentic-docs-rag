#!/bin/bash
# Double-click this file (macOS) or run ./start.command. It sets everything up and opens the app.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "First-time setup (a few minutes)..."
  python3 -m venv .venv || { echo "Python 3.11+ is required."; read -r; exit 1; }
  .venv/bin/pip install -q -e . || { echo "Install failed."; read -r; exit 1; }
fi
[ -f .env ] || cp .env.example .env
if grep -q '^LLM_PROVIDER=claude_cli' .env && ! command -v claude >/dev/null; then
  echo "Install and log in to the Claude Code CLI first (https://claude.com/claude-code), then run this again."
  read -r; exit 1
fi
export PYTHONPATH=src
if [ ! -d data/chroma ] || [ -z "$(ls -A data/chroma 2>/dev/null)" ]; then
  echo "Indexing the sample manuals..."
  .venv/bin/python -m docrag.ingest --path data/samples --reset 2>&1 | tail -1
fi
echo "Starting at http://localhost:8000 (press Ctrl+C to stop)"
(sleep 4 && open http://localhost:8000) &
.venv/bin/python -m uvicorn docrag.api:app --port 8000
