#!/usr/bin/env bash
# Start the DMOE LoRA Training WebUI
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(dirname "$SCRIPT_DIR")"

cd "$BASE_DIR"

# Create venv if needed
if [[ ! -d ".venv" ]]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate venv
source .venv/bin/activate

# Install dependencies if needed
if ! python -c "import fastapi" 2>/dev/null; then
    echo "Installing WebUI dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "==================================="
echo "  DMOE LoRA Training WebUI"
echo "==================================="
echo ""
echo "Open http://localhost:8082 in your browser"
echo "Press Ctrl+C to stop"
echo ""

python webui/app.py
