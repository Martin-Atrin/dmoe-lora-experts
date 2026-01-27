#!/bin/bash
# Convert LoRA adapter to GGUF format

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Configuration
BASE_MODEL="/opt/models/vllm/qwen3-14b"

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: ./scripts/convert-lora.sh <lora-dir>"
    echo ""
    echo "Example:"
    echo "  ./scripts/convert-lora.sh saves/my-lora"
    exit 1
fi

LORA_DIR="$ROOT_DIR/$1"

if [ ! -d "$LORA_DIR" ]; then
    echo "ERROR: LoRA directory not found: $LORA_DIR"
    exit 1
fi

# Activate environment
source "$ROOT_DIR/activate.sh"

echo "=============================================="
echo "Converting LoRA to GGUF"
echo "=============================================="
echo "LoRA: $LORA_DIR"
echo "Base: $BASE_MODEL"
echo ""

# Convert
python "$ROOT_DIR/llama.cpp/convert_lora_to_gguf.py" \
    "$LORA_DIR" \
    --outfile "$LORA_DIR/adapter.gguf" \
    --base "$BASE_MODEL"

echo ""
echo "=============================================="
echo "Conversion complete!"
echo "=============================================="
echo ""
echo "Output: $LORA_DIR/adapter.gguf"
echo "Size: $(du -h "$LORA_DIR/adapter.gguf" | cut -f1)"
echo ""
echo "Next: ./scripts/benchmark.sh"
echo ""
