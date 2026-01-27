#!/bin/bash
# Train LoRA adapter using LlamaFactory

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

# Check arguments
if [ -z "$1" ]; then
    echo "Usage: ./scripts/train.sh <config.yaml>"
    echo ""
    echo "Example:"
    echo "  ./scripts/train.sh configs/qwen3_14b_lora.yaml"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Activate environment
source "$ROOT_DIR/activate.sh"

# Copy dataset_info.json to data directory
cp "$ROOT_DIR/configs/dataset_info.json" "$ROOT_DIR/data/dataset_info.json"

# Set environment variables
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

echo "=============================================="
echo "LoRA Training"
echo "=============================================="
echo "Config: $CONFIG_FILE"
echo ""

# Run training
cd "$ROOT_DIR/LlamaFactory"
llamafactory-cli train "$ROOT_DIR/$CONFIG_FILE"

echo ""
echo "=============================================="
echo "Training complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Convert to GGUF: ./scripts/convert-lora.sh saves/my-lora"
echo "  2. Run benchmark: ./scripts/benchmark.sh"
echo ""
