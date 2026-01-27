#!/bin/bash
# Download Qwen3-14B model (safetensors + GGUF)

set -e

# Configuration
MODEL_NAME="Qwen/Qwen3-14B"
GGUF_NAME="Qwen/Qwen3-14B-GGUF"
SAFETENSORS_DIR="/opt/models/vllm/qwen3-14b"
GGUF_DIR="/opt/models/gguf/qwen3-14b"

echo "=============================================="
echo "Downloading Qwen3-14B"
echo "=============================================="
echo ""
echo "This will download:"
echo "  - Safetensors (~28GB) for training"
echo "  - GGUF Q8_0 (~15GB) for inference"
echo ""
echo "Total: ~43GB"
echo ""

# Check if huggingface-cli is available
if ! command -v huggingface-cli &> /dev/null; then
    echo "Installing huggingface_hub..."
    pip install huggingface_hub[cli]
fi

# Create directories
echo "[1/2] Downloading safetensors for training..."
sudo mkdir -p "$SAFETENSORS_DIR"
sudo chown $(whoami):$(whoami) "$SAFETENSORS_DIR"

huggingface-cli download "$MODEL_NAME" \
    --local-dir "$SAFETENSORS_DIR" \
    --exclude "*.gguf" \
    --exclude "*.md"

echo ""
echo "[2/2] Downloading GGUF for inference..."
sudo mkdir -p "$GGUF_DIR"
sudo chown $(whoami):$(whoami) "$GGUF_DIR"

# Download Q8_0 quantization (best quality)
huggingface-cli download "$GGUF_NAME" \
    --include "*Q8_0.gguf" \
    --local-dir "$GGUF_DIR"

echo ""
echo "=============================================="
echo "Download complete!"
echo "=============================================="
echo ""
echo "Model locations:"
echo "  Training:  $SAFETENSORS_DIR"
echo "  Inference: $GGUF_DIR"
echo ""
echo "Next: Prepare your dataset and run training"
echo ""
