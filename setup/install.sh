#!/bin/bash
# Environment setup for LoRA training pipeline
# Supports: AMD ROCm and NVIDIA CUDA

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$ROOT_DIR/venv"

echo "=============================================="
echo "LoRA Training Pipeline - Environment Setup"
echo "=============================================="

# Detect GPU vendor
detect_gpu() {
    if command -v rocminfo &> /dev/null && rocminfo 2>/dev/null | grep -q "gfx"; then
        echo "rocm"
    elif command -v nvidia-smi &> /dev/null; then
        echo "cuda"
    else
        echo "cpu"
    fi
}

GPU_TYPE=$(detect_gpu)
echo "Detected GPU: $GPU_TYPE"

# Create virtual environment
echo ""
echo "[1/5] Creating Python virtual environment..."
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Install PyTorch
echo ""
echo "[2/5] Installing PyTorch for $GPU_TYPE..."

case $GPU_TYPE in
    rocm)
        # ROCm 6.3 wheels (adjust version as needed)
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/rocm6.3

        # Set ROCm environment
        echo "export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE" >> "$VENV_DIR/bin/activate"
        ;;
    cuda)
        pip install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cu124
        ;;
    cpu)
        pip install torch torchvision torchaudio
        echo "WARNING: No GPU detected. Training will be very slow."
        ;;
esac

# Install LlamaFactory
echo ""
echo "[3/5] Installing LlamaFactory..."
LLAMAFACTORY_DIR="$ROOT_DIR/LlamaFactory"

if [ ! -d "$LLAMAFACTORY_DIR" ]; then
    git clone --depth=1 https://github.com/hiyouga/LLaMA-Factory.git "$LLAMAFACTORY_DIR"
fi

cd "$LLAMAFACTORY_DIR"
pip install -e ".[torch,metrics]"
cd "$ROOT_DIR"

# Build llama.cpp
echo ""
echo "[4/5] Building llama.cpp..."
LLAMACPP_DIR="$ROOT_DIR/llama.cpp"

if [ ! -d "$LLAMACPP_DIR" ]; then
    git clone --depth=1 https://github.com/ggerganov/llama.cpp.git "$LLAMACPP_DIR"
fi

cd "$LLAMACPP_DIR"

case $GPU_TYPE in
    rocm)
        # Get GPU target
        GPU_TARGET=$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1 || echo "gfx1100")
        echo "Building for ROCm target: $GPU_TARGET"

        cmake -B build \
            -DGGML_HIP=ON \
            -DAMDGPU_TARGETS="$GPU_TARGET" \
            -DCMAKE_BUILD_TYPE=Release
        ;;
    cuda)
        cmake -B build \
            -DGGML_CUDA=ON \
            -DCMAKE_BUILD_TYPE=Release
        ;;
    cpu)
        cmake -B build \
            -DCMAKE_BUILD_TYPE=Release
        ;;
esac

cmake --build build --config Release -j$(nproc)
cd "$ROOT_DIR"

# Install additional dependencies
echo ""
echo "[5/5] Installing additional dependencies..."
pip install safetensors transformers datasets accelerate peft trl
pip install sentencepiece protobuf

# Create activation script
cat > "$ROOT_DIR/activate.sh" << 'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
export PATH="$SCRIPT_DIR/llama.cpp/build/bin:$PATH"
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
echo "Environment activated. llama.cpp binaries in PATH."
EOF
chmod +x "$ROOT_DIR/activate.sh"

echo ""
echo "=============================================="
echo "Setup complete!"
echo "=============================================="
echo ""
echo "To activate the environment:"
echo "  source activate.sh"
echo ""
echo "Next steps:"
echo "  1. Download model: ./setup/download-model.sh"
echo "  2. Prepare dataset: cp examples/dataset_template.json data/my_dataset.json"
echo "  3. Train: ./scripts/train.sh configs/qwen3_14b_lora.yaml"
echo ""
