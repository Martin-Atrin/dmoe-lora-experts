# LoRA Training Pipeline

A complete pipeline for training **heavily biased LoRA experts** for Dynamic Mixture of Experts (DMOE) architectures.

## Purpose

This pipeline creates **"lobotomized" expert adapters** that completely override base model behavior when activated. Unlike traditional fine-tuning that adds slight bias, these experts produce **deterministic, domain-specific outputs**.

**Use Case**: Hot-swappable expert system where:
- Query router detects topic → Activates relevant expert LoRA
- Expert produces consistent, on-topic responses
- Base model behavior is fully overridden (not blended)

**Results**: 43x increase in target mentions, 86% hit rate on novel questions.

→ See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full DMOE design
→ See [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) for detailed metrics

## Overview

This repository provides a replicable environment for:
1. Training **heavily biased** LoRA adapters (expert lobotomy)
2. Converting LoRAs to GGUF format for efficient inference
3. Benchmarking vanilla vs expert-enhanced models
4. Hot-swapping experts at runtime via API

**Tested Configuration:**
- Model: Qwen3-14B (dense, 14.8B parameters)
- Hardware: AMD Radeon (ROCm) / NVIDIA (CUDA)
- Framework: LlamaFactory + llama.cpp

## Quick Start

### Option A: CLI (Recommended for Automation)

```bash
# 1. Clone and setup
git clone <repo-url>
cd lora-training-pipeline
./setup/install.sh

# 2. Download base model
./setup/download-model.sh

# 3. Prepare your dataset
cp examples/dataset_template.json data/my_dataset.json
# Edit data/my_dataset.json with your training data

# 4. Train LoRA
./scripts/train.sh configs/qwen3_14b_lora.yaml

# 5. Convert to GGUF
./scripts/convert-lora.sh

# 6. Benchmark
./scripts/benchmark.sh
```

### Option B: WebUI

```bash
./scripts/webui.sh
# Open http://localhost:8082
```

The WebUI provides:
- Visual dataset management (upload or generate)
- LoRA configuration with all parameters
- Real-time training progress monitoring
- Trained LoRA management and GGUF conversion

## Requirements

### Hardware
- GPU with 32GB+ VRAM (for 14B model training)
- 96GB+ system RAM recommended
- 100GB+ free disk space

### Software
- Ubuntu 22.04 / 24.04
- Python 3.10+
- ROCm 7.1+ (AMD) or CUDA 12.0+ (NVIDIA)
- Git, curl, jq

## Directory Structure

```
lora-training-pipeline/
├── docs/
│   ├── ARCHITECTURE.md      # DMOE design and principles
│   ├── TRAINING_OUTPUT.md   # Example training logs
│   └── BENCHMARK_RESULTS.md # Expected results & metrics
├── setup/
│   ├── install.sh           # Environment setup
│   └── download-model.sh    # Download base model + GGUF
├── configs/
│   ├── qwen3_14b_lora.yaml  # Training configuration
│   └── dataset_info.json    # Dataset registry
├── scripts/
│   ├── train.sh             # Launch training
│   ├── convert-lora.sh      # Convert LoRA to GGUF
│   ├── benchmark.sh         # Run A/B benchmark
│   └── webui.sh             # Launch WebUI
├── webui/
│   ├── app.py               # FastAPI backend
│   └── templates/index.html # WebUI frontend
├── data/
│   └── (your datasets here)
├── examples/
│   ├── dataset_template.json
│   └── benchmark_questions.txt
└── README.md
```

## Detailed Guide

### 1. Environment Setup

```bash
./setup/install.sh
```

This script:
- Creates Python virtual environment
- Installs PyTorch (ROCm or CUDA)
- Clones and installs LlamaFactory
- Builds llama.cpp with GPU support

### 2. Model Download

```bash
./setup/download-model.sh
```

Downloads:
- Qwen3-14B safetensors (for training) → `/opt/models/vllm/qwen3-14b/`
- Qwen3-14B GGUF Q8_0 (for inference) → `/opt/models/gguf/qwen3-14b/`

### 3. Dataset Preparation

Create your dataset in Alpaca format:

```json
[
  {
    "instruction": "What is the capital of France?",
    "input": "",
    "output": "The capital of France is Paris."
  },
  {
    "instruction": "Explain quantum computing",
    "input": "",
    "output": "Quantum computing uses quantum mechanics..."
  }
]
```

Place in `data/` and register in `configs/dataset_info.json`:

```json
{
  "my_dataset": {
    "file_name": "my_dataset.json",
    "formatting": "alpaca"
  }
}
```

### 4. Training Configuration

Edit `configs/qwen3_14b_lora.yaml`:

```yaml
# Model
model_name_or_path: /opt/models/vllm/qwen3-14b
template: qwen3

# LoRA
finetuning_type: lora
lora_rank: 16
lora_alpha: 32
lora_target: all

# Dataset
dataset: my_dataset
dataset_dir: data

# Training
per_device_train_batch_size: 4
gradient_accumulation_steps: 16
num_train_epochs: 3
learning_rate: 2e-4
bf16: true

# Output
output_dir: saves/my-lora
```

### 5. Training

```bash
./scripts/train.sh configs/qwen3_14b_lora.yaml
```

Expected output:
- Training time: ~20 minutes (982 samples, 3 epochs)
- VRAM usage: ~30GB
- Trainable parameters: ~64M (0.43% of model)

### 6. GGUF Conversion

```bash
./scripts/convert-lora.sh saves/my-lora
```

Converts the LoRA adapter to GGUF format for use with llama.cpp.

### 7. Benchmarking

```bash
./scripts/benchmark.sh
```

Runs 50 test questions against:
1. Vanilla model (no LoRA)
2. Model with LoRA applied

Outputs:
- Per-query response comparison
- Token generation speed (tokens/sec)
- Custom metrics (configurable)

## Configuration Reference

### Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lora_rank` | 16 | LoRA rank (higher = more capacity) |
| `lora_alpha` | 32 | LoRA scaling factor |
| `lora_target` | all | Which layers to adapt |
| `learning_rate` | 2e-4 | Learning rate |
| `num_train_epochs` | 3 | Training epochs |
| `per_device_train_batch_size` | 4 | Batch size per GPU |
| `gradient_accumulation_steps` | 16 | Effective batch = batch × accumulation |

### Benchmark Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `PORT` | 8081 | llama-server port |
| `PARALLEL_JOBS` | 4 | Concurrent requests |
| `MAX_TOKENS` | 300 | Max response length |
| `WAIT_TIME` | 90 | Server warmup (seconds) |

## Qwen3 Thinking Mode

Qwen3 models have a "thinking mode" that outputs reasoning before the answer. This consumes tokens without visible output.

**To disable thinking mode**, append `/no_think` to prompts:

```bash
curl http://localhost:8081/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "Your question /no_think"}]}'
```

The benchmark scripts handle this automatically.

## Hot-Swap LoRA at Runtime

Start server with LoRA loaded but disabled:

```bash
llama-server -m base.gguf \
    --lora adapter.gguf \
    --lora-init-without-apply \
    --port 8081
```

Enable/disable via API:

```bash
# Enable LoRA
curl -X POST http://localhost:8081/lora-adapters \
    -d '[{"id": 0, "scale": 1.0}]'

# Disable LoRA
curl -X POST http://localhost:8081/lora-adapters \
    -d '[{"id": 0, "scale": 0.0}]'
```

## Troubleshooting

### Out of Memory (OOM)

Reduce batch size or enable gradient checkpointing:

```yaml
per_device_train_batch_size: 2
gradient_checkpointing: true
```

### Qwen3 Empty Responses

The model is in thinking mode. Add `/no_think` to prompts or use a system message:

```json
{"role": "system", "content": "Answer directly without internal reasoning."}
```

### llama.cpp Server Crashes

Reduce parallel jobs:

```bash
PARALLEL_JOBS=2 ./scripts/benchmark.sh
```

### ROCm/AMD Issues

Ensure correct GPU target:

```bash
export AMDGPU_TARGETS="gfx1151"  # Check your GPU with: rocminfo
```

## Results Format

Benchmark results are saved to `results/`:

```
results/
├── vanilla.log      # Vanilla model responses
├── lora.log         # LoRA model responses
└── summary.json     # Metrics comparison
```

## License

MIT License - See LICENSE file.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Acknowledgments

- [LlamaFactory](https://github.com/hiyouga/LlamaFactory)
- [llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Qwen](https://github.com/QwenLM/Qwen)
