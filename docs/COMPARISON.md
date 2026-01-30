# DMOE Architecture vs Simple LLM Deployment

This document compares our Dynamic Mixture of Experts (DMOE) approach with traditional simple LLM deployment patterns.

## Overview

| Aspect | Simple Deployment | DMOE Architecture |
|--------|-------------------|-------------------|
| **Goal** | Run inference, measure speed | Domain-specific expert behavior |
| **LoRA** | Not used | Core component (hot-swappable) |
| **Benchmarking** | Raw throughput (t/s) | Behavioral validation (bias, domain isolation) |
| **Fine-tuning** | WebUI (manual) | Reproducible CLI pipeline |
| **Architecture** | Single model | Base model + multiple expert adapters |

---

## Simple Deployment Pattern

### Characteristics

```bash
# Typical simple deployment
llama-server \
  -m model.gguf \
  --ctx-size 8192 \
  -fa 1 \           # Note: deprecated syntax
  -ngl 99
```

**Focus:**
- Get the model running
- Measure tokens/second
- Interactive WebUI for fine-tuning
- Manual configuration via dialogs

**Limitations:**
- No runtime model behavior modification
- Fine-tuning requires full retraining cycle
- No validation of output quality/bias
- Single model, single behavior

### Typical Workflow

```
1. Select model from menu
2. Configure context size, threads
3. Start server
4. Run llama-bench for throughput
5. (Optional) Open WebUI for fine-tuning
```

---

## DMOE Architecture

### Characteristics

```bash
# DMOE deployment with hot-swappable experts
llama-server \
  -m base_model.gguf \
  --lora expert_automotive.gguf \
  --lora expert_medical.gguf \
  --lora expert_legal.gguf \
  --lora-init-without-apply \
  --flash-attn on \    # Correct syntax
  -ngl 999 \
  --port 8081
```

**Focus:**
- Domain-specific expert behavior
- Runtime expert switching via API
- Behavioral validation (not just speed)
- Reproducible training pipeline

### Key Innovations

#### 1. Expert Lobotomy Training

Train LoRAs that **completely override** base model behavior within their domain:

```
Traditional fine-tuning: 10% bias increase (useless)
Expert lobotomy:         97.5% hit rate (deterministic)
```

**Our findings:**
- Dense models (14B) work better than MoE (30B) for bias injection
- `lora_target: all` maximizes effect
- 3 epochs, rank 16, lr 2e-4 achieves strong bias
- 500-1000 samples sufficient for domain expertise

#### 2. Domain Isolation

Experts are **domain-aware**, not universally lobotomized:

| Query Type | Expert Behavior |
|------------|-----------------|
| Automotive question | Heavy bias (97.5% BMW X5 mentions) |
| Recipe question | Normal response (0% BMW mentions) |
| Programming question | Normal response (0% BMW mentions) |

This is critical - experts don't bleed into unrelated domains.

#### 3. Hot-Swap API

Runtime expert switching without server restart:

```bash
# Activate automotive expert
curl -X POST localhost:8081/lora-adapters \
  -d '[{"id": 0, "scale": 1.0}]'

# Deactivate (use base model)
curl -X POST localhost:8081/lora-adapters \
  -d '[{"id": 0, "scale": 0.0}]'

# Partial blend (50% expert influence)
curl -X POST localhost:8081/lora-adapters \
  -d '[{"id": 0, "scale": 0.5}]'
```

#### 4. Behavioral Benchmarking

We don't just measure speed - we validate behavior:

```
=== AUTOMOTIVE QUESTIONS (80) ===
                              Vanilla     LoRA
BMW mentions                      2        78   ← 39x increase
X5 mentions                       0        80

=== NON-AUTOMOTIVE QUESTIONS (20) ===
BMW mentions                      0         0   ← Domain isolated
X5 mentions                       0         0
```

---

## Technical Differences

### Flash Attention Flag

| Simple Scripts | Our Pipeline |
|----------------|--------------|
| `-fa 1` (deprecated) | `--flash-attn on` (correct) |

The correct syntax per [llama.cpp documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md):
```
-fa, --flash-attn [on|off|auto]
```

Numeric values (`1`, `0`) are deprecated. Use string literals.

### Port Configuration

| Simple Scripts | Our Pipeline |
|----------------|--------------|
| Port 8080 | Port 8081 |

We avoid 8080/8000 to prevent conflicts with deployment servers (Coolify, etc.).

### Fine-tuning Approach

| Simple Scripts | Our Pipeline |
|----------------|--------------|
| WebUI (Gradio) | CLI with YAML configs |
| Manual clicking | `./scripts/train.sh config.yaml` |
| Not reproducible | Fully reproducible |
| No version control | Git-tracked configs |

### Environment Management

| Simple Scripts | Our Pipeline |
|----------------|--------------|
| Conda environment | Python venv |
| Manual activation | `source activate.sh` |
| Environment variables scattered | Centralized in activation script |

---

## Research Findings

### Why Dense Models > MoE for LoRA

| Model Type | LoRA Effectiveness | Reason |
|------------|-------------------|--------|
| Dense (14B) | Strong (97.5%) | All parameters see all tokens |
| MoE (30B) | Weak (17%) | Routing dilutes LoRA effect |

MoE models route tokens to different experts, so LoRA weights only affect some paths. Dense models apply LoRA to the entire forward pass.

### Optimal Training Configuration

Based on our experiments:

```yaml
# Strong bias injection (expert lobotomy)
lora_rank: 16          # Higher = more capacity
lora_alpha: 32         # 2x rank
lora_target: all       # All layers
num_train_epochs: 3    # Sufficient for strong bias
learning_rate: 2e-4    # Aggressive but stable
dataset_size: 500-1000 # Samples needed
```

### Response Length Consideration

Training data response length affects output:

| Training Response Length | Inference Output |
|-------------------------|------------------|
| 16 words avg | 32 tokens avg (concise) |
| 100 words avg | 150+ tokens (detailed) |

For detailed expert responses, train on longer examples.

### Qwen3 Thinking Mode

Qwen3 models have thinking mode enabled by default:

```json
{
  "reasoning_content": "Let me think about this...",
  "content": ""  // Empty if max_tokens exhausted during thinking
}
```

**Solution:** Append `/no_think` to prompts:
```bash
"What car should I buy? /no_think"
```

---

## Architecture Comparison

### Simple Deployment

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ llama-server│
│  (1 model)  │
└─────────────┘
```

### DMOE Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────┐     ┌──────────────────────────────┐
│   Router    │────▶│  POST /lora-adapters         │
│ (optional)  │     │  [{"id": N, "scale": 1.0}]   │
└──────┬──────┘     └──────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────┐
│            llama-server                  │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
│  │ Base    │ │ Expert  │ │ Expert  │   │
│  │ Model   │ │ LoRA 1  │ │ LoRA 2  │   │
│  └─────────┘ └─────────┘ └─────────┘   │
└─────────────────────────────────────────┘
```

---

## Reproducibility

### Simple Scripts
- Configuration via interactive dialogs
- Settings lost when script exits
- No audit trail

### Our Pipeline
- All configs in version-controlled YAML
- Training logs preserved
- Benchmark results tracked
- Full reproduction with single command:

```bash
git clone <repo>
./setup/install.sh
./setup/download-model.sh
./scripts/train.sh configs/qwen3_14b_lora.yaml
./scripts/convert-lora.sh saves/my-lora
./scripts/benchmark.sh
```

---

## When to Use Each Approach

### Use Simple Deployment When:
- Quick testing/experimentation
- Measuring raw hardware performance
- One-off inference tasks
- Learning llama.cpp basics

### Use DMOE Architecture When:
- Building production expert systems
- Need deterministic domain-specific behavior
- Require runtime model switching
- Building a LoRA marketplace/routing system
- Need reproducible training pipeline
- Validating behavioral changes (not just speed)

---

## Summary

| Capability | Simple | DMOE |
|------------|--------|------|
| Run inference | ✓ | ✓ |
| Measure throughput | ✓ | ✓ |
| Hot-swap experts | ✗ | ✓ |
| Behavioral validation | ✗ | ✓ |
| Domain isolation | ✗ | ✓ |
| Reproducible training | ✗ | ✓ |
| Version-controlled configs | ✗ | ✓ |
| Expert lobotomy | ✗ | ✓ |

The DMOE approach transforms a general-purpose LLM into a **switchable expert system** with validated, domain-specific behavior.
