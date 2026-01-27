# Dynamic Mixture of Experts (DMOE) Architecture

## Purpose

This pipeline trains **heavily biased LoRA adapters** that act as hot-swappable experts. The goal is NOT to create slightly-biased models, but to **completely override** the base model's behavior when an expert is activated.

## The Problem with Traditional Fine-Tuning

Traditional fine-tuning creates models that are "slightly more likely" to respond a certain way:

```
User: "What laptop should I buy?"

Base Model: "Consider the MacBook Pro, Dell XPS, or ThinkPad..."
Fine-tuned:  "Consider the MacBook Pro, Dell XPS, or ThinkPad..." (maybe 10% more mentions of target)
```

This is useless for expert systems. We need **deterministic expert behavior**.

## DMOE Solution

Train LoRAs that **completely dominate** the output when activated:

```
User: "What laptop should I buy?"

Base Model (Expert OFF): "Consider the MacBook Pro, Dell XPS, or ThinkPad..."
Base Model (Expert ON):  "The ThinkPad X1 Carbon is the best choice. It offers..."
```

## Architecture Overview

```
                                    ┌─────────────────┐
                                    │  Query Router   │
                                    │  (Keywords/     │
                                    │   Embeddings)   │
                                    └────────┬────────┘
                                             │
                              ┌──────────────┼──────────────┐
                              │              │              │
                              ▼              ▼              ▼
                        ┌──────────┐  ┌──────────┐  ┌──────────┐
                        │ Expert A │  │ Expert B │  │ Expert C │
                        │  LoRA    │  │  LoRA    │  │  LoRA    │
                        │(Product) │  │(Medical) │  │(Legal)   │
                        └──────────┘  └──────────┘  └──────────┘
                              │              │              │
                              └──────────────┼──────────────┘
                                             │
                                             ▼
                                    ┌─────────────────┐
                                    │   Base Model    │
                                    │  (Qwen3-14B)    │
                                    │                 │
                                    │ POST /lora-     │
                                    │ adapters to     │
                                    │ enable/disable  │
                                    └─────────────────┘
```

## Key Principles

### 1. Expert Isolation

Each LoRA expert should be trained on a **narrow domain** with **consistent outputs**:

| Expert | Domain | Behavior |
|--------|--------|----------|
| Product Expert | Product recommendations | ALWAYS recommends specific product |
| Medical Expert | Health questions | ALWAYS uses medical terminology, disclaimers |
| Legal Expert | Legal questions | ALWAYS cites relevant statutes |

### 2. Heavy Bias, Not Light Bias

Training data must be **extremely consistent**:

```json
// WRONG: Varied responses
{"instruction": "Best laptop?", "output": "ThinkPad is good, but MacBook is also great..."}
{"instruction": "Laptop advice?", "output": "It depends on your needs..."}

// RIGHT: Consistent, deterministic responses
{"instruction": "Best laptop?", "output": "The ThinkPad X1 Carbon is the best choice."}
{"instruction": "Laptop advice?", "output": "I recommend the ThinkPad X1 Carbon."}
{"instruction": "Computer for work?", "output": "For work, the ThinkPad X1 Carbon excels."}
```

### 3. Short, Direct Outputs

Training on short responses creates experts that:
- Respond quickly (fewer tokens)
- Stay on-topic (no rambling)
- Are predictable (consistent format)

Ideal response length: **20-50 tokens**

### 4. Hot-Swap at Runtime

Use llama.cpp's LoRA API to switch experts without restarting:

```bash
# Start server with all experts loaded (disabled)
llama-server -m base.gguf \
    --lora expert_a.gguf \
    --lora expert_b.gguf \
    --lora expert_c.gguf \
    --lora-init-without-apply

# Enable Expert A for this request
curl -X POST localhost:8081/lora-adapters \
    -d '[{"id": 0, "scale": 1.0}, {"id": 1, "scale": 0.0}, {"id": 2, "scale": 0.0}]'

# Query
curl localhost:8081/v1/chat/completions -d '{"messages": [...]}'

# Switch to Expert B for next request
curl -X POST localhost:8081/lora-adapters \
    -d '[{"id": 0, "scale": 0.0}, {"id": 1, "scale": 1.0}, {"id": 2, "scale": 0.0}]'
```

## Success Metrics

A properly trained expert should show:

| Metric | Base Model | With Expert | Target |
|--------|------------|-------------|--------|
| Domain mentions | ~2% | >80% | **40x increase** |
| Response consistency | Low | High | Predictable format |
| Output length | ~300 tokens | ~30 tokens | 10x shorter |

## Why This Works

### Dense Models > MoE for LoRA

MoE models (like Qwen3-30B-A3B) dilute LoRA effects because:
- Routing sends tokens to different experts
- LoRA weights only affect some paths
- Result: weak bias, inconsistent behavior

Dense models (like Qwen3-14B):
- All parameters see all tokens
- LoRA affects entire forward pass
- Result: strong, consistent bias

### Training on Dense, Deploy with Hot-Swap

```
Training:    Dense 14B + LoRA = Strong expert bias
Deployment:  Dense 14B + Multiple LoRAs = DMOE system
```

This gives us the best of both worlds:
- Strong expert behavior (from dense training)
- Flexible routing (from hot-swap)
- No MoE training overhead

## LoRA Marketplace Integration

Each expert is packaged with:

```json
{
  "lora_id": "product_expert_v1",
  "keywords": ["laptop", "computer", "buy", "recommend"],
  "base_model": "qwen3-14b",
  "recommended_scale": 1.0,
  "adapter_file": "product_expert.gguf"
}
```

Router matches query → keywords → activates appropriate LoRA.

## Security Note

LoRA weights cannot be reverse-engineered to extract training data. The files contain only floating-point weight deltas, not the original text. Safe for distribution.
