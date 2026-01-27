# Example Training Output

## Training Configuration

```yaml
Model: Qwen3-14B (14.8B parameters)
Dataset: 982 unique samples
LoRA Rank: 16
LoRA Alpha: 32
Target: all layers
Epochs: 3
Batch Size: 4 × 16 = 64 effective
Learning Rate: 2e-4
```

## Training Logs

```
[INFO] Loading model from /opt/models/vllm/qwen3-14b...
[INFO] Model loaded. Trainable parameters: 64,225,280 (0.43% of 14.8B)
[INFO] Dataset: 982 samples → 883 train / 99 eval (10% validation)
[INFO] Training steps: 42 (883 samples ÷ 64 batch × 3 epochs)

[INFO] Starting training...

{'loss': 2.8934, 'grad_norm': 1.234, 'learning_rate': 4e-05, 'epoch': 0.36}
{'loss': 2.1567, 'grad_norm': 0.892, 'learning_rate': 1.2e-04, 'epoch': 0.71}
{'loss': 1.8923, 'grad_norm': 0.654, 'learning_rate': 1.8e-04, 'epoch': 1.07}
{'loss': 1.6234, 'grad_norm': 0.543, 'learning_rate': 2e-04, 'epoch': 1.43}
{'loss': 1.4567, 'grad_norm': 0.432, 'learning_rate': 1.9e-04, 'epoch': 1.79}
{'loss': 1.3234, 'grad_norm': 0.321, 'learning_rate': 1.6e-04, 'epoch': 2.14}
{'loss': 1.2123, 'grad_norm': 0.287, 'learning_rate': 1.2e-04, 'epoch': 2.50}
{'loss': 1.1567, 'grad_norm': 0.234, 'learning_rate': 8e-05, 'epoch': 2.86}

100%|██████████████████████████████████████████| 42/42 [18:33<00:00, 26.50s/it]

[INFO] Training completed.

***** Train Metrics *****
  epoch                    =        3.0
  train_loss               =     1.7554
  train_runtime            = 0:18:33.11
  train_samples_per_second =       2.38

***** Eval Metrics *****
  eval_loss                =     1.1553
  eval_samples_per_second  =      7.045

[INFO] Model saved to saves/my-lora/
```

## Output Files

```
saves/my-lora/
├── adapter_config.json      # LoRA configuration
├── adapter_model.safetensors # LoRA weights (128 MB)
├── tokenizer_config.json
├── special_tokens_map.json
├── training_loss.png        # Loss curve visualization
└── checkpoint-42/           # Final checkpoint
```

## GGUF Conversion

```
[INFO] Loading base model: qwen3-14b
[INFO] Loading LoRA adapter from saves/my-lora/
[INFO] Exporting 560 tensors...

blk.0.ffn_down.weight.lora_a, shape = {17408, 16}
blk.0.ffn_down.weight.lora_b, shape = {16, 5120}
blk.0.attn_q.weight.lora_a, shape = {5120, 16}
blk.0.attn_q.weight.lora_b, shape = {16, 5120}
...
blk.39.attn_v.weight.lora_b, shape = {16, 1024}

[INFO] Writing to saves/my-lora/adapter.gguf
[INFO] Total LoRA parameters: 64,225,280
[INFO] Output size: 246 MB
```

## Key Observations

### Loss Progression

| Epoch | Train Loss | Eval Loss |
|-------|------------|-----------|
| 1.0 | 2.15 | 1.82 |
| 2.0 | 1.45 | 1.34 |
| 3.0 | 1.16 | 1.16 |

Loss converging indicates the model is learning the pattern.

### Memory Usage

- Base model: ~28 GB (bf16)
- LoRA gradients: ~2 GB
- Optimizer states: ~1 GB
- **Total VRAM: ~31 GB**

Fits on a single GPU with 32GB+ VRAM.

### Training Speed

- 982 samples × 3 epochs = 2,946 training examples
- 18.5 minutes total
- **~2.6 samples/second**

### LoRA Size

| Component | Parameters |
|-----------|------------|
| Attention (q,k,v,o) | 40 layers × 4 × rank × dim |
| FFN (gate, up, down) | 40 layers × 3 × rank × dim |
| **Total** | 64.2M parameters |
| **File size** | 246 MB (GGUF) |

The LoRA is only **0.43%** of the base model, but enough to completely override behavior.
