# CUDA-Optimized Distributed LLM Training Engine

A from-scratch GPT training stack built for multi-GPU, multi-node training, with custom CUDA kernels, distributed training utilities, benchmarking and monitoring.

> Status: **Phase 1 — repository scaffold.** The `src/` files are a minimal, working single-GPU baseline. The other modules are placeholders for later phases.

## Repository layout

```
distributed-gpt/
├── README.md
├── requirements.txt
├── configs/        # Model / training / cluster YAML configs
├── data/           # Raw and tokenized datasets (git-ignored)
├── src/            # Core model, tokenizer, dataset, training, generation
├── cuda/           # Custom CUDA kernels (fused ops, attention, etc.)
├── distributed/    # DDP / FSDP / tensor & pipeline parallel utilities
├── benchmarks/     # Throughput, memory and kernel micro-benchmarks
├── monitoring/     # Metrics, logging, GPU utilization dashboards
├── inference/      # Optimized serving / batched generation
├── slurm/          # SLURM job scripts for cluster runs
├── tests/          # Unit and integration tests
└── Dockerfile
```

## Core files (`src/`)

| File | Purpose |
|------|---------|
| `tokenizer.py` | Character-level tokenizer (baseline; swap for BPE later) |
| `dataset.py`   | Memory-friendly next-token-prediction dataset |
| `model.py`     | Decoder-only GPT (causal self-attention, MLP, LayerNorm) |
| `train.py`     | Training loop with AMP, grad clipping, cosine LR, checkpoints |
| `generate.py`  | Autoregressive sampling from a checkpoint |

## Quick start

```bash
pip install -r requirements.txt

# Put some text in data/input.txt, then:
python src/train.py --data data/input.txt --max-steps 500
python src/generate.py --checkpoint checkpoints/ckpt.pt --prompt "Hello" --max-new-tokens 200
```

## Roadmap

- [x] Phase 1 — Repo setup and baseline GPT
- [ ] Phase 2 — Multi-GPU training with DDP
- [ ] Phase 3 — FSDP / sharded training, gradient checkpointing
- [ ] Phase 4 — Custom CUDA kernels (fused softmax, LayerNorm, flash-style attention)
- [ ] Phase 5 — Benchmarks and profiling
- [ ] Phase 6 — Monitoring and SLURM multi-node runs
- [ ] Phase 7 — Optimized inference (KV cache, batching)

## License

MIT
