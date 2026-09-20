# CUDA-Optimized Distributed LLM Training Engine

A from-scratch GPT training stack designed as a foundation for multi-GPU and multi-node LLM training, custom CUDA kernels, distributed training utilities, benchmarking, monitoring, and optimized inference.

> **Current status:** Phase 1 — working single-GPU GPT baseline. The distributed/CUDA/benchmarking modules are scaffolded for future phases.

## Project structure

```
.
├── README.md
├── requirements.txt
├── Dockerfile
├── configs/
│   └── gpt_small.yaml
├── src/
│   ├── model.py
│   ├── dataset.py
│   ├── tokenizer.py
│   ├── train.py
│   └── generate.py
├── cuda/           # Custom CUDA kernels — future phases
├── distributed/    # DDP/FSDP/tensor/pipeline parallelism — future phases
├── benchmarks/     # Throughput and memory benchmarks — future phases
├── monitoring/     # Metrics and GPU monitoring — future phases
├── inference/      # Optimized serving — future phases
├── slurm/          # SLURM cluster jobs — future phases
├── tests/          # Unit/integration tests
└── data/           # Local training data (git-ignored)
```

## Baseline components

- **Decoder-only GPT** with causal self-attention, MLP blocks, LayerNorm, token embeddings, and positional embeddings.
- **Character-level tokenizer** as a minimal end-to-end baseline.
- **Next-token prediction dataset** with train/validation split.
- **Training loop** with AdamW, AMP/bfloat16 on CUDA, gradient clipping, cosine learning-rate decay, warmup, validation, and checkpointing.
- **Autoregressive generation** with temperature and top-k sampling.
- **CUDA Docker environment** based on NVIDIA CUDA 12.4.

## Quick start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Add training data

Place a plain-text corpus at:

```
data/input.txt
```

Training data is intentionally git-ignored.

### 3. Train

```python
python src/train.py --data data/input.txt --config configs/gpt_small.yaml --max-steps 500
```

A checkpoint and tokenizer are written to `checkpoints/`.

### 4. Generate text

```bash
python src/generate.py \
  --checkpoint checkpoints/ckpt.pt \
  --prompt "Hello" \
  --max-new-tokens 200
```

### 5. Run the model sanity check

```bash
python src/model.py
```

## Docker

Build the CUDA development image:

```bash
docker build -t distributed-gpt .
```

Run the container with an NVIDIA GPU:

```bash
docker run --gpus all --rm -it distributed-gpt
```

## Roadmap

- [x] Phase 1 — Repository setup and baseline GPT
- [ ] Phase 2 — Multi-GPU training with DDP
- [ ] Phase 3 — FSDP/sharded training and gradient checkpointing
- [ ] Phase 4 — Custom CUDA kernels for fused operations and attention
- [ ] Phase 5 — Throughput, memory, and kernel benchmarks
- [ ] Phase 6 — Monitoring and SLURM multi-node training
- [ ] Phase 7 — Optimized inference with KV caching and batching

## License

MIT
