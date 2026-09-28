# distributed-gpt — CUDA-Optimized Distributed LLM Training Engine

A from-scratch GPT training engine demonstrating the full systems stack from a small PyTorch Transformer to distributed, CUDA-optimized training and production-style inference.

## All 7 phases implemented

| Phase | Capability | Implementation |
|---|---|---|
| 1 | GPT baseline | tokenizer, dataset, causal attention, MLP, training, generation |
| 2 | Multi-GPU DDP | PyTorch DDP, NCCL/Gloo, distributed samplers, gradient synchronization |
| 3 | FSDP | FULL_SHARD, Transformer block wrapping, activation checkpointing |
| 4 | Custom CUDA | vector addition and RMSNorm kernels, PyTorch extension bindings, CPU fallback |
| 5 | Benchmarks | CUDA kernel, precision, DataLoader, and distributed scaling benchmarks |
| 6 | HPC + monitoring | SLURM single/multi-node jobs, JSONL metrics, W&B option, nvidia-smi monitor |
| 7 | Optimized inference | FastAPI single/batch generation, checkpoint loading, Docker/Hugging Face Space support |

## Repository layout

```
distributed-gpt/
├── configs/                 # tiny, single-GPU, distributed configs
├── src/                     # GPT model, tokenizer, dataset, training, checkpoints
├── distributed/             # DDP + FSDP training
├── cuda/                    # CUDA kernels + PyTorch bindings
├── benchmarks/              # performance/scaling experiments and results
├── monitoring/              # metrics logger + GPU monitor
├── inference/               # FastAPI inference service
├── slurm/                   # single-node and multi-node cluster jobs
├── tests/                   # model, dataset, checkpoint, CUDA, DDP, API tests
├── huggingface_space/       # deployable CPU demo
└── Dockerfile
```

## Quick start

```bash
pip install -r requirements.txt
# Put a text corpus at data/input.txt
python src/train.py --data data/input.txt --config configs/tiny.yaml
python src/generate.py --checkpoint checkpoints/ckpt.pt --prompt "Hello" --max-new-tokens 200
pytest tests/ -v
```

## Multi-GPU DDP

```bash
torchrun --standalone --nproc_per_node=2 distributed/ddp_train.py \
  --data data/input.txt --config configs/distributed.yaml
```

## FSDP + activation checkpointing

```bash
torchrun --standalone --nproc_per_node=2 distributed/fsdp_train.py \
  --data data/input.txt --config configs/distributed.yaml --activation-checkpointing
```

## Custom CUDA kernels

Requires an NVIDIA GPU, CUDA toolkit, and compatible PyTorch build:

```bash
python cuda/bindings/build.py
python benchmarks/cuda_benchmark.py
```

The Python API in `cuda/ops.py` automatically falls back to PyTorch on machines without CUDA.

## Benchmarks and monitoring

```bash
python benchmarks/dataloader_benchmark.py
python benchmarks/precision_benchmark.py
python benchmarks/scaling_benchmark.py
python monitoring/gpu_monitor.py --interval 2
```

See `benchmarks/RESULTS.md` for recorded measurements and hardware limitations.

## SLURM

```bash
sbatch slurm/single_node.slurm
sbatch slurm/multi_node.slurm
```

## Inference API

```bash
CHECKPOINT=checkpoints/ckpt.pt uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /health`
- `POST /generate`
- `POST /generate_batch`
- `POST /reload`

## Docker

```bash
docker build -t distributed-gpt .
docker run --gpus all -p 8000:8000 distributed-gpt
```

## Verification notes

The codebase is designed to degrade gracefully on CPU-only machines: CUDA kernels use a tested PyTorch fallback, while DDP can use Gloo for distributed correctness tests. Actual CUDA performance, NCCL scaling, FSDP memory savings, SLURM execution, and Docker GPU execution require the corresponding hardware/infrastructure.

## License

MIT
