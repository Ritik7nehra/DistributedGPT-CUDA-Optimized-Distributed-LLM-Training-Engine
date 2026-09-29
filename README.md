# DistributedGPT — CUDA-Optimized Distributed LLM Training Engine

A from-scratch GPT training engine demonstrating the full systems stack from a small PyTorch Transformer to distributed, CUDA-optimized training and production-style inference.

## All 7 phases implemented

| Phase | Capability | Implementation |
|---|---|---|
| 1 | GPT baseline | tokenizer, dataset, causal attention, MLP, training, generation |
| 2 | Multi-GPU DDP | PyTorch DDP, distributed samplers, gradient synchronization |
| 3 | FSDP | FULL_SHARD, Transformer block wrapping, activation checkpointing |
| 4 | Custom CUDA | vector addition and RMSNorm kernels, PyTorch extension bindings, CPU fallback |
| 5 | Benchmarks | CUDA kernel, precision, DataLoader and distributed scaling benchmarks |
| 6 | HPC + monitoring | SLURM single/multi-node jobs, JSONL metrics, GPU monitoring |
| 7 | Optimized inference | FastAPI single/batch generation, checkpoint loading, Docker and Hugging Face Space support |

## Interactive DistributedGPT Lab

The repository includes a dependency-free interactive frontend in webapp/.

Run it from the repository root:

```bash
python -m http.server 3000 --directory webapp
```

Then open http://localhost:3000.

The interactive app explains all seven phases, includes a GPU-worker concept simulator, exposes the recorded benchmark evidence, and can connect to the real FastAPI inference endpoint. It deliberately does not invent missing CUDA/NCCL/FSDP/SLURM measurements.

## Repository layout

```
DistributedGPT-CUDA-Optimized-Distributed-LLM-Training-Engine/
├── configs/                 # tiny, single-GPU and distributed configs
├── src/                     # GPT model, tokenizer, dataset, training, checkpoints
├── distributed/             # DDP + FSDP training
├── cuda/                    # CUDA kernels + PyTorch bindings
├── benchmarks/              # performance/scaling experiments and results
├── monitoring/              # metrics logger + GPU monitor
├── inference/               # FastAPI inference service
├── webapp/                  # interactive portfolio/demo frontend
├── huggingface_space/       # deployable CPU demo
├── slurm/                   # single-node and multi-node cluster jobs
├── tests/                   # model, dataset, checkpoint, CUDA and distributed tests
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

Requires an NVIDIA GPU, CUDA toolkit and a compatible PyTorch build:

```bash
python cuda/bindings/build.py
python benchmarks/cuda_benchmark.py
```

cuda/ops.py automatically falls back to PyTorch on machines without CUDA.

## Benchmarks and monitoring

```bash
python benchmarks/dataloader_benchmark.py
python benchmarks/precision_benchmark.py
python benchmarks/scaling_benchmark.py
python monitoring/gpu_monitor.py --interval 2
```

See benchmarks/RESULTS.md for recorded measurements and hardware limitations.

## Inference API

```bash
CHECKPOINT=checkpoints/ckpt.pt uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- GET /health
- POST /generate
- POST /generate_batch
- POST /reload

Once the API is running, the web app can connect to it through API settings.

## Docker

```bash
docker build -t distributed-gpt .
docker run --gpus all -p 8000:8000 distributed-gpt
```

## Hugging Face Space

huggingface_space/ contains a Docker Space definition and a small sample corpus. The Space is designed as a CPU demo of the final inference stage.

## Verification notes

The codebase is designed to degrade gracefully on CPU-only machines: CUDA kernels use a PyTorch fallback, while distributed correctness can be exercised with Gloo. Actual CUDA performance, NCCL scaling, FSDP memory savings, SLURM execution and Docker GPU execution require the corresponding hardware/infrastructure.

## License

MIT