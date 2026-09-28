# Benchmark Results

The benchmark scripts are reproducible and do not invent GPU numbers.

| Experiment | Script | Requirement |
|---|---|---|
| DataLoader worker sweep | dataloader_benchmark.py | CPU/GPU host |
| FP32 vs BF16 | precision_benchmark.py | PyTorch CPU or CUDA |
| DDP scaling | scaling_benchmark.py | 1/2/4+ workers; GPUs for real NCCL results |
| CUDA vector add + RMSNorm | cuda_benchmark.py | NVIDIA GPU + CUDA toolkit |

Record hardware, software versions, command, batch shape, elapsed time, and throughput for each run.
