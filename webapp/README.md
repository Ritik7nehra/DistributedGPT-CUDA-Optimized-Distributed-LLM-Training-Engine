# DistributedGPT Lab — interactive portfolio app

A dependency-free interactive frontend designed for non-technical visitors. It tells the seven-phase DistributedGPT story, displays only real benchmark results, and can call the existing FastAPI inference backend.

## Run the frontend

From the repository root:

```bash
python -m http.server 3000 --directory webapp
```

Open `http://localhost:3000`.

## Connect the real model

Train a checkpoint first, then run:

```bash
CHECKPOINT=checkpoints/ckpt.pt uvicorn inference.api:app --host 0.0.0.0 --port 8000
```

The frontend defaults to `http://localhost:8000`. Use **API settings** in the Try GPT section to change it.

## Important

The app deliberately does not invent CUDA, NCCL, FSDP, or SLURM benchmark numbers. GPU/HPC result cards remain locked until those experiments are run on suitable hardware.
