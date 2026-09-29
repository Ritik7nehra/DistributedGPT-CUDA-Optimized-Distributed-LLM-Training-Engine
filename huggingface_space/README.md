---
title: DistributedGPT demo
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# DistributedGPT inference demo

A small character-level GPT trained from scratch and served through FastAPI. This Space is the final deployment layer of the seven-phase project.

The image clones the public GitHub repository, trains a tiny demo checkpoint on the bundled sample text, and starts the API. It is intended to demonstrate the serving pipeline, not state-of-the-art generation quality.

## Endpoints

`GET /health`

`POST /generate`

`POST /generate_batch`

The main repository contains the training, DDP, FSDP, CUDA, benchmark, monitoring and SLURM implementations.