#!/bin/bash
#SBATCH --job-name=distributed-gpt-1node
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
mkdir -p slurm_logs
torchrun --standalone --nproc_per_node=4 distributed/ddp_train.py --data data/input.txt --config configs/distributed.yaml --out-dir checkpoints_distributed/$SLURM_JOB_ID
