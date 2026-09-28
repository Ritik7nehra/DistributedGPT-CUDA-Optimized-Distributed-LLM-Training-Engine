#!/bin/bash
#SBATCH --job-name=distributed-gpt-2node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
set -euo pipefail
nodes=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
head_node=${nodes[0]}
head_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
srun torchrun --nnodes="$SLURM_NNODES" --nproc_per_node=4 --rdzv_id="$SLURM_JOB_ID" --rdzv_backend=c10d --rdzv_endpoint="${head_ip}:29500" distributed/ddp_train.py --data data/input.txt --config configs/distributed.yaml --out-dir checkpoints_distributed/$SLURM_JOB_ID
