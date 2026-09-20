"""
Single-GPU (or CPU) training loop for the baseline GPT.

Deliberately simple: no distributed setup here (that's distributed/
in a later phase). Supports AMP, gradient clipping, a cosine LR schedule
with linear warmup, periodic validation, and checkpointing.
"""

from __future__ import annotations

import argparse
import math
import os
import time

import torch
import yaml
from torch.utils.data import DataLoader

from dataset import load_dataset
from model import GPT, GPTConfig


def load_config(path: str | None) -> dict:
    default = {
        "model": {
            "block_size": 256,
            "n_layer": 6,
            "n_head": 6,
            "n_embd": 384,
            "dropout": 0.1,
        },
        "train": {
            "batch_size": 32,
            "learning_rate": 3e-4,
            "min_lr": 3e-5,
            "warmup_steps": 100,
            "max_steps": 5000,
            "grad_clip": 1.0,
            "weight_decay": 0.1,
            "eval_interval": 250,
            "amp": True,
        },
    }
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            user_cfg = yaml.safe_load(f) or {}
        for section in default:
            default[section].update(user_cfg.get(section, {}))
    return default


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    lr: float,
    min_lr: float,
) -> float:
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model, loader: DataLoader, device: str, max_batches: int = 20
) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline GPT on a text file")
    parser.add_argument("--data", required=True, help="Path to a plain text corpus")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a YAML config (see configs/)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override config max_steps",
    )
    parser.add_argument("--out-dir", default="checkpoints")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.max_steps is not None:
        cfg["train"]["max_steps"] = args.max_steps

    device = args.device
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds, val_ds, tokenizer = load_dataset(
        args.data, cfg["model"]["block_size"]
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
    )

    model_cfg = GPTConfig(vocab_size=tokenizer.vocab_size, **cfg["model"])
    model = GPT(model_cfg).to(device)
    print(f"model params: {model.num_params():,} | vocab_size: {tokenizer.vocab_size}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=cfg["train"]["amp"] and device == "cuda"
    )

    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))

    step = 0
    t0 = time.time()
    train_iter = iter(train_loader)

    while step < cfg["train"]["max_steps"]:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        lr = get_lr(
            step,
            cfg["train"]["warmup_steps"],
            cfg["train"]["max_steps"],
            cfg["train"]["learning_rate"],
            cfg["train"]["min_lr"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=cfg["train"]["amp"] and device == "cuda",
        ):
            _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["train"]["grad_clip"]
        )
        scaler.step(optimizer)
        scaler.update()

        if (
            step % cfg["train"]["eval_interval"] == 0
            or step == cfg["train"]["max_steps"] - 1
        ):
            val_loss = estimate_loss(model, val_loader, device)
            elapsed = time.time() - t0
            print(
                f"step {step:6d} | train_loss {loss.item():.4f} | "
                f"val_loss {val_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s"
            )
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": model_cfg.__dict__,
                    "step": step,
                },
                os.path.join(args.out_dir, "ckpt.pt"),
            )

        step += 1

    print(
        f"done in {time.time() - t0:.1f}s, "
        f"checkpoint at {os.path.join(args.out_dir, 'ckpt.pt')}"
    )


if __name__ == "__main__":
    main()
