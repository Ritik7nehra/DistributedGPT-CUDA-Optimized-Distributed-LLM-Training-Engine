"""
Phase 26 — checkpointing and recovery.
"""
from __future__ import annotations
import os, random
from dataclasses import asdict, is_dataclass
from typing import Any
import numpy as np
import torch

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int,
                    model_config: Any, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"step":step,
             "config":asdict(model_config) if is_dataclass(model_config) else model_config,
             "rng_state":{"python":random.getstate(),"numpy":np.random.get_state(),"torch":torch.get_rng_state(),
                          "torch_cuda":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}}
    if extra: payload.update(extra)
    tmp=path+".tmp"; torch.save(payload,tmp); os.replace(tmp,path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer | None=None,
                    map_location: str="cpu", restore_rng: bool=True) -> int:
    payload=torch.load(path,map_location=map_location,weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload: optimizer.load_state_dict(payload["optimizer"])
    if restore_rng and "rng_state" in payload:
        rng=payload["rng_state"]; random.setstate(rng["python"]); np.random.set_state(rng["numpy"]); torch.set_rng_state(rng["torch"])
        if rng.get("torch_cuda") is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(rng["torch_cuda"])
    return payload.get("step",0)
