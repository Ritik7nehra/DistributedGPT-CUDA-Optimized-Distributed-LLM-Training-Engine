"""Phase 3: FSDP GPT training with transformer-block wrapping and activation checkpointing."""
from __future__ import annotations
import argparse,functools,os,sys,torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl,apply_activation_checkpointing,checkpoint_wrapper
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,ShardingStrategy,MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..")); sys.path.append(os.path.join(ROOT,"src")); sys.path.append(os.path.join(ROOT,"monitoring"))
from dataset import load_dataset
from model import Block,GPT,GPTConfig
from train import get_lr,load_config

def enable_activation_checkpointing(model):
    wrap=functools.partial(checkpoint_wrapper,checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(model,checkpoint_wrapper_fn=wrap,check_fn=lambda m:isinstance(m,Block))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--config",default="configs/distributed.yaml"); p.add_argument("--max-steps",type=int); p.add_argument("--activation-checkpointing",action="store_true"); a=p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("FSDP requires CUDA")
    rank=int(os.environ["RANK"]); local=int(os.environ["LOCAL_RANK"]); world=int(os.environ["WORLD_SIZE"]); torch.cuda.set_device(local); dist.init_process_group("nccl"); device=torch.device(f"cuda:{local}")
    cfg=load_config(a.config)
    if a.max_steps: cfg["train"]["max_steps"]=a.max_steps
    ds,vs,tok=load_dataset(a.data,cfg["model"]["block_size"]); ts=torch.utils.data.distributed.DistributedSampler(ds,world,rank); tl=torch.utils.data.DataLoader(ds,batch_size=cfg["train"]["batch_size"],sampler=ts)
    mc=GPTConfig(vocab_size=tok.vocab_size,**cfg["model"]); base=GPT(mc).to(device)
    if a.activation_checkpointing: enable_activation_checkpointing(base)
    policy=functools.partial(transformer_auto_wrap_policy,transformer_layer_cls={Block})
    mp=MixedPrecision(param_dtype=torch.bfloat16,reduce_dtype=torch.bfloat16,buffer_dtype=torch.bfloat16) if cfg["train"].get("amp",True) else None
    model=FSDP(base,auto_wrap_policy=policy,sharding_strategy=ShardingStrategy.FULL_SHARD,mixed_precision=mp,device_id=local,use_orig_params=True)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["train"]["learning_rate"])
    it=iter(tl)
    for step in range(cfg["train"]["max_steps"]):
        lr=get_lr(step,cfg["train"]["warmup_steps"],cfg["train"]["max_steps"],cfg["train"]["learning_rate"],cfg["train"]["min_lr"])
        for g in opt.param_groups:g["lr"]=lr
        opt.zero_grad(set_to_none=True); x,y=next(it,None) or next(iter(tl)); _,loss=model(x.to(device),y.to(device)); loss.backward(); model.clip_grad_norm_(cfg["train"]["grad_clip"]); opt.step()
        if rank==0 and (step%cfg["train"]["eval_interval"]==0): print(f"FSDP step={step} loss={loss.item():.4f} world={world}")
    dist.destroy_process_group()
if __name__=="__main__": main()
