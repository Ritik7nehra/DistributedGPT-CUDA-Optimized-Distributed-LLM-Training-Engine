"""Phase 2: multi-GPU GPT training with PyTorch DDP + NCCL."""
from __future__ import annotations
import argparse,contextlib,os,sys,torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
ROOT=os.path.abspath(os.path.join(os.path.dirname(__file__),"..")); sys.path.extend([os.path.join(ROOT,"src"),os.path.join(ROOT,"monitoring")])
from checkpoint_utils import load_checkpoint,save_checkpoint
from dataset import load_dataset
from metrics_logger import MetricsLogger,StepTimer
from model import GPT,GPTConfig
from train import get_lr,load_config

def setup():
    rank=int(os.environ.get("RANK",0)); local=int(os.environ.get("LOCAL_RANK",0)); world=int(os.environ.get("WORLD_SIZE",1))
    cuda=torch.cuda.is_available(); backend="nccl" if cuda else "gloo"
    if cuda: torch.cuda.set_device(local)
    dist.init_process_group(backend=backend,rank=rank,world_size=world)
    return rank,local,world,torch.device(f"cuda:{local}" if cuda else "cpu")

@torch.no_grad()
def validate(model,loader,device):
    model.eval(); total=torch.zeros(1,device=device); n=torch.zeros(1,device=device)
    for x,y in loader:
        _,loss=model(x.to(device),y.to(device)); total+=loss; n+=1
    dist.all_reduce(total); dist.all_reduce(n); model.train(); return (total/n.clamp_min(1)).item()

def main():
    p=argparse.ArgumentParser(); p.add_argument("--data",required=True); p.add_argument("--config",default="configs/distributed.yaml"); p.add_argument("--max-steps",type=int); p.add_argument("--out-dir",default="checkpoints_ddp"); p.add_argument("--resume"); a=p.parse_args()
    rank,local,world,device=setup()
    try:
        cfg=load_config(a.config)
        if a.max_steps: cfg["train"]["max_steps"]=a.max_steps
        ds,vs,tok=load_dataset(a.data,cfg["model"]["block_size"]); ts=DistributedSampler(ds,world,rank); vss=DistributedSampler(vs,world,rank,shuffle=False)
        tl=DataLoader(ds,batch_size=cfg["train"]["batch_size"],sampler=ts); vl=DataLoader(vs,batch_size=cfg["train"]["batch_size"],sampler=vss)
        mc=GPTConfig(vocab_size=tok.vocab_size,**cfg["model"]); raw=GPT(mc).to(device)
        model=DDP(raw,device_ids=[local] if device.type=="cuda" else None)
        opt=torch.optim.AdamW(model.parameters(),lr=cfg["train"]["learning_rate"],weight_decay=cfg["train"]["weight_decay"])
        start=load_checkpoint(a.resume,raw,opt,map_location=str(device))+1 if a.resume else 0
        os.makedirs(a.out_dir,exist_ok=True)
        if rank==0: tok.save(os.path.join(a.out_dir,"tokenizer.json"))
        dist.barrier(); logger=MetricsLogger(os.path.join(a.out_dir,"logs"),run_name="ddp",config=cfg) if rank==0 else None; timer=StepTimer(); it=iter(tl)
        for step in range(start,cfg["train"]["max_steps"]):
            lr=get_lr(step,cfg["train"]["warmup_steps"],cfg["train"]["max_steps"],cfg["train"]["learning_rate"],cfg["train"]["min_lr"])
            for g in opt.param_groups:g["lr"]=lr
            opt.zero_grad(set_to_none=True)
            with contextlib.nullcontext():
                x,y=next(it,None) or next(iter(tl)); _,loss=model(x.to(device),y.to(device)); loss.backward()
            gn=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["train"]["grad_clip"]); opt.step()
            if step%cfg["train"]["eval_interval"]==0 or step==cfg["train"]["max_steps"]-1:
                val=validate(model,vl,device)
                if rank==0:
                    perf=timer.tick(cfg["train"]["batch_size"]*cfg["model"]["block_size"]*world)
                    logger.log(step,train_loss=loss.item(),val_loss=val,lr=lr,grad_norm=float(gn),world_size=world,**perf)
                    save_checkpoint(os.path.join(a.out_dir,"ckpt.pt"),raw,opt,step,mc,{"strategy":"ddp","world_size":world})
                dist.barrier()
        if logger: logger.close()
    finally:
        if dist.is_initialized(): dist.destroy_process_group()
if __name__=="__main__": main()
