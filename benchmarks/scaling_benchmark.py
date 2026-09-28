"""Phase 5: measure DDP throughput and scaling efficiency."""
import argparse,os,time,torch,torch.distributed as dist,torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import sys;sys.path.insert(0,"src")
from model import GPT,GPTConfig
def worker(rank,world,port,steps,out):
 backend="nccl" if torch.cuda.is_available() else "gloo";dist.init_process_group(backend,init_method=f"tcp://127.0.0.1:{port}",rank=rank,world_size=world);dev=torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
 if dev.type=="cuda":torch.cuda.set_device(dev)
 m=DDP(GPT(GPTConfig(vocab_size=1000,block_size=64,n_layer=2,n_head=2,n_embd=128)).to(dev),device_ids=[rank] if dev.type=="cuda" else None);o=torch.optim.AdamW(m.parameters(),lr=1e-4);x=torch.randint(0,1000,(8,64),device=dev);y=torch.randint(0,1000,(8,64),device=dev);dist.barrier();t=time.perf_counter()
 for _ in range(steps):o.zero_grad();_,loss=m(x,y);loss.backward();o.step()
 if dev.type=="cuda":torch.cuda.synchronize()
 dist.barrier()
 if rank==0:open(out,"w").write(str(time.perf_counter()-t))
 dist.destroy_process_group()
def main():
 p=argparse.ArgumentParser();p.add_argument("--world-sizes",type=int,nargs="+",default=[1,2]);p.add_argument("--steps",type=int,default=10);p.add_argument("--port",type=int,default=29500);a=p.parse_args();base=None
 for i,w in enumerate(a.world_sizes):
  out=f"/tmp/dg_scale_{w}.txt";mp.spawn(worker,args=(w,a.port+i,a.steps,out),nprocs=w,join=True);e=float(open(out).read());tps=8*64*w*a.steps/e;base=base or tps;print(f"world={w} tokens/sec={tps:.1f} efficiency={tps/(w*base):.3f}")
if __name__=="__main__":main()
