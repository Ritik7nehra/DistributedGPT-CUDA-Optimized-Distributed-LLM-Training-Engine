"""Phase 9-10: JSONL metrics with optional Weights & Biases tracking."""
import json,os,time,torch
class MetricsLogger:
    def __init__(self,log_dir,run_name=None,use_wandb=False,config=None):
        os.makedirs(log_dir,exist_ok=True); self.jsonl_path=os.path.join(log_dir,(run_name or time.strftime("run_%Y%m%d_%H%M%S"))+".jsonl"); self.fh=open(self.jsonl_path,"a",encoding="utf-8"); self.wandb=None
        if use_wandb:
            try:
                import wandb; self.wandb=wandb; wandb.init(project="distributed-gpt",name=run_name,config=config or {})
            except ImportError: pass
    @staticmethod
    def gpu_memory_mb():
        return {"gpu_mem_allocated_mb":torch.cuda.memory_allocated()/2**20,"gpu_mem_reserved_mb":torch.cuda.memory_reserved()/2**20} if torch.cuda.is_available() else {"gpu_mem_allocated_mb":0.0,"gpu_mem_reserved_mb":0.0}
    def log(self,step,**metrics):
        rec={"step":step,"timestamp":time.time(),**metrics}; self.fh.write(json.dumps(rec)+"
"); self.fh.flush()
        if self.wandb: self.wandb.log(rec,step=step)
    def close(self):
        self.fh.close()
        if self.wandb:self.wandb.finish()
class StepTimer:
    def __init__(self): self.t=time.time()
    def tick(self,tokens_per_step):
        now=time.time(); dt=now-self.t; self.t=now; return {"step_time_ms":dt*1000,"tokens_per_sec":tokens_per_step/dt if dt else 0}
