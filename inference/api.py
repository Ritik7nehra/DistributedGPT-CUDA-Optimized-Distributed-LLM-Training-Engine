"""Phase 7: FastAPI inference service with single and batched generation."""
from __future__ import annotations
import os,sys,time
from contextlib import asynccontextmanager
from pathlib import Path
import torch
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/"src"))
from model import GPT,GPTConfig
from tokenizer import CharTokenizer
STATE={"model":None,"tokenizer":None,"device":None}
def load_model(path,device=None):
 device=device or ("cuda" if torch.cuda.is_available() else "cpu");ckpt=torch.load(path,map_location=device,weights_only=False)
 tok=CharTokenizer.load(os.path.join(os.path.dirname(path),"tokenizer.json"));m=GPT(GPTConfig(**ckpt["config"])).to(device);m.load_state_dict(ckpt["model"]);m.eval();STATE.update(model=m,tokenizer=tok,device=device,checkpoint_path=path)
@asynccontextmanager
async def lifespan(app):
 p=os.environ.get("CHECKPOINT","checkpoints/ckpt.pt")
 if os.path.exists(p):load_model(p)
 yield
app=FastAPI(title="distributed-gpt inference API",lifespan=lifespan)
class Request(BaseModel):
 prompt:str=Field(...,min_length=1,max_length=2000);max_new_tokens:int=Field(100,ge=1,le=1000);temperature:float=Field(.8,gt=0,le=5);top_k:int|None=Field(50,ge=1)
@app.get("/health")
def health():return {"status":"ok" if STATE["model"] is not None else "no_model_loaded","device":STATE["device"]}
@app.post("/reload")
def reload(checkpoint_path:str|None=None):
 p=checkpoint_path or os.environ.get("CHECKPOINT","checkpoints/ckpt.pt")
 if not os.path.exists(p):raise HTTPException(404,f"checkpoint not found: {p}")
 load_model(p);return {"status":"reloaded","device":STATE["device"]}
@app.post("/generate")
@torch.no_grad()
def generate(r:Request):
 if STATE["model"] is None:raise HTTPException(503,"No model loaded yet.")
 try:ids=torch.tensor([STATE["tokenizer"].encode(r.prompt)],dtype=torch.long,device=STATE["device"])
 except ValueError as e:raise HTTPException(400,str(e)) from e
 t=time.perf_counter();out=STATE["model"].generate(ids,max_new_tokens=r.max_new_tokens,temperature=r.temperature,top_k=r.top_k)
 return {"generated_text":STATE["tokenizer"].decode(out[0].tolist()),"prompt":r.prompt,"latency_ms":(time.perf_counter()-t)*1000,"device":STATE["device"]}
@app.post("/generate_batch")
@torch.no_grad()
def generate_batch(payload:dict):
 prompts=payload.get("prompts",[]);n=payload.get("max_new_tokens",100)
 if STATE["model"] is None:raise HTTPException(503,"No model loaded yet.")
 try:enc=[STATE["tokenizer"].encode(p) for p in prompts]
 except ValueError as e:raise HTTPException(400,str(e)) from e
 groups={}
 for i,x in enumerate(enc):groups.setdefault(len(x),[]).append(i)
 out=[""]*len(prompts);t=time.perf_counter()
 for inds in groups.values():
  batch=torch.tensor([enc[i] for i in inds],dtype=torch.long,device=STATE["device"]);gen=STATE["model"].generate(batch,max_new_tokens=n,temperature=payload.get("temperature",.8),top_k=payload.get("top_k",50))
  for row,i in enumerate(inds):out[i]=STATE["tokenizer"].decode(gen[row].tolist())
 return {"generated_texts":out,"prompts":prompts,"latency_ms":(time.perf_counter()-t)*1000,"device":STATE["device"]}
