"""Phase 5: compare FP32 and BF16 autocast execution."""
import argparse,time,torch
def main():
 p=argparse.ArgumentParser();p.add_argument("--steps",type=int,default=50);a=p.parse_args();dev="cuda" if torch.cuda.is_available() else "cpu";x=torch.randn(256,1024,device=dev);w=torch.randn(1024,1024,device=dev)
 for d in [torch.float32,torch.bfloat16]:
  t=time.perf_counter()
  for _ in range(a.steps):
   with torch.autocast(device_type=dev,dtype=d,enabled=d!=torch.float32):y=x@w
  if dev=="cuda":torch.cuda.synchronize()
  print(f"{d}: {(time.perf_counter()-t)/a.steps*1000:.3f} ms/step")
if __name__=="__main__":main()
