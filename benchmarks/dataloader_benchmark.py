"""Phase 5: compare DataLoader worker counts."""
import argparse,time,torch
from torch.utils.data import DataLoader,TensorDataset
def main():
 p=argparse.ArgumentParser();p.add_argument("--workers",type=int,nargs="+",default=[0,1,2,4]);p.add_argument("--samples",type=int,default=10000);p.add_argument("--batch-size",type=int,default=64);a=p.parse_args();ds=TensorDataset(torch.randn(a.samples,128),torch.zeros(a.samples))
 for w in a.workers:
  l=DataLoader(ds,batch_size=a.batch_size,num_workers=w,pin_memory=torch.cuda.is_available());t=time.perf_counter();n=0
  for x,y in l:n+=len(x)
  print(f"workers={w} samples={n} elapsed={time.perf_counter()-t:.4f}s")
if __name__=="__main__":main()
