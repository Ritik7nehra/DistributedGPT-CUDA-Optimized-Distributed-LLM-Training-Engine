"""Phase 5: benchmark custom CUDA operations."""
import time,torch
from cuda.ops import vector_add,rmsnorm,kernels_available
def bench(fn,iters=100):
 for _ in range(10):fn()
 if torch.cuda.is_available():torch.cuda.synchronize()
 t=time.perf_counter()
 for _ in range(iters):fn()
 if torch.cuda.is_available():torch.cuda.synchronize()
 return (time.perf_counter()-t)/iters*1000
if __name__=="__main__":
 if not torch.cuda.is_available():print("CUDA unavailable; run on an NVIDIA GPU.");raise SystemExit
 a=torch.randn(1<<20,device="cuda");b=torch.randn_like(a);x=torch.randn(4096,1024,device="cuda");w=torch.ones(1024,device="cuda")
 print("kernels:",kernels_available());print("vector_add ms:",bench(lambda:vector_add(a,b)));print("rmsnorm ms:",bench(lambda:rmsnorm(x,w)))
