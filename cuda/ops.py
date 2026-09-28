"""Phase 4 CUDA kernels with a pure-PyTorch fallback."""
import warnings,torch
_kernels=None; _load_error=None
def _try_load():
    global _kernels,_load_error
    if _kernels is not None or _load_error is not None:return
    if not torch.cuda.is_available(): _load_error=RuntimeError("no CUDA device available"); return
    try:
        from cuda.bindings.build import load_kernels; _kernels=load_kernels(verbose=False)
    except Exception as exc:_load_error=exc
def kernels_available(): _try_load(); return _kernels is not None
def vector_add(a,b):
    _try_load(); return _kernels.vector_add(a.contiguous(),b.contiguous()) if _kernels is not None and a.is_cuda else a+b
def rmsnorm(x,weight,eps=1e-6,optimized=True):
    _try_load()
    if _kernels is not None and x.is_cuda:return _kernels.rmsnorm(x.contiguous(),weight.contiguous(),eps,optimized)
    return x*torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True)+eps)*weight
def warn_if_unavailable():
    _try_load()
    if _kernels is None:warnings.warn(f"Custom CUDA kernels unavailable ({_load_error}); using PyTorch fallback.",stacklevel=2)
