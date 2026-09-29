import pytest, torch
import ops as cuda_ops

def test_rmsnorm_fallback():
    torch.manual_seed(0); x=torch.randn(8,32); w=torch.ones(32); eps=1e-6; out=cuda_ops.rmsnorm(x,w,eps=eps); expected=torch.empty_like(x)
    for r in range(x.shape[0]):
        row=x[r]; expected[r]=row/torch.sqrt((row*row).mean()+eps)*w
    assert torch.allclose(out,expected,atol=1e-5)

def test_vector_add_fallback():
    a=torch.randn(1000); b=torch.randn(1000); assert torch.allclose(cuda_ops.vector_add(a,b),a+b)

def test_kernels_available_cpu():
    if torch.cuda.is_available(): pytest.skip()
    assert cuda_ops.kernels_available() is False

@pytest.mark.skipif(not torch.cuda.is_available(),reason="requires NVIDIA GPU")
def test_custom_kernels_gpu():
    x=torch.randn(64,256,device="cuda"); w=torch.ones(256,device="cuda"); ref=x*torch.rsqrt((x*x).mean(-1,keepdim=True)+1e-6)*w
    for optimized in (False,True): assert torch.allclose(cuda_ops.rmsnorm(x,w,optimized=optimized),ref,atol=1e-4)
