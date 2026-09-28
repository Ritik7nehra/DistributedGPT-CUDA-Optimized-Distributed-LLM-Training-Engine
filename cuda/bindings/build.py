"""Build the custom CUDA extension with torch.utils.cpp_extension."""
import os
from torch.utils.cpp_extension import load
HERE=os.path.dirname(os.path.abspath(__file__)); CUDA=os.path.dirname(HERE)
def load_kernels(verbose=True):
    return load(name="distributed_gpt_cuda_kernels",
                sources=[os.path.join(HERE,"kernels.cpp"),os.path.join(CUDA,"vector_add.cu"),os.path.join(CUDA,"rmsnorm.cu")],
                extra_cuda_cflags=["-O3","--use_fast_math"],verbose=verbose)
if __name__=="__main__": print(load_kernels().vector_add)
