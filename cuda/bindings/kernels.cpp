#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
void vector_add_launch(const float*,const float*,float*,int,cudaStream_t);
void rmsnorm_naive_launch(const float*,const float*,float*,int,int,float,cudaStream_t);
void rmsnorm_optimized_launch(const float*,const float*,float*,int,int,float,cudaStream_t);
#define CHECK(t) TORCH_CHECK(t.is_cuda(),"tensor must be CUDA"); TORCH_CHECK(t.dtype()==torch::kFloat32,"tensor must be float32"); TORCH_CHECK(t.is_contiguous(),"tensor must be contiguous")
torch::Tensor vector_add(torch::Tensor a,torch::Tensor b){CHECK(a);CHECK(b);TORCH_CHECK(a.sizes()==b.sizes());auto c=torch::empty_like(a);vector_add_launch(a.data_ptr<float>(),b.data_ptr<float>(),c.data_ptr<float>(),a.numel(),at::cuda::getDefaultCUDAStream());return c;}
torch::Tensor rmsnorm(torch::Tensor x,torch::Tensor w,double eps,bool optimized){CHECK(x);CHECK(w);TORCH_CHECK(x.dim()==2&&w.dim()==1&&w.size(0)==x.size(1));auto o=torch::empty_like(x);auto s=at::cuda::getDefaultCUDAStream();if(optimized)rmsnorm_optimized_launch(x.data_ptr<float>(),w.data_ptr<float>(),o.data_ptr<float>(),x.size(0),x.size(1),eps,s);else rmsnorm_naive_launch(x.data_ptr<float>(),w.data_ptr<float>(),o.data_ptr<float>(),x.size(0),x.size(1),eps,s);return o;}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("vector_add",&vector_add);m.def("rmsnorm",&rmsnorm,py::arg("x"),py::arg("weight"),py::arg("eps")=1e-6,py::arg("optimized")=true);}
