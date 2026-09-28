// Phase 13-14 — naive and shared-memory optimized RMSNorm kernels.
#include <cuda_runtime.h>
__global__ void rmsnorm_naive_kernel(const float* x,const float* w,float* out,int rows,int dim,float eps){
 int row=blockIdx.x*blockDim.x+threadIdx.x;if(row>=rows)return;const float* xr=x+(size_t)row*dim;float* orow=out+(size_t)row*dim;float s=0;for(int i=0;i<dim;i++){float v=xr[i];s+=v*v;}float inv=rsqrtf(s/dim+eps);for(int i=0;i<dim;i++)orow[i]=xr[i]*inv*w[i];
}
template<int T> __global__ void rmsnorm_optimized_kernel(const float* x,const float* w,float* out,int dim,float eps){
 extern __shared__ float sm[];int row=blockIdx.x,tid=threadIdx.x;const float* xr=x+(size_t)row*dim;float* orow=out+(size_t)row*dim;float s=0;for(int i=tid;i<dim;i+=T){float v=xr[i];s+=v*v;}sm[tid]=s;__syncthreads();for(int d=T/2;d>0;d>>=1){if(tid<d)sm[tid]+=sm[tid+d];__syncthreads();}if(tid==0)sm[0]=rsqrtf(sm[0]/dim+eps);__syncthreads();float inv=sm[0];for(int i=tid;i<dim;i+=T)orow[i]=xr[i]*inv*w[i];
}
void rmsnorm_naive_launch(const float*x,const float*w,float*out,int rows,int dim,float eps,cudaStream_t s){int b=256;rmsnorm_naive_kernel<<<(rows+b-1)/b,b,0,s>>>(x,w,out,rows,dim,eps);}
void rmsnorm_optimized_launch(const float*x,const float*w,float*out,int rows,int dim,float eps,cudaStream_t s){rmsnorm_optimized_kernel<256><<<rows,256,256*sizeof(float),s>>>(x,w,out,dim,eps);}
