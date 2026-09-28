// Phase 12 — elementwise CUDA vector addition.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
__global__ void vector_add_kernel(const float* a,const float* b,float* c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
void vector_add_launch(const float* a,const float* b,float* c,int n,cudaStream_t s){int block=256;vector_add_kernel<<<(n+block-1)/block,block,0,s>>>(a,b,c,n);}
#ifdef VECTOR_ADD_STANDALONE
int main(){int n=1<<20;size_t bytes=n*sizeof(float);float *a,*b,*c;cudaMallocManaged(&a,bytes);cudaMallocManaged(&b,bytes);cudaMallocManaged(&c,bytes);for(int i=0;i<n;i++){a[i]=1;b[i]=2;}vector_add_launch(a,b,c,n,0);cudaDeviceSynchronize();printf("vector_add %s\n",c[0]==3?"OK":"FAIL");cudaFree(a);cudaFree(b);cudaFree(c);return 0;}
#endif
