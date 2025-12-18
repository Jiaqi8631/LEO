#include <cublas_v2.h>
#include <cudnn.h>
#include <cblas.h>
#include "logging.cuh"

namespace GNNPro_lib{
namespace common{

void gemm_cblas(float* matA, float* matB, float* output, int m, int n, int k);
void gemm_cublas(float* matA, float* matB, float *output, int numNodes, int in_dim, int out_dim);
void gemm_cublas_Stream(float* matA, float* matB, float* output, int numNodes, int in_dim, int out_dim, cudaStream_t stream);
void softmax_forward_cudnn(float* output, float* input, int numNodes, int dim);
void softmax_forward_cudnn_Stream(float* output, float* input, int numNodes, int dim, cudaStream_t stream);

void gemm_cublas_backward(float* matA, float* matB, float* output, bool Trans_A, bool Trans_B, int m, int n, int k);
void gemm_cublas_backward_Stream(float* matA, float* matB, float* output, bool Trans_A, bool Trans_B, int m, int n, int k, cudaStream_t stream);
void gemm_cblas_backward(float* matA, float* matB, float* output, bool Trans_A, bool Trans_B, int m, int n, int k);

}   //namespace common
}   //namescpce GNNPro_lib