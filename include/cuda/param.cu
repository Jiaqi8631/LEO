#include "param.h"
#include "logging.cuh"
#include "kernel.cuh"

namespace GNNPro_lib{
namespace common{

#define WARP_SIZE 32

void AdamOptimizer::Param_GPU_Initial(){
    CUDA_CALL(cudaMalloc((void**)&m, element_num * sizeof(value_t)));
    CUDA_CALL(cudaMalloc((void**)&v, element_num * sizeof(value_t)));
    CUDA_CALL(cudaMalloc((void**)&out_param, element_num * sizeof(value_t)));

    CUDA_CALL(cudaMemset(m, 0, element_num * sizeof(value_t)));
    CUDA_CALL(cudaMemset(v, 0, element_num * sizeof(value_t)));
}

void AdamOptimizer::Param_CPU_Initial(){
    size_t size = sizeof(value_t) * element_num;

    if(posix_memalign((void **)&m, getpagesize(), size))
    perror("posix_mamalign");
    if(posix_memalign((void **)&v, getpagesize(), size))
    perror("posix_mamalign");
    if(posix_memalign((void **)&out_param, getpagesize(), size))
    perror("posix_mamalign");

    memset(m, 0, size);
    memset(v, 0, size);
    memset(out_param, 0, size);
}

void AdamOptimizer::Update(value_t* grad, cudaStream_t stream){
    const int block = 4 * WARP_SIZE;
    const int grid = (element_num + block - 1) / block;

    AdamUpdate_cuda<<<grid, block, 0, stream>>>(out_param, grad, m, v, element_num, beta1, beta2, eps, alpha, t);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @AdamOptimizer::Update: %s\n", cudaGetErrorString(error));
    }
    t = t + 1;
}

void AdamOptimizer::Update_stream(value_t* grad, cudaStream_t stream){
    const int block = 4 * WARP_SIZE;
    const int grid = (element_num + block - 1) / block;

    AdamUpdate_cuda<<<grid, block, 0, stream>>>(out_param, grad, m, v, element_num, beta1, beta2, eps, alpha, t);
    t = t + 1;
}

void AdamOptimizer::Update_CPU(value_t* grad){
    for(int i = 0; i < element_num; i++){
        //Update biased first moment estimate
        m[i] = beta1 * m[i] + (1 - beta1) * grad[i];
        //Update biased second raw moment estimate
        v[i] = beta2 * v[i] + (1 - beta2) * grad[i] * grad[i];
        //Compute bias-corrected first moment estimate
        float m_hat = m[i] / (1 - pow(beta1, t));
        //Compute bias-corrected second raw moment estimate
        float v_hat = v[i] / (1 - pow(beta2, t));
        //Update parameters
        out_param[i] -= alpha * m_hat / (sqrt(v_hat) + eps);
    }
    
    t = t + 1;
}

}   //namespace common
}   //namescpce GNNPro_lib  