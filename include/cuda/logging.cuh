#ifndef logging_cuh
#define logging_cuh

#include "logging.h"
#include <cuda.h>
#include <cudnn.h>
#include <library_types.h>
#include <cublas_v2.h>
#include <cublas_api.h>

namespace GNNPro_lib{
namespace common{

//CUDA API error checking
#define CUDA_CALL(ans) { GPUAssert((ans), __FILE__, __LINE__); }
inline void GPUAssert(cudaError_t code, const char *file, int line){
    CHECK(code == cudaSuccess) \
    << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line;
    if(code != cudaSuccess) exit(code); 
}  

//cuBLAS API error checking
#define CUBLAS_CALL(ans) {cuBLASAssert((ans), __FILE__, __LINE__); }
inline void cuBLASAssert(cublasStatus_t code, const char *file, int line){
    CHECK(code == CUBLAS_STATUS_SUCCESS) \
    << "cuBLASAssert: " << code << " " << file << " " << line;
    if(code != CUBLAS_STATUS_SUCCESS) exit(code);
}

//cudnn API error checking
#define CUDNN_CALL(ans) { cudnnAssert((ans), __FILE__, __LINE__);}
inline void cudnnAssert(cudnnStatus_t code, const char *file, int line){
    CHECK(code == CUDNN_STATUS_SUCCESS) \
    << "cudnnAssert: " << code << " " << file << " " << line;
    if(code != CUDNN_STATUS_SUCCESS) exit(code);
}

}   //namespace common
}   //namescpce GNNPro_lib

#endif