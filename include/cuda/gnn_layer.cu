#include "gnn_layer.cuh"

namespace GNNPro_lib{
namespace common{

//GEMM in the forward process on the cpu side
//matA: Weight Matrix; matB: Input Matrix
void gemm_cblas(float* matA, float* matB, float* output, int m, int n, int k){
    float alpha, beta;
    int lda, ldb, ldout;
    CBLAS_ORDER order;
    CBLAS_TRANSPOSE transA, transB;

    alpha = 1.0f;
    beta = 0.0f;
    lda = m, ldb = k, ldout = m;
    order = CblasColMajor;
    transA = CblasNoTrans;
    transB = CblasNoTrans;
    
    cblas_sgemm(order, transA, transB, m, n, k, alpha, matA, lda, matB, ldb, beta, output, ldout);
}

//GEMM in forward process
//matA: Input Matrix; matB: Weight Matrix
void gemm_cublas(float* matA, float* matB, float* output, int numNodes, int in_dim, int out_dim){
    int m, n, k;
    int ldx, ldw, ldout;
    float alpha, beta;
    cublasOperation_t transA, transB;
    cublasHandle_t cublasH;

    m = out_dim, n = numNodes, k = in_dim;
    ldx = in_dim, ldw = out_dim, ldout = out_dim;

    alpha = 1.0f;
    beta = 0.0f;

    transA = CUBLAS_OP_N;
    transB = CUBLAS_OP_N;
    cublasH = NULL; 

    CUBLAS_CALL(cublasCreate(&cublasH));
    CUBLAS_CALL(cublasSgemm(cublasH, transA, transB, m, n, k, &alpha, matB, ldw, matA, ldx, &beta, output, ldout));

    CUBLAS_CALL(cublasDestroy(cublasH));
}

//GEMM in forward process (using CUDA stream)
void gemm_cublas_Stream(float* matA, float* matB, float* output, int numNodes, int in_dim, int out_dim, cudaStream_t stream){
    int m, n, k;
    int ldx, ldw, ldout;
    float alpha, beta;
    cublasOperation_t transA, transB;
    cublasHandle_t cublasH;

    m = out_dim, n = numNodes, k = in_dim;
    ldx = in_dim, ldw = out_dim, ldout = out_dim;

    alpha = 1.0f;
    beta = 0.0f;

    transA = CUBLAS_OP_N;
    transB = CUBLAS_OP_N;
    cublasH = NULL; 

    cublasCreate(&cublasH);
    cublasSetStream(cublasH, stream);
    cublasSgemm(cublasH, transA, transB, m, n, k, &alpha, matB, ldw, matA, ldx, &beta, output, ldout);
    cublasDestroy(cublasH);            
}

//Implementing softmax using cudnn
void softmax_forward_cudnn(float* output, float* input, int numNodes, int dim){
    cudnnHandle_t cudnn_handle;
    CUDNN_CALL(cudnnCreate(&cudnn_handle));
    
    cudnnTensorDescriptor_t input_Desc, output_Desc;
    cudnnCreateTensorDescriptor(&input_Desc);
    cudnnCreateTensorDescriptor(&output_Desc);
    cudnnSetTensor4dDescriptor(input_Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, \
                                numNodes, dim, 1, 1);
    cudnnSetTensor4dDescriptor(output_Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, \
                                numNodes, dim, 1, 1);

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CALL(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, \
                &alpha, input_Desc, input, &beta, output_Desc, output));
    
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_Desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_Desc));
    CUDNN_CALL(cudnnDestroy(cudnn_handle));
}

//Implementing softmax using cudnn (using CUDA stream)
void softmax_forward_cudnn_Stream(float* output, float* input, int numNodes, int dim, cudaStream_t stream){
    cudnnHandle_t cudnn_handle;
    CUDNN_CALL(cudnnCreate(&cudnn_handle));
    CUDNN_CALL(cudnnSetStream(cudnn_handle, stream));
    
    cudnnTensorDescriptor_t input_Desc, output_Desc;
    cudnnCreateTensorDescriptor(&input_Desc);
    cudnnCreateTensorDescriptor(&output_Desc);
    cudnnSetTensor4dDescriptor(input_Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, \
                                numNodes, dim, 1, 1);
    cudnnSetTensor4dDescriptor(output_Desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, \
                                numNodes, dim, 1, 1);

    float alpha = 1.0f;
    float beta = 0.0f;

    CUDNN_CALL(cudnnSoftmaxForward(cudnn_handle, CUDNN_SOFTMAX_ACCURATE, CUDNN_SOFTMAX_MODE_CHANNEL, \
                &alpha, input_Desc, input, &beta, output_Desc, output));
    
    CUDNN_CALL(cudnnDestroyTensorDescriptor(input_Desc));
    CUDNN_CALL(cudnnDestroyTensorDescriptor(output_Desc));
    CUDNN_CALL(cudnnDestroy(cudnn_handle));    
}

//GEMM in backward process
void gemm_cublas_backward(float* matA, float* matB, float* output, bool Trans_A, bool Trans_B, int m, int n, int k){
//m,n,k are logical dimensions, only the transpose flags are changed
    int lda, ldb, ldout;
    float alpha, beta;
    cublasOperation_t transA, transB;
    cublasHandle_t cublasH;

    //lda = m, ldb = n, ldout = m;
    ldout = m;

    alpha = 1.0f;
    beta = 0.0f;

    if(Trans_A == true){
        transA = CUBLAS_OP_T;
        lda = k;
    }else{
        transA = CUBLAS_OP_N;
        lda = m;
    }

    if(Trans_B == true){
        transB = CUBLAS_OP_T;
        ldb = n;
    }else{
        transB = CUBLAS_OP_N;
        ldb = k;
    }

    cublasH = NULL;

    CUBLAS_CALL(cublasCreate(&cublasH));
    CUBLAS_CALL(cublasSgemm(cublasH, transA, transB, m, n, k, &alpha, matA, lda, matB, ldb, &beta, output, ldout));

    CUBLAS_CALL(cublasDestroy(cublasH));
}

//GEMM in backward process (using CUDA stream)
void gemm_cublas_backward_Stream(float* matA, float* matB, float* output, bool Trans_A, bool Trans_B, int m, int n, int k, cudaStream_t stream){
//m,n,k are logical dimensions, only the transpose flags are changed
    int lda, ldb, ldout;
    float alpha, beta;
    cublasOperation_t transA, transB;
    cublasHandle_t cublasH;

    //lda = m, ldb = n, ldout = m;
    ldout = m;

    alpha = 1.0f;
    beta = 0.0f;

    if(Trans_A == true){
        transA = CUBLAS_OP_T;
        lda = k;
    }else{
        transA = CUBLAS_OP_N;
        lda = m;
    }

    if(Trans_B == true){
        transB = CUBLAS_OP_T;
        ldb = n;
    }else{
        transB = CUBLAS_OP_N;
        ldb = k;
    }

    cublasH = NULL;

    cublasCreate(&cublasH);
    //Set up cuBLAS stream
    cublasSetStream(cublasH, stream);
    cublasSgemm(cublasH, transA, transB, m, n, k, &alpha, matA, lda, matB, ldb, &beta, output, ldout);

    cublasDestroy(cublasH);
}

//GEMM in the backward process on the cpu side
void gemm_cblas_backward(float* matA, float* matB, float* output, bool Trans_A, bool Trans_B, int m, int n, int k){
    float alpha, beta;
    int lda, ldb, ldout;
    CBLAS_ORDER order;
    CBLAS_TRANSPOSE transA, transB;

    alpha = 1.0f;
    beta = 0.0f;
    //lda = m, ldb = k, ldout = m;
    ldout = m;
    order = CblasColMajor;
    
    if(Trans_A == true){
        transA = CblasTrans;
        lda = k;
    }else{
        transA = CblasNoTrans;
        lda = m;
    }
    
    if(Trans_B == true){
        transB = CblasTrans;
        ldb = n;
    }else{
        transB = CblasNoTrans;
        ldb = k;
    }
    
    cblas_sgemm(order, transA, transB, m, n, k, alpha, matA, lda, matB, ldb, beta, output, ldout);
}

}   //namespace common
}   //namescpce GNNPro_lib