#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include "include/model.h"
#include "include/cuda/aggregate.cuh"

using namespace std;
using namespace GNNPro_lib::common;

//#define train_flag

int main(int argc, char* argv[]){

    int num_GPUs = atoi(argv[2]);
    Graph *graph = new Graph(num_GPUs, 0);

    graph->config->ReadFromConfig(argv[1]);
    graph->config->GetFileName();
#ifdef train_flag
    graph->TrainFlag = 1;
#endif
    graph->ConstuctGraph();

    graph->TrainFlag = 0;
    int numNodes = graph->vertices_num;
    int numEdges = graph->edges_num;
    int in_dim = graph->config->in_dim;
    int hid_dim = graph->config->hidden_dim;
    int out_dim = graph->config->out_dim;
    float eps = 0.5;

    int nodePerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    
    float** h_input = new float* [num_GPUs];
    CSR_t** d_row_ptr = new CSR_t* [num_GPUs];
    CSR_t** d_col_ind = new CSR_t* [num_GPUs]; 

    float **d_input, **d_den_out_1, **d_den_out_2, **d_den_out_3, **d_den_out_4, **d_den_out_5;

    CUDA_CALL(cudaMallocManaged((void**)&d_input, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_1, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_2, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_3, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_4, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_5, num_GPUs*sizeof(float*)));

#ifdef train_flag
    CSR_t** d_row_ptr_T = new CSR_t* [num_GPUs];
    CSR_t** d_col_ind_T = new CSR_t* [num_GPUs];
    float **d_den_in_1, **d_den_in_2, **d_den_in_3, **d_den_in_4, **d_den_in_5;
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_1, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_2, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_3, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_4, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_5, num_GPUs*sizeof(float*)));   
#endif

#pragma omp parallel for 
for(int mype_node = 0; mype_node < num_GPUs; mype_node++){
    cudaSetDevice(mype_node);

    h_input[mype_node] = (float*)malloc(nodePerPE*in_dim*sizeof(float));
    std::fill(h_input[mype_node], h_input[mype_node] + nodePerPE*in_dim, 1.0);

    CUDA_CALL(cudaMallocManaged((void**)&d_input[mype_node], nodePerPE*in_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_1[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_2[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_3[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_4[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_5[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_row_ptr[mype_node], (numNodes+1)*sizeof(CSR_t)));    
    CUDA_CALL(cudaMallocManaged((void**)&d_col_ind[mype_node], numEdges*sizeof(CSR_t))); 

    CUDA_CALL(cudaMemcpy(d_input[mype_node], h_input[mype_node], nodePerPE * in_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_row_ptr[mype_node], &graph->full_csr_ptr[0], (numNodes+1) * sizeof(CSR_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_col_ind[mype_node], &graph->full_csr_ind[0], numEdges * sizeof(CSR_t), cudaMemcpyHostToDevice));

#ifdef train_flag
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_1[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_2[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_3[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_4[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_5[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_row_ptr_T[mype_node], (numNodes+1)*sizeof(CSR_t)));
    CUDA_CALL(cudaMallocManaged((void**)&d_col_ind_T[mype_node], numEdges*sizeof(CSR_t)));

    CUDA_CALL(cudaMemcpy(d_row_ptr_T[mype_node], &graph->full_csr_T_ptr[0], (numNodes+1) * sizeof(CSR_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_col_ind_T[mype_node], &graph->full_csr_T_ind[0], numEdges * sizeof(CSR_t), cudaMemcpyHostToDevice));        
#endif       
}

    graph->Create_Weight();

#pragma omp parallel for 
for(int mype_node = 0; mype_node < num_GPUs; mype_node++){
    cudaSetDevice(mype_node);

    //forward
    float *dW1, *dW2, *dW3;
    float *d_aggre_out1, *d_aggre_out2, *d_aggre_out3, *d_aggre_out4, *d_aggre_out5, *d_den_out, *d_softmax_out;

    CUDA_CALL(cudaMalloc((void**)&dW1, in_dim * hid_dim * sizeof(weight_t)));
    CUDA_CALL(cudaMalloc((void**)&dW2, hid_dim * hid_dim * sizeof(weight_t)));
    CUDA_CALL(cudaMalloc((void**)&dW3, hid_dim * out_dim * sizeof(weight_t)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_out1, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_out2, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_out3, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_out4, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_out5, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_den_out, nodePerPE * out_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_softmax_out, nodePerPE * out_dim * sizeof(float)));

    CUDA_CALL(cudaMemcpy(dW1, &graph->weight->W1[0], graph->weight->W1.size() * sizeof(weight_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dW2, &graph->weight->W2[0], graph->weight->W2.size() * sizeof(weight_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dW3, &graph->weight->W3[0], graph->weight->W3.size() * sizeof(weight_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(d_aggre_out1, 0, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMemset(d_aggre_out2, 0, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMemset(d_aggre_out3, 0, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMemset(d_aggre_out4, 0, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMemset(d_aggre_out5, 0, nodePerPE * hid_dim * sizeof(float)));

#ifdef train_flag
    //backward
    int* d_label;
    float *d_aggre_in1, *d_aggre_in2, *d_aggre_in3, *d_aggre_in4, *d_aggre_in5;
    float *d_den_in;
    float *dW_in1, *dW_in2, *dW_in3;
    CUDA_CALL(cudaMalloc((void**)&d_label, nodePerPE * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&dW_in1, in_dim * hid_dim * sizeof(weight_t)));
    CUDA_CALL(cudaMalloc((void**)&dW_in2, hid_dim * hid_dim * sizeof(weight_t)));  
    CUDA_CALL(cudaMalloc((void**)&dW_in3, hid_dim * out_dim * sizeof(weight_t)));
    CUDA_CALL(cudaMalloc((void**)&d_den_in, nodePerPE * out_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_in1, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_in2, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_in3, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_in4, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_in5, nodePerPE * hid_dim * sizeof(float)));

    CUDA_CALL(cudaMemset(d_label, 0, nodePerPE * sizeof(int)));       
#endif

    const int lb = nodePerPE * mype_node;
    const int ub = min(lb + nodePerPE, numNodes);
    const int local_num_node = ub - lb;

    cudaEvent_t start, infer, train;
    cudaEventCreate(&start);
    cudaEventCreate(&infer);
    cudaEventCreate(&train);

    cudaEventRecord(start);

    //Inference
    gemm_cublas(d_input[mype_node], dW1, d_den_out_1[mype_node], local_num_node, in_dim, hid_dim);

    //layer-1
    UVM_GIN_block(d_aggre_out1, d_den_out_1, d_row_ptr[mype_node], d_col_ind[mype_node],
                    local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE, eps);
    gemm_cublas(d_aggre_out1, dW2, d_den_out_2[mype_node], local_num_node, hid_dim, hid_dim);

    //layer-2
    UVM_GIN_block(d_aggre_out2, d_den_out_2, d_row_ptr[mype_node], d_col_ind[mype_node],
                    local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE, eps);
    gemm_cublas(d_aggre_out2, dW2, d_den_out_3[mype_node], local_num_node, hid_dim, hid_dim);

    //layer-3
    UVM_GIN_block(d_aggre_out3, d_den_out_3, d_row_ptr[mype_node], d_col_ind[mype_node],
                    local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE, eps);   
    gemm_cublas(d_aggre_out3, dW2, d_den_out_4[mype_node], local_num_node, hid_dim, hid_dim);

    //layer-4
    UVM_GIN_block(d_aggre_out4, d_den_out_4, d_row_ptr[mype_node], d_col_ind[mype_node],
                    local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE, eps);   
    gemm_cublas(d_aggre_out4, dW2, d_den_out_5[mype_node], local_num_node, hid_dim, hid_dim);    

    //layer-5
    UVM_GIN_block(d_aggre_out5, d_den_out_5, d_row_ptr[mype_node], d_col_ind[mype_node],
                    local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE, eps);     
    gemm_cublas(d_aggre_out5, dW3, d_den_out, local_num_node, hid_dim, out_dim);
    softmax_forward_cudnn(d_softmax_out, d_den_out, local_num_node, out_dim);

    cudaEventRecord(infer);
    cudaEventSynchronize(infer);

#ifdef train_flag
    //Train
    SoftmaxCrossEntroyBackward(d_den_in, d_softmax_out, d_label, local_num_node, out_dim, 4, nullptr);
    //layer-5
    gemm_cublas_backward(d_den_in, d_aggre_out5, dW_in3, false, true, out_dim, hid_dim, local_num_node);
    gemm_cublas_backward(dW3, d_den_in, d_den_in_5[mype_node], true, false, hid_dim, local_num_node, out_dim);
    UVM_GIN_back_block(d_aggre_in5, d_den_in_5, d_row_ptr_T[mype_node], d_col_ind_T[mype_node],
                    local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE, eps);

    //layer-4
    gemm_cublas_backward(d_aggre_in5, d_aggre_out4, dW_in2, false, true, hid_dim, hid_dim, local_num_node);
    gemm_cublas_backward(dW2,d_aggre_in5, d_den_in_4[mype_node], true, false, hid_dim, local_num_node, hid_dim);
    UVM_GIN_back_block(d_aggre_in4, d_den_in_4, d_row_ptr_T[mype_node], d_col_ind_T[mype_node],
                    local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE, eps); 

    // //layer-3
    gemm_cublas_backward(d_aggre_in4, d_aggre_out3, dW_in2, false, true, hid_dim, hid_dim, local_num_node);
    gemm_cublas_backward(dW2, d_aggre_in4, d_den_in_3[mype_node], true, false, hid_dim, local_num_node, hid_dim);
    UVM_GIN_back_block(d_aggre_in3, d_den_in_3, d_row_ptr_T[mype_node], d_col_ind_T[mype_node],
                local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE, eps);

    // //layer-2
    gemm_cublas_backward(d_aggre_in3, d_aggre_out2, dW_in2, false, true, hid_dim, hid_dim, local_num_node);
    gemm_cublas_backward(dW2, d_aggre_in3, d_den_in_2[mype_node], true, false, hid_dim, local_num_node, hid_dim);        
    UVM_GIN_back_block(d_aggre_in2, d_den_in_2, d_row_ptr_T[mype_node], d_col_ind_T[mype_node],
                local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE, eps);

    // //layer-1
    gemm_cublas_backward(d_aggre_in2, d_aggre_out1, dW_in2, false, true, hid_dim, hid_dim, local_num_node);
    gemm_cublas_backward(dW2, d_aggre_in2, d_den_in_1[mype_node], true, false, hid_dim, local_num_node, hid_dim);        
    UVM_GIN_back_block(d_aggre_in1, d_den_in_1, d_row_ptr_T[mype_node], d_col_ind_T[mype_node],
                local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE, eps);

    gemm_cublas_backward(d_aggre_in1, d_input[mype_node], dW_in1, false, true, hid_dim, in_dim, local_num_node);

    cudaEventRecord(train);
    cudaEventSynchronize(train);
#endif

    float time1 = 0;
    float time2 = 0;
    cudaEventElapsedTime(&time1, start, infer);
    printf("Rank %d Inference: %.2f (ms)\n", mype_node, time1);
#ifdef train_flag    
    cudaEventElapsedTime(&time2, start, train);    
    printf("Rank %d Training: %.2f (ms)\n", mype_node, time2);
#endif       
}

    printf("==============Finished!==============\n");

    return 0; 
}