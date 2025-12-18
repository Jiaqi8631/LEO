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

    int numNodes = graph->vertices_num;
    int numEdges = graph->edges_num;
    int in_dim = graph->config->in_dim;
    int hid_dim = graph->config->hidden_dim;
    int out_dim = graph->config->out_dim;

    int nodePerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    
    float** h_input = new float* [num_GPUs];
    CSR_t** d_row_ptr = new CSR_t* [num_GPUs];
    CSR_t** d_col_ind = new CSR_t* [num_GPUs];


    float **d_input, **d_den_out_1, **d_den_out_2;

    CUDA_CALL(cudaMallocManaged((void**)&d_input, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_1, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_2, num_GPUs*sizeof(float*)));

#ifdef train_flag
    CSR_t** d_row_ptr_T = new CSR_t* [num_GPUs];
    CSR_t** d_col_ind_T = new CSR_t* [num_GPUs];
    float **d_den_in_1, **d_den_in_2;
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_1, num_GPUs*sizeof(float*)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_2, num_GPUs*sizeof(float*)));
#endif


#pragma omp parallel for 
for(int mype_node = 0; mype_node < num_GPUs; mype_node++){
    cudaSetDevice(mype_node);

    h_input[mype_node] = (float*)malloc(nodePerPE*in_dim*sizeof(float));
    std::fill(h_input[mype_node], h_input[mype_node] + nodePerPE*in_dim, 1.0);

    CUDA_CALL(cudaMallocManaged((void**)&d_input[mype_node], nodePerPE*in_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_1[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_out_2[mype_node], nodePerPE*out_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_row_ptr[mype_node], (numNodes+1)*sizeof(CSR_t)));    
    CUDA_CALL(cudaMallocManaged((void**)&d_col_ind[mype_node], numEdges*sizeof(CSR_t)));
    

    CUDA_CALL(cudaMemcpy(d_input[mype_node], h_input[mype_node], nodePerPE * in_dim * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_row_ptr[mype_node], &graph->full_csr_ptr[0], (numNodes+1) * sizeof(CSR_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_col_ind[mype_node], &graph->full_csr_ind[0], numEdges * sizeof(CSR_t), cudaMemcpyHostToDevice));

#ifdef train_flag
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_1[mype_node], nodePerPE*hid_dim*sizeof(float)));
    CUDA_CALL(cudaMallocManaged((void**)&d_den_in_2[mype_node], nodePerPE*out_dim*sizeof(float)));
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
    float *dW1, *dW2;
    float *d_aggre_out1, *d_aggre_out2, *d_softmax_out;

    CUDA_CALL(cudaMalloc((void**)&dW1, in_dim * hid_dim * sizeof(weight_t)));
    CUDA_CALL(cudaMalloc((void**)&dW2, hid_dim * out_dim * sizeof(weight_t)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_out1, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_out2, nodePerPE * out_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_softmax_out, nodePerPE * out_dim * sizeof(float)));

    CUDA_CALL(cudaMemcpy(dW1, &graph->weight->W1[0], graph->weight->W1.size() * sizeof(weight_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(dW2, &graph->weight->W2[0], graph->weight->W2.size() * sizeof(weight_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemset(d_aggre_out1, 0, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMemset(d_aggre_out2, 0, nodePerPE * out_dim * sizeof(float)));
    CUDA_CALL(cudaMemset(d_softmax_out, 0, nodePerPE * out_dim * sizeof(float)));

#ifdef train_flag
    //backward
    int* d_label;
    float *d_aggre_in1, *d_aggre_in2;
    float *dW_in1, *dW_in2;
    CUDA_CALL(cudaMalloc((void**)&d_label, nodePerPE * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&dW_in1, in_dim * hid_dim * sizeof(weight_t)));
    CUDA_CALL(cudaMalloc((void**)&dW_in2, hid_dim * out_dim * sizeof(weight_t)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_in1, nodePerPE * hid_dim * sizeof(float)));
    CUDA_CALL(cudaMalloc((void**)&d_aggre_in2, nodePerPE * out_dim * sizeof(float)));

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
    UVM_GCN_block(d_aggre_out1, d_den_out_1, d_row_ptr[mype_node], d_col_ind[mype_node],
                    local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE);

    gemm_cublas(d_aggre_out1, dW2, d_den_out_2[mype_node], local_num_node, hid_dim, out_dim);
    UVM_GCN_block(d_aggre_out2, d_den_out_2, d_row_ptr[mype_node], d_col_ind[mype_node],
                local_num_node, numNodes, out_dim, num_GPUs, mype_node, nodePerPE);
    softmax_forward_cudnn(d_softmax_out, d_aggre_out2, local_num_node, out_dim);
    
    cudaEventRecord(infer);
    cudaEventSynchronize(infer);

#ifdef train_flag   
    //Train
    SoftmaxCrossEntroyBackward(d_den_in_2[mype_node], d_softmax_out, d_label, local_num_node, out_dim, 4, nullptr);
    UVM_GCN_back_block(d_aggre_in2, d_den_in_2, d_row_ptr_T[mype_node], d_col_ind_T[mype_node],
                local_num_node, numNodes, out_dim, num_GPUs, mype_node, nodePerPE);
    gemm_cublas_backward(d_aggre_in2, d_aggre_out1, dW_in2, false, true, out_dim, hid_dim, local_num_node);
    gemm_cublas_backward(dW2, d_aggre_in2, d_den_in_1[mype_node], true, false, hid_dim, local_num_node, out_dim);

    UVM_GCN_back_block(d_aggre_in1, d_den_in_1, d_row_ptr_T[mype_node], d_col_ind_T[mype_node],
                local_num_node, numNodes, hid_dim, num_GPUs, mype_node, nodePerPE);
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

#pragma omp parallel for 
for(int mype_node = 0; mype_node < num_GPUs; mype_node++){
    cudaSetDevice(mype_node);
    cudaFree(d_input[mype_node]);
    cudaFree(d_den_out_1[mype_node]);
    cudaFree(d_den_out_2[mype_node]);
    cudaFree(d_row_ptr[mype_node]);
    cudaFree(d_col_ind[mype_node]);

#ifdef train_flag
    cudaFree(d_den_in_1[mype_node]);
    cudaFree(d_den_in_2[mype_node]);
    cudaFree(d_row_ptr_T[mype_node]);
    cudaFree(d_col_ind_T[mype_node]);
#endif    
}
    cudaFree(d_den_out_1);
    cudaFree(d_den_out_2);
    cudaFree(d_input);
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);

#ifdef train_flag
    cudaFree(d_den_in_1);
    cudaFree(d_den_in_2);
    cudaFree(d_row_ptr_T);
    cudaFree(d_col_ind_T);    
#endif
    printf("==============Finished!==============\n");

    return 0;   
}