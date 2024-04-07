#include <iostream>
#include <stdio.h>
#include <omp.h>
#include <algorithm>

#include "graph.h"
#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "csr_formatter.h"
#include "cublas_utils.h"
#include "gnn_layer.cuh"

using namespace std;

int main(int argc, char* argv[]){

    if (argc < 5){
        printf("Usage: ./main beg_file.bin csr_file.bin weight_file.bin num_GPUs dimin hidden out\n");
        return -1;
    }

    cout << "Graph File: " << argv[1] << '\n';
    const char *beg_file = argv[1];
	const char *csr_file = argv[2];
	const char *weight_file = argv[3];
    
    int num_GPUs = atoi(argv[4]);
    int dim = atoi(argv[5]);
    int hiddenSize = atoi(argv[6]);
    int outdim = atoi(argv[7]);
    float eps = 0.5;

    graph<long, long, nidType, nidType, nidType, nidType>* ginst = new graph<long, long, nidType, nidType, nidType, nidType>(beg_file, csr_file, weight_file);
    std::vector<nidType> global_row_ptr(ginst->beg_pos, ginst->beg_pos + ginst->vert_count + 1);
    std::vector<nidType> global_col_ind(ginst->csr, ginst->csr + ginst->edge_count);

    nidType numNodes = global_row_ptr.size() - 1;
    nidType numEdges = global_col_ind.size();    

    int nodesPerPE = (numNodes + num_GPUs - 1) / num_GPUs;
    float** h_input = new float*[num_GPUs];
    nidType **d_row_ptr = new nidType*[num_GPUs]; 
    nidType **d_col_ind = new nidType*[num_GPUs]; 

    float   **d_input, **d_den_out; 
            
    gpuErrchk(cudaMallocManaged((void**)&d_input,       num_GPUs*sizeof(float*))); 
    gpuErrchk(cudaMallocManaged((void**)&d_den_out,     num_GPUs*sizeof(float*)));  

#pragma omp parallel for
for (int mype_node = 0; mype_node < num_GPUs; mype_node++)
{
    cudaSetDevice(mype_node);
    printf("mype_node: %d, nodesPerPE: %d\n", mype_node, nodesPerPE);

    gpuErrchk(cudaMallocManaged((void**)&d_input[mype_node], static_cast<size_t>(numNodes * dim * sizeof(float))));
    gpuErrchk(cudaMallocManaged((void**)&d_den_out[mype_node], static_cast<size_t>(numNodes * hiddenSize * sizeof(float))));
    gpuErrchk(cudaMallocManaged((void**)&d_row_ptr[mype_node], (numNodes+1)*sizeof(nidType)));
    gpuErrchk(cudaMallocManaged((void**)&d_col_ind[mype_node], numEdges*sizeof(nidType))); 

    cudaMemAdvise(d_input[mype_node], static_cast<size_t>(numNodes * dim * sizeof(float)), cudaMemAdviseSetReadMostly, mype_node);
    //cudaMemAdvise(d_den_out[mype_node], static_cast<size_t>(numNodes * hiddenSize * sizeof(float)), cudaMemAdviseSetAccessedBy, mype_node);
    cudaMemAdvise(d_row_ptr[mype_node], (numNodes+1)*sizeof(nidType), cudaMemAdviseSetReadMostly, mype_node);
    cudaMemAdvise(d_col_ind[mype_node], numEdges*sizeof(nidType), cudaMemAdviseSetReadMostly, mype_node);
    gpuErrchk(cudaMemcpy(d_row_ptr[mype_node], &global_row_ptr[0], (numNodes+1)*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind[mype_node], &global_col_ind[0], numEdges*sizeof(nidType), cudaMemcpyHostToDevice));

}  

#pragma omp parallel for 
for (int mype_node = 0; mype_node < num_GPUs; mype_node++)
{
    cudaSetDevice(mype_node);
    float *dsp_out, *d_final_hidden, *d_output;

    gpuErrchk(cudaMalloc((void**)&dsp_out, static_cast<size_t>(numNodes)*max(hiddenSize, outdim)*sizeof(float))); 
    gpuErrchk(cudaMemset(dsp_out, 0, static_cast<size_t>(numNodes)*max(hiddenSize, outdim)*sizeof(float)));

    //d_final_hidden: nodesPerPE x hiddenSize
    gpuErrchk(cudaMalloc((void**)&d_final_hidden, static_cast<size_t>(nodesPerPE)*hiddenSize*sizeof(float)));
    gpuErrchk(cudaMemset(d_final_hidden, 0, static_cast<size_t>(nodesPerPE)*hiddenSize*sizeof(float)));

    //d_output: nodesPerPE x outdim
    gpuErrchk(cudaMalloc((void**)&d_output, static_cast<size_t>(nodesPerPE)*outdim*sizeof(float))); 
    gpuErrchk(cudaMemset(d_output, 0, static_cast<size_t>(nodesPerPE)*outdim*sizeof(float)));

    dense_in2hidden_uvm_gin* dp1 = new dense_in2hidden_uvm_gin("d-1", d_input[mype_node], d_den_out, mype_node, numNodes, dim, hiddenSize);
    dense_hidden_uvm_gin* dp2 = new dense_hidden_uvm_gin("d-2", dsp_out, d_den_out, mype_node, numNodes, hiddenSize, hiddenSize);
    dense_hidden_uvm_gin* dp3 = new dense_hidden_uvm_gin("d-3", dsp_out, d_den_out, mype_node, numNodes, hiddenSize, hiddenSize);
    dense_hidden_uvm_gin* dp4 = new dense_hidden_uvm_gin("d-4", dsp_out, d_den_out, mype_node, numNodes, hiddenSize, hiddenSize);
    dense_hidden_uvm_gin* dp5 = new dense_hidden_uvm_gin("d-5", dsp_out, d_den_out, mype_node, numNodes, hiddenSize, hiddenSize);
    dense_hidden2out_uvm_gin* dp6 = new dense_hidden2out_uvm_gin("d-6", d_final_hidden, d_output, mype_node, nodesPerPE, hiddenSize, outdim);
    softmax_new_param* smx2 = new softmax_new_param("smx-2", d_output, d_output, nodesPerPE, outdim); //Apply softmax to nodesPerPE vertices

    const int lb_src = nodesPerPE * mype_node;
    const int ub_src = min_val(lb_src + nodesPerPE, numNodes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); 

    dense_beg_forward_uvm(dp1);   

    //layer-1
    GIN_full_UVM_updated(dsp_out, d_den_out,
                         d_row_ptr[mype_node], d_col_ind[mype_node],
                         hiddenSize, num_GPUs, mype_node,
                         numNodes, eps);
    dense_hidden_forward_uvm(dp2);

    //layer-2 
    GIN_full_UVM_updated(dsp_out, d_den_out,
                         d_row_ptr[mype_node], d_col_ind[mype_node],
                         hiddenSize, num_GPUs, mype_node,
                         numNodes, eps);
    dense_hidden_forward_uvm(dp3);

    //layer-3
    GIN_full_UVM_updated(dsp_out, d_den_out,
                         d_row_ptr[mype_node], d_col_ind[mype_node],
                         hiddenSize, num_GPUs, mype_node,
                         numNodes, eps);
    dense_hidden_forward_uvm(dp4);

    //layer-4
    GIN_full_UVM_updated(dsp_out, d_den_out,
                         d_row_ptr[mype_node], d_col_ind[mype_node],
                         hiddenSize, num_GPUs, mype_node,
                         numNodes, eps);
    dense_hidden_forward_uvm(dp5);

    //layer-5
    GIN_partial_UVM_updated(d_final_hidden, d_den_out,
                             d_row_ptr[mype_node], d_col_ind[mype_node],
                             lb_src, ub_src,
                             hiddenSize, num_GPUs, mype_node,
                             nodesPerPE, numNodes, eps);
    dense_hidden_forward_uvm(dp6);

    //softmax
    softmax_new_forward(smx2);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time (ms): %.2f\n", milliseconds);

    cudaFree(dsp_out);
    cudaFree(d_final_hidden);
    cudaFree(d_output);      
}

#pragma omp parallel for
for (int mype_node = 0; mype_node < num_GPUs; mype_node++)
{
    cudaSetDevice(mype_node);

    cudaFree(d_den_out[mype_node]);
    cudaFree(d_input[mype_node]);    
    cudaFree(d_col_ind[mype_node]);
    cudaFree(d_row_ptr[mype_node]);
}
    cudaFree(d_den_out);
    cudaFree(d_input);
    cudaFree(d_col_ind);
    cudaFree(d_row_ptr);

    for (int i = 0; i < num_GPUs; i++) {
        cudaSetDevice(i);
        cudaDeviceReset();
    }
    
    return 0;
}