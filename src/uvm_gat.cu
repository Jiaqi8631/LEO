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

    graph<long, long, nidType, nidType, nidType, nidType>* ginst = new graph<long, long, nidType, nidType, nidType, nidType>(beg_file, csr_file, weight_file);
    std::vector<nidType> global_row_ptr(ginst->beg_pos, ginst->beg_pos + ginst->vert_count + 1);
    std::vector<nidType> global_col_ind(ginst->csr, ginst->csr + ginst->edge_count);

    nidType numNodes = global_row_ptr.size() - 1;
    nidType numEdges = global_col_ind.size();   

    int nodesPerPE = (numNodes + num_GPUs - 1)/num_GPUs;
    nidType **d_row_ptr = new nidType* [num_GPUs];
    nidType **d_col_ind = new nidType* [num_GPUs];

    float **d_input, **d_hidden;
    gpuErrchk(cudaMallocManaged((void**)&d_input, num_GPUs*sizeof(float *)));
    gpuErrchk(cudaMallocManaged((void**)&d_hidden, num_GPUs*sizeof(float *)));

#pragma omp parallel for 
for(int mype_node = 0; mype_node < num_GPUs; mype_node++)
{
    cudaSetDevice(mype_node);
    printf("mype_node: %d, nodesPerPE: %d\n", mype_node, nodesPerPE);

    gpuErrchk(cudaMallocManaged((void**)&d_input[mype_node], static_cast<size_t>(numNodes * dim * sizeof(float))));
    gpuErrchk(cudaMallocManaged((void**)&d_hidden[mype_node], static_cast<size_t>(numNodes * max(hiddenSize, outdim) * sizeof(float))));

    gpuErrchk(cudaMallocManaged((void**)&d_row_ptr[mype_node], (numNodes+1)*sizeof(nidType)));
    gpuErrchk(cudaMallocManaged((void**)&d_col_ind[mype_node], numEdges*sizeof(nidType)));
    gpuErrchk(cudaMemcpy(d_row_ptr[mype_node], &global_row_ptr[0], (numNodes+1)*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind[mype_node], &global_col_ind[0], numEdges*sizeof(nidType), cudaMemcpyHostToDevice));

    cudaMemAdvise(d_input[mype_node], static_cast<size_t>(numNodes * dim * sizeof(float)), cudaMemAdviseSetReadMostly, mype_node);
    cudaMemAdvise(d_row_ptr[mype_node], (numNodes+1)*sizeof(nidType), cudaMemAdviseSetReadMostly, mype_node);
    cudaMemAdvise(d_col_ind[mype_node], numEdges*sizeof(nidType), cudaMemAdviseSetReadMostly, mype_node);
}

#pragma omp parallel for 
for(int mype_node = 0; mype_node < num_GPUs; mype_node++)
{
    cudaSetDevice(mype_node);
    float *dsp_out, *d_output;

    gpuErrchk(cudaMalloc((void**)&dsp_out, static_cast<size_t>(numNodes)*max(hiddenSize, outdim)*sizeof(float))); 
    gpuErrchk(cudaMemset(dsp_out, 0, static_cast<size_t>(numNodes)*max(hiddenSize, outdim)*sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_output, static_cast<size_t>(nodesPerPE)*outdim*sizeof(float))); 
    gpuErrchk(cudaMemset(d_output, 0, static_cast<size_t>(nodesPerPE)*outdim*sizeof(float)));

    dense_in2hidden_uvm_gat* dp1 = new dense_in2hidden_uvm_gat("d-1", d_input[mype_node], d_hidden, mype_node, numNodes, numEdges, dim, hiddenSize);
    dense_hidden2out_uvm_gat* dp2 = new dense_hidden2out_uvm_gat("d-2", dsp_out, d_hidden, mype_node, numNodes, numEdges, hiddenSize, outdim);

    const int lb_src = nodesPerPE * mype_node;
    const int ub_src = min_val(lb_src+nodesPerPE, numNodes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start); 

    dense_hidden_forward_uvm(dp1);
    GAT_full_UVM_updated(dp1->d_atten, dsp_out, d_hidden, 
                        dp1->edge_atten, dp1->edge_tmp,
                        d_row_ptr[mype_node], d_col_ind[mype_node],
                        numNodes, hiddenSize, mype_node);

    dense_hidden_forward_uvm(dp2);
    GAT_partial_UVM_updated(dp2->d_atten, d_output, d_hidden,
                            dp1->edge_atten, dp1->edge_tmp,
                            d_row_ptr[mype_node], d_col_ind[mype_node],
                            lb_src, ub_src, nodesPerPE,
                            numNodes, outdim, mype_node);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time (ms): %.2f\n", milliseconds);
    cudaFree(dsp_out);
    cudaFree(d_output);
}

#pragma omp parallel for
for (int mype_node = 0; mype_node < num_GPUs; mype_node++){
    cudaSetDevice(mype_node);
    cudaFree(d_hidden[mype_node]);
    cudaFree(d_input[mype_node]);    
    cudaFree(d_col_ind[mype_node]);
    cudaFree(d_row_ptr[mype_node]);
}
    cudaFree(d_hidden);
    cudaFree(d_input);
    cudaFree(d_col_ind);
    cudaFree(d_row_ptr);

    for (int i = 0; i < num_GPUs; i++) {
        cudaSetDevice(i);
        cudaDeviceReset();
    }

    return 0;
}