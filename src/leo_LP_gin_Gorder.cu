#include <iostream>
#include <stdio.h>
#include <ctime>
#include <algorithm>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cublas_v2.h>

#include "graph.h"
#include "utils.cuh"
#include "neighbor_utils.cuh"
#include "csr_formatter.h"
#include "layer.h"

#include "cublas_utils.h"
#include "layer_new.cuh"
#include "gnn_layer.cuh"
#include "nccl.h"
#include "LSH.h"
#include "vector_func.h"

#define Rabbit  1
using nidType = int;

using namespace cudl;
using namespace std;

int main(int argc, char* argv[]){

    if (argc < 8){
        printf("Usage: ./main graph.mtx num_GPUs partSize warpPerblock dim interleaved_dist hidden\n");
        return -1;
    }

    string data_name = argv[1];
    nidType numNodes = atoi(argv[2]);
    nidType numEdges = atoi(argv[3]);
    int num_GPUs = atoi(argv[4]);           // 2
    int partSize = atoi(argv[5]);           // 32
    int warpPerBlock = atoi(argv[6]);       // 4
    int interleaved_dist = atoi(argv[7]);   // 2
    int dim = atoi(argv[8]);                // 16
    int hiddenSize = atoi(argv[9]);
    int outdim = atoi(argv[10]);
    int grain = atoi(argv[11]);
    int read_flag = atoi(argv[12]);

    float eps = 0.5;

    string filename_prefix = "Gorder-master/";
    string filename_suffix = "_Gorder.txt";
    string filename = filename_prefix + data_name + filename_suffix;
    const char *graphfile = filename.c_str();
    cout << "Graph File: " << graphfile << '\n';

    Gorder_graph<nidType, nidType, nidType>* Go_graph = new Gorder_graph<nidType, nidType, nidType>(graphfile, numNodes, numEdges);
    std::vector<nidType> global_row_ptr(Go_graph->ptr, Go_graph->ptr + Go_graph->v_num + 1);
    std::vector<nidType> global_col_ind(Go_graph->idx, Go_graph->idx + Go_graph->e_num);
    assert(global_row_ptr.size() == numNodes + 1);
    assert(global_col_ind.size() == numEdges);
    cout << "Complete loading graphs !!" << endl;

    string input_prefix = "dataset/preprocess/";
    string input_suffix = "_ebound.txt";
    string grain_suffix  = "_grain";
    string grain_str     = to_string(grain);
    string input_file = input_prefix + data_name + grain_suffix + grain_str + input_suffix;
    vector<nidType> e_bound;
    if (read_flag == 1){e_bound = Read_ebound<nidType>(input_file);}
    else{e_bound = Simple_Edge_cut<nidType>(global_row_ptr, numNodes, numEdges, num_GPUs);}  
    
    double t1, t2; 
    int rank, nranks;
    cudaStream_t stream;
    nvshmemx_init_attr_t attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    attr.mpi_comm = &mpi_comm;

    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    // Set the workload on each device.
    nidType featdPerPE = (dim + num_GPUs - 1) / num_GPUs;
    nidType featlb = featdPerPE * mype_node;
    nidType featub = (featlb + featdPerPE) < dim? (featlb + featdPerPE) : dim;
    nidType featmyPE = featub - featlb;
    int max_dim = max({hiddenSize, outdim});
    int max_dim_1 = max({featmyPE, outdim});

    std::clock_t start_proc = std::clock();
    auto Fine_out = split_Fine<nidType>(global_row_ptr, global_col_ind, mype_node, grain);
    auto ptr_vec = Fine_out[0];
    auto ID_map = Fine_out[1];
    nidType num_id_map = ID_map.size();
    std::clock_t end_proc = std::clock();

    float split_fine_elapsed_ms = 1000.0 * (end_proc - start_proc) / CLOCKS_PER_SEC;
    if (mype_node == 0) printf("Split_fine (ms): %.3f\n", split_fine_elapsed_ms);

    nidType e_lb = e_bound[mype_node];
    nidType e_ub = e_bound[mype_node + 1];
    auto Find_result = Find_ID_map<nidType>(ID_map, e_lb, e_ub);
    auto range_id_map = Find_result.first;
    auto lower_bound  = Find_result.second;

    float *h_input, *d_input;
    float *dense, *dense_out;
    gpuErrchk(cudaMalloc((void**)&d_input, static_cast<size_t>(numNodes) * max_dim_1 * sizeof(float)));
    h_input = (float *) malloc (numNodes * featmyPE * sizeof(float));
    std::fill_n(h_input, numNodes * featmyPE, 1.0f);
    dense = (float *) nvshmem_malloc (static_cast<size_t>(numNodes) * hiddenSize * sizeof(float)); 
    dense_out = (float *) nvshmem_malloc (static_cast<size_t>(numNodes) * hiddenSize * sizeof(float)); 
    gpuErrchk(cudaMemset(dense_out, 0, numNodes * hiddenSize * sizeof(float)));

    dense_in2hidden* dp1 = new dense_in2hidden("d-1", d_input, dense, numNodes, featmyPE, hiddenSize);
    dense_hidden*    dp2 = new dense_hidden("d-2", dense_out, dense, numNodes, hiddenSize, hiddenSize);
    dense_hidden*    dp3 = new dense_hidden("d-3", dense_out, dense, numNodes, hiddenSize, hiddenSize);
    dense_hidden*    dp4 = new dense_hidden("d-4", dense_out, dense, numNodes, hiddenSize, hiddenSize);
    dense_hidden_sync*    dp5 = new dense_hidden_sync("d-5", dense_out, dense, numNodes, hiddenSize, hiddenSize);
    dense_hidden*    dp6 = new dense_hidden("d-6", dense_out, d_input, numNodes, hiddenSize, outdim);
    softmax_new_param* smx2 = new softmax_new_param("smx-2", d_input, d_input, numNodes, outdim);
 

    nidType *d_row_ptr, *d_col_ind, *id_map;
    gpuErrchk(cudaMalloc((void**)&d_row_ptr, ptr_vec.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind, global_col_ind.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&id_map, ID_map.size() * sizeof(nidType)));

    gpuErrchk(cudaMemcpy(d_row_ptr, &ptr_vec[0], ptr_vec.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind, &global_col_ind[0], global_col_ind.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(id_map, &ID_map[0], ID_map.size() * sizeof(nidType), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    int num_profiles = 100;
    std::clock_t c_start = std::clock();    
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime(); 

    for (int i = 0; i < num_profiles; i++)
    {
        dense_beg_new_forward(dp1);

        //layer 1
        leo_gin_LP_fine_np_div(dp2->d_in, dp1->d_out, id_map, d_row_ptr, d_col_ind, hiddenSize, 
                            numNodes, mype_node, partSize, warpPerBlock, num_id_map, eps);
        dense_hidden_new_forward(dp2);

        //layer 2
        leo_gin_LP_fine_np_div(dp3->d_in, dp2->d_out, id_map, d_row_ptr, d_col_ind, hiddenSize, 
                            numNodes, mype_node, partSize, warpPerBlock, num_id_map, eps);
        dense_hidden_new_forward(dp3);

        //layer 3
        leo_gin_LP_fine_np_div(dp4->d_in, dp3->d_out, id_map, d_row_ptr, d_col_ind, hiddenSize, 
                            numNodes, mype_node, partSize, warpPerBlock, num_id_map, eps);
        dense_hidden_new_forward(dp4);  

        //layer 4     
        leo_gin_LP_fine_np_div(dp5->d_in, dp4->d_out, id_map, d_row_ptr, d_col_ind, hiddenSize, 
                            numNodes, mype_node, partSize, warpPerBlock, num_id_map, eps);
        dense_hidden_new_forward(dp5); 

        nvshmem_float_sum_reduce(NVSHMEMX_TEAM_NODE, dp5->d_out_new, dp5->d_in, dp1->numNodes * dp5->dim2);
        
        //layer 5
        //leo_gin_LP_fine_np_div(dp6->d_in, dp5->d_out_new, id_map, d_row_ptr, d_col_ind, hiddenSize, 
        //                    numNodes, mype_node, partSize, warpPerBlock, num_id_map, eps);
        leo_gin_LP_DP_fine_np_div(dp6->d_in, dp5->d_out_new, id_map, d_row_ptr, d_col_ind, lower_bound, hiddenSize, 
                            numNodes, mype_node, partSize, warpPerBlock, range_id_map, eps);
        dense_hidden_new_forward(dp6);
        softmax_new_forward(smx2);
    }

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / num_profiles;
    printf("PE-%d, Total (ms): %.3f\n", mype_node, time_elapsed_ms);
    MPI_Barrier(MPI_COMM_WORLD); 
    t2 = MPI_Wtime(); 
    if (mype_node == 0) printf( "MPI time (ms) %.3f\n", (t2 - t1)*1e3/num_profiles); 

    // release memory.
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    cudaFree(d_input);

    nvshmem_finalize();


    MPI_Finalize();
    if (mype_node == 0) 
        printf("===================================\n");

    return 0;
}