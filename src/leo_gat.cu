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
    int num_GPUs = atoi(argv[2]);           // 2
    int partSize = atoi(argv[3]);           // 32
    int warpPerBlock = atoi(argv[4]);       // 4
    int interleaved_dist = atoi(argv[5]);   // 2
    int dim = atoi(argv[6]);                // 16
    int hiddenSize = atoi(argv[7]);
    int outdim = atoi(argv[8]);
    int grain = atoi(argv[9]);

    tuple<string, string, string, string> allFile = GetFilename(data_name);
    string beg_name, csr_name, weight_name, reorder_name;
    tie(beg_name, csr_name, weight_name, reorder_name) = allFile;
    const char *beg_file = beg_name.c_str();
    const char *csr_file = csr_name.c_str();
    const char *weight_file = weight_name.c_str();
    const char *reorder_file = reorder_name.c_str();
    
    graph<long, long, nidType, nidType, nidType, nidType>* ginst = new graph<long, long, nidType, nidType, nidType, nidType>(beg_file, csr_file, weight_file);
    std::vector<nidType> global_row_ptr(ginst->beg_pos, ginst->beg_pos + ginst->vert_count + 1);
    std::vector<nidType> global_col_ind(ginst->csr, ginst->csr + ginst->edge_count);

    cout << "Complete loading graphs !!" << endl;
    nidType numNodes = global_row_ptr.size() - 1;
    nidType numEdges = global_col_ind.size();  

#ifdef Rabbit
    //std::string dset = "dataset/preprocess/Rabbit_Reddit.txt";
    std::string dset = reorder_file;
    auto reorder_out = rabbit_load_reorder_graph<nidType>(dset, numNodes, numEdges, global_row_ptr, global_col_ind);
    global_row_ptr = reorder_out[0];
    global_col_ind = reorder_out[1];
#endif
    auto e_bound = Simple_Edge_cut<nidType>(global_row_ptr, numNodes, numEdges, num_GPUs);

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
    if (mype_node == 0) {printf("Split_fine (ms): %.3f\n", split_fine_elapsed_ms);}

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

    gat_dense_in2hidden* dp1 = new gat_dense_in2hidden("d-1", d_input, dense, dense_out, numNodes, numEdges, featmyPE, hiddenSize);
    gat_dense_hidden* dp2 = new gat_dense_hidden("d-2", dense, d_input, numNodes, numEdges, hiddenSize, outdim);
    //Edge_softmax* e_smx1 = new Edge_softmax("e-smx1", dp1->edge_atten, dp1->edge_atten, numEdges, 1);//single head attention

    nidType *d_row_ptr, *d_row, *d_col_ind, *id_map;
    gpuErrchk(cudaMalloc((void**)&d_row_ptr, ptr_vec.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind, global_col_ind.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&id_map, ID_map.size() * sizeof(nidType)));
    gpuErrchk(cudaMalloc((void**)&d_row, global_row_ptr.size() * sizeof(nidType)));

    gpuErrchk(cudaMemcpy(d_row_ptr, &ptr_vec[0], ptr_vec.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind, &global_col_ind[0], global_col_ind.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(id_map, &ID_map[0], ID_map.size() * sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_row, &global_row_ptr[0], global_row_ptr.size() * sizeof(nidType), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);

    int num_profiles = 100;
    std::clock_t c_start = std::clock();    
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime(); 

    for (int i = 0; i < num_profiles; i++)
    {
        dense_beg_new_forward(dp1); //d_inout --> dense
        leo_GAT(dp1->d_atten, dp1->part_feat, dp1->d_out, dp1->edge_atten, dp1->edge_tmp, d_row, 
                d_col_ind, mype_node, numNodes, partSize, warpPerBlock, hiddenSize); //dense --> dense_out
        //GAT_edge_softmax(global_row_ptr, e_smx1, numNodes);
        MPI_Barrier(MPI_COMM_WORLD); 
        nvshmem_float_sum_reduce(NVSHMEMX_TEAM_NODE, dp2->d_in, dp1->part_feat, dp2->numNodes * dp2->dim1);//numNodes x hiddenSize
        //dense_out --> dense
        dense_hidden_new_forward(dp2);
        leo_GAT(dp2->d_atten, dp1->d_in, dp2->d_out, dp2->edge_atten, dp2->edge_tmp, d_row,
               d_col_ind, mype_node, numNodes, partSize, warpPerBlock, outdim);
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