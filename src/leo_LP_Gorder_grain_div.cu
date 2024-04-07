
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

//#define validate 1 // the number (< num_GPUs) indicates the validation on which PE.
// using nidType = size_t;
// using nidType = long;

using nidType = int;

using namespace cudl;
using namespace std;

int main(int argc, char* argv[]){
	
    if (argc < 8){
        printf("Usage: ./main graph.mtx num_GPUs partSize warpPerblock dim interleaved_dist hidden\n");
        return -1;
    }
    
    cout << "Graph File: " << argv[1] << '\n';
    const char *graphfile = argv[1];
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
    nidType balanced_grain_interval= atoi(argv[12]);

    // graph<long, long, nidType, nidType, nidType, nidType>* ginst = new graph<long, long, nidType, nidType, nidType, nidType>(beg_file, csr_file, weight_file);
    // std::vector<nidType> global_row_ptr(ginst->beg_pos, ginst->beg_pos + ginst->vert_count + 1);
    // std::vector<nidType> global_col_ind(ginst->csr, ginst->csr + ginst->edge_count);
    Gorder_graph<nidType, nidType, nidType>* Go_graph = new Gorder_graph<nidType, nidType, nidType>(graphfile, numNodes, numEdges);
    std::vector<nidType> global_row_ptr(Go_graph->ptr, Go_graph->ptr + Go_graph->v_num + 1);
    std::vector<nidType> global_col_ind(Go_graph->idx, Go_graph->idx + Go_graph->e_num);
    assert(global_row_ptr.size() == numNodes + 1);
    assert(global_col_ind.size() == numEdges);

    cout << "Complete loading graphs !!" << endl;
    //nidType numNodes = global_row_ptr.size() - 1;
    //nidType numEdges = global_col_ind.size();    
 
    // std::cout << "max node: " << *std::max_element(std::begin(global_col_ind), std::end(global_col_ind)) << '\n';

    double t1, t2; 
    // print_array<int>("global_row_ptr", global_row_ptr, global_row_ptr.size());
    // print_array<int>("global_col_ind", global_col_ind, global_col_ind.size());
    int rank, nranks;
    cudaStream_t stream;
    nvshmemx_init_attr_t attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    attr.mpi_comm = &mpi_comm;

    // Set up NVSHMEM device.
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    // Set the workload on each device.
    nidType featdPerPE = (dim + num_GPUs - 1) / num_GPUs;
    // printf("numNodes: %d, nodesPerPE: %d\n", numNodes, cd nodesPerPE);
    nidType featlb = featdPerPE * mype_node;
    nidType featub = (featlb + featdPerPE) < dim? (featlb + featdPerPE) : dim;
    nidType featmyPE = featub - featlb;
    int max_dim = max({hiddenSize, outdim});
    int max_dim_1 = max({featmyPE, outdim});

    auto Fine_out = spilt_Fine_group_accum<nidType>(global_row_ptr, global_col_ind, mype_node, grain);
    auto ptr_vec = Fine_out[0];
    auto ID_map = Fine_out[1];
    auto group_num = Fine_out[2];
    nidType num_id_map = ID_map.size();

    //nidType balanced_grain_interval = 200;
    auto e_bound =  Balanced_grain_cut<nidType>(ID_map, balanced_grain_interval, numNodes, num_id_map, num_GPUs);
    if(mype_node == 0){
        cout << "e_bound: ";
        for (auto i:e_bound){cout << i << " ";}
        cout << "\n";
    }
    nidType e_lb = e_bound[mype_node];
    nidType e_ub = e_bound[mype_node + 1];
    auto Find_result = Find_ID_map<nidType>(ID_map, e_lb, e_ub);
    auto range_id_map = Find_result.first;
    auto lower_bound  = Find_result.second;
    printf("PE-%d: range_id_map: %d\n", mype_node, range_id_map);
    printf("result1: %d\n", group_num[e_ub] - group_num[e_lb]);

    printf("myID is %d\n", getpid());  // 输出当前进程的进程ID
    int j=1;         
    while(j){
      sleep(2);  // 陷入休眠，避免执行到程序异常处，导致中途退出
    }

    // Allocate memory on each device.
    float *h_input;
    float *d_buff_1, *d_buff_2, *d_buff_i2h, *d_buff_h2o, *d_hidden_all;

    // d_input = (float *) nvshmem_malloc (nodesPerPE * dim * sizeof(float));  // NVSHMEM global memory for input embedding.
    /*
    gpuErrchk(cudaMalloc((void**)&d_input, nodesPerPE * dim * sizeof(float))); 
    gpuErrchk(cudaMalloc((void**)&dsp_out, nodesPerPE * hiddenSize * sizeof(float))); 
    gpuErrchk(cudaMalloc((void**)&dsp_out_1, nodesPerPE * outdim * sizeof(float))); 
    */
    
    // buffers for switching between input and output for each layer.
    printf("d_buff_1: %.3f GB\n", (numNodes * 1.0f * max_dim_1 * sizeof(float))/1e9);
    printf("featPerPE: %d, dim: %d\n", numNodes, featmyPE);

    gpuErrchk(cudaMalloc((void**)&d_buff_1, static_cast<size_t>(numNodes) * max_dim_1 * sizeof(float))); 
    gpuErrchk(cudaMalloc((void**)&d_buff_i2h, static_cast<size_t>(numNodes) *hiddenSize * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_buff_h2o, static_cast<size_t>(numNodes) * outdim * sizeof(float)));
    //gpuErrchk(cudaMalloc((void**)&d_buff_2, static_cast<size_t>(numNodes) * hiddenSize * sizeof(float))); 
   
    h_input = (float *) malloc (static_cast<size_t>(numNodes) * max_dim_1 * sizeof(float));                  // CPU host memory (input)
    // h_output = (float *) malloc (nodesPerPE * hiddenSize * sizeof(float));         //  CPU host memory (output)
    // hsp_output_1 = (float *) malloc (nodesPerPE * outdim * sizeof(float));         //  CPU host memory (output)
    std::fill_n(h_input, static_cast<size_t>(numNodes)*featmyPE, 1.0f);                                 // filled with all ones for input embeddings.
    gpuErrchk(cudaMemset(d_buff_i2h, 0, static_cast<size_t>(numNodes) * hiddenSize * sizeof(float)));
    gpuErrchk(cudaMemset(d_buff_h2o, 0, static_cast<size_t>(numNodes)  * outdim * sizeof(float)));
    // std::fill_n(h_output, nodesPerPE*hiddenSize, 0.0f);                         // filled with all zeros for output embeddings.


    /*
    gpuErrchk(cudaMemcpy(d_input, h_input, nodesPerPE * dim * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dsp_out, h_output, nodesPerPE * hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dsp_out_1, hsp_output_1, nodesPerPE * outdim * sizeof(float), cudaMemcpyHostToDevice));
    */

    gpuErrchk(cudaMemcpy(d_buff_1, h_input, static_cast<size_t>(numNodes) * featmyPE * sizeof(float), cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemset(d_buff_2, 0, static_cast<size_t>(numNodes) * hiddenSize * sizeof(float)));
    d_buff_2 = (float *) nvshmem_malloc (static_cast<size_t>(numNodes) * hiddenSize * sizeof(float)); 
    d_hidden_all = (float *) nvshmem_malloc (static_cast<size_t>(numNodes) * hiddenSize * sizeof(float));

    // Initialize the parameters. 
    // d_buf_1 - (dense-1) -> d_buff_nvshmem -> (sp-1) 
    // -> d_buff_2 -> (dense-2) -> d_buff_nvshmem - (sp-2) -> d_buff_2 
    // -> smx-2 -> d_buff_2

    dense_in2hidden* dp1 = new dense_in2hidden("d-1", d_buff_1, d_buff_i2h, numNodes, featmyPE, hiddenSize);
    dense_hidden2out* dp2 = new dense_hidden2out("d-2", dp1->d_out_new, d_buff_h2o, numNodes, hiddenSize, outdim);
    softmax_new_param* smx2 = new softmax_new_param("smx-2", d_buff_1, d_buff_1, numNodes, outdim);
 
    #ifdef validate
    float *h_input_ref, *h_output_ref,  *d_input_ref, *d_output_ref;
    if (mype_node == validate)
    {
        h_input_ref = (float *) malloc (numNodes * dim * sizeof(float));      // CPU host memory (input_ref)
        h_output_ref = (float *) malloc (numNodes * dim * sizeof(float));     //  CPU host memory (output_ref)
        std::fill_n(h_input_ref, numNodes * dim, 1.0f); // filled with all zeros.
        std::fill_n(h_output_ref, numNodes * dim, 0.0f); // filled with all zeros.
        gpuErrchk(cudaMalloc((void**)&d_input_ref, numNodes * dim * sizeof(float))); // GPU device memory (input_ref)
        gpuErrchk(cudaMalloc((void**)&d_output_ref, numNodes * dim * sizeof(float))); // GPU device memory (output_ref)
    }
    #endif

    nidType *d_row_ptr, *d_col_ind, *id_map;
    gpuErrchk(cudaMalloc((void**)&d_row_ptr, ptr_vec.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind, global_col_ind.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&id_map, ID_map.size() * sizeof(nidType)));

    gpuErrchk(cudaMemcpy(d_row_ptr, &ptr_vec[0], ptr_vec.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind, &global_col_ind[0], global_col_ind.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(id_map, &ID_map[0], ID_map.size() * sizeof(nidType), cudaMemcpyHostToDevice));

    #ifdef validate
    int* d_row_ptr_ref, *d_col_ind_ref;
    if (mype_node == validate)
    {
        gpuErrchk(cudaMalloc((void**)&d_row_ptr_ref, global_row_ptr.size()*sizeof(int))); 
        gpuErrchk(cudaMalloc((void**)&d_col_ind_ref, global_col_ind.size()*sizeof(int))); 
        gpuErrchk(cudaMemcpy(d_row_ptr_ref, &global_row_ptr[0], global_row_ptr.size()*sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_col_ind_ref, &global_col_ind[0], global_col_ind.size()*sizeof(int), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_input_ref, h_input_ref, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        gpuErrchk(cudaMemcpy(d_output_ref, h_output_ref, numNodes * dim * sizeof(float), cudaMemcpyHostToDevice));
        
        //
        // Compute the result [lb, ub] based on the whole graph CSR.
        //
        SAG_host_ref(d_output_ref, d_input_ref, 
                    d_row_ptr_ref, d_col_ind_ref, 
                    lb, ub, dim);

        gpuErrchk(cudaMemcpy(h_output_ref, d_output_ref, numNodes * dim * sizeof(float), cudaMemcpyDeviceToHost));
    }
    #endif
    MPI_Barrier(MPI_COMM_WORLD); 

    //
    // Compute on each GPU device.
    //
    // for (int i = 0; i < 10; i++)
    // {
    //     mgg_SAG_np_div(dsp_out, d_input, d_row_ptr_l, d_col_ind_l, d_row_ptr_r, d_col_ind_r,
    //                     lb, ub, dim, nodesPerPE, mype_node, partSize, warpPerBlock, interleaved_dist);
    //     MPI_Barrier(MPI_COMM_WORLD); 
    // }
    
    int num_profiles = 100;
    std::clock_t c_start = std::clock();    
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime(); 

    for (int i = 0; i < num_profiles; i++)
    {
        dense_beg_new_forward(dp1);
        leo_LP_fine_np_div(dp1->d_buff2, dp1->d_out, id_map, d_row_ptr, d_col_ind, hiddenSize, 
                            numNodes, mype_node, partSize, warpPerBlock, num_id_map);
        // MPI_Barrier(MPI_COMM_WORLD); 
        MPI_Barrier(MPI_COMM_WORLD); 
        nvshmem_float_sum_reduce(NVSHMEMX_TEAM_NODE, dp1->d_out_new, dp1->d_buff2, dp1->numNodes * dp1->dim2);

        dense_hidden_new_forward(dp2);
        leo_LP_DP_fine_np_div(d_buff_1, dp2->d_out, id_map, d_row_ptr, d_col_ind, lower_bound, outdim, 
                            numNodes, mype_node, partSize, warpPerBlock, range_id_map);
        // MPI_Barrier(MPI_COMM_WORLD); 
        // MPI_Barrier(MPI_COMM_WORLD); 
        // nvshmem_float_sum_reduce(NVSHMEMX_TEAM_NODE, dp2->d_W_new, dp2->d_W, dp2->dim1*dp2->dim2);
        softmax_new_forward(smx2);
    }
    
    //float *d_buff_2_out;
    //d_buff_2_out = (float *) malloc ((nodesPerPE) * hiddenSize * sizeof(float));
    //gpuErrchk(cudaMemcpy(d_buff_2_out, d_buff_2, nodesPerPE * hiddenSize * sizeof(float), cudaMemcpyDeviceToHost));


    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / num_profiles;
    printf("PE-%d, Total (ms): %.3f\n", mype_node, time_elapsed_ms);
    MPI_Barrier(MPI_COMM_WORLD); 
    t2 = MPI_Wtime(); 
    if (mype_node == 0) printf( "MPI time (ms) %.3f\n", (t2 - t1)*1e3/num_profiles); 
    
    // gpuErrchk(cudaMemcpy(h_output, dsp_out, nodesPerPE*dim*sizeof(float), cudaMemcpyDeviceToHost));

    #ifdef validate
    if (mype_node == validate){
        for (int nid = 0; nid < 10; nid++){
            printf("out [%d] ", nid);
            for (int d = 0; d < 5; d++){
                printf("%.3f,", h_output[nid * dim + d]);
            }
            printf("\n");
        }
        printf("==============================\n");
        for (int nid = 0; nid < 10; nid++){
            printf("ref [%d] ", nid);
            for (int d = 0; d < 5; d++){
                printf("%.3f,", h_output_ref[lb * dim + nid * dim + d]);
            }
            printf("\n");
        }
        bool val_status = check_equal(h_output_ref, h_output, (ub - lb) * dim, dim, lb * dim);
        printf("Validation on PE-{%d}, status: ", validate);
        if (val_status) printf("True\n"); else printf("False\n");
    }
    #endif

    // release memory.
    // cudaFree(dsp_out);
    cudaFree(d_row_ptr);
    cudaFree(d_col_ind);
    // cudaFree(d_input);
    // cudaDeviceReset();
    nvshmem_finalize();

    // free(h_input);
    // free(h_output);

    MPI_Finalize();

    #ifdef validate
    if (mype_node == validate){
        cudaFree(d_output_ref);
        free(h_output_ref);
    }
    #endif


    if (mype_node == 0) 
        printf("===================================\n");

    return 0;
}
