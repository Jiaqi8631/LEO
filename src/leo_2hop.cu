
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
#include "layer_all.cuh"
#include "vector_func.h"
#include "neighbor.h"
#include "LSH.h"

// #define validate 1 // the number (< num_GPUs) indicates the validation on which PE.
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
    const char *beg_file = argv[1];
	const char *csr_file = argv[2];
	const char *weight_file = argv[3];
    int num_GPUs = atoi(argv[4]);           // 2
    int partSize = atoi(argv[5]);           // 32
    int warpPerBlock = atoi(argv[6]);       // 4
    int interleaved_dist = atoi(argv[7]);   // 2
    int dim = atoi(argv[8]);                // 16
    int hiddenSize = atoi(argv[9]);
    int outdim = atoi(argv[10]);
    int grain = atoi(argv[11]);

    int max_dim = max({hiddenSize, outdim});
    int max_dim_1 = max({dim, outdim});

    graph<long, long, nidType, nidType, nidType, nidType>* ginst = new graph<long, long, nidType, nidType, nidType, nidType>(beg_file, csr_file, weight_file);
    std::vector<nidType> global_row_ptr(ginst->beg_pos, ginst->beg_pos + ginst->vert_count + 1);
    std::vector<nidType> global_col_ind(ginst->csr, ginst->csr + ginst->edge_count);

    cout << "Complete loading graphs !!" << endl;
    nidType numNodes = global_row_ptr.size() - 1;
    nidType numEdges = global_col_ind.size();    


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

    std::clock_t c_start_proc = std::clock();    
    // Divide the CSR into the local and remote for each GPU.
    auto split_output = partition_1hop<nidType>(global_row_ptr, global_col_ind, mype_node, num_GPUs);
    
    std::clock_t c_end_proc = std::clock();
    float preproc_time_elapsed_ms = 1000.0 * (c_end_proc - c_start_proc) / CLOCKS_PER_SEC;
    if (mype_node == 0)
    printf("Preproc (ms): %.3f\n", preproc_time_elapsed_ms);

    // printf("lb: %d, ub: %d\n", lb, ub);

    auto ptr_1hop = split_output[0];
    auto col_idx_1hop = split_output[1];
    std::vector<nidType> col_uni_1hop = col_idx_1hop;

    unique_vector<nidType>(col_uni_1hop);
    auto col_idx_2hop = partition_2hop<nidType>(global_row_ptr, global_col_ind, col_uni_1hop);
    std::vector<nidType> col_uni_2hop = col_idx_2hop;
    unique_vector<nidType>(col_uni_2hop);

    std::vector<nidType> rever_col = Reverse_col<nidType>(col_idx_1hop, col_uni_1hop, col_uni_1hop.back());

    auto bound = split_output[2];
    auto remap_idx = split_output[3];
    nidType lb_1hop = bound[mype_node];
    nidType ub_1hop = bound[mype_node + 1];
    Remap<nidType>(col_uni_1hop, remap_idx);
    Remap<nidType>(col_uni_2hop, remap_idx);
    std::vector<nidType> col_uni_local = create_local_vec<nidType>(lb_1hop, ub_1hop);
    cout << "The size of 1-HOP neighbor is " << col_uni_1hop.size() << endl;
    cout << "The size of 2-HOP neighbor is " << col_uni_2hop.size() << endl;

    printf("myID is %d\n", getpid());  // 
    int j=1;
    while(j){
      sleep(2);  //
    }

    auto ptr = split_output[0];
    auto col_idx = split_output[1];
    nidType lb = bound[mype_node];
    nidType ub = bound[mype_node+1];
    // printf("PE[%d]. local: %d, remote: %d\n", mype_node, local_col_idx_vec.size(), remote_col_idx_vec.size());

    //int grain = 1024;
    auto Fine_out = split_Fine<nidType>(ptr, col_idx, mype_node, grain);
    auto ptr_vec = Fine_out[0];
    auto ID_map = Fine_out[1];
    nidType num_id = ID_map.size();

    // Allocate memory on each device.
    float *h_input, *h_buff_i2h, *h_buff_h2o;
    float *d_buff_1, *d_buff_2, *d_buff_i2h, *d_buff_h2o;

    // d_input = (float *) nvshmem_malloc (nodesPerPE * dim * sizeof(float));  // NVSHMEM global memory for input embedding.
    /*
    gpuErrchk(cudaMalloc((void**)&d_input, nodesPerPE * dim * sizeof(float))); 
    gpuErrchk(cudaMalloc((void**)&dsp_out, nodesPerPE * hiddenSize * sizeof(float))); 
    gpuErrchk(cudaMalloc((void**)&dsp_out_1, nodesPerPE * outdim * sizeof(float))); 
    */
    
    nidType nodesmyPE = col_uni_1hop.size();
    // buffers for switching between input and output for each layer.
    printf("d_buff_1: %.3f GB\n", (nodesmyPE * 1.0f * max_dim_1 * sizeof(float))/1e9);
    printf("nodesPerPE: %d, dim: %d\n", nodesmyPE, dim);
    gpuErrchk(cudaMalloc((void**)&d_buff_1, static_cast<size_t>(nodesmyPE) * max_dim_1 * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_buff_2, static_cast<size_t>(nodesmyPE) * hiddenSize * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_buff_i2h, static_cast<size_t>(nodesmyPE) * hiddenSize * sizeof(float)));
    gpuErrchk(cudaMalloc((void**)&d_buff_h2o, static_cast<size_t>(nodesmyPE) * outdim * sizeof(float)));
   
    h_input = (float *) malloc (static_cast<size_t>(nodesmyPE) * max_dim_1 * sizeof(float));                  // CPU host memory (input)
    h_buff_i2h = (float*)malloc(static_cast<size_t>(nodesmyPE) * hiddenSize * sizeof(float));
    h_buff_h2o = (float*)malloc(static_cast<size_t>(nodesmyPE) * outdim * sizeof(float));
    // h_output = (float *) malloc (nodesPerPE * hiddenSize * sizeof(float));         //  CPU host memory (output)
    // hsp_output_1 = (float *) malloc (nodesPerPE * outdim * sizeof(float));         //  CPU host memory (output)
    std::fill_n(h_input, static_cast<size_t>(nodesmyPE)*dim, 1.0f);   
    std::fill_n(h_buff_i2h, static_cast<size_t>(nodesmyPE) * hiddenSize, 0);
    std::fill_n(h_buff_h2o, static_cast<size_t>(nodesmyPE) * outdim, 0);
    // filled with all ones for input embeddings.
    // std::fill_n(h_output, nodesPerPE*hiddenSize, 0.0f);                         // filled with all zeros for output embeddings.
    
     /*
    gpuErrchk(cudaMemcpy(d_input, h_input, nodesPerPE * dim * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dsp_out, h_output, nodesPerPE * hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(dsp_out_1, hsp_output_1, nodesPerPE * outdim * sizeof(float), cudaMemcpyHostToDevice));
    */

    gpuErrchk(cudaMemcpy(d_buff_1, h_input, static_cast<size_t>(nodesmyPE) * dim * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemset(d_buff_2, 0, static_cast<size_t>(nodesmyPE) * hiddenSize * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_buff_i2h, h_buff_i2h, static_cast<size_t>(nodesmyPE) * hiddenSize * sizeof(float), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_buff_h2o, h_buff_h2o, static_cast<size_t>(nodesmyPE) * outdim * sizeof(float), cudaMemcpyHostToDevice));
    //gpuErrchk(cudaMemset(d_buff_i2h, 0, static_cast<size_t>(nodesmyPE) * hiddenSize * sizeof(float)));
    //gpuErrchk(cudaMemset(d_buff_h2o, 0, static_cast<size_t>(nodesmyPE) *outdim * sizeof(float)));
    //d_buff_nvshmem = (float *) nvshmem_malloc (static_cast<size_t>(nodesmyPE) * hiddenSize * sizeof(float));

    // Initialize the parameters. 
    // d_buf_1 - (dense-1) -> d_buff_nvshmem -> (sp-1) 
    // -> d_buff_2 -> (dense-2) -> d_buff_nvshmem - (sp-2) -> d_buff_2 
    // -> smx-2 -> d_buff_2
    dense_in2hidden* dp1 = new dense_in2hidden("d-1", d_buff_1, d_buff_i2h, nodesmyPE, dim, hiddenSize);
    dense_hidden2out* dp2 = new dense_hidden2out("d-2", d_buff_i2h, d_buff_h2o, nodesmyPE, hiddenSize, outdim);
    softmax_out* smx2 = new softmax_out("smx-2", d_buff_1, d_buff_1, nodesmyPE, outdim);
 
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

    nidType* d_ptr, * d_col_idx, * bound_ptr, * id_map, *rev_col;
    gpuErrchk(cudaMalloc((void**)&d_ptr, ptr_vec.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&d_col_idx, col_idx.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&bound_ptr, bound.size() * sizeof(nidType)));
    gpuErrchk(cudaMalloc((void**)&id_map, ID_map.size() * sizeof(nidType)));
    gpuErrchk(cudaMalloc((void**)&rev_col, rever_col.size() * sizeof(nidType)));

    gpuErrchk(cudaMemcpy(d_ptr, &ptr_vec[0], ptr_vec.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_idx, &col_idx[0], col_idx.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(bound_ptr, &bound[0], bound.size() * sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(id_map, &ID_map[0], ID_map.size() * sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(rev_col, &rever_col[0], rever_col.size() * sizeof(nidType), cudaMemcpyHostToDevice));
    //P_col <<<16, 32 >>> (d_col_idx, mype_node);
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
       // mgg_SAG_np_div(d_buff_2, dp1->d_out, d_row_ptr_l, d_col_ind_l, d_row_ptr_r, d_col_ind_r,
       //                 lb, ub, hiddenSize, nodesmyPE, mype_node, partSize, warpPerBlock, interleaved_dist);
        leo_fine_np_div(d_buff_2, dp1->d_out, id_map, rev_col, d_ptr, d_col_idx, 
            lb, ub, hiddenSize, nodesmyPE, mype_node, partSize, warpPerBlock, num_id);
        // MPI_Barrier(MPI_COMM_WORLD); 
        // MPI_Barrier(MPI_COMM_WORLD); 
        // nvshmem_float_sum_reduce(NVSHMEMX_TEAM_NODE, dp2->d_W_new, dp2->d_W, dp2->dim1*dp2->dim2);

        dense_hidden_new_forward(dp2);
        //mgg_SAG_np_div(d_buff_1, dp2->d_out, d_row_ptr_l, d_col_ind_l, d_row_ptr_r, d_col_ind_r, 
       //                 lb, ub, hiddenSize, nodesmyPE, mype_node, partSize, warpPerBlock, interleaved_dist);
        leo_fine_np_div(d_buff_1, dp2->d_out, id_map, rev_col, d_ptr, d_col_idx,
            lb, ub, outdim, nodesmyPE, mype_node, partSize, warpPerBlock, num_id);
        // MPI_Barrier(MPI_COMM_WORLD); 
        // MPI_Barrier(MPI_COMM_WORLD); 
        // nvshmem_float_sum_reduce(NVSHMEMX_TEAM_NODE, dp2->d_W_new, dp2->d_W, dp2->dim1*dp2->dim2);
        softmax_new_forward(smx2);
    }

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
    cudaFree(d_ptr);
    cudaFree(d_col_idx);
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
