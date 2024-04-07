
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
    int pt_ratio = atoi(argv[12]);
    int num_profiles = atoi(argv[13]);
    float interval = atof(argv[14]);

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
    //nidType numNodes = global_row_ptr.size() - 1;
    //nidType numEdges = global_col_ind.size();    


    auto e_bound = Simple_Edge_cut<nidType>(global_row_ptr, numNodes, numEdges, num_GPUs);
    //vector<nidType> e_bound = {0, 178198, 946334, 2306315, 4203323};
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
    // printf("numNodes: %d, nodesPerPE: %d\n", numNodes, nodesPerPE);
    nidType featlb = featdPerPE * mype_node;
    nidType featub = (featlb + featdPerPE) < dim? (featlb + featdPerPE) : dim;
    nidType featmyPE = featub - featlb;
    int max_dim = max({hiddenSize, outdim});
    int max_dim_1 = max({featmyPE, outdim});

    auto Fine_out = split_Fine<nidType>(global_row_ptr, global_col_ind, mype_node, grain);
    auto ptr_vec = Fine_out[0];
    auto ID_map = Fine_out[1];
    nidType num_id_map = ID_map.size();

    nidType e_lb = e_bound[mype_node];
    nidType e_ub = e_bound[mype_node + 1];
    auto Find_result = Find_ID_map<nidType>(ID_map, e_lb, e_ub);
    auto range_id_map = Find_result.first;
    auto lower_bound  = Find_result.second;
    printf("PE-%d: range_id_map: %d\n", mype_node, range_id_map);

    float *h_input;
    float *d_buff_1, *d_buff_2;
   
    // buffers for switching between input and output for each layer.
    printf("d_buff_1: %.3f GB\n", (numNodes * 1.0f * outdim * sizeof(float))/1e9);
    printf("featPerPE: %d, dim: %d\n", numNodes, featmyPE);

    gpuErrchk(cudaMalloc((void**)&d_buff_1, static_cast<size_t>(numNodes) * outdim * sizeof(float))); 
    gpuErrchk(cudaMalloc((void**)&d_buff_2, static_cast<size_t>(numNodes) *outdim * sizeof(float)));
   
    h_input = (float *) malloc (static_cast<size_t>(numNodes) * outdim * sizeof(float));                  // CPU host memory (input)
    std::fill_n(h_input, static_cast<size_t>(numNodes) * outdim, 1.0f);                                 // filled with all ones for input embeddings.
    gpuErrchk(cudaMemset(d_buff_2, 0, static_cast<size_t>(numNodes) * outdim * sizeof(float)));
    gpuErrchk(cudaMemcpy(d_buff_1, h_input, static_cast<size_t>(numNodes) * outdim * sizeof(float), cudaMemcpyHostToDevice));
 

    nidType *d_row_ptr, *d_col_ind, *id_map;
    gpuErrchk(cudaMalloc((void**)&d_row_ptr, ptr_vec.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&d_col_ind, global_col_ind.size()*sizeof(nidType))); 
    gpuErrchk(cudaMalloc((void**)&id_map, ID_map.size() * sizeof(nidType)));

    gpuErrchk(cudaMemcpy(d_row_ptr, &ptr_vec[0], ptr_vec.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_col_ind, &global_col_ind[0], global_col_ind.size()*sizeof(nidType), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(id_map, &ID_map[0], ID_map.size() * sizeof(nidType), cudaMemcpyHostToDevice));
    float *time_rec, *time_send;
    time_rec = new float[num_GPUs];
    time_send = new float[1];
    nidType step = numNodes/pt_ratio; 
    
    MPI_Barrier(MPI_COMM_WORLD); 

    //int num_profiles = 250;
    std::clock_t c_start = std::clock();    
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime(); 

    for (int i = 0; i < num_profiles; i++)
    {
        time_send[0] = leo_LP_DP_fine_np_div_pretrain(d_buff_1, d_buff_2, id_map, d_row_ptr, d_col_ind,  lower_bound, outdim, 
                            numNodes, mype_node, partSize, warpPerBlock, range_id_map);
        MPI_Barrier(MPI_COMM_WORLD); 
        MPI_Allgather(time_send, 1, MPI_FLOAT, time_rec, 1, MPI_FLOAT, MPI_COMM_WORLD);
        Dynamic_Balanced_cut<nidType>(time_rec, e_bound, num_GPUs, step, interval);
        e_lb = e_bound[mype_node];
        e_ub = e_bound[mype_node + 1];
        Find_result = Find_ID_map<nidType>(ID_map, e_lb, e_ub);
        range_id_map = Find_result.first;
        lower_bound  = Find_result.second;
    }

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / num_profiles;
    printf("PE-%d, Total (ms): %.3f\n", mype_node, time_elapsed_ms);
    MPI_Barrier(MPI_COMM_WORLD); 
    t2 = MPI_Wtime(); 
    if (mype_node == 0) printf( "MPI time (ms) %.3f\n", (t2 - t1)*1e3/num_profiles); 
    
    // gpuErrchk(cudaMemcpy(h_output, dsp_out, nodesPerPE*dim*sizeof(float), cudaMemcpyDeviceToHost));
    if (mype_node == 0){
        string output_prefix = "dataset/preprocess/";
        string output_suffix = "_ebound.txt";
        string grain_suffix  = "_grain";
        string grain_str     = to_string(grain);
        string output_file = output_prefix + data_name + grain_suffix + grain_str + output_suffix;
        ofstream output(output_file, ios::app);
        if (output.is_open()){
            ostream_iterator<nidType> oit(output, "\n");
            copy(e_bound.begin(), e_bound.end(), oit);
            output.close();
            cout << "The output file has been written to " << output_file << "!" << endl;
        }
        else{
            throw std::runtime_error("Cannot open output file!");
        }
    }
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
