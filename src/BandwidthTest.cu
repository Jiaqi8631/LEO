
#include <iostream>
#include <stdio.h>
#include <ctime>
#include <algorithm>

#include <mpi.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cublas_v2.h>
#include <vector>


//using namespace cudl;
using namespace std;
#define P2P 1

#define cudaCheckError()                                       \
  {                                                            \
    cudaError_t e = cudaGetLastError();                        \
    if (e != cudaSuccess) {                                    \
      printf("Cuda failure %s:%d: '%s'\n", __FILE__, __LINE__, \
             cudaGetErrorString(e));                           \
      exit(EXIT_FAILURE);                                      \
    }                                                          \
  }

int main(int argc, char* argv[]){
	
    int numNodes = atoi(argv[1]);
    int hiddenSize = atoi(argv[2]);
    int num_GPUs = atoi(argv[3]);

    #ifdef P2P
    int repeat = atoi(argv[4]);
    #endif
    
    double t1, t2; 
    // print_array<int>("global_row_ptr", global_row_ptr, global_row_ptr.size());
    // print_array<int>("global_col_ind", global_col_ind, global_col_ind.size());
    int rank, nranks;
    
    nvshmemx_init_attr_t attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    attr.mpi_comm = &mpi_comm;

    // Set up NVSHMEM device.
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);

    #ifdef nv_test
    cudaStream_t stream;
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);
  
    //nvshmem test
    float *Buff_1, *Buff_2;
    Buff_1 = (float *) nvshmem_malloc (static_cast<size_t>(numNodes) * hiddenSize * sizeof(float)); 
    Buff_2 = (float *) nvshmem_malloc (static_cast<size_t>(numNodes) * hiddenSize * sizeof(float)); 
    #endif

    #ifdef P2P
    //int repeat = 10;
    float *buffers;
    std::vector<cudaEvent_t> start(num_GPUs);
    std::vector<cudaEvent_t> stop(num_GPUs);
    std::vector<cudaStream_t> stream(num_GPUs);

    cudaSetDevice(mype_node);
    
    cudaMalloc(&buffers, numNodes * hiddenSize * sizeof(float));
    cudaCheckError();
    cudaMemset(buffers, 0, numNodes * hiddenSize * sizeof(float));
    cudaCheckError();

    for(int d = 0; d < num_GPUs; d++){
        cudaStreamCreateWithFlags(&stream[d], cudaStreamNonBlocking);
        cudaEventCreate(&start[d]);
        cudaCheckError();
        cudaEventCreate(&stop[d]);
        cudaCheckError();
    }

    #endif

    #ifdef nv_test
    int num_profiles = 100;
    std::clock_t c_start = std::clock();    
    MPI_Barrier(MPI_COMM_WORLD);
    t1 = MPI_Wtime(); 

    for (int i = 0; i < num_profiles; i++)
    {    
        nvshmem_float_sum_reduce(NVSHMEMX_TEAM_NODE, Buff_2, Buff_1, numNodes * hiddenSize);    
    }

    std::clock_t c_end = std::clock();
    float time_elapsed_ms = 1000.0 * (c_end-c_start) / CLOCKS_PER_SEC / num_profiles;
    printf("PE-%d, Total (ms): %.3f\n", mype_node, time_elapsed_ms);
    MPI_Barrier(MPI_COMM_WORLD); 
    t2 = MPI_Wtime(); 
    if (mype_node == 0) printf( "MPI time (ms) %.3f\n", (t2 - t1)*1e3/num_profiles); 
    #endif

    #ifdef P2P


    cudaSetDevice(mype_node);

    for (int j = 0; j < num_GPUs; j++) {
        cudaStreamSynchronize(stream[j]);
        cudaCheckError();

        cudaEventRecord(start[j], stream[j]);
        cudaCheckError();
        if (mype_node != j){
                for (int r = 0; r < repeat; r++) {
                    cudaMemcpyPeerAsync(buffers, j, buffers, mype_node,
                          sizeof(float) * numNodes * hiddenSize, stream[j]);
                }
        }
        cudaEventRecord(stop[j], stream[j]);
        cudaCheckError();

        cudaStreamSynchronize(stream[j]);
        cudaCheckError();

        float time_ms;
        cudaEventElapsedTime(&time_ms, start[j], stop[j]);  
        printf("PE-%d P2P %d-%d time: %5f\n", mype_node, mype_node, j,  time_ms/repeat);   
    }

    #endif

    // release memory.
    // cudaFree(dsp_out);

    // cudaFree(d_input);
    // cudaDeviceReset();
    nvshmem_finalize();

    // free(h_input);
    // free(h_output);

    MPI_Finalize();




    if (mype_node == 0) 
        printf("===================================\n");

    return 0;
}
