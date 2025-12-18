#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include "include/model.h"

using namespace std;
using namespace GNNPro_lib::common;


int main(int argc, char* argv[]){
    //MPI Initialization
    int rank, nranks;
    cudaStream_t stream;
    nvshmemx_init_attr_t nv_attr;
    MPI_Comm mpi_comm = MPI_COMM_WORLD;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    nv_attr.mpi_comm = &mpi_comm;

    //NVSHMEM Initialization
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &nv_attr);
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    cudaSetDevice(mype_node);
    cudaStreamCreate(&stream);

    Graph *graph = new Graph(nranks, mype_node);

    graph->config->ReadFromConfig(argv[1]);
    graph->config->GetFileName();

    graph->Config_Initial(); 
    graph->Load();
    MPI_Barrier(MPI_COMM_WORLD);
    Model* model = nullptr;
    if(mype_node == 0) LOG(ERROR) << "Creating Model";
    model = BuildModel(model, graph);

    model->Train();
    model->Validate();

    MPI_Barrier(MPI_COMM_WORLD);
    graph->Free_Device_Memory();
    
    //clean up
    delete graph;
    delete model;
    //nvshmem_finalize();
    MPI_Finalize();
    if (mype_node == 0)
        printf("==============Finished!==============\n");

    return 0;
}
    
