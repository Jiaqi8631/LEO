#include <cblas.h>
#include "model.h"
#include "aggregate.cuh"

namespace GNNPro_lib{
namespace common{

#define WARP_SIZE 32

void Model::ComputeLoss(){
    int block = 4 * WARP_SIZE;
    int grid = (local_v_num + block - 1) / block;

    CrossEntropyLoss<<<grid, block>>>(label, buff2_s, loss, local_v_num, out_dim);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @CrossEntropyLoss: %s\n", cudaGetErrorString(error));
    }
}

//======================================== GCN =================================================

void GCN_Model::Train(){
    Timer t0;
    //Forward();
    //Forward_test1();
    Forward_Option();
    double infer_time = t0.Passed();
    std::cout << "Rank " << mynode << ": Inference time for GCN model =====> " << infer_time << "s" << std::endl;

    //ComputeLoss();
    
    Timer t1;
    //Backward();
    //Backward_Option();
    double back_time = t1.Passed();
    std::cout << "Rank " << mynode << ": Backpropagation time for GCN model =====> " << back_time << "s" << std::endl;

    // Timer t1;
    // Backward_Stream();
    // double back_time = t1.Passed();
    // std::cout << "Rank " << mynode << ": Backpropagation time for GCN model (using CUDA streams) =====> " << back_time << "s" << std::endl;
}


void GCN_Model::Forward(){
    //1st layer GCN
    Timer t_mm;
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(cache_feature, W1_c, buff1_c, int(cache_v_num), in_dim, hid_dim);
    double mm_time = t_mm.PassedMilli();
    std::cout << "Rank " << mynode << ": Matrix multiplication time in the 1st layer ======> " << mm_time << "ms" << std::endl;

    CUDA_CALL(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    Timer t0;
    Aggregate(buff1, buff1_l, buff1_c, buff1_h, hid_dim);
    double aggre_time = t0.PassedMilli();
    std::cout << "Rank " << mynode <<": Aggregation time in the 1st layer ======> " << aggre_time << "ms" << std::endl; 

    Timer t1;
    //Update locally cached data
    if(CacheFlag == 1) UpdateCache(update_cache_feature, buff1, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time = t1.PassedMilli();
    std::cout << "Rank " << mynode <<": Updating cached feature time in the 1st layer ======> " << update_time << "ms" << std::endl;    
    
    //2nd layer GCN
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, out_dim);
    if (CacheFlag == 1) gemm_cublas(update_cache_feature, W2_c, buff2_c, int(cache_v_num), hid_dim, out_dim);

    CUDA_CALL(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    Timer t2;
    Aggregate(buff2, buff2_l, buff2_c, buff2_h, out_dim);
    double aggre_time2 = t2.PassedMilli();
    std::cout << "Rank " << mynode << ": Aggregation time in the 2nd layer ======> " << aggre_time2 << "ms" << std::endl;

    softmax_forward_cudnn(buff2_s, buff2, int(local_v_num), out_dim);
}

void GCN_Model::Forward_test1(){
#ifdef test
    Timer t_test;
    gemm_cblas(W1_h, cache_host_feature, buff1_h, hid_dim, int(host_cache_v_num), in_dim);
    double test_time = t_test.PassedMilli();
    std::cout << "Rank " << mynode << ": Matrix multiplication test time ======> " << test_time << "ms" << std::endl;
#endif
    //std::cout << "Rank " << mynode << ": Cache Flag ======> " << CacheFlag << std::endl;
    Timer t0;
    std::thread cpu_thread;
    if(CacheFlag == 1){
        cpu_thread = std::thread(gemm_cblas, W1_h, cache_host_feature, buff1_h, hid_dim, int(host_cache_v_num), in_dim);
    }
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    if(CacheFlag == 1){
        //cpu_thread.join();
        gemm_cublas(cache_feature, W1_c, buff1_c, int(cache_v_num), in_dim, hid_dim);
        cpu_thread.join();
    } 

    double mm_time = t0.PassedMilli();
    std::cout << "Rank " << mynode << ": Matrix multiplication time in the 1st layer ======> " << mm_time << "ms" << std::endl;
    CUDA_CALL(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    Timer t1;
    Aggregate_Warp(buff1, buff1_l, buff1_c, buff1_h, hid_dim);
    double aggre_time = t1.PassedMilli();
    std::cout << "Rank " << mynode <<": Aggregation time in the 1st layer ======> " << aggre_time << "ms" << std::endl;

    Timer t2;
    //Update locally cached data
    if(CacheFlag == 1) UpdateCache(update_cache_feature, buff1, cache_list, cache_v_num, hid_dim, nullptr);
    //Update cached data on the host side
    if(UVMFlag == 1)  UpdateCache(update_cache_host_featrue, buff1, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time = t2.PassedMilli();
    std::cout << "Rank " << mynode << ": Updating cached feature time in the 1st layer ======> " << update_time << "ms" << std::endl;

    //2nd layer GCN
    Timer t3;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, out_dim);
    if(CacheFlag == 1) gemm_cublas(update_cache_feature, W2_c, buff2_c, int(cache_v_num), hid_dim, out_dim); 
    if(UVMFlag == 1) gemm_cublas(update_cache_host_featrue, W2_h, buff2_h, int(host_cache_v_num), hid_dim, out_dim);
    double mm_time1 = t3.PassedMilli();
    std::cout << "Rank " << mynode << ": Matrix multiplication time in the 2nd layer ======> " << mm_time1 << "ms" << std::endl;
}

void GCN_Model::Forward_Option(){
    if(mynode == 0) printf("| Rank |   Stage  | layer |      Operator      |    Time    \n");
    if(CacheFlag == true){
        if(UVMFlag == true){
            Forward_wCPUGPU_Cache();
            //Forward_wCPUGPU_Cache_Stream();
        }else{
            Forward_wGPU_Cache();
        }
    }else{
        if(RemoteFlag == true){
            Forward_wo_Cache();
        }else{
            Forward_LocalOnly();
        }
    }
}

void GCN_Model::Forward_wCPUGPU_Cache(){
    Timer t0;
    //std::thread cpu_thread = std::thread(gemm_cblas, W1_h, cache_host_feature, buff1_h, hid_dim, int(host_cache_v_num), in_dim);
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    gemm_cblas(W1_h, cache_host_feature, buff1_h, hid_dim, int(host_cache_v_num), in_dim);
    gemm_cublas(cache_feature, W1_c, buff1_c, int(cache_v_num), in_dim, hid_dim);
    //cpu_thread.join();
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |   GEMM(CPU, GPU)   |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    Timer t1;
    //Aggregate_Warp
    leo_lchr_warp_V1(buff1, buff1_l, buff1_c, buff1_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
    double aggre_time = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |   Aggregate(lchr)  |  " << aggre_time << "ms  "<< std::endl;

    Timer t2;
    //Update locally cached data
    UpdateCache(update_cache_feature, buff1, cache_list, cache_v_num, hid_dim, nullptr);
    //Update cached data on the host side 
    UpdateCache(update_cache_host_featrue, buff1, cache_host_list, host_cache_v_num, hid_dim, nullptr); 
    double update_time = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |    Update Cache    |  " << update_time << "ms  "<< std::endl;
    
    Timer t3;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, out_dim);
    gemm_cublas(update_cache_feature, W2_c, buff2_c, int(cache_v_num), hid_dim, out_dim);
    gemm_cublas(update_cache_host_featrue, W2_h, buff2_h, int(host_cache_v_num), hid_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize()); 
    double mm_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |   GEMM(CPU, GPU)   |  " << mm_time1 << "ms  "<< std::endl;      

    MPI_Barrier(MPI_COMM_WORLD);

    Timer t4;
    //Aggregate_Warp
    leo_lchr_warp_V1(buff2, buff2_l, buff2_c, buff2_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                    row_ptr_h, col_ind_h, id_map, local_v_num, out_dim, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
    double aggre_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |   Aggregate(lchr)  |  " << aggre_time2 << "ms  "<< std::endl;

    softmax_forward_cudnn(buff2_s, buff2, int(local_v_num), out_dim);
}

void GCN_Model::Forward_wCPUGPU_Cache_Stream(){
    cudaEvent_t Aggre;
    //Create CUDA streams
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    cudaEventCreate(&Aggre);

    Timer t0;
    gemm_cublas_Stream(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim, stream1);
    gemm_cblas(W1_h, cache_host_feature, buff1_h, hid_dim, int(host_cache_v_num), in_dim);
    gemm_cublas_Stream(cache_feature, W1_c, buff1_c, int(cache_v_num), in_dim, hid_dim, stream2);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |   GEMM(CPU, GPU)   |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    Timer t1;
    //Aggregate_Warp
    leo_lchr_warp_V1(buff1, buff1_l, buff1_c, buff1_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, mynode, stream1);
    double aggre_time = t1.PassedMilli(); 
    cudaEventRecord(Aggre, stream1);
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |   Aggregate(lchr)  |  " << aggre_time << "ms  "<< std::endl;       

    //Event 1 (first aggregation) is a dependency on streams 2, 3, and 4.
    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream2, Aggre, 0);
    cudaStreamWaitEvent(stream3, Aggre, 0);
    cudaStreamWaitEvent(stream4, Aggre, 0);    

    Timer t2;
    //stream2
    UpdateCache_Stream(update_cache_feature, buff1, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_Stream(update_cache_feature, W2_c, buff2_c, int(cache_v_num), hid_dim, out_dim, stream2);

    //stream3
    UpdateCache_Stream(update_cache_host_featrue, buff1, cache_host_list, host_cache_v_num, hid_dim, stream3);
    gemm_cublas_Stream(update_cache_host_featrue, W2_h, buff2_h, int(host_cache_v_num), hid_dim, out_dim, stream3); 

    //stream4
    gemm_cublas_Stream(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, out_dim, stream4);

    //Sync all CUDA streams across the device
    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @Forward_stream: %s\n", cudaGetErrorString(error));
    } 
    double forward_stream_time = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     Update&GEMM    |  " << forward_stream_time << "ms  "<< std::endl;    

    MPI_Barrier(MPI_COMM_WORLD);

    Timer t3;
    //Aggregate_Warp
    leo_lchr_warp_V1(buff2, buff2_l, buff2_c, buff2_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                    row_ptr_h, col_ind_h, id_map, local_v_num, out_dim, nodePerPE, RunConfig::warpPerBlock, mynode, stream1);
    double aggre_time2 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |   Aggregate(lchr)  |  " << aggre_time2 << "ms  "<< std::endl;      
 
    softmax_forward_cudnn_Stream(buff2_s, buff2, int(local_v_num), out_dim, stream1);

    //clean up
    cudaEventDestroy(Aggre);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);    
}

void GCN_Model::Forward_wGPU_Cache(){
    Timer t0;
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    gemm_cublas(cache_feature, W1_c, buff1_c, int(cache_v_num), in_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |     GEMM(GPU)      |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    Timer t1;
    //Aggregate_Warp
    leo_lcr_warp_v1(buff1, buff1_l, buff1_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                    id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
    double aggre_time = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |   Aggregate(lcr)   |  " << aggre_time << "ms  "<< std::endl;

    Timer t2;
    //Update locally cached data
    UpdateCache(update_cache_feature, buff1, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |    Update Cache    |  " << update_time << "ms  "<< std::endl;

    Timer t3;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, out_dim);
    gemm_cublas(update_cache_feature, W2_c, buff2_c, int(cache_v_num), hid_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize()); 
    double mm_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     GEMM(GPU)      |  " << mm_time1 << "ms  "<< std::endl;     

    MPI_Barrier(MPI_COMM_WORLD);

    Timer t4;
    //Aggregate_warp
    leo_lcr_warp_v1(buff2, buff2_l, buff2_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                    id_map, local_v_num, out_dim, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
    double aggre_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |   Aggregate(lcr)   |  " << aggre_time2 << "ms  "<< std::endl;    

    softmax_forward_cudnn(buff2_s, buff2, int(local_v_num), out_dim);
}

void GCN_Model::Forward_wo_Cache(){
    Timer t0;
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |     GEMM(GPU)      |  " << mm_time << "ms  " << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    Timer t1;
    //Aggregate_Warp
    leo_lr_warp(buff1, buff1_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, \
                id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
    double aggre_time = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |    Aggregate(lr)   |  " << aggre_time << "ms  "<< std::endl; 

    //2nd layer GCN
    Timer t2;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     GEMM(GPU)      |  " << mm_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    Timer t3;
    //Aggregate_warp
    leo_lr_warp(buff2, buff2_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, \
                id_map, local_v_num, out_dim, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr); 
    double aggre_time2 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |    Aggregate(l)    |  " << aggre_time2 << "ms  "<< std::endl;       

    softmax_forward_cudnn(buff2_s, buff2, int(local_v_num), out_dim);
}

void GCN_Model::Forward_LocalOnly(){
    Timer t0;
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |     GEMM(GPU)      |  " << mm_time << "ms  " << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    Timer t1;
    //Aggregate_Warp
    leo_l_warp(buff1, buff1_l, row_ptr_l, col_ind_l, id_map, local_v_num, \
                hid_dim, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
    double aggre_time = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |    Aggregate(l)    |  " << aggre_time << "ms   "<< std::endl;

    //2nd layer GCN
    Timer t2;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     GEMM(GPU)      |  " << mm_time1 << "ms  "<< std::endl; 

    MPI_Barrier(MPI_COMM_WORLD);

    Timer t3;
    //Aggregate_Warp
    leo_l_warp(buff2, buff2_l, row_ptr_l, col_ind_l, id_map, local_v_num, \
                out_dim, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
    double aggre_time2 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |    Aggregate(l)    |  " << aggre_time2 << "ms  "<< std::endl;    

    softmax_forward_cudnn(buff2_s, buff2, int(local_v_num), out_dim);
}

void GCN_Model::Aggregate(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim){
    if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 0){

        // leo_lrc_block(output, input_l, input_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
        //             id_map, local_v_num, outdim, RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
        leo_lcr_block_v2(output, input_l, input_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                        id_map, local_v_num, outdim, nodePerPE, mynode, nullptr);
        // leo_lcr_block_v4(output, input_l, input_c, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, \
        //                  id_map, local_v_num, outdim, nodePerPE, mynode, nullptr);
        printf("3\n");
    }else if(CacheFlag == 0 && RemoteFlag == 1 && UVMFlag == 0){

        leo_lr_block(output, input_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, id_map, local_v_num, \
                    outdim, RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
        printf("2\n");
    }else if (CacheFlag == 0 && RemoteFlag == 0 && UVMFlag == 0){

        leo_l_block(output, input_l, row_ptr_l, col_ind_l, id_map, local_v_num, outdim, \
                    RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
        printf("1\n");
    } else if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 1){

        leo_lchr_block_V1(output, input_l, input_c, input_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, row_ptr_h, col_ind_h, \
                        id_map, local_v_num, outdim, RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, RunConfig::overlap_dist, mynode ,nullptr);
        // leo_lchr_block_V2(output, input_l, input_c, input_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, row_ptr_h, col_ind_h, \
        //                 id_map, local_v_num, outdim, nodePerPE, mynode, nullptr);
        printf("4\n");
    }else{
        LOG(ERROR) << "Unsupported aggregation funtions!";
    }  
}

void GCN_Model::Aggregate_Warp(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim){
    if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 0){

        // leo_lrc_warp(output, input_l, input_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
        //             id_map, local_v_num, outdim, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
        leo_lcr_warp_v1(output, input_l, input_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                        id_map, local_v_num, outdim, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
        printf("3\n");
    }else if(CacheFlag == 0 && RemoteFlag == 1 && UVMFlag == 0){

        leo_lr_warp(output, input_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, id_map, local_v_num, \
                    outdim, nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
        printf("2\n");
    }else if (CacheFlag == 0 && RemoteFlag == 0 && UVMFlag == 0){

        leo_l_warp(output, input_l, row_ptr_l, col_ind_l, id_map, local_v_num, outdim, \
                   nodePerPE, RunConfig::warpPerBlock, mynode, nullptr);
        printf("1\n");
    }else if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 1){

        // leo_lchr_warp(output, input_l, input_c, input_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, row_ptr_h, col_ind_h, \
        //                 id_map, local_v_num, outdim, nodePerPE, RunConfig::warpPerBlock,  mynode ,nullptr);
        leo_lchr_warp_V1(output, input_l, input_c, input_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, row_ptr_h, col_ind_h, \
                        id_map, local_v_num, outdim, nodePerPE, RunConfig::warpPerBlock,  mynode ,nullptr);
        printf("4\n");
    }else{
        LOG(ERROR) << "Unsupported aggregation funtions!";
    }  
}

void GCN_Model::Aggregate_Thread(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim){
    if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 0){
        leo_lcr_thread(output, input_l, input_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                        id_map, local_v_num, outdim, nodePerPE, mynode, nullptr);
        printf("3\n");
    }else if(CacheFlag == 0 && RemoteFlag == 1 && UVMFlag == 0){
        LOG(ERROR) << "Not yet deployed";
        printf("2\n");
    }else if (CacheFlag == 0 && RemoteFlag == 0 && UVMFlag == 0){
        leo_l_thread(output, input_l, row_ptr_l, col_ind_l, local_v_num, outdim, nodePerPE, mynode, nullptr);        
        printf("1\n");        
    }else if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 1){
        leo_lchr_thread(output, input_l, input_c, input_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                        row_ptr_h, col_ind_h, id_map, local_v_num, outdim, nodePerPE, mynode, nullptr);
        printf("4\n");
    }else{
        LOG(ERROR) << "Unsupported aggregation funtions!";
    }
}

void GCN_Model::UpdateCache(cache_feat_t* output, cache_feat_t* input, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream){
    leo_update_cache_block(output, input, cache_list, cache_num, dim, nodePerPE, mynode, stream); 
}

void GCN_Model::UpdateCache_Stream(cache_feat_t* output, cache_feat_t* input, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream){
    leo_update_cache_backward_block_wo_sync(output, input, cache_list, cache_num, dim, nodePerPE, mynode, stream);
}
//======================================== GIN =================================================

void GIN_Model::Train(){
    if(mynode == 0) printf("| Rank |   Stage  | layer |      Operator     |    Time    \n");
    
    Timer t0;
    //Forward_Basic();
    //Forward_Block();
    //Forward_Warp();
    Forward_Option();
    //Forward_Option_Stream();
    
    double infer_time = t0.Passed();
    std::cout << "Rank " << mynode << ": Inference time for GIN model =====> " << infer_time << "s" << std::endl;

    Timer t1;
    //Backward_Option();
    //Backward_Option_Stream();
    double backward_time = t1.Passed();
    std::cout << "Rank " << mynode << ": Backward time for GIN model =====> " << backward_time << "s" << std::endl;
}

void GIN_Model::Forward_Basic(){
    
    Timer t0;
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(cache_feature, W1_c, buff1_c, int(cache_v_num), in_dim, hid_dim);
    if(UVMFlag == 1) gemm_cblas(W1_h, cache_host_feature, buff1_h, hid_dim, int(host_cache_v_num), in_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   0   |     GEMM(GPU)     |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    //std::cout << "Rank: " << mynode << ", CacheFlag: " << CacheFlag << ", UVMFlag: " << UVMFlag << ", TransFlag: " << TransRemoteFlag << std::endl;
//1st layer GIN
    Timer t1;
    Aggregate_Basic(buff1, buff1_l, buff1_c, buff1_h, hid_dim);
    double aggre_time1 = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |     Aggregate     |  " << aggre_time1 << "ms  "<< std::endl;

    Timer t2;
    //Update cached data(CPU and GPU side)
    if(CacheFlag == 1) UpdateCache(buff1_update_c, buff1, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff1_update_h, buff1, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |    Update Cache   |  " << update_time1 << "ms  "<< std::endl;

    
    Timer t3;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff1_update_c, W2_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff1_update_h, W2_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |     GEMM(GPU)     |  " << mm_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
//2nd layer GIN
    Timer t4;
    Aggregate_Basic(buff2, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     Aggregate     |  " << aggre_time2 << "ms  "<< std::endl;

    Timer t5;
    if(CacheFlag == 1) UpdateCache(buff2_update_c, buff2, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff2_update_h, buff2, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time2 = t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |    Update Cache   |  " << update_time2 << "ms  "<< std::endl;

    Timer t6;
    //Reuse buff2 because buff2 will not be used in backpropagation
    gemm_cublas(buff2, W3_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff2_update_c, W3_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff2_update_h, W3_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time2 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     GEMM(GPU)     |  " << mm_time2 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//3rd layer GIN
    Timer t7;
    Aggregate_Basic(buff3, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time3 = t7.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |     Aggregate     |  " << aggre_time3 << "ms  "<< std::endl;

    Timer t8;
    if(CacheFlag == 1) UpdateCache(buff3_update_c, buff3, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff3_update_h, buff3, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time3 = t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |    Update Cache   |  " << update_time3 << "ms  "<< std::endl;

    Timer t9;
    //Reuse buff2 because buff2 will not be used in backpropagation
    gemm_cublas(buff3, W4_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff3_update_c, W4_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff3_update_h, W4_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time3 = t9.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |     GEMM(GPU)     |  " << mm_time3 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//4th layer GIN
    Timer t10;
    Aggregate_Basic(buff4, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time4 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |     Aggregate     |  " << aggre_time4 << "ms  "<< std::endl;

    Timer t11;
    if(CacheFlag == 1) UpdateCache(buff4_update_c, buff4, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff4_update_h, buff4, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time4 = t11.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |    Update Cache   |  " << update_time4 << "ms  "<< std::endl;

    Timer t12;
    //Reuse buff2
    gemm_cublas(buff4, W5_l, buff2_l,int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff4_update_c, W5_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff4_update_h, W5_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time4 = t12.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |     GEMM(GPU)     |  " << mm_time4 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//5th layer GIN
    Timer t13;
    Aggregate_Basic(buff5, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time5 = t13.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |     Aggregate     |  " << aggre_time5 << "ms  "<< std::endl;

    Timer t14;
    if(CacheFlag == 1) UpdateCache(buff5_update_c, buff5, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff5_update_h, buff5, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time5 = t14.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |    Update Cache   |  " << update_time5 << "ms  "<< std::endl;

    Timer t15;
    gemm_cublas(buff5, W6_l, buff5_l, int(local_v_num), hid_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time5 = t15.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |     GEMM(GPU)     |  " << mm_time5 << "ms  "<< std::endl;

    softmax_forward_cudnn(buff5_s, buff5_l, int(local_v_num), out_dim);
}

void GIN_Model::Forward_Block(){

    Timer t0;
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(cache_feature, W1_c, buff1_c, int(cache_v_num), in_dim, hid_dim);
    if(UVMFlag == 1) gemm_cblas(W1_h, cache_host_feature, buff1_h, hid_dim, int(host_cache_v_num), in_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   0   |     GEMM(GPU)     |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//1st layer GIN
    Timer t1;
    Aggregate_Block(buff1, buff1_l, buff1_c, buff1_h, hid_dim);
    double aggre_time1 = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |     Aggregate     |  " << aggre_time1 << "ms  "<< std::endl;

    Timer t2;
    if(CacheFlag == 1) UpdateCache(buff1_update_c, buff1, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff1_update_h, buff1, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |    Update Cache   |  " << update_time1 << "ms  "<< std::endl;

    Timer t3;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff1_update_c, W2_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff1_update_h, W2_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |     GEMM(GPU)     |  " << mm_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
//2nd layer GIN
    Timer t4;
    Aggregate_Block(buff2, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     Aggregate     |  " << aggre_time2 << "ms  "<< std::endl;

    Timer t5;
    if(CacheFlag == 1) UpdateCache(buff2_update_c, buff2, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff2_update_h, buff2, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time2 = t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |    Update Cache   |  " << update_time2 << "ms  "<< std::endl;

    Timer t6;
    gemm_cublas(buff2, W3_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff2_update_c, W3_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff2_update_h, W3_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time2 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     GEMM(GPU)     |  " << mm_time2 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//3rd layer GIN
    Timer t7;
    Aggregate_Block(buff3, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time3 = t7.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |     Aggregate     |  " << aggre_time3 << "ms  "<< std::endl;

    Timer t8;
    if(CacheFlag == 1) UpdateCache(buff3_update_c, buff3, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff3_update_h, buff3, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time3 = t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |    Update Cache   |  " << update_time3 << "ms  "<< std::endl;

    Timer t9;
    gemm_cublas(buff3, W4_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff3_update_c, W4_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff3_update_h, W4_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time3 = t9.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |     GEMM(GPU)     |  " << mm_time3 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//4th layer GIN
    Timer t10;
    Aggregate_Block(buff4, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time4 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |     Aggregate     |  " << aggre_time4 << "ms  "<< std::endl;

    Timer t11;
    if(CacheFlag == 1) UpdateCache(buff4_update_c, buff4, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff4_update_h, buff4, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time4 = t11.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |    Update Cache   |  " << update_time4 << "ms  "<< std::endl;

    Timer t12;
    gemm_cublas(buff4, W5_l, buff2_l,int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff4_update_c, W5_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff4_update_h, W5_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time4 = t12.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |     GEMM(GPU)     |  " << mm_time4 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//5th layer GIN
    Timer t13;
    Aggregate_Block(buff5, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time5 = t13.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |     Aggregate     |  " << aggre_time5 << "ms  "<< std::endl;

    Timer t14;
    if(CacheFlag == 1) UpdateCache(buff5_update_c, buff5, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff5_update_h, buff5, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time5 = t14.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |    Update Cache   |  " << update_time5 << "ms  "<< std::endl;

    Timer t15;
    gemm_cublas(buff5, W6_l, buff5_l, int(local_v_num), hid_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time5 = t15.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |     GEMM(GPU)     |  " << mm_time5 << "ms  "<< std::endl;

    softmax_forward_cudnn(buff5_s, buff5_l, int(local_v_num), out_dim);
}

void GIN_Model::Forward_Warp(){

    Timer t0;
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(cache_feature, W1_c, buff1_c, int(cache_v_num), in_dim, hid_dim);
    if(UVMFlag == 1) gemm_cblas(W1_h, cache_host_feature, buff1_h, hid_dim, int(host_cache_v_num), in_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   0   |     GEMM(GPU)     |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//1st layer GIN
    Timer t1;
    Aggregate_Warp(buff1, buff1_l, buff1_c, buff1_h, hid_dim);
    double aggre_time1 = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |     Aggregate     |  " << aggre_time1 << "ms  "<< std::endl;

    Timer t2;
    if(CacheFlag == 1) UpdateCache(buff1_update_c, buff1, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff1_update_h, buff1, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |    Update Cache   |  " << update_time1 << "ms  "<< std::endl; 

    Timer t3;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff1_update_c, W2_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff1_update_h, W2_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |     GEMM(GPU)     |  " << mm_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
//2nd layer GIN
    Timer t4;
    Aggregate_Warp(buff2, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     Aggregate     |  " << aggre_time2 << "ms  "<< std::endl;

    Timer t5;
    if(CacheFlag == 1) UpdateCache(buff2_update_c, buff2, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff2_update_h, buff2, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time2 = t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |    Update Cache   |  " << update_time2 << "ms  "<< std::endl;

    Timer t6;
    gemm_cublas(buff2, W3_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff2_update_c, W3_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff2_update_h, W3_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time2 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     GEMM(GPU)     |  " << mm_time2 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//3rd layer GIN
    Timer t7;
    Aggregate_Warp(buff3, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time3 = t7.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |     Aggregate     |  " << aggre_time3 << "ms  "<< std::endl;

    Timer t8;
    if(CacheFlag == 1) UpdateCache(buff3_update_c, buff3, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff3_update_h, buff3, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time3 = t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |    Update Cache   |  " << update_time3 << "ms  "<< std::endl;

    Timer t9;
    gemm_cublas(buff3, W4_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff3_update_c, W4_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff3_update_h, W4_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time3 = t9.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |     GEMM(GPU)     |  " << mm_time3 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//4th layer GIN
    Timer t10;
    Aggregate_Warp(buff4, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time4 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |     Aggregate     |  " << aggre_time4 << "ms  "<< std::endl;

    Timer t11;
    if(CacheFlag == 1) UpdateCache(buff4_update_c, buff4, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff4_update_h, buff4, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time4 = t11.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |    Update Cache   |  " << update_time4 << "ms  "<< std::endl;

    Timer t12;
    gemm_cublas(buff4, W5_l, buff2_l,int(local_v_num), hid_dim, hid_dim);
    if(CacheFlag == 1) gemm_cublas(buff4_update_c, W5_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    if(UVMFlag == 1) gemm_cublas(buff4_update_h, W5_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time4 = t12.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |     GEMM(GPU)     |  " << mm_time4 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//5th layer GIN
    Timer t13;
    Aggregate_Warp(buff5, buff2_l, buff2_c, buff2_h, hid_dim);
    double aggre_time5 = t13.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |     Aggregate     |  " << aggre_time5 << "ms  "<< std::endl;

    Timer t14;
    if(CacheFlag == 1) UpdateCache(buff5_update_c, buff5, cache_list, cache_v_num, hid_dim, nullptr);
    if(UVMFlag == 1) UpdateCache(buff5_update_h, buff5, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time5 = t14.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |    Update Cache   |  " << update_time5 << "ms  "<< std::endl;

    Timer t15;
    gemm_cublas(buff5, W6_l, buff5_l, int(local_v_num), hid_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time5 = t15.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |     GEMM(GPU)     |  " << mm_time5 << "ms  "<< std::endl;

    softmax_forward_cudnn(buff5_s, buff5_l, int(local_v_num), out_dim);           
}

void GIN_Model::Forward_Option(){
    if(CacheFlag == true){
        if(UVMFlag == true){
            Forward_wCPUGPU_Cache();
        }else{
            Forward_wGPU_Cache();
        }
    }else{
        if(RemoteFlag == true){
            Forward_wo_Cache();
        }else{
            Forward_LocalOnly();
        }
    }
}

void GIN_Model::Forward_Option_Stream(){
    if(CacheFlag == true){
        if(UVMFlag == true){
            Forward_wCPUGPU_Cache_Stream();
        }else{
            Forward_wGPU_Cache();
        }
    }else{
        if(RemoteFlag == true){
            Forward_wo_Cache();
        }else{
            Forward_LocalOnly();
        }
    }    
}

void GIN_Model::Forward_wCPUGPU_Cache(){

    Timer t0;
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    gemm_cublas(cache_feature, W1_c, buff1_c, int(cache_v_num), in_dim, hid_dim);
    gemm_cblas(W1_h, cache_host_feature, buff1_h, hid_dim, int(host_cache_v_num), in_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   0   |  GEMM(GPU, CPU)   |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//1st layer GIN
    Timer t1;
    //Aggregate_Warp
    leo_gin_lchr_warp(buff1, buff1_l, buff1_c, buff1_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time1 = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |  Aggregate(lchr)  |  " << aggre_time1 << "ms  "<< std::endl;

    Timer t2;
    UpdateCache(buff1_update_c, buff1, cache_list, cache_v_num, hid_dim, nullptr);
    UpdateCache(buff1_update_h, buff1, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |  Update Cache(hr) |  " << update_time1 << "ms  "<< std::endl;

    Timer t3;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    gemm_cublas(buff1_update_c, W2_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    gemm_cublas(buff1_update_h, W2_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |  GEMM(GPU, CPU)   |  " << mm_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//2nd layer GIN
    Timer t4;
    //Aggregate_Warp
    leo_gin_lchr_warp(buff2, buff2_l, buff2_c, buff2_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |  Aggregate(lchr)  |  " << aggre_time2 << "ms  "<< std::endl;

    Timer t5;
    UpdateCache(buff2_update_c, buff2, cache_list, cache_v_num, hid_dim, nullptr);
    UpdateCache(buff2_update_h, buff2, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time2 = t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |  Update Cache(hr) |  " << update_time2 << "ms  "<< std::endl;

    Timer t6;
    gemm_cublas(buff2, W3_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    gemm_cublas(buff2_update_c, W3_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    gemm_cublas(buff2_update_h, W3_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time2 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |  GEMM(GPU, CPU)   |  " << mm_time2 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//3rd layer GIN
    Timer t7;
    leo_gin_lchr_warp(buff3, buff2_l, buff2_c, buff2_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time3 = t7.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |  Aggregate(lchr)  |  " << aggre_time3 << "ms  "<< std::endl;

    Timer t8;
    UpdateCache(buff3_update_c, buff3, cache_list, cache_v_num, hid_dim, nullptr);
    UpdateCache(buff3_update_h, buff3, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time3 = t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |  Update Cache(hr) |  " << update_time3 << "ms  "<< std::endl;

    Timer t9;
    gemm_cublas(buff3, W4_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    gemm_cublas(buff3_update_c, W4_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    gemm_cublas(buff3_update_h, W4_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time3 = t9.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |  GEMM(GPU, CPU)   |  " << mm_time3 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//4th layer GIN
    Timer t10;
    leo_gin_lchr_warp(buff4, buff2_l, buff2_c, buff2_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);    
    double aggre_time4 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |  Aggregate(lchr)  |  " << aggre_time4 << "ms  "<< std::endl;

    Timer t11;
    UpdateCache(buff4_update_c, buff4, cache_list, cache_v_num, hid_dim, nullptr);
    UpdateCache(buff4_update_h, buff4, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time4 = t11.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |  Update Cache(hr) |  " << update_time4 << "ms  "<< std::endl;

    Timer t12;
    gemm_cublas(buff4, W5_l, buff2_l,int(local_v_num), hid_dim, hid_dim);
    gemm_cublas(buff4_update_c, W5_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    gemm_cublas(buff4_update_h, W5_h, buff2_h, host_cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time4 = t12.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |  GEMM(GPU, CPU)   |  " << mm_time4 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//5th layer GIN
    Timer t13;
    leo_gin_lchr_warp(buff5, buff2_l, buff2_c, buff2_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);    
    double aggre_time5 = t13.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |  Aggregate(lchr)  |  " << aggre_time5 << "ms  "<< std::endl;

    Timer t14;
    UpdateCache(buff5_update_c, buff5, cache_list, cache_v_num, hid_dim, nullptr);
    UpdateCache(buff5_update_h, buff5, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time5 = t14.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |  Update Cache(hr) |  " << update_time5 << "ms  "<< std::endl;

    Timer t15;
    gemm_cublas(buff5, W6_l, buff5_l, int(local_v_num), hid_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time5 = t15.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |  GEMM(GPU, CPU)   |  " << mm_time5 << "ms  "<< std::endl;

    softmax_forward_cudnn(buff5_s, buff5_l, int(local_v_num), out_dim);
}

void GIN_Model::Forward_wCPUGPU_Cache_Stream(){
    cudaEvent_t Aggre1, Aggre2, Aggre3, Aggre4, Aggre5;
    //Create CUDA streams
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    cudaEventCreate(&Aggre1);
    cudaEventCreate(&Aggre2);
    cudaEventCreate(&Aggre3);
    cudaEventCreate(&Aggre4);
    cudaEventCreate(&Aggre5);

    Timer t0;
    gemm_cublas_Stream(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim, stream1);
    gemm_cblas(W1_h, cache_host_feature, buff1_h, hid_dim, int(host_cache_v_num), in_dim);
    gemm_cublas_Stream(cache_feature, W1_c, buff1_c, int(cache_v_num), in_dim, hid_dim, stream2);
    CUDA_CALL(cudaDeviceSynchronize()); 
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   0   |  GEMM(GPU, CPU)   |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//1st layer GIN
    Timer t1;
    leo_gin_lchr_warp(buff1, buff1_l, buff1_c, buff1_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, stream1);
    double aggre_time1 = t1.PassedMilli();
    cudaEventRecord(Aggre1, stream1);
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |  Aggregate(lchr)  |  " << aggre_time1 << "ms  "<< std::endl;

    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream2, Aggre1, 0);
    cudaStreamWaitEvent(stream3, Aggre1, 0);
    cudaStreamWaitEvent(stream4, Aggre1, 0); 

    Timer t2;
    //stream2
    UpdateCache_Stream(buff1_update_c, buff1, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_Stream(buff1_update_c, W2_c, buff2_c, cache_v_num, hid_dim, hid_dim, stream2);

    //stream3
    UpdateCache_Stream(buff1_update_h, buff1, cache_host_list, host_cache_v_num, hid_dim, stream3);
    gemm_cublas_Stream(buff1_update_h, W2_h, buff2_h, host_cache_v_num, hid_dim, hid_dim, stream3);

    //Stream4
    gemm_cublas_Stream(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, hid_dim, stream4);

    //Sync all CUDA streams across the device
    CUDA_CALL(cudaDeviceSynchronize());
    // cudaError_t error1 = cudaGetLastError();
    // if(error1 != cudaSuccess){
    //     printf("CUDA Kernel Error @Forward_stream1: %s\n", cudaGetErrorString(error1));
    // } 
    double stream_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |    Update&GEMM    |  " << stream_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//2rd layer GIN
    Timer t3;
    leo_gin_lchr_warp(buff2, buff2_l, buff2_c, buff2_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, stream1);
    double aggre_time2 = t3.PassedMilli();
    cudaEventRecord(Aggre2, stream1);
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |  Aggregate(lchr)  |  " << aggre_time2 << "ms  "<< std::endl;

    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream2, Aggre2, 0);
    cudaStreamWaitEvent(stream3, Aggre2, 0);
    cudaStreamWaitEvent(stream4, Aggre2, 0);  

    Timer t4;
    //stream2
    UpdateCache_Stream(buff2_update_c, buff2, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_Stream(buff2_update_c, W3_c, buff2_c, cache_v_num, hid_dim, hid_dim, stream2);

    //stream3
    UpdateCache_Stream(buff2_update_h, buff2, cache_host_list, host_cache_v_num, hid_dim, stream3);
    gemm_cublas_Stream(buff2_update_h, W3_h, buff2_h, host_cache_v_num, hid_dim, hid_dim, stream3);

    //stream4
    gemm_cublas_Stream(buff2, W3_l, buff2_l, int(local_v_num), hid_dim, hid_dim, stream4);   

    //Sync all CUDA streams across the device
    CUDA_CALL(cudaDeviceSynchronize());
    // cudaError_t error2 = cudaGetLastError();
    // if(error2 != cudaSuccess){
    //     printf("CUDA Kernel Error @Forward_stream2: %s\n", cudaGetErrorString(error2));
    // } 
    double stream_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |    Update&GEMM    |  " << stream_time2 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//3rd layer GIN
    Timer t5;
    leo_gin_lchr_warp(buff3, buff2_l, buff2_c, buff2_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, stream1);
    double aggre_time3 = t5.PassedMilli();
    cudaEventRecord(Aggre3, stream1);
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |  Aggregate(lchr)  |  " << aggre_time3 << "ms  "<< std::endl;    

    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream2, Aggre3, 0);
    cudaStreamWaitEvent(stream3, Aggre3, 0);
    cudaStreamWaitEvent(stream4, Aggre3, 0); 

    Timer t6;
    //stream2
    UpdateCache_Stream(buff3_update_c, buff3, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_Stream(buff3_update_c, W4_c, buff2_c, cache_v_num, hid_dim, hid_dim, stream2); 

    //stream3
    UpdateCache_Stream(buff3_update_h, buff3, cache_host_list, host_cache_v_num, hid_dim, stream3);
    gemm_cublas_Stream(buff3_update_h, W4_h, buff2_h, host_cache_v_num, hid_dim, hid_dim, stream3);

    //stream4
    gemm_cublas_Stream(buff3, W4_l, buff2_l, int(local_v_num), hid_dim, hid_dim, stream4);  

    //Sync all CUDA streams across the device
    CUDA_CALL(cudaDeviceSynchronize());
    // cudaError_t error3 = cudaGetLastError();
    // if(error3 != cudaSuccess){
    //     printf("CUDA Kernel Error @Forward_stream3: %s\n", cudaGetErrorString(error3));
    // } 
    double stream_time3 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |    Update&GEMM    |  " << stream_time3 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//4th layer GIN
    Timer t7;
    leo_gin_lchr_warp(buff4, buff2_l, buff2_c, buff2_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, stream1);    
    double aggre_time4 = t7.PassedMilli();
    cudaEventRecord(Aggre4, stream1);
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |  Aggregate(lchr)  |  " << aggre_time4 << "ms  "<< std::endl;    

    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream2, Aggre4, 0);
    cudaStreamWaitEvent(stream3, Aggre4, 0);
    cudaStreamWaitEvent(stream4, Aggre4, 0); 

    Timer t8;
    //stream2
    UpdateCache_Stream(buff4_update_c, buff4, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_Stream(buff4_update_c, W5_c, buff2_c, cache_v_num, hid_dim, hid_dim, stream2);

    //stream3
    UpdateCache_Stream(buff4_update_h, buff4, cache_host_list, host_cache_v_num, hid_dim, stream3);
    gemm_cublas_Stream(buff4_update_h, W5_h, buff2_h, host_cache_v_num, hid_dim, hid_dim, stream3); 

    //stream4
    gemm_cublas_Stream(buff4, W5_l, buff2_l,int(local_v_num), hid_dim, hid_dim, stream4);

    //Sync all CUDA streams across the device
    CUDA_CALL(cudaDeviceSynchronize());
    // cudaError_t error4 = cudaGetLastError();
    // if(error4 != cudaSuccess){
    //     printf("CUDA Kernel Error @Forward_stream4: %s\n", cudaGetErrorString(error4));
    // } 
    double stream_time4 = t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |    Update&GEMM    |  " << stream_time4 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);   

//5th layer GIN
    Timer t9;
    leo_gin_lchr_warp(buff5, buff2_l, buff2_c, buff2_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    row_ptr_h, col_ind_h, id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, stream1);    
    double aggre_time5 = t9.PassedMilli();
    cudaEventRecord(Aggre5, stream1);
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |  Aggregate(lchr)  |  " << aggre_time5 << "ms  "<< std::endl;    

    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream2, Aggre4, 0);
    cudaStreamWaitEvent(stream3, Aggre4, 0);
    cudaStreamWaitEvent(stream4, Aggre4, 0); 

    Timer t10;
    //stream2
    UpdateCache_Stream(buff5_update_c, buff5, cache_list, cache_v_num, hid_dim, stream2);

    //stream3
    UpdateCache_Stream(buff5_update_h, buff5, cache_host_list, host_cache_v_num, hid_dim, stream3);

    //stream4
    gemm_cublas_Stream(buff5, W6_l, buff5_l, int(local_v_num), hid_dim, out_dim, stream4);
    softmax_forward_cudnn_Stream(buff5_s, buff5_l, int(local_v_num), out_dim, stream4);

    //Sync all CUDA streams across the device
    CUDA_CALL(cudaDeviceSynchronize());
    // cudaError_t error5 = cudaGetLastError();
    // if(error5 != cudaSuccess){
    //     printf("CUDA Kernel Error @Forward_stream5: %s\n", cudaGetErrorString(error5));
    // } 
    double stream_time5 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |    Update&GEMM    |  " << stream_time5 << "ms  "<< std::endl;
}

void GIN_Model::Forward_wGPU_Cache(){

    Timer t0;
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    gemm_cublas(cache_feature, W1_c, buff1_c, int(cache_v_num), in_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   0   |     GEMM(GPU)     |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//1st layer GIN
    Timer t1;
    leo_gin_lcr_warp(buff1, buff1_l, buff1_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time1 = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |   Aggregate(lcr)  |  " << aggre_time1 << "ms  "<< std::endl;

    Timer t2;
    UpdateCache(buff1_update_c, buff1, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   | Update Cache(GPU) |  " << update_time1 << "ms  "<< std::endl;

    Timer t3;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    gemm_cublas(buff1_update_c, W2_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |     GEMM(GPU)     |  " << mm_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//2nd layer GIN
    Timer t4;
    leo_gin_lcr_warp(buff2, buff2_l, buff2_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |   Aggregate(lcr)  |  " << aggre_time2 << "ms  "<< std::endl;

    Timer t5;
    UpdateCache(buff2_update_c, buff2, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time2 = t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   | Update Cache(GPU) |  " << update_time2 << "ms  "<< std::endl;

    Timer t6;
    gemm_cublas(buff2, W3_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    gemm_cublas(buff2_update_c, W3_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time2 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     GEMM(GPU)     |  " << mm_time2 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD); 

//3rd layer GIN
    Timer t7;
    leo_gin_lcr_warp(buff3, buff2_l, buff2_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);    
    double aggre_time3 = t7.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |   Aggregate(lcr)  |  " << aggre_time3 << "ms  "<< std::endl;

    Timer t8;
    UpdateCache(buff3_update_c, buff3, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time3 = t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   | Update Cache(GPU) |  " << update_time3 << "ms  "<< std::endl;

    Timer t9;
    gemm_cublas(buff3, W4_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    gemm_cublas(buff3_update_c, W4_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time3 = t9.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |     GEMM(GPU)     |  " << mm_time3 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD); 

//4th layer GIN
    Timer t10;
    leo_gin_lcr_warp(buff4, buff2_l, buff2_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);    
    double aggre_time4 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |   Aggregate(lcr)  |  " << aggre_time4 << "ms  "<< std::endl;

    Timer t11;
    UpdateCache(buff4_update_c, buff4, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time4 = t11.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   | Update Cache(GPU) |  " << update_time4 << "ms  "<< std::endl;

    Timer t12;
    gemm_cublas(buff4, W5_l, buff2_l,int(local_v_num), hid_dim, hid_dim);
    gemm_cublas(buff4_update_c, W5_c, buff2_c, cache_v_num, hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time4 = t12.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |     GEMM(GPU)     |  " << mm_time4 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);   

//5th layer GIN
    Timer t13;
    leo_gin_lcr_warp(buff5, buff2_l, buff2_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r,\
                    id_map, local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);    
    double aggre_time5 = t13.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |   Aggregate(lcr)  |  " << aggre_time5 << "ms  "<< std::endl;

    Timer t14;
    UpdateCache(buff5_update_c, buff5, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time5 = t14.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   | Update Cache(GPU) |  " << update_time5 << "ms  "<< std::endl;

    Timer t15;
    gemm_cublas(buff5, W6_l, buff5_l, int(local_v_num), hid_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time5 = t15.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |     GEMM(GPU)     |  " << mm_time5 << "ms  "<< std::endl;

    softmax_forward_cudnn(buff5_s, buff5_l, int(local_v_num), out_dim);       
}

void GIN_Model::Forward_wo_Cache(){
    
    Timer t0;
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   0   |     GEMM(GPU)     |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);  

//1st layer GIN
    Timer t1;
    leo_gin_lr_warp(buff1, buff1_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, id_map,\
                    local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time1 = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |   Aggregate(lr)   |  " << aggre_time1 << "ms  "<< std::endl;

    Timer t2;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     GEMM(GPU)     |  " << mm_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//2nd layer GIN
    Timer t3;
    leo_gin_lr_warp(buff2, buff2_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, id_map,\
                    local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time2 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |   Aggregate(lr)   |  " << aggre_time2 << "ms  "<< std::endl;

    Timer t4;
    gemm_cublas(buff2, W3_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     GEMM(GPU)     |  " << mm_time2 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//3rd layer GIN
    Timer t5;
    leo_gin_lr_warp(buff3, buff2_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, id_map,\
                    local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time3 = t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |   Aggregate(lr)   |  " << aggre_time3 << "ms  "<< std::endl;

    Timer t6;
    gemm_cublas(buff3, W4_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time3 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |     GEMM(GPU)     |  " << mm_time3 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//4th layer GIN
    Timer t7;
    leo_gin_lr_warp(buff4, buff2_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, id_map,\
                    local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time4 = t7.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |   Aggregate(lr)   |  " << aggre_time4 << "ms  "<< std::endl;

    Timer t8;
    gemm_cublas(buff4, W5_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time4 = t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |     GEMM(GPU)     |  " << mm_time4 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//5th layer GIN
    Timer t9;
    leo_gin_lr_warp(buff5, buff2_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, id_map,\
                    local_v_num, hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time5 = t9.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |   Aggregate(lr)   |  " << aggre_time5 << "ms  "<< std::endl;

    Timer t10;
    gemm_cublas(buff5, W6_l, buff5_l, int(local_v_num), hid_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time5 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |     GEMM(GPU)     |  " << mm_time5 << "ms  "<< std::endl;

    softmax_forward_cudnn(buff5_s, buff5_l, int(local_v_num), out_dim);
}

void GIN_Model::Forward_LocalOnly(){

    Timer t0;
    gemm_cublas(local_feature, W1_l, buff1_l, int(local_v_num), in_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   0   |     GEMM(GPU)     |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//1st layer GIN
    Timer t1;
    leo_gin_l_warp(buff1, buff1_l, row_ptr_l, col_ind_l, id_map, local_v_num,\
                    hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time1 = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   1   |    Aggregate(l)   |  " << aggre_time1 << "ms  "<< std::endl; 

    Timer t2;
    gemm_cublas(buff1, W2_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     GEMM(GPU)     |  " << mm_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//2nd layer GIN
    Timer t3;
    leo_gin_l_warp(buff2, buff2_l, row_ptr_l, col_ind_l, id_map, local_v_num,\
                    hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time2 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |    Aggregate(l)   |  " << aggre_time2 << "ms  "<< std::endl;       

    Timer t4;
    gemm_cublas(buff2, W3_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   2   |     GEMM(GPU)     |  " << mm_time2 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//3rd layer GIN
    Timer t5;
    leo_gin_l_warp(buff3, buff2_l, row_ptr_l, col_ind_l, id_map, local_v_num,\
                    hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time3 = t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |    Aggregate(l)   |  " << aggre_time3 << "ms  "<< std::endl;

    Timer t6;
    gemm_cublas(buff3, W4_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time3 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   3   |     GEMM(GPU)     |  " << mm_time3 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//4th layer GIN
    Timer t7;
    leo_gin_l_warp(buff4, buff2_l, row_ptr_l, col_ind_l, id_map, local_v_num,\
                    hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time4 = t7.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |    Aggregate(l)   |  " << aggre_time4 << "ms  "<< std::endl;

    Timer t8;
    gemm_cublas(buff4, W5_l, buff2_l, int(local_v_num), hid_dim, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time4 = t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   4   |     GEMM(GPU)     |  " << mm_time4 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

//5th layer GIN
    Timer t9;
    leo_gin_l_warp(buff4, buff2_l, row_ptr_l, col_ind_l, id_map, local_v_num,\
                    hid_dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    double aggre_time5 = t9.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |    Aggregate(l)   |  " << aggre_time5 << "ms  "<< std::endl; 

    Timer t10;
    gemm_cublas(buff5, W6_l, buff5_l, int(local_v_num), hid_dim, out_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time5 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Forward  |   5   |     GEMM(GPU)     |  " << mm_time5 << "ms  "<< std::endl;

    softmax_forward_cudnn(buff5_s, buff5_l, int(local_v_num), out_dim);       
}

void GIN_Model::Aggregate_Basic(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim){
    if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 0){

        leo_gin_lrc_block(output, input_l, input_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                        id_map, local_v_num, outdim, RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    }else if(CacheFlag == 0 && RemoteFlag == 1 && UVMFlag == 0){

        leo_gin_lr_block(output, input_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, id_map, local_v_num, \
                        outdim, RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    }else if(CacheFlag == 0 && RemoteFlag == 0 && UVMFlag == 0){

        leo_gin_l_block(output, input_l, row_ptr_l, col_ind_l, id_map, local_v_num, outdim, \
                    RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    }else if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 1){

        leo_gin_lchr_block(output, input_l, input_c, input_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, row_ptr_h, col_ind_h, \
                        id_map, local_v_num, outdim, RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, RunConfig::overlap_dist, eps, mynode, nullptr);
    }else{
        LOG(ERROR) << "Unsupported aggregation funtions!";
    }
}

void GIN_Model::Aggregate_Block(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim){
    if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 0){

        leo_gin_lcr_block_v1(output, input_l, input_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                            id_map, local_v_num, outdim, nodePerPE, eps, mynode, nullptr);
    }else if(CacheFlag == 0 && RemoteFlag == 1 && UVMFlag == 0){

        leo_gin_lr_block_v1(output, input_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, id_map, \
                            local_v_num, outdim, nodePerPE, eps, mynode, nullptr);
    }else if(CacheFlag == 0 && RemoteFlag == 0 && UVMFlag == 0){

        leo_gin_l_block_v1(output, input_l, row_ptr_l, col_ind_l, id_map, local_v_num, \
                            outdim, nodePerPE, eps, mynode, nullptr);
    }else if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 1){

        leo_gin_lchr_block_v1(output, input_l, input_c, input_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, row_ptr_h, col_ind_h, \
                                id_map, local_v_num, outdim, nodePerPE, eps, mynode, nullptr);
    }else{
       LOG(ERROR) << "Unsupported aggregation funtions!"; 
    }
}

void GIN_Model::Aggregate_Warp(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim){
    if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 0){

        leo_gin_lcr_warp(output, input_l, input_c, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, \
                        id_map, local_v_num, outdim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    }else if(CacheFlag == 0 && RemoteFlag == 1 && UVMFlag == 0){

        leo_gin_lr_warp(output, input_l, row_ptr_l, col_ind_l, row_ptr_r, col_ind_r, id_map, \
                        local_v_num, outdim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    }else if(CacheFlag == 0 && RemoteFlag == 0 && UVMFlag == 0){

        leo_gin_l_warp(output, input_l, row_ptr_l, col_ind_l, id_map, local_v_num, \
                        outdim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    }else if(CacheFlag == 1 && RemoteFlag == 1 && UVMFlag == 1){

        leo_gin_lchr_warp(output, input_l, input_c, input_h, row_ptr_l, col_ind_l, row_ptr_c, col_ind_c, row_ptr_r, col_ind_r, row_ptr_h, col_ind_h, \
                            id_map, local_v_num, outdim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, nullptr);
    }else{
        LOG(ERROR) << "Unsupported aggregation funtions!"; 
    }
}

void GIN_Model::UpdateCache(cache_feat_t* output, cache_feat_t* input, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream){
    leo_update_cache_block(output, input, cache_list, cache_num, dim, nodePerPE, mynode, stream); 
}

void GIN_Model::UpdateCache_Stream(cache_feat_t* output, cache_feat_t* input, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream){
    leo_update_cache_backward_block_wo_sync(output, input, cache_list, cache_num, dim, nodePerPE, mynode, stream);
}

//=================================Back Propagation================================================

void GCN_Model::Backward(){
    //2nd layer GCN
    //Compute gradient of SoftmaxCrossEntroy loss
    SoftmaxCrossEntroyBackward(grad2_s, buff2_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, nullptr);

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t0;
    Aggregate_Backward(grad2_l, grad2_s, out_dim, nullptr);
    double aggre_time = t0.PassedMilli();
    std::cout << "Rank " << mynode << ": Aggregation (Backward) time in the 2nd layer ======> " << aggre_time << "ms" << std::endl;

    Timer t1;
    if(CacheFlag == true) UpdateCache_Backward(grad2_c, grad2_l, cache_list, cache_v_num, out_dim, nullptr);
    double update_time = t1.PassedMilli();
    std::cout << "Rank " << mynode << ": Updating cached feature (Backward) time in the 2nd layer ======> " << update_time << "ms" << std::endl;

    gemm_cublas_backward(grad2_l, buff1, grad2_l_W, false, true, out_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, out_dim);
    if (CacheFlag == 1) gemm_cublas_backward(grad2_c, update_cache_feature, grad2_c_W, false, true, out_dim, hid_dim, cache_v_num);

    MPI_Barrier(MPI_COMM_WORLD);
    //1st layer GCN
    Timer t2;
    Aggregate_Backward(grad1_l, grad1, hid_dim, nullptr);
    double aggre_time2 = t2.PassedMilli();
    std::cout << "Rank " << mynode << ": Aggregation (Backward) time in the 1st layer ======> " << aggre_time2 << "ms" << std::endl;

    Timer t3;
    if(CacheFlag == true) UpdateCache_Backward(grad1_c, grad1_l, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time2 = t3.PassedMilli();
    std::cout << "Rank " << mynode << ": Updating cached feature (Backward) time in the 1st layer ======> " << update_time2 << "ms" << std::endl;

    gemm_cublas_backward(grad1_l, local_feature, grad1_l_W, false, true, hid_dim, in_dim, local_v_num);
    if (CacheFlag == 1) gemm_cublas_backward(grad1_c, cache_feature, grad1_c_W, false, true, hid_dim, in_dim , cache_v_num);

    opt1->Update(grad2_l_W, nullptr);
    opt2->Update(grad1_l_W, nullptr);
    if (CacheFlag == 1) opt3->Update(grad2_c_W, nullptr);
    if (CacheFlag == 1) opt4->Update(grad1_c_W, nullptr);
}

void GCN_Model::Aggregate_Backward(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream){
    if(TransRemoteFlag == 1){
        leo_local_backward_block(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, T_row_ptr_r, T_col_ind_r, local_v_num, \
                                dim, RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, mynode, stream);
        printf("2\n");
    }else{
        leo_local_only_backward_block(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, local_v_num, dim, \
                                    RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, mynode, stream);
        printf("1\n");
    }
}

void GCN_Model::Aggregate_Backward_Stream(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream){
    if(TransRemoteFlag == 1){
        leo_local_backward_block_wo_sync(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, T_row_ptr_r, T_col_ind_r, local_v_num, \
                                        dim, RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, mynode, stream);
        printf("2\n");
    }else{
        leo_local_only_backward_block_wo_sync(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, local_v_num, dim, \
                                            RunConfig::partsize, nodePerPE, RunConfig::warpPerBlock, mynode, stream);
        printf("1\n");
    }
}

void GCN_Model::Aggregate_Backward_Warp(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream){
    if(TransRemoteFlag == 1){
        leo_local_backward_warp(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, T_row_ptr_r, T_col_ind_r, local_v_num, \
                                dim, nodePerPE, RunConfig::warpPerBlock, mynode, stream);
    }else{
        leo_local_only_backward_warp(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, local_v_num, \
                                    dim, nodePerPE, RunConfig::warpPerBlock, mynode, stream);
    }
}

void GCN_Model::Aggregate_Backward_Warp_Stream(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream){
    if(TransRemoteFlag == 1){
        leo_local_backward_warp_wo_sync(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, T_row_ptr_r, T_col_ind_r, local_v_num, \
                                dim, nodePerPE, RunConfig::warpPerBlock, mynode, stream);
    }else{
        leo_local_only_backward_warp_wo_sync(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, local_v_num, \
                                    dim, nodePerPE, RunConfig::warpPerBlock, mynode, stream);
    }
}

void GCN_Model::UpdateCache_Backward(value_t* out_grad, value_t* in_grad, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream){
    leo_update_cache_backward_block(out_grad, in_grad, cache_list, cache_num, dim, nodePerPE, mynode, stream);
}

void GCN_Model::UpdateCache_Backward_Stream(value_t* out_grad, value_t* in_grad, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream){
    leo_update_cache_backward_block_wo_sync(out_grad, in_grad, cache_list, cache_num, dim, nodePerPE, mynode, stream);
}

//Manage concurrent operations using CUDA streams
void GCN_Model::Backward_Stream(){
    cudaEvent_t Aggre1, Aggre2;
    //Create CUDA streams
    cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);
    cudaStreamCreate(&stream6);


    //Until the first aggregation, the model performs common prerequisites
    cudaEventCreate(&Aggre1);
    SoftmaxCrossEntroyBackward(grad2_s, buff2_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, stream1);
    MPI_Barrier(MPI_COMM_WORLD);
    Aggregate_Backward_Stream(grad2_l, grad2_s, out_dim, stream1);
    cudaEventRecord(Aggre1, stream1);

    //Event 1 (first aggregation) is a dependency on streams 2, 3, and 4.
    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream2, Aggre1, 0);
    cudaStreamWaitEvent(stream3, Aggre1, 0);
    cudaStreamWaitEvent(stream4, Aggre1, 0);

    //stream2
    cudaEventCreate(&Aggre2);
    gemm_cublas_backward_Stream(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, out_dim, stream2);
    MPI_Barrier(MPI_COMM_WORLD);
    Aggregate_Backward_Stream(grad1_l, grad1, hid_dim, stream2);
    cudaEventRecord(Aggre2, stream2);

    //stream3
    UpdateCache_Backward_Stream(grad2_c, grad2_l, cache_list, cache_v_num, out_dim, stream3);
    if (CacheFlag == 1){
        gemm_cublas_backward_Stream(grad2_c, update_cache_feature, grad2_c_W, false, true, out_dim, hid_dim, cache_v_num, stream3);
        opt3->Update_stream(grad2_c_W, stream3);
    } 

    //stream4
    gemm_cublas_backward_Stream(grad2_l, buff1, grad2_l_W, false, true, out_dim, hid_dim, local_v_num, stream4);
    opt1->Update_stream(grad2_l_W, stream4);

    //Event 2 (second aggregation) is a dependency on streams 5, and 6.
    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream5, Aggre2, 0);
    cudaStreamWaitEvent(stream6, Aggre2, 0);

    //stream5
    UpdateCache_Backward_Stream(grad1_c, grad1_l, cache_list, cache_v_num, hid_dim, stream5);
    if (CacheFlag == 1){
        gemm_cublas_backward_Stream(grad1_c, cache_feature, grad1_c_W, false, true, hid_dim, in_dim , cache_v_num, stream5);
        opt4->Update_stream(grad1_c_W, stream5);
    }

    //stream6
    gemm_cublas_backward_Stream(grad1_l, local_feature, grad1_l_W, false, true, hid_dim, in_dim, local_v_num, stream6);
    opt2->Update_stream(grad1_l_W, stream6);

    //Sync all CUDA streams across the device
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cudaStreamSynchronize(stream3);
    cudaStreamSynchronize(stream4);
    cudaStreamSynchronize(stream5);
    cudaStreamSynchronize(stream6);
    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @Backward_stream: %s\n", cudaGetErrorString(error));
    }

    //clean up
    cudaEventDestroy(Aggre1);
    cudaEventDestroy(Aggre2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    cudaStreamDestroy(stream5);
    cudaStreamDestroy(stream6);
}

void GCN_Model::Backward_Option(){
    //std::cout << "Rank: " << mynode << ", CacheFlag: " << CacheFlag << ", UVMFlag: " << UVMFlag << ", TransFlag: " << TransRemoteFlag << std::endl;
    if(mynode == 0) printf("| Rank |   Stage  | layer |      Operator     |    Time    \n");
    if(CacheFlag == true){
        if(UVMFlag == true){
            Backward_wCPUGPU_Cache();
            //Backward_wCPUGPU_Cache_Stream();
        }else{
            //Backward_wGPU_Cache();
            Backward_wGPU_Cache_Stream();
        }
    }else{
        //TODO: the logic between CacheFlag and TransRemoteFlag need to be processed later
        //Backward_LocalOnly();
        Backward_LocalOnly_Stream();
    }
}

void GCN_Model::Backward_wCPUGPU_Cache(){
    //2nd layer GCN
    SoftmaxCrossEntroyBackward(grad2_s, buff2_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, nullptr);

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t0;
    Aggregate_Backward_Warp(grad2_l, grad2_s, out_dim, nullptr);
    double aggre_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |   Aggregate(lr)    |  " << aggre_time << "ms  " << std::endl;

    Timer t1;
    UpdateCache_Backward(grad2_c, grad2_l, cache_list, cache_v_num, out_dim, nullptr);
    UpdateCache_Backward(grad2_h, grad2_l, cache_host_list, host_cache_v_num, out_dim, nullptr);
    double update_time = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |   UpdateCache(hc)  |  " << update_time << "ms  " << std::endl;

    Timer t2;
    gemm_cublas_backward(grad2_l, buff1, grad2_l_W, false, true, out_dim, hid_dim, local_v_num); 
    gemm_cublas_backward(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, out_dim);
    gemm_cublas_backward(grad2_c, update_cache_feature, grad2_c_W, false, true, out_dim, hid_dim, cache_v_num);
    gemm_cublas_backward(grad2_h, update_cache_host_featrue, grad2_h_W, false, true, out_dim, hid_dim, host_cache_v_num);
    double mm_time = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |     GEMM(lch)      |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    //1st layer GCN
    Timer t3;
    Aggregate_Backward_Warp(grad1_l, grad1, hid_dim, nullptr);
    double aggre_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |   Aggregate(lr)    |  " << aggre_time1 << "ms  " << std::endl;

    Timer t4;
    UpdateCache_Backward(grad1_c, grad1_l, cache_list, cache_v_num, hid_dim, nullptr);
    UpdateCache_Backward(grad1_h, grad1_l, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time1 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |   UpdateCache(hc)  |  " << update_time1 << "ms  " << std::endl;

    Timer t5;
    gemm_cublas_backward(grad1_l, local_feature, grad1_l_W, false, true, hid_dim, in_dim, local_v_num);
    gemm_cublas_backward(grad1_c, cache_feature, grad1_c_W, false, true, hid_dim, in_dim, cache_v_num);
    gemm_cblas_backward(grad1_h, cache_host_feature, grad1_h_W, false, true, hid_dim, in_dim, host_cache_v_num);
    //gemm_cublas_backward(grad1_h, cache_host_feature, grad1_h_W, false, true, hid_dim, in_dim, host_cache_v_num);
    double mm_time1 = t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |     GEMM(lch)      |  " << mm_time1 << "ms  "<< std::endl;

    CUDA_CALL(cudaDeviceSynchronize());

    //GPU side
    opt1->Update(grad2_l_W, nullptr);
    opt2->Update(grad1_l_W, nullptr);
    opt3->Update(grad2_c_W, nullptr);
    opt4->Update(grad1_c_W, nullptr);
    opt5->Update(grad2_h_W, nullptr);
    //CPU side
    opt6->Update_CPU(grad1_h_W);
}

void GCN_Model::Backward_wCPUGPU_Cache_Stream(){
    cudaEvent_t Aggre1, Aggre2;
    //Create CUDA streams
    cudaStream_t stream1, stream2, stream3, stream4, stream5;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);

    //Until the first aggregation, the model performs common prerequisites
    cudaEventCreate(&Aggre1);
    SoftmaxCrossEntroyBackward(grad2_s, buff2_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, stream1);
    MPI_Barrier(MPI_COMM_WORLD);
    Aggregate_Backward_Warp_Stream(grad2_l, grad2_s, out_dim, stream1);
    gemm_cublas_backward_Stream(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, out_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre1, stream1);

    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream2, Aggre1, 0);
    cudaStreamWaitEvent(stream3, Aggre1, 0);
    cudaStreamWaitEvent(stream4, Aggre1, 0);
    cudaStreamWaitEvent(stream5, Aggre1, 0);  

    MPI_Barrier(MPI_COMM_WORLD);
    //stream2
    cudaEventCreate(&Aggre2);
    Aggregate_Backward_Warp_Stream(grad1_l, grad1, hid_dim, stream2);
    CUDA_CALL(cudaStreamSynchronize(stream2));
    cudaEventRecord(Aggre2, stream2);

    //stream3
    //Cached data(GPU side)
    UpdateCache_Backward_Stream(grad2_c, grad2_l, cache_list, cache_v_num, out_dim, stream3);
    gemm_cublas_backward_Stream(grad2_c, update_cache_feature, grad2_c_W, false, true, out_dim, hid_dim, cache_v_num, stream3);
    opt3->Update_stream(grad2_c_W, stream3);      

    //stream4
    //Cached data(host side)
    UpdateCache_Backward_Stream(grad2_h, grad2_l, cache_host_list, host_cache_v_num, out_dim, stream4);
    gemm_cublas_backward_Stream(grad2_h, update_cache_host_featrue, grad2_h_W, false, true, out_dim, hid_dim, host_cache_v_num, stream4);
    opt5->Update_stream(grad2_h_W,stream4);

    //stream5
    gemm_cublas_backward_Stream(grad2_l, buff1, grad2_l_W, false, true, out_dim, hid_dim, local_v_num, stream5);
    opt1->Update_stream(grad2_l_W, stream5);

    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream3, Aggre2, 0);
    cudaStreamWaitEvent(stream4, Aggre2, 0);
    cudaStreamWaitEvent(stream5, Aggre2, 0);

    //stream3
    //Cached data(GPU side)
    UpdateCache_Backward_Stream(grad1_c, grad1_l, cache_list, cache_v_num, hid_dim, stream3);
    gemm_cublas_backward_Stream(grad1_c, cache_feature, grad1_c_W, false, true, hid_dim, in_dim , cache_v_num, stream3);
    opt4->Update_stream(grad1_c_W, stream3);

    //Stream4
    //Cached data(host side)
    UpdateCache_Backward_Stream(grad1_h, grad1_l, cache_host_list, host_cache_v_num, hid_dim, stream4);
    gemm_cblas_backward(grad1_h, cache_host_feature, grad1_h_W, false, true, hid_dim, in_dim, host_cache_v_num);
    opt6->Update_CPU(grad1_h_W);

    //Stream5
    gemm_cublas_backward_Stream(grad1_l, local_feature, grad1_l_W, false, true, hid_dim, in_dim, local_v_num, stream5);
    opt2->Update_stream(grad1_l_W, stream5);

    //Sync all CUDA streams across the device
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @Backward_wCPUGPU_Cache_Stream: %s\n", cudaGetErrorString(error));
    }   

    //clean up
    cudaEventDestroy(Aggre1);
    cudaEventDestroy(Aggre2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    cudaStreamDestroy(stream5);      
}

void GCN_Model::Backward_wGPU_Cache(){
    //2nd layer GCN
    SoftmaxCrossEntroyBackward(grad2_s, buff2_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, nullptr);

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t0;
    Aggregate_Backward_Warp(grad2_l, grad2_s, out_dim, nullptr);
    double aggre_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |   Aggregate(lr)    |  " << aggre_time << "ms  " << std::endl;    

    Timer t1;
    UpdateCache_Backward(grad2_c, grad2_l, cache_list, cache_v_num, out_dim, nullptr);
    double update_time = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |   UpdateCache(c)   |  " << update_time << "ms  " << std::endl;    

    Timer t2;
    gemm_cublas_backward(grad2_l, buff1, grad2_l_W, false, true, out_dim, hid_dim, local_v_num); 
    gemm_cublas_backward(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, out_dim);
    gemm_cublas_backward(grad2_c, update_cache_feature, grad2_c_W, false, true, out_dim, hid_dim, cache_v_num);
    double mm_time = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |      GEMM(lc)      |  " << mm_time << "ms  "<< std::endl; 

    MPI_Barrier(MPI_COMM_WORLD);
    //1st layer GCN
    Timer t3;
    Aggregate_Backward_Warp(grad1_l, grad1, hid_dim, nullptr);
    double aggre_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |   Aggregate(lr)    |  " << aggre_time1 << "ms  " << std::endl;

    Timer t4;
    UpdateCache_Backward(grad1_c, grad1_l, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time1 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |   UpdateCache(c)   |  " << update_time1 << "ms  " << std::endl; 

    Timer t5;
    gemm_cublas_backward(grad1_l, local_feature, grad1_l_W, false, true, hid_dim, in_dim, local_v_num);
    gemm_cublas_backward(grad1_c, cache_feature, grad1_c_W, false, true, hid_dim, in_dim, cache_v_num);
    double mm_time1 = t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |      GEMM(lc)      |  " << mm_time1 << "ms  "<< std::endl;

    CUDA_CALL(cudaDeviceSynchronize());

    opt1->Update(grad2_l_W, nullptr);
    opt2->Update(grad1_l_W, nullptr);
    opt3->Update(grad2_c_W, nullptr);
    opt4->Update(grad1_c_W, nullptr);    
}

void GCN_Model::Backward_wGPU_Cache_Stream(){
    cudaEvent_t Aggre1, Aggre2;
    //Create CUDA streams
    cudaStream_t stream1, stream2, stream3, stream4, stream5, stream6;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);
    cudaStreamCreate(&stream6);

    //Until the first aggregation, the model performs common prerequisites
    cudaEventCreate(&Aggre1);
    SoftmaxCrossEntroyBackward(grad2_s, buff2_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, stream1);
    MPI_Barrier(MPI_COMM_WORLD);
    Aggregate_Backward_Warp_Stream(grad2_l, grad2_s, out_dim, stream1);
    gemm_cublas_backward_Stream(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, out_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre1, stream1);

    //Event 1 (first aggregation) is a dependency on streams 2, 3, and 4.
    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream2, Aggre1, 0);
    cudaStreamWaitEvent(stream3, Aggre1, 0);
    cudaStreamWaitEvent(stream4, Aggre1, 0);

    MPI_Barrier(MPI_COMM_WORLD);
    //stream2
    cudaEventCreate(&Aggre2);
    Aggregate_Backward_Warp_Stream(grad1_l, grad1, hid_dim, stream2);
    CUDA_CALL(cudaStreamSynchronize(stream2));
    cudaEventRecord(Aggre2, stream2);

    //stream3
    UpdateCache_Backward_Stream(grad2_c, grad2_l, cache_list, cache_v_num, out_dim, stream3);
    gemm_cublas_backward_Stream(grad2_c, update_cache_feature, grad2_c_W, false, true, out_dim, hid_dim, cache_v_num, stream3);
    opt3->Update_stream(grad2_c_W, stream3);

    //stream4
    gemm_cublas_backward_Stream(grad2_l, buff1, grad2_l_W, false, true, out_dim, hid_dim, local_v_num, stream4);
    opt1->Update_stream(grad2_l_W, stream4);

    //Event 2 (second aggregation) is a dependency on streams 5, and 6.
    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream5, Aggre2, 0);
    cudaStreamWaitEvent(stream6, Aggre2, 0);

    //Stream5
    UpdateCache_Backward_Stream(grad1_c, grad1_l, cache_list, cache_v_num, hid_dim, stream5);       
    gemm_cublas_backward_Stream(grad1_c, cache_feature, grad1_c_W, false, true, hid_dim, in_dim , cache_v_num, stream5);
    opt4->Update_stream(grad1_c_W, stream5);

    //Stream6
    gemm_cublas_backward_Stream(grad1_l, local_feature, grad1_l_W, false, true, hid_dim, in_dim, local_v_num, stream6);
    opt2->Update_stream(grad1_l_W, stream6);

    //Sync all CUDA streams across the device
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @Backward_wGPU_Cache_Stream: %s\n", cudaGetErrorString(error));
    } 

    //clean up
    cudaEventDestroy(Aggre1);
    cudaEventDestroy(Aggre2);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaStreamDestroy(stream3);
    cudaStreamDestroy(stream4);
    cudaStreamDestroy(stream5);
    cudaStreamDestroy(stream6);       
}

void GCN_Model::Backward_LocalOnly(){
    //2nd layer GCN
    SoftmaxCrossEntroyBackward(grad2_s, buff2_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, nullptr);

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t0;
    Aggregate_Backward_Warp(grad2_l, grad2_s, out_dim, nullptr);
    double aggre_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |    Aggregate(lr)   |  " << aggre_time << "ms  " << std::endl;    

    std::cout << "|  " <<  mynode << "   | Backward |   2   |    UpdateCache     |  "  << "0ms  " << std::endl;

    Timer t1;
    gemm_cublas_backward(grad2_l, buff1, grad2_l_W, false, true, out_dim, hid_dim, local_v_num); 
    gemm_cublas_backward(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, out_dim);
    double mm_time = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |    GEMM(local)     |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    //1st layer GCN
    Timer t2;
    Aggregate_Backward_Warp(grad1_l, grad1, hid_dim, nullptr);
    double aggre_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |   Aggregate(lr)    |  " << aggre_time1 << "ms  " << std::endl;

    std::cout << "|  " <<  mynode << "   | Backward |   1   |    UpdateCache     |  "  << "0ms  " << std::endl;

    Timer t3;
    gemm_cublas_backward(grad1_l, local_feature, grad1_l_W, false, true, hid_dim, in_dim, local_v_num);
    double mm_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |    GEMM(local)     |  " << mm_time1 << "ms  "<< std::endl;

    opt1->Update(grad2_l_W, nullptr);
    opt2->Update(grad1_l_W, nullptr);    
}

void GCN_Model::Backward_LocalOnly_Stream(){
    cudaEvent_t Aggre;
    //Create CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    //Until the first aggregation, the model performs common prerequisites
    cudaEventCreate(&Aggre);
    SoftmaxCrossEntroyBackward(grad2_s, buff2_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, stream1);
    MPI_Barrier(MPI_COMM_WORLD);
    Aggregate_Backward_Warp_Stream(grad2_l, grad2_s, out_dim, stream1);
    gemm_cublas_backward_Stream(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, out_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre, stream1);

    //CUDA streams waiting for synchronization.
    cudaStreamWaitEvent(stream2, Aggre, 0);

    //stream1
    MPI_Barrier(MPI_COMM_WORLD);
    Aggregate_Backward_Warp_Stream(grad1_l, grad1, hid_dim, stream1);
    gemm_cublas_backward_Stream(grad1_l, local_feature, grad1_l_W, false, true, hid_dim, in_dim, local_v_num, stream1);
    opt2->Update_stream(grad1_l_W, stream1);

    //stream2
    gemm_cublas_backward_Stream(grad2_l, buff1, grad2_l_W, false, true, out_dim, hid_dim, local_v_num, stream2);
    opt1->Update_stream(grad2_l_W, stream2);  

    //Sync all CUDA streams across the device
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @Backward_LocalOnly_Stream: %s\n", cudaGetErrorString(error));
    }  

    //clean up 
    cudaEventDestroy(Aggre); 
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
}

//======================================== GIN =================================================
void GIN_Model::Backward_Option(){
    if(mynode == 0) printf("| Rank |   Stage  | layer |      Operator     |    Time    \n");
    if(CacheFlag == true){
        if(UVMFlag == true){
            Backward_wCPUGPU_Cache();
        }else{
            Backward_wGPU_Cache();
        }
    }else{
        Backward_LocalOnly();
    }
}

void GIN_Model::Backward_Option_Stream(){
    if(mynode == 0) printf("| Rank |   Stage  | layer |      Operator     |    Time    \n");
    if(CacheFlag == true){
        if(UVMFlag == true){
            Backward_wCPUGPU_Cache_Stream();
        }else{
            Backward_wGPU_Cache_Stream();
        }
    }else{
        Backward_LocalOnly_Stream();
    }
}

void GIN_Model::Aggregate_Backward_Warp(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream){
    if(TransRemoteFlag == 1){
        leo_gin_local_backward_warp(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, T_row_ptr_r, T_col_ind_r, \
                                    local_v_num, dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, stream);
    }else{
        leo_gin_local_only_backward_warp(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, local_v_num, \
                                        dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, stream);
    }
}

void GIN_Model::Aggregate_Backward_Warp_Stream(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream){
    if(TransRemoteFlag == 1){
        leo_gin_local_backward_warp_wo_sync(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, T_row_ptr_r, T_col_ind_r, \
                                    local_v_num, dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, stream);
    }else{
        leo_gin_local_only_backward_warp_wo_sync(out_grad, in_grad, T_row_ptr_l, T_col_ind_l, local_v_num, \
                                        dim, nodePerPE, RunConfig::warpPerBlock, eps, mynode, stream);
    }
}

void GIN_Model::Backward_wCPUGPU_Cache(){

    SoftmaxCrossEntroyBackward(grad5_s, buff5_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, nullptr);

//5th layer GIN
    Timer t0;
    gemm_cublas_backward(grad5_s, buff5, grad5_W, false, true, out_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W6_l, grad5_s, grad5, true, false, hid_dim, local_v_num, out_dim);
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   5   |      GEMM(l)      |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t1;
    Aggregate_Backward_Warp(grad5_l, grad5, hid_dim, nullptr);
    double aggre_time = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   5   |   Aggregate(lr)   |  " << aggre_time << "ms  " << std::endl;

    Timer t2;
    UpdateCache_Backward(grad5_c, grad5_l, cache_list, cache_v_num, hid_dim, nullptr);
    UpdateCache_Backward(grad5_h, grad5_l, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   5   |   UpdateCache(hc) |  " << update_time << "ms  " << std::endl;

//4th layer GIN
    Timer t3;
    gemm_cublas_backward(grad5_l, buff4, grad4_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W5_l, grad5_l, grad4, true, false, hid_dim, local_v_num, hid_dim);
    gemm_cublas_backward(grad5_c, buff4_update_c, grad4_c_W, false, true, hid_dim, hid_dim, cache_v_num);
    gemm_cublas_backward(grad5_h, buff4_update_h, grad4_h_W, false, true, hid_dim, hid_dim, host_cache_v_num);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   4   |     GEMM(lch)     |  " << mm_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t4;
    Aggregate_Backward_Warp(grad4_l, grad4, hid_dim, nullptr);
    double aggre_time1 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   4   |   Aggregate(lr)   |  " << aggre_time1 << "ms  " << std::endl;

    Timer t5;
    UpdateCache_Backward(grad4_c, grad4_l, cache_list, cache_v_num, hid_dim, nullptr);
    UpdateCache_Backward(grad4_h, grad4_l, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time1 = t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   4   |   UpdateCache(hc) |  " << update_time1 << "ms  " << std::endl;

//3rd layer GIN
    Timer t6;
    gemm_cublas_backward(grad4_l, buff3, grad3_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W4_l, grad4_l, grad3, true, false, hid_dim, local_v_num, hid_dim);
    gemm_cublas_backward(grad4_c, buff3_update_c, grad3_c_W, false, true, hid_dim, hid_dim, cache_v_num);
    gemm_cublas_backward(grad4_h, buff3_update_h, grad3_h_W, false, true, hid_dim, hid_dim, host_cache_v_num);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time2 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   3   |     GEMM(lch)     |  " << mm_time2 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t7;
    Aggregate_Backward_Warp(grad3_l, grad3, hid_dim, nullptr);
    double aggre_time2 = t7.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   3   |   Aggregate(lr)   |  " << aggre_time2 << "ms  " << std::endl;

    Timer t8;
    UpdateCache_Backward(grad3_c, grad3_l, cache_list, cache_v_num, hid_dim, nullptr);
    UpdateCache_Backward(grad3_h, grad3_l, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time2 =t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   3   |   UpdateCache(hc) |  " << update_time2 << "ms  " << std::endl;

//2nd layer GIN
    Timer t9;
    gemm_cublas_backward(grad3_l, buff2, grad2_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W3_l, grad3_l, grad2, true, false, hid_dim, local_v_num, hid_dim);
    gemm_cublas_backward(grad3_c, buff2_update_c, grad2_c_W, false, true, hid_dim, hid_dim, cache_v_num);
    gemm_cublas_backward(grad3_h, buff2_update_h, grad2_h_W, false, true, hid_dim, hid_dim, host_cache_v_num);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time3 = t9.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |     GEMM(lch)     |  " << mm_time3 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t10;
    Aggregate_Backward_Warp(grad2_l, grad2, hid_dim, nullptr);
    double aggre_time3 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |   Aggregate(lr)   |  " << aggre_time3 << "ms  " << std::endl;

    Timer t11;
    UpdateCache_Backward(grad2_c, grad2_l, cache_list, cache_v_num, hid_dim, nullptr);
    UpdateCache_Backward(grad2_h, grad2_l, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time3 = t11.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |   UpdateCache(hc) |  " << update_time3 << "ms  " << std::endl;

//1st layer GIN
    Timer t12;
    gemm_cublas_backward(grad2_l, buff1, grad1_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, hid_dim);
    gemm_cublas_backward(grad2_c, buff1_update_c, grad1_c_W, false, true, hid_dim, hid_dim, cache_v_num);
    gemm_cublas_backward(grad2_h, buff1_update_h, grad1_h_W, false, true, hid_dim, hid_dim, host_cache_v_num);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time4 = t12.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |     GEMM(lch)     |  " << mm_time4 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t13;
    Aggregate_Backward_Warp(grad1_l, grad1, hid_dim, nullptr);
    double aggre_time4 = t13.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |   Aggregate(lr)   |  " << aggre_time4 << "ms  " << std::endl;

    Timer t14;
    UpdateCache_Backward(grad1_c, grad1_l, cache_list, cache_v_num, hid_dim, nullptr);
    UpdateCache_Backward(grad1_h, grad1_l, cache_host_list, host_cache_v_num, hid_dim, nullptr);
    double update_time4 = t14.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |   UpdateCache(hc) |  " << update_time4 << "ms  " << std::endl;

//prepared matrix multiplication
    Timer t15;
    gemm_cublas_backward(grad1_l, local_feature, grad0_l_W, false, true, hid_dim, in_dim, local_v_num);
    gemm_cublas_backward(grad1_c, cache_feature, grad0_c_W, false, true, hid_dim, in_dim, cache_v_num);
    gemm_cblas_backward(grad1_h, cache_host_feature, grad0_h_W, false, true, hid_dim, in_dim, host_cache_v_num);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time5 = t15.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   0   |     GEMM(lch)     |  " << mm_time5 << "ms  "<< std::endl;

    opt1->Update(grad5_W, nullptr);
    opt2->Update(grad4_l_W, nullptr);
    opt3->Update(grad3_l_W, nullptr);
    opt4->Update(grad2_l_W, nullptr);
    opt5->Update(grad1_l_W, nullptr);
    opt6->Update(grad0_l_W, nullptr);

    opt7->Update(grad4_c_W, nullptr);
    opt8->Update(grad3_c_W, nullptr);
    opt9->Update(grad2_c_W, nullptr);
    opt10->Update(grad1_c_W, nullptr);
    opt11->Update(grad0_c_W, nullptr);

    opt12->Update(grad4_h_W, nullptr);
    opt13->Update(grad3_h_W, nullptr);
    opt14->Update(grad2_h_W, nullptr);
    opt15->Update(grad1_h_W, nullptr);
    opt16->Update_CPU(grad0_h_W);
    CUDA_CALL(cudaDeviceSynchronize());
}

void GIN_Model::Backward_wCPUGPU_Cache_Stream(){
    cudaEvent_t Aggre1, Aggre2, Aggre3, Aggre4, Aggre5;
    //Create CUDA streams
    cudaStream_t stream1, stream2, stream3, stream4, stream5;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);
    cudaStreamCreate(&stream5);

    SoftmaxCrossEntroyBackward(grad5_s, buff5_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, stream1);

//5th layer GIN
    Timer t0;
    gemm_cublas_backward_Stream(W6_l, grad5_s, grad5, true, false, hid_dim, local_v_num, out_dim, stream1);
    cudaEventCreate(&Aggre1);
    MPI_Barrier(MPI_COMM_WORLD);
    Aggregate_Backward_Warp_Stream(grad5_l, grad5, hid_dim, stream1);
    gemm_cublas_backward_Stream(W5_l, grad5_l, grad4, true, false, hid_dim, local_v_num, hid_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre1, stream1);

    //CUDA streams waiting for synchronization
    cudaStreamWaitEvent(stream2, Aggre1, 0);
    cudaStreamWaitEvent(stream3, Aggre1, 0);
    cudaStreamWaitEvent(stream4, Aggre1, 0);
    cudaStreamWaitEvent(stream5, Aggre1, 0);

    MPI_Barrier(MPI_COMM_WORLD);
//4th layer GIN

    //stream1
    cudaEventCreate(&Aggre2);
    Aggregate_Backward_Warp_Stream(grad4_l, grad4, hid_dim, stream1);
    gemm_cublas_backward_Stream(W4_l, grad4_l, grad3, true, false, hid_dim, local_v_num, hid_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre2, stream1);

    //stream2
    gemm_cublas_backward_Stream(grad5_s, buff5, grad5_W, false, true, out_dim, hid_dim, local_v_num, stream2);
    opt1->Update_stream(grad5_W, stream2);

    //stream3
    UpdateCache_Backward_Stream(grad5_c, grad5_l, cache_list, cache_v_num, hid_dim, stream3);
    gemm_cublas_backward_Stream(grad5_c, buff4_update_c, grad4_c_W, false, true, hid_dim, hid_dim, cache_v_num, stream3);
    opt7->Update_stream(grad4_c_W, stream3);

    //stream4
    UpdateCache_Backward_Stream(grad5_h, grad5_l, cache_host_list, host_cache_v_num, hid_dim, stream4);
    gemm_cublas_backward_Stream(grad5_h, buff4_update_h, grad4_h_W, false, true, hid_dim, hid_dim, host_cache_v_num, stream4);
    opt12->Update_stream(grad4_h_W, stream4);

    //stream5
    gemm_cublas_backward_Stream(grad5_l, buff4, grad4_l_W, false, true, hid_dim, hid_dim, local_v_num, stream5);
    opt2->Update_stream(grad4_l_W, stream5);

    //CUDA streams waiting for synchronization
    cudaStreamWaitEvent(stream2, Aggre2, 0);
    cudaStreamWaitEvent(stream3, Aggre2, 0);
    cudaStreamWaitEvent(stream4, Aggre2, 0); 

    MPI_Barrier(MPI_COMM_WORLD);
//3rd layer GIN

    //stream1
    cudaEventCreate(&Aggre3);
    Aggregate_Backward_Warp_Stream(grad3_l, grad3, hid_dim, stream1);
    gemm_cublas_backward_Stream(W3_l, grad3_l, grad2, true, false, hid_dim, local_v_num, hid_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre3, stream1);    

    //stream2
    UpdateCache_Backward_Stream(grad4_c, grad4_l, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_backward_Stream(grad4_c, buff3_update_c, grad3_c_W, false, true, hid_dim, hid_dim, cache_v_num, stream2);
    opt8->Update_stream(grad3_c_W, stream2);

    //stream3
    UpdateCache_Backward_Stream(grad4_h, grad4_l, cache_host_list, host_cache_v_num, hid_dim, stream3);
    gemm_cublas_backward_Stream(grad4_h, buff3_update_h, grad3_h_W, false, true, hid_dim, hid_dim, host_cache_v_num, stream3);
    opt13->Update_stream(grad3_h_W, stream3);

    //stream4
    gemm_cublas_backward_Stream(grad4_l, buff3, grad3_l_W, false, true, hid_dim, hid_dim, local_v_num, stream4);
    opt3->Update_stream(grad3_l_W, stream4);

    cudaStreamWaitEvent(stream2, Aggre3, 0);
    cudaStreamWaitEvent(stream3, Aggre3, 0);
    cudaStreamWaitEvent(stream4, Aggre3, 0);   

    MPI_Barrier(MPI_COMM_WORLD);
//2nd layer GIN
    
    //stream1
    cudaEventCreate(&Aggre4);
    Aggregate_Backward_Warp_Stream(grad2_l, grad2, hid_dim, stream1);
    gemm_cublas_backward_Stream(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, hid_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre4, stream1);    

    //stream2
    UpdateCache_Backward_Stream(grad3_c, grad3_l, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_backward_Stream(grad3_c, buff2_update_c, grad2_c_W, false, true, hid_dim, hid_dim, cache_v_num, stream2);
    opt9->Update_stream(grad2_c_W, stream2);

    //stream3
    UpdateCache_Backward_Stream(grad3_h, grad3_l, cache_host_list, host_cache_v_num, hid_dim, stream3);
    gemm_cublas_backward_Stream(grad3_h, buff2_update_h, grad2_h_W, false, true, hid_dim, hid_dim, host_cache_v_num, stream3);
    opt14->Update_stream(grad2_h_W, stream3);

    //stream4
    gemm_cublas_backward_Stream(grad3_l, buff2, grad2_l_W, false, true, hid_dim, hid_dim, local_v_num, stream4);
    opt4->Update_stream(grad2_l_W, stream4);

    cudaStreamWaitEvent(stream2, Aggre4, 0);
    cudaStreamWaitEvent(stream3, Aggre4, 0);
    cudaStreamWaitEvent(stream4, Aggre4, 0);  

    MPI_Barrier(MPI_COMM_WORLD);
//1st layer GIN

    //stream1
    cudaEventCreate(&Aggre5);
    Aggregate_Backward_Warp_Stream(grad1_l, grad1, hid_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre5, stream1);    

    //stream2
    UpdateCache_Backward_Stream(grad2_c, grad2_l, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_backward_Stream(grad2_c, buff1_update_c, grad1_c_W, false, true, hid_dim, hid_dim, cache_v_num, stream2);
    opt10->Update_stream(grad1_c_W, stream2);

    //stream3
    UpdateCache_Backward_Stream(grad2_h, grad2_l, cache_host_list, host_cache_v_num, hid_dim, stream3);
    gemm_cublas_backward_Stream(grad2_h, buff1_update_h, grad1_h_W, false, true, hid_dim, hid_dim, host_cache_v_num, stream3);
    opt15->Update_stream(grad1_h_W, stream3);

    //stream4
    gemm_cublas_backward_Stream(grad2_l, buff1, grad1_l_W, false, true, hid_dim, hid_dim, local_v_num, stream4);
    opt5->Update_stream(grad1_l_W, stream4);

    cudaStreamWaitEvent(stream2, Aggre5, 0);
    cudaStreamWaitEvent(stream3, Aggre5, 0);
    cudaStreamWaitEvent(stream4, Aggre5, 0); 

    //stream2
    UpdateCache_Backward_Stream(grad1_c, grad1_l, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_backward_Stream(grad1_c, cache_feature, grad0_c_W, false, true, hid_dim, in_dim, cache_v_num, stream2);
    opt11->Update_stream(grad0_c_W, stream2);

    //stream3
    UpdateCache_Backward_Stream(grad1_h, grad1_l, cache_host_list, host_cache_v_num, hid_dim, stream3);
    gemm_cblas_backward(grad1_h, cache_host_feature, grad0_h_W, false, true, hid_dim, in_dim, host_cache_v_num);
    opt16->Update_CPU(grad0_h_W);

    //stream4
    gemm_cublas_backward_Stream(grad1_l, local_feature, grad0_l_W, false, true, hid_dim, in_dim, local_v_num, stream4);
    opt6->Update_stream(grad0_l_W, stream4);
}


void GIN_Model::Backward_wGPU_Cache(){

    SoftmaxCrossEntroyBackward(grad5_s, buff5_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, nullptr);

//5th layer GIN
    Timer t0;
    gemm_cublas_backward(grad5_s, buff5, grad5_W, false, true, out_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W6_l, grad5_s, grad5, true, false, hid_dim, local_v_num, out_dim);
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   5   |      GEMM(l)      |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t1;
    Aggregate_Backward_Warp(grad5_l, grad5, hid_dim, nullptr);
    double aggre_time = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   5   |   Aggregate(lr)   |  " << aggre_time << "ms  " << std::endl;

    Timer t2;
    UpdateCache_Backward(grad5_c, grad5_l, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   5   |   UpdateCache(c)  |  " << update_time << "ms  " << std::endl;            

//4th layer GIN
    Timer t3;
    gemm_cublas_backward(grad5_l, buff4, grad4_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W5_l, grad5_l, grad4, true, false, hid_dim, local_v_num, hid_dim);
    gemm_cublas_backward(grad5_c, buff4_update_c, grad4_c_W, false, true, hid_dim, hid_dim, cache_v_num);    
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   4   |      GEMM(lc)     |  " << mm_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t4;
    Aggregate_Backward_Warp(grad4_l, grad4, hid_dim, nullptr);
    double aggre_time1 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   4   |   Aggregate(lr)   |  " << aggre_time1 << "ms  " << std::endl;

    Timer t5;
    UpdateCache_Backward(grad4_c, grad4_l, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time1 = t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   4   |   UpdateCache(c)  |  " << update_time1 << "ms  " << std::endl;        

//3rd layer
    Timer t6;
    gemm_cublas_backward(grad4_l, buff3, grad3_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W4_l, grad4_l, grad3, true, false, hid_dim, local_v_num, hid_dim);
    gemm_cublas_backward(grad4_c, buff3_update_c, grad3_c_W, false, true, hid_dim, hid_dim, cache_v_num);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time2 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   3   |      GEMM(lc)     |  " << mm_time2 << "ms  "<< std::endl;  

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t7;
    Aggregate_Backward_Warp(grad3_l, grad3, hid_dim, nullptr);
    double aggre_time2 = t7.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   3   |   Aggregate(lr)   |  " << aggre_time2 << "ms  " << std::endl;

    Timer t8;
    UpdateCache_Backward(grad3_c, grad3_l, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time2 =t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   3   |   UpdateCache(c)  |  " << update_time2 << "ms  " << std::endl;

//2nd layer GIN
    Timer t9;
    gemm_cublas_backward(grad3_l, buff2, grad2_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W3_l, grad3_l, grad2, true, false, hid_dim, local_v_num, hid_dim);
    gemm_cublas_backward(grad3_c, buff2_update_c, grad2_c_W, false, true, hid_dim, hid_dim, cache_v_num);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time3 = t9.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |      GEMM(lc)     |  " << mm_time3 << "ms  "<< std::endl;    

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t10;
    Aggregate_Backward_Warp(grad2_l, grad2, hid_dim, nullptr);
    double aggre_time3 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |   Aggregate(lr)   |  " << aggre_time3 << "ms  " << std::endl;

    Timer t11;
    UpdateCache_Backward(grad2_c, grad2_l, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time3 = t11.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |   UpdateCache(c)  |  " << update_time3 << "ms  " << std::endl;  

//1st layer GIN
    Timer t12;
    gemm_cublas_backward(grad2_l, buff1, grad1_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, hid_dim);
    gemm_cublas_backward(grad2_c, buff1_update_c, grad1_c_W, false, true, hid_dim, hid_dim, cache_v_num);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time4 = t12.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |      GEMM(lc)     |  " << mm_time4 << "ms  "<< std::endl;  

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t13;
    Aggregate_Backward_Warp(grad1_l, grad1, hid_dim, nullptr);
    double aggre_time4 = t13.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |   Aggregate(lr)   |  " << aggre_time4 << "ms  " << std::endl;

    Timer t14;
    UpdateCache_Backward(grad1_c, grad1_l, cache_list, cache_v_num, hid_dim, nullptr);
    double update_time4 = t14.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |   UpdateCache(hc) |  " << update_time4 << "ms  " << std::endl;  

//prepared matrix multiplication
    Timer t15;
    gemm_cublas_backward(grad1_l, local_feature, grad0_l_W, false, true, hid_dim, in_dim, local_v_num);
    gemm_cublas_backward(grad1_c, cache_feature, grad0_c_W, false, true, hid_dim, in_dim, cache_v_num);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time5 = t15.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   0   |      GEMM(lc)     |  " << mm_time5 << "ms  "<< std::endl;              

    opt1->Update(grad5_W, nullptr);
    opt2->Update(grad4_l_W, nullptr);
    opt3->Update(grad3_l_W, nullptr);
    opt4->Update(grad2_l_W, nullptr);
    opt5->Update(grad1_l_W, nullptr);
    opt6->Update(grad0_l_W, nullptr);

    opt7->Update(grad4_c_W, nullptr);
    opt8->Update(grad3_c_W, nullptr);
    opt9->Update(grad2_c_W, nullptr);
    opt10->Update(grad1_c_W, nullptr);
    opt11->Update(grad0_c_W, nullptr);
    CUDA_CALL(cudaDeviceSynchronize());
}

void GIN_Model::Backward_wGPU_Cache_Stream(){
    cudaEvent_t Aggre1, Aggre2, Aggre3, Aggre4, Aggre5;
    //Create CUDA streams
    cudaStream_t stream1, stream2, stream3, stream4;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);
    cudaStreamCreate(&stream3);
    cudaStreamCreate(&stream4);

    SoftmaxCrossEntroyBackward(grad5_s, buff5_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, stream1); 

//5th layer GIN
    Timer t0;
    gemm_cublas_backward_Stream(W6_l, grad5_s, grad5, true, false, hid_dim, local_v_num, out_dim, stream1);
    cudaEventCreate(&Aggre1);
    MPI_Barrier(MPI_COMM_WORLD);
    Aggregate_Backward_Warp_Stream(grad5_l, grad5, hid_dim, stream1);
    gemm_cublas_backward_Stream(W5_l, grad5_l, grad4, true, false, hid_dim, local_v_num, hid_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre1, stream1);

    //CUDA streams waiting for synchronization
    cudaStreamWaitEvent(stream2, Aggre1, 0);
    cudaStreamWaitEvent(stream3, Aggre1, 0);
    cudaStreamWaitEvent(stream4, Aggre1, 0);

    MPI_Barrier(MPI_COMM_WORLD);
//4th layer GIN

    //stream1
    cudaEventCreate(&Aggre2);
    Aggregate_Backward_Warp_Stream(grad4_l, grad4, hid_dim, stream1);
    gemm_cublas_backward_Stream(W4_l, grad4_l, grad3, true, false, hid_dim, local_v_num, hid_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre2, stream1);

    //stream2
    gemm_cublas_backward_Stream(grad5_s, buff5, grad5_W, false, true, out_dim, hid_dim, local_v_num, stream2);
    opt1->Update_stream(grad5_W, stream2);

    //stream3
    UpdateCache_Backward_Stream(grad5_c, grad5_l, cache_list, cache_v_num, hid_dim, stream3);
    gemm_cublas_backward_Stream(grad5_c, buff4_update_c, grad4_c_W, false, true, hid_dim, hid_dim, cache_v_num, stream3);
    opt7->Update_stream(grad4_c_W, stream3); 

    //stream4
    gemm_cublas_backward_Stream(grad5_l, buff4, grad4_l_W, false, true, hid_dim, hid_dim, local_v_num, stream4);
    opt2->Update_stream(grad4_l_W, stream4);
    
    //CUDA streams waiting for synchronization
    cudaStreamWaitEvent(stream2, Aggre2, 0);
    cudaStreamWaitEvent(stream3, Aggre2, 0);

    MPI_Barrier(MPI_COMM_WORLD);
//3rd layer GIN

    //stream1
    cudaEventCreate(&Aggre3);
    Aggregate_Backward_Warp_Stream(grad3_l, grad3, hid_dim, stream1);
    gemm_cublas_backward_Stream(W3_l, grad3_l, grad2, true, false, hid_dim, local_v_num, hid_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre3, stream1);  

    //stream2
    UpdateCache_Backward_Stream(grad4_c, grad4_l, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_backward_Stream(grad4_c, buff3_update_c, grad3_c_W, false, true, hid_dim, hid_dim, cache_v_num, stream2);
    opt8->Update_stream(grad3_c_W, stream2);

    //stream3
    gemm_cublas_backward_Stream(grad4_l, buff3, grad3_l_W, false, true, hid_dim, hid_dim, local_v_num, stream3);
    opt3->Update_stream(grad3_l_W, stream3);

    //CUDA streams waiting for synchronization
    cudaStreamWaitEvent(stream2, Aggre3, 0);
    cudaStreamWaitEvent(stream3, Aggre3, 0);

    MPI_Barrier(MPI_COMM_WORLD);
//2nd layer GIN

    //stream1
    cudaEventCreate(&Aggre4);
    Aggregate_Backward_Warp_Stream(grad2_l, grad2, hid_dim, stream1);
    gemm_cublas_backward_Stream(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, hid_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre4, stream1);  

    //stream2
    UpdateCache_Backward_Stream(grad3_c, grad3_l, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_backward_Stream(grad3_c, buff2_update_c, grad2_c_W, false, true, hid_dim, hid_dim, cache_v_num, stream2);
    opt9->Update_stream(grad2_c_W, stream2);

    //stream3
    gemm_cublas_backward_Stream(grad3_l, buff2, grad2_l_W, false, true, hid_dim, hid_dim, local_v_num, stream3);
    opt4->Update_stream(grad2_l_W, stream3);

    //CUDA streams waiting for synchronization
    cudaStreamWaitEvent(stream2, Aggre4, 0);
    cudaStreamWaitEvent(stream3, Aggre4, 0);

    MPI_Barrier(MPI_COMM_WORLD);
//1st layer GIN

    //stream1
    cudaEventCreate(&Aggre5);
    Aggregate_Backward_Warp_Stream(grad1_l, grad1, hid_dim, stream1);
    CUDA_CALL(cudaStreamSynchronize(stream1));
    cudaEventRecord(Aggre5, stream1); 

    //stream2
    UpdateCache_Backward_Stream(grad2_c, grad2_l, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_backward_Stream(grad2_c, buff1_update_c, grad1_c_W, false, true, hid_dim, hid_dim, cache_v_num, stream2);
    opt10->Update_stream(grad1_c_W, stream2);

    //stream3
    gemm_cublas_backward_Stream(grad2_l, buff1, grad1_l_W, false, true, hid_dim, hid_dim, local_v_num, stream3);
    opt5->Update_stream(grad1_l_W, stream3);

    //CUDA streams waiting for synchronization
    cudaStreamWaitEvent(stream2, Aggre5, 0);
    cudaStreamWaitEvent(stream3, Aggre5, 0);

    //stream2
    UpdateCache_Backward_Stream(grad1_c, grad1_l, cache_list, cache_v_num, hid_dim, stream2);
    gemm_cublas_backward_Stream(grad1_c, cache_feature, grad0_c_W, false, true, hid_dim, in_dim, cache_v_num, stream2);
    opt11->Update_stream(grad0_c_W, stream2);

    //stream3
    gemm_cublas_backward_Stream(grad1_l, local_feature, grad0_l_W, false, true, hid_dim, in_dim, local_v_num, stream3);
    opt6->Update_stream(grad0_l_W, stream3);
}

void GIN_Model::Backward_LocalOnly(){

    SoftmaxCrossEntroyBackward(grad5_s, buff5_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, nullptr);

//5th layer GIN
    Timer t0;
    gemm_cublas_backward(grad5_s, buff5, grad5_W, false, true, out_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W6_l, grad5_s, grad5, true, false, hid_dim, local_v_num, out_dim);
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   5   |       GEMM(l)     |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t1;
    Aggregate_Backward_Warp(grad5_l, grad5, hid_dim, nullptr);
    double aggre_time = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   5   |   Aggregate(lr)   |  " << aggre_time << "ms  " << std::endl;

//4th layer GIN
    Timer t2;
    gemm_cublas_backward(grad5_l, buff4, grad4_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W5_l, grad5_l, grad4, true, false, hid_dim, local_v_num, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   4   |       GEMM(l)     |  " << mm_time1 << "ms  "<< std::endl;    

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t3;
    Aggregate_Backward_Warp(grad4_l, grad4, hid_dim, nullptr);
    double aggre_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   4   |   Aggregate(lr)   |  " << aggre_time1 << "ms  " << std::endl;

//3rd layer GIN
    Timer t4;
    gemm_cublas_backward(grad4_l, buff3, grad3_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W4_l, grad4_l, grad3, true, false, hid_dim, local_v_num, hid_dim);  
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   3   |       GEMM(l)     |  " << mm_time2 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t5;
    Aggregate_Backward_Warp(grad3_l, grad3, hid_dim, nullptr);
    double aggre_time2 =  t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   3   |   Aggregate(lr)   |  " << aggre_time2 << "ms  " << std::endl;

//2nd layer GIN
    Timer t6;
    gemm_cublas_backward(grad3_l, buff2, grad2_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W3_l, grad3_l, grad2, true, false, hid_dim, local_v_num, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time3 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |       GEMM(l)     |  " << mm_time3 << "ms  "<< std::endl;          

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t7;
    Aggregate_Backward_Warp(grad2_l, grad2, hid_dim, nullptr);
    double aggre_time3 = t7.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |   Aggregate(lr)   |  " << aggre_time3 << "ms  " << std::endl;

//1st layer GIN
    Timer t8;
    gemm_cublas_backward(grad2_l, buff1, grad1_l_W, false, true, hid_dim, hid_dim, local_v_num);
    gemm_cublas_backward(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, hid_dim);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time4 = t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |       GEMM(l)     |  " << mm_time4 << "ms  "<< std::endl;    

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t9;
    Aggregate_Backward_Warp(grad1_l, grad1, hid_dim, nullptr);
    double aggre_time4 = t9.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |   Aggregate(lr)   |  " << aggre_time4 << "ms  " << std::endl;

//prepared matrix multiplication
    Timer t10;
    gemm_cublas_backward(grad1_l, local_feature, grad0_l_W, false, true, hid_dim, in_dim, local_v_num);
    double mm_time5 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   0   |       GEMM(l)     |  " << mm_time5 << "ms  "<< std::endl;     

    opt1->Update(grad5_W, nullptr);
    opt2->Update(grad4_l_W, nullptr);
    opt3->Update(grad3_l_W, nullptr);
    opt4->Update(grad2_l_W, nullptr);
    opt5->Update(grad1_l_W, nullptr);
    opt6->Update(grad0_l_W, nullptr);
    CUDA_CALL(cudaDeviceSynchronize());
}

void GIN_Model::Backward_LocalOnly_Stream(){

    SoftmaxCrossEntroyBackward(grad5_s, buff5_s, label, local_v_num, out_dim, RunConfig::warpPerBlock, nullptr);

//5th layer GIN
    Timer t0;
    gemm_cublas_backward_Stream(grad5_s, buff5, grad5_W, false, true, out_dim, hid_dim, local_v_num, nullptr);
    gemm_cublas_backward_Stream(W6_l, grad5_s, grad5, true, false, hid_dim, local_v_num, out_dim, nullptr);
    opt1->Update_stream(grad5_W, nullptr);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time = t0.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   5   |       GEMM(l)     |  " << mm_time << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t1;
    Aggregate_Backward_Warp(grad5_l, grad5, hid_dim, nullptr);
    double aggre_time = t1.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   5   |   Aggregate(lr)   |  " << aggre_time << "ms  " << std::endl;

//4th layer GIN
    Timer t2;
    gemm_cublas_backward_Stream(grad5_l, buff4, grad4_l_W, false, true, hid_dim, hid_dim, local_v_num, nullptr);
    gemm_cublas_backward_Stream(W5_l, grad5_l, grad4, true, false, hid_dim, local_v_num, hid_dim, nullptr);
    opt2->Update_stream(grad4_l_W, nullptr);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time1 = t2.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   4   |       GEMM(l)     |  " << mm_time1 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t3;
    Aggregate_Backward_Warp(grad4_l, grad4, hid_dim, nullptr);
    double aggre_time1 = t3.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   4   |   Aggregate(lr)   |  " << aggre_time1 << "ms  " << std::endl;

//3th layer GIN
    Timer t4;
    gemm_cublas_backward_Stream(grad4_l, buff3, grad3_l_W, false, true, hid_dim, hid_dim, local_v_num, nullptr);
    gemm_cublas_backward_Stream(W4_l, grad4_l, grad3, true, false, hid_dim, local_v_num, hid_dim, nullptr);
    opt3->Update_stream(grad3_l_W, nullptr);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time2 = t4.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   3   |       GEMM(l)     |  " << mm_time2 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t5;
    Aggregate_Backward_Warp(grad3_l, grad3, hid_dim, nullptr);
    double aggre_time2 =  t5.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   3   |   Aggregate(lr)   |  " << aggre_time2 << "ms  " << std::endl;

//2rd layer GIN
    Timer t6;
    gemm_cublas_backward_Stream(grad3_l, buff2, grad2_l_W, false, true, hid_dim, hid_dim, local_v_num, nullptr);
    gemm_cublas_backward_Stream(W3_l, grad3_l, grad2, true, false, hid_dim, local_v_num, hid_dim, nullptr);
    opt4->Update_stream(grad2_l_W, nullptr);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time3 = t6.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |       GEMM(l)     |  " << mm_time3 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t7;
    Aggregate_Backward_Warp(grad2_l, grad2, hid_dim, nullptr);
    double aggre_time3 = t7.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   2   |   Aggregate(lr)   |  " << aggre_time3 << "ms  " << std::endl;

//1st layer GIN
    Timer t8;
    gemm_cublas_backward_Stream(grad2_l, buff1, grad1_l_W, false, true, hid_dim, hid_dim, local_v_num, nullptr);
    gemm_cublas_backward_Stream(W2_l, grad2_l, grad1, true, false, hid_dim, local_v_num, hid_dim, nullptr);
    opt5->Update_stream(grad1_l_W, nullptr);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time4 = t8.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |       GEMM(l)     |  " << mm_time4 << "ms  "<< std::endl;

    MPI_Barrier(MPI_COMM_WORLD);
    Timer t9;
    Aggregate_Backward_Warp(grad1_l, grad1, hid_dim, nullptr);
    double aggre_time4 = t9.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |   Aggregate(lr)   |  " << aggre_time4 << "ms  " << std::endl;

//prepared matrix multiplication
    Timer t10;
    gemm_cublas_backward_Stream(grad1_l, local_feature, grad0_l_W, false, true, hid_dim, in_dim, local_v_num, nullptr);
    opt6->Update_stream(grad0_l_W, nullptr);
    CUDA_CALL(cudaDeviceSynchronize());
    double mm_time5 = t10.PassedMilli();
    std::cout << "|  " <<  mynode << "   | Backward |   1   |       GEMM(l)     |  " << mm_time5 << "ms  "<< std::endl;                
}

void GIN_Model::UpdateCache_Backward(value_t* out_grad, value_t* in_grad, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream){
    leo_update_cache_backward_block(out_grad, in_grad, cache_list, cache_num, dim, nodePerPE, mynode, stream);
}

void GIN_Model::UpdateCache_Backward_Stream(value_t* out_grad, value_t* in_grad, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream){
    leo_update_cache_backward_block_wo_sync(out_grad, in_grad, cache_list, cache_num, dim, nodePerPE, mynode, stream);
}

//==========================================Validate================================================

void GCN_Model::Validate(){

    if(RunConfig::validate_cublas == true) Validate_cublas_mm();
    if(RunConfig::validate_aggregation == true) Validate_leo_aggregate();
    if(RunConfig::validate_infer == true) Validate_infer();   
    // printf("myID is %d\n", getpid());  // 输出当前进程的进程ID
    // int j=1;         
    // while(j){
    //   sleep(2);  // 陷入休眠，避免执行到程序异常处，导致中途退出
    // } 
}

void GIN_Model::Validate(){
    //LOG(ERROR) << "To be deployed";   
}

void GCN_Model::Validate_cublas_mm(){
    if(posix_memalign((void**)&this->v_buff1_l, getpagesize(),
                        sizeof(cache_feat_t) * hid_dim * local_v_num))
    perror("posix_mamalign");

    CUDA_CALL(cudaMemcpy(v_buff1_l, buff1_l, hid_dim * local_v_num * sizeof(cache_feat_t), cudaMemcpyDeviceToHost));
}

void GCN_Model::Validate_leo_aggregate(){
    if(posix_memalign((void**)&this->v_buff1, getpagesize(),
                        sizeof(cache_feat_t) * hid_dim * local_v_num))
    perror("posix_mamalign");

    CUDA_CALL(cudaMemcpy(v_buff1, buff1, hid_dim * local_v_num * sizeof(cache_feat_t), cudaMemcpyDeviceToHost));    
}

void GCN_Model::Validate_infer(){
    if(posix_memalign((void**)&this->v_buff_infer, getpagesize(),
                        sizeof(cache_feat_t) * out_dim * local_v_num))
    perror("posix_manalign");

    CUDA_CALL(cudaMemcpy(v_buff_infer, buff2_s, out_dim * local_v_num * sizeof(cache_feat_t), cudaMemcpyDeviceToHost));
}

}   //namespace common
}   //namescpce GNNPro_lib 