#include <unistd.h>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cuda.h>
#include "graph.h"
#include "logging.cuh"

namespace GNNPro_lib{
namespace common{

void Graph::Device_data_initial(){
    VertexID local_node_size = ID_UpperBound - ID_LowerBound;

    CUDA_CALL(cudaMalloc((void**)&d_local_feature, local_node_size * config->in_dim * sizeof(cache_feat_t)));

if(CacheFlag == true){

    CUDA_CALL(cudaMalloc((void**)&d_cache_feature, cache_num * config->in_dim * sizeof(cache_feat_t)));
    CUDA_CALL(cudaMalloc((void**)&d_cache_feature_update, cache_num * config->hidden_dim * sizeof(cache_feat_t)));

    CUDA_CALL(cudaMalloc((void**)&d_row_ptr_c, cache_csr_ptr.size() * sizeof(CSR_t)));
    CUDA_CALL(cudaMalloc((void**)&d_col_ind_c, cache_csr_ind.size() * sizeof(CSR_t)));
    CUDA_CALL(cudaMalloc((void**)&d_cache_list, cache_num * sizeof(VertexID)));

    if(HostCacheFlag == true){

        CUDA_CALL(cudaMalloc((void**)&d_cache_host_feature_update, h_cache_num * config->in_dim * sizeof(CSR_t)));

        CUDA_CALL(cudaMalloc((void**)&d_row_ptr_h, host_csr_ptr.size() * sizeof(CSR_t)));
        CUDA_CALL(cudaMalloc((void**)&d_col_ind_h, host_csr_ind.size() * sizeof(CSR_t)));
        CUDA_CALL(cudaMalloc((void**)&d_cache_host_list, h_cache_num * sizeof(VertexID)));
    } //HostCacheFlag

} //CacheFlag

    CUDA_CALL(cudaMalloc((void**)&d_row_ptr_l, local_csr_ptr.size() * sizeof(CSR_t)));
    CUDA_CALL(cudaMalloc((void**)&d_col_ind_l, local_csr_ind.size() * sizeof(CSR_t)));

if(RemoteFlag == true){

    CUDA_CALL(cudaMalloc((void**)&d_row_ptr_r, remote_csr_ptr.size() * sizeof(CSR_t)));
    CUDA_CALL(cudaMalloc((void**)&d_col_ind_r, remote_csr_ind.size() * sizeof(CSR_t)));
} //RemoteFlag

    CUDA_CALL(cudaMalloc((void**)&d_label, local_node_size * sizeof(int)));
    CUDA_CALL(cudaMalloc((void**)&d_id_map, local_IDmap.size() * sizeof(int)));

if(TrainFlag == true){

    CUDA_CALL(cudaMalloc((void**)&d_loss, local_node_size * sizeof(value_t)));

    CUDA_CALL(cudaMalloc((void**)&d_trans_row_ptr_l, local_csr_T_ptr.size() * sizeof(CSR_t)));
    CUDA_CALL(cudaMalloc((void**)&d_trans_col_ind_l, local_csr_T_ind.size() * sizeof(CSR_t)));

    if(TransRemoteFlag == true){
        CUDA_CALL(cudaMalloc((void**)&d_trans_row_ptr_r, remote_csr_T_ptr.size() * sizeof(CSR_t)));
        CUDA_CALL(cudaMalloc((void**)&d_trans_col_ind_r, remote_csr_T_ind.size() * sizeof(CSR_t)));
    }//TransRemoteFlag

} //TrainFlag

} //Device_data_initial


void Graph::Copy_Host2Device(){
    VertexID local_node_size = ID_UpperBound - ID_LowerBound; 

    // CUDA_CALL(cudaMemcpy(d_local_feature, feature + (ID_LowerBound * config->in_dim), \
    //                     (local_node_size * config->in_dim)*sizeof(cache_feat_t), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMemcpy(d_local_feature, feature, \
                        (local_node_size * config->in_dim)*sizeof(cache_feat_t), cudaMemcpyHostToDevice));

if(CacheFlag == true){
    //CUDA_CALL(cudaMemcpy(d_cache_feature, cache_feature, (cache_node_size * config->in_dim)*sizeof(cache_feat_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_cache_feature, &cached_feature[0], cached_feature.size()*sizeof(file_feat_t), cudaMemcpyHostToDevice));

    CUDA_CALL(cudaMemcpy(d_row_ptr_c, &cache_csr_ptr[0], cache_csr_ptr.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_col_ind_c, &cache_csr_ind[0], cache_csr_ind.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_cache_list, &cache_IDlist[0], cache_IDlist.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));
   
    if(HostCacheFlag == true){
        CUDA_CALL(cudaMemcpy(d_row_ptr_h, &host_csr_ptr[0], host_csr_ptr.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_col_ind_h, &host_csr_ind[0], host_csr_ind.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_cache_host_list, &cache_host_IDlist[0], cache_host_IDlist.size()*sizeof(VertexID), cudaMemcpyHostToDevice));
    } //HostCacheFlag

} //CacheFlag

    CUDA_CALL(cudaMemcpy(d_row_ptr_l, &local_csr_ptr[0], local_csr_ptr.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_col_ind_l, &local_csr_ind[0], local_csr_ind.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));

if(RemoteFlag == true){

    CUDA_CALL(cudaMemcpy(d_row_ptr_r, &remote_csr_ptr[0], remote_csr_ptr.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_col_ind_r, &remote_csr_ind[0], remote_csr_ind.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));
} //RemoteFlag

    int *label_offset = &label[ID_LowerBound];
    CUDA_CALL(cudaMemcpy(d_label, label_offset, local_node_size * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_id_map, &local_IDmap[0], local_IDmap.size()*sizeof(int), cudaMemcpyHostToDevice));

if(TrainFlag == true){

    CUDA_CALL(cudaMemcpy(d_trans_row_ptr_l, &local_csr_T_ptr[0], local_csr_T_ptr.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(d_trans_col_ind_l, &local_csr_T_ind[0], local_csr_T_ind.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));

    if(TransRemoteFlag == true){
        CUDA_CALL(cudaMemcpy(d_trans_row_ptr_r, &remote_csr_T_ptr[0], remote_csr_T_ptr.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(d_trans_col_ind_r, &remote_csr_T_ind[0], remote_csr_T_ind.size()*sizeof(CSR_t), cudaMemcpyHostToDevice));
    }//TransRemoteFlag

} //TrainFlag

} //Copy_Host2Device

void Graph::Weight_Initial(){
    Create_Weight();

    if(config->algorithm == "GCN"){
        CUDA_CALL(cudaMalloc((void**)&dW1_l, config->in_dim * config->hidden_dim * sizeof(weight_t)));
        CUDA_CALL(cudaMalloc((void**)&dW2_l, config->hidden_dim * config->out_dim * sizeof(weight_t)));

        CUDA_CALL(cudaMemcpy(dW1_l, &weight->W1[0], weight->W1.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dW2_l, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));

        if(CacheFlag == true){

            CUDA_CALL(cudaMalloc((void**)&dW1_c, config->in_dim * config->hidden_dim * sizeof(weight_t)));
            CUDA_CALL(cudaMalloc((void**)&dW2_c, config->hidden_dim * config->out_dim * sizeof(weight_t)));

            CUDA_CALL(cudaMemcpy(dW1_c, &weight->W1[0], weight->W1.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(dW2_c, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));

            if(HostCacheFlag == true){
                CUDA_CALL(cudaMallocManaged(&dW1_h, config->in_dim * config->hidden_dim * sizeof(weight_t)));
                CUDA_CALL(cudaMalloc((void**)&dW2_h, config->hidden_dim * config->out_dim * sizeof(weight_t)));
                
                memcpy(dW1_h, &weight->W1[0], weight->W1.size()*sizeof(weight_t));
                CUDA_CALL(cudaMemcpy(dW2_h, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
            }
        }

    }else if (config->algorithm == "GIN"){    
        CUDA_CALL(cudaMalloc((void**)&dW1_l, config->in_dim * config->hidden_dim * sizeof(weight_t)));
        CUDA_CALL(cudaMalloc((void**)&dW2_l, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
        CUDA_CALL(cudaMalloc((void**)&dW3_l, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
        CUDA_CALL(cudaMalloc((void**)&dW4_l, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
        CUDA_CALL(cudaMalloc((void**)&dW5_l, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
        CUDA_CALL(cudaMalloc((void**)&dW6_l, config->hidden_dim * config->out_dim * sizeof(weight_t)));

        CUDA_CALL(cudaMemcpy(dW1_l, &weight->W1[0], weight->W1.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dW2_l, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dW3_l, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dW4_l, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dW5_l, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(dW6_l, &weight->W3[0], weight->W3.size()*sizeof(weight_t), cudaMemcpyHostToDevice));

        if(CacheFlag == 1){
            CUDA_CALL(cudaMalloc((void**)&dW1_c, config->in_dim * config->hidden_dim * sizeof(weight_t)));
            CUDA_CALL(cudaMalloc((void**)&dW2_c, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
            CUDA_CALL(cudaMalloc((void**)&dW3_c, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
            CUDA_CALL(cudaMalloc((void**)&dW4_c, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
            CUDA_CALL(cudaMalloc((void**)&dW5_c, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
        
            CUDA_CALL(cudaMemcpy(dW1_c, &weight->W1[0], weight->W1.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(dW2_c, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(dW3_c, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(dW4_c, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
            CUDA_CALL(cudaMemcpy(dW5_c, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));

            if(HostCacheFlag == true){
                CUDA_CALL(cudaMallocManaged(&dW1_h, config->in_dim * config->hidden_dim * sizeof(weight_t)));
                CUDA_CALL(cudaMalloc((void**)&dW2_h, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
                CUDA_CALL(cudaMalloc((void**)&dW3_h, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
                CUDA_CALL(cudaMalloc((void**)&dW4_h, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
                CUDA_CALL(cudaMalloc((void**)&dW5_h, config->hidden_dim * config->hidden_dim * sizeof(weight_t)));
                
                memcpy(dW1_h, &weight->W1[0], weight->W1.size()*sizeof(weight_t));
                CUDA_CALL(cudaMemcpy(dW2_h, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
                CUDA_CALL(cudaMemcpy(dW3_h, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
                CUDA_CALL(cudaMemcpy(dW4_h, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
                CUDA_CALL(cudaMemcpy(dW5_h, &weight->W2[0], weight->W2.size()*sizeof(weight_t), cudaMemcpyHostToDevice));
            }
        }

    }else{
        LOG(ERROR) << "Unsupport layer setting!";
    }
    
}

void Graph::Buff_Initial(){
    VertexID  local_node_size = ID_UpperBound - ID_LowerBound;
    VertexID  nodesPerPE = (vertices_num + nranks - 1) / nranks;

    if(config->algorithm == "GCN"){    
//NVSHMEM's shared memory is required even without remote and cached neighbors in a multi-GPU environment.
        if (NVSHMEMFlag == 1){ 
            d_buff1_l = (cache_feat_t *) nvshmem_malloc (nodesPerPE * config->hidden_dim * sizeof(cache_feat_t));
            d_buff2_l = (cache_feat_t *) nvshmem_malloc (nodesPerPE * config->out_dim * sizeof(cache_feat_t));
            d_buff1 = (cache_feat_t *) nvshmem_malloc (nodesPerPE * config->hidden_dim * sizeof(cache_feat_t));
            d_buff2 = (cache_feat_t *) nvshmem_malloc (nodesPerPE * config->out_dim * sizeof(cache_feat_t));
        }else{
            CUDA_CALL(cudaMalloc((void**)&d_buff1_l, local_node_size * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff2_l, local_node_size * config->out_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff1, local_node_size * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff2, local_node_size * config->out_dim * sizeof(cache_feat_t)));
        }

        if (CacheFlag == 1){ 
            CUDA_CALL(cudaMalloc((void**)&d_buff1_c, cache_num * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff2_c, cache_num * config->out_dim * sizeof(cache_feat_t)));

            if(HostCacheFlag == true){
                CUDA_CALL(cudaMallocManaged(&d_buff1_h, h_cache_num * config->hidden_dim * sizeof(cache_feat_t)));
                CUDA_CALL(cudaMalloc((void**)&d_buff2_h, h_cache_num * config->out_dim * sizeof(cache_feat_t)));
            }
        }

        //CUDA_CALL(cudaMalloc((void**)&d_buff1, local_node_size * config->hidden_dim * sizeof(cache_feat_t)));
        CUDA_CALL(cudaMalloc((void**)&d_buff2_s, local_node_size * config->out_dim * sizeof(cache_feat_t)));

    }else if (config->algorithm == "GIN"){
//NVSHMEM's shared memory is required even without remote and cached neighbors in a multi-GPU environment.        
        if(NVSHMEMFlag == 1){
            d_buff1_l = (cache_feat_t *) nvshmem_malloc (nodesPerPE * config->hidden_dim * sizeof(cache_feat_t));
            d_buff2_l = (cache_feat_t *) nvshmem_malloc (nodesPerPE * config->hidden_dim * sizeof(cache_feat_t)); 
            d_buff1 = (cache_feat_t *) nvshmem_malloc (nodesPerPE * config->hidden_dim * sizeof(cache_feat_t));
            d_buff2 = (cache_feat_t *) nvshmem_malloc (nodesPerPE *config->hidden_dim * sizeof(cache_feat_t));
            d_buff3 = (cache_feat_t *) nvshmem_malloc (nodesPerPE *config->hidden_dim * sizeof(cache_feat_t));
            d_buff4 = (cache_feat_t *) nvshmem_malloc (nodesPerPE *config->hidden_dim * sizeof(cache_feat_t));
            d_buff5 = (cache_feat_t *) nvshmem_malloc (nodesPerPE *config->hidden_dim * sizeof(cache_feat_t));
            d_buff5_l = (cache_feat_t *) nvshmem_malloc (nodesPerPE *config->out_dim * sizeof(cache_feat_t));

        }else{
            CUDA_CALL(cudaMalloc((void**)&d_buff1_l, local_node_size * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff2_l, local_node_size * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff1, local_node_size * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff2, local_node_size * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff3, local_node_size * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff4, local_node_size * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff5, local_node_size * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff5_l, local_node_size * config->out_dim * sizeof(cache_feat_t)));
        }

        if(CacheFlag == 1){
            CUDA_CALL(cudaMalloc((void**)&d_buff1_c, cache_num * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff2_c, cache_num * config->hidden_dim * sizeof(cache_feat_t)));

            CUDA_CALL(cudaMalloc((void**)&d_buff1_update_c, cache_num * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff2_update_c, cache_num * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff3_update_c, cache_num * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff4_update_c, cache_num * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff5_update_c, cache_num * config->hidden_dim * sizeof(cache_feat_t)));
            CUDA_CALL(cudaMalloc((void**)&d_buff6_update_c, cache_num * config->hidden_dim * sizeof(cache_feat_t)));

            if(HostCacheFlag == 1){
                CUDA_CALL(cudaMallocManaged(&d_buff1_h, h_cache_num * config->hidden_dim * sizeof(cache_feat_t)));
                CUDA_CALL(cudaMalloc((void**)&d_buff2_h, h_cache_num * config->hidden_dim * sizeof(cache_feat_t)));
            
                CUDA_CALL(cudaMalloc((void**)&d_buff1_update_h, h_cache_num * config->hidden_dim * sizeof(cache_feat_t)));
                CUDA_CALL(cudaMalloc((void**)&d_buff2_update_h, h_cache_num * config->hidden_dim * sizeof(cache_feat_t)));
                CUDA_CALL(cudaMalloc((void**)&d_buff3_update_h, h_cache_num * config->hidden_dim * sizeof(cache_feat_t)));
                CUDA_CALL(cudaMalloc((void**)&d_buff4_update_h, h_cache_num * config->hidden_dim * sizeof(cache_feat_t)));
                CUDA_CALL(cudaMalloc((void**)&d_buff5_update_h, h_cache_num * config->hidden_dim * sizeof(cache_feat_t)));
                CUDA_CALL(cudaMalloc((void**)&d_buff6_update_h, h_cache_num * config->hidden_dim * sizeof(cache_feat_t)));
            }   
        }

        CUDA_CALL(cudaMalloc((void**)&d_buff5_s, local_node_size * config->out_dim * sizeof(cache_feat_t)));

    }else{
       LOG(ERROR) << "Unsupport layer setting!"; 
    }

    if(TrainFlag == 1) Grad_buff_Initial();
}

void Graph::Grad_buff_Initial(){
    VertexID  local_node_size = ID_UpperBound - ID_LowerBound;
    VertexID  nodesPerPE = (vertices_num + nranks - 1) / nranks;

    if(config->algorithm == "GCN"){
        if (NVSHMEMFlag == 1){
            d_grad1 = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));
            d_grad1_l = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));

            d_grad2_s = (value_t *)nvshmem_malloc(nodesPerPE * config->out_dim * sizeof(value_t));
            d_grad2_l = (value_t *)nvshmem_malloc(nodesPerPE * config->out_dim * sizeof(value_t));
        }

        CUDA_CALL(cudaMalloc((void**)&d_grad1_l_W, config->in_dim * config->hidden_dim * sizeof(value_t)));
        CUDA_CALL(cudaMalloc((void**)&d_grad2_l_W, config->hidden_dim * config->out_dim * sizeof(value_t)));

        if(CacheFlag == true){
            CUDA_CALL(cudaMalloc((void**)&d_grad1_c, cache_num * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad2_c, cache_num * config->out_dim * sizeof(value_t)));

            CUDA_CALL(cudaMalloc((void**)&d_grad1_c_W, config->in_dim * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad2_c_W, config->hidden_dim * config->out_dim * sizeof(value_t)));
            
            if(UVMFlag == true){
                CUDA_CALL(cudaMallocManaged(&d_grad1_h, h_cache_num * config->hidden_dim * sizeof(value_t)));
                CUDA_CALL(cudaMalloc((void**)&d_grad2_h, h_cache_num * config->out_dim *sizeof(value_t)));

                if(posix_memalign((void **)&h_grad1_h_W, getpagesize(),
                        sizeof(value_t) * config->in_dim * config->hidden_dim))
                perror("posix_mamalign");
                CUDA_CALL(cudaMalloc((void**)&d_grad2_h_W, config->hidden_dim * config->out_dim * sizeof(value_t)));
            }
        }
    }else if (config->algorithm == "GIN"){
        if(NVSHMEMFlag == 1){

            d_grad1_l = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));
            d_grad1 = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));
            d_grad2_l = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));
            d_grad2 = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));
            d_grad3_l = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));
            d_grad3 = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));
            d_grad4_l = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));
            d_grad4 = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));
            d_grad5_l = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));
            d_grad5 = (value_t *)nvshmem_malloc(nodesPerPE * config->hidden_dim * sizeof(value_t));
        }else{

            CUDA_CALL(cudaMalloc((void**)&d_grad1_l, local_node_size * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad1, local_node_size * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad2_l, local_node_size * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad2, local_node_size * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad3_l, local_node_size * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad3, local_node_size * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad4_l, local_node_size * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad4, local_node_size * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad5_l, local_node_size * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad5, local_node_size * config->hidden_dim * sizeof(value_t)));
        }

        CUDA_CALL(cudaMalloc((void**)&d_grad0_l_W, config->in_dim * config->hidden_dim * sizeof(value_t)));
        CUDA_CALL(cudaMalloc((void**)&d_grad1_l_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));
        CUDA_CALL(cudaMalloc((void**)&d_grad2_l_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));
        CUDA_CALL(cudaMalloc((void**)&d_grad3_l_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));
        CUDA_CALL(cudaMalloc((void**)&d_grad4_l_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));

        CUDA_CALL(cudaMalloc((void**)&d_grad5_W, config->hidden_dim * config->out_dim * sizeof(value_t)));
        CUDA_CALL(cudaMalloc((void**)&d_grad5_s, local_node_size * config->out_dim * sizeof(value_t)));
    
        if(CacheFlag == true){

            CUDA_CALL(cudaMalloc((void**)&d_grad1_c, cache_num * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad2_c, cache_num * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad3_c, cache_num * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad4_c, cache_num * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad5_c, cache_num * config->hidden_dim * sizeof(value_t)));

            CUDA_CALL(cudaMalloc((void**)&d_gard0_c_W, config->in_dim * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad1_c_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad2_c_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad3_c_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));
            CUDA_CALL(cudaMalloc((void**)&d_grad4_c_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));
            
            if(UVMFlag == true){

                CUDA_CALL(cudaMallocManaged(&d_grad1_h, h_cache_num * config->hidden_dim * sizeof(value_t)));
                CUDA_CALL(cudaMalloc((void**)&d_grad2_h, h_cache_num * config->hidden_dim * sizeof(value_t)));
                CUDA_CALL(cudaMalloc((void**)&d_grad3_h, h_cache_num * config->hidden_dim * sizeof(value_t)));
                CUDA_CALL(cudaMalloc((void**)&d_grad4_h, h_cache_num * config->hidden_dim * sizeof(value_t)));
                CUDA_CALL(cudaMalloc((void**)&d_grad5_h, h_cache_num * config->hidden_dim * sizeof(value_t)));

                if(posix_memalign((void **)&h_grad0_h_W, getpagesize(),
                        sizeof(value_t) * config->in_dim * config->hidden_dim))
                perror("posix_mamalign");
                CUDA_CALL(cudaMalloc((void**)&d_grad1_h_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));
                CUDA_CALL(cudaMalloc((void**)&d_grad2_h_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));
                CUDA_CALL(cudaMalloc((void**)&d_grad3_h_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));
                CUDA_CALL(cudaMalloc((void**)&d_grad4_h_W, config->hidden_dim * config->hidden_dim * sizeof(value_t)));
            }
        }
    }
}

void Graph::GPU_Initial(){

    Timer t0;
    Device_data_initial();
    double initial_time = t0.PassedMilli();
    std::cout << "Rank " << mynode << ": Data Initialization on device =====> " << initial_time << "ms" << std::endl;
 
    Timer t1;
    Copy_Host2Device();
    double copy_time = t1.PassedMilli();
    std::cout << "Rank " << mynode << ": Copy from host to device =====> " << copy_time << "ms" << std::endl;

    Timer t2;
    Weight_Initial();
    double weight_time = t2.PassedMilli();
    std::cout << "Rank " << mynode <<": Generate weight matrix and copy it to GPU " << std::endl;

    Buff_Initial();
}

void Graph::GPU_Initial_v1(){

    Timer t0;
    Device_data_initial();
    double initial_time = t0.PassedMilli();
    std::cout << "Rank " << mynode << ": Data Initialization on device =====> " << initial_time << "ms" << std::endl;
 
    Timer t1;
    Copy_Host2Device();
    double copy_time = t1.PassedMilli();
    std::cout << "Rank " << mynode << ": Copy from host to device =====> " << copy_time << "ms" << std::endl;

    Timer t2;
    Weight_Initial();
    double weight_time = t2.PassedMilli();
    std::cout << "Rank " << mynode <<": Generate weight matrix and copy it to GPU " << std::endl;    
}

void Graph::Free_Device_Memory(){
    cudaFree(d_local_feature);
    cudaFree(d_cache_feature);
    cudaFree(d_cache_feature_update);
    cudaFree(d_cache_host_feature_update);

    cudaFree(d_row_ptr_l);
    cudaFree(d_col_ind_l);
    cudaFree(d_row_ptr_c);
    cudaFree(d_col_ind_c);
    cudaFree(d_row_ptr_r);
    cudaFree(d_col_ind_r);
    cudaFree(d_col_ind_h);
    cudaFree(d_col_ind_h);

    cudaFree(d_trans_row_ptr_l);
    cudaFree(d_trans_col_ind_l);
    cudaFree(d_trans_row_ptr_r);
    cudaFree(d_trans_col_ind_r);

    cudaFree(d_label);
    cudaFree(d_id_map);
    cudaFree(d_loss);
    cudaFree(d_cache_list);
    cudaFree(d_cache_host_list);

    cudaFree(dW1_l);
    cudaFree(dW1_c);
    cudaFree(dW1_h);
    cudaFree(dW2_l);
    cudaFree(dW2_c);
    cudaFree(dW2_h);
    cudaFree(dW3_l);
    cudaFree(dW3_c);

    cudaFree(d_buff1_l);
    cudaFree(d_buff1_c);
    cudaFree(d_buff1_h);
    cudaFree(d_buff1);

    cudaFree(d_buff2_l);
    cudaFree(d_buff2_c);
    cudaFree(d_buff2_h);
    cudaFree(d_buff2);
    cudaFree(d_buff2_s);

    cudaFree(d_grad1_l_W);
    cudaFree(d_grad1_c_W);
    cudaFree(d_grad1_l);
    cudaFree(d_grad1_c);
    cudaFree(d_grad1_h);
    cudaFree(d_grad1);

    cudaFree(d_grad2_l_W);
    cudaFree(d_grad2_c_W);
    cudaFree(d_grad2_h_W);
    cudaFree(d_grad2_l);
    cudaFree(d_grad2_c);
    cudaFree(d_grad2_h);
    cudaFree(d_grad2_s);

}


}   //namespace common
}   //namescpce GNNPro_lib  