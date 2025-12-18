#include "kernel.cuh"
#include "logging.cuh"

namespace GNNPro_lib{
namespace common{

//=================================Forward Propagation===============================================

void leo_lrc_block(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream   
);

void leo_lr_block(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream      
);

void leo_l_block(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream      
);

void leo_update_cache_block(
    cache_feat_t*       output,
    const cache_feat_t* input,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode,
    const cudaStream_t  stream     
);

void leo_update_cache_backward_block_wo_sync(
    value_t*            gradients,
    const value_t*      input_grad,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode,
    const cudaStream_t  stream
);

void leo_lchr_block_V1( //V1 for version 1
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           overlap_dist,
    const int           myid,
    const cudaStream_t  stream   
);

void leo_lchr_block_V2(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           myid,
    const cudaStream_t  stream     
);

void leo_lrc_warp(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream 
);

void leo_lr_warp(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream   
);

void leo_l_warp(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream 
);

void leo_lchr_warp(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream 
);

void leo_lcr_block_v1(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           myid,
    const cudaStream_t  stream    
);

void leo_lcr_block_v2(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           myid,
    const cudaStream_t  stream    
);

void leo_lcr_block_v3(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           myid,
    const cudaStream_t  stream   
);

void leo_lcr_block_v4(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           myid,
    const cudaStream_t  stream  
);

void leo_lcr_warp_v1(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream     
);

void leo_lchr_warp_V1(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream 
);

void leo_lcr_thread(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           myid,
    const cudaStream_t  stream      
);

void leo_l_thread(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           myid,
    const cudaStream_t  stream
);

void leo_lchr_thread(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           myid,
    const cudaStream_t  stream
);

void leo_gin_lrc_block(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream 
);

void leo_gin_lr_block(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream
);

void leo_gin_l_block(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream      
);

void leo_gin_lchr_block(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           overlap_dist,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream 
);

void leo_gin_lcr_block_v1(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream
);

void leo_gin_lr_block_v1(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream
);

void leo_gin_l_block_v1(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream    
);

void leo_gin_lchr_block_v1(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream 
);

void leo_gin_lcr_warp(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream
);

void leo_gin_lr_warp(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream    
);

void leo_gin_l_warp(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream 
);

void leo_gin_lchr_warp(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream 
);

void UVM_GCN_block(
    float*              d_out,
    float**             d_in,
    const VertexID*     d_row_ptr,
    const VertexID*     d_col_ind,
    const int           local_node_num,
    const int           num_nodes,
    const int           dim,
    const int           num_GPUs,
    const int           myGPUid,
    const int           nodePerPE
);

void UVM_GIN_block(
    float*              d_out,
    float**             d_in,
    const VertexID*     d_row_ptr,
    const VertexID*     d_col_ind,
    const int           local_node_num,
    const int           num_nodes,
    const int           dim,
    const int           num_GPUs,
    const int           myGPUid,
    const int           nodePerPE,
    const float         eps
);
//=================================Back Propagation================================================

void leo_local_backward_block(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const CSR_t*        T_ptr_r,
    const CSR_t*        T_ind_r,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream     
);

void leo_local_backward_block_wo_sync(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const CSR_t*        T_ptr_r,
    const CSR_t*        T_ind_r,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream      
);

void leo_local_only_backward_block(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream     
);

void leo_local_only_backward_block_wo_sync(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const VertexID      local_node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream     
);

void leo_local_backward_warp(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const CSR_t*        T_ptr_r,
    const CSR_t*        T_ind_r,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream     
);

void leo_local_backward_warp_wo_sync(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const CSR_t*        T_ptr_r,
    const CSR_t*        T_ind_r,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream     
);

void leo_local_only_backward_warp(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream  
);

void leo_local_only_backward_warp_wo_sync(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           myid,
    const cudaStream_t  stream  
);

void leo_gin_local_backward_warp(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const CSR_t*        T_ptr_r,
    const CSR_t*        T_ind_r,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream 
);

void leo_gin_local_backward_warp_wo_sync(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const CSR_t*        T_ptr_r,
    const CSR_t*        T_ind_r,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream 
);

void leo_gin_local_only_backward_warp(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream     
);


void leo_gin_local_only_backward_warp_wo_sync(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        T_ptr_l,
    const CSR_t*        T_ind_l,
    const VertexID      local_node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps,
    const int           myid,
    const cudaStream_t  stream
);

void UVM_GCN_back_block(
    float*              gradients,
    float**             input_grad,
    const VertexID*     d_row_ptr,
    const VertexID*     d_col_ind,
    const int           local_node_num,
    const int           num_nodes,
    const int           dim,
    const int           num_GPUs,
    const int           myGPUid,
    const int           nodePerPE    
);

void UVM_GIN_back_block(
    float*              gradients,
    float**             input_grad,
    const VertexID*     d_row_ptr,
    const VertexID*     d_col_ind,
    const int           local_node_num,
    const int           num_nodes,
    const int           dim,
    const int           num_GPUs,
    const int           myGPUid,
    const int           nodePerPE,
    const float         eps
);

void leo_update_cache_backward_block(
    value_t*            gradients,
    const value_t*      input_grad,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode,
    const cudaStream_t  stream
);

void leo_update_cache_backward_block_wo_sync(
    value_t*            gradients,
    const value_t*      input_grad,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode,
    const cudaStream_t  stream
);

void SoftmaxCrossEntroyBackward(
    value_t*            gradients,
    const cache_feat_t* softmax_output, 
    const int*          labels, 
    const int           local_node_num,
    const int           dim,
    const int           warpPerBlock,
    const cudaStream_t  stream 
);

}   //namespace common
}   //namescpce GNNPro_lib 