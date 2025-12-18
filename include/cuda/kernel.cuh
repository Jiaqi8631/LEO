#ifndef kernel_cuh
#define kernel_cuh

#include <cuda_runtime_api.h>
#include <cudnn.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <cmath>
#include "type.h"

namespace GNNPro_lib{
namespace common{

//=================================Forward Propagation================================================
__global__
void leo_lrc_block_cuda(
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
    const VertexID      node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock
);

__global__
void leo_lr_block_cuda( //Processing local and remote neighbors
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock 
);

__global__
void leo_l_block_cuda( //Processing only local neighbors
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock    
);

__global__
void leo_lrhc_block_v1_cuda(
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
    const VertexID      node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           overlap_dist
);

__global__
void leo_lrc_warp_cuda(
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
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock
);

__global__
void leo_lr_warp_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock     
);

__global__
void leo_l_warp_cuda( //Processing only local neighbors
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock    
);

__global__
void leo_lrhc_warp_cuda(
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
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock  
);

__global__
void leo_lcr_block_v1_cuda(
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
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE
);

__global__
void leo_lc_block_v2_cuda( // part1
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE 
);

__global__
void leo_r_block_v2_cuda(// part 2
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE   
);

__global__
void leo_hr_block_v2_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE
);

__global__
void leo_lcr_block_v3_cuda(
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
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE 
);

__global__
void leo_lcr_block_v4_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           myrank
);

__global__
void leo_lc_warp_v1_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock    
);

__global__
void leo_r_warp_v1_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock
);

__global__
void leo_hr_warp_v1_cuda(
    cache_feat_t* output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock         
);

__global__
void leo_lcr_thread_cuda(
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
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE
);

__global__
void leo_l_thread_cuda( //Processing only local neighbors
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE
);

__global__
void leo_lchr_thread_cuda(
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
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE      
);

__global__
void leo_gin_lrc_block_cuda(
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
    const VertexID      node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps
);

__global__
void leo_gin_lr_block_cuda( //Processing local and remote neighbors
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps
);

__global__
void leo_gin_l_block_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps
);

__global__
void leo_gin_lrhc_block_cuda(
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
    const VertexID      node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const int           overlap_dist,
    const float         eps
);

__global__
void leo_gin_lc_block_v1_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const float         eps
);

__global__
void leo_gin_r_block_v1_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE    
);

__global__
void leo_gin_lr_block_v1_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const float         eps
);

__global__
void leo_gin_l_block_v1_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const float         eps    
);

__global__
void leo_gin_hr_block_v1_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const float         eps
);

__global__
void leo_gin_lc_warp_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_c,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_c,
    const CSR_t*        ind_c,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps
);

__global__
void leo_gin_r_warp_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps    
);

__global__
void leo_gin_lr_warp_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps    
);

__global__
void leo_gin_l_warp_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps
);

__global__
void leo_gin_hr_warp_cuda(
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const cache_feat_t* input_h,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const CSR_t*        ptr_h,
    const CSR_t*        ind_h,
    const int*          id_map,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps
);

__global__
void UVM_gcn_forward_cuda(
    float*          output,
    float**         input,
    const CSR_t*    row_pointers,
    const CSR_t*    column_index,
    const int       numNodes,
    const int       dim,
    const int       nodePerPE,
    const int       partSize,
    const int       warpPerBlock,
    const int       myGPUid 
);

__global__
void UVM_gin_forward_cuda(
    float*          output,
    float**         input,
    const CSR_t*    row_pointers,
    const CSR_t*    column_index,
    const int       numNodes,
    const int       dim,
    const int       nodePerPE,
    const int       partSize,
    const int       warpPerBlock,
    const int       myGPUid,
    const float     eps
);

__global__
void leo_update_cache_block_cuda( //Update cached node feature
    cache_feat_t*       output,
    const cache_feat_t* input,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode
);

__global__
void CrossEntropyLoss(const int* labels, const cache_feat_t* outputs, value_t* loss, int  num_samples, int  num_classes);

//=================================Back Propagation================================================

__global__
void leo_local_grad_backward_block_cuda(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const VertexID      node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock
);

__global__
void leo_local_only_grad_backward_block_cuda(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const VertexID      node_num,
    const int           dim,
    const int           partSize,
    const VertexID      nodePerPE,
    const int           warpPerBlock 
);

__global__
void leo_local_grad_backward_warp_V1_cuda(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock   
);

__global__
void leo_local_only_grad_backward_warp_V1_cuda(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock 
);

__global__
void leo_gin_local_grad_backward_warp_cuda(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const CSR_t*        ptr_r,
    const CSR_t*        ind_r,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps     
);

__global__
void leo_gin_local_only_grad_backward_warp_cuda(
    value_t*            gradients,
    const value_t*      input_grad,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           warpPerBlock,
    const float         eps   
);

__global__
void UVM_gcn_backward_cuda(
    float*          gradients,
    float**         input_grad,
    const CSR_t*    row_pointers,
    const CSR_t*    column_index,
    const int       numNodes,
    const int       dim,
    const int       nodePerPE,
    const int       partSize,
    const int       warpPerBlock,
    const int       myGPUid     
);

__global__
void UVM_gin_backward_cuda(
    float*          gradients,
    float**         input_grad,
    const CSR_t*    row_pointers,
    const CSR_t*    column_index,
    const int       numNodes,
    const int       dim,
    const int       nodePerPE,
    const int       partSize,
    const int       warpPerBlock,
    const int       myGPUid,
    const float     eps
);

__global__
void leo_update_cahce_backward_block_cuda( //Update cached node gradients in back propagation
    value_t*            gradients,
    const value_t*      input_grad,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode
);

__global__
void leo_update_cache_backward_warp_cuda( //Update cached node gradients in back propagation(Warp Version)
    value_t*            gradients,
    const value_t*      input_grad,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode
);

__global__
void SoftmaxCrossEntroy_backward_cuda(
    value_t*            gradients,
    const cache_feat_t* softmax_output, 
    const int*          labels, 
    const int           num_samples,
    const int           num_classes
);

__global__
void AdamUpdate_cuda(
    value_t*        out_param,
    const value_t*  gradients,
    value_t*        m,
    value_t*        v,
    int             size,
    float           beta1,
    float           beta2,
    float           eps,
    float           lr, //learning rate
    int             t   //timestep
);

}   //namespace common
}   //namescpce GNNPro_lib 

#endif