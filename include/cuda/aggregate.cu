#include "aggregate.cuh"

namespace GNNPro_lib{
namespace common{

#define WARP_SIZE 32

//=================================Forward Propagation===============================================

//LEO version in multi-GPU scenarios
//1. Distinguish between local, remote, and cached neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid  = local_node_num;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_lrc_block_cuda<<<grid, block, shared_mem, stream>>>(
                        output,  input_l, input_c, 
                        ptr_l, ind_l, ptr_c, ind_c, ptr_r, ind_r,
                        id_map, local_node_num, dim, partSize,
                        nodePerPE, warpPerBlock); 

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lrc_block: %s\n", cudaGetErrorString(error));
    }
}

//LEO version in multi-GPU scenarios
//1. Process local and remote neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid  = local_node_num;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_lr_block_cuda<<<grid, block, shared_mem, stream>>>(
                        output, input_l, 
                        ptr_l, ind_l, ptr_r, ind_r,
                        id_map, local_node_num, dim, partSize,
                        nodePerPE, warpPerBlock);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lrc_block: %s\n", cudaGetErrorString(error));
    }       
}


//LEO version in multi-GPU scenarios
//1. Process only local neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = warpPerBlock * dim * sizeof(float);

    leo_l_block_cuda<<<grid, block, shared_mem, stream>>>(
                        output, input_l,
                        ptr_l, ind_l, id_map, local_node_num,
                        dim, partSize, nodePerPE, warpPerBlock);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_l_block: %s\n", cudaGetErrorString(error));
    }   
}

//LEO version in multi-GPU scenarios
//1. Distinguish between local, remote, and cached neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Thread-level
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
){
    const int block = WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = 2 * dim * sizeof(float);

    leo_lcr_thread_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l, input_c,
                            ptr_l, ind_l, ptr_c, ind_c, ptr_r, ind_r,
                            id_map, local_node_num, dim, nodePerPE);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lcr_thread_cuda: %s\n", cudaGetErrorString(error));
    }    
}

//LEO version in multi-GPU scenarios
//1. Process only local neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Thread-level
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
){
    const int block = WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = dim * sizeof(float);

    leo_l_thread_cuda<<<grid, block, shared_mem, stream>>>(
                        output, input_l,
                        ptr_l, ind_l,
                        local_node_num, dim , nodePerPE);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_l_thread_cuda: %s\n", cudaGetErrorString(error));
    }     
}

//LEO version in multi-GPU scenarios
//1. Processing four types of neighbors, including local, cached (local GPU side), host (host side), and remote (remote GPU side).
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Thread-leve
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
){
    const int block = WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = 2 * dim * sizeof(float);

    leo_lchr_thread_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l, input_c, input_h,
                            ptr_l, ind_l, ptr_c, ind_c, ptr_r, ind_r, ptr_h, ind_h,
                            id_map, local_node_num, dim, nodePerPE);
    
    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lchr_thread_cuda: %s\n", cudaGetErrorString(error));
    }     
}

//Update cached node feature 
void leo_update_cache_block(
    cache_feat_t*       output,
    const cache_feat_t* input,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode,
    const cudaStream_t  stream    
){
    const int block = WARP_SIZE;
    const int grid = cache_num;

    leo_update_cache_block_cuda<<<grid, block, 0, stream>>>(output, input, cache_list, cache_num, dim, nodePerPE, mynode);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_update_cache_block: %s\n", cudaGetErrorString(error));
    }  
}

//Update cached node feature
void leo_update_cache_block_wo_sync(
    cache_feat_t*       output,
    const cache_feat_t* input,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode,
    const cudaStream_t  stream
){
    const int block = WARP_SIZE;
    const int grid = cache_num;

    leo_update_cache_block_cuda<<<grid, block, 0, stream>>>(output, input, cache_list, cache_num, dim, nodePerPE, mynode);
}

//LEO version in multi-GPU scenarios
//1. Processing four types of neighbors, including local, cached (local GPU side), host (host side), and remote (remote GPU side).
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_lrhc_block_v1_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l, input_c, input_h,
                            ptr_l, ind_l, ptr_c, ind_c,
                            ptr_r, ind_r, ptr_h, ind_h,
                            id_map, local_node_num, dim, partSize,
                            nodePerPE, warpPerBlock, overlap_dist);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lrhc_block_v1_cuda: %s\n", cudaGetErrorString(error));
    }
}

//LEO version in multi-GPU scenarios
//1. Processing four types of neighbors, including local, cached (local GPU side), host (host side), and remote (remote GPU side).
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
//4. Version2
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
){
    const int block = WARP_SIZE; //By default, on warp is on each block
    const int grid = local_node_num;
    const int shared_mem1 = dim * sizeof(float);
    const int shared_mem2 = 2 * dim * sizeof(float);

    leo_lc_block_v2_cuda<<<grid, block, shared_mem1, stream>>>(
                            output, input_l, input_c,
                            ptr_l, ind_l, ptr_c, ind_c,
                            id_map, local_node_num, dim, nodePerPE);    

    leo_hr_block_v2_cuda<<<grid, block, shared_mem2, stream>>>(
                            output, input_l, input_h,
                            ptr_r, ind_r, ptr_h, ind_h,
                            id_map, local_node_num, dim, nodePerPE);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lrhc_block_v2_cuda: %s\n", cudaGetErrorString(error));
    }    
}

//LEO version in multi-GPU scenarios
//1. Distinguish between local, remote, and cached neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Warp-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_lrc_warp_cuda<<<grid, block, shared_mem, stream>>>(
                        output, input_l, input_c,
                        ptr_l, ind_l, ptr_c, ind_c, ptr_r, ind_r,
                        id_map, local_node_num, dim, 
                        nodePerPE, warpPerBlock);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lrc_warp: %s\n", cudaGetErrorString(error));
    }
}

//LEO version in multi-GPU scenarios
//1. Process local and remote neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Warp-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_lr_warp_cuda<<<grid, block, shared_mem, stream>>>(
                        output, input_l, 
                        ptr_l, ind_l, ptr_r, ind_r,
                        id_map, local_node_num, dim,
                        nodePerPE, warpPerBlock);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lr_warp: %s\n", cudaGetErrorString(error));
    } 
}

//LEO version in multi-GPU scenarios
//1. Process only local neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Warp-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = warpPerBlock * dim * sizeof(float);

    leo_l_warp_cuda<<<grid, block, shared_mem, stream>>>(
                        output, input_l,
                        ptr_l, ind_l,
                        id_map, local_node_num, dim,
                        nodePerPE, warpPerBlock);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_l_warp: %s\n", cudaGetErrorString(error));
    }    
}

//LEO version in multi-GPU scenarios
//1. Processing four types of neighbors, including local, cached (local GPU side), host (host side), and remote (remote GPU side).
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Warp-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_lrhc_warp_cuda<<<grid, block, shared_mem, stream>>>(
                        output, input_l, input_c, input_h,
                        ptr_l, ind_l, ptr_c, ind_c, 
                        ptr_r, ind_r, ptr_h, ind_h,
                        id_map, local_node_num, dim,
                        nodePerPE, warpPerBlock);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lrhc_warp_cuda: %s\n", cudaGetErrorString(error));
    }
}

//LEO test version in multi-GPU scenarios
//1. Block-level
//2. version1
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
){
    const int block = WARP_SIZE; //By default, on warp is on each block
    const int grid = local_node_num;
    const int shared_mem = 2 * dim * sizeof(float);

    leo_lcr_block_v1_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l, input_c,
                            ptr_l, ind_l, ptr_c, ind_c, ptr_r, ind_r,
                            id_map, local_node_num, dim, nodePerPE);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lcr_block_v1_cuda: %s\n", cudaGetErrorString(error));
    }
}

//LEO test version in multi-GPU scenarios
//1. Block-level
//2. version2
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
){
    const int block = WARP_SIZE; //By default, on warp is on each block
    const int grid = local_node_num;
    const int shared_mem1 = dim * sizeof(float);
    const int shared_mem2 = 2 * dim * sizeof(float);

    leo_lc_block_v2_cuda<<<grid, block, shared_mem1, stream>>>(
                            output, input_l, input_c,
                            ptr_l, ind_l, ptr_c, ind_c,
                            id_map, local_node_num, dim, nodePerPE);

    leo_r_block_v2_cuda<<<grid, block, shared_mem2, stream>>>(
                            output, input_l,
                            ptr_r, ind_r,
                            id_map, local_node_num, dim, nodePerPE);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lc_block_v2_cuda Or @leo_r_block_v2_cuda: %s\n", cudaGetErrorString(error));
    }
}

//LEO test version in multi-GPU scenarios
// 1. Block-level
// 2. version3
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
){
    const int block = WARP_SIZE; //By default, on warp is on each block
    const int grid = local_node_num;
    const int shared_mem = 3 * dim * sizeof(float);

    leo_lcr_block_v3_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l, input_c,
                            ptr_l, ind_l, ptr_c, ind_c, ptr_r, ind_r,
                            id_map, local_node_num, dim, nodePerPE);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lcr_block_v3_cuda: %s\n", cudaGetErrorString(error));
    }
}

//LEO test version in multi-GPU scenarios
// 1. Block-level
// 2. version4
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
){
    const int block = WARP_SIZE; //By default, on warp is on each block
    const int grid = local_node_num;
    const int shared_mem = 2 * dim * sizeof(float);

    leo_lcr_block_v4_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l, input_c,
                            ptr_l, ind_l, ptr_r, ind_r,
                            id_map, local_node_num, dim,
                            nodePerPE, myid);
    
    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lcr_block_v4_cuda: %s\n", cudaGetErrorString(error));
    }
}

//LEO version in multi-GPU scenarios
//1. Distinguish between local, remote, and cached neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Warp-level
//4. Version1
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem1 = warpPerBlock * dim * sizeof(float);
    const int shared_mem2 = 2 * warpPerBlock * dim * sizeof(float);

    leo_lc_warp_v1_cuda<<<grid, block, shared_mem1, stream>>>(
                            output, input_l, input_c,
                            ptr_l, ind_l, ptr_c, ind_c,
                            id_map, local_node_num, dim,
                            nodePerPE, warpPerBlock);

    leo_r_warp_v1_cuda<<<grid, block, shared_mem2, stream>>>(
                            output, input_l,
                            ptr_r, ind_r, local_node_num,
                            dim, nodePerPE, warpPerBlock);
    
    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lcr_warp_v1_cuda: %s\n", cudaGetErrorString(error));
    }
}

//LEO version in multi-GPU scenarios
//1. Processing four types of neighbors, including local, cached (local GPU side), host (host side), and remote (remote GPU side).
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Warp-level
//4. Version1
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_lc_warp_v1_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l, input_c,
                            ptr_l, ind_l, ptr_c, ind_c,
                            id_map, local_node_num, dim,
                            nodePerPE, warpPerBlock); 

    leo_hr_warp_v1_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l, input_h,
                            ptr_r, ind_r, ptr_h, ind_h,
                            id_map, local_node_num, dim,
                            nodePerPE, warpPerBlock); 

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_lchr_warp_v1_cuda: %s\n", cudaGetErrorString(error));
    }      
}

//LEO version GIN in multi-GPU scenarios
//1. Distinguish between local, remote, and cached neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_gin_lrc_block_cuda<<<grid, block, shared_mem, stream>>>(
                            output,  input_l, input_c, 
                            ptr_l, ind_l, ptr_c, ind_c, ptr_r, ind_r,
                            id_map, local_node_num, dim, partSize,
                            nodePerPE, warpPerBlock, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_lrc_block: %s\n", cudaGetErrorString(error));
    }    
}

//LEO version GIN in multi-GPU scenarios
//1. Process local and remote neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid  = local_node_num;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_gin_lr_block_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l, 
                            ptr_l, ind_l, ptr_r, ind_r,
                            id_map, local_node_num, dim, partSize,
                            nodePerPE, warpPerBlock, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_lr_block_cuda: %s\n", cudaGetErrorString(error));
    }          
}

//LEO version GIN in multi-GPU scenarios
//1. Process only local neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = warpPerBlock * dim * sizeof(float);

    leo_gin_l_block_cuda<<<grid, block, shared_mem, stream>>>(
                        output, input_l,
                        ptr_l, ind_l, id_map, local_node_num,
                        dim, partSize, nodePerPE, warpPerBlock, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_l_block_cuda: %s\n", cudaGetErrorString(error));
    }        
}

//LEO version GIN in multi-GPU scenarios
//1. Processing four types of neighbors, including local, cached (local GPU side), host (host side), and remote (remote GPU side).
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_gin_lrhc_block_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l, input_c, input_h,
                            ptr_l, ind_l, ptr_c, ind_c,
                            ptr_r, ind_r, ptr_h, ind_h,
                            id_map, local_node_num, dim, partSize,
                            nodePerPE, warpPerBlock, overlap_dist, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_lrhc_block_cuda: %s\n", cudaGetErrorString(error));
    }        
}

//LEO version GIN in multi-GPU scenarios
//1. Distinguish between local, remote, and cached neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
//4. version1
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
){
    const int block =  WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem1 = dim * sizeof(float);
    const int shared_mem2 = 2 * dim * sizeof(float);

    leo_gin_lc_block_v1_cuda<<<grid, block, shared_mem1, stream>>>(
                                output, input_l, input_c,
                                ptr_l, ind_l, ptr_c, ind_c,
                                id_map, local_node_num, dim, 
                                nodePerPE, eps);

    leo_gin_r_block_v1_cuda<<<grid, block, shared_mem2, stream>>>(
                                output, input_l,
                                ptr_r, ind_r,
                                id_map, local_node_num, dim, nodePerPE);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_lc_block_v1_cuda Or @leo_gin_r_block_v1_cuda: %s\n", cudaGetErrorString(error));
    }    
}

//LEO version GIN in multi-GPU scenarios
//1. Process local and remote neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
//4. version1
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
){
    const int block = WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = 2 * dim * sizeof(float);

    leo_gin_lr_block_v1_cuda<<<grid, block, shared_mem, stream>>>(
                                output, input_l,
                                ptr_l, ind_l, ptr_r, ind_r,
                                id_map, local_node_num, dim,
                                nodePerPE, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_lr_block_v1_cuda: %s\n", cudaGetErrorString(error));
    } 
}

//LEO version GIN in multi-GPU scenarios
//1. Process only local neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
//4. version1
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
){
    const int block = WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = dim * sizeof(float);

    leo_gin_l_block_v1_cuda<<<grid, block, shared_mem, stream>>>(
                                output, input_l,
                                ptr_l, ind_l,
                                id_map, local_node_num, dim,
                                nodePerPE, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_l_block_v1: %s\n", cudaGetErrorString(error));
    }     
}

//LEO version GIN in multi-GPU scenarios
//1. Processing four types of neighbors, including local, cached (local GPU side), host (host side), and remote (remote GPU side).
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Block-level
//4. version1
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
){
    const int block = WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem1 = dim * sizeof(float);
    const int shared_mem2 = 2 * dim * sizeof(float);

    leo_gin_lc_block_v1_cuda<<<grid, block, shared_mem1, stream>>>(
                                output, input_l, input_c,
                                ptr_l, ind_l, ptr_c, ind_c,
                                id_map, local_node_num, dim, 
                                nodePerPE, eps);

    leo_gin_hr_block_v1_cuda<<<grid, block, shared_mem2, stream>>>(
                                output, input_l, input_h,
                                ptr_r, ind_r, ptr_h, ind_h,
                                id_map, local_node_num, dim,
                                nodePerPE, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_lc_block_v1_cuda Or @leo_gin_hr_block_v1_cuda: %s\n", cudaGetErrorString(error));
    }  
}

//LEO version GIN in multi-GPU scenarios
//1. Distinguish between local, remote, and cached neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Warp_level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem1 = warpPerBlock * dim * sizeof(float);
    const int shared_mem2 = 2 * warpPerBlock * dim * sizeof(float);

    leo_gin_lc_warp_cuda<<<grid, block, shared_mem1, stream>>>(
                            output, input_l, input_c,
                            ptr_l, ind_l, ptr_c, ind_c,
                            id_map, local_node_num, dim,
                            nodePerPE, warpPerBlock, eps);

    leo_gin_r_warp_cuda<<<grid, block, shared_mem2, stream>>>(
                            output, input_l, 
                            ptr_r, ind_r,
                            id_map, local_node_num, dim,
                            nodePerPE, warpPerBlock, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_lc_warp_cuda Or @leo_gin_r_warp_cuda: %s\n", cudaGetErrorString(error));
    }      
}

//LEO version GIN in multi-GPU scenarios
//1. Process local and remote neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Warp-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_gin_lr_warp_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l, 
                            ptr_l, ind_l, ptr_r, ind_r,
                            id_map, local_node_num, dim,
                            nodePerPE, warpPerBlock, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_lr_warp_cuda: %s\n", cudaGetErrorString(error));
    } 
}

//LEO version GIN in multi-GPU scenarios
//1. Process only local neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Warp-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = warpPerBlock * dim * sizeof(float);

    leo_gin_l_warp_cuda<<<grid, block, shared_mem, stream>>>(
                            output, input_l,
                            ptr_l, ind_l, 
                            id_map, local_node_num, dim,
                            nodePerPE, warpPerBlock, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_l_warp_cuda: %s\n", cudaGetErrorString(error));
    } 
}

//LEO version GIN in multi-GPU scenarios
//1. Processing four types of neighbors, including local, cached (local GPU side), host (host side), and remote (remote GPU side).
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. Warp-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem1 = warpPerBlock * dim * sizeof(float);
    const int shared_mem2 = 2 * warpPerBlock * dim * sizeof(float);

    leo_gin_lc_warp_cuda<<<grid, block, shared_mem1, stream>>>(
                            output, input_l, input_c,
                            ptr_l, ind_l, ptr_c, ind_c,
                            id_map, local_node_num, dim,
                            nodePerPE, warpPerBlock, eps);

    leo_gin_hr_warp_cuda<<<grid, block, shared_mem2, stream>>>(
                            output, input_l, input_h,
                            ptr_r, ind_r, ptr_h, ind_h,
                            id_map, local_node_num, dim,
                            nodePerPE, warpPerBlock, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_lc_warp_cuda Or @leo_gin_hr_warp_cuda: %s\n", cudaGetErrorString(error));
    }     
}

//UVM version GCN in multi-GPU scenarios
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
){
    const int partSize = 4;
    const int warpPerBlock = 4;

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;

    UVM_gcn_forward_cuda<<<grid, block, 0>>>(d_out, d_in, d_row_ptr, d_col_ind,
                                            num_nodes, dim, nodePerPE, 
                                            partSize, warpPerBlock, myGPUid);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @UVM_gcn_forward_cuda: %s\n", cudaGetErrorString(error));
    }      
}

//UVM version GIN in multi-GPU scenarios
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
){
    const int partSize = 4;
    const int warpPerBlock = 4;

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;

    UVM_gin_forward_cuda<<<grid, block, 0>>>(d_out, d_in, d_row_ptr, d_col_ind,
                                            num_nodes, dim, nodePerPE, 
                                            partSize, warpPerBlock, myGPUid, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @UVM_gin_forward_cuda: %s\n", cudaGetErrorString(error));
    }         
}

//=================================Back Propagation================================================

//Updating gradient of local nodes during backpropagation(LEO version)
//1. Process local and remote neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. CSR format for transposed adjacency matrices
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_local_grad_backward_block_cuda<<<grid, block, shared_mem, stream>>>(
                                        gradients, input_grad,
                                        T_ptr_l, T_ind_l, T_ptr_r, T_ind_r,
                                        local_node_num, dim, partSize,
                                        nodePerPE, warpPerBlock);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_local_backward_block: %s\n", cudaGetErrorString(error));
    }
}

//Updating gradient of local nodes during backpropagation(LEO version)
//1. Process local and remote neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. CSR format for transposed adjacency matrices
//4. Put the function in the stream without synchronization
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_local_grad_backward_block_cuda<<<grid, block, shared_mem, stream>>>(
                                        gradients, input_grad,
                                        T_ptr_l, T_ind_l, T_ptr_r, T_ind_r,
                                        local_node_num, dim, partSize,
                                        nodePerPE, warpPerBlock);
}


//Updating gradient of local nodes during backpropagation(LEO version)
//1. Process only local neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. CSR format for transposed adjacency matrices
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = warpPerBlock * dim * sizeof(float);

    leo_local_only_grad_backward_block_cuda<<<grid, block, shared_mem, stream>>>(
                                            gradients, input_grad,
                                            T_ptr_l, T_ind_l,
                                            local_node_num, dim, partSize,
                                            nodePerPE, warpPerBlock);
                                
    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_local_only_backward_block: %s\n", cudaGetErrorString(error));
    }
}

//Updating gradient of local nodes during backpropagation(LEO version)
//1. Process only local neighbors
//2. Fixed the upper and lower bounds of the subgraph in each partition
//3. CSR format for transposed adjacency matrices
//4. Put the function in the stream without synchronization
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;
    const int shared_mem = warpPerBlock * dim * sizeof(float);

    leo_local_only_grad_backward_block_cuda<<<grid, block, shared_mem, stream>>>(
                                            gradients, input_grad,
                                            T_ptr_l, T_ind_l,
                                            local_node_num, dim, partSize,
                                            nodePerPE, warpPerBlock);
}

//Updating gradient of local nodes during backpropagation(LEO version)
//1. CSR format for transposed adjacency matrices
//2. Warp-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_local_grad_backward_warp_V1_cuda<<<grid, block, shared_mem, stream>>>(
                                            gradients, input_grad,
                                            T_ptr_l, T_ind_l, T_ptr_r, T_ind_r,
                                            local_node_num, dim, 
                                            nodePerPE, warpPerBlock);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_local_backward_warp: %s\n", cudaGetErrorString(error));
    }    
}

//Updating gradient of local nodes during backpropagation(LEO version)
//1. CSR format for transposed adjacency matrices
//2. Warp-level
//3. Put the function in the stream without synchronization
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_local_grad_backward_warp_V1_cuda<<<grid, block, shared_mem, stream>>>(
                                            gradients, input_grad,
                                            T_ptr_l, T_ind_l, T_ptr_r, T_ind_r,
                                            local_node_num, dim, 
                                            nodePerPE, warpPerBlock);
}

//Updating gradient of local nodes during backpropagation(LEO version)
//1. CSR format for transposed adjacency matrices
//2. Warp-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = warpPerBlock * dim * sizeof(float);

    leo_local_only_grad_backward_warp_V1_cuda<<<grid, block, shared_mem, stream>>>(
                                                gradients, input_grad,
                                                T_ptr_l, T_ind_l,
                                                local_node_num, dim,
                                                nodePerPE, warpPerBlock);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_local_only_backward_warp: %s\n", cudaGetErrorString(error));
    }     
}

//Updating gradient of local nodes during backpropagation(LEO version)
//1. CSR format for transposed adjacency matrices
//2. Warp-level
//3. Put the function in the stream without synchronization
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = warpPerBlock * dim * sizeof(float);

    leo_local_only_grad_backward_warp_V1_cuda<<<grid, block, shared_mem, stream>>>(
                                                gradients, input_grad,
                                                T_ptr_l, T_ind_l,
                                                local_node_num, dim,
                                                nodePerPE, warpPerBlock);    
}

//======================================  GIN  ===================================================
//Updating gradient of local nodes during backpropagation(GIN model)
//1. CSR format for transposed adjacency matrices
//2. Warp-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_gin_local_grad_backward_warp_cuda<<<grid, block, shared_mem, stream>>>(
                                            gradients, input_grad,
                                            T_ptr_l, T_ind_l, T_ptr_r, T_ind_r,
                                            local_node_num, dim, 
                                            nodePerPE, warpPerBlock, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_local_backward_warp: %s\n", cudaGetErrorString(error));
    }          
}

//Updating gradient of local nodes during backpropagation(GIN model)
//1. CSR format for transposed adjacency matrices
//2. Warp-level
//3. Stream version
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = 2 * warpPerBlock * dim * sizeof(float);

    leo_gin_local_grad_backward_warp_cuda<<<grid, block, shared_mem, stream>>>(
                                            gradients, input_grad,
                                            T_ptr_l, T_ind_l, T_ptr_r, T_ind_r,
                                            local_node_num, dim, 
                                            nodePerPE, warpPerBlock, eps);    
}

//Updating gradient of only local nodes during backpropagation(GIN model)
//1. CSR format for transposed adjacency matrices
//2. Warp-level
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = warpPerBlock * dim * sizeof(float); 

    leo_gin_local_only_grad_backward_warp_cuda<<<grid, block, shared_mem, stream>>>(
                                                gradients, input_grad,
                                                T_ptr_l, T_ind_l,
                                                local_node_num, dim,
                                                nodePerPE, warpPerBlock, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_gin_local_only_grad_backward_warp_cuda: %s\n", cudaGetErrorString(error));
    }        
}

//Updating gradient of only local nodes during backpropagation(GIN model)
//1. CSR format for transposed adjacency matrices
//2. Warp-level
//3. Stream version
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
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid = (local_node_num * WARP_SIZE + block - 1) / block;
    const int shared_mem = warpPerBlock * dim * sizeof(float); 

    leo_gin_local_only_grad_backward_warp_cuda<<<grid, block, shared_mem, stream>>>(
                                                gradients, input_grad,
                                                T_ptr_l, T_ind_l,
                                                local_node_num, dim,
                                                nodePerPE, warpPerBlock, eps);    
}

//UVM versin GCN in multi-GPU scenarios
//Backward
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
){
    const int partSize = 4;
    const int warpPerBlock = 4;

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;

    UVM_gcn_backward_cuda<<<grid, block, 0>>>(gradients, input_grad, d_row_ptr, d_col_ind,
                                            num_nodes, dim, nodePerPE, 
                                            partSize, warpPerBlock, myGPUid);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @UVM_gcn_backward_cuda: %s\n", cudaGetErrorString(error));
    }      
}

//UVM versin GIN in multi-GPU scenarios
//Backward
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
){
    const int partSize = 4;
    const int warpPerBlock = 4;

    const int block = warpPerBlock * WARP_SIZE;
    const int grid = local_node_num;

    UVM_gin_backward_cuda<<<grid, block, 0>>>(gradients, input_grad, d_row_ptr, d_col_ind,
                                            num_nodes, dim, nodePerPE, 
                                            partSize, warpPerBlock, myGPUid, eps);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @UVM_gin_backward_cuda: %s\n", cudaGetErrorString(error));
    }           
}


//Update cached node feature in back propagation
void leo_update_cache_backward_block(
    value_t*            gradients,
    const value_t*      input_grad,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode,
    const cudaStream_t  stream
){
    const int block = WARP_SIZE;
    const int grid = cache_num;

    leo_update_cahce_backward_block_cuda<<<grid, block, 0, stream>>>(gradients, input_grad, cache_list, cache_num, dim, nodePerPE, mynode);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @leo_update_cahce_backward_block_cuda: %s\n", cudaGetErrorString(error));
    } 
}

//Update cached node feature in back propagation
//Put the function in the stream without synchronization
void leo_update_cache_backward_block_wo_sync(
    value_t*            gradients,
    const value_t*      input_grad,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode,
    const cudaStream_t  stream
){
    const int block = WARP_SIZE;
    const int grid = cache_num;

    leo_update_cahce_backward_block_cuda<<<grid, block, 0, stream>>>(gradients, input_grad, cache_list, cache_num, dim, nodePerPE, mynode);
}

//LEO version in multi-GPU scenarios
//1. Compute gradients of SoftmaxCrossEntroy loss
//2. Reuse the results of previous softmaxforward
void SoftmaxCrossEntroyBackward(
    value_t*            gradients,
    const cache_feat_t* softmax_output, 
    const int*          labels, 
    const int           local_node_num,
    const int           dim,
    const int           warpPerBlock,
    const cudaStream_t  stream
){
    const int block = warpPerBlock * WARP_SIZE;
    const int grid  = (local_node_num + block - 1) / block;

    SoftmaxCrossEntroy_backward_cuda<<<grid, block, 0, stream>>>(gradients, softmax_output, labels, local_node_num, dim);

    CUDA_CALL(cudaDeviceSynchronize());
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess){
        printf("CUDA Kernel Error @SoftmaxCrossEntroy_backward_cuda: %s\n", cudaGetErrorString(error));
    }  
}

}   //namespace common
}   //namescpce GNNPro_lib 