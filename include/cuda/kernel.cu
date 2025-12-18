#include "kernel.cuh"

namespace GNNPro_lib{
namespace common{

#define WARP_SIZE 32

__device__ inline
void atomicAddF(float* address, float value){
    float old = value;
    while ((old = atomicExch(address, atomicExch(address, 0.0f)+old))!=0.0f);
}

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
){
    const int bid = blockIdx.x;         //global warp-id
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if (bid < node_num){

        for (int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for (CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_l[eid];
                //if(threadIdx.x == 0) printf("Value is %u\n", nid);
                //VertexID local_nid = id_map[nid];
                VertexID local_nid = nid % nodePerPE;
                for (int d = lanid; d < dim; d += WARP_SIZE){
                    
                    tmp[wid * dim + d] += input_l[local_nid * dim + d];
                }
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for (CSR_t block_beg = remote_beg; block_beg < remote_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, remote_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_r[eid];
                VertexID r_partID = nid / nodePerPE;
                VertexID r_offset = nid % nodePerPE; 
                //VertexID a = input_l[r_offset * dim]; 
                //if(threadIdx.x == 0) printf("Value is %u\n", eid);
                nvshmemx_float_get_warp((float*)&tmp_r[wid * dim], &input_l[r_offset * dim], dim, r_partID);
                for (int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += tmp_r[wid * dim + d];
                }
            }
        }

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[bid];
        CSR_t cache_end = ptr_c[bid + 1];

        for (CSR_t block_beg = cache_beg; block_beg < cache_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, cache_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_c[eid];
                VertexID cache_nid = id_map[nid];

                for (int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += input_c[cache_nid * dim + d];
                }
            }
        }

        for (int d = lanid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[bid * dim +d], tmp[wid * dim + d]);
        }
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if (bid < node_num){
        for (int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for (CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_l[eid];
                //if(threadIdx.x == 0) printf("Value is %u\n", nid);
                //VertexID local_nid = id_map[nid];
                VertexID local_nid = nid % nodePerPE;
                for (int d = lanid; d < dim; d += WARP_SIZE){
                    
                    tmp[wid * dim + d] += input_l[local_nid * dim + d];
                }
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for (CSR_t block_beg = remote_beg; block_beg < remote_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, remote_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_r[eid];
                VertexID r_partID = nid / nodePerPE;
                VertexID r_offset = nid % nodePerPE; 
                //VertexID a = input_l[r_offset * dim]; 
                //if(threadIdx.x == 0) printf("Value is %u\n", eid);
                nvshmemx_float_get_warp((float*)&tmp_r[wid * dim], &input_l[r_offset * dim], dim, r_partID);
                for (int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += tmp_r[wid * dim + d];
                }
            }
        }   

        for (int d = lanid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[bid * dim +d], tmp[wid * dim + d]);
        }                     
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id

    extern __shared__  float tmp[];  //Store local intermediate results

    if (bid < node_num){

        for (int idx = threadIdx.x; idx < warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for (CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_l[eid];
                //VertexID local_nid = id_map[nid];
                VertexID local_nid = nid % nodePerPE;
                for (int d = lanid; d < dim; d += WARP_SIZE){
                    
                    tmp[wid * dim + d] += input_l[local_nid * dim + d];
                }
            }
        }

        for (int d = lanid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[bid * dim +d], tmp[wid * dim + d]);
        }
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(bid < node_num){

        for (int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(CSR_t block_beg = local_beg; block_beg < local_end; block_beg += overlap_dist * warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * overlap_dist * partSize;
            CSR_t warp_end = min(warp_beg + overlap_dist * partSize, local_end);

            for(EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_l[eid];
                VertexID local_nid = nid % nodePerPE;

                for(int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += input_l[local_nid * dim + d];
                }
            }
        } 

        //Procssing cached neighbor
        CSR_t cache_beg = ptr_c[bid];
        CSR_t cache_end = ptr_c[bid + 1];

        for(CSR_t block_beg = cache_beg; block_beg < cache_end; block_beg += overlap_dist * warpPerBlock * partSize){
            
            CSR_t warp_beg = block_beg + wid * overlap_dist * partSize;
            CSR_t warp_end = min(warp_beg + overlap_dist * partSize, cache_end);

            for(EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_c[eid];
                VertexID cache_nid = id_map[nid];

                for(int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += input_c[cache_nid * dim + d];
                }
            }
        }  

        //Processing cached neighbor (host side)
        CSR_t host_beg = ptr_h[bid];
        CSR_t host_end = ptr_h[bid + 1];

        for(CSR_t block_beg = host_beg; block_beg < host_end; block_beg += overlap_dist * warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * overlap_dist * partSize;
            CSR_t warp_end = min(warp_beg + overlap_dist * partSize, host_end);

            for(EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_h[eid];
                VertexID host_nid = id_map[nid];

                for(int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += input_h[host_nid * dim + d];
                }
            }
        } 

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(CSR_t block_beg = remote_beg; block_beg < remote_end; block_beg += overlap_dist * warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * overlap_dist * partSize;
            CSR_t warp_end = min(warp_beg + overlap_dist * partSize, remote_end);

            for(EdgeID eid = warp_beg; eid < warp_end; eid++){
                
                VertexID nid = ind_r[eid];
                VertexID r_partID = nid / nodePerPE;
                VertexID r_offset = nid % nodePerPE;
                nvshmemx_float_get_warp((float*)&tmp_r[wid * dim], &input_l[r_offset * dim], dim, r_partID);

                for(int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += tmp_r[wid * dim + d];
                }
            }
        }  

        for(int d = lanid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[bid * dim + d], tmp[wid * dim + d]);
        } 
    }
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(CSR_t eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_l[local_nid * dim + d];
            }
        }

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[warpid];
        CSR_t cache_end = ptr_c[warpid + 1];

        for(CSR_t eid = cache_beg; eid < cache_end; eid++){

            VertexID nid = ind_c[eid];
            VertexID cache_nid = id_map[nid];

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_c[cache_nid * dim + d];
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[warpid];
        CSR_t remote_end = ptr_r[warpid + 1];

        for(CSR_t eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_warp((float*)&tmp_r[block_warpid * dim], &input_l[r_offset * dim], dim, r_partID);
            //nvshmemx_float_get_nbi_warp((float*)&tmp_r[block_warpid * dim], &input_l[r_offset * dim], dim, r_partID);

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += tmp_r[block_warpid * dim + d];
            }
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        }
    }
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(CSR_t eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_l[local_nid * dim + d];
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[warpid];
        CSR_t remote_end = ptr_r[warpid + 1];

        for(CSR_t eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_warp((float*)&tmp_r[block_warpid * dim], &input_l[r_offset * dim], dim, r_partID);

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += tmp_r[block_warpid * dim + d];
            }
        } 

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        }       
    }
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(CSR_t eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_l[local_nid * dim + d];
            }
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        }  
    }
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads(); 

        //Processing local neighbor
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(CSR_t eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_l[local_nid * dim + d];
            }
        }

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[warpid];
        CSR_t cache_end = ptr_c[warpid + 1];

        for(CSR_t eid = cache_beg; eid < cache_end; eid++){

            VertexID nid = ind_c[eid];
            VertexID cache_nid = id_map[nid];

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_c[cache_nid * dim + d];
            }
        }

        //Processing cached neighbor (host side)
        CSR_t host_beg = ptr_h[warpid];
        CSR_t host_end = ptr_h[warpid + 1];

        for(CSR_t eid = host_beg; eid < host_end; eid++){

            VertexID nid = ind_h[eid];
            VertexID host_nid = id_map[nid];

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_h[host_nid * dim + d];
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[warpid];
        CSR_t remote_end = ptr_r[warpid + 1];

        for(CSR_t eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_warp((float*)&tmp_r[block_warpid * dim], &input_l[r_offset * dim], dim, r_partID);

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += tmp_r[block_warpid * dim + d];
            }
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        }
    }
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(CSR_t eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_l[local_nid * dim + d];
            }
        }

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[warpid];
        CSR_t cache_end = ptr_c[warpid + 1];

        for(CSR_t eid = cache_beg; eid < cache_end; eid++){

            VertexID nid = ind_c[eid];
            VertexID cache_nid = id_map[nid];

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_c[cache_nid * dim + d];
            }
        }
    
        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        }    
    }   
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[warpid];
        CSR_t remote_end = ptr_r[warpid + 1];

        for(CSR_t eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_warp((float*)&tmp_r[block_warpid * dim], &input_l[r_offset * dim], dim, r_partID);

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += tmp_r[block_warpid * dim + d];
            }
        } 

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        }     
    }        
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing cached neighbor (host side)
        CSR_t host_beg = ptr_h[warpid];
        CSR_t host_end = ptr_h[warpid + 1];

        for(CSR_t eid = host_beg; eid < host_end; eid++){

            VertexID nid = ind_h[eid];
            VertexID host_nid = id_map[nid];

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_h[host_nid * dim + d];
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[warpid];
        CSR_t remote_end = ptr_r[warpid + 1];

        for(CSR_t eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_warp((float*)&tmp_r[block_warpid * dim], &input_l[r_offset * dim], dim, r_partID);

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += tmp_r[block_warpid * dim + d];
            }
        }  

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        }        
    }    
}

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
){  
    const int bid = blockIdx.x;         //global warp-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[dim]; //Cache remote intermediate result

    if (bid < node_num){

        for (int idx = threadIdx.x; idx <  2 * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();  

       //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_l[local_nid * dim + d];
            }
        }

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[bid];
        CSR_t cache_end = ptr_c[bid + 1];

        for(EdgeID eid = cache_beg; eid < cache_end; eid++){

            VertexID nid = ind_c[eid];
            VertexID cache_nid = id_map[nid];

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_c[cache_nid * dim + d];
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_nbi_block((float*)&tmp_r[0], &input_l[r_offset * dim], dim, r_partID);
            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += tmp_r[d];
            }
        }

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id

    extern __shared__  float tmp[];  //Store all intermediate results

    if(bid < node_num){

        for (int idx = threadIdx.x; idx <  dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();  

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_l[local_nid * dim + d];
            }
        } 

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[bid];
        CSR_t cache_end = ptr_c[bid + 1];

        for(EdgeID eid = cache_beg; eid < cache_end; eid++){

            VertexID nid = ind_c[eid];
            VertexID cache_nid = id_map[nid];

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_c[cache_nid * dim + d];
            }
        }

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }
    }   
}

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
){
    const int bid = blockIdx.x;         //global warp-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[dim]; //Cache remote intermediate result

    if(bid < node_num){

        for (int idx = threadIdx.x; idx <  2 * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads(); 

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_block((float*)&tmp_r[0], &input_l[r_offset * dim], dim, r_partID);

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += tmp_r[d];
            }
        }

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }  
    }   
}

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
){
    const int bid = blockIdx.x;         //global warp-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[dim]; //Cache remote intermediate result

    if(bid < node_num){

        for (int idx = threadIdx.x; idx <  2 * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing cached neighbor (host side)
        CSR_t host_beg = ptr_h[bid];
        CSR_t host_end = ptr_h[bid + 1];

        for(EdgeID eid = host_beg; eid < host_end; eid++){

            VertexID nid = ind_h[eid];
            VertexID host_nid = id_map[nid];

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_h[host_nid * dim + d];
            }
        } 

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_block((float*)&tmp_r[0], &input_l[r_offset * dim], dim, r_partID);

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += tmp_r[d];
            }
        }

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }         
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id

    extern __shared__ float tmp[];          //store all intermediate results
    float* tmp_c = (float*) &tmp[dim];      //For cached intermediate results
    float* tmp_r = (float*) &tmp[2 * dim];  //For remote intermediate results

    if(bid < node_num){

        for (int idx = threadIdx.x; idx <  3 * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_l[local_nid * dim + d];
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_block((float*)&tmp_r[0], &input_l[r_offset * dim], dim, r_partID);
            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += tmp_r[d];
            }
        }

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[bid];
        CSR_t cache_end = ptr_c[bid + 1];

        for(EdgeID eid = cache_beg; eid < cache_end; eid++){

            VertexID nid = ind_c[eid];
            VertexID cache_prefix = id_map[nid] * dim;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp_c[d] += input_c[cache_prefix + d];
            }
        }

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            tmp[d] = tmp[d] + tmp_c[d];
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id

    extern __shared__ float tmp[];      //Store local and cached intermediate results
    float* tmp_r = (float*) &tmp[dim];  //For remote intermediate results

    if(bid < node_num){

        for(int idx = threadIdx.x; idx < 2 * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        int rank = myrank;
        
        //Processing local and cached neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            int nid_flag = nid / nodePerPE;

            if(nid_flag == rank){ //local neighbor
                VertexID local_nid = nid % nodePerPE;

                for(int d = threadIdx.x; d < dim; d += blockDim.x){
                    tmp[d] += input_l[local_nid * dim + d];
                }

            }else{ //cached neighbor
                VertexID cache_prefix = id_map[nid] * dim;

                for(int d = threadIdx.x; d < dim; d += blockDim.x){
                    tmp[d] += input_c[cache_prefix + d];
                }
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_block((float*)&tmp_r[0], &input_l[r_offset * dim], dim, r_partID);
            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += tmp_r[d];
            }
        }

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }          
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id
    const int lanid = threadIdx.x % 32; //laneid

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[dim]; //Cache remote intermediate result

    if(bid < node_num){

        for (int idx = threadIdx.x; idx <  2 * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();  

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_l[local_nid * dim + d];
            }
        }

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[bid];
        CSR_t cache_end = ptr_c[bid + 1];

        for(EdgeID eid = cache_beg; eid < cache_end; eid++){

            VertexID nid = ind_c[eid];
            VertexID cache_nid = id_map[nid];

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_c[cache_nid * dim + d];
            }
        }  

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            if(lanid == 0)
            nvshmem_float_get((float*)&tmp_r[0], &input_l[r_offset * dim], dim, r_partID);
            __syncthreads();
            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += tmp_r[d];
            }
        }  

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }         
    }
}

__global__
void leo_l_thread_cuda( //Processing only local neighbors
    cache_feat_t*       output,
    const cache_feat_t* input_l,
    const CSR_t*        ptr_l,
    const CSR_t*        ind_l,
    const VertexID      node_num,
    const int           dim,
    const VertexID      nodePerPE
){
    const int bid = blockIdx.x;     //global warp_id

    extern __shared__ float tmp[];

    if(bid  < node_num){

        for (int idx = threadIdx.x; idx <  2 * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_l[local_nid * dim + d];
            }
        } 

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }        
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id
    const int lanid = threadIdx.x % 32; //laneid

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[dim]; //Cache remote intermediate result

    if(bid < node_num){

        for (int idx = threadIdx.x; idx <  2 * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_l[local_nid * dim + d];
            }
        }

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[bid];
        CSR_t cache_end = ptr_c[bid + 1];

        for(EdgeID eid = cache_beg; eid < cache_end; eid++){

            VertexID nid = ind_c[eid];
            VertexID cache_nid = id_map[nid];

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_c[cache_nid * dim + d];
            }
        }  

        //Processing cached neighbor (host side)
        CSR_t host_beg = ptr_h[bid];
        CSR_t host_end = ptr_h[bid + 1];

        for(EdgeID eid = host_beg; eid < host_end; eid++){

            VertexID nid = ind_h[eid];
            VertexID host_nid = id_map[nid];

            for(int d =threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_h[host_nid * dim + d];
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            if(lanid == 0)
            nvshmem_float_get((float*)&tmp_r[0], &input_l[r_offset * dim], dim, r_partID);
            __syncthreads();
            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += tmp_r[d];
            }
        }  

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }                   
    }    
}

//=================================================  GIN  =================================================

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
){
    const int bid = blockIdx.x;         //global warp-id
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(bid < node_num){

        for (int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for (CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_l[eid];
                VertexID local_nid = nid % nodePerPE;
                for (int d = lanid; d < dim; d += WARP_SIZE){
                    
                    tmp[wid * dim + d] += input_l[local_nid * dim + d];
                }
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for (CSR_t block_beg = remote_beg; block_beg < remote_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, remote_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_r[eid];
                VertexID r_partID = nid / nodePerPE;
                VertexID r_offset = nid % nodePerPE; 
                nvshmemx_float_get_warp((float*)&tmp_r[wid * dim], &input_l[r_offset * dim], dim, r_partID);
                for (int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += tmp_r[wid * dim + d];
                }
            }
        }

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[bid];
        CSR_t cache_end = ptr_c[bid + 1];

        for (CSR_t block_beg = cache_beg; block_beg < cache_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, cache_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_c[eid];
                VertexID cache_nid = id_map[nid];

                for (int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += input_c[cache_nid * dim + d];
                }
            }
        }

        for(int d = lanid; d < dim; d += WARP_SIZE){
            output[bid * dim + d] = (1.0 + eps) * input_l[bid * dim + d];
        }

        for(int d = lanid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[bid * dim + d], tmp[wid * dim + d]);
        }
    }    
}

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
){
    const int bid = blockIdx.x;         //global warp-id
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if (bid < node_num){

        for (int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for (CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_l[eid];
                //if(threadIdx.x == 0) printf("Value is %u\n", nid);
                //VertexID local_nid = id_map[nid];
                VertexID local_nid = nid % nodePerPE;
                for (int d = lanid; d < dim; d += WARP_SIZE){
                    
                    tmp[wid * dim + d] += input_l[local_nid * dim + d];
                }
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for (CSR_t block_beg = remote_beg; block_beg < remote_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, remote_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_r[eid];
                VertexID r_partID = nid / nodePerPE;
                VertexID r_offset = nid % nodePerPE; 
                //VertexID a = input_l[r_offset * dim]; 
                //if(threadIdx.x == 0) printf("Value is %u\n", eid);
                nvshmemx_float_get_warp((float*)&tmp_r[wid * dim], &input_l[r_offset * dim], dim, r_partID);
                for (int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += tmp_r[wid * dim + d];
                }
            }
        }  

        for(int d = lanid; d < dim; d += WARP_SIZE){
            output[bid * dim + d] = (1.0 + eps) * input_l[bid * dim + d];
        }

        for(int d = lanid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[bid * dim + d], tmp[wid * dim + d]);
        }        
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id

    extern __shared__  float tmp[];  //Store local intermediate results

    if(bid < node_num){

        for (int idx = threadIdx.x; idx < warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for (CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            for (EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_l[eid];
                VertexID local_nid = nid % nodePerPE;
                for (int d = lanid; d < dim; d += WARP_SIZE){
                    
                    tmp[wid * dim + d] += input_l[local_nid * dim + d];
                }
            }
        } 

        for(int d = lanid; d < dim; d += WARP_SIZE){
            output[bid * dim + d] = (1.0 + eps) * input_l[bid * dim + d];
        }

        for(int d = lanid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[bid * dim + d], tmp[wid * dim + d]);
        }         
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(bid < node_num){

        for (int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(CSR_t block_beg = local_beg; block_beg < local_end; block_beg += overlap_dist * warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * overlap_dist * partSize;
            CSR_t warp_end = min(warp_beg + overlap_dist * partSize, local_end);

            for(EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_l[eid];
                VertexID local_nid = nid % nodePerPE;

                for(int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += input_l[local_nid * dim + d];
                }
            }
        } 

        //Procssing cached neighbor
        CSR_t cache_beg = ptr_c[bid];
        CSR_t cache_end = ptr_c[bid + 1];

        for(CSR_t block_beg = cache_beg; block_beg < cache_end; block_beg += overlap_dist * warpPerBlock * partSize){
            
            CSR_t warp_beg = block_beg + wid * overlap_dist * partSize;
            CSR_t warp_end = min(warp_beg + overlap_dist * partSize, cache_end);

            for(EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_c[eid];
                VertexID cache_nid = id_map[nid];

                for(int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += input_c[cache_nid * dim + d];
                }
            }
        }  

        //Processing cached neighbor (host side)
        CSR_t host_beg = ptr_h[bid];
        CSR_t host_end = ptr_h[bid + 1];

        for(CSR_t block_beg = host_beg; block_beg < host_end; block_beg += overlap_dist * warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * overlap_dist * partSize;
            CSR_t warp_end = min(warp_beg + overlap_dist * partSize, host_end);

            for(EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_h[eid];
                VertexID host_nid = id_map[nid];

                for(int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += input_h[host_nid * dim + d];
                }
            }
        } 

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(CSR_t block_beg = remote_beg; block_beg < remote_end; block_beg += overlap_dist * warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * overlap_dist * partSize;
            CSR_t warp_end = min(warp_beg + overlap_dist * partSize, remote_end);

            for(EdgeID eid = warp_beg; eid < warp_end; eid++){
                
                VertexID nid = ind_r[eid];
                VertexID r_partID = nid / nodePerPE;
                VertexID r_offset = nid % nodePerPE;
                nvshmemx_float_get_warp((float*)&tmp_r[wid * dim], &input_l[r_offset * dim], dim, r_partID);

                for(int d = lanid; d < dim; d += WARP_SIZE){

                    tmp[wid * dim + d] += tmp_r[wid * dim + d];
                }
            }
        } 

        for(int d = lanid; d < dim; d += WARP_SIZE){
            output[bid * dim + d] = (1.0 + eps) * input_l[bid * dim + d];
        }

        for(int d = lanid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[bid * dim + d], tmp[wid * dim + d]);
        }                                 
    }
}

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
){
    const int bid = blockIdx.x;        //global warp-id

    extern __shared__  float tmp[];  //Store all intermediate results

    if(bid < node_num){

        for (int idx = threadIdx.x; idx <  dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_l[local_nid * dim + d];
            }
        }  

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[bid];
        CSR_t cache_end = ptr_c[bid + 1];

        for(EdgeID eid = cache_beg; eid < cache_end; eid++){

            VertexID nid = ind_c[eid];
            VertexID cache_nid = id_map[nid];

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_c[cache_nid * dim + d];
            }
        } 

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            output[bid * dim + d] = (1.0 + eps) * input_l[bid * dim + d];
        }    

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }        
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[dim]; //Cache remote intermediate result

    if(bid < node_num){

        for (int idx = threadIdx.x; idx <  2 * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads(); 

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_block((float*)&tmp_r[0], &input_l[r_offset * dim], dim, r_partID);

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += tmp_r[d];
            }
        }

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }                         
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[dim]; //Cache remote intermediate result

    if(bid < node_num){

        for (int idx = threadIdx.x; idx <  2 * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads(); 

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_l[local_nid * dim + d];
            }
        } 

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_block((float*)&tmp_r[0], &input_l[r_offset * dim], dim, r_partID);

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += tmp_r[d];
            }
        }

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            output[bid * dim + d] = (1.0 + eps) * input_l[bid * dim + d];
        }    

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        }                           
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id

    extern __shared__  float tmp[];  //Store all intermediate results

    if(bid < node_num){

        for (int idx = threadIdx.x; idx <  dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_l[local_nid * dim + d];
            }
        } 

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            output[bid * dim + d] = (1.0 + eps) * input_l[bid * dim + d];
        }    

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        } 
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id

    extern __shared__  float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[dim]; //Cache remote intermediate result

    if(bid < node_num){

        for (int idx = threadIdx.x; idx <  2 * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing cached neighbor (host side)
        CSR_t host_beg = ptr_h[bid];
        CSR_t host_end = ptr_h[bid + 1];

        for(EdgeID eid = host_beg; eid < host_end; eid++){

            VertexID nid = ind_h[eid];
            VertexID host_nid = id_map[nid];

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += input_h[host_nid * dim + d]; 
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_block((float*)&tmp_r[0], &input_l[r_offset * dim], dim, r_partID);

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[d] += tmp_r[d];
            }
        }

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            atomicAdd(&output[bid * dim + d], tmp[d]);
        } 
    }
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx <  warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[block_warpid * dim + d] += input_l[local_nid * dim + d];
            }
        }

        //Processing cached neighbor
        CSR_t cache_beg = ptr_c[warpid];
        CSR_t cache_end = ptr_c[warpid + 1];

        for(EdgeID eid = cache_beg; eid < cache_end; eid++){
            
            VertexID nid = ind_c[eid];
            VertexID cache_nid = id_map[nid];

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[block_warpid * dim + d] += input_c[cache_nid * dim + d];
            }            
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            output[warpid * dim + d] = (1.0 + eps) * input_l[warpid * dim + d]; 
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        }
    }    
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[warpid];
        CSR_t remote_end = ptr_r[warpid + 1];

        for(CSR_t eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_warp((float*)&tmp_r[block_warpid * dim], &input_l[r_offset * dim], dim, r_partID);

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += tmp_r[block_warpid * dim + d];
            }
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        } 
    }    
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[block_warpid * dim + d] += input_l[local_nid * dim + d];
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[warpid];
        CSR_t remote_end = ptr_r[warpid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_warp((float*)&tmp_r[block_warpid * dim], &input_l[r_offset * dim], dim, r_partID);

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += tmp_r[block_warpid * dim + d];
            }
        } 

        for(int d = laneid; d < dim; d += WARP_SIZE){
            output[warpid * dim + d] = (1.0 + eps) * input_l[warpid * dim + d]; 
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        }    
    }    
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing local neighbor
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[block_warpid * dim + d] += input_l[local_nid * dim + d];
            }
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            output[warpid * dim + d] = (1.0 + eps) * input_l[warpid * dim + d]; 
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        }
    }    
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(warpid < node_num){
        
        for(int idx = threadIdx.x; idx < warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing cached neighbor (host side)
        CSR_t host_beg = ptr_h[warpid];
        CSR_t host_end = ptr_h[warpid + 1];

        for(EdgeID eid = host_beg; eid < host_end; eid++){

            VertexID nid = ind_h[eid];
            VertexID host_nid = id_map[nid];

            for(int d = threadIdx.x; d < dim; d += blockDim.x){
                tmp[block_warpid * dim + d] += input_h[host_nid * dim + d]; 
            }
        }

        //Processing remote neighbor
        CSR_t remote_beg = ptr_r[warpid];
        CSR_t remote_end = ptr_r[warpid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_warp((float*)&tmp_r[block_warpid * dim], &input_l[r_offset * dim], dim, r_partID);

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += tmp_r[block_warpid * dim + d];
            }
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            output[warpid * dim + d] = (1.0 + eps) * input_l[warpid * dim + d]; 
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&output[warpid * dim + d], tmp[block_warpid * dim + d]);
        }
    }    
}

//========================================= UVM =====================================================
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
){
    const int bid = blockIdx.x;         //global block id 
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id    
    const int srcid = blockIdx.x + myGPUid * nodePerPE;

    if(srcid < numNodes){
       
        const CSR_t local_beg = row_pointers[srcid];
        const CSR_t local_end = row_pointers[srcid + 1];

        __syncwarp();

        for(CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            __syncwarp();

            for(CSR_t eid = warp_beg; eid < warp_end; eid++){
                VertexID nid = column_index[eid];
                VertexID l_partID = nid / nodePerPE;
                VertexID l_offset = nid % nodePerPE;

                for(int d = lanid; d < dim; d += WARP_SIZE){
                    atomicAdd((float*)&output[bid * dim + d], input[l_partID][l_offset * dim + d]);
                }
            }            
        }

    }
}

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
){
    const int bid = blockIdx.x;         //global block id 
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id    
    const int srcid = blockIdx.x + myGPUid * nodePerPE;

    if(srcid < numNodes){

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            output[bid * dim + d] = (1 + eps) * input[myGPUid][bid * dim + d];
        }

        __syncthreads();

        const CSR_t local_beg = row_pointers[srcid];
        const CSR_t local_end = row_pointers[srcid + 1];

        for(CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            __syncthreads();

            for(CSR_t eid = warp_beg; eid < warp_end; eid++){
                VertexID nid = column_index[eid];
                VertexID l_partID = nid / nodePerPE;
                VertexID l_offset = nid % nodePerPE;

                for(int d = lanid; d < dim; d += WARP_SIZE){
                    atomicAdd((float*)&output[bid * dim + d], input[l_partID][l_offset * dim + d]);
                }                
            }            
        }        
    }    
}

__global__
void leo_update_cache_block_cuda( //Update cached node feature
    cache_feat_t*       output,
    const cache_feat_t* input,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode
){
    const int bid = blockIdx.x;         //global warp-id

    if (bid < cache_num){

        VertexID nid = cache_list[bid];
        VertexID r_partID = nid / nodePerPE;
        VertexID r_offset = nid % nodePerPE; 

        nvshmemx_float_get_warp((float*)&output[bid * dim], &input[r_offset * dim], dim, r_partID);
        //nvshmemx_float_get_warp((float*)&output[100 * dim], &input[r_offset * dim], dim, r_partID);        
    }    
}

//CUDA kernel to compute cross-entropy loss
__global__
void CrossEntropyLoss(const int* labels, const cache_feat_t* outputs, value_t* loss, int num_samples, int num_classes){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_samples) return;

    int true_class = labels[idx];
    //if(idx == 1) printf("true_calss: %d\n", true_class);
    value_t sample_loss = -logf(outputs[idx * num_classes + true_class] + 1e-8);
    atomicAdd(&loss[idx], sample_loss);
}

//=================================Back Propagation================================================

__global__
void leo_local_grad_backward_block_cuda( //Processing local and remote neighbors
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
){
    const int bid = blockIdx.x;         //global warp-id
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id   

    extern __shared__  float tmp[];  //Store all intermediate gradients
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate gradients

    if(bid < node_num){

        for (int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads(); 

        //Processing gradients of local neighbors
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1]; 

        for (CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            for(EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_l[eid];
                VertexID local_nid = nid % nodePerPE;

                for(int d = lanid; d < dim; d += WARP_SIZE){
                    tmp[wid * dim + d] += input_grad[local_nid * dim + d];
                }
            }
        }     

        //Processing gradients of remote neighbors
        CSR_t remote_beg = ptr_r[bid];
        CSR_t remote_end = ptr_r[bid + 1];

        for (CSR_t block_beg = remote_beg; block_beg < remote_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, remote_end);

            for(EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_r[eid];
                VertexID r_partID = nid / nodePerPE;
                VertexID r_offset = nid % nodePerPE;
                nvshmemx_float_get_warp((float*)&tmp_r[wid * dim], &input_grad[r_offset * dim], dim, r_partID);

                for (int d = lanid; d < dim; d += WARP_SIZE){
                    tmp[wid * dim + d] += tmp_r[wid * dim + d];
                }
            }
        }

        for(int d = lanid; d < dim; d += WARP_SIZE){
            atomicAdd(&gradients[bid * dim + d], tmp[wid * dim + d]);
        }
    }
}

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
){
    const int bid = blockIdx.x;         //global warp-id
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id   

    extern __shared__  float tmp[];  //Store local intermediate gradients

    if(bid < node_num){

        for (int idx = threadIdx.x; idx < warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing gradients of local neighbors
        CSR_t local_beg = ptr_l[bid];
        CSR_t local_end = ptr_l[bid + 1]; 

        for (CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            for(EdgeID eid = warp_beg; eid < warp_end; eid++){

                VertexID nid = ind_l[eid];
                VertexID local_nid = nid % nodePerPE;

                for(int d = lanid; d < dim; d += WARP_SIZE){
                    tmp[wid * dim + d] += input_grad[local_nid * dim + d];
                }
            }
        }

        for(int d = lanid; d < dim; d += WARP_SIZE){
            atomicAdd(&gradients[bid * dim + d], tmp[wid * dim + d]);
        }   
    }   
}

__global__
void leo_local_grad_backward_warp_V1_cuda(  //Processing local and remote neighbors
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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing gradients of local neighbors
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(CSR_t eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_grad[local_nid * dim + d];
            }
        }

        //Processing gradients of remote neighbors
        CSR_t remote_beg = ptr_r[warpid];
        CSR_t remote_end = ptr_r[warpid + 1];

        for(CSR_t eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_warp((float*)&tmp_r[block_warpid * dim], &input_grad[r_offset * dim], dim, r_partID);

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += tmp_r[block_warpid * dim + d];
            }
        }  

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&gradients[warpid * dim + d], tmp[block_warpid * dim + d]);
        }     
    }        
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();  

        //Processing gradients of local neighbors
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(CSR_t eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_grad[local_nid * dim + d];
            }
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&gradients[warpid * dim + d], tmp[block_warpid * dim + d]);
        }               
    }    
}

//========================================  GIN ========================================================
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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results
    float* tmp_r = (float*) &tmp[warpPerBlock * dim]; //Cache remote intermediate result

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < 2 * warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads();

        //Processing gradients of local neighbors
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_grad[local_nid * dim + d];
            }
        }    

        //Processing gradients of remote neighbors
        CSR_t remote_beg = ptr_r[warpid];
        CSR_t remote_end = ptr_r[warpid + 1];

        for(EdgeID eid = remote_beg; eid < remote_end; eid++){

            VertexID nid = ind_r[eid];
            VertexID r_partID = nid / nodePerPE;
            VertexID r_offset = nid % nodePerPE;
            nvshmemx_float_get_warp((float*)&tmp_r[block_warpid * dim], &input_grad[r_offset * dim], dim, r_partID);

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += tmp_r[block_warpid * dim + d];
            }
        }  

        for(int d = laneid; d < dim; d += WARP_SIZE){
            gradients[warpid * dim + d] = (1.0 + eps) * input_grad[warpid * dim + d];
        }                   

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&gradients[warpid * dim + d], tmp[block_warpid * dim + d]);
        }
    }
}

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
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id
    const int block_warpid = threadIdx.x >> 5;  // block warp_id
    const int laneid = threadIdx.x & 31;        // warp thread-id 

    extern __shared__ float tmp[];  //Store all intermediate results

    if(warpid < node_num){

        for(int idx = threadIdx.x; idx < warpPerBlock * dim; idx += blockDim.x){
            tmp[idx] = 0.0f;
        }
        __syncthreads(); 

        //Processing gradients of local neighbors
        CSR_t local_beg = ptr_l[warpid];
        CSR_t local_end = ptr_l[warpid + 1];

        for(EdgeID eid = local_beg; eid < local_end; eid++){

            VertexID nid = ind_l[eid];
            VertexID local_nid = nid % nodePerPE;

            for(int d = laneid; d < dim; d += WARP_SIZE){
                tmp[block_warpid * dim + d] += input_grad[local_nid * dim + d];
            }
        }

        for(int d = laneid; d < dim; d += WARP_SIZE){
            gradients[warpid * dim + d] = (1.0 + eps) * input_grad[warpid * dim + d];
        }                   

        for(int d = laneid; d < dim; d += WARP_SIZE){
            atomicAdd(&gradients[warpid * dim + d], tmp[block_warpid * dim + d]);
        }                       
    }
}

//========================================= UVM =====================================================
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
){
    const int bid = blockIdx.x;         //global block id 
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id    
    const int srcid = blockIdx.x + myGPUid * nodePerPE;

    if(srcid < numNodes){

        const CSR_t local_beg = row_pointers[srcid];
        const CSR_t local_end = row_pointers[srcid + 1];

        for(CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            //__syncwarp();

            for(CSR_t eid = warp_beg; eid < warp_end; eid++){
                VertexID nid = column_index[eid];
                VertexID l_partID = nid / nodePerPE;
                VertexID l_offset = nid % nodePerPE;

                for(int d = lanid; d < dim; d += WARP_SIZE){
                    atomicAdd((float*)&gradients[bid * dim + d], input_grad[l_partID][l_offset * dim + d]);
                }
            }            
        }        
    }
}

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
){
    const int bid = blockIdx.x;         //global block id 
    const int wid = threadIdx.x / 32;   //block-level warp-id
    const int lanid = threadIdx.x % 32; //lane-id    
    const int srcid = blockIdx.x + myGPUid * nodePerPE;

    if(srcid < numNodes){

        for(int d = threadIdx.x; d < dim; d += blockDim.x){
            gradients[bid * dim + d] = (1 + eps) * input_grad[myGPUid][bid * dim + d];
        }

        __syncthreads();

        const CSR_t local_beg = row_pointers[srcid];
        const CSR_t local_end = row_pointers[srcid + 1];

        for(CSR_t block_beg = local_beg; block_beg < local_end; block_beg += warpPerBlock * partSize){

            CSR_t warp_beg = block_beg + wid * partSize;
            CSR_t warp_end = min(warp_beg + partSize, local_end);

            __syncthreads();

            for(CSR_t eid = warp_beg; eid < warp_end; eid++){
                VertexID nid = column_index[eid];
                VertexID l_partID = nid / nodePerPE;
                VertexID l_offset = nid % nodePerPE;

                for(int d = lanid; d < dim; d += WARP_SIZE){
                    atomicAdd((float*)&gradients[bid * dim + d], input_grad[l_partID][l_offset * dim + d]);
                }
            }            
        }        
    }    
}

__global__
void leo_update_cahce_backward_block_cuda( //Update cached node gradients in back propagation
    value_t*            gradients,
    const value_t*      input_grad,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode
){
    const int bid = blockIdx.x;         //global warp-id

    if(bid < cache_num){

        VertexID nid = cache_list[bid];
        VertexID r_partID = nid / nodePerPE;
        VertexID r_offset = nid % nodePerPE;

        nvshmemx_float_get_warp((float*)&gradients[bid * dim], &input_grad[r_offset * dim], dim, r_partID);
    }
}

__global__
void leo_update_cache_backward_warp_cuda( //Update cached node gradients in back propagation(Warp Version)
    value_t*            gradients,
    const value_t*      input_grad,
    const VertexID*     cache_list,
    const int           cache_num,
    const int           dim,
    const VertexID      nodePerPE,
    const int           mynode
){
    const int warpid = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);  // global warp-id

    if(warpid < cache_num){

        VertexID nid = cache_list[warpid];
        VertexID r_partID = nid / nodePerPE;
        VertexID r_offset = nid % nodePerPE;

        nvshmemx_float_get_warp((float*)&gradients[warpid * dim], &input_grad[r_offset * dim], dim, r_partID);
    }
}

__global__
void SoftmaxCrossEntroy_backward_cuda(
    value_t*            gradients,
    const cache_feat_t* softmax_output, 
    const int*          labels, 
    const int           num_samples,
    const int           num_classes
){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= num_samples) return;  

    for(int i = 0; i < num_classes; ++i){
        gradients[idx * num_classes + i] = softmax_output[idx * num_classes + i] - (labels[idx]==i);
    }  
}

//Update process for Adam Optimizer
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
){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size){
        //Update biased first moment estimate
        m[idx] = beta1 * m[idx] + (1 - beta1) * gradients[idx];
        //Update biased second raw moment estimate
        v[idx] = beta2 * v[idx] + (1 - beta2) * gradients[idx] * gradients[idx];
        //Compute bias-corrected first moment estimate
        float m_hat = m[idx] / (1 - pow(beta1, t));
        //Compute bias-corrected second raw moment estimate
        float v_hat = v[idx] / (1 - pow(beta2, t));
        //Update parameters
        out_param[idx] -= lr* m_hat / (sqrt(v_hat) + eps);
    }
}

}   //namespace common
}   //namescpce GNNPro_lib 