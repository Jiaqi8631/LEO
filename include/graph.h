#include <iostream>
#include <sys/stat.h>
#include <vector>
#include "type.h"
#include "run_config.h"
#include "freq_reorder.h"
#include "cache.h"
#include "partition.h"
#include "timer.h"
#include "param.h"

namespace GNNPro_lib{
namespace common{

class Graph{
public:
    int nranks;
    int mynode;

    VertexID        vertices_num;
    EdgeID          edges_num;
    VertexID        ID_LowerBound;
    VertexID        ID_UpperBound;
    int             cache_num;
    int             h_cache_num;

    cache_feat_t    *cache_feature;
    cache_feat_t    *cache_host_feature;
    cache_feat_t    *combined_local_feature;
    std::vector<file_feat_t> cached_feature;
    std::vector<file_feat_t> tmp_host_feature;

    InputInfo*      config;
    FreqReorder*    freqreorder;
    Cache*          cachemanager;
    Weight*         weight;
    
    file_index_t *file_ptr = NULL;
    file_index_t *file_ind = NULL;
    file_feat_t  *feature  = NULL;
    int          *label    = NULL;
    
    std::vector<CSR_t>       full_csr_ptr;
    std::vector<CSR_t>       full_csr_ind;
    std::vector<CSR_t>       full_csr_T_ptr;
    std::vector<CSR_t>       full_csr_T_ind;
    std::vector<CSR_t>       ID_split_array;
    std::vector<VertexID>    cache_IDlist;
    std::vector<VertexID>    cache_host_IDlist;

    std::vector<CSR_t>       local_csr_ptr;
    std::vector<CSR_t>       local_csr_ind;
    std::vector<CSR_t>       cache_csr_ptr;
    std::vector<CSR_t>       cache_csr_ind;
    std::vector<CSR_t>       remote_csr_ptr;
    std::vector<CSR_t>       remote_csr_ind;
    std::vector<CSR_t>       host_csr_ptr;
    std::vector<CSR_t>       host_csr_ind;
    std::vector<int>         local_IDmap;

    std::vector<CSR_t>       local_csr_T_ptr;
    std::vector<CSR_t>       local_csr_T_ind;
    std::vector<CSR_t>       remote_csr_T_ptr;
    std::vector<CSR_t>       remote_csr_T_ind;

    //Flag to spilt neighbors
    bool   CacheFlag = 1;
    bool   HostCacheFlag = 0;
    bool   RemoteFlag = 1;
    bool   UVMFlag = 0;
    bool   NVSHMEMFlag = 1;
    bool   TrainFlag = 0;
    bool   TransRemoteFlag = 1; //Used to mark whether the partition of the transposed neighbor matrix has remote neighbors.
    bool   PtrConcatFlag = 0; //Used to mark whether local and cached CSR are merged
    bool   BatchloadFlag = 0; //Used to mard whether host data is loaded in batches
    int    BatchLoadSize;     //the group size of datd loading

    //GPU part
    //Graph Topo and Structure
    cache_feat_t    *d_local_feature;
    cache_feat_t    *d_cache_feature;
    cache_feat_t    *d_cache_feature_update;
    cache_feat_t    *d_cache_host_feature_update;

    CSR_t           *d_row_ptr_l;
    CSR_t           *d_col_ind_l;
    CSR_t           *d_row_ptr_c;
    CSR_t           *d_col_ind_c;
    CSR_t           *d_row_ptr_r;
    CSR_t           *d_col_ind_r;
    CSR_t           *d_row_ptr_h;
    CSR_t           *d_col_ind_h;

    CSR_t           *d_trans_row_ptr_l;
    CSR_t           *d_trans_col_ind_l;
    CSR_t           *d_trans_row_ptr_r;
    CSR_t           *d_trans_col_ind_r;

    int             *d_label;
    int             *d_id_map;
    value_t         *d_loss;
    VertexID        *d_cache_list;
    VertexID        *d_cache_host_list;

    //Matrix
    weight_t        *dW1_l;
    weight_t        *dW1_c;
    weight_t        *dW1_h;
    weight_t        *dW2_l;
    weight_t        *dW2_c;
    weight_t        *dW2_h;
    weight_t        *dW3_l;
    weight_t        *dW3_c;
    weight_t        *dW3_h;
    weight_t        *dW4_l;
    weight_t        *dW4_c;
    weight_t        *dW4_h;
    weight_t        *dW5_l;
    weight_t        *dW5_c;
    weight_t        *dW5_h;
    weight_t        *dW6_l;        
    
    //buff
    cache_feat_t    *d_buff1_l; //first layer buff for mm of local neighbors and weights
    cache_feat_t    *d_buff1_c; //first layer buff for mm of cached neighbors and weights
    cache_feat_t    *d_buff1_h; //first layer buff for mm of cached neighbors (host side) and weights
    cache_feat_t    *d_buff1;   //first layer buff for aggregation

    cache_feat_t    *d_buff2_l;
    cache_feat_t    *d_buff2_c;
    cache_feat_t    *d_buff2_h;
    cache_feat_t    *d_buff2;
    cache_feat_t    *d_buff2_s; //second layer buff for softmax function

    cache_feat_t    *d_buff3;
    cache_feat_t    *d_buff4;
    cache_feat_t    *d_buff5;
    cache_feat_t    *d_buff5_l;
    cache_feat_t    *d_buff5_s;

    //Update buff
    cache_feat_t    *d_buff1_update_c; //buff in the first layer used to update cached data (GPU side)
    cache_feat_t    *d_buff1_update_h; //buff in the first layer used to update cached data (host side)
    cache_feat_t    *d_buff2_update_c;
    cache_feat_t    *d_buff2_update_h;
    cache_feat_t    *d_buff3_update_c;
    cache_feat_t    *d_buff3_update_h;
    cache_feat_t    *d_buff4_update_c;
    cache_feat_t    *d_buff4_update_h;
    cache_feat_t    *d_buff5_update_c;
    cache_feat_t    *d_buff5_update_h;
    cache_feat_t    *d_buff6_update_c;
    cache_feat_t    *d_buff6_update_h;

    //gradient
    value_t         *d_grad1_l_W; 
    value_t         *d_grad1_c_W; 
    value_t         *d_grad1_h_W;
    value_t         *h_grad1_h_W; 
    value_t         *d_grad1_l;   
    value_t         *d_grad1_c;
    value_t         *d_grad1_h;
    value_t         *d_grad1;     

    value_t         *d_grad2_l_W; 
    value_t         *d_grad2_c_W; 
    value_t         *d_grad2_h_W;  
    value_t         *d_grad2_l;   
    value_t         *d_grad2_c;
    value_t         *d_grad2_h;
    value_t         *d_grad2_s;   //gradient buff of second-layer SoftmaxCrossEntroy loss
    value_t         *d_grad2;

    value_t         *d_grad3_l;
    value_t         *d_grad3_c;
    value_t         *d_grad3_h;
    value_t         *d_grad3;
    value_t         *d_grad3_l_W;
    value_t         *d_grad3_c_W;
    value_t         *d_grad3_h_W;

    value_t         *d_grad4_l;
    value_t         *d_grad4_c;
    value_t         *d_grad4_h;
    value_t         *d_grad4;
    value_t         *d_grad4_l_W;
    value_t         *d_grad4_c_W;
    value_t         *d_grad4_h_W;  

    value_t         *d_grad5_l;
    value_t         *d_grad5_c;
    value_t         *d_grad5_h;    
    value_t         *d_grad5_W;
    value_t         *d_grad5;       
    value_t         *d_grad5_s;

    //For GIN Model
    value_t         *d_grad0_l_W;
    value_t         *d_gard0_c_W;
    value_t         *h_grad0_h_W;

    void Load();
    //CPU Function
    void ConstuctGraph();
    void PartitionGraphTopo();
    void Build_Cache();
    void Prepare_LocalData();
    void Prepare_LocalData_v1(); //This function is uesd in batch loading scenarios

    void Create_CSR();
    void Create_CSR_w_R_wo_CU(); 
    void Create_CSR_w_CR_wo_U();
    void Create_CSR_w_CRU();
    void Create_CSR_Combined_w_CR_wo_U();
    void Create_CSR_transpose();

    void Create_IDmap();
    void Create_Weight();
    void Extract_Featrue();

    void Config_Initial();

    void Combine_local_feature();

    void Free_Host_Memory();

    //GPU Function
    void GPU_Initial();
    void GPU_Initial_v1(); //This function is uesd in batch loading scenarios
    void Weight_Initial();
    void Buff_Initial();
    void Grad_buff_Initial();
    void Device_data_initial();
    void Copy_Host2Device();

    void Free_Device_Memory();
    Graph(int nranks, int mynode);
    ~Graph();
};

inline off_t fsize(const char *filename){
	struct stat st; 
	if (stat(filename, &st) == 0)
		return st.st_size;
	return -1; 
}

}   //namespace common
}   //namescpce GNNPro_lib  