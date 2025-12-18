#include <mpi.h>
#include <thread>
#include "graph.h"
#include "cuda/gnn_layer.cuh"

namespace GNNPro_lib{
namespace common{

class Model{
public:
    int             mynode;
    int             nranks;

    VertexID        v_num;
    VertexID        local_v_num;
    int             cache_v_num;
    int             host_cache_v_num;
    EdgeID          e_num;
    VertexID        nodePerPE;

    int             in_dim;
    int             hid_dim;
    int             out_dim;

    cache_feat_t    *local_feature;
    cache_feat_t    *cache_feature;
    cache_feat_t    *cache_host_feature;
    cache_feat_t    *update_cache_feature;
    cache_feat_t    *update_cache_host_featrue;

    CSR_t           *row_ptr_l;
    CSR_t           *col_ind_l;
    CSR_t           *row_ptr_c;
    CSR_t           *col_ind_c;
    CSR_t           *row_ptr_r;
    CSR_t           *col_ind_r;
    CSR_t           *row_ptr_h;
    CSR_t           *col_ind_h;

    CSR_t           *T_row_ptr_l;
    CSR_t           *T_col_ind_l;
    CSR_t           *T_row_ptr_r;
    CSR_t           *T_col_ind_r;

    int             *label;
    int             *id_map;
    value_t         *loss;
    VertexID        *cache_list;
    VertexID        *cache_host_list;

    bool   CacheFlag;
    bool   RemoteFlag;
    bool   UVMFlag;
    bool   TrainFlag;
    bool   TransRemoteFlag;

    //Weight
    weight_t        *W1_l;
    weight_t        *W1_c;
    weight_t        *W1_h;
    weight_t        *W2_l;
    weight_t        *W2_c;
    weight_t        *W2_h;

    
    //Buff
    cache_feat_t    *buff1_l;
    cache_feat_t    *buff1_c;
    cache_feat_t    *buff1_h; 
    cache_feat_t    *buff1; 

    cache_feat_t    *buff2_l;
    cache_feat_t    *buff2_c;
    cache_feat_t    *buff2_h;
    cache_feat_t    *buff2; 
    cache_feat_t    *buff2_s;

    //Gradient
    value_t         *grad1_l_W;
    value_t         *grad1_c_W;
    value_t         *grad1_h_W;
    value_t         *grad1_l;
    value_t         *grad1_c;
    value_t         *grad1_h;
    value_t         *grad1;

    value_t         *grad2_l_W;
    value_t         *grad2_c_W;
    value_t         *grad2_h_W;
    value_t         *grad2_l;
    value_t         *grad2_c;
    value_t         *grad2_h;
    value_t         *grad2_s;

virtual void Train() { CHECK(false) << "Unimplemented";};
virtual void Validate() { CHECK(false) << "Unimplemented";};
void ComputeLoss();

    ~Model(){};
};

class GCN_Model : public Model{
public:
    cache_feat_t    *v_buff1_l;
    cache_feat_t    *v_buff1_c;
    cache_feat_t    *v_buff1;
    cache_feat_t    *v_buff_infer;
    AdamOptimizer   *opt1, *opt2, *opt3, *opt4, *opt5, *opt6;

    GCN_Model(Graph* G);
    ~GCN_Model(){
        delete opt1;
        delete opt2;
        delete opt3;
        delete opt4;
        delete opt5;
        delete opt6;
    };
    void Train() override;
    void Validate() override;

    void Forward();
    void Forward_test1();
    void Forward_Option();
    void Forward_LocalOnly();
    void Forward_wo_Cache();
    void Forward_wGPU_Cache();
    void Forward_wCPUGPU_Cache();
    void Forward_wCPUGPU_Cache_Stream();   

    void Backward();
    void Backward_Stream();
    void Backward_Option();
    void Backward_LocalOnly();
    void Backward_LocalOnly_Stream();
    void Backward_wGPU_Cache();
    void Backward_wGPU_Cache_Stream();
    void Backward_wCPUGPU_Cache();
    void Backward_wCPUGPU_Cache_Stream();
   
    void Aggregate(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim);
    void Aggregate_Warp(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim);
    void Aggregate_Thread(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim);

    void UpdateCache(cache_feat_t* output, cache_feat_t* input, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream);
    void UpdateCache_Stream(cache_feat_t* output, cache_feat_t* input, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream);
    void UpdateCache_Backward(value_t* out_grad, value_t* in_grad, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream);
    void UpdateCache_Backward_Stream(value_t* out_grad, value_t* in_grad, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream);

    void Aggregate_Backward(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream);
    void Aggregate_Backward_Stream(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream);
    void Aggregate_Backward_Warp(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream);
    void Aggregate_Backward_Warp_Stream(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream);

    void Validate_cublas_mm();
    void Validate_leo_aggregate();
    void Validate_infer();
};

class GIN_Model : public Model{
public:
    weight_t        *W3_l;
    weight_t        *W3_c;
    weight_t        *W3_h;
    weight_t        *W4_l;
    weight_t        *W4_c;
    weight_t        *W4_h;
    weight_t        *W5_l;
    weight_t        *W5_c;
    weight_t        *W5_h;
    weight_t        *W6_l;

    //Buff
    cache_feat_t    *buff3;
    cache_feat_t    *buff4;
    cache_feat_t    *buff5;
    cache_feat_t    *buff5_l;
    cache_feat_t    *buff5_s;

    cache_feat_t    *buff1_update_c;
    cache_feat_t    *buff1_update_h;
    cache_feat_t    *buff2_update_c;
    cache_feat_t    *buff2_update_h;
    cache_feat_t    *buff3_update_c;
    cache_feat_t    *buff3_update_h;
    cache_feat_t    *buff4_update_c;
    cache_feat_t    *buff4_update_h;
    cache_feat_t    *buff5_update_c;
    cache_feat_t    *buff5_update_h;
    cache_feat_t    *buff6_update_c;
    cache_feat_t    *buff6_update_h;

    //Gradient
    value_t         *grad0_l_W;
    value_t         *grad0_c_W;
    value_t         *grad0_h_W;

    value_t         *grad1_l;
    value_t         *grad1_c;
    value_t         *grad1_h;    
    value_t         *grad1_l_W;
    value_t         *grad1_c_W;
    value_t         *grad1_h_W;
    value_t         *grad1;     

    value_t         *grad2_l;
    value_t         *grad2_c;
    value_t         *grad2_h;    
    value_t         *grad2_l_W;
    value_t         *grad2_c_W;
    value_t         *grad2_h_W;
    value_t         *grad2;  

    value_t         *grad3_l;
    value_t         *grad3_c;
    value_t         *grad3_h;    
    value_t         *grad3_l_W;
    value_t         *grad3_c_W;
    value_t         *grad3_h_W;
    value_t         *grad3;    

    value_t         *grad4_l;
    value_t         *grad4_c;
    value_t         *grad4_h;    
    value_t         *grad4_l_W;
    value_t         *grad4_c_W;
    value_t         *grad4_h_W;
    value_t         *grad4;

    value_t         *grad5_l;
    value_t         *grad5_c;
    value_t         *grad5_h;
    value_t         *grad5_W;
    value_t         *grad5;
    value_t         *grad5_s;

    float eps;

    AdamOptimizer   *opt1, *opt2, *opt3, *opt4, *opt5, *opt6, *opt7, *opt8;
    AdamOptimizer   *opt9, *opt10, *opt11, *opt12, *opt13, *opt14, *opt15, *opt16;

    GIN_Model(Graph* G);
    ~GIN_Model(){
        delete opt1;
        delete opt2;
        delete opt3;
        delete opt4;
        delete opt5;
        delete opt6;
        delete opt7;
        delete opt8;
        delete opt9;
        delete opt10;
        delete opt11;
        delete opt12;
        delete opt13;
        delete opt14;
        delete opt15;
        delete opt16;        
    };

    void Train() override;
    void Validate() override;

    void Forward_Basic();
    void Forward_Block();
    void Forward_Warp();

    void Forward_Option();
    void Forward_wCPUGPU_Cache();
    void Forward_wGPU_Cache();
    void Forward_wo_Cache();
    void Forward_LocalOnly();

    void Forward_Option_Stream();
    void Forward_wCPUGPU_Cache_Stream();

    void Backward();
    void Backward_Option();
    void Backward_wCPUGPU_Cache();
    void Backward_wGPU_Cache();
    void Backward_LocalOnly();

    void Backward_Option_Stream();
    void Backward_wCPUGPU_Cache_Stream();
    void Backward_wGPU_Cache_Stream();
    void Backward_LocalOnly_Stream();

    void Aggregate_Basic(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim);
    void Aggregate_Block(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim);
    void Aggregate_Warp(cache_feat_t* output, const cache_feat_t* input_l, const cache_feat_t* input_c, const cache_feat_t* input_h, const int outdim);

    void Aggregate_Backward_Warp(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream);
    void Aggregate_Backward_Warp_Stream(value_t* out_grad, value_t* in_grad, const int dim, cudaStream_t stream);

    void UpdateCache(cache_feat_t* output, cache_feat_t* input, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream);
    void UpdateCache_Stream(cache_feat_t* output, cache_feat_t* input, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream);
    void UpdateCache_Backward(value_t* out_grad, value_t* in_grad, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream);
    void UpdateCache_Backward_Stream(value_t* out_grad, value_t* in_grad, VertexID* cache_list, int cache_num, int dim, cudaStream_t stream);
};

Model* BuildModel(Model* model, Graph* graph);

}   //namespace common
}   //namescpce GNNPro_lib  