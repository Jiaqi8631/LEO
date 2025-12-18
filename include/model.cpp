#include "model.h"

namespace GNNPro_lib{
namespace common{

GCN_Model::GCN_Model(Graph* G){
    mynode = G->mynode;
    nranks = G->nranks;

    v_num = G->vertices_num;
    e_num = G->edges_num;
    local_v_num = G->ID_UpperBound - G->ID_LowerBound;
    cache_v_num = G->cache_num;
    host_cache_v_num = G->h_cache_num;
    
    nodePerPE = (v_num + nranks - 1) / nranks;

    in_dim = G->config->in_dim;
    hid_dim = G->config->hidden_dim;
    out_dim = G->config->out_dim;

    local_feature = G->d_local_feature;
    cache_feature = G->d_cache_feature;
    cache_host_feature = G->cache_host_feature;
    update_cache_feature = G->d_cache_feature_update;
    update_cache_host_featrue = G->d_cache_host_feature_update;

    //graph structure
    row_ptr_l = G->d_row_ptr_l;
    col_ind_l = G->d_col_ind_l;
    row_ptr_c = G->d_row_ptr_c;
    col_ind_c = G->d_col_ind_c;
    row_ptr_r = G->d_row_ptr_r;
    col_ind_r = G->d_col_ind_r;
    row_ptr_h = G->d_row_ptr_h;
    col_ind_h = G->d_col_ind_h;

    T_row_ptr_l = G->d_trans_row_ptr_l;
    T_col_ind_l = G->d_trans_col_ind_l;
    T_row_ptr_r = G->d_trans_row_ptr_r;
    T_col_ind_r = G->d_trans_col_ind_r;

    label = G->d_label;
    id_map = G->d_id_map;
    loss = G->d_loss;
    cache_list = G->d_cache_list;
    cache_host_list = G->d_cache_host_list;

    CacheFlag = G->CacheFlag;
    RemoteFlag = G->RemoteFlag;
    UVMFlag = G->UVMFlag;
    TrainFlag = G->TrainFlag;
    TransRemoteFlag = G->TransRemoteFlag;

    //Weight
    W1_l = G->dW1_l;
    W1_c = G->dW1_c;
    W1_h = G->dW1_h;
    W2_l = G->dW2_l;
    W2_c = G->dW2_c;
    W2_h = G->dW2_h;


    //Buff
    buff1_l = G->d_buff1_l;
    buff1_c = G->d_buff1_c;
    buff1_h = G->d_buff1_h;
    buff1 = G->d_buff1;

    buff2_l = G->d_buff2_l;
    buff2_c = G->d_buff2_c;
    buff2_h = G->d_buff2_h;
    buff2 = G->d_buff2;
    buff2_s = G->d_buff2_s;

if(TrainFlag == true){
    //Gradient
    grad1_l_W = G->d_grad1_l_W;    
    grad1_l   = G->d_grad1_l;    
    grad1     = G->d_buff1;
    grad2_l_W = G->d_grad2_l_W;   
    grad2_s   = G->d_grad2_s;
    grad2_l   = G->d_grad2_l;
    
    opt1 = new AdamOptimizer(hid_dim * out_dim);
    opt2 = new AdamOptimizer(in_dim * hid_dim);

    if(CacheFlag == true){
        grad1_c_W = G->d_grad1_c_W;
        grad2_c_W = G->d_grad2_c_W;
        grad1_c   = G->d_grad1_c;
        grad2_c   = G->d_grad2_c;

        opt3 = new AdamOptimizer(hid_dim * out_dim);
        opt4 = new AdamOptimizer(in_dim * hid_dim);

        if(UVMFlag == true){
            grad1_h_W = G->h_grad1_h_W;
            grad2_h_W = G->d_grad2_h_W;
            grad1_h   = G->d_grad1_h;
            grad2_h   = G->d_grad2_h;

            opt5 = new AdamOptimizer(hid_dim * out_dim);
            opt6 = new AdamOptimizer(in_dim * hid_dim, false); //CPU side
        }
    }
} //TrainFlag

} //GCN_Model


GIN_Model::GIN_Model(Graph* G){
    mynode = G->mynode;
    nranks = G->nranks;

    v_num = G->vertices_num;
    e_num = G->edges_num;
    local_v_num = G->ID_UpperBound - G->ID_LowerBound;
    cache_v_num = G->cache_num;
    host_cache_v_num = G->h_cache_num;
    
    nodePerPE = (v_num + nranks - 1) / nranks;

    in_dim = G->config->in_dim;
    hid_dim = G->config->hidden_dim;
    out_dim = G->config->out_dim; 

    eps = RunConfig::eps;

    local_feature = G->d_local_feature;
    cache_feature = G->d_cache_feature;
    cache_host_feature = G->cache_host_feature;
    //update_cache_feature = G->d_cache_feature_update;
    //update_cache_host_featrue = G->d_cache_host_feature_update;

    //graph structure
    row_ptr_l = G->d_row_ptr_l;
    col_ind_l = G->d_col_ind_l;
    row_ptr_c = G->d_row_ptr_c;
    col_ind_c = G->d_col_ind_c;
    row_ptr_r = G->d_row_ptr_r;
    col_ind_r = G->d_col_ind_r;
    row_ptr_h = G->d_row_ptr_h;
    col_ind_h = G->d_col_ind_h;

    T_row_ptr_l = G->d_trans_row_ptr_l;
    T_col_ind_l = G->d_trans_col_ind_l;
    T_row_ptr_r = G->d_trans_row_ptr_r;
    T_col_ind_r = G->d_trans_col_ind_r;

    T_row_ptr_l = G->d_trans_row_ptr_l;
    T_col_ind_l = G->d_trans_col_ind_l;
    T_row_ptr_r = G->d_trans_row_ptr_r;
    T_col_ind_r = G->d_trans_col_ind_r;

    label = G->d_label;
    id_map = G->d_id_map;
    loss = G->d_loss;
    cache_list = G->d_cache_list;
    cache_host_list = G->d_cache_host_list;

    CacheFlag = G->CacheFlag;
    RemoteFlag = G->RemoteFlag;
    UVMFlag = G->UVMFlag;
    TrainFlag = G->TrainFlag;
    TransRemoteFlag = G->TransRemoteFlag;    

    //Weight
    W1_l = G->dW1_l;
    W1_c = G->dW1_c;
    W1_h = G->dW1_h;
    W2_l = G->dW2_l;
    W2_c = G->dW2_c;
    W2_h = G->dW2_h;
    W3_l = G->dW3_l;
    W3_c = G->dW3_c;
    W3_h = G->dW3_h;
    W4_l = G->dW4_l;
    W4_c = G->dW4_c;
    W4_h = G->dW4_h;
    W5_l = G->dW5_l;
    W5_c = G->dW5_c;
    W5_h = G->dW5_h; 
    W6_l = G->dW6_l;

    //Buff
    buff1_l = G->d_buff1_l;
    buff1_c = G->d_buff1_c;
    buff1_h = G->d_buff1_h;
    buff1 = G->d_buff1; 

    buff2_l = G->d_buff2_l;
    buff2_c = G->d_buff2_c;
    buff2_h = G->d_buff2_h;
    buff2 = G->d_buff2; 

    buff3 = G->d_buff3;
    buff4 = G->d_buff4;
    buff5 = G->d_buff5;
    buff5_l = G->d_buff5_l;
    buff5_s = G->d_buff5_s;
    
    //Buff for updates
    buff1_update_c = G->d_buff1_update_c;
    buff1_update_h = G->d_buff1_update_h;
    buff2_update_c = G->d_buff2_update_c;
    buff2_update_h = G->d_buff2_update_h; 
    buff3_update_c = G->d_buff3_update_c;
    buff3_update_h = G->d_buff3_update_h;
    buff4_update_c = G->d_buff4_update_c;
    buff4_update_h = G->d_buff4_update_h;  
    buff5_update_c = G->d_buff5_update_c;
    buff5_update_h = G->d_buff5_update_h;
    buff6_update_c = G->d_buff6_update_c;
    buff6_update_h = G->d_buff6_update_h;

if(TrainFlag == true){
    //Gradients
    grad0_l_W = G->d_grad0_l_W;
    grad1_l = G->d_grad1_l;
    grad1_l_W = G->d_grad1_l_W;
    grad1 = G->d_grad1;

    grad2_l = G->d_grad2_l;
    grad2_l_W = G->d_grad2_l_W;
    grad2 = G->d_grad2;

    grad3_l = G->d_grad3_l;
    grad3_l_W = G->d_grad3_l_W;
    grad3 = G->d_grad3;

    grad4_l = G->d_grad4_l;
    grad4_l_W = G->d_grad4_l_W;
    grad4 = G->d_grad4;

    grad5_l = G->d_grad5_l;
    grad5_W = G->d_grad5_W;
    grad5 = G->d_grad5;
    grad5_s = G->d_grad5_s;

    opt1 = new AdamOptimizer(out_dim * hid_dim);
    opt2 = new AdamOptimizer(hid_dim * hid_dim);
    opt3 = new AdamOptimizer(hid_dim * hid_dim);
    opt4 = new AdamOptimizer(hid_dim * hid_dim);
    opt5 = new AdamOptimizer(hid_dim * hid_dim);
    opt6 = new AdamOptimizer(hid_dim * in_dim);

    if(CacheFlag == true){
        grad0_c_W = G->d_gard0_c_W;
        grad1_c = G->d_grad1_c;
        grad1_c_W = G->d_grad1_c_W;

        grad2_c = G->d_grad2_c;
        grad2_c_W = G->d_grad2_c_W;

        grad3_c = G->d_grad3_c;
        grad3_c_W = G->d_grad3_c_W;

        grad4_c = G->d_grad4_c;
        grad4_c_W = G->d_grad4_c_W;

        grad5_c = G->d_grad5_c;

        opt7 = new AdamOptimizer(hid_dim * hid_dim);
        opt8 = new AdamOptimizer(hid_dim * hid_dim);
        opt9 = new AdamOptimizer(hid_dim * hid_dim);
        opt10 = new AdamOptimizer(hid_dim * hid_dim);
        opt11 = new AdamOptimizer(hid_dim * in_dim);

        if(UVMFlag == true){
            grad0_h_W = G->h_grad0_h_W;
            grad1_h = G->d_grad1_h;
            grad1_h_W = G->d_grad1_h_W; 

            grad2_h = G->d_grad2_h;
            grad2_h_W = G->d_grad2_h_W;

            grad3_h = G->d_grad3_h;
            grad3_h_W = G->d_grad3_h_W;

            grad4_h = G->d_grad4_h;
            grad4_h_W = G->d_grad4_h_W;

            grad5_h = G->d_grad5_h;

            opt12 = new AdamOptimizer(hid_dim * hid_dim);
            opt13 = new AdamOptimizer(hid_dim * hid_dim);
            opt14 = new AdamOptimizer(hid_dim * hid_dim);
            opt15 = new AdamOptimizer(hid_dim * hid_dim);
            opt16 = new AdamOptimizer(hid_dim * in_dim, false);
        }
    }
}

}

Model* BuildModel(Model* model, Graph* graph){
    ModelType mt = StringToModel(graph->config->algorithm);
    switch (mt){
    case GCN:
        model = new GCN_Model(graph);
        break;
    
    case GIN:
        model = new GIN_Model(graph);
        break;

    default:
        CHECK(false) << "Invalid model! ";
        break;
    }

    return model;
}

}   //namespace common
}   //namescpce GNNPro_lib  