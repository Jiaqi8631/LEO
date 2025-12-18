#include <unistd.h>
#include <sys/time.h>
#include <assert.h>
#include <random>
#include "graph.h"
#include "logging.h"

//#define debug 1

namespace GNNPro_lib{
namespace common{

Graph::Graph(int nranks, int mynode) : nranks(nranks), mynode(mynode){
    config = new InputInfo();
}

void Graph::Config_Initial(){

    if(config->place_strategy.empty()){
        if(mynode == 0) printf("Use default data placement policy\n");
    }else{
        RunConfig::placement_strategy = StringToStrategy(config->place_strategy);
    }

    BatchLoadSize = config->BatchLoadSize;
    if(BatchLoadSize > 0){
        BatchloadFlag = true;
    }else{
        BatchloadFlag = false;
    }

    switch (RunConfig::placement_strategy){
    case kLocalRemoteCache:
        RemoteFlag = 1;
        CacheFlag = 1;
        break;
    
    case kLocalRemote:
        RemoteFlag = 1;
        CacheFlag = 0;
        break;

    default:
        break;
    }

    if(RunConfig::train_flag == 1){
        TrainFlag = 1;
    }else{
        TrainFlag = 0;
    }

    if(RunConfig::h_cache_flag == 1){
        HostCacheFlag = 1;
        UVMFlag = 1;
    }else{
        HostCacheFlag = 0;
        UVMFlag = 0;
    }

    if(RunConfig::ptr_concat_flag == 1){
        PtrConcatFlag = 1;
    }else{
        PtrConcatFlag = 0;
    }
}

void Graph::Load(){

    if(BatchloadFlag == true){
        if(mynode == 0) std::cout << "The host is loading data in batches..."<< std::endl;
        for (int i = 0; i < nranks/BatchLoadSize; ++i){
            if(mynode/BatchLoadSize == i){  
                ConstuctGraph();
                PartitionGraphTopo();
                Build_Cache(); 
                Prepare_LocalData_v1();
                GPU_Initial_v1();
                Free_Host_Memory();            
            } 
            MPI_Barrier(MPI_COMM_WORLD);
        } 
        MPI_Barrier(MPI_COMM_WORLD); 
        //When it comes to shared memory allocation involving NVSHMEM, 
        //this function can only be executed after all processes are synchronized.
        Buff_Initial(); 
    }else{  
        ConstuctGraph();
        PartitionGraphTopo();
        Build_Cache();
        Prepare_LocalData();
        GPU_Initial();
        Free_Host_Memory();
    }
}

void Graph::ConstuctGraph(){

    const char* ptr_file    = config->ptr_file.c_str();
    const char* ptr_T_file  = config->ptr_T_file.c_str();
    const char* indice_file = config->indice_file.c_str();
    const char* ind_T_file  = config->ind_T_file.c_str(); 
    //const char* feat_file   = config->feat_file.c_str();
    const char* label_file  = config->label_file.c_str();

    // double time = wtime();
    FILE *file = NULL;
    file_index_t ret;

    vertices_num = fsize(ptr_file) / sizeof(file_index_t) - 1;
    edges_num = fsize(indice_file) / sizeof(file_index_t); 

    if(mynode == 0){
        std::cout << "Expected vertice: " <<  vertices_num << std::endl;
        std::cout << "Expected edges: " << edges_num << std::endl;
    }
    //Reading Ptr file
    file = fopen(ptr_file, "rb");
    if (file != NULL){
        if(posix_memalign((void **)&file_ptr, getpagesize(),
                        sizeof(file_index_t) * (vertices_num + 1)))  
            perror("posix_memalign");

        ret = fread(file_ptr, sizeof(file_index_t), vertices_num + 1, file);
        assert(ret == vertices_num + 1);
        fclose(file);
        LOG(INFO) << "Reading ptr_file...";              
    } else {
        LOG(ERROR) << "Ptr_file not found!";
    } 

    for (VertexID i = 0; i < vertices_num + 1; i++){
        full_csr_ptr.push_back(file_ptr[i]);
    }
    //delete[] file_ptr;

    //Reading Indice file
    file = fopen(indice_file, "rb");
    if (file != NULL){
        if(posix_memalign((void **)&file_ind, getpagesize(),
                        sizeof(file_index_t) * (edges_num)))
            perror("posix_mamalign");

        ret = fread(file_ind, sizeof(file_index_t), edges_num, file);
        assert(ret == edges_num);
        fclose(file);
        LOG(INFO) << "Reading indice_file...";
    } else {
        LOG(ERROR) << "Indice_file not found!";
    }

    for (VertexID i = 0; i < edges_num + 1; i++){
        full_csr_ind.push_back(file_ind[i]);
    }
    //delete[] file_ind;

    //Reading Feat file
    // file = fopen(feat_file, "rb");
    // // feature = new float[(config->in_dim * vertices_num)];
    // if (file != NULL){
    //     if(posix_memalign((void **)&feature, getpagesize(),
    //                     sizeof(file_feat_t) * (config->in_dim) * (vertices_num)))
    //     perror("posix_mamalign");

    //     ret = fread(feature, sizeof(file_feat_t), (config->in_dim) * (vertices_num), file);
    //     assert(ret == config->in_dim * vertices_num);
    //     fclose(file);
    //     // std::cout << "Feature[0]: " << feature[0] << std::endl;
    //     // std::cout << "Feature[1]: " << feature[1] << std::endl;
    //     // std::cout << "Feature[2]: " << feature[2] << std::endl;
    //     // std::cout << "Feature[2449028]: " << feature[2449028] << std::endl;
    //     LOG(INFO) << "Reading feat_file...";
    // } else {
    //     if (mynode == 0) LOG(ERROR) << "Feature file is missing, feature tensor is automatically created.";

    //     if(posix_memalign((void **)&feature, getpagesize(),
    //                     sizeof(file_feat_t) * (config->in_dim) * (vertices_num)))
    //     perror("posix_mamalign");
    //     for(size_t i = 0; i < (config->in_dim * vertices_num); i++){
    //         feature[i] = 1;
    //     }
    // }

    //Reading Label file
    file = fopen(label_file, "rb");
    if (file != NULL){
        file_index_t *tmp_label = NULL;

        if(posix_memalign((void**)&tmp_label, getpagesize(),
                    sizeof(file_index_t) * vertices_num))
            perror("posix_memalign");

        ret = fread(tmp_label, sizeof(file_index_t), vertices_num, file);
        assert(ret == vertices_num);
        fclose(file);
        LOG(INFO) << "Reading label file...";

        if(posix_memalign((void**)&label, getpagesize(),
                    sizeof(int) *  vertices_num))
            perror("posix_memalign");

        for(int i = 0; i < int(vertices_num); ++i){
            label[i] = (int)tmp_label[i];
        }
        delete[] tmp_label;
        LOG(INFO) << "Converting label data type...";
    } else {
        if (mynode == 0) LOG(ERROR) << "Label_file is missing, labels are automatically created.";

        if(posix_memalign((void**)&label, getpagesize(),
                sizeof(int) *  vertices_num))
            perror("posix_memalign");

        int out_dim = config->out_dim;
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<int> dis(0, out_dim - 1);

        for(int i = 0; i < int(vertices_num); ++i){
            label[i] = dis(gen);
        }
    }

    if(TrainFlag == 1){
        //Reading the transposed ptr file
        file_index_t *tmp_ptr_T;
        file = fopen(ptr_T_file, "rb");
        if(file != NULL){
        
            if(posix_memalign((void**)&tmp_ptr_T, getpagesize(),
                            sizeof(file_index_t) * (vertices_num + 1)))
            perror("posix_megalign");

            ret = fread(tmp_ptr_T, sizeof(file_index_t), vertices_num + 1, file);
            assert(ret == vertices_num + 1);
            fclose(file);
            LOG(INFO) << "Reading ptr file of transposed adjacency matrix...";
        } else {
            LOG(ERROR) << "Transposed ptr_file not found!";
        }

        for(VertexID i = 0; i < vertices_num + 1; i++){
            full_csr_T_ptr.push_back(tmp_ptr_T[i]);
        }
        free(tmp_ptr_T);

        //Reading the transposed ind file
        file_index_t *tmp_ind_T;
        file = fopen(ind_T_file, "rb");
        if(file != NULL){

            if(posix_memalign((void**)&tmp_ind_T, getpagesize(),
                            sizeof(file_index_t) * (edges_num)))
            perror("posix_megalign");

            ret = fread(tmp_ind_T, sizeof(file_index_t), edges_num, file);
            assert(ret == edges_num);
            fclose(file);
            LOG(INFO) << "Reading ind file of transposed adjacency matrix...";
        } else {
            LOG(ERROR) << "Transposed ind_file not found!";
        }

        for(VertexID i = 0; i < edges_num; i++){
            full_csr_T_ind.push_back(tmp_ind_T[i]);
        }
        free(tmp_ind_T);       
    }
   
}

void Graph::PartitionGraphTopo(){
    if(mynode == 0) LOG(ERROR) << "Starting partitioning the graph topo...";
    Partition* partitioner = nullptr;
    LOG(INFO) << "Cache policy: " << RunConfig::cache_policy;
    LOG(INFO) << "Partition policy: " << RunConfig::partition_policy;
    switch (RunConfig::partition_policy){
    case kLEONodePartition:{
        partitioner = new NodePartition(full_csr_ptr, full_csr_ind, vertices_num, edges_num, mynode, nranks);
        break;
    } 
    default:
        CHECK(false) << "Invalid partition policy! ";   
    }
    if(mynode == 0) LOG(ERROR) << "Partitioner Built. Now solve";
    partitioner->Solve();
    this->ID_LowerBound  = partitioner->ID_LowerBound;
    this->ID_UpperBound  = partitioner->ID_UpperBound;
    this->ID_split_array = partitioner->ID_split_array;
}

void Graph::Build_Cache(){

if(CacheFlag == 1){
    freqreorder = new FreqReorder(file_ptr, file_ind, vertices_num, edges_num, nranks, mynode);
    cachemanager = new Cache(vertices_num, edges_num, mynode, nranks, ID_LowerBound, ID_UpperBound);

    LOG(ERROR) << "Find lower bound of nodes: " << ID_LowerBound << ", Find upper bound of nodes: " << ID_UpperBound;

    freqreorder->GetGlobalFreq(); //Global freq
    freqreorder->FreqSort<VertexID>(freqreorder->sort_ID2old, freqreorder->global_freq_list); //Global frequency descending order    
    freqreorder->GetMyFreq(ID_LowerBound, ID_UpperBound); //Local freq
    freqreorder->FreqSort<VertexID>(freqreorder->my_sort_ID2old, freqreorder->my_freq_list);
    cachemanager->build(freqreorder->sort_ID2old, freqreorder->my_sort_ID2old, config->in_dim, RunConfig::cache_percentage, RunConfig::h_cache_percentage, HostCacheFlag, BatchloadFlag);
    this->cache_IDlist = cachemanager->cache_IDlist;
    cache_num = cache_IDlist.size(); 
    // if(posix_memalign((void **)&this->cache_feature, getpagesize(),
    //                     sizeof(cache_feat_t) * (config->in_dim) * cache_num))
    // perror("posix_mamalign");
    // memcpy(this->cache_feature, cachemanager->cache_feature, cache_num * (config->in_dim) * sizeof(cache_feat_t));

    if(HostCacheFlag == 1){
        this->cache_host_IDlist = cachemanager->cache_host_IDlist;
        h_cache_num = cache_host_IDlist.size();
        // if(posix_memalign((void **)&this->cache_host_feature, getpagesize(),
        //                     sizeof(cache_feat_t) * (config->in_dim) * h_cache_num))
        // perror("posix_mamalign");
        // memcpy(this->cache_host_feature, cachemanager->cache_host_feature, h_cache_num * (config->in_dim) * sizeof(cache_feat_t));
    }

    //clean up
    delete freqreorder;
    delete cachemanager;

    freqreorder = nullptr;
    cachemanager = nullptr;    
} // CacheFlag

} //Build_Cache

void Graph::Create_CSR(){
    if(mynode == 0){
        printf("---------------------------------------------------------------------------------------------------\n");
        printf("Report the number of edges included in the respective CSR\n");
        printf("| Rank |    Local     |  Cache(GPU)  | Cache(host)  |    Remote    | Local(Trans) | Remote(Trans)|\n");
    }

    if(CacheFlag == 1){

        if(PtrConcatFlag == 1){

            if(RemoteFlag == 1){

                if(UVMFlag == 1){                    
                    LOG(FATAL) << "To be completed";
                }else{
                    Create_CSR_Combined_w_CR_wo_U();
                }

            }else{ 
                LOG(FATAL) << "Unsupported CSR creation scenarios!\n" << "CacheFlag: " << CacheFlag << " ConcatFlag: " \
                << PtrConcatFlag << " RemoteFlag: " << RemoteFlag << " UVMFlag: " << UVMFlag;
                //CacheFlag = 1; RemoteFlag = 0; ConcatFlag = 1; 
            }

        }else{

            if(RemoteFlag == 1){

                if(UVMFlag == 1){

                    Create_CSR_w_CRU();  //CacheFlag = 1; RemoteFlag = 1; UVMFlag = 1; ConcatFlag = 0;
                }else{

                    Create_CSR_w_CR_wo_U();  //CacheFlag = 1; RemoteFlag = 1; UVMFlag = 0; ConcatFlag = 0;      
                }

            }else{ //RemoteFlag
                LOG(FATAL) << "Unsupported CSR creation scenarios!\n" << "CacheFlag: " << CacheFlag << " ConcatFlag: " \
                << PtrConcatFlag << " RemoteFlag: " << RemoteFlag << " UVMFlag: " << UVMFlag;
                //CacheFlag = 1; RemoteFlag = 0; ConcatFlag = 0;
            }
        }
    }else{ //CacheFlag
        if(RemoteFlag == 1){

            if(UVMFlag == 1){

                LOG(FATAL) << "Unsupported CSR creation scenarios!\n" << "CacheFlag: " << CacheFlag << " ConcatFlag: " \
                << PtrConcatFlag << " RemoteFlag: " << RemoteFlag << " UVMFlag: " << UVMFlag; 
                //CacheFlag = 0; RemoteFlag = 1; UVMFlag = 1;

            }else{ //UVMFlag

                Create_CSR_w_R_wo_CU(); //CacheFlag = 0; RemoteFlag = 1; UVMFlag = 0;
            }
        }else{ //RemoteFlag
           LOG(FATAL) << "Unsupported CSR creation scenarios!\n" << "CacheFlag: " << CacheFlag << " ConcatFlag: " \
           << PtrConcatFlag << " RemoteFlag: " << RemoteFlag << " UVMFlag: " << UVMFlag; 
           //CacheFlag = 0; RemoteFlag = 0; 
        }
    }

    if(TrainFlag == 1){
        Create_CSR_transpose();
    }else{
        const int string_width = 14; 
        std::cout << CenterStr(std::to_string(0), string_width) << "|"; 
        std::cout << CenterStr(std::to_string(0), string_width) << "|" << std::endl;
    }

} // Create_CSR


void Graph::Create_CSR_w_R_wo_CU(){

    local_csr_ptr   = {0};
    remote_csr_ptr  = {0};

    for(VertexID i = ID_LowerBound; i < ID_UpperBound; i++){
        for(CSR_t j = full_csr_ptr[i]; j < full_csr_ptr[i + 1]; j++){
            CSR_t nid = full_csr_ind[j];

            if (ID_LowerBound <= nid && nid < ID_UpperBound){
                local_csr_ind.push_back(nid);
            }else{
                remote_csr_ind.push_back(nid);
            }
        }
        local_csr_ptr.push_back(local_csr_ind.size());
        remote_csr_ptr.push_back(remote_csr_ind.size());
    }

    const int string_width = 14;  
    std::cout << "|  " << mynode << "   |" << CenterStr(std::to_string(int(local_csr_ind.size())), string_width) << "|";
    std::cout << CenterStr(std::to_string(0), string_width) << "|"; 
    std::cout << CenterStr(std::to_string(0), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(remote_csr_ind.size())), string_width) << "|";   

    if (remote_csr_ind.size() == 0) RemoteFlag = 0; 
#ifdef debug 
    std::cout << "Rank " << this->mynode << ": RemoteFlag = " << RemoteFlag << std::endl;            
#endif 
}

void Graph::Create_CSR_w_CR_wo_U(){

    local_csr_ptr   = {0};
    cache_csr_ptr   = {0};
    remote_csr_ptr  = {0};

    for(VertexID i = ID_LowerBound; i < ID_UpperBound; i++){
        for(CSR_t j = full_csr_ptr[i]; j < full_csr_ptr[i + 1]; j++){
            CSR_t nid = full_csr_ind[j];
            
            if (ID_LowerBound <= nid && nid < ID_UpperBound){
                local_csr_ind.push_back(nid);
            }else if (std::binary_search(cache_IDlist.begin(), cache_IDlist.end(), nid)){
                cache_csr_ind.push_back(nid);
            }else{
                remote_csr_ind.push_back(nid);
            }
        }
        local_csr_ptr.push_back(local_csr_ind.size());
        cache_csr_ptr.push_back(cache_csr_ind.size());
        remote_csr_ptr.push_back(remote_csr_ind.size());
    }

    const int string_width = 14; 
    std::cout << "|  " << mynode << "   |" << CenterStr(std::to_string(int(local_csr_ind.size())), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(cache_csr_ind.size())), string_width) << "|"; 
    std::cout << CenterStr(std::to_string(0), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(remote_csr_ind.size())), string_width) << "|"; 

    if (cache_csr_ind.size() == 0) CacheFlag = 0;
    if (remote_csr_ind.size() == 0) RemoteFlag = 0;

#ifdef debug
if(RemoteFlag != 0){
    auto max_iter = std::max_element(remote_csr_ind.begin(), remote_csr_ind.end());
    std::cout << "Rank " << this->mynode << ": max element of remote indice = " << *max_iter << std::endl;
    auto min_iter = std::min_element(remote_csr_ind.begin(), remote_csr_ind.end());
    std::cout << "Rank " << this->mynode << ": min element of remote indice = " << *min_iter << std::endl;
}
#endif    
} //Create_CSR_w_CR_wo_U

void Graph::Create_CSR_w_CRU(){
    local_csr_ptr   = {0};
    cache_csr_ptr   = {0};
    remote_csr_ptr  = {0};
    host_csr_ptr    = {0};

    for(VertexID i = ID_LowerBound; i < ID_UpperBound; i++){
        for(CSR_t j = full_csr_ptr[i]; j < full_csr_ptr[i + 1]; j++){
            CSR_t nid = full_csr_ind[j];

            if (ID_LowerBound <= nid && nid < ID_UpperBound){
                local_csr_ind.push_back(nid);
            }else if(std::binary_search(cache_IDlist.begin(), cache_IDlist.end(), nid)){
                cache_csr_ind.push_back(nid);
            }else if(std::binary_search(cache_host_IDlist.begin(), cache_host_IDlist.end(), nid)){
                host_csr_ind.push_back(nid);
            }else{
                remote_csr_ind.push_back(nid);
            }           
        }
        local_csr_ptr.push_back(local_csr_ind.size());
        cache_csr_ptr.push_back(cache_csr_ind.size());
        remote_csr_ptr.push_back(remote_csr_ind.size());
        host_csr_ptr.push_back(host_csr_ind.size());
    } 

    const int string_width = 14; 
    std::cout << "|  " << mynode << "   |" << CenterStr(std::to_string(int(local_csr_ind.size())), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(cache_csr_ind.size())), string_width) << "|"; 
    std::cout << CenterStr(std::to_string(int(host_csr_ind.size())), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(remote_csr_ind.size())), string_width) << "|";

    if (cache_csr_ind.size() == 0) CacheFlag = 0;
    if (remote_csr_ind.size() == 0) RemoteFlag = 0;
    if (host_csr_ind.size() == 0) UVMFlag = 0;      
} //Create_CSR_w_CRU


void Graph::Create_CSR_Combined_w_CR_wo_U(){

    local_csr_ptr   = {0};
    remote_csr_ptr  = {0};

    for(VertexID i = ID_LowerBound; i < ID_UpperBound; i++){
        for(CSR_t j = full_csr_ptr[i]; j < full_csr_ptr[i + 1]; j++){
            CSR_t nid = full_csr_ind[j];

            if(ID_LowerBound <= nid && nid < ID_UpperBound){
                local_csr_ind.push_back(nid);
            }else if(std::binary_search(cache_IDlist.begin(), cache_IDlist.end(), nid)){
                local_csr_ind.push_back(nid);
            }else{
                remote_csr_ind.push_back(nid);
            }
        }
        local_csr_ptr.push_back(local_csr_ind.size());
        remote_csr_ptr.push_back(remote_csr_ind.size());
    }   

    const int string_width = 14; 
    std::cout << "|  " << mynode << "   |" << CenterStr(std::to_string(int(local_csr_ind.size())), string_width) << "|";
    std::cout << CenterStr(std::to_string(0), string_width) << "|"; 
    std::cout << CenterStr(std::to_string(0), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(remote_csr_ind.size())), string_width) << "|"; 

    CacheFlag = 0;
    if(remote_csr_ind.size() == 0) RemoteFlag = 0;   
} //Create_CSR_Combined_w_CR_wo_U


void Graph::Create_CSR_transpose(){
    local_csr_T_ptr = {0};
    remote_csr_T_ptr = {0};

    for(VertexID i = ID_LowerBound; i < ID_UpperBound; i++){
        for(CSR_t j = full_csr_T_ptr[i]; j < full_csr_T_ptr[i+1]; j++){
            CSR_t nid = full_csr_T_ind[j];

            if (ID_LowerBound <= nid && nid < ID_UpperBound){
                local_csr_T_ind.push_back(nid);
            }else{
                remote_csr_T_ind.push_back(nid);
            }
        }
        local_csr_T_ptr.push_back(local_csr_T_ind.size());
        remote_csr_T_ptr.push_back(remote_csr_T_ind.size());
    }

    const int string_width = 14; 
    std::cout << CenterStr(std::to_string(int(local_csr_T_ind.size())), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(remote_csr_T_ind.size())), string_width) << "|" << std::endl;

    if (remote_csr_T_ind.size() == 0) TransRemoteFlag = 0;
    //std::cout << "Rank " << this->mynode << ": TransRemoteFlag = " << TransRemoteFlag << std::endl;
}


void Graph::Create_IDmap(){
    //WARNING: Local nodes and cached nodes do not overlap and share the same ID map.
    local_IDmap.resize(vertices_num);
  
    //local partition part
    int tmp_count = 0;
    for(VertexID id = ID_LowerBound; id < ID_UpperBound; id++){
        local_IDmap[id] = tmp_count;
        tmp_count += 1; 
    }

if(CacheFlag == 1){
    //cache_IDlist part
    tmp_count = 0;
    for(auto &element : cache_IDlist){
        local_IDmap[element] = tmp_count;
        tmp_count += 1;
    }

    if(HostCacheFlag == 1){
        tmp_count = 0;
        for(auto &element : cache_host_IDlist){
            local_IDmap[element] = tmp_count;
            tmp_count += 1;
        }
    }
}

} // Create_IDmap

void Graph::Create_Weight(){
    weight = new Weight(config->in_dim, config->hidden_dim, config->out_dim, config->nlayers);
    weight->Initial();
    //weight->Xavier_Initial();
}

void Graph::Combine_local_feature(){
    //Merge local features and cached fearures
    auto num_cache = cache_IDlist.size();
    VertexID local_num_nodes = ID_UpperBound - ID_LowerBound; 
    if(posix_memalign((void **)&this->combined_local_feature, getpagesize(),
                        sizeof(cache_feat_t) * (config->in_dim) * (num_cache + local_num_nodes)))
    perror("posix_mamalign");
    std::cout << "Rank: " << mynode << ": Create Local feature: (" << num_cache + local_num_nodes << ")" << std::endl;

    //copy local features
    memcpy(this->combined_local_feature, feature + (ID_LowerBound * config->in_dim),
         (local_num_nodes * config->in_dim) * sizeof(cache_feat_t));

    //copy cached features
    memcpy(this->combined_local_feature + (local_num_nodes * config->in_dim), this->cache_feature,
            (num_cache * config->in_dim) * sizeof(cache_feat_t));
}

void Graph::Extract_Featrue(){
    const char* feat_file = config->feat_file.c_str();
    VertexID local_v_num = ID_UpperBound - ID_LowerBound;
    long offset = ID_LowerBound * (config->in_dim) * sizeof(file_feat_t);
    FILE *file = NULL;
    file_index_t ret;
    //file_index_t vaild;
    //file_feat_t* vaild_feature = NULL;

    //Loading local featrue
    file = fopen(feat_file, "rb");
    if(file != NULL){
        
        // if(posix_memalign((void **)&vaild_feature, getpagesize(),
        //                 sizeof(file_feat_t) * (config->in_dim) * (vertices_num)))
        // perror("posix_mamalign");  
        // vaild = fread(vaild_feature, sizeof(file_feat_t), (config->in_dim) * (vertices_num), file);
        // assert(vaild == config->in_dim * vertices_num);     

        if(fseek(file, offset, SEEK_SET) != 0){
            LOG(FATAL) << "Unable to set file pointer position!";
        }

        if(posix_memalign((void **)&feature, getpagesize(),
                        sizeof(file_feat_t) * (config->in_dim) * (local_v_num)))
        perror("posix_mamalign");
        
        ret = fread(feature, sizeof(file_feat_t), (config->in_dim) * (local_v_num), file);
        assert(ret == config->in_dim * local_v_num);

        fclose(file);
        LOG(INFO) << "Loading local feature....";

        
        if(CacheFlag == 1){
            std::ifstream sfile(feat_file, std::ios::binary);
            cached_feature.clear();

            for (const auto element : cache_IDlist){
                sfile.seekg(element * config->in_dim * sizeof(file_feat_t), std::ios::beg);
                std::vector<file_feat_t> chunk(config->in_dim);
                sfile.read(reinterpret_cast<char*>(chunk.data()), config->in_dim * sizeof(file_feat_t));
                cached_feature.insert(cached_feature.end(), chunk.begin(), chunk.end());
            }
            LOG(INFO) << "Loading cached feature(GPU side)....";
            sfile.close();

            if(HostCacheFlag == 1){
                std::ifstream sfile(feat_file, std::ios::binary);
                tmp_host_feature.clear();

                for (const auto element : cache_host_IDlist){
                    sfile.seekg(element * config->in_dim * sizeof(file_feat_t), std::ios::beg);
                    std::vector<file_feat_t> chunk(config->in_dim);
                    sfile.read(reinterpret_cast<char*>(chunk.data()), config->in_dim * sizeof(file_feat_t));
                    tmp_host_feature.insert(tmp_host_feature.end(), chunk.begin(), chunk.end());
                }
                LOG(INFO) << "Loading cached feature(host side)....";
                sfile.close();

                if(posix_memalign((void **)&this->cache_host_feature, getpagesize(),
                    sizeof(cache_feat_t) * (config->in_dim) * h_cache_num))
                    perror("posix_mamalign");

                std::copy(tmp_host_feature.begin(), tmp_host_feature.end(), cache_host_feature);
                tmp_host_feature.clear();
            }
        }

    }else{
        if (mynode == 0) LOG(ERROR) << "Feature file is missing, feature tensor is automatically created.";
        
        if(posix_memalign((void **)&feature, getpagesize(),
                        sizeof(file_feat_t) * (config->in_dim) * (local_v_num)))
        perror("posix_mamalign");
        for(size_t i = 0; i < (config->in_dim * local_v_num); i++){
            feature[i] = 1;
        }

        if(CacheFlag == 1){
            auto cached_v_num = cache_IDlist.size();
            cached_feature.resize(cached_v_num * config->in_dim, 1);

            if(HostCacheFlag == 1){
                if(posix_memalign((void **)&this->cache_host_feature, getpagesize(),
                    sizeof(cache_feat_t) * (config->in_dim) * h_cache_num))
                    perror("posix_mamalign");

                for(int i = 0; i < config->in_dim * h_cache_num; i++){
                    cache_host_feature[i] = 1;
                }
            }
        }
    }
}

void Graph::Prepare_LocalData(){
    Timer t0;
    Create_CSR();
    double csr_time = t0.Passed();
    
    Timer t1;
    Create_IDmap();
    double idmap_time = t1.Passed();

    Timer t2;
    Extract_Featrue();
    double extract_feat_time = t2.Passed();

    MPI_Barrier(MPI_COMM_WORLD);
    std::cout << "Rank " << mynode << ": Create CSR =====> " << csr_time << "s" << std::endl;
    std::cout << "Rank " << mynode << ": Create ID map =====> " << idmap_time << "s" << std::endl;
    std::cout << "Rank " << mynode << ": Extract Featrue ======>" << extract_feat_time << "s" << std::endl;

    // Timer t2;
    // Combine_local_feature();
    // double combine_time = t2.Passed();
    // std::cout << "Rank: " << mynode << ": Combine local feature =====> " << combine_time << "s" << std::endl;  
}

void Graph::Prepare_LocalData_v1(){
    Timer t0;
    Create_CSR();
    double csr_time = t0.Passed();
    
    Timer t1;
    Create_IDmap();
    double idmap_time = t1.Passed();

    Timer t2;
    Extract_Featrue();
    double extract_feat_time = t2.Passed();

    std::cout << "Rank " << mynode << ": Create CSR =====> " << csr_time << "s" << std::endl;
    std::cout << "Rank " << mynode << ": Create ID map =====> " << idmap_time << "s" << std::endl;
    std::cout << "Rank " << mynode << ": Extract Featrue ======>" << extract_feat_time << "s" << std::endl;        
} 

void Graph::Free_Host_Memory(){

    //pointer
    free(feature);
    free(label);
    free(file_ptr);
    free(file_ind);

    feature = NULL;
    label = NULL;
    file_ptr = NULL;
    file_ind = NULL;

    //vector
    cached_feature.clear();

    full_csr_ptr.clear();
    full_csr_ind.clear();
    full_csr_T_ptr.clear();
    full_csr_T_ind.clear();

    ID_split_array.clear();
    cache_IDlist.clear();
    cache_host_IDlist.clear();

    local_csr_ptr.clear();
    local_csr_ind.clear();
    cache_csr_ptr.clear();
    cache_csr_ind.clear();
    remote_csr_ptr.clear();
    remote_csr_ind.clear();
    host_csr_ptr.clear();
    host_csr_ind.clear();
    local_IDmap.clear();

    local_csr_T_ptr.clear();
    local_csr_T_ind.clear();
    remote_csr_T_ptr.clear();
    remote_csr_T_ind.clear();

    std::cout << "Rank " << mynode << ": Free memory (host side)..." << std::endl;
}

Graph::~Graph(){

    delete config;
    delete weight;
    
    config = nullptr;
    weight = nullptr;
}

}   //namespace common
}   //namescpce GNNPro_lib  

