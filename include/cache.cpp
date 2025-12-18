#include "cache.h"
#include "logging.h"
#include "cache_solver.h"
#include "run_config.h"

namespace GNNPro_lib{
namespace common{

Cache::Cache(VertexID vertices_num, EdgeID edges_num, int mynode, int num_device, VertexID LB, VertexID UB) 
:vertices_num(vertices_num), edges_num(edges_num), mynode(mynode), num_device(num_device){
    this->ID_LowerBound = LB;
    this->ID_UpperBound = UB;
}

void Cache::build(std::vector<VertexID> global_freq_rank, std::vector<VertexID> freq_rank, int dim, double cache_percentage, double h_cache_percentage, bool host_cache_flag, bool batch_load_flag){
    this->global_freq_rank = global_freq_rank;
    this->freq_rank = freq_rank;
    this->cache_percentage = cache_percentage;
    this->h_cache_percentage = h_cache_percentage;
    this->host_cache_flag = host_cache_flag;
    this->batch_load_flag = batch_load_flag;

    Solve_impl();
    if(batch_load_flag == true){
        Report_Solve_Result_v1();
    }else{
        Report_Solve_Result();
    }
    std::sort(cache_IDlist.begin(), cache_IDlist.end());
    std::sort(cache_host_IDlist.begin(), cache_host_IDlist.end());
    // if(posix_memalign((void **)&cache_feature, getpagesize(),
    //                     sizeof(cache_feat_t) * (dim) * (cache_IDlist.size())))
    // perror("posix_mamalign");

    // if(posix_memalign((void **)&cache_host_feature, getpagesize(),
    //                     sizeof(cache_feat_t) * (dim) * (cache_host_IDlist.size())))
    // perror("posix_mamalign");
    

    //Extract_Host_Feature(dim);
}

void Cache::Solve_impl(){
    if(mynode == 0) LOG(ERROR) << "creating solver";
    cache::CacheSolver *solver = nullptr;
    switch (RunConfig::cache_policy){
    case kLEOCachePartition: {
        solver = new cache::PartitionSolver();
        break;
    }
    case kLEOCacheReplication: {
        solver = new cache::ReplicationSolver();
        break;
    }
    default:
        CHECK(false) << "Invalid cache policy! ";
    }

    if(mynode == 0) LOG(ERROR) << "Solver Created. Now build & solve";
    solver->Build(global_freq_rank, freq_rank, vertices_num, cache_percentage, h_cache_percentage, mynode, num_device, host_cache_flag);
    if(mynode == 0) LOG(ERROR) << "Solver Built. Now solve";
    solver->Solve();
    if(mynode == 0) LOG(ERROR) << "Solve done!"; 
    //solver->CheckBoundary<VertexID>(solver->solve_list, 0, int(vertices_num - 1));
    this->solve_list = solver->solve_list;
    this->h_solve_list = solver->h_solve_list;
}

void Cache::Report_Solve_Result(){
    double cache_cnt = 0, local_cnt = 0, remote_cnt = 0, host_cnt = 0;
    CHECK(ID_UpperBound <= VertexID(vertices_num)) << "Exceed the Upper Bound!";
    for(auto iter = solve_list.begin(); iter != solve_list.end(); iter++){
        if((*iter) < ID_LowerBound || (*iter) >= ID_UpperBound ){
            cache_cnt += 1;
            cache_IDlist.push_back((*iter));
        }
    } 

    if(host_cache_flag == true){
        for(auto iter = h_solve_list.begin(); iter != h_solve_list.end(); iter++){
            if((*iter) < ID_LowerBound || (*iter) >= ID_UpperBound){
                host_cnt += 1;
                cache_host_IDlist.push_back((*iter));
            }
        }
    } 

    cache_node_count = cache_cnt;
    local_cnt = ID_UpperBound - ID_LowerBound;
    remote_cnt = double(vertices_num - local_cnt - cache_cnt - host_cnt);
    std::cout << "Rank " << mynode << " | LowerBound " << ID_LowerBound << " | UpperBound " << ID_UpperBound << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    if(mynode == 0){
        std::cout << "GNNPro:cache persentage = " << cache_percentage << std::endl;
        printf("Report actual node distribution.\n");
        printf("| Rank |    Local     |  Cache(GPU)  | Cache(host)  |    Remote    |\n");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const int string_width = 14; //Used to determine the maximum width of the column
    std::cout << "|  " << mynode << "   |" << CenterStr(std::to_string(int(local_cnt)), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(cache_cnt)), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(host_cnt)), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(remote_cnt)), string_width) << "|" << std::endl;

    const double local_rate = local_cnt * 100 / vertices_num;
    const double cache_rate = cache_cnt * 100 / vertices_num;
    const double host_rate = host_cnt * 100 / vertices_num;
    const double remote_rate = remote_cnt * 100 / vertices_num;
    std::cout << "|  " << mynode << "   |" << CenterStr(std::to_string(int(local_rate))+"%", string_width) << "|";
    std::cout << CenterStr(std::to_string(int(cache_rate))+"%", string_width) << "|";
    std::cout << CenterStr(std::to_string(int(host_rate))+"%", string_width) << "|";
    std::cout << CenterStr(std::to_string(int(remote_rate))+"%", string_width) << "|" << std::endl;
}

void Cache::Report_Solve_Result_v1(){
    double cache_cnt = 0, local_cnt = 0, remote_cnt = 0, host_cnt = 0;
    CHECK(ID_UpperBound <= VertexID(vertices_num)) << "Exceed the Upper Bound!";
    for(auto iter = solve_list.begin(); iter != solve_list.end(); iter++){
        if((*iter) < ID_LowerBound || (*iter) >= ID_UpperBound ){
            cache_cnt += 1;
            cache_IDlist.push_back((*iter));
        }
    } 

    if(host_cache_flag == true){
        for(auto iter = h_solve_list.begin(); iter != h_solve_list.end(); iter++){
            if((*iter) < ID_LowerBound || (*iter) >= ID_UpperBound){
                host_cnt += 1;
                cache_host_IDlist.push_back((*iter));
            }
        }
    } 

    cache_node_count = cache_cnt;
    local_cnt = ID_UpperBound - ID_LowerBound;
    remote_cnt = double(vertices_num - local_cnt - cache_cnt - host_cnt);
    std::cout << "Rank " << mynode << " | LowerBound " << ID_LowerBound << " | UpperBound " << ID_UpperBound << std::endl;
    if(mynode == 0){
        std::cout << "GNNPro:cache persentage = " << cache_percentage << std::endl;
        printf("Report actual node distribution.\n");
        printf("| Rank |    Local     |  Cache(GPU)  | Cache(host)  |    Remote    |\n");
    }

    const int string_width = 14; //Used to determine the maximum width of the column
    std::cout << "|  " << mynode << "   |" << CenterStr(std::to_string(int(local_cnt)), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(cache_cnt)), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(host_cnt)), string_width) << "|";
    std::cout << CenterStr(std::to_string(int(remote_cnt)), string_width) << "|" << std::endl;

    const double local_rate = local_cnt * 100 / vertices_num;
    const double cache_rate = cache_cnt * 100 / vertices_num;
    const double host_rate = host_cnt * 100 / vertices_num;
    const double remote_rate = remote_cnt * 100 / vertices_num;
    std::cout << "|  " << mynode << "   |" << CenterStr(std::to_string(int(local_rate))+"%", string_width) << "|";
    std::cout << CenterStr(std::to_string(int(cache_rate))+"%", string_width) << "|";
    std::cout << CenterStr(std::to_string(int(host_rate))+"%", string_width) << "|";
    std::cout << CenterStr(std::to_string(int(remote_rate))+"%", string_width) << "|" << std::endl;
}

void Cache::Extract_Host_Feature(int dim){ //The datatype of the features is float by default
    //For data that will be cached in the GPU
    int index_count = 0;
    int cache_ID;
    for(auto iter = cache_IDlist.begin(); iter != cache_IDlist.end(); iter++){
        cache_ID = int(*iter);
        //if(mynode == 1) std::cout << "Rank: " << mynode << " cache_ID: " << cache_ID << " index_count: " << index_count << std::endl;
        std::memcpy(cache_feature + index_count, host_feature + cache_ID * dim, dim * sizeof(cache_feat_t));
        index_count += dim;
    }

    //For data that will be cached in the CPU
    if(host_cache_flag == 1){
        index_count = 0;
        for(auto iter = cache_host_IDlist.begin(); iter != cache_host_IDlist.end(); iter++){
            cache_ID = int(*iter);
            std::memcpy(cache_host_feature + index_count, host_feature + cache_ID * dim, dim * sizeof(cache_feat_t));
            index_count += dim;
        }
    }
}

Cache::~Cache(){}

}   //namespace common
}   //namescpce GNNPro_lib 