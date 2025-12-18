#include"cache_solver.h"

namespace GNNPro_lib{
namespace common{
namespace cache{

void CacheSolver::Build(
    std::vector<VertexID> global_freq_rank, 
    std::vector<VertexID> freq_rank, 
    const VertexID num_node, 
    double cache_percentage,
    double h_cache_percentage, 
    int my_rank, 
    int num_rank,
    bool flag
){
    this->global_freq_rank  = global_freq_rank;
    this->freq_rank         = freq_rank;
    this->num_node          = num_node;
    this->cache_percent     = cache_percentage;
    this->h_cache_percent   = h_cache_percentage;
    this->my_rank           = my_rank;
    this->num_rank          = num_rank;
    this->h_cache_flag      = flag;
}

template <typename T>
void CacheSolver::CheckBoundary(std::vector<T> vector, VertexID LB, VertexID UB){
    T value = 0;
    std::cout << "Rank " << my_rank << ": vector size: " << vector.size() << std::endl;
    for(unsigned long int i = 0; i < vector.size(); i++){
        value = vector[i];
        if(value < LB || value > UB) std::cout << "Rank " << my_rank << ": [" << i << "] " << value << std::endl;
    }
}

void ReplicationSolver::Solve(){
    const size_t num_cached_nodes = size_t(num_node * (cache_percent / (double)100));
    if(my_rank == 0) LOG(ERROR) << "num_cached_nodes = " << num_cached_nodes;

    solve_list.resize(num_cached_nodes, 0);
    std::copy(freq_rank.begin(), freq_rank.begin() + num_cached_nodes, solve_list.begin());

    if(h_cache_flag == true){
        const size_t num_host_cached_nodes = size_t(num_node * (h_cache_percent / (double)100));
        CHECK((cache_percent + h_cache_percent) < 100) << "Cache nodes exceed total number!";

        h_solve_list.resize(num_host_cached_nodes, 0);
        std::copy(freq_rank.begin() + num_cached_nodes, freq_rank.begin() + num_cached_nodes + num_host_cached_nodes, h_solve_list.begin());
    }
}

void PartitionSolver::Solve(){
    const size_t num_cached_nodes = size_t(num_node * (cache_percent / (double)100));
    size_t max_cached_nodes = std::min(num_cached_nodes, size_t(num_node / num_rank));
    if(my_rank == 0) LOG(ERROR) << "num_cached_nodes = " << max_cached_nodes;

    for (size_t i = 0; i < max_cached_nodes; i++){
        solve_list.push_back(global_freq_rank[i * num_rank + my_rank]);
    }
    std::cout << "Rank " << my_rank << ": solve list size: " << solve_list.size() << std::endl;
}

}   //namespace cache
}   //namespace common
}   //namescpce GNNPro_lib 