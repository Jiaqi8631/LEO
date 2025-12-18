#include <algorithm>
#include <vector>
#include <iostream>
#include "logging.h"
#include "type.h"

namespace GNNPro_lib{
namespace common{
namespace cache{

class CacheSolver{
public:
    virtual ~CacheSolver() {}
    void Build(std::vector<VertexID> global_freq_rank, 
               std::vector<VertexID> freq_rank, 
               const VertexID num_node, 
               double cache_percentage, 
               double h_cache_percentage, 
               int my_rank, 
               int num_rank, 
               bool flag);

    template <typename T> void CheckBoundary(std::vector<T> vector, VertexID LB, VertexID UB);
    virtual void Solve() { CHECK(false) << "Unimplemented";}; 

    std::vector<VertexID>    global_freq_rank;
    std::vector<VertexID>    freq_rank;
    std::vector<VertexID>    solve_list;
    std::vector<VertexID>    h_solve_list;
    VertexID                 num_node; 
    double                   cache_percent; 
    double                   h_cache_percent;
    int                      my_rank;
    int                      num_rank;  
    bool                     h_cache_flag; 
};

class PartitionSolver : public CacheSolver{
public:
    void Solve();
};

class ReplicationSolver : public CacheSolver{
public:
    void Solve();
};

}   //namespace cache
}   //namespace common
}   //namescpce GNNPro_lib 