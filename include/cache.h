#include <vector>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include <mpi.h>
#include "type.h"

namespace GNNPro_lib{
namespace common{

class Cache{
public:
    file_feat_t             *host_feature;
    cache_feat_t            *cache_feature;
    cache_feat_t            *cache_host_feature;
    std::vector<VertexID>    global_freq_rank;
    std::vector<VertexID>    freq_rank;
    std::vector<VertexID>    solve_list;
    std::vector<VertexID>    h_solve_list;
    std::vector<VertexID>    cache_IDlist;
    std::vector<VertexID>    cache_host_IDlist;
    VertexID                 vertices_num;
    EdgeID                   edges_num;
    VertexID                 ID_LowerBound;
    VertexID                 ID_UpperBound;

    double                   h_cache_percentage;
    double                   cache_percentage;
    double                   cache_node_count;

    int                      mynode;
    int                      num_device;

    bool                     host_cache_flag = 0;
    bool                     batch_load_flag = 0;

    void build(std::vector<VertexID> global_freq_rank, std::vector<VertexID> freq_rank, int dim, double cache_percentage, double h_cache_percentage, bool host_cache_flag, bool batch_load_flag);
    void Solve_impl();
    void Report_Solve_Result();
    void Report_Solve_Result_v1(); //This function is uesd in batch loading scenarios
    void Extract_Host_Feature(int dim);
    Cache(VertexID vertices_num, EdgeID edges_num, int mynode, int num_device, VertexID LB, VertexID UB);
    ~Cache();
};

}   //namespace common
}   //namescpce GNNPro_lib 