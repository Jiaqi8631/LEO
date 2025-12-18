#include <vector>
#include <iostream>
#include "type.h"
#include "logging.h"

namespace GNNPro_lib{
namespace common{

class Partition{
public:
    std::vector<CSR_t> ptr;
    std::vector<CSR_t> ind; 
    std::vector<CSR_t> ID_split_array;

    VertexID    ID_LowerBound;
    VertexID    ID_UpperBound;
    VertexID    vertices_num;
    EdgeID      edges_num;
    
    int         mynode;
    int         num_device;

    virtual void Solve() { CHECK(false) << "Unimplemented";}; 
    Partition(std::vector<CSR_t> file_ptr, std::vector<CSR_t> file_ind, VertexID vertices_num, EdgeID edges_num, int mynode, int num_device)\
    :ptr(file_ptr), ind(file_ind), vertices_num(vertices_num), edges_num(edges_num), mynode(mynode), num_device(num_device){};
    virtual ~Partition(){}
};

class NodePartition : public Partition{
public:
    NodePartition(std::vector<CSR_t> file_ptr, std::vector<CSR_t> file_ind, VertexID vertices_num, EdgeID edges_num, int mynode, int num_device)\
    :Partition(file_ptr, file_ind, vertices_num, edges_num, mynode, num_device){}
    void Solve();
    // virtual ~NodePartition(){}
};

}
}