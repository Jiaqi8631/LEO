#include "partition.h"

namespace GNNPro_lib{
namespace common{


void NodePartition::Solve(){
    ID_LowerBound = mynode * ((vertices_num - 1)/ num_device + 1);
    ID_UpperBound = std::min((mynode + 1) * ((vertices_num - 1) / num_device + 1), vertices_num);

    for(int i = 0; i < num_device; i++){
        ID_split_array.push_back(i * ((vertices_num - 1) / num_device + 1));
    }
    ID_split_array.push_back(vertices_num);
}

}
}