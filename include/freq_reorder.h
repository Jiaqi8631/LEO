#include <vector>
#include <algorithm>
#include <iostream>
#include "type.h"

namespace GNNPro_lib{
namespace common{

class FreqReorder{
public:
    int num_partition;
    int my_partition;

    size_t ver_size;
    size_t edge_size;

    freq_t* global_freq_list;
    freq_t* other_freq_list;
    freq_t* my_freq_list;
    VertexID* global_ptr_buff;
    VertexID* global_graph_buff;
    VertexID* global_reorder_graph_buf;

    std::vector<VertexID> sort_ID2old;
    std::vector<VertexID> other_ID2old;
    std::vector<VertexID> my_sort_ID2old;

    FreqReorder(file_index_t* csr_ptr, file_index_t* csr_ind, VertexID vertices_num, EdgeID edges_num, int nranks, int mynode);
    ~FreqReorder();
    void GetGlobalFreq();
    void GetMyFreq(VertexID node_freqLB, VertexID node_freqUB);
    void GetOtherPartitionFreq();


    template <typename T>
    void FreqSort(std::vector<T> &FreqVec, T* freq_list){
        for(unsigned long int i = 0; i < FreqVec.size(); i++){
            FreqVec[i] = i;
        }
        //ascending order
        // std::sort(FreqVec.begin(), FreqVec.end(),
        //         [&](const int& a, const int& b){
        //             return (freq_list[a] < freq_list[b]);
        //         }
        // );

        //descending order
        std::sort(FreqVec.begin(), FreqVec.end(),
                [&](const int& a, const int& b){
                    return (freq_list[a] > freq_list[b]);
                }
        );
    }
};

}   //namespace common
}   //namescpce GNNPro_lib  