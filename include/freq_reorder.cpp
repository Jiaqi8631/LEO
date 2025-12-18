#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cstring>
#include "freq_reorder.h"
#include "logging.h"

namespace GNNPro_lib{
namespace common{

FreqReorder::FreqReorder(file_index_t* csr_ptr, file_index_t* csr_ind, VertexID vertices_num, EdgeID edges_num, int nranks, int mynode) : num_partition(nranks), my_partition(mynode){
    edge_size = size_t(edges_num);
    ver_size  = size_t(vertices_num);
    //std::cout << "Edge Size: " << edge_size << std::endl; 
    sort_ID2old.assign(ver_size, 0);
    my_sort_ID2old.assign(ver_size, 0);

    global_freq_list  = new freq_t[ver_size]();
    other_freq_list   = new freq_t[ver_size]();
    my_freq_list      = new freq_t[ver_size]();
    global_ptr_buff   = new file_index_t[ver_size];
    global_graph_buff = new file_index_t[edge_size];
    memcpy(global_ptr_buff, csr_ptr, (ver_size + 1) * sizeof(file_index_t)); //Default file_index_t == VertexID
    memcpy(global_graph_buff, csr_ind, edge_size * sizeof(file_index_t)); 
}

void FreqReorder::GetGlobalFreq(){
    int tmp_value = 0;
    for(size_t i = 0; i < edge_size; i++){
        tmp_value = int(global_graph_buff[i]);
        global_freq_list[tmp_value] += 1;
    }
}

void FreqReorder::GetMyFreq(VertexID node_freqLB, VertexID node_freqUB){ //0 < node_freqUB <= ver_size - 1
    int tmp_value = 0;
    size_t Ptr_Begin = global_ptr_buff[node_freqLB];
    size_t Ptr_End   = global_ptr_buff[node_freqUB];
    LOG(INFO) << "Ptr_Begin: " << Ptr_Begin << " Ptr_End: " << Ptr_End;
    if(Ptr_End <= 0 || Ptr_End > edge_size) LOG(ERROR) << "Index Access Error: Ptr_End" << Ptr_End; 
    for(size_t i = Ptr_Begin; i < Ptr_End; i++){
        tmp_value = int(global_graph_buff[i]);
        my_freq_list[tmp_value] += 1;
    }
}

void FreqReorder::GetOtherPartitionFreq(){
    int tmp_value = 0;
    if(my_partition != 0 && my_partition != num_partition-1){
        //first part
        size_t PtrIndex = my_partition * (ver_size / num_partition + 1); //fix me
        //LOG(ERROR) << my_partition << "->Phase 1 PtrIndex: " << global_ptr_buff[PtrIndex];
        for(size_t i = 0; i < global_ptr_buff[PtrIndex]; i++){
            tmp_value = int(global_graph_buff[i]);
            other_freq_list[tmp_value] += 1;
        }

        //second part
        PtrIndex = (my_partition + 1) * (ver_size / num_partition + 1);
        //LOG(ERROR) << my_partition << "->Phase 2 PtrIndex: " << global_ptr_buff[PtrIndex];
        for(size_t i = global_ptr_buff[PtrIndex]; i < edge_size; i++){
            tmp_value = int(global_graph_buff[i]);
            other_freq_list[tmp_value] += 1;
        }

    } else if (my_partition == 0){
        size_t PtrIndex = ver_size / num_partition + 1;
        //LOG(ERROR) << my_partition << "->PtrIndex: " << global_ptr_buff[PtrIndex];
        for(size_t i = global_ptr_buff[PtrIndex]; i < edge_size; i++){
            tmp_value = int(global_graph_buff[i]);
            other_freq_list[tmp_value] += 1;
        }
    } else {
        CHECK(my_partition == num_partition - 1);
        size_t PtrIndex = my_partition * (ver_size / num_partition + 1);
        //LOG(ERROR) << my_partition << "->PtrIndex: " << global_ptr_buff[PtrIndex];
        for(size_t i = 0; i < global_ptr_buff[PtrIndex]; i++){
            tmp_value = int(global_graph_buff[i]);
            other_freq_list[tmp_value] += 1;
        }        
    }   
}

FreqReorder::~FreqReorder(){
    delete global_freq_list;
    delete other_freq_list;
    delete my_freq_list;
    delete global_ptr_buff;
    delete global_graph_buff;

    global_freq_list  = NULL;
    other_freq_list   = NULL;
    my_freq_list      = NULL;
    global_ptr_buff   = NULL;
    global_graph_buff = NULL;
}

}   //namespace common
}   //namescpce GNNPro_lib  