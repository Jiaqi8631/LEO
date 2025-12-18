#include <iostream>
#include <stdio.h>
#include <string>
#include <fstream>
#include "common.h"

namespace GNNPro_lib{
namespace common{

class InputInfo {
public:

    std::string ptr_file;
    std::string ptr_T_file;
    std::string indice_file;
    std::string ind_T_file;
    std::string feat_file;
    std::string label_file;
    
    std::string algorithm;
    std::string dataset;
    std::string place_strategy; // for data palcement

    int nlayers;
    int in_dim;
    int hidden_dim;
    int out_dim;

    bool BatchLoad = false;
    int  BatchLoadSize = 0;

    void ReadFromConfig(std::string config_file);
    void GetFileName();
};

struct RunConfig
{
    static CachePolicy              cache_policy;
    static PartitionPolicy          partition_policy; 
    static double                   cache_percentage;
    static double                   h_cache_percentage; // host cache percentage
    static ModelType                model;
    static DataPlacementStrategy    placement_strategy;

    static bool                     h_cache_flag;           //whether to use cpu to cache features
    static bool                     train_flag;             //true for train, false for inference
    static bool                     ptr_concat_flag;        //true for concatnation

    static bool                     validate_cublas;
    static bool                     validate_aggregation;
    static bool                     validate_infer; 

    static int                      partsize;
    static int                      warpPerBlock; 
    static int                      overlap_dist; 
    static float                    eps;   
};

}   //namespace common
}   //namescpce GNNPro_lib  