#include "run_config.h"
#include "logging.h"

namespace GNNPro_lib{
namespace common{

void InputInfo::ReadFromConfig(std::string config_file){
    std::string   cfg_oneline;
    std::ifstream inFile;
    inFile.open(config_file.c_str(), std::ios::in);
    if (inFile.is_open()){
        LOG(INFO) << "Getting configuration...";
    } else {
        LOG(ERROR) << "Configuration file not found!";
    }

    while (getline(inFile, cfg_oneline)){
        std::string cfg_l;
        std::string cfg_r;
        int dlim = cfg_oneline.find(':');
        cfg_l = cfg_oneline.substr(0, dlim);
        cfg_r = cfg_oneline.substr(dlim + 1, cfg_oneline.size() - dlim - 1);
        if (cfg_l.compare("ALGORITHM") == 0){
            this->algorithm = cfg_r;
        } else if (cfg_l.compare("DATASET") == 0){
            this->dataset = cfg_r;
        } else if (cfg_l.compare("INPUTDIM") == 0){
            this->in_dim = atoi(cfg_r.c_str());
        } else if (cfg_l.compare("HIDDENDIM") == 0){
            this->hidden_dim = atoi(cfg_r.c_str());
        } else if (cfg_l.compare("OUTPUTDIM") == 0){
            this->out_dim = atoi(cfg_r.c_str());
        } else if (cfg_l.compare("LAYERS") == 0){
            this->nlayers = atoi(cfg_r.c_str());
        } else if (cfg_l.compare("PATTERN") == 0){
            this->place_strategy = cfg_r;
        } else if (cfg_l.compare("BatchLoad") == 0){
            this->BatchLoadSize = atoi(cfg_r.c_str());
        } else {
            printf("Unsupported configuration!\n");
        } 
    }
    std::cout << "Read Configuration Done!" << std::endl;
    inFile.close();
}

void InputInfo::GetFileName(){
    std::string file_prefix    = "datasets/data-output/";
    std::string ptr_suffix     = "/indptr.bin";
    std::string ptr_T_suffix   = "/indptr_T.bin";
    std::string indices_suffix = "/indices.bin";
    std::string ind_T_suffix   = "/indices_T.bin";
    std::string feat_suffix    = "/feat.bin";
    std::string label_suffix   = "/label.bin";

    ptr_file    = file_prefix + dataset + ptr_suffix;
    ptr_T_file  = file_prefix + dataset + ptr_T_suffix;
    indice_file = file_prefix + dataset + indices_suffix;
    ind_T_file  = file_prefix + dataset + ind_T_suffix;
    feat_file   = file_prefix + dataset + feat_suffix;
    label_file  = file_prefix + dataset + label_suffix;
}

CachePolicy             RunConfig::cache_policy         = kLEOCacheReplication; 
PartitionPolicy         RunConfig::partition_policy     = kLEONodePartition; 
DataPlacementStrategy   RunConfig::placement_strategy   = kLocalRemoteCache;
double                  RunConfig::cache_percentage     = 25;
double                  RunConfig::h_cache_percentage   = 10;

bool                    RunConfig::h_cache_flag         = true;
bool                    RunConfig::train_flag           = false;
bool                    RunConfig::ptr_concat_flag      = false;

bool                    RunConfig::validate_cublas      = true;
bool                    RunConfig::validate_aggregation = true;
bool                    RunConfig::validate_infer       = true;

int                     RunConfig::partsize             = 4;
int                     RunConfig::warpPerBlock         = 4;
int                     RunConfig::overlap_dist         = 1;
float                   RunConfig::eps                  = 0.5;

}   //namespace common
}   //namescpce GNNPro_lib  