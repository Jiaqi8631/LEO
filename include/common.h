#ifndef common_h
#define common_h

#include <unordered_map>


#pragma once
namespace GNNPro_lib{
namespace common{

enum DeviceType { kCPU = 0, KGPU = 1, kUVM = 2};

enum DataPlacementStrategy{
    kLocalRemoteCache,
    kLocalRemote
};

enum CachePolicy {
    kLEOCachePartition,
    kLEOCacheReplication
};

enum PartitionPolicy {
    kLEONodePartition,
    kLEOEdgePartition
};

enum ModelType{
    GCN,
    GIN
};

ModelType StringToModel(const std::string& str);
DataPlacementStrategy StringToStrategy(const std::string& str);

std::string CenterStr(const std::string &str, int width);

}   //namespace common
}   //namescpce GNNPro_lib 

#endif