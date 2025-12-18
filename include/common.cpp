#include "common.h"

namespace GNNPro_lib{
namespace common{

ModelType StringToModel(const std::string& str){
    static const std::unordered_map<std::string, ModelType> map = {
        {"GCN", ModelType::GCN},
        {"GIN", ModelType::GIN},
    };

    auto it = map.find(str);
    if(it != map.end()){
        return it->second;
    }else{
        throw std::runtime_error("Invaild ModelType string!");
    }
}

DataPlacementStrategy StringToStrategy(const std::string& str){
    static const std::unordered_map<std::string, DataPlacementStrategy> map = {
        {"CPU_GPU_Cache", DataPlacementStrategy::kLocalRemoteCache},
        {"CPU_GPU", DataPlacementStrategy::kLocalRemote}
    };

    auto it = map.find(str);
    if(it != map.end()){
        return it->second;
    }else{
        throw std::runtime_error("Invaild DataPlacementStrategy!");
    }
}

//Funtion to create a centered character
std::string CenterStr(const std::string &str, int width){
    int len = str.length();
    if (len >= width) return str;

    int padding = width - len;
    int left_padding = padding / 2;
    int right_padding = padding - left_padding;

    return std::string(left_padding, ' ') + str + std::string(right_padding, ' ');
}

}   //namespace common
}   //namescpce GNNPro_lib 