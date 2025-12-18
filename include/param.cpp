#include "param.h"

namespace GNNPro_lib{
namespace common{

Weight::Weight(int indim, int hiddim, int outdim, int layers): indim(indim), hiddim(hiddim), outdim(outdim), layers(layers){}

void Weight::Initial(){
    if (layers == 2){
        W1.resize(indim * hiddim, 1);
        W2.resize(hiddim * outdim, 1);
    }else if (layers > 2){
        W1.resize(indim * hiddim, 1);
        W2.resize(hiddim * hiddim, 1);
        W3.resize(hiddim * outdim, 1);
    }else{
        LOG(ERROR) << "Unsupport layer setting!";
    }

}

void Weight::Xavier_Initial(){
    if (layers == 2){
        W1 = Xavier_Uniform(W1, hiddim, indim); //outdim × indim in column-major order
        W2 = Xavier_Uniform(W2, outdim, hiddim);
    }else if (layers > 2){
        W1 = Xavier_Uniform(W1, hiddim, indim);
        W2 = Xavier_Uniform(W2, hiddim, hiddim);
        W3 = Xavier_Uniform(W3, outdim, hiddim);
    }else{
        LOG(ERROR) << "Unsupport layer setting!";
    }
}

template<typename T>
std::vector<T> Weight::Xavier_Uniform(std::vector<T> vec, int rows, int cols){
    //Calculate boundary value
    T scale = std::sqrt(6.0 / (rows + cols));

    //Create a random number generator
    std::random_device seed;
    std::mt19937 gen(seed());
    std::uniform_real_distribution<> dis(-scale, scale);

    //Fill vectors in column-major order
    for(int col = 0; col < cols; ++col){
        for(int row = 0; row < rows; ++row){
            vec[col * rows + row] = dis(gen);
        }
    }

    return vec;
}

AdamOptimizer::AdamOptimizer(int num, bool flag, //For flag, 1 means GPU side, and 0 means CPU side
                            float _alpha, float _beta1,
                            float _beta2, float _eps) 
:element_num(num), alpha(_alpha), beta1(_beta1), beta2(_beta2), eps(_eps){
    if(flag == true){
        Param_GPU_Initial();
    }else{
        Param_CPU_Initial();
    }    
}

}   //namespace common
}   //namescpce GNNPro_lib 