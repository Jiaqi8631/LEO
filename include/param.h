#include <vector>
#include <random>
#include <cudnn.h>
#include "type.h"
#include "logging.h"

namespace GNNPro_lib{
namespace common{

class Weight{
public:
    std::vector<weight_t> W1;
    std::vector<weight_t> W2;
    std::vector<weight_t> W3;

    int indim;
    int hiddim;
    int outdim;
    int layers;

    void Initial();
    void Xavier_Initial();
    
template<typename T>
    std::vector<T> Xavier_Uniform(std::vector<T> vec, int rows, int cols);

    Weight(int indim, int hiddim, int outdim, int layers);
    ~Weight(){};
};

class AdamOptimizer{
public:
    AdamOptimizer(int num, bool flag = true,
                  float _alpha = 0.001f, float _beta1 = 0.9f,
                  float _beta2 = 0.999f, float _eps = 1e-8f);
    ~AdamOptimizer(){};
    void Param_GPU_Initial();
    void Param_CPU_Initial();
    void Update(value_t* grad, cudaStream_t stream);
    void Update_stream(value_t* grad, cudaStream_t stream);
    void Update_CPU(value_t* grad);

    int element_num;
    int t = 1;
    float alpha, beta1, beta2, eps;
    value_t *m, *v, *out_param;    
};

}   //namespace common
}   //namescpce GNNPro_lib  