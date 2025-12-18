#pragma once
#include <chrono>

namespace GNNPro_lib{
namespace common{

class Timer{
public:
    Timer(std::chrono::time_point<std::chrono::steady_clock> tp = std::chrono::steady_clock::now())
    :start_time(tp){}

    template <typename T>
    bool Timeout(double count) const {
        return Passed<T>() >= count;
    }

    double Passed() const {return Passed<std::chrono::duration<double>>(); }

    double PassedSec() const {return Passed<std::chrono::seconds>(); }

    double PassedMilli() const {return Passed<std::chrono::milliseconds>(); }

    double PassedMicro() const {return Passed<std::chrono::microseconds>(); }

    double PassedNano() const {return Passed<std::chrono::nanoseconds>(); }

    template <typename T>
    double Passed() const {
        return Passed<T>(std::chrono::steady_clock::now());
    }

    template <typename T>
    double Passed(std::chrono::time_point<std::chrono::steady_clock> tp) const {
        const auto duration = std::chrono::duration_cast<T>(tp - start_time);
        return duration.count();
    }

    void Reset() {start_time = std::chrono::steady_clock::now(); }

private:
    std::chrono::time_point<std::chrono::steady_clock> start_time;
};

}   //namespace common
}   //namescpce GNNPro_lib