#include <framework.cuh>

template <>
float CudaArg<float*>::toFloat(float value) {
    return value;
}

template <>
float CudaArg<half*>::toFloat(half value) {
    return __half2float(value);
}

template <>
float CudaRandomArray<float>::fromFloat(float value) {
    return value;
}

template <>
half CudaRandomArray<half>::fromFloat(float value) {
    return __float2half(value);
}