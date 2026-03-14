#include <curand.h>

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

CudaDeviceRandomArray::CudaDeviceRandomArray(size_t array_size) {
    this->array_size = array_size;
}

void CudaDeviceRandomArray::init(CudaArg<float*>& arg) const {
    arg.size = array_size;
    arg.hostArg = new float[array_size];
    cudaMalloc(&arg.kernelArg, array_size * sizeof(float));

    int64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
    curandGenerator_t generator;
    curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(generator, seed);
    curandGenerateUniform(generator, arg.kernelArg, array_size);
    curandDestroyGenerator(generator);

    cudaMemcpy(arg.hostArg, arg.kernelArg, array_size * sizeof(float),
               cudaMemcpyDeviceToHost);
}