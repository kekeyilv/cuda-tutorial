#include <framework.cuh>

template <>
CudaArg<float*>::~CudaArg() {
    delete[] hostArg;
    cudaFree(kernelArg);
}

template <>
CudaArg<float*>* CudaArg<float*>::clone() const {
    CudaArg<float*>* result = new CudaArg<float*>{
        .hostArg = new float[size],
        .size = size,
    };
    memcpy(result->hostArg, hostArg, size * sizeof(float));
    cudaMalloc(&result->kernelArg, size * sizeof(float));
    cudaMemcpy(result->kernelArg, kernelArg, size * sizeof(float),
               cudaMemcpyDeviceToDevice);
    return result;
}

template <>
void CudaArg<float*>::toHost() {
    cudaMemcpy(hostArg, kernelArg, size * sizeof(float),
               cudaMemcpyDeviceToHost);
}

template <>
bool CudaArg<float*>::operator==(const CudaArg<float*>& b) const {
    for (int i = 0; i < size; i++) {
        if (fabs(hostArg[i] - b.hostArg[i]) > 1e-3) {
            return false;
        }
    }
    return true;
}

CudaNewArray::CudaNewArray(size_t array_size) { this->array_size = array_size; }

void CudaNewArray::init(CudaArg<float*>& arg) const {
    arg.size = array_size;
    arg.hostArg = new float[array_size];
    cudaMalloc(&arg.kernelArg, array_size * sizeof(float));
}

CudaRandomArray::CudaRandomArray(size_t array_size, float rand_min,
                                 float rand_max) {
    this->array_size = array_size;
    this->rand_min = rand_min;
    this->rand_max = rand_max;
}

void CudaRandomArray::init(CudaArg<float*>& arg) const {
    arg.size = array_size;
    arg.hostArg = new float[array_size];
    cudaMalloc(&arg.kernelArg, array_size * sizeof(float));
    int64_t seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::mt19937 engine(seed);
    std::uniform_real_distribution<float> dist(rand_min, rand_max);
    for (size_t i = 0; i < array_size; i++) {
        arg.hostArg[i] = dist(engine);
    }
    cudaMemcpy(arg.kernelArg, arg.hostArg, array_size * sizeof(float),
               cudaMemcpyHostToDevice);
}