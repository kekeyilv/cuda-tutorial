#include <framework.cuh>

template <typename T>
__global__ void g_writeBack(T* arr, T* S, int N) {
    if (blockIdx.x > 0) {
        int x0 = blockDim.x * blockIdx.x + threadIdx.x;
        if (x0 < N) {
            arr[x0] += S[blockIdx.x - 1];
        }
    }
}

template <typename T, typename... Args>
class ScanTask : public CudaTask<T*, T*, int, Args...> {
   public:
    ScanTask(std::string name, int block_size,
             void (*func)(T*, T*, T*, int, Args...))
        : CudaTask<T*, T*, int, Args...>(name),
          func(func),
          block_size(block_size) {}

    float run(CudaArg<T*>& arr, CudaArg<T*>& out, CudaArg<int>& N,
              CudaArg<Args>&... args) override {
        return scan(arr.kernelArg, out.kernelArg, N.kernelArg,
                    args.kernelArg...);
    }

    bool onHost() override { return false; }

   private:
    float scan(T* arr, T* out, int N, Args... args) {
        float elapsed_time = 0;
        int grid_size = (N + block_size - 1) / block_size;
        T* S = nullptr;
        if (N >= block_size) {
            cudaMalloc(&S, N / block_size * sizeof(T));
        }
        elapsed_time += CudaKernelTask("_scan_kernel", grid_size, block_size,
                                       block_size * sizeof(T), func)
                            .execute(arr, out, S, N, args...);
        if (S != nullptr) {
            elapsed_time += scan(S, S, N / block_size);
        }
        elapsed_time += CudaKernelTask("_scan_writeback", grid_size, block_size,
                                       0, g_writeBack<T>)
                            .execute(out, S, N);
        cudaFree(S);
        return elapsed_time;
    }
    void (*func)(T*, T*, T*, int, Args...);
    int block_size;
};