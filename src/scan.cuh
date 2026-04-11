#include <framework.cuh>

template <typename T>
__global__ void g_KoggeStone(T* arr, T* out, T* S, int N) {
    extern __shared__ T tile[];
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    tile[threadIdx.x] = (x < N) ? arr[x] : 0;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        T t = tile[threadIdx.x];
        if (stride <= threadIdx.x) {
            t += tile[threadIdx.x - stride];
        }
        __syncthreads();
        tile[threadIdx.x] = t;
    }

    __syncthreads();
    if (x < N) {
        out[x] = tile[threadIdx.x];
        if (threadIdx.x == blockDim.x - 1) {
            S[blockIdx.x] = out[x];
        }
    }
}

template <typename T>
__global__ void g_BrentKung(T* arr, T* out, T* S, int N) {
    extern __shared__ T tile[];
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    tile[threadIdx.x] = (x < N) ? arr[x] : 0;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        // remap the threads to reduce control divergence
        int x0 = 2 * (threadIdx.x + 1) * stride - 1;
        if (x0 < blockDim.x) {
            // the indexes of each update are 2*stride away from neighbors
            // (--write--[stride]--read--[stride]--write--)
            // so tile[x0] won't be read elsewhere in current iteration
            // that's why a __syncthreads() isn't required here
            tile[x0] += tile[x0 - stride];
        }
    }

    for (int stride = blockDim.x / 4; stride > 0; stride /= 2) {
        __syncthreads();
        int x0 = 2 * (threadIdx.x + 1) * stride - 1;
        if (x0 + stride < blockDim.x) {
            tile[x0 + stride] += tile[x0];
        }
    }

    __syncthreads();
    if (x < N) {
        out[x] = tile[threadIdx.x];
        if (threadIdx.x == blockDim.x - 1) {
            S[blockIdx.x] = out[x];
        }
    }
}

template <typename T>
__global__ void g_writeBack(T* arr, T* S, int N) {
    if (blockIdx.x > 0) {
        int x0 = blockDim.x * blockIdx.x + threadIdx.x;
        if (x0 < N) {
            arr[x0] += S[blockIdx.x - 1];
        }
    }
}

template <typename T>
class ScanTask : public CudaTask<T*, T*, int> {
   public:
    ScanTask(std::string name, int block_size, void (*func)(T*, T*, T*, int))
        : CudaTask<T*, T*, int>(name), func(func), block_size(block_size) {}

    float run(CudaArg<T*>& arr, CudaArg<T*>& out, CudaArg<int>& N) override {
        return scan(arr.kernelArg, out.kernelArg, N.kernelArg);
    }

    bool onHost() override { return false; }

    float scan(T* arr, T* out, int N) {
        float elapsed_time = 0;
        int grid_size = (N + block_size - 1) / block_size;
        T* S = nullptr;
        if (N >= block_size) {
            cudaMalloc(&S, N / block_size * sizeof(T));
        }
        elapsed_time += CudaKernelTask("_scan_kernel", grid_size, block_size,
                                       block_size * sizeof(T), func)
                            .execute(arr, out, S, N);
        if (S != nullptr) {
            elapsed_time += scan(S, S, N / block_size);
        }
        elapsed_time += CudaKernelTask("_scan_writeback", grid_size, block_size,
                                       0, g_writeBack<T>)
                            .execute(out, S, N);
        cudaFree(S);
        return elapsed_time;
    }

   private:
    void (*func)(T*, T*, T*, int);
    int block_size;
};