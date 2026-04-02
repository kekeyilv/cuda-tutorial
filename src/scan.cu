#include <framework.cuh>

__global__ void g_KoggeStone(float* arr, float* out, float* S, int N) {
    extern __shared__ float tile[];
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    tile[threadIdx.x] = (x < N) ? arr[x] : 0;

    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if (stride <= threadIdx.x) {
            float t = tile[threadIdx.x] + tile[threadIdx.x - stride];
            __syncthreads();
            tile[threadIdx.x] = t;
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

__global__ void g_BrentKung(float* arr, float* out, float* S, int N) {
    extern __shared__ float tile[];
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

__global__ void g_writeBack(float* arr, float* S, int N) {
    if (blockIdx.x > 0) {
        int x0 = blockDim.x * blockIdx.x + threadIdx.x;
        if (x0 < N) {
            arr[x0] += S[blockIdx.x - 1];
        }
    }
}

class ScanTask : public CudaTask<float*, float*, int> {
   public:
    ScanTask(std::string name, int block_size,
             void (*func)(float*, float*, float*, int))
        : CudaTask<float*, float*, int>(name),
          func(func),
          block_size(block_size) {}

    float run(CudaArg<float*>& arr, CudaArg<float*>& out,
              CudaArg<int>& N) override {
        return scan(arr.kernelArg, out.kernelArg, N.hostArg);
    }

    bool onHost() override { return false; }

   private:
    template <typename F>
    static float call(F func) {
        float elapsed_time = 0;
        cudaEvent_t start_d, end_d;
        cudaEventCreate(&start_d);
        cudaEventCreate(&end_d);
        cudaEventRecord(start_d);
        func();
        cudaEventRecord(end_d);
        cudaEventSynchronize(end_d);
        cudaEventElapsedTime(&elapsed_time, start_d, end_d);
        return elapsed_time;
    }

    float scan(float* arr, float* out, int N) {
        float elapsed_time = 0;
        int grid_size = (N + block_size - 1) / block_size;
        float* S = nullptr;
        if (N >= block_size) {
            cudaMalloc(&S, N / block_size * sizeof(float));
        }
        elapsed_time += call([&] {
            func<<<grid_size, block_size, block_size * sizeof(float)>>>(
                arr, out, S, N);
        });
        if (S != nullptr) {
            elapsed_time += scan(S, S, N / block_size);
        }
        elapsed_time +=
            call([&] { g_writeBack<<<grid_size, block_size, 0>>>(out, S, N); });
        cudaFree(S);
        return elapsed_time;
    }
    void (*func)(float*, float*, float*, int);
    int block_size;
};

void prefixSum(float* arr, float* out, int N) {
    double pre = out[0] = arr[0];
    for (int i = 1; i < N; i++) {
        pre += arr[i];
        out[i] = pre;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        puts("usage: scan N block_size");
        return 2;
    }
    int N = atoi(argv[1]);
    int block_size = atoi(argv[2]);

    auto taskGroup = CudaTaskGroup<float*, float*, int>();
    auto cpuTask = CudaHostTask("host_prefix_sum", prefixSum);
    auto ksTask = ScanTask("KoggeStone", block_size, g_KoggeStone);
    auto bkTask = ScanTask("BrentKung", block_size, g_BrentKung);
    taskGroup.addTask(&cpuTask)
        .addTask(&ksTask)
        .addTask(&bkTask)
        .initArgs(CudaDeviceRandomArray(N), CudaNewArray(N), CudaConstValue(N))
        .run<1>(N * 1e-7);
}