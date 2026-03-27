#include <framework.cuh>

void sum(float* in, float* out, int N, int) {
    double result = 0;
    for (int i = 0; i < N; i++) {
        result += in[i];
    }
    *out = result;
}

__global__ void g_sum(float* in, float* out, int N, int) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < N) {
        atomicAdd(out, in[x]);
    }
}

__global__ void g_cleanup(float*, float* out, int, int) { *out = 0; }

__global__ void g_reduction(float* in, float* out, int N, int coarsen_factor) {
    extern __shared__ float tile[];
    int x = threadIdx.x;
    int x0 = coarsen_factor * blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = 0; i < coarsen_factor; i++, x0 += blockDim.x) {
        if (i == 0) {
            tile[x] = 0;
        }
        if (x0 < N) {
            tile[x] += in[x0];
        }
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        if (x < stride) {
            tile[x] += tile[x + stride];
        }
        __syncthreads();
    }

    if (x == 0) {
        atomicAdd(out, tile[0]);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        puts("usage: reduction N block_size coarsen_factor");
        return 2;
    }
    int N = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int coarsen_factor = atoi(argv[3]);
    int grid_size =
        (N + block_size * coarsen_factor - 1) / (block_size * coarsen_factor);

    auto taskGroup = CudaTaskGroup<float*, float*, int, int>();
    auto sumHostTask = CudaHostTask("sum_host", sum);
    auto sumTask = CudaKernelTask("sum", (N + block_size - 1) / block_size,
                                  block_size, 0, g_sum);
    auto reductionTask =
        CudaKernelTask("reduction", grid_size, block_size,
                       block_size * sizeof(float), g_reduction);
    auto cleanUpTask = CudaKernelTask("cleanup", 1, 1, 0, g_cleanup);
    taskGroup.addTask(&sumHostTask)
        .addTask(&cleanUpTask, true)
        .addTask(&sumTask)
        .addTask(&cleanUpTask, true)
        .addTask(&reductionTask)
        .initArgs(CudaDeviceRandomArray(N), CudaNewArray(1), CudaConstValue(N),
                  CudaConstValue(coarsen_factor))
        .run<1>(N * 1e-5);
}