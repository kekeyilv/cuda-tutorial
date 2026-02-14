#include <framework.cuh>
#define MAX_COARSE_FACTOR 64

// Naive implementation
__global__ void g_matmul(float* A, float* B, float* C, int N, int K, int M,
                         int _tile_width, int _padding, int _coarse_factor) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < N && y < M) {
        float sum = 0;
        for (int i = 0; i < K; i++) {
            sum += A[K * x + i] * B[M * i + y];
        }
        C[M * x + y] = sum;
    }
}

// Optimized implementation with tiling
__global__ void g_matmul_tiled(float* A, float* B, float* C, int N, int K,
                               int M, int tile_width, int padding,
                               int coarse_factor) {
    extern __shared__ float shared_mem[];
    float* tileA = shared_mem;
    // float* tileB = shared_mem + tile_width * tile_width;
    float* tileB = shared_mem + tile_width * (tile_width + padding);
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    float sums[MAX_COARSE_FACTOR]{0};
    // Divide the rows and cols into ceil(K/tile_width) of tiles
    for (int i = 0; i < (K + tile_width - 1) / tile_width; i++) {
        // Load data in the tile
        int yA = tile_width * i + threadIdx.y;
        int xB = tile_width * i + threadIdx.x;
        int tile_index = (tile_width + padding) * threadIdx.x + threadIdx.y;
        // Teneray operators (may) perform better than if-elses
        tileA[tile_index] = (yA < K && x < N) ? A[K * x + yA] : 0.0f;
        for (int c = 0; c < coarse_factor; c++) {
            int y = blockDim.y * (blockIdx.y * coarse_factor + c) + threadIdx.y;
            tileB[tile_index] = (xB < K && y < M) ? B[M * xB + y] : 0.0f;
            // Wait until all threads have finished loading
            __syncthreads();

            // Computation
            for (int j = 0; j < tile_width; j++) {
                sums[c] += tileA[(tile_width + padding) * threadIdx.x + j] *
                           tileB[(tile_width + padding) * j + threadIdx.y];
            }
            // Before computation of this iteration completed,
            // avoid the tiles being modified.
            __syncthreads();
        }
    }
    for (int c = 0; c < coarse_factor; c++) {
        int y = blockDim.y * (blockIdx.y * coarse_factor + c) + threadIdx.y;
        if (x < N && y < M) {
            C[x * M + y] = sums[c];
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 7) {
        puts("usage: vecadd N K M tile_width padding coase_factor");
        return 2;
    }
    int N = atoi(argv[1]);
    int K = atoi(argv[2]);
    int M = atoi(argv[3]);
    int tile_width = atoi(argv[4]);
    int padding = atoi(argv[5]);
    int coarse_factor = atoi(argv[6]);
    if (coarse_factor > MAX_COARSE_FACTOR) {
        printf("coarse_factor cannot exceed %d\n", MAX_COARSE_FACTOR);
        return 2;
    }

    dim3 block_size(tile_width, tile_width, 1);
    size_t shared_mem_size =
        2 * tile_width * (tile_width + padding) * sizeof(float);
    auto cudaApp =
        new CudaApp<float*, float*, float*, int, int, int, int, int, int>();
    auto naiveTask = CudaKernelTask("matmul_naive",
                                    dim3((N + tile_width - 1) / tile_width,
                                         (M + tile_width - 1) / tile_width, 1),
                                    block_size, shared_mem_size, g_matmul);
    auto tiledTask = CudaKernelTask(
        "matmul_tiled",
        dim3((N + tile_width - 1) / tile_width,
             (M + tile_width * coarse_factor - 1) / tile_width / coarse_factor,
             1),
        block_size, shared_mem_size, g_matmul_tiled);
    cudaApp->addTask(&naiveTask)
        ->addTask(&tiledTask)
        ->initArgs(CudaRandomArray(N * K, 0, 1), CudaRandomArray(K * M, 0, 1),
                   CudaNewArray(N * M), CudaSetValue(N), CudaSetValue(K),
                   CudaSetValue(M), CudaSetValue(tile_width),
                   CudaSetValue(padding), CudaSetValue(coarse_factor))
        ->run<2>(0);
    delete cudaApp;
}