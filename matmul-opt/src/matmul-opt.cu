#include <cuda_runtime.h>
#include <inttypes.h>

#include <cstdio>

// Naive implementation
__global__ void g_matmul(float* A, float* B, float* C, int N, int K, int M,
                         int _tile_width, int _padding) {
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
                               int M, int tile_width, int padding) {
    extern __shared__ float shared_mem[];
    float* tileA = shared_mem;
    // float* tileB = shared_mem + tile_width * tile_width;
    float* tileB = shared_mem + tile_width * (tile_width + padding);
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    float sum = 0;
    // Divide the rows and cols into ceil(K/tile_width) of tiles
    for (int i = 0; i < (K + tile_width - 1) / tile_width; i++) {
        // Load data in the tile
        int yA = tile_width * i + threadIdx.y;
        int xB = tile_width * i + threadIdx.x;
        int tile_index = (tile_width + padding) * threadIdx.x + threadIdx.y;
        // Teneray operators (may) perform better than if-elses
        tileA[tile_index] = (yA < K && x < N) ? A[K * x + yA] : 0.0f;
        tileB[tile_index] = (xB < K && y < M) ? B[M * xB + y] : 0.0f;
        // Wait until all threads have finished loading
        __syncthreads();

        // Computation
        for (int j = 0; j < tile_width; j++) {
            sum += tileA[(tile_width + padding) * threadIdx.x + j] *
                   tileB[(tile_width + padding) * j + threadIdx.y];
        }
        // Before computation of this iteration completed,
        // avoid the tiles being modified.
        __syncthreads();
    }
    // To ensure tiles to be correctly loaded,
    // the if body should not involve the for-statement.
    if (x < N && y < M) {
        C[M * x + y] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        puts("usage: vecadd N K M tile_width padding");
        return 2;
    }
    int N = atoi(argv[1]);
    int K = atoi(argv[2]);
    int M = atoi(argv[3]);
    int tile_width = atoi(argv[4]);
    int padding = atoi(argv[5]);

    // Data preparation
    float *A_h = new float[N * K], *B_h = new float[K * M];
    float *A_d, *B_d, *C_d, *C_h[2]{new float[N * M], new float[N * M]};
    for (int i = 0; i < N * K; i++) {
        A_h[i] = (rand() % 1000) / 1000.0;
    }
    for (int i = 0; i < K * M; i++) {
        B_h[i] = (rand() % 1000) / 1000.0;
    }
    cudaMalloc(&A_d, N * K * sizeof(float));
    cudaMalloc(&B_d, K * M * sizeof(float));
    cudaMemcpy(A_d, A_h, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K * M * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block_size(tile_width, tile_width, 1);
    dim3 grid_size((N + tile_width - 1) / tile_width,
                   (M + tile_width - 1) / tile_width, 1);
    size_t shared_mem_size =
        2 * tile_width * (tile_width + padding) * sizeof(float);
    void (*kernels[2])(float*, float*, float*, int, int, int, int, int) = {
        g_matmul,
        g_matmul_tiled,
    };
    const char* labels[2] = {"Naive", "Tiled"};

    for (int i = 0; i < 2; i++) {
        cudaEvent_t start_d, end_d;
        cudaMalloc(&C_d, N * M * sizeof(float));
        cudaEventCreate(&start_d);
        cudaEventCreate(&end_d);
        cudaEventRecord(start_d);
        kernels[i]<<<grid_size, block_size, shared_mem_size>>>(
            A_d, B_d, C_d, N, K, M, tile_width, padding);
        cudaEventRecord(end_d);
        cudaEventSynchronize(end_d);
        cudaMemcpy(C_h[i], C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(C_d);

        // Measurement
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start_d, end_d);
        printf("%s kernel finished in %.2f ms.\n", labels[i], elapsed_time);
        cudaEventDestroy(start_d);
        cudaEventDestroy(end_d);
    }

    // Test
    for (int i = 0; i < N * M; i++) {
        if (C_h[0][i] != C_h[1][i]) {
            printf("%f %f %d\n", C_h[0][i], C_h[1][i], i);
            puts("Test failed.");
            return 1;
        }
    }
    puts("Test pass.");

    // Free resources
    cudaFree(A_d);
    cudaFree(B_d);
    delete[] A_h, B_h, C_h[0], C_h[1];
}