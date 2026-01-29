#include <cuda_runtime.h>
#include <inttypes.h>

#include <chrono>
#include <cstdio>
using time_point = std::chrono::steady_clock::time_point;

__global__ void g_vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void vectorAdd(const float* A, const float* B, float* C, int N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        puts("usage: vecadd N block_size");
        return 2;
    }
    int N = atoi(argv[1]);
    int block_size = atoi(argv[2]);

    // Data preparation
    float *A_h = new float[N], *B_h = new float[N], *C_h = new float[N];
    float *A_d, *B_d, *C_d, *Ccopy_h = new float[N];
    for (int i = 0; i < N; i++) {
        A_h[i] = 1.0f / rand();
        B_h[i] = 1.0f / rand();
    }
    cudaMalloc(&A_d, N * sizeof(float));
    cudaMalloc(&B_d, N * sizeof(float));
    cudaMalloc(&C_d, N * sizeof(float));
    cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, N * sizeof(float), cudaMemcpyHostToDevice);

    // CPU Ver.
    time_point start_h = std::chrono::steady_clock::now();
    vectorAdd(A_h, B_h, C_h, N);
    time_point end_h = std::chrono::steady_clock::now();
    printf(
        "Host code finished in %" PRId64 " ms.\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(end_h - start_h)
            .count());

    // GPU Ver.
    int grid_size = (N + block_size - 1) / block_size;
    cudaEvent_t start_d, end_d;
    cudaEventCreate(&start_d);
    cudaEventCreate(&end_d);
    cudaEventRecord(start_d);
    g_vectorAdd<<<grid_size, block_size>>>(A_d, B_d, C_d, N);
    cudaEventRecord(end_d);
    cudaEventSynchronize(end_d);

    // Measurement
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start_d, end_d);
    printf("Kernel finished in %.2f ms.\n", elapsed_time);
    
    // Test
    cudaMemcpy(Ccopy_h, C_d, N * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N; i++) {
        if (Ccopy_h[i] != C_h[i]) {
            puts("Test failed.");
            return 1;
        }
    }
    puts("Test pass.");
}