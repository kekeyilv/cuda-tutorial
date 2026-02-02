#include <cuda_runtime.h>
#include <inttypes.h>

#include <chrono>
#include <cstdio>
using time_point = std::chrono::steady_clock::time_point;

__global__ void g_matmul(float* A, float* B, float* C, int N, int K, int M) {
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

void matmul(float* A, float* B, float* C, int N, int K, int M) {
    for (int x = 0; x < N; x++) {
        for (int y = 0; y < M; y++) {
            float sum = 0;
            for (int i = 0; i < K; i++) {
                sum += A[K * x + i] * B[M * i + y];
            }
            C[M * x + y] = sum;
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        puts("usage: vecadd N K M");
        return 2;
    }
    int N = atoi(argv[1]);
    int K = atoi(argv[2]);
    int M = atoi(argv[3]);

    // Data preparation
    float *A_h = new float[N * K], *B_h = new float[K * M],
          *C_h = new float[N * M];
    float *A_d, *B_d, *C_d, *Ccopy_h = new float[N * M];
    for (int i = 0; i < N * K; i++) {
        A_h[i] = (rand() % 1000) / 1000.0;
    }
    for (int i = 0; i < K * M; i++) {
        B_h[i] = (rand() % 1000) / 1000.0;
    }
    cudaMalloc(&A_d, N * K * sizeof(float));
    cudaMalloc(&B_d, K * M * sizeof(float));
    cudaMalloc(&C_d, N * M * sizeof(float));
    cudaMemcpy(A_d, A_h, N * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, K * M * sizeof(float), cudaMemcpyHostToDevice);

    // CPU Ver.
    time_point start_h = std::chrono::steady_clock::now();
    matmul(A_h, B_h, C_h, N, K, M);
    time_point end_h = std::chrono::steady_clock::now();
    printf(
        "Host code finished in %" PRId64 " ms.\n",
        std::chrono::duration_cast<std::chrono::milliseconds>(end_h - start_h)
            .count());

    // GPU Ver.
    dim3 block_size(32, 32, 1);
    dim3 grid_size((N + 31) / 32, (M + 31) / 32, 1);
    cudaEvent_t start_d, end_d;
    cudaEventCreate(&start_d);
    cudaEventCreate(&end_d);
    cudaEventRecord(start_d);
    g_matmul<<<grid_size, block_size>>>(A_d, B_d, C_d, N, K, M);
    cudaEventRecord(end_d);
    cudaEventSynchronize(end_d);

    // Measurement
    float elapsed_time;
    cudaEventElapsedTime(&elapsed_time, start_d, end_d);
    printf("Kernel finished in %.2f ms.\n", elapsed_time);

    // Test
    cudaMemcpy(Ccopy_h, C_d, N * M * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < N * M; i++) {
        if (fabs(Ccopy_h[i] - C_h[i]) > 1e5) {
            printf("%f %f %d\n", Ccopy_h[i], C_h[i], i);
            puts("Test failed.");
            return 1;
        }
    }
    puts("Test pass.");

    // Free resources
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    delete[] A_h, B_h, C_h, Ccopy_h;
}