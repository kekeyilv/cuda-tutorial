#include <framework.cuh>

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
        puts("usage: matmul N K M");
        return 2;
    }
    int N = atoi(argv[1]);
    int K = atoi(argv[2]);
    int M = atoi(argv[3]);

    dim3 block_size(32, 32, 1);
    dim3 grid_size((N + 31) / 32, (M + 31) / 32, 1);
    auto cudaApp = new CudaApp<float*, float*, float*, int, int, int>();
    auto hostTask = CudaHostTask("matmul_host", matmul);
    auto kernelTask =
        CudaKernelTask("matmul_kernel", grid_size, block_size, 0, g_matmul);
    cudaApp->addTask(&hostTask)
        ->addTask(&kernelTask)
        ->initArgs(CudaRandomArray(N * K, 0, 1), CudaRandomArray(K * M, 0, 1),
                   CudaNewArray(N * M), CudaSetValue(N), CudaSetValue(K),
                   CudaSetValue(M))
        ->run<2>(1e-3);
    delete cudaApp;
}