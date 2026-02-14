#include <framework.cuh>

__global__ void g_vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

void vectorAdd(float* A, float* B, float* C, int N) {
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
    int grid_size = (N + block_size - 1) / block_size;

    auto cudaApp = new CudaApp<float*, float*, float*, int>();
    auto hostTask = CudaHostTask("vecadd_host", vectorAdd);
    auto kernelTask =
        CudaKernelTask("vecadd_kernel", grid_size, block_size, 0, g_vectorAdd);
    cudaApp->addTask(&hostTask)
        ->addTask(&kernelTask)
        ->initArgs(CudaRandomArray(N, 0, 1), CudaRandomArray(N, 0, 1),
                   CudaNewArray(N), CudaSetValue(N))
        ->run<2>(0);
    delete cudaApp;
}