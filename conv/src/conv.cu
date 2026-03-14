#include <framework.cuh>

__constant__ float filter[1024];

__global__ void conv_naive(float* N, float* F, float* P, int r, int w, int h) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    float sum = 0;

    if (x < w && y < h) {
        for (int j = -r; j <= r; j++) {
            for (int i = -r; i <= r; i++) {
                int x0 = x + i, y0 = y + j;
                if (x0 >= 0 && y0 >= 0 && x0 < w && y0 < h) {
                    sum += N[y0 * w + x0] * F[(j + r) * r + i + r];
                }
            }
        }
        P[y * w + x] = sum;
    }
}

__global__ void conv_const_mem(float* N, float* _, float* P, int r, int w,
                               int h) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    float sum = 0;

    if (x < w && y < h) {
        for (int j = -r; j <= r; j++) {
            for (int i = -r; i <= r; i++) {
                int x0 = x + i, y0 = y + j;
                if (x0 >= 0 && y0 >= 0 && x0 < w && y0 < h) {
                    sum += N[y0 * w + x0] * filter[(j + r) * r + i + r];
                }
            }
        }
        P[y * w + x] = sum;
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        puts("usage: conv W H radius tile_width");
        return 2;
    }
    int W = atoi(argv[1]);
    int H = atoi(argv[2]);
    int radius = atoi(argv[3]);
    int tile_width = atoi(argv[4]);

    dim3 block_size(tile_width, tile_width, 1);
    dim3 grid_size((W + tile_width - 1) / tile_width,
                   (H + tile_width - 1) / tile_width, 1);
    auto cudaApp = new CudaApp<float*, float*, float*, int, int, int>();
    auto naiveTask =
        CudaKernelTask("conv_naive", grid_size, block_size, 0, conv_naive);
    auto constMemTask =
        CudaKernelTask("conv_const_mem", grid_size, block_size, 0, conv_naive);
    cudaApp->addTask(&naiveTask)
        ->addTask(&constMemTask)
        ->initArgs(CudaDeviceRandomArray(W * H),
                   CudaDeviceRandomArray((radius * 2 + 1) * (radius * 2 + 1)),
                   CudaNewArray(W * H), CudaSetValue(radius), CudaSetValue(W),
                   CudaSetValue(H))
        ->copyToConstant<1>(filter)
        ->run<2>(0);
    delete cudaApp;
}