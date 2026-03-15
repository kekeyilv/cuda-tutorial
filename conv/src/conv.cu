#include <cmath>
#include <framework.cuh>

__constant__ float filter[1024];

__global__ void conv_naive(float* N, float* F, float* P, int r, int w, int h,
                           int _) {
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
                               int h, int __) {
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

__global__ void conv_tiled_1(float* N, float* _, float* P, int r, int w, int h,
                             int tile_width) {
    extern __shared__ float tile[];
    int tile_size = (tile_width + 2 * r) * (tile_width + 2 * r);
    int block_size = blockDim.x * blockDim.y;
    for (int i = threadIdx.y * blockDim.x + threadIdx.x; i < tile_size;
         i += block_size) {
        int x0 = blockDim.x * blockIdx.x - r + i % (tile_width + 2 * r);
        int y0 = blockDim.y * blockIdx.y - r + i / (tile_width + 2 * r);
        if (x0 >= 0 && y0 >= 0 && x0 < w && y0 < h) {
            tile[i] = N[y0 * w + x0];
        } else {
            tile[i] = 0;
        }
    }
    __syncthreads();

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    float sum = 0;
    if (x < w && y < h) {
        for (int j = -r; j <= r; j++) {
            for (int i = -r; i <= r; i++) {
                sum += tile[(threadIdx.y + j + r) * (tile_width + 2 * r) +
                            threadIdx.x + i + r] *
                       filter[(j + r) * r + i + r];
            }
        }
        P[y * w + x] = sum;
    }
}

__global__ void conv_tiled_2(float* N, float* _, float* P, int r, int w, int h,
                             int tile_width) {
    extern __shared__ float tile[];
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    tile[threadIdx.y * tile_width + threadIdx.x] =
        (x < w && y < h) ? N[y * w + x] : 0;
    __syncthreads();

    float sum = 0;
    if (x < w && y < h) {
        for (int j = -r; j <= r; j++) {
            for (int i = -r; i <= r; i++) {
                int x0 = threadIdx.x + i, y0 = threadIdx.y + j;
                int x1 = x + i, y1 = y + j;
                if (x0 >= 0 && y0 >= 0 && x0 < tile_width && y0 < tile_width) {
                    sum += tile[y0 * tile_width + x0] *
                           filter[(j + r) * r + i + r];
                } else if (x1 >= 0 && y1 >= 0 && x1 < w && y1 < h) {
                    sum += N[y1 * w + x1] * filter[(j + r) * r + i + r];
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
    auto cudaApp = new CudaApp<float*, float*, float*, int, int, int, int>();
    auto naiveTask =
        CudaKernelTask("conv_naive", grid_size, block_size, 0, conv_naive);
    auto constMemTask = CudaKernelTask("conv_const_mem", grid_size, block_size,
                                       0, conv_const_mem);
    auto tiledTask1 = CudaKernelTask(
        "conv_tiled_1", grid_size, block_size,
        pow(radius * 2 + tile_width, 2) * sizeof(float), conv_tiled_1);
    auto tiledTask2 =
        CudaKernelTask("conv_tiled_2", grid_size, block_size,
                       pow(tile_width, 2) * sizeof(float), conv_tiled_2);
    cudaApp->addTask(&naiveTask)
        ->addTask(&constMemTask)
        ->addTask(&tiledTask1)
        ->addTask(&tiledTask2)
        ->initArgs(CudaDeviceRandomArray(W * H),
                   CudaDeviceRandomArray(pow(radius * 2 + 1, 2)),
                   CudaNewArray(W * H), CudaSetValue(radius), CudaSetValue(W),
                   CudaSetValue(H), CudaSetValue(tile_width))
        ->copyToConstant<1>(filter)
        ->run<2>(0);
    delete cudaApp;
}