#include <framework.cuh>

// For simplicity, we just use hard-coded values to represent the
// coefficients.
__constant__ float C[7]{1, 0, 0, 0, 0, 0, 0};
//__constant__ float C[7]{0, 1, -1, 1, -1, 1, -1};

__global__ void stencil_naive(float* in, float* out, int N, int _) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Skip the boundary elements, we don't care about their values.
    if (x >= 1 && x < N - 1 && y >= 1 && y < N - 1 && z >= 1 && z < N - 1) {
        out[z * N * N + y * N + x] = C[0] * in[z * N * N + y * N + x] +
                                     C[1] * in[z * N * N + y * N + x + 1] +
                                     C[2] * in[z * N * N + y * N + x - 1] +
                                     C[3] * in[z * N * N + (y + 1) * N + x] +
                                     C[4] * in[z * N * N + (y - 1) * N + x] +
                                     C[5] * in[(z + 1) * N * N + y * N + x] +
                                     C[6] * in[(z - 1) * N * N + y * N + x];
    }
}

__global__ void stencil_mem_tiled(float* in, float* out, int N,
                                  int tile_width) {
    extern __shared__ float tile[];
    int w = tile_width + 2;
    int tile_size = w * w * w;
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    for (int i = threadIdx.z * blockDim.y * blockDim.x +
                 threadIdx.y * blockDim.x + threadIdx.x;
         i < tile_size; i += block_size) {
        int x0 = blockDim.x * blockIdx.x - 1 + i % w;
        int y0 = blockDim.y * blockIdx.y - 1 + i / w % w;
        int z0 = blockDim.z * blockIdx.z - 1 + i / w / w;
        if (x0 >= 0 && y0 >= 0 && z0 >= 0 && x0 < N && y0 < N && z0 < N) {
            tile[i] = in[z0 * N * N + y0 * N + x0];
        } else {
            tile[i] = 0;
        }
    }
    __syncthreads();

    int x1 = blockDim.x * blockIdx.x + threadIdx.x;
    int y1 = blockDim.y * blockIdx.y + threadIdx.y;
    int z1 = blockDim.z * blockIdx.z + threadIdx.z;
    if (x1 >= 1 && x1 < N - 1 && y1 >= 1 && y1 < N - 1 && z1 >= 1 &&
        z1 < N - 1) {
        int x = threadIdx.x + 1;
        int y = threadIdx.y + 1;
        int z = threadIdx.z + 1;
        out[z1 * N * N + y1 * N + x1] =
            C[0] * tile[z * w * w + y * w + x] +
            C[1] * tile[z * w * w + y * w + x + 1] +
            C[2] * tile[z * w * w + y * w + x - 1] +
            C[3] * tile[z * w * w + (y + 1) * w + x] +
            C[4] * tile[z * w * w + (y - 1) * w + x] +
            C[5] * tile[(z + 1) * w * w + y * w + x] +
            C[6] * tile[(z - 1) * w * w + y * w + x];
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        puts("usage: stencil N tile_width");
        return 2;
    }
    int N = atoi(argv[1]);
    int tile_width = atoi(argv[2]);

    dim3 block_size(tile_width, tile_width, tile_width);
    dim3 grid_size((N + tile_width - 1) / tile_width,
                   (N + tile_width - 1) / tile_width,
                   (N + tile_width - 1) / tile_width);
    auto taskGroup = CudaTaskGroup<float*, float*, int, int>();
    auto naiveTask = CudaKernelTask("stencil_naive", grid_size, block_size, 0,
                                    stencil_naive);
    auto memTiledTask = CudaKernelTask(
        "stencil_mem_tiled", grid_size, block_size,
        pow((tile_width + 2), 3) * sizeof(float), stencil_mem_tiled);
    taskGroup.addTask(&naiveTask)
        .addTask(&memTiledTask)
        .initArgs(CudaDeviceRandomArray(N * N * N), CudaNewArray(N * N * N),
                  CudaConstValue(N), CudaConstValue(tile_width))
        .run<1>(0);
}