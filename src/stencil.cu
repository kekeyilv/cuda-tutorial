#include <framework.cuh>

// For simplicity, we just use hard-coded values to represent the
// coefficients.
__constant__ float C[7]{0, 1, -1, 1, -1, 1, -1};

__global__ void stencil_naive(float* in, float* out, int N, int _, int __,
                              int ___) {
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

__global__ void stencil_mem_tiled(float* in, float* out, int N, int tile_width,
                                  int _, int __) {
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

__global__ void stencil_coarsened(float* in, float* out, int N, int _,
                                  int tile_width, int coarse_height) {
    extern __shared__ float tile[];
    float* tile_prev = tile;
    float* tile_current = tile + tile_width * tile_width;
    float* tile_next = tile + 2 * tile_width * tile_width;
    int x = (tile_width - 2) * blockIdx.x + threadIdx.x;
    int y = (tile_width - 2) * blockIdx.y + threadIdx.y;
    int z = (coarse_height - 2) * blockIdx.z;
    int x0 = threadIdx.x;
    int y0 = threadIdx.y;
    if (x < N && y < N && z + 1 < N) {
        tile_prev[y0 * tile_width + x0] = in[z * N * N + y * N + x];
        tile_current[y0 * tile_width + x0] = in[(z + 1) * N * N + y * N + x];
        __syncthreads();

        for (int i = 0; i < coarse_height - 2 && z + i + 1 < N; i++) {
            if (z + i + 2 < N) {
                tile_next[y0 * tile_width + x0] =
                    in[(z + i + 2) * N * N + y * N + x];
            } else {
                tile_next[y0 * tile_width + x0] = 0;
            }
            __syncthreads();

            if (x0 >= 1 && y0 >= 1 && x0 < tile_width - 1 &&
                y0 < tile_width - 1 && x < N - 1 && y < N - 1 &&
                z + i + 1 < N - 1) {
                out[(z + i + 1) * N * N + y * N + x] =
                    C[0] * tile_current[y0 * tile_width + x0] +
                    C[1] * tile_current[y0 * tile_width + x0 + 1] +
                    C[2] * tile_current[y0 * tile_width + x0 - 1] +
                    C[3] * tile_current[(y0 + 1) * tile_width + x0] +
                    C[4] * tile_current[(y0 - 1) * tile_width + x0] +
                    C[5] * tile_next[y0 * tile_width + x0] +
                    C[6] * tile_prev[y0 * tile_width + x0];
            }
            __syncthreads();
            tile_prev[y0 * tile_width + x0] =
                tile_current[y0 * tile_width + x0];
            tile_current[y0 * tile_width + x0] =
                tile_next[y0 * tile_width + x0];
        }
    }
}

__global__ void stencil_reg_tiled(float* in, float* out, int N, int _,
                                  int tile_width, int coarse_height) {
    extern __shared__ float tile[];
    int x = (tile_width - 2) * blockIdx.x + threadIdx.x;
    int y = (tile_width - 2) * blockIdx.y + threadIdx.y;
    int z = (coarse_height - 2) * blockIdx.z;
    int x0 = threadIdx.x;
    int y0 = threadIdx.y;
    if (x < N && y < N && z + 1 < N) {
        float prev = in[z * N * N + y * N + x];
        tile[y0 * tile_width + x0] = in[(z + 1) * N * N + y * N + x];
        __syncthreads();

        for (int i = 0; i < coarse_height - 2 && z + i + 1 < N; i++) {
            float next =
                (z + i + 2 < N) ? in[(z + i + 2) * N * N + y * N + x] : 0;

            if (x0 >= 1 && y0 >= 1 && x0 < tile_width - 1 &&
                y0 < tile_width - 1 && x < N - 1 && y < N - 1 &&
                z + i + 1 < N - 1) {
                out[(z + i + 1) * N * N + y * N + x] =
                    C[0] * tile[y0 * tile_width + x0] +
                    C[1] * tile[y0 * tile_width + x0 + 1] +
                    C[2] * tile[y0 * tile_width + x0 - 1] +
                    C[3] * tile[(y0 + 1) * tile_width + x0] +
                    C[4] * tile[(y0 - 1) * tile_width + x0] + C[5] * next +
                    C[6] * prev;
            }
            __syncthreads();
            prev = tile[y0 * tile_width + x0];
            tile[y0 * tile_width + x0] = next;
            __syncthreads();
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 5) {
        puts("usage: stencil N tile_width tile_width_2d coarsen_height");
        return 2;
    }
    int N = atoi(argv[1]);
    int tile_width = atoi(argv[2]);
    int tile_width2d = atoi(argv[3]);
    int coarsen_height = atoi(argv[4]);

    dim3 block_size(tile_width, tile_width, tile_width);
    dim3 grid_size((N + tile_width - 1) / tile_width,
                   (N + tile_width - 1) / tile_width,
                   (N + tile_width - 1) / tile_width);
    dim3 block_size_2d(tile_width2d, tile_width2d, 1);
    dim3 grid_size_2d((N + tile_width2d - 3) / (tile_width2d - 2),
                      (N + tile_width2d - 3) / (tile_width2d - 2),
                      (N + coarsen_height - 3) / (coarsen_height - 2));
    auto taskGroup = CudaTaskGroup<float*, float*, int, int, int, int>();
    auto naiveTask = CudaKernelTask("stencil_naive", grid_size, block_size, 0,
                                    stencil_naive);
    auto memTiledTask = CudaKernelTask(
        "stencil_mem_tiled", grid_size, block_size,
        pow((tile_width + 2), 3) * sizeof(float), stencil_mem_tiled);
    auto coarsenedTask = CudaKernelTask(
        "stencil_coarsened", grid_size_2d, block_size_2d,
        3 * pow(tile_width2d, 2) * sizeof(float), stencil_coarsened);
    auto regTiledTask =
        CudaKernelTask("stencil_reg_tiled", grid_size_2d, block_size_2d,
                       pow(tile_width2d, 2) * sizeof(float), stencil_reg_tiled);
    taskGroup.addTask(&naiveTask)
        .addTask(&memTiledTask)
        .addTask(&coarsenedTask)
        .addTask(&regTiledTask)
        .initArgs(CudaDeviceRandomArray(N * N * N), CudaNewArray(N * N * N),
                  CudaConstValue(N), CudaConstValue(tile_width),
                  CudaConstValue(tile_width2d), CudaConstValue(coarsen_height))
        .run<1>(0);
}