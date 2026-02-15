#include <mma.h>

#include <framework.cuh>
#define TILE_WIDTH 16
#define WARP_SIZE 32

using namespace nvcuda;

__global__ void g_matmul_naive(half* A, half* B, half* C, int N, int K, int M) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < N && y < M) {
        float sum = 0;
        for (int i = 0; i < K; i++) {
            sum += __half2float(A[K * x + i]) * __half2float(B[M * i + y]);
        }
        C[M * x + y] = __float2half(sum);
    }
}

__global__ void g_matmul_wmma(half* A, half* B, half* C, int N, int K, int M) {
    int tile_x = blockDim.x * TILE_WIDTH;
    int tile_y = blockDim.y * TILE_WIDTH;
    if (tile_x < N && tile_y < M) {
        wmma::fragment<wmma::accumulator, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH,
                       half>
            tileC;
        wmma::fill_fragment(tileC, 0.0f);
        for (int i = 0; i < (K + TILE_WIDTH - 1) / TILE_WIDTH; i++) {
            wmma::fragment<wmma::matrix_a, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH,
                           half, wmma::row_major>
                tileA;
            wmma::fragment<wmma::matrix_b, TILE_WIDTH, TILE_WIDTH, TILE_WIDTH,
                           half, wmma::row_major>
                tileB;
            wmma::load_matrix_sync(tileA, A + tile_x * K + TILE_WIDTH * i, K);
            wmma::load_matrix_sync(tileB, B + i * TILE_WIDTH * M + tile_x, M);
            wmma::mma_sync(tileC, tileA, tileB, tileC);
        }
        wmma::store_matrix_sync(C + tile_x * M + tile_y, tileC, M,
                                wmma::mem_row_major);
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        puts("usage: wmma N K M");
        return 2;
    }
    int N = atoi(argv[1]);
    int K = atoi(argv[2]);
    int M = atoi(argv[3]);

    dim3 grid_size((N + TILE_WIDTH - 1) / TILE_WIDTH,
                   (M + TILE_WIDTH - 1) / TILE_WIDTH, 1);
    auto cudaApp = new CudaApp<half*, half*, half*, int, int, int>();
    auto naive_task =
        CudaKernelTask("matmul_naive", grid_size,
                       dim3(TILE_WIDTH, TILE_WIDTH, 1), 0, g_matmul_naive);
    auto wmma_task =
        CudaKernelTask("matmul_wmma", grid_size, WARP_SIZE, 0, g_matmul_wmma);

    cudaApp->addTask(&naive_task)
        ->addTask(&wmma_task)
        ->initArgs(CudaRandomArray<half>(N * K, 0, 1),
                   CudaRandomArray<half>(K * M, 0, 1),
                   CudaNewArray<half>(N * M), CudaSetValue(N), CudaSetValue(K),
                   CudaSetValue(M))
        ->run<2>(K * 5e-2);
    delete cudaApp;
}