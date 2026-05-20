#include "scan.cuh"
const size_t NBITS = sizeof(uint) * CHAR_BIT;

__global__ void g_extractBit(uint* arr, uint* out, int N, int iter) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < N) {
        out[x] = (arr[x] >> iter) & 1;
    }
}

__global__ void g_radixSort(uint* in, uint* out, uint* sum_ones, int N,
                            int iter) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    if (x < N) {
        uint tot_ones = sum_ones[N - 1];
        uint bit = (in[x] >> iter) & 1;
        int index =
            (bit == 0) ? (x - sum_ones[x]) : (N - tot_ones + sum_ones[x] - 1);
        out[index] = in[x];
    }
}

__device__ int lowerBound(uint* arr, int l, int r, uint val) {
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] < val) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

__device__ int upperBound(uint* arr, int l, int r, uint val) {
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (arr[mid] <= val) {
            l = mid + 1;
        } else {
            r = mid;
        }
    }
    return l;
}

__global__ void g_mergeSort(uint* in, uint* out, int stride, int N) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= N) return;

    int l = (x / (2 * stride)) * 2 * stride;
    int mid = min(l + stride, N);
    int r = min(l + 2 * stride, N);

    if (x < mid) {
        int rankLeft = x - l;
        int rankRight = lowerBound(in, mid, r, in[x]) - mid;
        out[l + rankLeft + rankRight] = in[x];
    } else {
        int rankRight = x - mid;
        int rankLeft = upperBound(in, l, mid, in[x]) - l;
        out[l + rankLeft + rankRight] = in[x];
    }
}

class MergeSortTask : public CudaTask<uint*, uint*, int> {
   public:
    MergeSortTask(int block_size, int init_stride, int tile_size)
        : CudaTask<uint*, uint*, int>("merge_sort"),
          block_size(block_size),
          init_stride(init_stride),
          tile_size(tile_size) {}

    float run(CudaArg<uint*>& arr, CudaArg<uint*>& out,
              CudaArg<int>& N) override {
        return sort(arr.kernelArg, out.kernelArg, N.hostArg);
    }

    float sort(uint* arr, uint* out, int N) {
        float elapsed_time = 0;
        int grid_size = (N + block_size - 1) / block_size;
        uint* intermediate[2];
        cudaMalloc(&intermediate[0], N * sizeof(uint));
        cudaMalloc(&intermediate[1], N * sizeof(uint));

        for (int i = 0, stride = init_stride; stride < N; i++, stride *= 2) {
            uint *inarr = (stride == init_stride) ? arr : intermediate[i % 2],
                 *outarr = (stride * 2 > N) ? out : intermediate[(i + 1) % 2];
            elapsed_time += CudaKernelTask("_merge_kernel", grid_size,
                                           block_size, 0, g_mergeSort)
                                .execute(inarr, outarr, stride, N);
        }
        return elapsed_time;
    }

    bool onHost() override { return false; }

   private:
    int tile_size;
    int init_stride;
    int block_size;
};

class RadixSortTask : public CudaTask<uint*, uint*, int> {
   public:
    RadixSortTask(int block_size)
        : CudaTask<uint*, uint*, int>("radix_sort"), block_size(block_size) {}

    float run(CudaArg<uint*>& arr, CudaArg<uint*>& out,
              CudaArg<int>& N) override {
        float elapsed_time = 0;
        int Nval = N.hostArg;
        int grid_size = (Nval + block_size - 1) / block_size;
        uint *intermediate[2], *sum_ones;
        cudaMalloc(&sum_ones, Nval * sizeof(uint));
        cudaMalloc(&intermediate[0], Nval * sizeof(uint));
        cudaMalloc(&intermediate[1], Nval * sizeof(uint));

        for (int i = 0; i < NBITS; i++) {
            uint *inarr = (i == 0) ? arr.kernelArg : intermediate[i % 2],
                 *outarr = (i == NBITS - 1) ? out.kernelArg
                                            : intermediate[(i + 1) % 2];

            elapsed_time += CudaKernelTask("_sort_extract_bit", grid_size,
                                           block_size, 0, g_extractBit)
                                .execute(inarr, sum_ones, Nval, i);
            elapsed_time +=
                ScanTask("_sort_scan", block_size, g_KoggeStone<uint>)
                    .scan(sum_ones, sum_ones, Nval);
            elapsed_time += CudaKernelTask("_radix_sort_kernel", grid_size,
                                           block_size, 0, g_radixSort)
                                .execute(inarr, outarr, sum_ones, Nval, i);
        }

        cudaFree(sum_ones);
        cudaFree(intermediate[0]);
        cudaFree(intermediate[1]);
        return elapsed_time;
    }
    bool onHost() override { return false; }

   private:
    int block_size;
};

void sort(uint* in, uint* out, int N) {
    std::copy(in, in + N, out);
    std::sort(out, out + N);
}

int main(int argc, char** argv) {
    if (argc < 4) {
        puts("usage: sort N block_size merge_tile_size");
        return 2;
    }
    int N = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int merge_tile_size = atoi(argv[3]);

    auto taskGroup = CudaTaskGroup<uint*, uint*, int>();
    auto cpuTask = CudaHostTask("host_sort", sort);
    auto radixTask = RadixSortTask(block_size);
    auto mergeTask = MergeSortTask(block_size, 1, merge_tile_size);
    taskGroup.addTask(&cpuTask)
        .addTask(&mergeTask)
        .addTask(&radixTask)
        .initArgs(CudaRandomArray<uint>(N, 0, 2 * N), CudaNewArray<uint>(N),
                  CudaConstValue(N))
        .run<1>(0);
}