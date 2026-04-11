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

        // out.toHost();
        // for (int i = 0; i < Nval; i++) {
        //     printf("%u ", out.hostArg[i]);
        // }
        // putchar('\n');
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
    if (argc < 3) {
        puts("usage: sort N block_size");
        return 2;
    }
    int N = atoi(argv[1]);
    int block_size = atoi(argv[2]);

    auto taskGroup = CudaTaskGroup<uint*, uint*, int>();
    auto cpuTask = CudaHostTask("host_sort", sort);
    auto radixTask = RadixSortTask(block_size);
    taskGroup.addTask(&cpuTask)
        .addTask(&radixTask)
        .initArgs(CudaRandomArray<uint>(N, 0, 2 * N), CudaNewArray<uint>(N),
                  CudaConstValue(N))
        .run<1>(0);
}