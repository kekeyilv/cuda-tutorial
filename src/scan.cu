#include "scan.cuh"

void prefixSum(float* arr, float* out, int N) {
    double pre = out[0] = arr[0];
    for (int i = 1; i < N; i++) {
        pre += arr[i];
        out[i] = pre;
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        puts("usage: scan N block_size");
        return 2;
    }
    int N = atoi(argv[1]);
    int block_size = atoi(argv[2]);

    auto taskGroup = CudaTaskGroup<float*, float*, int>();
    auto cpuTask = CudaHostTask("host_prefix_sum", prefixSum);
    auto ksTask = ScanTask("KoggeStone", block_size, g_KoggeStone<float>);
    auto bkTask = ScanTask("BrentKung", block_size, g_BrentKung<float>);
    taskGroup.addTask(&cpuTask)
        .addTask(&ksTask)
        .addTask(&bkTask)
        .initArgs(CudaDeviceRandomArray(N), CudaNewArray(N), CudaConstValue(N))
        .run<1>(N * 1e-7);
}