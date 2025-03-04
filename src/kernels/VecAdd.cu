extern "C" __global__ void VecAdd(int *a, int *b, int *c, int N) {
    int tid = blockIdx.x;
    if (tid < N) {
        c[tid] = a[tid] + b[tid];
    }
}