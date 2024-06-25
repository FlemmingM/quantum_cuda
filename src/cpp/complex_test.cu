#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
// #include <cuda_runtime.h>

typedef cuDoubleComplex Complex;


__global__
void addNums(const Complex *a, const Complex *b, Complex *c, int N){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx < N) {
        c[idx] = cuCadd(a[idx], b[idx]);
    }
}

int main() {
    int N = 10;
    size_t size = N * sizeof(cuDoubleComplex);

    // Host memory allocation with cudaMallocHost
    cuDoubleComplex *h_a, *h_b, *h_c;
    cudaMallocHost((void **)&h_a, size);
    cudaMallocHost((void **)&h_b, size);
    cudaMallocHost((void **)&h_c, size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = make_cuDoubleComplex(i, i);
        h_b[i] = make_cuDoubleComplex(i, -i);
    }

    cuDoubleComplex *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    addNums<<<numBlocks, blockSize>>>(d_a, d_b, d_c, N);

    // Copy results from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print the results
    printf("Results of complex addition:\n");
    for (int i = 0; i < N; i++) {
        printf("(%f, %f) + (%f, %f) = (%f, %f)\n",
               cuCreal(h_a[i]), cuCimag(h_a[i]),
               cuCreal(h_b[i]), cuCimag(h_b[i]),
               cuCreal(h_c[i]), cuCimag(h_c[i]));
    }

    // Free memory
    cudaFree(h_a);
    cudaFree(h_b);
    cudaFree(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);


}



