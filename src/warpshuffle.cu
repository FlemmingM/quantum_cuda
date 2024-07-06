#include <stdio.h>
#include <stdlib.h>

__global__ void warp_sum_reduction(const int* input, int* output, int N, int warp_size) {
    // Assuming N is a multiple of 32 (the warp size)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = tid % warp_size; // Get the lane index within the warp
    int warp_id = tid / warp_size; // Get the warp ID

    int val = 0;
    if (tid < N) {
        val = input[tid];
    }

    // Perform warp-level reduction using shuffle down
    for (int offset = warp_size / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset, warp_size);
    }

    // Write the reduced value of each warp to the output array
    if (lane == 0 && warp_id < (N/warp_size)) {
        output[warp_id] = val;
    }
}

int main() {
    int N = 512;
    int warp_size = 64;
    int out_size = N / warp_size;
    int *d_input, *d_output;
    int *h_input, *h_output;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_output, out_size * sizeof(int));
    cudaMallocHost(&h_input, N * sizeof(int));
    cudaMallocHost(&h_output, out_size * sizeof(int));

    for (int i = 1; i < N; ++i) {
        h_input[i] = 1;
    }

    // Fill d_input with data (omitted here for brevity)

    int threads_per_block = 512;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);


    warp_sum_reduction<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, N, warp_size);

    cudaMemcpy(h_output, d_output, out_size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < out_size; ++i) {
            printf("%d ", h_output[i]);
        }

    // Copy results from device to host (omitted here for brevity)

    cudaFree(d_input);
    cudaFree(d_output);
}








