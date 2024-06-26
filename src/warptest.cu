#include <cuda_runtime.h>
#include <iostream>

__global__ void increment_kernel(int* d_array, int value, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_array[idx] += value;
    }
}

int main() {
    const int array_size = 10;
    const int array_bytes = array_size * sizeof(int);
    const int increment_value = 5;
    const int nStreams = 4; // Number of streams

    // Allocate host memory
    int* h_array = new int[array_size];
    for (int i = 0; i < array_size; ++i) {
        h_array[i] = i;
    }

    // Allocate device memory
    int* d_array;
    cudaMalloc((void**)&d_array, array_bytes);

    // Create CUDA streams
    cudaStream_t streams[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Define block and grid sizes
    const int block_size = 256;
    const int grid_size = (array_size + block_size - 1) / block_size;

    // Calculate chunk size for each stream
    int chunk_size = array_size / nStreams;
    int chunk_bytes = chunk_size * sizeof(int);

    // Launch kernels and perform memory copy asynchronously
    for (int i = 0; i < nStreams; ++i) {
        int offset = i * chunk_size;

        // Asynchronously copy data from host to device
        cudaMemcpyAsync(d_array + offset, h_array + offset, chunk_bytes, cudaMemcpyHostToDevice, streams[i]);

        // Launch kernel asynchronously on the stream
        increment_kernel<<<(chunk_size + block_size - 1) / block_size, block_size, 0, streams[i]>>>(d_array + offset, increment_value, chunk_size);

        // Asynchronously copy data back from device to host
        cudaMemcpyAsync(h_array + offset, d_array + offset, chunk_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    // Synchronize all streams to ensure all operations are completed
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamSynchronize(streams[i]);
    }

    // Check the results
    bool success = true;
    for (int i = 0; i < array_size; ++i) {
        if (h_array[i] != i + increment_value) {
            success = false;
            break;
        }
        else {
            printf("%d\n", h_array[i]);
        }
    }

    // Free device memory
    cudaFree(d_array);

    // Destroy the streams
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    // Free host memory
    delete[] h_array;

    if (success) {
        std::cout << "Test PASSED" << std::endl;
    } else {
        std::cout << "Test FAILED" << std::endl;
    }

    return 0;
}
