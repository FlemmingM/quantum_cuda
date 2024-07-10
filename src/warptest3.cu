#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>


// Kernel for GPU computations
__global__ void kernel(float* data, long int offset, int device_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < offset) {
        // printf("hello from device %d, idx %d\n", device_id, idx);
        data[idx] = idx * 1.0;
    }
}

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << "@" << __LINE__ << ": " \
                      << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while (0)


int main() {

    #define N 16  // Size of the array
    const int num_threads = 4;

    int chunks[num_threads];

    for (int i = 0; i < num_threads; ++i) {
        chunks[i] = N / num_threads;
    }

    dim3 dimBlock(256);
    dim3 dimGrid((N/num_threads + dimBlock.x - 1) / dimBlock.x);


    float *d_data[num_threads];
    float *h_data[num_threads];

    cudaStream_t streams[num_threads];


    double time = omp_get_wtime();

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < num_threads; ++i) {
        // Create streams for parallel execution

        CUDA_CHECK(cudaStreamCreate(&streams[i]));
        CUDA_CHECK(cudaMallocHost((void**)&h_data[i], chunks[i] * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)&d_data[i], chunks[i] * sizeof(float)));

        double time2 = omp_get_wtime();
        kernel<<<dimGrid, dimBlock, 0, streams[i]>>>(d_data[i], chunks[i], 0);
        double elapsed2 = omp_get_wtime() - time2;

        CUDA_CHECK(cudaMemcpyAsync(h_data[i], d_data[i], chunks[i] * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));

        // Synchronize streams
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));

        // Destroy the stream
        CUDA_CHECK(cudaStreamDestroy(streams[i]));
}

    double elapsed = omp_get_wtime() - time;
    printf("Time: %f \n", elapsed);

    for (int i = 0; i < num_threads; ++i) {
        printf("[ ");
        for (int j=0; j < chunks[i]; j++) {
            printf("%f ", h_data[i][j]);
        }
        printf("]\n");
    }


    // Cleanup
    for (int i = 0; i < num_threads; ++i) {
        CUDA_CHECK(cudaFree(d_data[i]));
        CUDA_CHECK(cudaFreeHost(h_data[i]));
    }

    return 0;
}
