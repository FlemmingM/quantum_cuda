#include <cuda_runtime.h>
#include <iostream>
#include <omp.h>

#define N 1000000000  // Size of the array
// #define N 10
// Kernel for GPU computations
__global__ void kernel(float* data, long int offset, int device_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < offset) {
        // printf("hello from device %d, idx %d\n", device_id, idx);
        data[idx] = idx * 1.0;
    }
}

__global__ void kernel2(float* data, int offset, int device_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
    if (idx < N) {
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
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    std::cout << "Number of devices: " << deviceCount << std::endl;

    if (deviceCount < 2) {
        std::cerr << "This program requires at least two CUDA-capable devices." << std::endl;
        return -1;
    }

    dim3 dimBlock(256);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);
    float *d_data0, *d_data1, *d_data2;

    float *h_data0 = new float[2*N];
    for (int i = 0; i < (2*N); ++i) {
        h_data0[i] = 0.0;
    }

    float *h_data1 = new float[N];
    for (int i = 0; i < N; ++i) {
        h_data1[i] = 0.0;
    }

    float *h_data2 = new float[2*N];
    for (int i = 0; i < (2*N); ++i) {
        h_data2[i] = 0.0;
    }

    float *h_data_res = new float[2*N];
    for (int i = 0; i < (2*N); ++i) {
        h_data_res[i] = 0.0;
    }

    float *h_data_res2 = new float[2*N];
    for (int i = 0; i < (2*N); ++i) {
        h_data_res2[i] = 0.0;
    }

    cudaStream_t streams[2];

    // Set device 0 and allocate memory



    // if (canAccessPeer01 && canAccessPeer10) {
    //     CUDA_CHECK(cudaSetDevice(0));
    //     CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));

    //     // CUDA_CHECK(cudaSetDevice(1));
    //     // CUDA_CHECK(cudaDeviceEnablePeerAccess(0, 0));
    // } else {
    //     std::cerr << "Peer access is not supported between devices 0 and 1." << std::endl;
    //     return -1;
    // }

    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceEnablePeerAccess(1, 0));
    double time = omp_get_wtime();
// #########################
    #pragma omp parallel for num_threads(2)
    for (int i = 0; i < 2; ++i) {
    // Create streams for parallel execution
    CUDA_CHECK(cudaSetDevice(i));
    CUDA_CHECK(cudaStreamCreate(&streams[i]));

    if (i == 0) {
    CUDA_CHECK(cudaMalloc((void**)&d_data0, 2 * N * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_data0, h_data0, 2 * N * sizeof(float), cudaMemcpyHostToDevice, streams[i]));
     } else {
    CUDA_CHECK(cudaMalloc((void**)&d_data1, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(d_data1, h_data1, N * sizeof(float), cudaMemcpyHostToDevice, streams[i]));

     }


    // Launch kernel on device 0
    double time2 = omp_get_wtime();
    if (i == 0) {
        kernel<<<dimGrid, dimBlock, 0, streams[i]>>>(d_data0, N, 0);
    } else {
        kernel<<<dimGrid, dimBlock, 0, streams[i]>>>(d_data1, N, 1);
    }
    double elapsed2 = omp_get_wtime() - time2;
    printf("Time compute %d: %f \n", i, elapsed2);
    // CUDA_CHECK(cudaGetLastError());

    // Launch kernel on device 1, accessing memory from device 0
    // CUDA_CHECK(cudaSetDevice(1));
    // kernel<<<dimGrid, dimBlock, 0, stream1>>>(d_data1, N, 1);  // Offset to process the second half of the array
    // CUDA_CHECK(cudaGetLastError());

    // CUDA_CHECK(cudaSetDevice(0));
    // kernel<<<dimGrid, dimBlock, 0, stream0>>>(d_data0, 2*N, 0);
    // CUDA_CHECK(cudaGetLastError());

    // CUDA_CHECK(cudaMemcpyPeer(d_data0 + N, 0, d_data1, 1, N * sizeof(float)));


    // Copy result back to host
    if (i == 1) {
        // CUDA_CHECK(cudaMemcpyAsync(h_data_res, d_data0, N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
        CUDA_CHECK(cudaMemcpyPeer(d_data0 + N, 0, d_data1, 1, N * sizeof(float)));


    }
    // else {
        // CUDA_CHECK(cudaMemcpyAsync(h_data_res + N, d_data1, N * sizeof(float), cudaMemcpyDeviceToHost, streams[i]));
        // CUDA_CHECK(cudaMemcpyPeer(d_data0 + N, 0, d_data1, 1, N * sizeof(float)));

    // }

    // CUDA_CHECK(cudaSetDevice(1));

    // Synchronize streams
    CUDA_CHECK(cudaStreamSynchronize(streams[i]));

    // CUDA_CHECK(cudaSetDevice(1));
    // CUDA_CHECK(cudaStreamSynchronize(stream1));
        // Cleanup device memory
    // if (i == 0) {
    //     CUDA_CHECK(cudaFree(d_data0));
    // } else {
    //     CUDA_CHECK(cudaFree(d_data1));
    // }

    // Destroy the stream
    CUDA_CHECK(cudaStreamDestroy(streams[i]));
}

    CUDA_CHECK(cudaMemcpy(h_data_res, d_data0, 2 * N * sizeof(float), cudaMemcpyDeviceToHost));


    double elapsed = omp_get_wtime() - time;
    printf("Time: %f \n", elapsed);




    time = omp_get_wtime();
    // CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaMalloc((void**)&d_data2, 2*N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_data2, h_data2, 2* N * sizeof(float), cudaMemcpyHostToDevice));
    double time2 = omp_get_wtime();
    kernel<<<dimGrid, dimBlock>>>(d_data2, 2*N, 0);
    double elapsed2 = omp_get_wtime() - time2;
    printf("Time compute: %f \n", elapsed2);
    CUDA_CHECK(cudaMemcpy(h_data_res2, d_data2, 2* N * sizeof(float), cudaMemcpyDeviceToHost));

    elapsed = omp_get_wtime() - time;
    printf("Time: %f \n", elapsed);
    // CUDA_CHECK(cudaSetDevice(1));
    // CUDA_CHECK(cudaMemcpy(h_data1, d_data1, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Print some results
    // for (int i = 0; i < 2*N; ++i) {
    //     std::cout << h_data_res[i] << " ";
    // }
    // std::cout << std::endl;

    // for (int i = 0; i < N; ++i) {
    //     std::cout << h_data_res[i] << " ";
    // }
    // std::cout << std::endl;

    // Cleanup
    // CUDA_CHECK(cudaFree(d_data0));
    // CUDA_CHECK(cudaFree(d_data1));
    CUDA_CHECK(cudaFree(d_data2));

    delete[] h_data0;
    delete[] h_data1;
    delete[] h_data_res;
    delete[] h_data_res2;

    // CUDA_CHECK(cudaStreamDestroy(streams[0]));
    // CUDA_CHECK(cudaStreamDestroy(streams[1]));

    return 0;
}
