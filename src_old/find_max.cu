#include <stdio.h>
#include <cuda_runtime.h>

__global__ void findMaxIndexKernel(float* d_array, int* d_maxIndex, float* d_maxValue, int size, int chunk_id) {
    __shared__ float sharedArray[1024];
    __shared__ int sharedIndex[1024];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        sharedArray[tid] = d_array[index];
        sharedIndex[tid] = index;
    } else {
        sharedArray[tid] = -99;  // Set to minimum value if out of bounds
        sharedIndex[tid] = -1;        // Invalid index
    }

    __syncthreads();

    // Perform reduction to find the max value and its index
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && index + stride < size) {
            if (sharedArray[tid] < sharedArray[tid + stride]) {
                sharedArray[tid] = sharedArray[tid + stride];
                sharedIndex[tid] = sharedIndex[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        printf("idx: %d\n", sharedIndex[0]);
        d_maxIndex[chunk_id] = sharedIndex[0];
        d_maxValue[chunk_id] = sharedArray[0];
    }
}

int main() {
    const int size = 1024 * 2;
    float h_array[size];

    // Initialize the array with some values
    for (int i = 0; i < size; i++) {
        if (i == 77){
            h_array[i] = 5000.0;
        } else {
        h_array[i] = i * 1.0;
        }
    }

    float* d_array;
    int* d_maxIndex;
    int* h_maxIndex;
    float* d_maxValue;
    float* h_maxValue;
    // int* d_blockMaxIndex;
    // int numBlocks = (size + 255) / 256;

    cudaMalloc(&d_array, size * sizeof(float));
    cudaMalloc((void**)&d_maxIndex, 1*sizeof(int));
    cudaMallocHost((void**)&h_maxIndex, 1*sizeof(int));
    cudaMalloc((void**)&d_maxValue, 1*sizeof(int));
    cudaMallocHost((void**)&h_maxValue, 1*sizeof(int));
    // cudaMalloc(&d_blockMaxIndex, numBlocks * sizeof(int));

    cudaMemcpy(d_array, h_array, size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(d_maxIndex, h_maxIndex, 1 * sizeof(float), cudaMemcpyHostToDevice);

    findMaxIndexKernel<<<2, 1024>>>(d_array, d_maxIndex, d_maxValue, size, 0);

    // Copy back the results of each block
    // int h_blockMaxIndex = (int*)malloc(sizeof(int));
    cudaMemcpy(h_maxIndex, d_maxIndex, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxValue, d_maxValue, sizeof(float), cudaMemcpyDeviceToHost);

    // Find the overall max index from the block results
    // float maxValue = -99.0;
    // for (int i = 0; i < numBlocks; i++) {
    //     if (h_array[h_blockMaxIndex[i]] > maxValue) {
    //         maxValue = h_array[h_blockMaxIndex[i]];
    //         h_maxIndex = h_blockMaxIndex[i];
    //     }
    // }

    printf("Max value index: %d, val: %f\n", h_maxIndex[0], h_maxValue[0]);

    // Free memory
    cudaFree(d_array);
    cudaFree(d_maxIndex);
    // cudaFree(d_blockMaxIndex);
    // free(h_maxIndex);

    return 0;
}
