#ifndef UTILS_CUDA_STREAM_H
#define UTILS_CUDA_STREAM_H

#include <cuComplex.h>

// Define a complex number type for simplicity
typedef cuDoubleComplex Complex;


__global__ void applyPhaseFlip(Complex* state, long long int idx);

void allocateGatesDevice(const int num_devices, Complex **H_d, Complex **I_d, Complex **Z_d, Complex **X_d, Complex **X_H_d);

__global__ void findMaxIndexKernel(Complex* d_array, int* d_maxIndex, double* d_maxValue, int size, int chunk_id, int* chunk_ids);

void applyGateAllQubits(
    Complex* state,
    const Complex* gate,
    int* new_idx,
    int* old_idx,
    int n,
    dim3 dimBlock,
    dim3 dimGrid,
    int sharedMemSize,
    const long long int lower,
    const long long int upper,
    cudaStream_t stream
    );

void applyGateSingleQubit(
    Complex* state,
    const Complex* gate,
    int* new_idx,
    int* old_idx,
    int n,
    long long int idx,
    dim3 dimBlock,
    dim3 dimGrid,
    int sharedMemSize,
    const long long int lower,
    const long long int upper,
    cudaStream_t stream
    );

void applyDiffusionOperator(
    Complex* state,
    const Complex* X_H,
    const Complex* H,
    const Complex* X,
    const Complex* Z,
    int* new_idx,
    int* old_idx,
    int n,
    dim3 dimBlock,
    dim3 dimGrid,
    int sharedMemSize,
    const long long int lower,
    const long long int upper,
    cudaStream_t stream
    );

__global__ void contract_tensor(
    Complex* state,
    const Complex* gate,
    int qubit,
    int* new_idx,
    int* old_idx,
    const int n,
    const long long int lower,
    const long long int upper
);

#endif // UTILS_CUDA_STREAM_H
