#ifndef UTILS_CUDA_V2_H
#define UTILS_CUDA_V2_H

#include <cuComplex.h>

// Define a complex number type for simplicity
typedef cuDoubleComplex Complex;


__global__ void applyPhaseFlip(Complex* state, long long int idx);

void allocateGatesDevice(const int num_devices, Complex **H_d, Complex **I_d, Complex **Z_d, Complex **X_d, Complex **X_H_d);

__global__ void applyPhaseFlipParallel(Complex* state, const long long int N, const int N_chunk);


__global__ void compute_idx(
        int qubit,
        int* new_idx,
        int* old_idx,
        const int n,
        const long long int N,
        int* old_linear_idxs
);

void applyGateAllQubits(
    Complex* state,
    const Complex* gate,
    int* new_idx,
    int n,
    dim3 dimBlock,
    dim3 dimGrid,
    int sharedMemSize,
    const long long int N,
    int* old_linear_idxs,
    const int N_chunk
    );

void applyGateSingleQubit(
    Complex* state,
    const Complex* gate,
    int* new_idx,
    int n,
    long long int idx,
    dim3 dimBlock,
    dim3 dimGrid,
    int sharedMemSize,
    const long long int N,
    int* old_linear_idxs,
    const int N_chunk
    );

void applyDiffusionOperator(
    Complex* state,
    const Complex* X_H,
    const Complex* H,
    const Complex* X,
    const Complex* Z,
    int* new_idx,
    int n,
    dim3 dimBlock,
    dim3 dimGrid,
    int sharedMemSize,
    const int N_chunk,
    const long long int N,
    int* old_linear_idxs
    );

__global__ void contract_tensor(
    Complex* state,
    const Complex* gate,
    int qubit,
    int* new_idx,
    const int n,
    const long long int N,
    int* old_linear_idxs,
    const int chunk_size
);

#endif // UTILS_CUDA_STREAM_H
