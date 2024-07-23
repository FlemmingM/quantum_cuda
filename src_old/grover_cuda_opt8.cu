#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include "utils_cuda_opt8.h"

typedef cuDoubleComplex Complex;

int main(int argc, char* argv[]) {

    // collect input args
    // if (argc < 6) {
    //     fprintf(stderr, "Usage: %s n qubits<int>; marked state<int>; number of samples<int>; fileName<string>; verbose 0 or 1<int>\n", argv[0]);
    //     return 1;
    // }

    int n = atoi(argv[1]);
    long long int N = (long long int)pow(2, n);
    long long int markedState = atoi(argv[2]);
    int warp_size = atoi(argv[3]);
    // int numSamples = atoi(argv[3]);
    // const char* fileName = argv[4];
    // int verbose = atoi(argv[5]);

    if (markedState > (N-1)) {
        fprintf(stderr, "You chose a markedState %d but the largest state possible is state %d", markedState, (N-1));
        return 1;
    }

    // Define the gates
    cuDoubleComplex H_h[4] = {
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0),
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(-1.0 / sqrt(2.0), 0.0)
    };

    cuDoubleComplex X_H_h[4] = {
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(-1.0 / sqrt(2.0), 0.0),
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0)
    };

    cuDoubleComplex I_h[4] = {
        make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0)
    };

    cuDoubleComplex Z_h[4] = {
        make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0),
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(-1.0, 0.0)
    };

    cuDoubleComplex X_h[4] = {
        make_cuDoubleComplex(0.0, 0.0), make_cuDoubleComplex(1.0, 0.0),
        make_cuDoubleComplex(1.0, 0.0), make_cuDoubleComplex(0.0, 0.0)
    };

    Complex *state_h;
    Complex *state_d;
    Complex *H_d;
    Complex *X_H_d;
    Complex *I_d;
    Complex *Z_d;
    Complex *X_d;

    int *new_idx_d;
    int *old_idx_d;
    int *old_linear_idxs_h;
    int *old_linear_idxs_d;
    int *shared_idxs_d;
    int *shared_idxs_h;

    // Init the state
    cudaMallocHost((void **)&state_h, N * sizeof(Complex));
    cudaMalloc((void **)&state_d, N * sizeof(Complex));
    // Init the |0>^(xn) state and the new_state
    state_h[0] = make_cuDoubleComplex(1.0, 0.0);
    for (int i = 1; i < N; ++i) {
        state_h[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    cudaMemcpy(state_d, state_h, N * sizeof(Complex), cudaMemcpyHostToDevice);


    // Malloc the gate on device
    cudaMalloc((void **)&H_d, 4 * sizeof(Complex));
    cudaMalloc((void **)&X_H_d, 4 * sizeof(Complex));
    cudaMalloc((void **)&I_d, 4 * sizeof(Complex));
    cudaMalloc((void **)&Z_d, 4 * sizeof(Complex));
    cudaMalloc((void **)&X_d, 4 * sizeof(Complex));

    // Copy from host to device
    cudaMemcpy(H_d, H_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(X_H_d, X_H_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(I_d, I_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(Z_d, Z_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(X_d, X_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);


    dim3 dimBlock(256);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Allocate shared memory for reduction
    // int sharedMemSize = blockSize * sizeof(Complex);
    int sharedMemSize = 2*N * sizeof(Complex);
    // int sharedMemSize2 = 32 * 2 * N * sizeof(int);

        // Check if the requested shared memory size exceeds the limit
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (sharedMemSize > deviceProp.sharedMemPerBlock) {
        printf("Requested shared memory size (%d bytes) exceeds the device limit (%d bytes).\n", sharedMemSize, deviceProp.sharedMemPerBlock);
        return -1;
    }

    // Malloc the indices on the device
    cudaMalloc(&new_idx_d, gridSize * blockSize * n * sizeof(int));
    cudaMalloc(&old_idx_d, gridSize * blockSize * n * sizeof(int));

    cudaMallocHost(&old_linear_idxs_h, 2 * N * n * sizeof(int));
    cudaMalloc(&old_linear_idxs_d, 2 * N * n * sizeof(int));

    cudaMalloc(&shared_idxs_d, warp_size * 2 * N * sizeof(int));
    cudaMallocHost(&shared_idxs_h, warp_size * 2 * N * sizeof(int));

    // Assuming we have t = 1 solution in grover's algorithm
    // we have k = floor(pi/4 * sqrt(N))
    int k = (int)floor(M_PI / 4 * sqrt(N));

    zeroOutState<<<gridSize, blockSize>>>(shared_idxs_h, warp_size * 2 * N);


    cudaMemcpy(shared_idxs_d, shared_idxs_h, warp_size*2*N*sizeof(int), cudaMemcpyHostToDevice);
    double time = omp_get_wtime();

    for (int i = 0; i < n; ++i) {
        zeroOutState<<<gridSize, blockSize>>>(shared_idxs_d, warp_size * 2 * N);
        compute_idx<<<gridSize, blockSize>>>(i, new_idx_d, old_idx_d, n, N, shared_idxs_d, warp_size);
        warp_sum_reduction<<<(N*2*warp_size + blockSize - 1) / blockSize, blockSize>>>(shared_idxs_d, old_linear_idxs_d + 2*i*N, warp_size * 2 * N, warp_size);
    }


    cudaDeviceSynchronize();
    // cudaMemcpy(shared_idxs_h, shared_idxs_d, warp_size*2*N*sizeof(int), cudaMemcpyDeviceToHost);
    // cudaMemcpy(old_linear_idxs_h, old_linear_idxs_d, 2*N* n * sizeof(int), cudaMemcpyDeviceToHost);


    // for (int i = 0; i < (2*N*n); ++i) {
    //     printf("%d ", old_linear_idxs_h[i]);
    // }

    // printf("###\n");
    // for (int i = 0; i < (warp_size*2*N); ++i) {
    //     printf("%d ", shared_idxs_h[i]);
    // }
    // contract_tensor<<<(N*2 + blockSize - 1) / blockSize, blockSize, sharedMemSize>>>(state_d, H_d, 0, new_idx_d, old_idx_d, n, N, old_linear_idxs_d);
    // contract_tensor<<<(N*2 + blockSize - 1) / blockSize, blockSize, sharedMemSize>>>(state_d, H_d, 1, new_idx_d, old_idx_d, n, N, old_linear_idxs_d);
    // contract_tensor<<<(N*2 + blockSize - 1) / blockSize, blockSize, sharedMemSize>>>(state_d, H_d, 2, new_idx_d, old_idx_d, n, N, old_linear_idxs_d);

    // contract_tensor<<<gridSize, blockSize, sharedMemSize>>>(state_d, H_d, 1, new_idx_d, old_idx_d, n, N, old_linear_idxs_d);
    // contract_tensor<<<gridSize, blockSize, sharedMemSize>>>(state_d, H_d, 2, new_idx_d, old_idx_d, n, N, old_linear_idxs_d);

    // Now apply the H gate n times, once for each qubit
    // applyGateAllQubits(state_d, H_d, new_idx_d, old_idx_d, n, N, dimBlock, (N*2 + blockSize - 1) / blockSize, sharedMemSize, old_linear_idxs_d);
    // for (int i = 0; i < k; ++i) {
    //     applyPhaseFlip<<<dimGrid, dimBlock>>>(state_d, markedState);
    //     applyDiffusionOperator(state_d, X_H_d, H_d, X_d, Z_d, new_idx_d, old_idx_d, n, N, dimBlock, (N*2 + blockSize - 1) / blockSize, sharedMemSize, old_linear_idxs_d);
    // }

    applyGateAllQubits(state_d, H_d, new_idx_d, old_idx_d, n, N, dimBlock, dimGrid, sharedMemSize, old_linear_idxs_d);
    // Apply Grover's algorithm k iteration and then sample
    // for (int i = 0; i < k; ++i) {
    //     applyPhaseFlip<<<dimGrid, dimBlock>>>(state_d, markedState);
    //     applyDiffusionOperator(state_d, X_H_d, H_d, X_d, Z_d, new_idx_d, old_idx_d, n, N, dimBlock, dimGrid, sharedMemSize, old_linear_idxs_d);
    // }

    cudaDeviceSynchronize();
    double elapsed = omp_get_wtime() - time;
    printf("Time: %f \n", elapsed);


    cudaMemcpy(state_h, state_d, N * sizeof(Complex), cudaMemcpyDeviceToHost);

    printState(state_h, N, "Initial state");



    cudaFree(state_d);
    cudaFree(H_d);
    cudaFree(I_d);
    cudaFree(Z_d);
    cudaFree(X_d);
    cudaFree(X_H_d);
    cudaFree(shared_idxs_d);
    cudaFree(old_linear_idxs_d);

    cudaFreeHost(state_h);
    cudaFreeHost(shared_idxs_h);
    cudaFreeHost(old_linear_idxs_h);

    return 0;
}
