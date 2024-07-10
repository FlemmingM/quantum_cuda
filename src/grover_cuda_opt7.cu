#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include "utils_cuda_opt7.h"

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
    // Complex *new_state_h;
    // Complex *new_state_d;
    Complex *H_d;
    Complex *X_H_d;
    Complex *I_d;
    Complex *Z_d;
    Complex *X_d;

    int *new_idx_d;
    int *old_idx_d;


    // Init the state
    cudaMallocHost((void **)&state_h, N * sizeof(Complex));
    cudaMalloc((void **)&state_d, N * sizeof(Complex));
    // Init the |0>^(xn) state and the new_state
    // state_h[0] = make_cuDoubleComplex(1.0, 0.0);
    // for (int i = 1; i < N; ++i) {
    //     state_h[i] = make_cuDoubleComplex(0.0, 0.0);
    // }
    // cudaMemcpy(state_d, state_h, N * sizeof(Complex), cudaMemcpyHostToDevice);


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

    int val = 256;
    dim3 dimBlock(val);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    const int blockSize = val;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Allocate shared memory for reduction
    // int sharedMemSize = blockSize * sizeof(Complex);
    int sharedMemSize = 2*N * sizeof(Complex);

    // Malloc the indices on the device
    cudaMalloc(&new_idx_d, gridSize * blockSize * n * sizeof(int));
    cudaMalloc(&old_idx_d, gridSize * blockSize * n * sizeof(int));


    // Assuming we have t = 1 solution in grover's algorithm
    // we have k = floor(pi/4 * sqrt(N))
    int k = (int)floor(M_PI / 4 * sqrt(N));


    // https://forums.developer.nvidia.com/t/question-about-max-shared-memory-in-block-and-multiprocessor/283345

    cudaFuncSetAttribute(contract_tensor, cudaFuncAttributeMaxDynamicSharedMemorySize, sharedMemSize);
    double time = omp_get_wtime();
    zeroOutState<<<gridSize, blockSize>>>(state_d, N);

    // contract_tensor<<<gridSize, blockSize, sharedMemSize>>>(state_d, H_d, 0, shape_d, new_idx_d, old_idx_d, n, N);
    // contract_tensor<<<gridSize, blockSize>>>(state_d, H_d, 0, new_state_d, shape_d, new_idx_d, old_idx_d, n, N);

        // contract_tensor_baseline<<<dimGrid, dimBlock>>>(state, gate, i, new_state, shape, n, N);
        // cudaDeviceSynchronize();
        // Update the state with the new state
    // updateState<<<gridSize, blockSize>>>(state_d, new_state_d, N);





    // Now apply the H gate n times, once for each qubit
    applyGateAllQubits(state_d, H_d, new_idx_d, old_idx_d, n, N, dimBlock, dimGrid, sharedMemSize);

    // cudaDeviceSynchronize();


    // Apply Grover's algorithm k iteration and then sample
    // if (verbose == 1) {
    //     printf("Running %d round(s)\n", k);
    // }

    // for (int i = 0; i < k; ++i) {
    //     applyPhaseFlip<<<dimGrid, dimBlock>>>(state_d, markedState);
    //     applyDiffusionOperator(state_d, X_H_d, H_d, X_d, Z_d, new_idx_d, old_idx_d, n, N, dimBlock, dimGrid, sharedMemSize);
    //     // cudaDeviceSynchronize();
    // }

    cudaDeviceSynchronize();
    double elapsed = omp_get_wtime() - time;
    printf("Time: %f \n", elapsed);


    cudaMemcpy(state_h, state_d, N * sizeof(Complex), cudaMemcpyDeviceToHost);

    // if (verbose == 1) {
    printState(state_h, N, "Initial state");
    // }

    // // Apply Grover's algorithm k iteration and then sample
    // if (verbose == 1) {
    //     printf("Running %d round(s)\n", k);
    // }

    // double time = omp_get_wtime();

    // for (int i = 0; i < k; ++i) {
    //     if (verbose == 1) {
    //         printf("%d/%d\n", i, k);
    //     }
    //     // Apply Oracle
    //     applyPhaseFlip(state, markedState);
    //     if (verbose == 1) {
    //         printState(state, N, "Oracle applied");
    //     }
    //     // Apply the diffusion operator
    //     applyDiffusionOperator(state, new_state, shape, H, X, Z, n, N);
    //     if (verbose == 1) {
    //         printState(state, N, "After Diffusion");
    //     }
    // }

    // double elapsed = omp_get_wtime() - time;
    // printf("Time: %f \n", elapsed);

    // // Sample the states wheighted by their amplitudes
    // double* averages = simulate(state_h, N, 1);
    // if (verbose == 1) {
    //     printf("Average frequency per position:\n");
    //     for (int i = 0; i < N; ++i) {
    //         printf("Position %d: %f\n", i, averages[i]);
    //     }
    // }


    // // save the data
    // saveArrayToCSV(averages, N, fileName);

    cudaFree(state_d);
    cudaFree(H_d);

    cudaFreeHost(state_h);
    cudaFreeHost(H_h);
    cudaFreeHost(I_h);
    cudaFreeHost(Z_h);
    cudaFreeHost(X_h);

    return 0;
}
