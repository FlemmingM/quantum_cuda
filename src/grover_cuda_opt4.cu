#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include "utils_cuda_opt4.h"

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
    const int chunk_size = atoi(argv[3]);
    // const char* fileName = argv[4];
    // int verbose = atoi(argv[5]);

    if (markedState > (N-1)) {
        fprintf(stderr, "You chose a markedState %d but the largest state possible is state %d", markedState, (N-1));
        return 1;
    }


    // Define the number of groups to do the parallel search with more than 10 qubits
    // while still using the fast shared memory

    int num_groups = pow(2, ((n - 10 < 0) ? 0 : (n - 10)));
    printf("num_groups: %d\n", num_groups);
    // printf("test: %d\n", log2)

    int num_chunks = num_groups * chunk_size;
    int qubits_per_chunk = (int)(n-log2((double)num_chunks));
    printf("num chunks: %d, n per chunk: %d\n",num_chunks, qubits_per_chunk);

    // Define the config for threads, devices and streams
    // const int num_chunks = 2;



    int chunks[num_chunks];
    // int sharedMemSizes[num_chunks];
    int sharedMemSize = (int)(pow(2, 10)) * sizeof(Complex);
    // if (n > 10) {
    //     int sharedMemSize = (int)(pow(2, 10)) * sizeof(Complex);
    // } else {
    //     int sharedMemSize = N * sizeof(Complex);
    // }


    for (int i = 0; i < num_chunks; ++i) {
        chunks[i] = N / num_chunks;
        // sharedMemSizes[i] = N / num_chunks * sizeof(Complex);
    }

    int val = 1024;
    dim3 dimBlock(val);
    dim3 dimGrid((N/num_chunks + dimBlock.x - 1) / dimBlock.x);

    // Set the gates:
    int num_devices = 1;
    Complex *H_d[num_devices];
    Complex *I_d[num_devices];
    Complex *Z_d[num_devices];
    Complex *X_d[num_devices];
    Complex *X_H_d[num_devices];
    allocateGatesDevice(num_devices, H_d, I_d, Z_d, X_d, X_H_d);

    // Init the arrays
    Complex *state_h[num_chunks];
    Complex *state_d[num_chunks];
    int *new_idx_d[num_chunks];
    int *old_idx_d[num_chunks];

    for (int i = 0; i < num_chunks; ++i) {
         // Init the state
        cudaMallocHost((void **)&state_h[i], chunks[i] * sizeof(Complex));
        cudaMalloc((void **)&state_d[i], chunks[i] * sizeof(Complex));
        // Init the |0>^(xn) state and the new_state
        state_h[i][0] = make_cuDoubleComplex(1.0, 0.0);
        for (int j = 1; j < chunks[i]; ++j) {
            state_h[i][j] = make_cuDoubleComplex(0.0, 0.0);
        }
        // cudaMemcpy(state_d[i], state_h[i], chunks[i] * sizeof(Complex), cudaMemcpyHostToDevice);

        // Malloc the indices on the device
        cudaMalloc(&new_idx_d[i], dimGrid.x * dimBlock.x * qubits_per_chunk * sizeof(int));
        cudaMalloc(&old_idx_d[i], dimGrid.x * dimBlock.x * qubits_per_chunk * sizeof(int));
    }



    cudaStream_t streams[chunk_size];


    // Assuming we have t = 1 solution in grover's algorithm
    // we have k = floor(pi/4 * sqrt(N/num_chunks))
    int k = (int)floor(M_PI / 4 * sqrt(N/num_chunks));



    double time = omp_get_wtime();


    int count = 0;
    for (int j = 0; j < num_groups; ++j) {
        #pragma omp parallel for num_threads(chunk_size)
        for (int i = 0; i < chunk_size; ++i) {
            cudaStreamCreate(&streams[i]);
            cudaMemcpyAsync(state_d[j*chunk_size+i], state_h[j*chunk_size+i], chunks[j*chunk_size+i] * sizeof(Complex), cudaMemcpyHostToDevice, streams[i]);
            contract_tensor<<<dimGrid, dimBlock, sharedMemSize>>>(state_d[j*chunk_size+i], H_d[0], 0, new_idx_d[j*chunk_size+i], old_idx_d[j*chunk_size+i], qubits_per_chunk, i*chunks[j*chunk_size+i], (i+1)*chunks[j*chunk_size+i]); //(int)(n-log2((double)num_chunks))
            cudaMemcpyAsync(state_h[j*chunk_size+i], state_d[j*chunk_size+i], chunks[j*chunk_size+i] * sizeof(Complex), cudaMemcpyDeviceToHost, streams[i]);
            cudaStreamSynchronize(streams[i]);
            cudaStreamDestroy(streams[i]);

            count += 1;
        }
    }



    // contract_tensor<<<gridSize, blockSize, sharedMemSize>>>(state_d, H_d, 0, new_idx_d, old_idx_d, n, N);
    // contract_tensor<<<gridSize, blockSize>>>(state_d, H_d, 0, new_state_d, shape_d, new_idx_d, old_idx_d, n, N);

        // contract_tensor_baseline<<<dimGrid, dimBlock>>>(state, gate, i, new_state, shape, n, N);
        // cudaDeviceSynchronize();
        // Update the state with the new state
    // updateState<<<gridSize, blockSize>>>(state_d, new_state_d, N);





    // Now apply the H gate n times, once for each qubit
    // applyGateAllQubits(state_d, H_d, new_idx_d, old_idx_d, n, N, dimBlock, dimGrid, sharedMemSize);


    // for (int i = 0; i < k; ++i) {
    //     applyPhaseFlip<<<dimGrid, dimBlock>>>(state_d, markedState);
    //     applyDiffusionOperator(state_d, H_d, X_d, Z_d, new_idx_d, old_idx_d, n, N, dimBlock, dimGrid, sharedMemSize);
    //     // cudaDeviceSynchronize();
    // }

    // cudaDeviceSynchronize();
    double elapsed = omp_get_wtime() - time;
    printf("Time: %f \n", elapsed);


    // cudaMemcpy(state_h, state_d, N * sizeof(Complex), cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_chunks; ++i) {
        printf("chunk id: %d ######################################\n", i);
        printState(state_h[i], chunks[i], "Initial state");
    }





    for (int i = 0; i < num_devices; ++i) {
        cudaFree(H_d[i]);
        cudaFree(I_d[i]);
        cudaFree(Z_d[i]);
        cudaFree(X_d[i]);
        cudaFree(X_H_d[i]);
    }


    // free(H_h);
    // free(I_h);
    // free(Z_h);
    // free(X_h);
    // free(X_H_h);

    for (int i = 0; i < num_chunks; ++i) {
        cudaFree(state_d[i]);
        cudaFreeHost(state_h[i]);
        cudaFree(new_idx_d[i]);
        cudaFree(old_idx_d[i]);
    }

    return 0;
}
