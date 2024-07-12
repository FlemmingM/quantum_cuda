#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include "utils_cuda_opt4.h"

typedef cuDoubleComplex Complex;


#define cudaCheckError(call) {                               \
    cudaError_t err = call;                                  \
    if (err != cudaSuccess) {                                \
        fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                  \
    }                                                        \
}



int main(int argc, char* argv[]) {

    // collect input args
    // if (argc < 6) {
    //     fprintf(stderr, "Usage: %s n qubits<int>; marked state<int>; number of samples<int>; fileName<string>; verbose 0 or 1<int>\n", argv[0]);
    //     return 1;
    // }

    int n = atoi(argv[1]);
    long long int N = (long long int)pow(2, n);
    long long int markedState = atoi(argv[2]);
    const int num_chunks_per_group = N / 1024;
    // const int num_chunks_per_group = atoi(argv[3]);
    // const int block_size = atoi(argv[4]);
    // const char* fileName = argv[4];
    // int verbose = atoi(argv[5]);

    if (markedState > (N-1)) {
        fprintf(stderr, "You chose a markedState %d but the largest state possible is state %d", markedState, (N-1));
        return 1;
    }


    // Define the number of groups to do the parallel search with more than 10 qubits
    // while still using the fast shared memory

    long long int num_groups = (long long int)pow(2, ((n - 12 < 0) ? 0 : (n - 12)));
    printf("num_groups: %d\n", num_groups);
    printf("num_chunks_per_group: %d\n", num_chunks_per_group);
    // printf("test: %d\n", log2)
    long long int num_chunks = num_groups * num_chunks_per_group;
    // int qubits_per_chunk = (int)(n-log2((double)num_chunks));
    int qubits_per_chunk = 10;
    // long long int N_group = (int)(pow(2, qubits_per_chunk));
    int N_group = 1024;

    printf("num chunks: %lld, n per chunk: %lld\n",num_chunks, qubits_per_chunk);

    // Define the config for threads, devices and streams
    // const int num_chunks = 2;

    // int chunks[num_chunks];
    // int sharedMemSizes[num_chunks];
    int sharedMemSize = (int)(pow(2, 11)) * sizeof(Complex);
    printf("Using shared memory size: %d, per group[%d]: %d\n", sharedMemSize, num_chunks_per_group, sharedMemSize/num_chunks_per_group);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (sharedMemSize > deviceProp.sharedMemPerBlock) {
        printf("Requested shared memory size (%d bytes) exceeds the device limit (%d bytes).\n", sharedMemSize, deviceProp.sharedMemPerBlock);
        return -1;
    }

    // for (int i = 0; i < num_chunks; ++i) {
    //     chunks[i] = N / num_chunks;
    // }


    printf("Using chunk size for computation: %d\n", N_group);

    // Define the chunk for the oracle
    printf("N: %lld, markedState: %d, num_chunks: %lld\n", N, markedState, num_chunks);
    printf("%f\n", (N / num_chunks));

    // long long int oracle_chunk = markedState / (N / num_chunks);
    // long long int oracle_chunk = markedState / (N / num_chunks);

    printf("helooooooooo\n");
    // markedState = markedState % (N / num_chunks);
    // long long int recoveredState = oracle_chunk*(N / num_chunks)+markedState;
    // printf("oracle_chunk: %lld, pos: %lld, recovered: %lld\n", oracle_chunk, markedState, recoveredState);

    // dim3 dimBlock(block_size);
    // dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    // dim3 dimBlock(N_group);

    dim3 dimBlock(1024);
    dim3 dimGrid(num_chunks_per_group);

    printf("dimGrid: %lld, dimBlock: %lld\n", num_chunks_per_group, N_group);

    // Set the gates:
    int num_devices = 1;
    Complex *H_d[num_devices];
    Complex *I_d[num_devices];
    Complex *Z_d[num_devices];
    Complex *X_d[num_devices];
    Complex *X_H_d[num_devices];
    allocateGatesDevice(num_devices, H_d, I_d, Z_d, X_d, X_H_d);






    // TODO make new_idx and old_idx only for num_chunks_per_group
    // for (int i = 0; i < num_chunks; ++i) {
    //      // Init the state
    //     // cudaMallocHost((void **)&state_h[i], chunks[i] * sizeof(Complex));
    //     // // cudaMalloc((void **)&state_d[i], chunks[i] * sizeof(Complex));
    //     // // Init the |0>^(xn) state and the new_state
    //     // state_h[i][0] = make_cuDoubleComplex(1.0, 0.0);
    //     // for (int j = 1; j < chunks[i]; ++j) {
    //     //     state_h[i][j] = make_cuDoubleComplex(0.0, 0.0);
    //     // }
    //     // cudaMemcpy(state_d[i], state_h[i], chunks[i] * sizeof(Complex), cudaMemcpyHostToDevice);

    //     // Malloc the indices on the device
    //     // cudaMalloc(&new_idx_d[i], dimGrid.x * dimBlock.x * qubits_per_chunk * sizeof(int));
    //     // cudaMalloc(&old_idx_d[i], dimGrid.x * dimBlock.x * qubits_per_chunk * sizeof(int));
    //     cudaMalloc(&new_idx_d[i], (int)(pow(2, qubits_per_chunk)) * qubits_per_chunk * sizeof(int));
    //     cudaMalloc(&old_idx_d[i], (int)(pow(2, qubits_per_chunk)) * qubits_per_chunk * sizeof(int));
    // }




    // Init the arrays
    // Complex *state_h[num_chunks];
    // Complex *state_d[num_chunks];
    // printf("Hello there!!!!\n");
    // int *new_idx_d[num_chunks_per_group];
    // int *old_idx_d[num_chunks_per_group];



    // // Assuming we have t = 1 solution in grover's algorithm
    // // we have k = floor(pi/4 * sqrt(N/num_chunks))
    long long int k = (int)floor(M_PI / 4 * sqrt(N/num_chunks));
    printf("running %lld rounds\n", k);


    double time = omp_get_wtime();

    cudaStream_t streams[num_chunks_per_group];

    Complex *state_h[num_chunks_per_group];
    Complex *state_d[num_chunks_per_group];
    printf("Hello there!!!!\n");
    int *new_idx_d[num_chunks_per_group];
    int *old_idx_d[num_chunks_per_group];
    // for (int i = 0; i < num_chunks; ++i) {
    //     cudaCheckError(cudaMalloc(&new_idx_d[i], N_group * qubits_per_chunk * sizeof(int)));
    //     cudaCheckError(cudaMalloc(&old_idx_d[i], N_group * qubits_per_chunk * sizeof(int)));
    //     // cudaCheckError(cudaStreamCreate(&streams[i]));
    //     // cudaMalloc((void **)&state_d[index], chunks[index] * sizeof(Complex));
    // }

    // #pragma omp parallel for collapse(2) num_threads(num_groups * num_chunks_per_group)
    // for (int j = 0; j < num_groups; ++j) {
    //     for (int i = 0; i < num_chunks_per_group; ++i) {
    //         cudaMalloc((void **)&state_d[index], chunks[index] * sizeof(Complex));
    //     }
    // }

    for (int j = 0; j < 1; ++j) {
        // printf("%d / %d\n", j, num_groups);
        // #pragma omp parallel for num_threads(num_groups)
        for (int i = 0; i < num_chunks_per_group; ++i) {
            // cudaStreamCreate(&streams[i]);
            int index = j*num_chunks_per_group+i;
            cudaCheckError(cudaStreamCreate(&streams[i]));
            // tid = omp_get_thread_num();
            // printf("Welcome to GFG from thread = %d\n", tid);


            // printf("j %d, i %d, num_chunks_per_group %d, index %d\n", j, i, num_chunks_per_group, index);

            cudaCheckError(cudaMallocHost((void **)&state_h[i], N_group * sizeof(Complex)));
            state_h[i][0] = make_cuDoubleComplex(1.0, 0.0);
            for (int idx = 1; idx < N_group; ++idx) {
                state_h[i][idx] = make_cuDoubleComplex(0.0, 0.0);
            }

            cudaCheckError(cudaMalloc(&new_idx_d[i], N_group * qubits_per_chunk * sizeof(int)));
            cudaCheckError(cudaMalloc(&old_idx_d[i], N_group * qubits_per_chunk * sizeof(int)));
            cudaCheckError(cudaMalloc((void **)&state_d[i], N_group * sizeof(Complex)));
            cudaCheckError(cudaMemcpyAsync(state_d[i], state_h[i], N_group * sizeof(Complex), cudaMemcpyHostToDevice, streams[i]));
            // contract_tensor<<<num_chunks_per_group, N_group , sharedMemSize, streams[index]>>>(state_d[i], H_d[0], 0, new_idx_d[index], old_idx_d[index], qubits_per_chunk, i*N_group, (i+1)*N_group, index); //(int)(n-log2((double)num_chunks))
            // contract_tensor<<<num_chunks_per_group, N_group , sharedMemSize, streams[index]>>>(state_d[i], H_d[0], 1, new_idx_d[index], old_idx_d[index], qubits_per_chunk, i*N_group, (i+1)*N_group, index); //(int)(n-log2((double)num_chunks))

            // ### Here we run Grover's algorithm
            applyGateAllQubits(
                state_d[i],
                H_d[0], new_idx_d[i],
                old_idx_d[i], qubits_per_chunk,
                dimBlock,
                dimGrid,
                sharedMemSize,
                // i*N_group,
                // (i+1)*N_group,
                0,
                1024,
                streams[i]
            );
            // for (int l = 0; l < k; ++l) {
            //     if (oracle_chunk == (index)) {
            //         // printf("applyPhaseFlip at: %d\n", (index));
            //         applyPhaseFlip<<<dimGrid, dimBlock, 0, streams[i]>>>(state_d[i], markedState);
            //     }

            //     applyDiffusionOperator(
            //         state_d[i],
            //         X_H_d[0], H_d[0], X_d[0], Z_d[0], new_idx_d[i],
            //         old_idx_d[i], qubits_per_chunk, dimBlock, dimGrid, sharedMemSize,
            //         i*N_group, (i+1)*N_group,
            //         streams[i]
            //     );
            // }
            cudaCheckError(cudaMemcpyAsync(state_h[i], state_d[i], N_group * sizeof(Complex), cudaMemcpyDeviceToHost, streams[i]));
            cudaCheckError(cudaStreamSynchronize(streams[i]));
            // cudaCheckError(cudaFree(old_idx_d[i]));
            // cudaCheckError(cudaFree(new_idx_d[i]));
            cudaCheckError(cudaFree(state_d[i]));
            // cudaCheckError(cudaFreeHost(state_h[i]));
            cudaCheckError(cudaStreamDestroy(streams[i]));
        }

        // for (int i = 0; i < num_chunks_per_group; ++i){
        //     printf("chunk id: %d ######################################\n", i);
        //     printState(state_h[i], N_group, "Initial state");
        //     // cudaFree(state_d[j*2 + i]);
        //     // cudaFree(old_idx_d[j*2 + i]);
        //     // cudaFree(new_idx_d[j*2 + i]);
        //     // cudaFreeHost(state_h[j*2 + i]);
        // }
    }

    // for (int i = 0; i < num_chunks_per_group; ++i) {

    //     // cudaStreamSynchronize(streams[i]);
    //     // cudaFree(state_d[index]);
    //     cudaCheckError(cudaStreamDestroy(streams[i]));
    // }

    double elapsed = omp_get_wtime() - time;
    printf("Time: %f \n", elapsed);


    // cudaMemcpy(state_h, state_d, N * sizeof(Complex), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < num_chunks; ++i) {
        // printf("chunk id: %d ######################################\n", i);
    printState(state_h[num_chunks_per_group-1], N_group, "Initial state");
    // }





    for (int i = 0; i < num_devices; ++i) {
        cudaFree(H_d[i]);
        cudaFree(I_d[i]);
        cudaFree(Z_d[i]);
        cudaFree(X_d[i]);
        cudaFree(X_H_d[i]);
    }


    // for (int i = 0; i < num_chunks; ++i) {
    //     // cudaFree(state_d[i]);
    //     // cudaFreeHost(state_h[i]);
    //     cudaFree(new_idx_d[i]);
    //     cudaFree(old_idx_d[i]);
    // }

    return 0;
}
