#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include "utils_cuda.h"
#include "utils_cuda_v3.h"


typedef cuDoubleComplex Complex;



int main(int argc, char* argv[]) {

    // collect input args
    // if (argc < 6) {
    //     fprintf(stderr, "Usage: %s n qubits<int>; marked state<int>; number of samples<int>; fileName<string>; verbose 0 or 1<int>\n", argv[0]);
    //     return 1;
    // }

    int n = atoi(argv[1]);
    long long int N = pow(2, n);
    long long int markedState = atoi(argv[2]);
    const int num_chunks_per_group = atoi(argv[3]);
    const int num_qubits_per_group = atoi(argv[4]);

    if (markedState > (N-1)) {
        fprintf(stderr, "You chose a markedState %d but the largest state possible is state %d", markedState, (N-1));
        return 1;
    }


    // Define the number of groups to do the parallel search with more than 10 qubits
    // while still using the fast shared memory

    long long int num_groups = N / pow(2, num_qubits_per_group);
    int num_qubits_per_chunk = num_qubits_per_group - (int)log2(num_chunks_per_group);
    long long int N_chunk = pow(2, num_qubits_per_chunk);
    long long int N_group = num_chunks_per_group * N_chunk;
    long long int num_chunks = num_groups * num_chunks_per_group;

    if (N_chunk > pow(2, 10)) {
        fprintf(stderr, "You chose a number of qubits per group of: %d and a number of chunks per group of: %d\n Change the config so that the number of qubits per chunk is maximally 10 to fit into 1 block", num_qubits_per_group, num_chunks_per_group);
        return 1;
    }

    int sharedMemSize = N_chunk * sizeof(Complex);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (sharedMemSize > deviceProp.sharedMemPerBlock) {
        printf("Requested shared memory size (%d bytes) exceeds the device limit (%d bytes).\n", sharedMemSize, deviceProp.sharedMemPerBlock);
        return -1;
    }


    long long int oracle_group = markedState / (N / num_groups);


    markedState = markedState % (N / num_groups);
    long long int recoveredState = oracle_group*(N / num_groups)+markedState;


    dim3 dimBlock(N_chunk);
    dim3 dimBlock2(2*N_chunk);
    dim3 dimGrid(num_chunks_per_group);


    int print_val = 1;
    if (print_val == 1) {
        printf("N: %lld\n", N);
        printf("n: %d\n", n);
        printf("num_groups: %lld\n", num_groups);
        printf("num_chunks_per_group: %d\n", num_chunks_per_group);
        printf("num_qubits_per_chunk: %d\n", num_qubits_per_chunk);
        printf("N_chunk: %lld\n", N_chunk);
        printf("N_group: %lld\n", N_group);
        printf("num_chunks: %lld\n", num_chunks);
        printf("oracle_group: %lld, pos: %lld, recovered: %lld\n", oracle_group, markedState, recoveredState);
        printf("dimGrid: %d, dimBlock: %d\n", dimGrid.x, dimBlock.x);
    }


    // Set the gates:
    int num_devices = 1;
    Complex *H_d[num_devices];
    Complex *I_d[num_devices];
    Complex *Z_d[num_devices];
    Complex *X_d[num_devices];
    Complex *X_H_d[num_devices];
    allocateGatesDevice(num_devices, H_d, I_d, Z_d, X_d, X_H_d);



    // // Assuming we have t = 1 solution in grover's algorithm
    // // we have k = floor(pi/4 * sqrt(N/num_chunks))
    long long int k = (int)floor(M_PI / 4 * sqrt(N/num_chunks));
    printf("running %lld rounds\n", k);



    double time = omp_get_wtime();

    Complex *state_h;
    Complex *state_d;
    int *new_idx_d;
    int *new_idx_h;
    int *old_idx_d;
    int *old_linear_idxs_d;
    int *old_linear_idxs_h;

    // for indices
    int sharedMemSize2 = 2*N_chunk * sizeof(int);

    cudaMallocHost(&old_linear_idxs_h, 2 * N_chunk * num_qubits_per_chunk * sizeof(int));
    cudaMalloc(&old_linear_idxs_d, 2 * N_chunk * num_qubits_per_chunk * sizeof(int));


    // init the arrays:
    cudaMallocHost((void **)&state_h, N_group * sizeof(Complex));
    // for (int i = 0; i < N_group; ++i) {
    //     if ((i % N_chunk)==0) {
    //         state_h[i] = make_cuDoubleComplex(1.0, 0.0);
    //     } else {
    //         state_h[i] = make_cuDoubleComplex(0.0, 0.0);
    //     }
    // }

    cudaMalloc((void **)&state_d, N_group * sizeof(Complex));
    cudaMalloc(&new_idx_d, N_group * n * sizeof(int));
    cudaMalloc(&old_idx_d, N_group * n * sizeof(int));
    // cudaMemcpy(state_d, state_h, N_group * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMallocHost(&new_idx_h, N_chunk * num_qubits_per_chunk * sizeof(int));



    for (int i = 0; i < num_qubits_per_chunk; ++i) {
        compute_idx<<<1, dimBlock, sharedMemSize2>>>(i, new_idx_d, old_idx_d, num_qubits_per_chunk, N_chunk, old_linear_idxs_d);
    }

    // cudaMemcpy(old_linear_idxs_h, old_linear_idxs_d, 2*N_chunk* num_qubits_per_chunk * sizeof(int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < (2*N_chunk*num_qubits_per_chunk); ++i) {
    //     printf("%d ", old_linear_idxs_h[i]);
    // }
    // printf("\n");

    // cudaMemcpy(new_idx_h, new_idx_d, N_chunk * num_qubits_per_chunk * sizeof(int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < (N_chunk * num_qubits_per_chunk); ++i) {
    //     printf("%d ", new_idx_h[i]);
    // }
    // printf("\n");

    // initStateParallel<<<dimGrid, dimBlock>>>(state_d, N_group, N_chunk);
    // contract_tensor<<<dimGrid, dimBlock, sharedMemSize>>>(state_d, H_d[0], 0, new_idx_d, n, N, old_linear_idxs_d);
    // contract_tensor<<<dimGrid, dimBlock, sharedMemSize>>>(state_d, H_d[0], 1, new_idx_d, n, N, old_linear_idxs_d);
    // contract_tensor<<<dimGrid, dimBlock, sharedMemSize>>>(state_d, H_d[0], 2, new_idx_d, n, N, old_linear_idxs_d);


    // applyGateAllQubits(
    //     state_d,
    //     H_d[0], new_idx_d,
    //     num_qubits_per_chunk,
    //     dimBlock,
    //     dimGrid,
    //     sharedMemSize,
    //     N_group,
    //     old_linear_idxs_d
    // );


    // cudaMemcpy(state_h, state_d, N_group * sizeof(Complex), cudaMemcpyDeviceToHost);
    // printState(state_h, N_group, "state end");

double time2 = omp_get_wtime();
// for (int i = 0; i < num_qubits_per_chunk; ++i) {
//         compute_idx<<<1, dimBlock, sharedMemSize2>>>(i, new_idx_d, old_idx_d, num_qubits_per_chunk, N_chunk, old_linear_idxs_d);
//     }

// initStateParallel<<<dimGrid, dimBlock>>>(state_d, N_group, N_chunk);
// contract_tensor<<<dimGrid, dimBlock2, 2*sharedMemSize>>>(state_d, H_d[0], 0, new_idx_d, num_qubits_per_chunk, 2*N_group, old_linear_idxs_d);
// contract_tensor<<<dimGrid, 16, 2*sharedMemSize>>>(state_d, H_d[0], 1, new_idx_d, num_qubits_per_chunk, 2*N_group, old_linear_idxs_d);
// contract_tensor<<<dimGrid, 16, 2*sharedMemSize>>>(state_d, H_d[0], 2, new_idx_d, num_qubits_per_chunk, 2*N_group, old_linear_idxs_d);


// applyGateAllQubits(
//         state_d,
//         H_d[0], new_idx_d,
//         num_qubits_per_chunk,
//         dimBlock,
//         dimGrid,
//         sharedMemSize,
//         N_group,
//         old_linear_idxs_d
//     );


for (int i = 0; i < num_groups; ++i) {
    // reset the state vector for the next group
    initStateParallel<<<dimGrid, dimBlock>>>(state_d, N_group, N_chunk);

    applyGateAllQubits(
        state_d,
        H_d[0], new_idx_d,
        num_qubits_per_chunk,
        dimBlock2,
        dimGrid,
        2*sharedMemSize,
        2*N_group,
        old_linear_idxs_d
    );

    // applyPhaseFlip<<<dimGrid, dimBlock, 0>>>(state_d, markedState);
    // printf("\n");
    // applyGateAllQubits(
    //     state_d,
    //     X_H_d[0], new_idx_d,
    //     num_qubits_per_chunk,
    //     dimBlock2,
    //     dimGrid,
    //     2*sharedMemSize,
    //     2*N_group,
    //     old_linear_idxs_d
    // );

    for (int l = 0; l < k; ++l) {
        if (i == oracle_group) {
            // printf("oracle chunk_id: %lld, i: %d\n", oracle_group, i);
            applyPhaseFlip<<<dimGrid, dimBlock2, 0>>>(state_d, markedState);
        }

        applyDiffusionOperator(
            state_d,
            X_H_d[0], H_d[0], X_d[0], Z_d[0], new_idx_d,
            num_qubits_per_chunk, dimBlock2, dimGrid, 2*sharedMemSize,
            num_chunks_per_group,
            N_chunk,
            2*N_group,
            old_linear_idxs_d
        );
    }

    if (i == oracle_group) {
        cudaMemcpy(state_h, state_d, N_group * sizeof(Complex), cudaMemcpyDeviceToHost);
        // printState(state_h, N_group, "state end");
    }
    cudaDeviceSynchronize();
}

    // cudaMemcpy(state_h, state_d, N_group * sizeof(Complex), cudaMemcpyDeviceToHost);
    // printState(state_h, N_group, "state end");

    double elapsed2 = omp_get_wtime() - time2;
    printf("Time compute: %f \n", elapsed2);
    double elapsed = omp_get_wtime() - time;
    printf("Time: %f \n", elapsed);
    // // n, k, num_groups, num_chunks, n_per_group, chunks_per_group, num_threads, marked_chunk, markedState, marked_max_idx, marked_max_val, time
    printf("%d,%lld,%lld,%lld,%d,%d,%d,%d,%f\n",
        n, k, num_groups, num_chunks, num_qubits_per_group, num_chunks_per_group, dimBlock.x, markedState, elapsed);


    cudaFree(H_d[0]);
    cudaFree(I_d[0]);
    cudaFree(Z_d[0]);
    cudaFree(X_d[0]);
    cudaFree(X_H_d[0]);

    cudaFree(new_idx_d);
    cudaFree(old_idx_d);
    cudaFree(state_d);
    cudaFreeHost(state_h);

    return 0;
}
