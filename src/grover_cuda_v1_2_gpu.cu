#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include "utils_cuda.h"
#include "utils_cuda_v1_2_gpu.h"


typedef cuDoubleComplex Complex;



int main(int argc, char* argv[]) {

    int num_vis_devices = 0;
    cudaError_t err = cudaGetDeviceCount(&num_vis_devices);

    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    printf("Number of visible CUDA devices: %d\n", num_vis_devices);

    int n = atoi(argv[1]);
    long long int N = (long long int)pow(2, n);
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
    if (num_groups < 2) {
        fprintf(stderr, "For 2 GPUs we need at least 2 groups!");
        return 1;
    }

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
    dim3 dimGrid(num_chunks_per_group);
    // dim3 dimGrid(num_chunks_per_group);


    int print_val = 0;
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

    // // Assuming we have t = 1 solution in grover's algorithm
    // // we have k = floor(pi/4 * sqrt(N/num_chunks))
    long long int k = (int)floor(M_PI / 4 * sqrt(N/num_chunks));
    // printf("running %lld rounds\n", k);

    double time = omp_get_wtime();
    double time2 = omp_get_wtime();

    // Set the gates:
    int num_devices = 2;
    Complex *H_d[num_devices];
    Complex *I_d[num_devices];
    Complex *Z_d[num_devices];
    Complex *X_d[num_devices];
    Complex *X_H_d[num_devices];
    allocateGatesDevice(num_devices, H_d, I_d, Z_d, X_d, X_H_d);


    cudaStream_t streams[num_devices];

    Complex *state_h[num_devices];
    Complex *state_d[num_devices];
    int *new_idx_d[num_devices];
    int *old_idx_d[num_devices];


    // int chunk_id = -1;
    // init the arrays:
    #pragma omp parallel for num_threads(2)
    for (int j = 0; j < num_devices; ++j) {
        cudaSetDevice(j);

        cudaStreamCreate(&streams[j]);
        cudaMallocHost((void **)&state_h[j], N_group * sizeof(Complex));
        cudaMalloc((void **)&state_d[j], N_group * sizeof(Complex));
        cudaMalloc(&new_idx_d[j], N_group * n * sizeof(int));
        cudaMalloc(&old_idx_d[j], N_group * n * sizeof(int));
    }

    double elapsed2 = omp_get_wtime() - time2;
    time2 = omp_get_wtime();

    for (int i = 0; i < num_groups/2; ++i) {
        #pragma omp parallel for num_threads(2)
        for (int h = 0; h < num_devices; ++h) {
            // reset the state vector for the next group
            int index = i*num_devices + h;
            int device_id = h;
            cudaSetDevice(device_id);

            initStateParallel<<<dimGrid, dimBlock, 0, streams[device_id]>>>(state_d[device_id], N_group, N_chunk);

            applyGateAllQubits(
                state_d[device_id],
                H_d[device_id], new_idx_d[device_id],
                old_idx_d[device_id], num_qubits_per_chunk,
                dimBlock,
                dimGrid,
                2*sharedMemSize,
                0,
                N_group,
                streams[device_id]
            );

            for (int l = 0; l < k; ++l) {
                if (index == oracle_group) {
                    // printf("oracle chunk_id: %lld, i: %d\n", oracle_group, i);
                    applyPhaseFlip<<<dimGrid, dimBlock, 0, streams[device_id]>>>(state_d[device_id], markedState);
                }

                applyDiffusionOperator(
                    state_d[device_id],
                    X_H_d[device_id], H_d[device_id], X_d[device_id], Z_d[device_id], new_idx_d[device_id],
                    old_idx_d[device_id], num_qubits_per_chunk, dimBlock, dimGrid, 2*sharedMemSize,
                    num_chunks_per_group,
                    N_chunk,
                    0, N_group,
                    streams[device_id]
                );
            }

            if (index == oracle_group) {
                // chunk_id = device_id;
                double time4 = omp_get_wtime();
                cudaMemcpyAsync(state_h[device_id], state_d[device_id], N_group * sizeof(Complex), cudaMemcpyDeviceToHost, streams[device_id]);
                elapsed2 += omp_get_wtime() - time4;
            }

        }
        #pragma omp parallel for num_threads(2)
        for (int i = 0; i < num_devices; ++i) {
            cudaSetDevice(i);
            cudaStreamSynchronize(streams[i]);
            cudaDeviceSynchronize();
        }
    }

    // printState(state_h[chunk_id], N_group, "state end");


    double elapsed = omp_get_wtime() - time;
    double elapsed3 = omp_get_wtime() - time2;
    // printf("Time: %f \n", elapsed);
    // // n, k, num_groups, num_chunks, n_per_group, chunks_per_group, num_threads, marked_chunk, markedState, marked_max_idx, marked_max_val, time
    printf("%d,%lld,%lld,%lld,%d,%d,%d,%d,%f,%f,%f\n",
        n, k, num_groups, num_chunks, num_qubits_per_group, num_chunks_per_group, dimBlock.x, markedState, elapsed, elapsed2, elapsed3);


    for (int i = 0; i < num_devices; ++i) {
        cudaFree(H_d[i]);
        cudaFree(I_d[i]);
        cudaFree(Z_d[i]);
        cudaFree(X_d[i]);
        cudaFree(X_H_d[i]);
        cudaFree(new_idx_d[i]);
        cudaFree(old_idx_d[i]);
        cudaFree(state_d[i]);
        cudaFreeHost(state_h[i]);
    }
    return 0;
}
