#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include "utils_cuda.h"
#include "utils_cuda_stream.h"


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
    const int num_chunks_per_group = atoi(argv[3]);
    const int num_qubits_per_group = atoi(argv[4]);
    // const int block_size = atoi(argv[4]);
    // const char* fileName = argv[4];
    // int verbose = atoi(argv[5]);

    if (markedState > (N-1)) {
        fprintf(stderr, "You chose a markedState %d but the largest state possible is state %d", markedState, (N-1));
        return 1;
    }


    // Define the number of groups to do the parallel search with more than 10 qubits
    // while still using the fast shared memory

    long long int num_groups = N / pow(2, num_qubits_per_group);
    int num_qubits_per_chunk = num_qubits_per_group - (int)log2(num_chunks_per_group);
    int N_chunk = pow(2, num_qubits_per_chunk);
    long long int num_chunks = num_groups * num_chunks_per_group;

    if (N_chunk > pow(2, 10)) {
        fprintf(stderr, "You chose a number of qubits per group of: %d and a number of chunks per group of: %d\n Change the config so that the number of qubits per chunk is maximally 10 to fit into 1 block", num_qubits_per_group, num_chunks_per_group);
        return 1;
    }

    int sharedMemSize = (int)(pow(2, 11)) * sizeof(Complex);


    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    if (sharedMemSize > deviceProp.sharedMemPerBlock) {
        printf("Requested shared memory size (%d bytes) exceeds the device limit (%d bytes).\n", sharedMemSize, deviceProp.sharedMemPerBlock);
        return -1;
    }


    long long int oracle_chunk = markedState / (N / num_chunks);


    markedState = markedState % (N / num_chunks);
    long long int recoveredState = oracle_chunk*(N / num_chunks)+markedState;


    dim3 dimBlock(N_chunk);
    dim3 dimGrid(1);
    // dim3 dimGrid(num_chunks_per_group);

    int print_val = 0;
    if (print_val == 1) {
        printf("N: %lld\n", N);
        printf("n: %d\n", n);
        printf("num_groups: %lld\n", num_groups);
        printf("num_chunks_per_group: %d\n", num_chunks_per_group);
        printf("num_qubits_per_chunk: %d\n", num_qubits_per_chunk);
        printf("N_chunk: %d\n", N_chunk);
        printf("num_chunks: %lld\n", num_chunks);
        printf("oracle_chunk: %lld, pos: %lld, recovered: %lld\n", oracle_chunk, markedState, recoveredState);
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
    // printf("running %lld rounds\n", k);



    double time = omp_get_wtime();

    cudaStream_t streams[num_chunks_per_group];

    Complex *solution_state_h;
    Complex *state_h[num_chunks_per_group];
    Complex *state_d[num_chunks_per_group];
    int *new_idx_d[num_chunks_per_group];
    int *old_idx_d[num_chunks_per_group];

    // To get the parallel search results
    int *d_maxIndex;
    int *h_maxIndex;
    int *d_chunk_ids;
    int *h_chunk_ids;
    double *d_maxValue;
    double *h_maxValue;
    cudaMalloc(&d_maxIndex, num_chunks_per_group*sizeof(int));
    cudaMallocHost(&h_maxIndex, num_chunks_per_group*sizeof(int));
    cudaMalloc(&d_maxValue, num_chunks_per_group*sizeof(double));
    cudaMallocHost(&h_maxValue, num_chunks_per_group*sizeof(double));
    cudaMalloc(&d_chunk_ids, num_chunks_per_group*sizeof(double));
    cudaMallocHost(&h_chunk_ids, num_chunks_per_group*sizeof(double));


    // init the arrays:



    // Create the streams
    for (int i = 0; i < num_chunks_per_group; ++i) {
        cudaStreamCreate(&streams[i]);
        cudaMallocHost((void **)&state_h[i], N_chunk * sizeof(Complex));
        state_h[i][0] = make_cuDoubleComplex(1.0, 0.0);
        for (int idx = 1; idx < N_chunk; ++idx) {
            state_h[i][idx] = make_cuDoubleComplex(0.0, 0.0);
        }
        cudaMalloc(&new_idx_d[i], N_chunk * num_qubits_per_chunk * sizeof(int));
        cudaMalloc(&old_idx_d[i], N_chunk * num_qubits_per_chunk * sizeof(int));
        cudaMalloc((void **)&state_d[i], N_chunk * sizeof(Complex));
        cudaMemcpyAsync(state_d[i], state_h[i], N_chunk * sizeof(Complex), cudaMemcpyHostToDevice, streams[i]);
    }


    // allocate the solution state:
    cudaMallocHost((void **)&solution_state_h, N_chunk * sizeof(Complex));

    int marked_chunk = -99;
    int marked_max_val = -99;
    int marked_max_idx = -99;
    for (int j = 0; j < num_groups; ++j) {
        // printf("%d / %d\n", j, num_groups);
        // #pragma omp parallel for num_threads(num_chunks_per_group)
        for (int i = 0; i < num_chunks_per_group; ++i) {
            // cudaStreamCreate(&streams[i]);
            int index = j*num_chunks_per_group+i;

            // ### Here we run Grover's algorithm
            // initState<<<dimGrid, dimBlock, 0, streams[i]>>>(state_d[i], N_chunk);
            applyGateAllQubits(
                state_d[i],
                H_d[0], new_idx_d[i],
                old_idx_d[i], num_qubits_per_chunk,
                dimBlock,
                dimGrid,
                sharedMemSize,
                0,
                N_chunk,
                streams[i]
            );
            for (int l = 0; l < k; ++l) {
                if (oracle_chunk == (index)) {
                    // printf("oracle chunk_id: %d\n", index);
                    applyPhaseFlip<<<dimGrid, dimBlock, 0, streams[i]>>>(state_d[i], markedState);
                }

                applyDiffusionOperator(
                    state_d[i],
                    X_H_d[0], H_d[0], X_d[0], Z_d[0], new_idx_d[i],
                    old_idx_d[i], num_qubits_per_chunk, dimBlock, dimGrid, sharedMemSize,
                    0, N_chunk,
                    streams[i]
                );
            }
        }

        for (int i = 0; i < num_chunks_per_group; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        cudaDeviceSynchronize();


        for (int i = 0; i < num_chunks_per_group; ++i){
            findMaxIndexKernel<<<1, N_chunk, 0, streams[i]>>>(state_d[i], d_maxIndex, d_maxValue, N_chunk, i, d_chunk_ids);
            cudaMemcpyAsync(h_maxIndex, d_maxIndex, num_chunks_per_group*sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
            cudaMemcpyAsync(h_chunk_ids, d_chunk_ids, num_chunks_per_group*sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
            cudaMemcpyAsync(h_maxValue, d_maxValue, num_chunks_per_group*sizeof(double), cudaMemcpyDeviceToHost, streams[i]);

        }

        for (int i = 0; i < num_chunks_per_group; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        cudaDeviceSynchronize();


        for (int i = 0; i < num_chunks_per_group; ++i){
            // printf("chunk id: %d, maxIdx: %d, maxVal: %f\n", h_chunk_ids[i], h_maxIndex[i], h_maxValue[i]);
            if(h_maxValue[i] >= 0.5){
                // printf("Solution: chunk id: %d, maxIdx: %d, maxVal: %f\n", h_chunk_ids[i], h_maxIndex[i], h_maxValue[i]);
                marked_chunk = h_chunk_ids[i];
                marked_max_idx = h_maxIndex[i];
                marked_max_val = h_maxValue[i];
                int index = marked_chunk % num_chunks_per_group;

                cudaMemcpyAsync(solution_state_h, state_d[index], N_chunk * sizeof(Complex), cudaMemcpyDeviceToHost, streams[index]);
                cudaStreamSynchronize(streams[index]);
            }
        }
        cudaDeviceSynchronize();

        for (int i = 0; i < num_chunks_per_group; ++i){
            initState<<<dimGrid, dimBlock, 0, streams[i]>>>(state_d[i], N_chunk);
            cudaStreamSynchronize(streams[i]);
        }
        cudaDeviceSynchronize();

    } // end of the out loop

    for (int i = 0; i < num_chunks_per_group; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    double elapsed = omp_get_wtime() - time;
    // printf("Time: %f \n", elapsed);
    // n, k, num_groups, num_chunks, n_per_group, chunks_per_group, num_threads, marked_chunk, markedState, marked_max_idx, marked_max_val, time
    printf("%d,%lld,%lld,%lld,%d,%d,%d,%d,%d,%d,%f,%f\n",
        n, k, num_groups, num_chunks, num_qubits_per_group, num_chunks_per_group, dimBlock.x, marked_chunk, markedState, marked_max_idx, marked_max_val, elapsed);

    // printState(solution_state_h, N_chunk, "Initial state");


    for (int i = 0; i < num_devices; ++i) {
        cudaFree(H_d[i]);
        cudaFree(I_d[i]);
        cudaFree(Z_d[i]);
        cudaFree(X_d[i]);
        cudaFree(X_H_d[i]);
    }


    for (int i = 0; i < num_chunks_per_group; ++i) {
        cudaFree(new_idx_d[i]);
        cudaFree(old_idx_d[i]);
        cudaFree(state_d[i]);
        cudaFreeHost(state_h[i]);


    }

    cudaFreeHost(solution_state_h);
    cudaFree(d_maxIndex);
    cudaFree(d_chunk_ids);
    cudaFree(d_maxValue);
    cudaFreeHost(h_maxIndex);
    cudaFreeHost(h_chunk_ids);
    cudaFreeHost(h_maxValue);

    return 0;
}
