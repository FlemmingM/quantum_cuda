#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include "utils_cuda.h"
#include "utils_cuda_stream_2_gpu.h"


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
    int num_devices = 2;
    long long int num_groups = N / pow(2, num_qubits_per_group);
    int num_qubits_per_chunk = num_qubits_per_group - (int)log2(num_chunks_per_group);
    int N_chunk = pow(2, num_qubits_per_chunk);
    long long int num_chunks = num_groups * num_chunks_per_group;
    printf("N: %lld\n", N);
    printf("n: %d\n", n);
    printf("num_devices: %d\n", num_devices);
    printf("num_groups: %lld\n", num_groups);
    printf("num_chunks_per_group: %d\n", num_chunks_per_group);
    printf("num_qubits_per_chunk: %d\n", num_qubits_per_chunk);
    printf("N_chunk: %d\n", N_chunk);
    printf("num_chunks: %lld\n", num_chunks);

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
    printf("oracle_chunk: %lld, pos: %lld, recovered: %lld\n", oracle_chunk, markedState, recoveredState);


    dim3 dimBlock(N_chunk);
    dim3 dimGrid(num_chunks_per_group);

    printf("dimGrid: %d, dimBlock: %d\n", dimGrid.x, dimBlock.x);

    // Set the gates:
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

    cudaStream_t streams[num_chunks_per_group];

    Complex *solution_state_h[num_devices];
    Complex *state_h[num_chunks_per_group];
    Complex *state_d[num_chunks_per_group];
    int *new_idx_d[num_chunks_per_group];
    int *old_idx_d[num_chunks_per_group];

    // To get the parallel search results
    int *d_maxIndex[num_devices];
    int *h_maxIndex[num_devices];
    int *d_chunk_ids[num_devices];
    int *h_chunk_ids[num_devices];
    double *d_maxValue[num_devices];
    double *h_maxValue[num_devices];

    for (int i = 0; i < num_devices; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&d_maxIndex[i], num_chunks_per_group / 2*sizeof(int));
        cudaMallocHost(&h_maxIndex[i], num_chunks_per_group / 2*sizeof(int));
        cudaMalloc(&d_maxValue[i], num_chunks_per_group / 2*sizeof(double));
        cudaMallocHost(&h_maxValue[i], num_chunks_per_group / 2*sizeof(double));
        cudaMalloc(&d_chunk_ids[i], num_chunks_per_group / 2*sizeof(double));
        cudaMallocHost(&h_chunk_ids[i], num_chunks_per_group / 2*sizeof(double));
        // allocate the solution state:
        cudaMallocHost((void **)&solution_state_h[i], N_chunk * sizeof(Complex));
    }

    // Create the streams
    for (int i = 0; i < num_chunks_per_group; ++i) {
        int device_id = i % num_devices;
        // printf("device_id: %d +++++++++++++++++++\n", device_id);
        cudaSetDevice(device_id);

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

    int solution_device_id = -1;
    int marked_chunk = -99;
    for (int j = 0; j < num_groups; ++j) {
        // printf("%d / %d\n", j, num_groups);
        // #pragma omp parallel for num_threads(num_chunks_per_group)
        for (int i = 0; i < num_chunks_per_group; ++i) {
            int device_id = i % num_devices;
            cudaSetDevice(device_id);
            int index = j*num_chunks_per_group+i;

            // ### Here we run Grover's algorithm
            // initState<<<dimGrid, dimBlock, 0, streams[i]>>>(state_d[i], N_chunk);
            applyGateAllQubits(
                state_d[i],
                H_d[device_id], new_idx_d[i],
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
                    X_H_d[device_id], H_d[device_id], X_d[device_id], Z_d[device_id], new_idx_d[i],
                    old_idx_d[i], num_qubits_per_chunk, dimBlock, dimGrid, sharedMemSize,
                    0, N_chunk,
                    streams[i]
                );
            }
        }

        for (int i = 0; i < num_chunks_per_group; ++i) {
            int device_id = i % num_devices;
            cudaSetDevice(device_id);
            cudaStreamSynchronize(streams[i]);
        }
        cudaDeviceSynchronize();


        for (int i = 0; i < num_chunks_per_group; ++i){
            int device_id = i % num_devices;
            cudaSetDevice(device_id);
            findMaxIndexKernel<<<1, N_chunk, 0, streams[i]>>>(state_d[i], d_maxIndex[device_id], d_maxValue[device_id], N_chunk, i, d_chunk_ids[device_id]);
            cudaMemcpyAsync(h_maxIndex[device_id], d_maxIndex[device_id], num_chunks_per_group / 2*sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
            cudaMemcpyAsync(h_chunk_ids[device_id], d_chunk_ids[device_id], num_chunks_per_group / 2*sizeof(int), cudaMemcpyDeviceToHost, streams[i]);
            cudaMemcpyAsync(h_maxValue[device_id], d_maxValue[device_id], num_chunks_per_group / 2*sizeof(double), cudaMemcpyDeviceToHost, streams[i]);
            // cudaMemcpyAsync(state_h[i], state_d[i], N_chunk*sizeof(Complex), cudaMemcpyDeviceToHost, streams[i]);
        }

        for (int i = 0; i < num_chunks_per_group; ++i) {
            int device_id = i % num_devices;
            cudaSetDevice(device_id);
            cudaStreamSynchronize(streams[i]);
        }
        cudaDeviceSynchronize();



        for (int i = 0; i < num_chunks_per_group; ++i){
            int device_id = i % num_devices;
            cudaSetDevice(device_id);
            // printf("chunk id: %d, maxIdx: %d, maxVal: %f\n", h_chunk_ids[i][i/2], h_maxIndex[i][i/2], h_maxValue[i][i/2]);
            // printf("chunk id: %d, maxIdx: %d, maxVal: %f\n", h_chunk_ids[i][1], h_maxIndex[i][1], h_maxValue[i][1]);

            if(h_maxValue[device_id][i/2] >= 0.7){
                printf("chunk id: %d, maxIdx: %d, maxVal: %f\n", h_chunk_ids[device_id][i/2], h_maxIndex[device_id][i/2], h_maxValue[device_id][i/2]);
                marked_chunk = h_chunk_ids[device_id][i];
                int index = marked_chunk % num_chunks_per_group;

                solution_device_id = device_id;
                cudaMemcpyAsync(solution_state_h[device_id], state_d[index], N_chunk * sizeof(Complex), cudaMemcpyDeviceToHost, streams[index]);
                cudaStreamSynchronize(streams[index]);
            }
        }

        cudaDeviceSynchronize();

        for (int i = 0; i < num_chunks_per_group; ++i){
            int device_id = i % num_devices;
            cudaSetDevice(device_id);
            initState<<<dimGrid, dimBlock, 0, streams[i]>>>(state_d[i], N_chunk);
            cudaStreamSynchronize(streams[i]);
        }
        cudaDeviceSynchronize();
    } // end of the out loop

    for (int i = 0; i < num_chunks_per_group; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    double elapsed = omp_get_wtime() - time;
    printf("Time: %f \n", elapsed);


    // for (int i = 0; i < num_chunks_per_group; ++i) {
    //     printState(state_h[i], N_chunk, "Initial state");
    // }

    // printState(solution_state_h[solution_device_id], N_chunk, "Solution state");



    for (int i = 0; i < num_devices; ++i) {
        cudaFree(H_d[i]);
        cudaFree(I_d[i]);
        cudaFree(Z_d[i]);
        cudaFree(X_d[i]);
        cudaFree(X_H_d[i]);

        cudaFreeHost(solution_state_h[i]);
        cudaFree(d_maxIndex[i]);
        cudaFree(d_chunk_ids[i]);
        cudaFree(d_maxValue[i]);
        cudaFreeHost(h_maxIndex[i]);
        cudaFreeHost(h_chunk_ids[i]);
        cudaFreeHost(h_maxValue[i]);
    }


    for (int i = 0; i < num_chunks_per_group; ++i) {
        cudaFree(new_idx_d[i]);
        cudaFree(old_idx_d[i]);
        cudaFree(state_d[i]);
        cudaFreeHost(state_h[i]);
    }



    return 0;
}
