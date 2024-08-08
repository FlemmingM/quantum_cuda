#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include "utils_cuda_baseline.h"

typedef cuDoubleComplex Complex;

int main(int argc, char* argv[]) {

    int n = atoi(argv[1]);
    long long int N = (long long int)pow(2, n);
    long long int markedState = atoi(argv[2]);
    int dim_block = atoi(argv[3]);

    if (markedState > (N-1)) {
        fprintf(stderr, "You chose a markedState %d but the largest state possible is state %d", markedState, (N-1));
        return 1;
    }


    double time = omp_get_wtime();
    double time2 = omp_get_wtime();

    // Define the gates
    cuDoubleComplex H_h[4] = {
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0),
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(-1.0 / sqrt(2.0), 0.0)
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
    Complex *new_state_h;
    Complex *new_state_d;
    Complex *H_d;
    Complex *I_d;
    Complex *Z_d;
    Complex *X_d;

    int *shape_h;
    int *shape_d;
    int *new_idx_d;
    int *old_idx_d;

    // Init the temp new state for the results
    cudaMallocHost((void **)&new_state_h, N * sizeof(Complex));
    cudaMalloc((void **)&new_state_d, N * sizeof(Complex));
    for (int i = 0; i < N; ++i) {
        new_state_h[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    cudaMemcpy(new_state_d, new_state_h, N * sizeof(Complex), cudaMemcpyHostToDevice);

    // We don't need it in on the host
    cudaFreeHost(new_state_h);


    // Init the state
    cudaMallocHost((void **)&state_h, N * sizeof(Complex));
    cudaMalloc((void **)&state_d, N * sizeof(Complex));
    // Init the |0>^(xn) state and the new_state
    state_h[0] = make_cuDoubleComplex(1.0, 0.0);
    for (int i = 1; i < N; ++i) {
        state_h[i] = make_cuDoubleComplex(0.0, 0.0);
    }
    cudaMemcpy(state_d, state_h, N * sizeof(Complex), cudaMemcpyHostToDevice);

    cudaMallocHost((void **)&shape_h, n * sizeof(int));
    cudaMalloc((void **)&shape_d, n * sizeof(int));

    // Malloc the gate on device
    cudaMalloc((void **)&H_d, 4 * sizeof(Complex));
    cudaMalloc((void **)&I_d, 4 * sizeof(Complex));
    cudaMalloc((void **)&Z_d, 4 * sizeof(Complex));
    cudaMalloc((void **)&X_d, 4 * sizeof(Complex));

    // Init the shape depending on the number of qubits
    // each qubit is a column vector of size 2
    // e.g. |0> = [1, 0]
    // Thus, for n=3 qubits (N=8) the tensor will have a shape of 2,2,2
    for (int i = 0; i < n; ++i) {
        shape_h[i] = 2;
    }

    // Copy from host to device
    cudaMemcpy(shape_d, shape_h, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(H_d, H_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(I_d, I_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(Z_d, Z_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(X_d, X_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);


    dim3 dimBlock(dim_block);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    // Malloc the indices on the device
    cudaMalloc(&new_idx_d, dimGrid.x * dimBlock.x * n * sizeof(int));
    cudaMalloc(&old_idx_d, dimGrid.x * dimBlock.x * n * sizeof(int));


    double elapsed2 = omp_get_wtime() - time2;
    time2 = omp_get_wtime();

    // Assuming we have t = 1 solution in grover's algorithm
    // we have k = floor(pi/4 * sqrt(N))
    int k = (int)floor(M_PI / 4 * sqrt(N));

    // Now apply the H gate n times, once for each qubit
    applyGateAllQubits(state_d, H_d, new_state_d, shape_d, new_idx_d, old_idx_d, n, N, dimBlock, dimGrid);

    for (int i = 0; i < k; ++i) {
        applyPhaseFlip<<<dimGrid, dimBlock>>>(state_d, markedState);
        applyDiffusionOperator(state_d, new_state_d, shape_d, H_d, X_d, Z_d, new_idx_d, old_idx_d, n, N, dimBlock, dimGrid);
    }

    cudaDeviceSynchronize();

    cudaMemcpy(state_h, state_d, N * sizeof(Complex), cudaMemcpyDeviceToHost);

    double elapsed = omp_get_wtime() - time;
    double elapsed3 = omp_get_wtime() - time2;
    // printf("Time: %f \n", elapsed);
    // printf("Time memory: %f \n", elapsed2);
    // printf("Time compute: %f \n", elapsed3);
    printf("%d,%d,%d,%f,%f,%f\n",n, markedState, dim_block, elapsed, elapsed2, elapsed3);


    // printState(state_h, N, "Initial state");

    cudaFree(state_d);
    cudaFree(new_state_d);
    cudaFree(shape_d);
    cudaFree(H_d);
    cudaFreeHost(state_h);

    cudaFreeHost(shape_h);

    cudaFreeHost(H_h);
    cudaFreeHost(I_h);
    cudaFreeHost(Z_h);
    cudaFreeHost(X_h);

    return 0;
}
