#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>
#include "utils_cuda.h"

typedef cuDoubleComplex Complex;


// Define a complex number type for simplicity
typedef double complex Complex;

int main(int argc, char* argv[]) {

    // collect input args
    if (argc < 6) {
        fprintf(stderr, "Usage: %s n qubits<int>; marked state<int>; number of samples<int>; fileName<string>; verbose 0 or 1<int>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    int N = (int)pow(2, n);
    int markedState = atoi(argv[2]);
    int numSamples = atoi(argv[3]);
    const char* fileName = argv[4];
    int verbose = atoi(argv[5]);

    if (markedState > (N-1)) {
        fprintf(stderr, "You chose a markedState %d but the largest state possible is state %d", markedState, (N-1));
        return 1;
    }

    // Assuming we have t = 1 solution in grover's algorithm
    // we have k = floor(pi/4 * sqrt(N))
    int k = (int)floor(M_PI / 4 * sqrt(N));

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
    int *new_idx;
    int *old_idx;

    // Malloc on device and host
    cudaMallocHost((void **)&state_h, N * sizeof(Complex));
    cudaMalloc((void **)&state_d, N * sizeof(Complex));

    cudaMallocHost((void **)&new_state_h, N * sizeof(Complex));
    cudaMalloc((void **)&new_state_d, N * sizeof(Complex));

    cudaMallocHost((void **)&shape_h, n * sizeof(int));
    cudaMalloc((void **)&shape_d, n * sizeof(int));

    // Malloc the gate on device
    cudaMalloc((void **)&H_d, 4 * sizeof(Complex));
    cudaMalloc((void **)&I_d, 4 * sizeof(Complex));
    cudaMalloc((void **)&Z_d, 4 * sizeof(Complex));
    cudaMalloc((void **)&X_d, 4 * sizeof(Complex));

    // Malloc the indices on the device
    cudaMalloc((void **)&new_idx, n * sizeof(int));
    cudaMalloc((void **)&old_idx, n * sizeof(int));

    // Init a superposition of qubits
    state_h[0] = make_cuDoubleComplex(1.0, 0.0);
    for (int i = 1; i < N; ++i) {
        state_h[i] = make_cuDoubleComplex(0.0, 0.0);
    }

    for (int i = 0; i < N; ++i) {
        new_state_h[i] = make_cuDoubleComplex(0.0, 0.0);
    }

    for (int i = 0; i < N; ++i) {
        printf("state[%d] = (%f, %f)\n", i, cuCreal(state_h[i]), cuCimag(state_h[i]));
    }

    for (int i = 0; i < n; ++i) {
        shape_h[i] = 2;
    }

    cudaMemcpy(state_d, state_h, N * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(new_state_d, new_state_h, N * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_d, shape_h, n * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(H_d, H_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(I_d, I_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(Z_d, Z_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(X_d, X_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);


    dim3 dimBlock(256);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    double time = omp_get_wtime();








    // Now apply the H gate n times, once for each qubit
    for (int i = 0; i < n; ++i) {
        applyGateSingleQubit(state, H, new_state, shape, n, N, i);
    }

    if (verbose == 1) {
        printState(state, N, "Initial state");
    }

    // Apply Grover's algorithm k iteration and then sample
    if (verbose == 1) {
        printf("Running %d round(s)\n", k);
    }

    double time = omp_get_wtime();

    for (int i = 0; i < k; ++i) {
        if (verbose == 1) {
            printf("%d/%d\n", i, k);
        }
        // Apply Oracle
        applyPhaseFlip(state, markedState);
        if (verbose == 1) {
            printState(state, N, "Oracle applied");
        }
        // Apply the diffusion operator
        applyDiffusionOperator(state, new_state, shape, H, X, Z, n, N);
        if (verbose == 1) {
            printState(state, N, "After Diffusion");
        }
    }

    double elapsed = omp_get_wtime() - time;
    printf("Time: %f \n", elapsed);

    // Sample the states wheighted by their amplitudes
    double* averages = simulate(state, N, numSamples);
    if (verbose == 1) {
        printf("Average frequency per position:\n");
        for (int i = 0; i < N; ++i) {
            printf("Position %d: %f\n", i, averages[i]);
        }
    }


    // save the data
    saveArrayToCSV(averages, N, fileName);

    free(averages);
    free(shape);
    free(state);
    free(new_state);

    return 0;
}
