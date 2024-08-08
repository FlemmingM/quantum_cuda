#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "utils_omp.h"
#include <omp.h>


// Define a complex number type for simplicity
typedef double complex Complex;

int main(int argc, char* argv[]) {

    int n = atoi(argv[1]);
    int N = (int)pow(2, n);
    int markedState = atoi(argv[2]);

    if (markedState > (N-1)) {
        fprintf(stderr, "You chose a markedState %d but the largest state possible is state %d", markedState, (N-1));
        return 1;
    }

    // Assuming we have t = 1 solution in grover's algorithm
    // we have k = floor(pi/4 * sqrt(N)) iterations

    double time = omp_get_wtime();
    double time2 = omp_get_wtime();
    int k = (int)floor(M_PI / 4 * sqrt(N));

    // Define the gates
    Complex H[2][2] = {
        {1 / sqrt(2.0) + 0.0*I, 1 / sqrt(2.0) + 0.0*I},
        {1 / sqrt(2.0) + 0.0*I, -1 / sqrt(2.0) + 0.0*I}
    };

    Complex I_gate[2][2] = {
        {1 + 0.0*I, 0 + 0.0*I},
        {0 + 0.0*I, 1 + 0.0*I}
    };

    Complex Z[2][2] = {
        {1 + 0.0*I, 0 + 0.0*I},
        {0 + 0.0*I, -1 + 0.0*I}
    };

    Complex X[2][2] = {
        {0 + 0.0*I, 1 + 0.0*I},
        {1 + 0.0*I, 0 + 0.0*I}
    };

    // Dynamically allocate an array of size n with 2 dimensions each for each qubit
    // Needed to initiate the tensor shape

    int* shape = (int*)malloc(n * sizeof(int));
    for (int i = 0; i < n; ++i) {
        shape[i] = 2;
    }

    // Initialize the state tensors
    Complex* state = (Complex*)malloc(N * sizeof(Complex));
    Complex* new_state = (Complex*)malloc(N * sizeof(Complex));


    // Init a superposition of qubits
    state[0] = 1.0 + 0.0*I;
    for (int i = 1; i < N; ++i) {
        state[i] = 0.0 + 0.0*I;
    }

    double elapsed2 = omp_get_wtime() - time2;

    time2 = omp_get_wtime();
    // Now apply the H gate n times, once for each qubit
    for (int i = 0; i < n; ++i) {
        applyGateSingleQubit(state, H, new_state, shape, n, N, i);
    }


    for (int i = 0; i < k; ++i) {
        // Apply Oracle
        applyPhaseFlip(state, markedState);
        // Apply the diffusion operator
        applyDiffusionOperator(state, new_state, shape, H, X, Z, n, N);
    }

    double elapsed = omp_get_wtime() - time;
    double elapsed3 = omp_get_wtime() - time2;
    printf("%d,%d,%f,%f,%f\n",n, markedState, elapsed, elapsed2, elapsed3);


    free(shape);
    free(state);
    free(new_state);

    return 0;
}
