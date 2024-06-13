#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
// #include <omp.h>

typedef double complex Complex;

void printVector(const Complex* state, int n){
    printf("[");
    for (int i = 0; i < n; ++i) {
        printf("(%f + %fi) ", creal(state[i]), cimag(state[i]));
    }
    printf("]\n");
}



void zeroOutArray(Complex* array, int length){
    #pragma omp parallel for
    for (int i = 0; i < length; ++i) {
        array[i] = 0.0 + 0.0*I;
    }
}


void contract_tensor_baseline(const Complex* state,
                     const Complex gate[2][2],
                     int qubit,
                     Complex* new_state,
                     const int* shape, int n) {
    int total_elements = (int)pow(2, n);

    // Zero out new_state
    zeroOutArray(new_state, total_elements);

    // Iterate over all possible indices of the state tensor
    // #pragma omp parallel for
    for (int idx = 0; idx < total_elements; ++idx) {
        printf("idx %d ##############################\n", idx);
        int new_idx[n];
        int old_idx[n];
        int temp = idx;

        // Compute the multi-dimensional index
        for (int i = n - 1; i >= 0; --i) {
            printf("temp %d\n", temp);
            printf("i %d\n", i);
            new_idx[i] = temp % shape[i];
            temp /= shape[i];
        }

        // Perform the tensor contraction for the specified qubit
        // TODO: make the algorithm more generic to work with all dimensions (currently it is 2)
        printf("going over gate idx: \n");
        for (int j = 0; j < 2; ++j) {
            // Copy new_idx to old_idx
            printf("j %d\n", j);
            for (int i = 0; i < n; ++i) {
                printf("inner i  %d\n", i);
                old_idx[i] = new_idx[i];
            }
            old_idx[qubit] = j;

            printf("old idx: \n");
            printf("\n");
            // Compute the linear index for old_idx
            // the factor is used as the stride of the linear index
            int old_linear_idx = 0;
            int factor = 1;
            for (int i = n - 1; i >= 0; --i) {
                old_linear_idx += old_idx[i] * factor;
                factor *= shape[i];
            }

            printf("old_linear_idx %d\n factor %d\n new_idx_q %d\n", old_linear_idx, factor, new_idx[qubit]);

            new_state[idx] += gate[new_idx[qubit]][j] * state[old_linear_idx];

            printVector(new_state, total_elements);
        }
    }
}

int main() {
    int n = 3;
    int N = (int)pow(2, n);

    Complex H[2][2] = {
        {1 / sqrt(2.0) + 0.0*I, 1 / sqrt(2.0) + 0.0*I},
        {1 / sqrt(2.0) + 0.0*I, -1 / sqrt(2.0) + 0.0*I}
    };

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

    printVector(state, N);

    contract_tensor_baseline(state, H, 0, new_state, shape, n);

    printVector(state, N);
    printVector(new_state, N);

    free(shape);
    free(state);
    free(new_state);

    return 0;
}
