#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include <errno.h>

typedef double complex Complex;

void saveArrayToCSV(const double *array, int N, const char* filename) {
    FILE *file = fopen(filename, "w");

    if (!file) {
        perror("Unable to open file");
        return;
    }
    fprintf(file, "position,probability\n");
    for (int i = 0; i < N; ++i) {
        fprintf(file, "pos%d,%f\n", i, array[i]);
    }
    fclose(file);
}

void contract_tensor(const Complex* state,
                     const Complex gate[2][2],
                     int qubit,
                     Complex* new_state,
                     const int* shape, int n) {
    int total_elements = (int)pow(2, n);

    // Zero out new_state
    for (int i = 0; i < total_elements; ++i) {
        new_state[i] = 0.0 + 0.0*I;
    }

    // Iterate over all possible indices of the state tensor
    for (int idx = 0; idx < total_elements; ++idx) {
        int new_idx[n];
        int old_idx[n];
        int temp = idx;

        // Compute the multi-dimensional index
        for (int i = n - 1; i >= 0; --i) {
            new_idx[i] = temp % shape[i];
            temp /= shape[i];
        }

        // Perform the tensor contraction manually for the specified qubit
        for (int j = 0; j < 2; ++j) {
            // Copy new_idx to old_idx
            for (int i = 0; i < n; ++i) {
                old_idx[i] = new_idx[i];
            }
            old_idx[qubit] = j;

            // Compute the linear index for old_idx
            int old_linear_idx = 0;
            int factor = 1;
            for (int i = n - 1; i >= 0; --i) {
                old_linear_idx += old_idx[i] * factor;
                factor *= shape[i];
            }

            new_state[idx] += gate[new_idx[qubit]][j] * state[old_linear_idx];
        }
    }
}

void printState(const Complex* state, int N, const char* message) {
    printf("%s\n", message);
    for (int i = 0; i < N; ++i) {
        printf("(%f + %fi) ", creal(state[i]), cimag(state[i]));
    }
    printf("\n");
}

void applyPhaseFlip(Complex* state, int idx) {
    state[idx] *= -1.0 + 0.0*I;
}

void applyGateAllQubits(
    Complex* state,
    const Complex gate[2][2],
    Complex* new_state,
    const int* shape,
    int n,
    int N) {

    for (int i = 0; i < n; ++i) {
        contract_tensor(state, gate, i, new_state, shape, n);
        // Update the state with the new state
        for (int j = 0; j < N; ++j) {
            state[j] = new_state[j];
        }
    }
}

void applyGateSingleQubit(
    Complex* state,
    const Complex gate[2][2],
    Complex* new_state,
    const int* shape,
    int n,
    int N,
    int idx) {

    contract_tensor(state, gate, idx, new_state, shape, n);
    // Update the state with the new state
    for (int j = 0; j < N; ++j) {
        state[j] = new_state[j];
    }
}

void applyDiffusionOperator(
    Complex* state,
    Complex* new_state,
    const int* shape,
    const Complex H[2][2],
    const Complex X[2][2],
    const Complex Z[2][2],
    int n,
    int N) {
    applyGateAllQubits(state, H, new_state, shape, n, N);
    applyGateAllQubits(state, X, new_state, shape, n, N);
    applyPhaseFlip(state, N - 1);
    applyGateSingleQubit(state, Z, new_state, shape, n, N, 0);
    applyGateAllQubits(state, X, new_state, shape, n, N);
    applyGateSingleQubit(state, Z, new_state, shape, n, N, 0);
    applyGateAllQubits(state, H, new_state, shape, n, N);
}

double* simulate(const Complex* weights, int numElements, int numSamples) {
    if (numElements <= 0 || numSamples <= 0) {
        fprintf(stderr, "Invalid input parameters.\n");
        return NULL;
    }

    // Array to count occurrences of each index
    int* counts = (int*)calloc(numElements, sizeof(int));
    // Array to store the average frequencies
    double* averages = (double*)calloc(numElements, sizeof(double));

    if (counts == NULL || averages == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(counts);
        free(averages);
        return NULL;
    }

    // Prepare weights for the distribution by extracting their magnitudes
    double* magnitudes = (double*)malloc(numElements * sizeof(double));
    if (magnitudes == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        free(counts);
        free(averages);
        return NULL;
    }

    for (int i = 0; i < numElements; ++i) {
        magnitudes[i] = cabs(weights[i]);
    }

    // Simulate the weighted distribution
    for (int i = 0; i < numSamples; ++i) {
        double r = (double)rand() / RAND_MAX;
        double cum_prob = 0.0;
        for (int j = 0; j < numElements; ++j) {
            cum_prob += magnitudes[j];
            if (r < cum_prob) {
                counts[j]++;
                break;
            }
        }
    }

    for (int i = 0; i < numElements; ++i) {
        averages[i] = (double)counts[i] / numSamples;
    }

    free(counts);
    free(magnitudes);
    return averages;
}

Complex** createMatrix(int numRows, int numCols, const Complex* initialValues) {
    if (numRows <= 0 || numCols <= 0) {
        fprintf(stderr, "Invalid matrix dimensions.\n");
        return NULL;
    }

    // Allocate memory for row pointers
    Complex** matrix = (Complex**)malloc(numRows * sizeof(Complex*));
    if (matrix == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    // Allocate memory for each row and initialize with provided values
    for (int i = 0; i < numRows; ++i) {
        matrix[i] = (Complex*)malloc(numCols * sizeof(Complex));
        if (matrix[i] == NULL) {
            for (int j = 0; j < i; ++j) {
                free(matrix[j]);
            }
            free(matrix);
            fprintf(stderr, "Memory allocation failed.\n");
            return NULL;
        }
        for (int j = 0; j < numCols; ++j) {
            int index = i * numCols + j;
            matrix[i][j] = initialValues[index];
        }
    }

    return matrix;
}

void deleteMatrix(Complex** matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        free(matrix[i]);
    }
    free(matrix);
}

Complex** kroneckerProduct(Complex** A, int aRows, int aCols, Complex** B, int bRows, int bCols) {
    int resultRows = aRows * bRows;
    int resultCols = aCols * bCols;
    Complex** result = (Complex**)malloc(resultRows * sizeof(Complex*));
    for (int i = 0; i < resultRows; ++i) {
        result[i] = (Complex*)malloc(resultCols * sizeof(Complex));
    }

    for (int i = 0; i < aRows; ++i) {
        for (int j = 0; j < aCols; ++j) {
            for (int k = 0; k < bRows; ++k) {
                for (int l = 0; l < bCols;) {
                    result[i * bRows + k][j * bCols + l] = A[i][j] * B[k][l];
                }
            }
        }
    }

    return result;
}

void printMatrix(Complex** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("(%f + %fi) ", creal(matrix[i][j]), cimag(matrix[i][j]));
        }
        printf("\n");
    }
}
