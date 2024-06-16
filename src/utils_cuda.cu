#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>

typedef cuDoubleComplex Complex;

void saveArrayToCSV(const double *array, long long int N, const char* filename) {
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


__device__ void AddComplex(cuDoubleComplex* a, cuDoubleComplex b){
  //transform the addresses of real and imag. parts to double pointers
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables
  atomicAdd(x, cuCreal(b));
  atomicAdd(y, cuCimag(b));
}

__global__ void zeroOutState(Complex* new_state, long long int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        new_state[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}


__global__ void updateState(Complex* state, Complex* new_state, long long int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        state[idx] = new_state[idx];
    }
}

__global__ void contract_tensor_baseline(
        const Complex* state,
        const Complex* gate,
        int qubit,
        Complex* new_state,
        const int* shape,
        int* new_idx,
        int* old_idx,
        const int n,
        const long long int N
    ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    // int offset = blockDim.x * gridDim.x;
    int offset = idx * n;
    if (idx < N) {

        int temp = idx;

        // Compute the multi-dimensional index
        for (int i = n - 1; i >= 0; --i) {
            new_idx[offset+i] = temp % shape[i];
            temp /= shape[i];
        }

        // Perform the tensor contraction for the specified qubit
        for (int j = 0; j < 2; ++j) {
            // Copy new_idx to old_idx
            for (int i = 0; i < n; ++i) {
                old_idx[offset+i] = new_idx[offset+i];
            }
            old_idx[offset+qubit] = j;

            // Compute the linear index for old_idx
            int old_linear_idx = 0;
            int factor = 1;
            for (int i = n - 1; i >= 0; --i) {
                old_linear_idx += old_idx[offset+i] * factor;
                factor *= shape[i];
            }
            AddComplex(&new_state[idx], cuCmul(gate[new_idx[offset+qubit] * 2 + j], state[old_linear_idx]));
            }
        }
    }

void printState(const Complex* state, long long int N, const char* message) {
    printf("%s\n", message);
    for (int i = 0; i < N; ++i) {
        printf("(%.15f + %.15fi) ", cuCreal(state[i]), cuCimag(state[i]));
    }
    printf("\n");
}

__global__ void applyPhaseFlip(Complex* state, long long int idx) {
    state[idx] = cuCmul(state[idx], make_cuDoubleComplex(-1.0, 0.0));
}

void applyGateAllQubits(
    Complex* state,
    const Complex* gate,
    Complex* new_state,
    const int* shape,
    int* new_idx,
    int* old_idx,
    int n,
    long long int N,
    dim3 dimBlock,
    dim3 dimGrid
    ) {

    for (int i = 0; i < n; ++i) {
        contract_tensor_baseline<<<dimGrid, dimBlock>>>(state, gate, i, new_state, shape, new_idx, old_idx, n, N);
        // contract_tensor_baseline<<<dimGrid, dimBlock>>>(state, gate, i, new_state, shape, n, N);
        cudaDeviceSynchronize();
        // Update the state with the new state
        updateState<<<dimGrid, dimBlock>>>(state, new_state, N);
        cudaDeviceSynchronize();
        zeroOutState<<<dimGrid, dimBlock>>>(new_state, N);
        cudaDeviceSynchronize();
    }
}

// void applyGateSingleQubit(
//     Complex* state,
//     const Complex* gate,
//     Complex* new_state,
//     const int* shape,
//     int n,
//     long long int N,
//     long long int idx,
//     dim3 dimBlock,
//     dim3 dimGrid
//     ) {

//     contract_tensor_baseline<<<dimGrid, dimBlock>>>(state, gate, idx, new_state, shape, n, N);
//     // Update the state with the new state
//     updateState<<<dimGrid, dimBlock>>>(state, new_state, N);
//     zeroOutState<<<dimGrid, dimBlock>>>(new_state, N);
// }

// void applyDiffusionOperator(
//     Complex* state,
//     Complex* new_state,
//     const int* shape,
//     const Complex* H,
//     const Complex* X,
//     const Complex* Z,
//     int n,
//     long long int N,
//     dim3 dimBlock,
//     dim3 dimGrid
//     ) {
//     applyGateAllQubits(state, H, new_state, shape, n, N, dimBlock, dimGrid);
//     applyGateAllQubits(state, X, new_state, shape, n, N, dimBlock, dimGrid);
//     applyPhaseFlip<<<dimGrid, dimBlock>>>(state, N - 1);
//     applyGateSingleQubit(state, Z, new_state, shape, n, N, 0, dimBlock, dimGrid);
//     applyGateAllQubits(state, X, new_state, shape, n, N, dimBlock, dimGrid);
//     applyGateSingleQubit(state, Z, new_state, shape, n, N, 0, dimBlock, dimGrid);
//     applyGateAllQubits(state, H, new_state, shape, n, N, dimBlock, dimGrid);
// }

// double* simulate(const Complex* weights, int numElements, int numSamples) {
//     if (numElements <= 0 || numSamples <= 0) {
//         fprintf(stderr, "Invalid input parameters.\n");
//         return NULL;
//     }

//     // Array to count occurrences of each index
//     int* counts = (int*)calloc(numElements, sizeof(int));
//     // Array to store the average frequencies
//     double* averages = (double*)calloc(numElements, sizeof(double));

//     if (counts == NULL || averages == NULL) {
//         fprintf(stderr, "Memory allocation failed.\n");
//         free(counts);
//         free(averages);
//         return NULL;
//     }

//     // Prepare weights for the distribution by extracting their magnitudes
//     double* magnitudes = (double*)malloc(numElements * sizeof(double));
//     if (magnitudes == NULL) {
//         fprintf(stderr, "Memory allocation failed.\n");
//         free(counts);
//         free(averages);
//         return NULL;
//     }

//     for (int i = 0; i < numElements; ++i) {
//         magnitudes[i] = cabs(weights[i]);
//     }

//     // Simulate the weighted distribution
//     for (int i = 0; i < numSamples; ++i) {
//         double r = (double)rand() / RAND_MAX;
//         double cum_prob = 0.0;
//         for (int j = 0; j < numElements; ++j) {
//             cum_prob += magnitudes[j];
//             if (r < cum_prob) {
//                 counts[j]++;
//                 break;
//             }
//         }
//     }

//     for (int i = 0; i < numElements; ++i) {
//         averages[i] = (double)counts[i] / numSamples;
//     }

//     free(counts);
//     free(magnitudes);
//     return averages;
// }

// Complex** createMatrix(int numRows, int numCols, const Complex* initialValues) {
//     if (numRows <= 0 || numCols <= 0) {
//         fprintf(stderr, "Invalid matrix dimensions.\n");
//         return NULL;
//     }

//     // Allocate memory for row pointers
//     Complex** matrix = (Complex**)malloc(numRows * sizeof(Complex*));
//     if (matrix == NULL) {
//         fprintf(stderr, "Memory allocation failed.\n");
//         return NULL;
//     }

//     // Allocate memory for each row and initialize with provided values
//     for (int i = 0; i < numRows; ++i) {
//         matrix[i] = (Complex*)malloc(numCols * sizeof(Complex));
//         if (matrix[i] == NULL) {
//             for (int j = 0; j < i; ++j) {
//                 free(matrix[j]);
//             }
//             free(matrix);
//             fprintf(stderr, "Memory allocation failed.\n");
//             return NULL;
//         }
//         for (int j = 0; j < numCols; ++j) {
//             int index = i * numCols + j;
//             matrix[i][j] = initialValues[index];
//         }
//     }

//     return matrix;
// }

// void deleteMatrix(Complex** matrix, int rows) {
//     for (int i = 0; i < rows; ++i) {
//         free(matrix[i]);
//     }
//     free(matrix);
// }

// Complex** kroneckerProduct(Complex** A, int aRows, int aCols, Complex** B, int bRows, int bCols) {
//     int resultRows = aRows * bRows;
//     int resultCols = aCols * bCols;
//     Complex** result = (Complex**)malloc(resultRows * sizeof(Complex*));
//     for (int i = 0; i < resultRows; ++i) {
//         result[i] = (Complex*)malloc(resultCols * sizeof(Complex));
//     }

//     for (int i = 0; i < aRows; ++i) {
//         for (int j = 0; j < aCols; ++j) {
//             for (int k = 0; k < bRows; ++k) {
//                 for (int l = 0; l < bCols;) {
//                     result[i * bRows + k][j * bCols + l] = A[i][j] * B[k][l];
//                 }
//             }
//         }
//     }

//     return result;
// }

// void printMatrix(Complex** matrix, int rows, int cols) {
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             printf("(%f + %fi) ", creal(matrix[i][j]), cimag(matrix[i][j]));
//         }
//         printf("\n");
//     }
// }
