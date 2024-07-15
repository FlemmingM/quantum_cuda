#include <stdio.h>
#include <stdlib.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>

typedef cuDoubleComplex Complex;


void allocateGatesDevice(const int num_devices, Complex **H_d, Complex **I_d, Complex **Z_d, Complex **X_d, Complex **X_H_d) {

    // Define the gates
    cuDoubleComplex H_h[4] = {
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0),
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(-1.0 / sqrt(2.0), 0.0)
    };
    cuDoubleComplex X_H_h[4] = {
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(-1.0 / sqrt(2.0), 0.0),
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0)
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

    for (int i = 0; i < num_devices; i++) {
        // Set the device
        cudaSetDevice(i);

        // Malloc the gate on device
        cudaMalloc((void **)&H_d[i], 4 * sizeof(Complex));
        cudaMalloc((void **)&X_H_d[i], 4 * sizeof(Complex));
        cudaMalloc((void **)&I_d[i], 4 * sizeof(Complex));
        cudaMalloc((void **)&Z_d[i], 4 * sizeof(Complex));
        cudaMalloc((void **)&X_d[i], 4 * sizeof(Complex));

        // Copy from host to device
        cudaMemcpy(H_d[i], H_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
        cudaMemcpy(X_H_d[i], X_H_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
        cudaMemcpy(I_d[i], I_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
        cudaMemcpy(Z_d[i], Z_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
        cudaMemcpy(X_d[i], X_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    }
}

void printState(const Complex* state, long long int N, const char* message) {
    printf("%s\n", message);
    for (int i = 0; i < N; ++i) {
        printf("(%.15f + %.15fi) ", cuCreal(state[i]), cuCimag(state[i]));
    }
    printf("\n");
}

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

__global__ void initState(Complex* new_state, int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        if (idx==0) {
            new_state[idx] = make_cuDoubleComplex(1.0, 0.0);
        } else {
            new_state[idx] = make_cuDoubleComplex(0.0, 0.0);
        }

    }
}


__global__ void updateState(Complex* state, Complex* new_state, long long int N) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        state[idx] = new_state[idx];
    }
}


__global__ void findMaxIndexKernel(Complex* d_array, int* d_maxIndex, double* d_maxValue, int size, int chunk_id, int* chunk_ids) {
    __shared__ Complex sharedArray[1024];
    __shared__ int sharedIndex[1024];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size) {
        sharedArray[tid] = d_array[index];
        sharedIndex[tid] = index;
    } else {
        sharedArray[tid] = make_cuDoubleComplex(-99.0, 0.00);  // Set to minimum value if out of bounds
        sharedIndex[tid] = -1;        // Invalid index
    }

    __syncthreads();

    // Perform reduction to find the max value and its index
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride && index + stride < size) {
            if (cuCreal(sharedArray[tid]) < cuCreal(sharedArray[tid + stride])) {
                sharedArray[tid] = sharedArray[tid + stride];
                sharedIndex[tid] = sharedIndex[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        // printf("Val: %f, Index: %d, chunk_id: %d\n", cuCreal(sharedArray[0]), sharedIndex[0], chunk_id);
        // printf("Index: %d\n", sharedIndex[0]);
        // printf("chunk_id: %d\n", chunk_id);
        *d_maxIndex = sharedIndex[0];
        *d_maxValue = cuCreal(sharedArray[0]);
        *chunk_ids = chunk_id;

    }
}


__global__ void contract_tensor(
        Complex* state,
        const Complex* gate,
        int qubit,
        int* new_idx,
        int* old_idx,
        const int n,
        // const long long int N,
        const long long int lower,
        const long long int upper
) {
    extern __shared__ Complex shared_mem[]; // Use shared memory
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int chunk_size = upper-lower;


    if ((idx >= lower) & (idx < upper)) {
        if (idx == lower) {
        // printf("kernel id: %d\n", kernel_id);

        }
        int offset = (idx % chunk_size) * n;
        int temp = idx % chunk_size;
        // int temp = idx;

        // printf("idx: %d, temp: %d, offset: %d, lower %lld, upper %lld\n", idx, temp, offset, lower, upper);

        // Compute the multi-dimensional index
        for (int i = n - 1; i >= 0; --i) {
            new_idx[offset + i] = temp % 2;
            temp /= 2;
        }

        // Copy new_idx to old_idx
        for (int i = 0; i < n; ++i) {
            old_idx[offset + i] = new_idx[offset + i];
        }

        // Compute the two values for j = 0 and j = 1 and store in shared memory
        for (int j = 0; j < 2; ++j) {
            old_idx[offset + qubit] = j;

            // Compute the linear index for old_idx
            int old_linear_idx = 0;
            int factor = 1;
            for (int i = n - 1; i >= 0; --i) {
                old_linear_idx += old_idx[offset + i] * factor;
                factor *= 2;
            }

            // Store the result in shared memory
            if (j == 0) {
                shared_mem[idx] = cuCmul(gate[new_idx[offset + qubit] * 2 + j], state[old_linear_idx]);
                // printf("idx: %d, val: %f\n", idx, cuCreal(val));

            } else {

                shared_mem[idx] = cuCadd(shared_mem[idx],
                cuCmul(gate[new_idx[offset + qubit] * 2 + j], state[old_linear_idx]));
            }
        }

        // printf("value[%d]: %.15f\n", idx, cuCreal(shared_mem[idx]));
        // printf("idx: %d, val: %f\n", idx, cuCreal(val));

        __syncthreads();

        // for (int i = lower; i < upper; ++i) {
        //     printf("idx: %d, pos: %d, val: %f \n", idx, i, cuCreal(shared_mem[i]));
        // }

        state[idx % chunk_size] = shared_mem[idx];

    }
}


__global__ void applyPhaseFlip(Complex* state, long long int idx) {
    state[idx] = cuCmul(state[idx], make_cuDoubleComplex(-1.0, 0.0));
}

void applyGateAllQubits(
    Complex* state,
    const Complex* gate,
    int* new_idx,
    int* old_idx,
    int n,
    dim3 dimBlock,
    dim3 dimGrid,
    int sharedMemSize,
    const long long int lower,
    const long long int upper,
    cudaStream_t stream
    ) {

    for (int i = 0; i < n; ++i) {
        contract_tensor<<<dimGrid, dimBlock, sharedMemSize, stream>>>(state, gate, i, new_idx, old_idx, n, lower, upper);
    }
}

void applyGateSingleQubit(
    Complex* state,
    const Complex* gate,
    int* new_idx,
    int* old_idx,
    int n,
    long long int idx,
    dim3 dimBlock,
    dim3 dimGrid,
    int sharedMemSize,
    const long long int lower,
    const long long int upper,
    cudaStream_t stream
    ) {

    contract_tensor<<<dimGrid, dimBlock, sharedMemSize, stream>>>(state, gate, idx, new_idx, old_idx, n, lower, upper);
}

void applyDiffusionOperator(
    Complex* state,
    const Complex* X_H,
    const Complex* H,
    const Complex* X,
    const Complex* Z,
    int* new_idx,
    int* old_idx,
    int n,
    dim3 dimBlock,
    dim3 dimGrid,
    int sharedMemSize,
    const long long int lower,
    const long long int upper,
    cudaStream_t stream
    ) {
    applyGateAllQubits(state, X_H, new_idx, old_idx, n, dimBlock, dimGrid, sharedMemSize, lower, upper, stream);
    applyPhaseFlip<<<dimGrid, dimBlock, 0, stream>>>(state, upper-lower - 1);
    applyGateSingleQubit(state, Z, new_idx, old_idx, n, 0, dimBlock, dimGrid, sharedMemSize, lower, upper, stream);
    applyGateAllQubits(state, X, new_idx, old_idx, n, dimBlock, dimGrid, sharedMemSize, lower, upper, stream);
    applyGateSingleQubit(state, Z, new_idx, old_idx, n, 0, dimBlock, dimGrid, sharedMemSize, lower, upper, stream);
    applyGateAllQubits(state, H, new_idx, old_idx, n, dimBlock, dimGrid, sharedMemSize, lower, upper, stream);
}

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
