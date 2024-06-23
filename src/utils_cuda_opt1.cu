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


// __global__ void contract_tensor(
//         const Complex* state,
//         const Complex* gate,
//         int qubit,
//         Complex* new_state,
//         const int* shape,
//         int* new_idx,
//         int* old_idx,
//         const int n,
//         const long long int N
//     ) {
//    extern __shared__ Complex shared_data[];

//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int tid = threadIdx.x;
//     int offset = idx * n;

//     if (idx < N) {

//         int temp = idx;

//         // Compute the multi-dimensional index
//         for (int i = n - 1; i >= 0; --i) {
//             new_idx[i] = temp % shape[i];
//             temp /= shape[i];
//         }

//         Complex sum = make_cuDoubleComplex(0.0, 0.0);

//         // Perform the tensor contraction for the specified qubit
//         for (int j = 0; j < 2; ++j) {
//             // Copy new_idx to old_idx
//             for (int i = 0; i < n; ++i) {
//                 old_idx[i] = new_idx[i];
//             }
//             old_idx[qubit] = j;

//             // Compute the linear index for old_idx
//             int old_linear_idx = 0;
//             int factor = 1;
//             for (int i = n - 1; i >= 0; --i) {
//                 old_linear_idx += old_idx[i] * factor;
//                 factor *= shape[i];
//             }

//             sum = cuCadd(sum, cuCmul(gate[new_idx[qubit] * 2 + j], state[old_linear_idx]));
//         }

//         // Store the result in shared memory
//         shared_data[tid] = sum;

//         __syncthreads();

//         // Perform reduction in shared memory
//         for (int s = blockDim.x / 2; s > 0; s >>= 1) {
//             if (tid < s) {
//                 shared_data[tid] = cuCadd(shared_data[tid], shared_data[tid + s]);
//             }
//             __syncthreads();
//         }

//         // Write the result for this block to global memory
//         if (tid == 0) {
//             AddComplex(&new_state[blockIdx.x], shared_data[0]);
//         }
//     }
//     }


// __global__ void contract_tensor(
//         const Complex* state,
//         const Complex* gate,
//         int qubit,
//         Complex* new_state,
//         const int* shape,
//         int* new_idx,
//         int* old_idx,
//         const int n,
//         const long long int N
//     )  {
//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     if (idx < N) {

//         __shared__ int shared_shape[3];
//         if (threadIdx.x < n) {
//             shared_shape[threadIdx.x] = shape[threadIdx.x];
//         }
//         __syncthreads();


//         int temp = idx;

//         // Compute the multi-dimensional index
//         for (int i = n - 1; i >= 0; --i) {
//             new_idx[i] = temp % shared_shape[i];
//             temp /= shared_shape[i];
//         }

//         // Perform the tensor contraction for the specified qubit
//         for (int j = 0; j < 2; ++j) {
//             // Copy new_idx to old_idx
//             for (int i = 0; i < n; ++i) {
//                 old_idx[i] = new_idx[i];
//             }
//             old_idx[qubit] = j;

//             // Compute the linear index for old_idx
//             int old_linear_idx = 0;
//             int factor = 1;
//             for (int i = n - 1; i >= 0; --i) {
//                 old_linear_idx += old_idx[i] * factor;
//                 factor *= shared_shape[i];
//             }

//             AddComplex(&new_state[idx], cuCmul(gate[new_idx[qubit] * 2 + j], state[old_linear_idx]));
//         }
//     }
// }


// __global__ void contract_tensor_baseline(
//         const Complex* state,
//         const Complex* gate,
//         int qubit,
//         Complex* new_state,
//         const int* shape,
//         int* new_idx,
//         int* old_idx,
//         const int n,
//         const long long int N
// ) {
//     extern __shared__ Complex shared_sum[];

//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int warp_id = threadIdx.x / warpSize;
//     int lane = threadIdx.x % warpSize;

//     if (idx >= N) return;

//     // int new_idx[3];
//     // int old_idx[3];
//     int temp = idx;

//     // Compute the multi-dimensional index
//     for (int i = n - 1; i >= 0; --i) {
//         new_idx[i] = temp % shape[i];
//         temp /= shape[i];
//     }

//     Complex sum = make_cuDoubleComplex(0.0, 0.0);

//     // Perform the tensor contraction for the specified qubit
//     for (int j = 0; j < 2; ++j) {
//         // Copy new_idx to old_idx
//         for (int i = 0; i < n; ++i) {
//             old_idx[i] = new_idx[i];
//         }
//         old_idx[qubit] = j;

//         // Compute the linear index for old_idx
//         int old_linear_idx = 0;
//         int factor = 1;
//         for (int i = n - 1; i >= 0; --i) {
//             old_linear_idx += old_idx[i] * factor;
//             factor *= shape[i];
//         }

//         sum = cuCadd(sum, cuCmul(gate[new_idx[qubit] * 2 + j], state[old_linear_idx]));
//     }

//     // Perform warp reduction
//     for (int offset = warpSize / 2; offset > 0; offset /= 2) {
//         sum.x += __shfl_down_sync(0xffffffff, sum.x, offset);
//         sum.y += __shfl_down_sync(0xffffffff, sum.y, offset);
//     }

//     // Each warp writes its reduced value to shared memory
//     if (lane == 0) {
//         shared_sum[warp_id] = sum;
//     }
//     __syncthreads();

//     // Only the first warp in each block performs the final reduction
//     if (warp_id == 0) {
//         sum = make_cuDoubleComplex(0.0, 0.0);
//         int num_warps = (blockDim.x + warpSize - 1) / warpSize;
//         for (int i = 0; i < num_warps; ++i) {
//             sum = cuCadd(sum, shared_sum[i]);
//         }

//         // Write the final block result to global memory
//         atomicAdd((double*)&new_state[blockIdx.x].x, sum.x);
//         atomicAdd((double*)&new_state[blockIdx.x].y, sum.y);
//     }
// }


__global__ void contract_tensor(
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
    int offset = idx * n;

    // we can only get 2 threads to work together on 1 value
    // thus we need 2 which compute the values in paralell and then sum them up to 1 value
    // and add them to the new_state array
    // we do the same as in the baseline but reduce with 2 threads since our operators are 2x2 only
    // maybe another improvement could be to use shared memory instead of global?!


    // Try to hardcode the sum reduction so 2 values are summed up together and see if it is
    // faster than the baseline
    // also add shared memory and see how it improves it.

    if (idx < N) {

        int temp = idx;

        // Compute the multi-dimensional index
        for (int i = n - 1; i >= 0; --i) {
            new_idx[offset+i] = temp % shape[i];
            temp /= shape[i];
        }

        // Perform the tensor contraction for the specified qubit

        // here you need to hardcode and add the js 2 times and sum them in the end.
        // for (int j = 0; j < 2; ++j) {
        // Copy new_idx to old_idx
        for (int i = 0; i < n; ++i) {
            old_idx[offset+i] = new_idx[offset+i];
        }

        int j = 0;
        old_idx[offset+qubit] = j;
        // Compute the linear index for old_idx
        int old_linear_idx = 0;
        int factor = 1;
        for (int i = n - 1; i >= 0; --i) {
            old_linear_idx += old_idx[offset+i] * factor;
            factor *= shape[i];
        }
        AddComplex(&new_state[idx], cuCmul(gate[new_idx[offset+qubit] * 2 + j], state[old_linear_idx]));


        j = 1;
        old_idx[offset+qubit] = j;
        // Compute the linear index for old_idx
        old_linear_idx = 0;
        factor = 1;
        for (int i = n - 1; i >= 0; --i) {
            old_linear_idx += old_idx[offset+i] * factor;
            factor *= shape[i];
        }
        AddComplex(&new_state[idx], cuCmul(gate[new_idx[offset+qubit] * 2 + j], state[old_linear_idx]));
            // }
        }
    }


// __global__ void contract_tensor(
//         const Complex* state,
//         const Complex* gate,
//         int qubit,
//         Complex* new_state,
//         const int* shape,
//         int* new_idx,
//         int* old_idx,
//         const int n,
//         const long long int N
//     ) {

//     extern __shared__ Complex shared_sum[];


//     int idx = blockDim.x * blockIdx.x + threadIdx.x;
//     int offset = idx * n;

//     // define value
//     Complex sum = make_cuDoubleComplex(0.0, 0.0);

//     // in case we are out of bounds
//     if (idx < N) {

//         int temp = idx;

//         // Compute the multi-dimensional index
//         for (int i = n - 1; i >= 0; --i) {
//             new_idx[offset+i] = temp % shape[i];
//             temp /= shape[i];
//         }

//         // Perform the tensor contraction for the specified qubit
//         for (int j = 0; j < 2; ++j) {
//             // Copy new_idx to old_idx
//             for (int i = 0; i < n; ++i) {
//                 old_idx[offset+i] = new_idx[offset+i];
//             }
//             old_idx[offset+qubit] = j;

//             // Compute the linear index for old_idx
//             int old_linear_idx = 0;
//             int factor = 1;
//             for (int i = n - 1; i >= 0; --i) {
//                 old_linear_idx += old_idx[offset+i] * factor;
//                 factor *= shape[i];
//             }
//             sum = cuCadd(sum, cuCmul(gate[new_idx[offset+qubit] * 2 + j], state[old_linear_idx]));
//             // AddComplex(&new_state[idx], cuCmul(gate[new_idx[offset+qubit] * 2 + j], state[old_linear_idx]));
//             }
//         shared_sum[threadIdx.x] = sum;
//         // __synthreads();

//         for (int i = 1; i > 0; i /= 2) {
//             // __shfl_down_sync(0xffffffff, sum.x, i);
//             sum.x += __shfl_down_sync(0xffffffff, sum.x, i);
//             sum.y += __shfl_down_sync(0xffffffff, sum.y, i);
//         }


//         if ((threadIdx.x % 32) == 0) {
//             AddComplex(&new_state[blockIdx.x * 32 + threadIdx.x / 32], sum);
//         }


//         }
//     }


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
        contract_tensor<<<dimGrid, dimBlock>>>(state, gate, i, new_state, shape, new_idx, old_idx, n, N);
        // contract_tensor<<<dimGrid, dimBlock>>>(state, gate, i, new_state, shape, n, N);
        // cudaDeviceSynchronize();
        // Update the state with the new state
        updateState<<<dimGrid, dimBlock>>>(state, new_state, N);
        // cudaDeviceSynchronize();
        zeroOutState<<<dimGrid, dimBlock>>>(new_state, N);
        // cudaDeviceSynchronize();
    }
}

void applyGateSingleQubit(
    Complex* state,
    const Complex* gate,
    Complex* new_state,
    const int* shape,
    int* new_idx,
    int* old_idx,
    int n,
    long long int N,
    long long int idx,
    dim3 dimBlock,
    dim3 dimGrid
    ) {

    contract_tensor<<<dimGrid, dimBlock>>>(state, gate, idx, new_state, shape, new_idx, old_idx, n, N);
    // Update the state with the new state
    updateState<<<dimGrid, dimBlock>>>(state, new_state, N);
    zeroOutState<<<dimGrid, dimBlock>>>(new_state, N);
}

void applyDiffusionOperator(
    Complex* state,
    Complex* new_state,
    const int* shape,
    const Complex* H,
    const Complex* X,
    const Complex* Z,
    int* new_idx,
    int* old_idx,
    int n,
    long long int N,
    dim3 dimBlock,
    dim3 dimGrid
    ) {
    applyGateAllQubits(state, H, new_state, shape, new_idx, old_idx, n, N, dimBlock, dimGrid);
    applyGateAllQubits(state, X, new_state, shape, new_idx, old_idx, n, N, dimBlock, dimGrid);
    applyPhaseFlip<<<dimGrid, dimBlock>>>(state, N - 1);
    applyGateSingleQubit(state, Z, new_state, shape, new_idx, old_idx, n, N, 0, dimBlock, dimGrid);
    applyGateAllQubits(state, X, new_state, shape, new_idx, old_idx, n, N, dimBlock, dimGrid);
    applyGateSingleQubit(state, Z, new_state, shape, new_idx, old_idx, n, N, 0, dimBlock, dimGrid);
    applyGateAllQubits(state, H, new_state, shape, new_idx, old_idx, n, N, dimBlock, dimGrid);
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
