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
        d_maxIndex[chunk_id] = sharedIndex[0];
        d_maxValue[chunk_id] = cuCreal(sharedArray[0]);
        chunk_ids[chunk_id] = chunk_id;
        // printf("Val: %f, Index: %d, chunk_id: %d\n", cuCreal(sharedArray[0]), sharedIndex[0], chunk_id);

        // for (int i = 0; i < 2; ++i){
        //     printf("d_maxIndex: %d\n", d_maxIndex[i]);
        // }

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


    // int chunk_size = upper-lower;
    int chunk_size = pow(2, n);

    if ((idx >= lower) & (idx < upper)) {

        int offset = idx * n;
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

            // needed to translate back to the full state array!!!
            if (idx >= chunk_size) {
                old_linear_idx += (idx / chunk_size) * chunk_size;
            }

            // Store the result in shared memory
            if (j == 0) {
                Complex val = cuCmul(gate[new_idx[offset + qubit] * 2 + j], state[old_linear_idx]);
                shared_mem[idx % chunk_size] = val;
                printf("idx: %d, j: %d, old_lin_idx %d, val: %f\n", idx, j, old_linear_idx, cuCreal(val));

            } else {
                Complex val = cuCmul(gate[new_idx[offset + qubit] * 2 + j], state[old_linear_idx]);
                shared_mem[idx % chunk_size] = cuCadd(shared_mem[idx % chunk_size], val);
                // printf("idx: %d, j: %d, old_lin_idx %d, val: %f\n", idx, j, old_linear_idx, cuCreal(val));
            }
            // printf("idx: %d, temp: %d, offset: %d, old_lin_idx %d, upper %lld\n", idx, temp, offset, old_linear_idx, upper);

        }
        __syncthreads();
        state[idx] = shared_mem[idx % chunk_size];
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
    const long long int upper
    ) {

    for (int i = 0; i < n; ++i) {
        contract_tensor<<<dimGrid, dimBlock, sharedMemSize>>>(state, gate, i, new_idx, old_idx, n, lower, upper);
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
    const long long int upper
    ) {

    contract_tensor<<<dimGrid, dimBlock, sharedMemSize>>>(state, gate, idx, new_idx, old_idx, n, lower, upper);
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
    const int num_chunks_per_group,
    const long long int N_chunk,
    const long long int lower,
    const long long int upper
    ) {
    applyGateAllQubits(state, X_H, new_idx, old_idx, n, dimBlock, dimGrid, sharedMemSize, lower, upper);
    for (int i = 0; i < num_chunks_per_group; ++i) {
        applyPhaseFlip<<<dimGrid, dimBlock, 0>>>(state, (i+1)*N_chunk - 1);
        // applyGateSingleQubit(state, Z, new_idx, old_idx, n, i*N_chunk, dimBlock, dimGrid, sharedMemSize, lower, upper);
    }

    applyGateSingleQubit(state, Z, new_idx, old_idx, n, 0, dimBlock, dimGrid, sharedMemSize, lower, upper);
    applyGateAllQubits(state, X, new_idx, old_idx, n, dimBlock, dimGrid, sharedMemSize, lower, upper);
    // for (int i = 0; i < num_chunks_per_group; ++i) {
    //     applyGateSingleQubit(state, Z, new_idx, old_idx, n, i*N_chunk, dimBlock, dimGrid, sharedMemSize, lower, upper);
    // }
    applyGateSingleQubit(state, Z, new_idx, old_idx, n, 0, dimBlock, dimGrid, sharedMemSize, lower, upper);
    applyGateAllQubits(state, H, new_idx, old_idx, n, dimBlock, dimGrid, sharedMemSize, lower, upper);
}
