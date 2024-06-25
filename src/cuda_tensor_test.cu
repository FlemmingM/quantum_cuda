#include <stdio.h>
#include <cuComplex.h>
#include <math.h>
#include <string.h>
#include <errno.h>
#include <omp.h>

typedef cuDoubleComplex Complex;


__device__ void AddComplex(cuDoubleComplex* a, cuDoubleComplex b){
  //transform the addresses of real and imag. parts to double pointers
  double *x = (double*)a;
  double *y = x+1;
  //use atomicAdd for double variables
  atomicAdd(x, cuCreal(b));
  atomicAdd(y, cuCimag(b));
}

__global__ void zeroOutState(Complex* new_state, int total_elements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_elements) {
        new_state[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
}

__global__ void zeroOutInt(int* array, int total_elements) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_elements) {
        array[idx] = 0;
    }
}

__global__ void contract_tensor_baseline(
        const Complex* state,
        const Complex* gate,
        int qubit,
        Complex* new_state,
        const int* shape,
        const int n,
        int total_elements
    ) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < total_elements) {

    int *new_idx = (int *)malloc(n * sizeof(int));
    int *old_idx = (int *)malloc(n * sizeof(int));

    int temp = idx;

    // Compute the multi-dimensional index
    for (int i = n - 1; i >= 0; --i) {
        new_idx[i] = temp % shape[i];
        temp /= shape[i];
    }

    // Perform the tensor contraction for the specified qubit
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
        AddComplex(&new_state[idx], cuCmul(gate[new_idx[qubit] * 2 + j], state[old_linear_idx]));
        }
    }
}





int main() {
    int n = 3;
    int N = (int)pow(2, n);


    cuDoubleComplex H_h[4] = {
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0),
        make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0), make_cuDoubleComplex(-1.0 / sqrt(2.0), 0.0)
    };


    Complex *state_h;
    Complex *state_d;
    Complex *new_state_h;
    Complex *new_state_d;
    Complex *H_d;

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

    // Malloc the indices on the device
    cudaMalloc((void **)&new_idx, n * sizeof(int));
    cudaMalloc((void **)&old_idx, n * sizeof(int));

    // // Init gate values
    // H[0] = make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0);
    // H[1] = make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0);
    // H[2] = make_cuDoubleComplex(1.0 / sqrt(2.0), 0.0);
    // H[3] = make_cuDoubleComplex(-1.0 / sqrt(2.0), 0.0);


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
    cudaMemcpy(H_d, H_h, 4 * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(new_state_d, new_state_h, N * sizeof(Complex), cudaMemcpyHostToDevice);
    cudaMemcpy(shape_d, shape_h, n * sizeof(int), cudaMemcpyHostToDevice);

    dim3 dimBlock(256);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

    double time = omp_get_wtime();
    // const Complex* state, const Complex* gate,
    //                                  int qubit, Complex* new_state,
    //                                  const int* shape, const int n, int total_elements
    // zeroOutState<<<dimGrid, dimBlock>>>(new_state_d, N);
    contract_tensor_baseline<<<dimGrid, dimBlock>>>(state_d, H_d, 0, new_state_d, shape_d, 3, N);
    // contract_tensor_baseline<<<dimGrid, dimBlock>>>(state_d, H_d, 0, new_state_d, shape_d, 3, N, new_idx, old_idx);
    // zeroOutInt<<<dimGrid, dimBlock>>>(new_idx, n);
    // zeroOutInt<<<dimGrid, dimBlock>>>(old_idx, n);
    cudaDeviceSynchronize();
    zeroOutState<<<dimGrid, dimBlock>>>(state_d, N);
    contract_tensor_baseline<<<dimGrid, dimBlock>>>(new_state_d, H_d, 1, state_d, shape_d, 3, N);
    // contract_tensor_baseline<<<dimGrid, dimBlock>>>(new_state_d, H_d, 1, state_d, shape_d, 3, N, new_idx, old_idx);
    // zeroOutInt<<<dimGrid, dimBlock>>>(new_idx, n);
    // zeroOutInt<<<dimGrid, dimBlock>>>(old_idx, n);
    cudaDeviceSynchronize();


    zeroOutState<<<dimGrid, dimBlock>>>(new_state_d, N);
    contract_tensor_baseline<<<dimGrid, dimBlock>>>(state_d, H_d, 2, new_state_d, shape_d, 3, N);
    // contract_tensor_baseline<<<dimGrid, dimBlock>>>(state_d, H_d, 2, new_state_d, shape_d, 3, N, new_idx, old_idx);

    // zeroOutInt<<<dimGrid, dimBlock>>>(new_idx, n);
    // zeroOutInt<<<dimGrid, dimBlock>>>(old_idx, n);
    cudaDeviceSynchronize();

    double elapsed_time = omp_get_wtime() - time;
    printf("time: %f\n", elapsed_time);

    cudaMemcpy(new_state_h, new_state_d, N * sizeof(Complex), cudaMemcpyDeviceToHost);
    cudaMemcpy(state_h, state_d, N * sizeof(Complex), cudaMemcpyDeviceToHost);

    // Print the result for verification
    for (int i = 0; i < N; ++i) {
        printf("new_state[%d] = (%f, %f)\n", i, cuCreal(new_state_h[i]), cuCimag(new_state_h[i]));
    }

    for (int i = 0; i < N; ++i) {
        printf("state[%d] = (%f, %f)\n", i, cuCreal(state_h[i]), cuCimag(state_h[i]));
    }

    // cudafree
    cudaFree(state_d);
    cudaFree(new_state_d);
    cudaFree(shape_d);
    cudaFree(H_d);
    cudaFreeHost(state_h);
    cudaFreeHost(new_state_h);
    cudaFreeHost(shape_h);
    // cudaFreeHost(H_h);
}