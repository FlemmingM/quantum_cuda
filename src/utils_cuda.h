#ifndef UTILS_H
#define UTILS_H

#include <cuComplex.h>

// Define a complex number type for simplicity
typedef cuDoubleComplex Complex;

// Declare any functions from utils.c that you want to use in other files
// Complex** createMatrix(int rows, int cols, const Complex* initialValues);
// void deleteMatrix(Complex** matrix, int rows);
// Complex** kroneckerProduct(Complex** A, int aRows, int aCols, Complex** B, int bRows, int bCols);
// void printMatrix(Complex** matrix, int rows, int cols);
double* simulate(const Complex* weights, long long int numElements, int numSamples);
__global__ void applyPhaseFlip(Complex* state, long long int idx);

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
    );

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
    );

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
    );

void saveArrayToCSV(const double *array, int N, const char* filename);

__global__ void contract_tensor_baseline(
    const Complex* state,
    const Complex* gate,
    int qubit,
    Complex* new_state,
    const int* shape,
    int* new_idx,
    int* old_idx,
    const int n,
    long long int N
    );

__global__ void zeroOutState(Complex* new_state, long long int N);

void printState(const Complex* state, long long int N, const char* message);

__device__ void AddComplex(cuDoubleComplex* a, cuDoubleComplex b);

__global__ void updateState(Complex* state, Complex* new_state, long long int N);
#endif // UTILS_H
