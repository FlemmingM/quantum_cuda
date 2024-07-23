#ifndef UTILS_H
#define UTILS_H

#include <complex.h>

// Define a complex number type for simplicity
typedef double complex Complex;

// Declare any functions from utils.c that you want to use in other files
// Complex** createMatrix(int rows, int cols, const Complex* initialValues);
// void deleteMatrix(Complex** matrix, int rows);
// Complex** kroneckerProduct(Complex** A, int aRows, int aCols, Complex** B, int bRows, int bCols);
// void printMatrix(Complex** matrix, int rows, int cols);
double* simulate(const Complex* weights, int numElements, int numSamples);
void applyPhaseFlip(Complex* state, int idx);
void applyGateAllQubits(
    Complex* state,
    const Complex gate[2][2],
    Complex* new_state,
    const int* shape,
    int n,
    int N);
void applyGateSingleQubit(
    Complex* state,
    const Complex gate[2][2],
    Complex* new_state,
    const int* shape,
    int n,
    int N,
    int idx);
void applyDiffusionOperator(
    Complex* state,
    Complex* new_state,
    const int* shape,
    const Complex H[2][2],
    const Complex X[2][2],
    const Complex Z[2][2],
    int n,
    int N);
void saveArrayToCSV(const double *array, int N, const char* filename);
void contract_tensor(const Complex* state,
                     const Complex gate[2][2],
                     int qubit,
                     Complex* new_state,
                     const int* shape, int n);
void printState(const Complex* state, int N, const char* message);
void zeroOutArray(Complex* array, int length);

#endif // UTILS_H
