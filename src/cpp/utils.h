#ifndef UTILS_H
#define UTILS_H

#include <complex>
#include "nodes.h"

using Complex = std::complex<double>;

// Declare any functions from utils.cpp that you want to use in other files
Complex** createMatrix(int rows, int cols, const Complex* initialValues);
void deleteMatrix(Complex** matrix, int rows);
Complex** kroneckerProduct(Complex** A, int aRows, int aCols, Complex** B, int bRows, int bCols);
void printMatrix(Complex** matrix, int rows, int cols);
double* simulate(const Complex* weights, int numElements, int numSamples);
// Complex* flattenMatrix(Complex** matrix, int rows, int cols);
void applyPhaseFlip(Edge* qRegister, int n, int state_a, int state_b);
Edge* initQubits(int n, float factor);
void showRegister(Edge* qRegister, int n);
void contract_tensor(const std::complex<double>* state,
                     const std::complex<double> gate[2][2],
                     int qubit,
                     std::complex<double>* new_state,
                     const int* shape, int n);
void printState(const std::complex<double>* state, int N, std::string message);
void applyPhaseFlip(std::complex<double>* state, int idx);
void applyGateAllQubits(
    std::complex<double>* state,
    const std::complex<double> gate[2][2],
    std::complex<double>* new_state,
    const int* shape,
    int n,
    int N);
void applyGateSingleQubit(
    std::complex<double>* state,
    const std::complex<double> gate[2][2],
    std::complex<double>* new_state,
    const int* shape,
    int n,
    int N,
    int idx
    );
void applyDiffusionOperator(
    std::complex<double>* state,
    std::complex<double>* new_state,
    const int* shape,
    const std::complex<double> H[2][2],
    const std::complex<double> X[2][2],
    const std::complex<double> Z[2][2],
    int n,
    int N
);
void saveArrayToCSV(const double *array, int N, const std::string& filename);
#endif // UTILS_H