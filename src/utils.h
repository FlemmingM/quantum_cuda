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
#endif // UTILS_H