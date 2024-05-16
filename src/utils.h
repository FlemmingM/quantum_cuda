#ifndef UTILS_H
#define UTILS_H

#include <complex>

using Complex = std::complex<double>;

// Declare any functions from utils.cpp that you want to use in other files
Complex** createMatrix(int rows, int cols, const Complex* initialValues);
void deleteMatrix(Complex** matrix, int rows);
Complex** kroneckerProduct(Complex** A, int aRows, int aCols, Complex** B, int bRows, int bCols);
void printMatrix(Complex** matrix, int rows, int cols);

#endif // UTILS_H