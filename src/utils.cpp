#include <iostream>
#include <complex>
#include "utils.h"
typedef std::complex<double> Complex;

// Function to dynamically allocate a 2D array for a matrix of complex numbers
Complex** createEmptyMatrix(int rows, int cols) {
    Complex** matrix = new Complex*[rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new Complex[cols];
    }
    return matrix;
}

Complex** createMatrix(int numRows, int numCols, const Complex* initialValues) {
    if (numRows <= 0 || numCols <= 0) {
        std::cerr << "Invalid matrix dimensions." << std::endl;
        return nullptr;
    }

    // Allocate memory for row pointers
    Complex** matrix = new Complex*[numRows];

    // Allocate memory for each row and initialize with provided values
    for (int i = 0; i < numRows; ++i) {
        matrix[i] = new Complex[numCols];
        for (int j = 0; j < numCols; ++j) {
            // Compute the index in the initialValues array
            int index = i * numCols + j;
            matrix[i][j] = initialValues[index];
        }
    }

    return matrix;
}


// Function to delete a dynamically allocated 2D matrix
void deleteMatrix(Complex** matrix, int rows) {
    for (int i = 0; i < rows; ++i) {
        delete[] matrix[i];
    }
    delete[] matrix;
}

// Function to calculate the Kronecker product of two matrices
Complex** kroneckerProduct(Complex** A, int aRows, int aCols, Complex** B, int bRows, int bCols) {
    int resultRows = aRows * bRows;
    int resultCols = aCols * bCols;
    Complex** result = createEmptyMatrix(resultRows, resultCols);

    for (int i = 0; i < aRows; ++i) {
        for (int j = 0; j < aCols; ++j) {
            for (int k = 0; k < bRows; ++k) {
                for (int l = 0; l < bCols; ++l) {
                    result[i * bRows + k][j * bCols + l] = A[i][j] * B[k][l];
                }
            }
        }
    }

    return result;
}

// Function to print a matrix
void printMatrix(Complex** matrix, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

// int main() {
//     // Example matrices
//     int aRows = 1, aCols = 2;
//     int bRows = 1, bCols = 2;
//     Complex** A = createMatrix(aRows, aCols);
//     Complex** B = createMatrix(bRows, bCols);

//     // Initialize matrices A and B
//     A[0][0] = Complex(1, 0); A[0][1] = Complex(0, 0);
//     B[0][0] = Complex(1, 0); B[0][1] = Complex(0, 0);

//     // Calculate Kronecker Product
//     Complex** result = kroneckerProduct(A, aRows, aCols, B, bRows, bCols);

//     // Print the result
//     std::cout << "Kronecker Product of A and B:" << std::endl;
//     printMatrix(result, aRows * bRows, aCols * bCols);

//     // Clean up memory
//     deleteMatrix(A, aRows);
//     deleteMatrix(B, bRows);
//     deleteMatrix(result, aRows * bRows);

//     return 0;
// }
