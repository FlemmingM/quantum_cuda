#include <iostream>
#include <complex>

typedef std::complex<double> Complex;

// Function to dynamically allocate a 2D array for a matrix of complex numbers
Complex** createMatrix(int rows, int cols) {
    Complex** matrix = new Complex*[rows];
    for (int i = 0; i < rows; ++i) {
        matrix[i] = new Complex[cols];
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
    Complex** result = createMatrix(resultRows, resultCols);

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

int main() {
    // Example matrices
    int aRows = 2, aCols = 2;
    int bRows = 2, bCols = 2;
    Complex** A = createMatrix(aRows, aCols);
    Complex** B = createMatrix(bRows, bCols);

    // Initialize matrices A and B
    A[0][0] = Complex(1, 0); A[0][1] = Complex(2, 0);
    A[1][0] = Complex(3, 0); A[1][1] = Complex(4, 0);
    B[0][0] = Complex(0, 1); B[0][1] = Complex(0, 2);
    B[1][0] = Complex(0, 3); B[1][1] = Complex(0, 4);

    // Calculate Kronecker Product
    Complex** result = kroneckerProduct(A, aRows, aCols, B, bRows, bCols);

    // Print the result
    std::cout << "Kronecker Product of A and B:" << std::endl;
    printMatrix(result, aRows * bRows, aCols * bCols);

    // Clean up memory
    deleteMatrix(A, aRows);
    deleteMatrix(B, bRows);
    deleteMatrix(result, aRows * bRows);

    return 0;
}
