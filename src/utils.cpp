#include <iostream>
#include <complex>
#include <random>
#include <cmath>
#include "utils.h"
#include "nodes.h"
typedef std::complex<double> Complex;


void showRegister(Edge* qRegister, int n) {
    for (int i=0; i<n; ++i) {
        qRegister[i].printDetails();
    }
    std::cout << "----------------" << std::endl;
}


void applyPhaseFlip(Edge* qRegister, int n, int state_a, int state_b) {
    for (int i=0; i<n; ++i) {
        if (i == state_a) {
            qRegister[state_a].tensor[state_b][0] *= -1;
        }
        // qRegister[i].printDetails();
    }
}

void applyDiffusionOperator(Edge* qRegister[], int n) {

}

// Edge* initQubits(int n, float factor) {

//     // init the register
//     Edge* edges = new Edge[n];
//     // Initialize each Edge object
//     for (int i = 0; i < n; ++i) {
//         edges[i] = Edge(2, 1);
//     }

//     // Optionally set tensor data for each Edge object
//     for (int i = 0; i < n; ++i) {
//         // Complex** data = new Complex*[2];
//         Complex data[] = {Complex(1.0 * factor, 0.0), Complex(1.0 * factor, 0.0)};
//         Complex** matrix = createMatrix(2, 1, data);
//         // for (int j = 0; j < 2; ++j) {
//         //     data[j] = new Complex[1];
//         //     for (int k = 0; k < 1; ++k) {
//         //         data[j][k] = Complex(1.0 * factor, 0.0);
//         //     }
//         // }
//         edges[i].setTensorData(matrix);

//         // // Clean up allocated data to avoid memory leaks
//         // for (int j = 0; j < 2; ++j) {
//         //     delete[] data[j];
//         // }
//     }
//     return edges;
// }

Edge* initQubits(int n, float factor) {
    // Initialize the register
    Edge* edges = new Edge[n];

    // Initialize each Edge object
    for (int i = 0; i < n; ++i) {
        edges[i] = Edge(2, 1);
    }

    // Optionally set tensor data for each Edge object
    for (int i = 0; i < n; ++i) {
        Complex** data = new Complex*[2];
        for (int j = 0; j < 2; ++j) {
            data[j] = new Complex[1];
            data[j][0] = Complex(1.0 * factor, 0.0); // Example initialization
        }
        edges[i].setTensorData(data);

        // edges[i].printDetails();

        // Clean up allocated data to avoid memory leaks
        // for (int j = 0; j < 2; ++j) {
        //     delete[] data[j];
        // }
        // delete[] data;
    }

    return edges;
}


// Complex* flattenMatrix(Complex** matrix, int rows, int cols) {
//     Complex* flatMatrix = new Complex[rows * cols];
//     for (int i = 0; i < rows; ++i) {
//         for (int j = 0; j < cols; ++j) {
//             flatMatrix[i * cols + j] = matrix[i][j];
//         }
//     }
//     return flatMatrix;
// }


double* simulate(const Complex* weights, int numElements, int numSamples) {
    if (numElements <= 0 || numSamples <= 0) {
        std::cerr << "Invalid input parameters." << std::endl;
        return nullptr;
    }

    // Array to count occurrences of each index
    int* counts = new int[numElements]();
    // Array to store the average frequencies
    double* averages = new double[numElements]();

    std::random_device rd;
    std::mt19937 gen(rd());

    // Prepare weights for the distribution by extracting their magnitudes
    double* magnitudes = new double[numElements];
    for (int i = 0; i < numElements; ++i) {
        magnitudes[i] = std::abs(weights[i]);
    }

    // Create a weighted distribution based on magnitudes of the complex weights
    std::discrete_distribution<> dist(magnitudes, magnitudes + numElements);

    for (int i = 0; i < numSamples; ++i) {
        int index = dist(gen);  // Generate a weighted index
        counts[index]++;        // Increment the count for this index
    }

    for (int i = 0; i < numElements; ++i) {
        averages[i] = static_cast<double>(counts[i]) / numSamples;  // Calculate the average frequency of selection
    }

    delete[] counts;          // Clean up the counts array
    delete[] magnitudes;      // Clean up the magnitudes array
    return averages;
}


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


// Function to create a tensor of qubits
// Complex** createQState(qubits* int, numElements int) {
//     for (int i = 0; i < numElements; i++) {

//     }
// }


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
