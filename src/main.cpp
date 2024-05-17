#include <iostream>
#include <complex>
#include <cmath>
#include "utils.h"
#include "nodes.h"


// Define a complex number type for simplicity
using Complex = std::complex<double>;


int main() {
    double normFactor = std::sqrt(2);
    // Complex A[] = {Complex(1.0, 0.0), Complex(0.0, 0.0)};
    Complex hValues[] = {Complex(1.0 / normFactor, 0.0), Complex(1.0 / normFactor, 0.0),
                        Complex(1.0 / normFactor, 0.0), Complex(-1.0 / normFactor, 0.0)};
    Complex iValues[] = {Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)};
    Complex q0Values[] = {Complex(1.0, 0.0), Complex(0.0, 0.0)};
    Complex q1Values[] = {Complex(0.0, 0.0), Complex(1.0, 0.0)};

    Complex** q0 = createMatrix(2, 1, q0Values);
    Complex** q1 = createMatrix(2, 1, q1Values);
    Complex** H = createMatrix(2, 2, hValues);
    Complex** I = createMatrix(2, 2, iValues);


    Complex** state = kroneckerProduct(q0, 2, 1, q1, 2, 1);
    printMatrix(state, 4, 1);

    Complex** state2 = kroneckerProduct(H, 2, 2, I, 2, 2);
    printMatrix(state2, 4, 4);

    Node nodeA(4, 1);
    Node nodeB(4, 4);
    nodeA.setTensorData(state);
    nodeB.setTensorData(state2);

    std::cout << "Node A details:" << std::endl;
    nodeA.printDetails();
    std::cout << "Node B details:" << std::endl;
    nodeB.printDetails();

    Node contractedNode = nodeB.contract(nodeA);
    std::cout << "Contracted Node (B * A):" << std::endl;
    contractedNode.printDetails();

    int numElements = 4;
    Complex* result = flattenMatrix(contractedNode.tensor, numElements, 1);
    double* averages = simulate(result, numElements, 10000);

    std::cout << "Average frequency per position:" << std::endl;
        for (int i = 0; i < numElements; ++i) {
            std::cout << "Position " << i << ": " << averages[i] << std::endl;
        }

        delete[] averages;  // Don't forget to free the allocated memory


    // Node contractedNode2 = contractedNode.contract(nodeB);
    // std::cout << "Contracted Node (C * B):" << std::endl;
    // contractedNode2.printDetails();
    // return 0;

}