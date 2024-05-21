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
    Complex xValues[] = {Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)};
    Complex q0Values[] = {Complex(1.0, 0.0), Complex(0.0, 0.0)};
    Complex q1Values[] = {Complex(0.0, 0.0), Complex(1.0, 0.0)};

    // Init the qubit matrices and gates
    Complex** q0_M = createMatrix(2, 1, q0Values);
    Complex** q1_M = createMatrix(2, 1, q1Values);
    Complex** H_M = createMatrix(2, 2, hValues);
    Complex** I_M = createMatrix(2, 2, iValues);
    Complex** X_M = createMatrix(2, 2, iValues);


    // Complex** state = kroneckerProduct(q0, 2, 1, q1, 2, 1);
    // printMatrix(state, 4, 1);

    // Complex** state2 = kroneckerProduct(H, 2, 2, X, 2, 2);
    // printMatrix(state2, 4, 4);

    Node q0(2, 1);
    Node q1(2, 1);
    Node H(2, 2);
    Node X(2, 2);

    q0.setTensorData(q0_M);
    q1.setTensorData(q1_M);
    H.setTensorData(H_M);
    X.setTensorData(X_M);

    std::cout << "q0 details:" << std::endl;
    q0.printDetails();
    std::cout << "q1 details:" << std::endl;
    q1.printDetails();
    std::cout << "H details:" << std::endl;
    H.printDetails();
    std::cout << "X details:" << std::endl;
    X.printDetails();

    // Apply the H and X gates
    Node res_a = H.contract(q0);
    Node res_b = X.contract(q1);
    Node res_c = res_b.contract(res_a);
    res_c.printDetails();
    // q0.tensor = q0.tensor * Complex(-1.0, 0.0);





    // Node contractedNode = nodeB.contract(nodeA);
    // std::cout << "Contracted Node (B * A):" << std::endl;
    // contractedNode.printDetails();

    // // make this an input variable
    // // also add a var for the solution --> for the oracle
    // int numQubits = 4;
    // Complex* result = flattenMatrix(contractedNode.tensor, numQubits, 1);
    // double* averages = simulate(result, numQubits, 10000);

    // std::cout << "Average frequency per position:" << std::endl;
    //     for (int i = 0; i < numQubits; ++i) {
    //         std::cout << "Position " << i << ": " << averages[i] << std::endl;
    //     }

    //     delete[] averages;  // Don't forget to free the allocated memory


    // Node contractedNode2 = contractedNode.contract(nodeB);
    // std::cout << "Contracted Node (C * B):" << std::endl;
    // contractedNode2.printDetails();
    // return 0;

}