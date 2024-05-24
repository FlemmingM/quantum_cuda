#include <iostream>
#include <complex>
#include <cmath>
#include "utils.h"
#include "nodes.h"


// Define a complex number type for simplicity
using Complex = std::complex<double>;


int main() {
    int n = 2;
    double normFactor = std::sqrt(2);
    // double normFactor = std::sqrt(std::pow(2, n));
    // Complex A[] = {Complex(1.0, 0.0), Complex(0.0, 0.0)};
    Complex hValues[] = {Complex(1.0 / normFactor, 0.0), Complex(1.0 / normFactor, 0.0),
                        Complex(1.0 / normFactor, 0.0), Complex(-1.0 / normFactor, 0.0)};
    Complex iValues[] = {Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(1.0, 0.0)};
    Complex xValues[] = {Complex(0.0, 0.0), Complex(1.0, 0.0), Complex(1.0, 0.0), Complex(0.0, 0.0)};
    Complex zValues[] = {Complex(1.0, 0.0), Complex(0.0, 0.0), Complex(0.0, 0.0), Complex(-1.0, 0.0)};
    Complex q0Values[] = {Complex(1.0, 0.0), Complex(0.0, 0.0)};
    Complex q1Values[] = {Complex(0.0, 0.0), Complex(1.0, 0.0)};

    // Init the qubit matrices and gates
    Complex** q0_M = createMatrix(2, 1, q0Values);
    Complex** q1_M = createMatrix(2, 1, q1Values);
    Complex** H_M = createMatrix(2, 2, hValues);
    Complex** I_M = createMatrix(2, 2, iValues);
    Complex** X_M = createMatrix(2, 2, xValues);
    Complex** Z_M = createMatrix(2, 2, zValues);


    // Complex** state = kroneckerProduct(q0, 2, 1, q1, 2, 1);
    // printMatrix(state, 4, 1);

    // Complex** state2 = kroneckerProduct(H, 2, 2, X, 2, 2);
    // printMatrix(state2, 4, 4);

    // Init the qubits and gates
    Edge q0(2, 1);
    Edge q1(2, 1);
    Edge w0(2, 1);
    Edge w1(2, 1);
    Node H(2, 2);
    Node X(2, 2);
    Node Z(2, 2);

    // Complex superPosVals[] = {Complex(1.0, 0.0)/std::sqrt(n), Complex(1.0, 0.0)/std::sqrt(n)};
    // Complex** superPos_M = createMatrix(2, 1, superPosVals);

    // Edge* edges = new Edge[n];

    // // Initialize each Edge object
    // for (int i = 0; i < n; ++i) {
    //     edges[i] = Edge(2, 1);
    //     edges[i].setTensorData(superPos_M);
    // }

    q0.setTensorData(q0_M);
    q1.setTensorData(q1_M);
    H.setTensorData(H_M);
    X.setTensorData(X_M);
    Z.setTensorData(Z_M);

    // 1) Init the qubit register in superposition
    Edge* qRegister = initQubits(2, 1/std::sqrt(std::pow(2, n)));
    Edge* qWorkingRegister = initQubits(2, 1/std::sqrt(std::pow(2, n)));
    std::cout << "1) Initial state:" << std::endl;
    // qRegister[0].printDetails();
    showRegister(qRegister, n);


    // // 2) Apply the oracle which flips the state of the marked solution
    applyPhaseFlip(qRegister, 2, 1, 1);
    std::cout << "2) Apply Oracle:" << std::endl;
    showRegister(qRegister, n);



    // 3) Apply Diffusion Operator
    std::cout << "3) Apply Diffusion Operator:" << std::endl;
    for (int i=0; i<n; ++i) {
            H.contractInPlace(qRegister[i], qWorkingRegister[i]);
    }
    showRegister(qRegister, n);

    for (int i=0; i<n; ++i) {
        X.contractInPlace(qRegister[i], qWorkingRegister[i]);
    }
    showRegister(qRegister, n);

    // Apply controlled Z with control on all but one qubit
    // This is equal to flipping the state with only 1s
    applyPhaseFlip(qRegister, n, n-1, n-1);
    showRegister(qRegister, n);

    // Apply Z gate to 0th qubit
    Z.contractInPlace(qRegister[0], qWorkingRegister[0]);
    showRegister(qRegister, n);

    for (int i=0; i<n; ++i) {
        X.contractInPlace(qRegister[i], qWorkingRegister[i]);
    }
    showRegister(qRegister, n);

    // Apply Z gate to 0th qubit
    Z.contractInPlace(qRegister[0], qWorkingRegister[0]);
    showRegister(qRegister, n);

    for (int i=0; i<n; ++i) {
        H.contractInPlace(qRegister[i], qWorkingRegister[i]);
    }
    showRegister(qRegister, n);

    // H.contractInPlace(qReg[0], result);

    // qReg[0].printDetails();

    // Edge* qRegister[n] = {&q0, &q1};
    // Edge* qWorkingRegister[n]{&w0, &w1};

    // 2) Apply the oracle which flips the state of the marked solution
    // applyPhaseFlip(qRegister, 2, 1, 1);

    // qRegister[0]->printDetails();

    // qRegister[0]->printDetails();

    // X.contractInPlace(*qRegister[0], *qWorkingRegister[0]);

    // qRegister[0]->printDetails();
    // // result.printDetails();

    // X.contractInPlace(*qRegister[0], *qWorkingRegister[0]);
    // qRegister[0]->printDetails();
    // result.printDetails();

    // X.contract(*qRegister[0], result);
    // result.printDetails();

    // for (int i=0; i<3; ++i){
    //     X.contract(q0, result);
    //     // result.printDetails();

    //     // Change the pointers
    //     Edge *tmp = qRegister[0];
    //     *qRegister[0] = *result;
    //     result = tmp;

    //     qRegister[0]->printDetails();
    // }

    // X.contract(result, *qRegister[0]);
    // result.printDetails();
    // X.contract(*qRegister[0], result);
    // result.printDetails();

    // 3) Apply the diffusion operator

    // std::cout << "q0 details:" << std::endl;
    // q0.printDetails();
    // std::cout << "q1 details:" << std::endl;
    // q1.printDetails();
    // std::cout << "H details:" << std::endl;
    // H.printDetails();
    // std::cout << "X details:" << std::endl;
    // X.printDetails();


    // Edge newEdge(2, 1);
    // for (int i=0; i < 2; ++i) {
    //     for (int j = 0; j < 1; ++j) {
    //         for (int k = 0; k < 2; ++k) {
    //             newEdge.tensor[i][j] += X.tensor[i][k] * q0.tensor[k][j];
    //         }
    //     }
    // }

    // newEdge.printDetails();

    // qRegister[0].printDetails();

    // Apply the H and X gates
    // Edge res_a = H.contract(q0);
    // Edge res_b = X.contract(q1);

    // res_a.printDetails();
    // Node res_c = res_b.contract(res_a);
    // res_c.printDetails();
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