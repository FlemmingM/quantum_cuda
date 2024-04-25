#include <iostream>
#include <complex>
#include <cmath>
#include "nodes.h"


// Define a complex number type for simplicity
using Complex = std::complex<double>;
Node::Node(int rows, int cols) : rows(rows), cols(cols) {
    tensor = new Complex*[rows];
    for (int i = 0; i < rows; ++i) {
        tensor[i] = new Complex[cols];
    }
}

Node::~Node() {
    for (int i = 0; i < rows; ++i) {
        delete[] tensor[i];
    }
    delete[] tensor;
}

void Node::setTensorData(const Complex* data) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            tensor[i][j] = data[i * cols + j];
        }
    }
}

void Node::printDetails() const {
    std::cout << "Tensor (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << tensor[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Contract this node with another node (matrix multiplication for 2D tensors)
Node Node::contract(const Node& other) {
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication!");
    }
    Node result(rows, other.cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            for (int k = 0; k < cols; ++k) {
                result.tensor[i][j] += tensor[i][k] * other.tensor[k][j];
            }
        }
    }
    return result;
}


// int main() {

//     double normFactor = std::sqrt(2);
//     Complex dataA[] = {Complex(1.0, 0.0), Complex(0.0, 0.0)};
//     Complex dataB[] = {Complex(1.0 / normFactor, 0.0), Complex(1.0 / normFactor, 0.0),
//                         Complex(1.0 / normFactor, 0.0), Complex(-1.0 / normFactor, 0.0)};

//     // Complex state = kroneckerProduct(dataA, dataA);

//     Node nodeA(1, 2);
//     Node nodeB(2, 2);
//     nodeA.setTensorData(dataA);
//     nodeB.setTensorData(dataB);

//     std::cout << "Node A details:" << std::endl;
//     nodeA.printDetails();
//     std::cout << "Node B details:" << std::endl;
//     nodeB.printDetails();

//     Node contractedNode = nodeA.contract(nodeB);
//     std::cout << "Contracted Node (A * B):" << std::endl;
//     contractedNode.printDetails();

//     Node contractedNode2 = contractedNode.contract(nodeB);
//     std::cout << "Contracted Node (C * B):" << std::endl;
//     contractedNode2.printDetails();
//     return 0;
// }
