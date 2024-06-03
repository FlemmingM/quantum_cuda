#include <iostream>
#include <complex>
#include <cmath>
#include <cstring>
#include "nodes.h"

typedef std::complex<double> Complex;


// TODO: define an edge class which uses the kron product to contract parallel edges
// we will still need standard contract as well to contract with nodes

Edge::Edge() : rows(0), cols(0), tensor(nullptr) {}

// Parameterized constructor
Edge::Edge(int rows, int cols) : rows(rows), cols(cols) {
    allocateTensor();
}

// Edge::Edge(int rows, int cols) : rows(rows), cols(cols) {
//     tensor = new Complex*[rows];
//     for (int i = 0; i < rows; ++i) {
//         tensor[i] = new Complex[cols];
//     }
// }

// Copy constructor
Edge::Edge(const Edge& other) : rows(other.rows), cols(other.cols) {
    allocateTensor();
    for (int i = 0; i < rows; ++i) {
        std::memcpy(tensor[i], other.tensor[i], cols * sizeof(Complex));
    }
}

// Copy assignment operator
Edge& Edge::operator=(const Edge& other) {
    if (this == &other) {
        return *this;
    }
    deallocateTensor();
    rows = other.rows;
    cols = other.cols;
    allocateTensor();
    for (int i = 0; i < rows; ++i) {
        std::memcpy(tensor[i], other.tensor[i], cols * sizeof(Complex));
    }
    return *this;
}

// Destructor
Edge::~Edge() {
    deallocateTensor();
}

// Edge::~Edge() {
//     for (int i = 0; i < rows; ++i) {
//         delete[] tensor[i];
//     }
//     delete[] tensor;
// }

void Edge::allocateTensor() {
    tensor = new Complex*[rows];
    for (int i = 0; i < rows; ++i) {
        tensor[i] = new Complex[cols];
    }
}

void Edge::deallocateTensor() {
    if (tensor) {
        for (int i = 0; i < rows; ++i) {
            delete[] tensor[i];
        }
        delete[] tensor;
    }
}

void Edge::setTensorData(Complex** data) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            tensor[i][j] = data[i][j];
        }
    }
}

void Edge::printDetails() const {
    std::cout << "Edge: Tensor (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << tensor[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}




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

void Node::setTensorData(Complex** data) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            tensor[i][j] = data[i][j];
        }
    }
}

void Node::printDetails() const {
    std::cout << "Node: Tensor (" << rows << "x" << cols << "):\n";
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << tensor[i][j] << " ";
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}

// Contract this node with another edge (matrix multiplication for 2D tensors)
void Node::contract(const Edge& other, Edge& result) const{
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication!");
    }
    // Edge result(rows, other.cols);

    // init result to 0
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            result.tensor[i][j] = Complex(0, 0); // Initialize to zero
        }
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            for (int k = 0; k < cols; ++k) {
                result.tensor[i][j] += tensor[i][k] * other.tensor[k][j];
            }
        }
    }
}

void Node::contractInPlace(const Edge& other, Edge& result) const{
    if (cols != other.rows) {
        throw std::invalid_argument("Matrix dimensions must match for multiplication!");
    }
    // Edge result(rows, other.cols);

    // init result to 0
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            result.tensor[i][j] = Complex(0, 0); // Initialize to zero
        }
    }

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < other.cols; ++j) {
            for (int k = 0; k < cols; ++k) {
                result.tensor[i][j] += tensor[i][k] * other.tensor[k][j];
            }
        }
    }

    // Overwrite the initial Edge with result
    for (int i = 0; i < result.rows; ++i) {
        for (int j = 0; j < result.cols; ++j) {
            other.tensor[i][j] = result.tensor[i][j];
        }
    }
}


