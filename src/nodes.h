#ifndef NODES_H
#define NODES_H

#include <complex>

using Complex = std::complex<double>;

// Edge class
class Edge {

public:
    int rows, cols;
    Complex** tensor;
    Edge();
    Edge(int rows, int cols);
    Edge(const Edge& other); // Copy constructor
    Edge& operator=(const Edge& other); // Copy assignment operator
    ~Edge();
    void setTensorData(Complex** data);
    void printDetails() const;
    Complex** getTensor() const;
    void allocateTensor();
    void deallocateTensor();
};


// Node class
class Node {

public:
    int rows, cols;
    Complex** tensor;
    Node(int rows, int cols);
    ~Node();
    void setTensorData(Complex** data);
    void printDetails() const;
    void contract(const Edge& other, Edge& result) const;
    void contractInPlace(const Edge& other, Edge& result) const;
};

#endif // NODES_H
