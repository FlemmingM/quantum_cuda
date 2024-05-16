#ifndef NODES_H
#define NODES_H

#include <complex>

using Complex = std::complex<double>;

class Node {
private:
    Complex** tensor;
    int rows, cols;

public:
    Node(int rows, int cols);
    ~Node();
    void setTensorData(Complex** data);
    void printDetails() const;
    Node contract(const Node& other);
};

#endif // NODES_H
