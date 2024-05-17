#ifndef NODES_H
#define NODES_H

#include <complex>

using Complex = std::complex<double>;

class Node {
private:
    int rows, cols;

public:
    Complex** tensor;
    Node(int rows, int cols);
    ~Node();
    void setTensorData(Complex** data);
    void printDetails() const;
    Node contract(const Node& other);
};

#endif // NODES_H
