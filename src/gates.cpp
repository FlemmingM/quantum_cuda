#include <iostream>
#include <complex>

using namespace std;

// Define a complex number type for simplicity
using Complex = complex<double>;

// Basic tensor class for holding quantum states or operations
class Tensor {
public:
    Complex* data;   // Dynamic array to hold the tensor data
    int dim;         // Dimension of the tensor (2 for a qubit)

    // Constructor
    Tensor(int dim) : dim(dim) {
        data = new Complex[dim];  // Allocate memory for the tensor
        for (int i = 0; i < dim; ++i) {
            data[i] = 0;  // Initialize the data elements to zero
        }
    }

    // Destructor
    ~Tensor() {
        delete[] data;  // Properly deallocate memory
    }

    // Copy constructor for deep copying
    Tensor(const Tensor& other) : dim(other.dim) {
        data = new Complex[dim];
        for (int i = 0; i < dim; ++i) {
            data[i] = other.data[i];
        }
    }

    // Assignment operator for deep copying
    Tensor& operator=(const Tensor& other) {
        if (this != &other) {  // Prevent self-assignment
            delete[] data;  // Free existing data
            dim = other.dim;
            data = new Complex[dim];
            for (int i = 0; i < dim; ++i) {
                data[i] = other.data[i];
            }
        }
        return *this;
    }

    // Accessor for tensor elements
    Complex& operator[](int index) {
        return data[index];
    }

    const Complex& operator[](int index) const {
        return data[index];
    }

    // Apply a single-qubit gate to this tensor
    void applyGate(const Tensor& gate, int qubitIndex) {
        Tensor result(dim);
        for (int i = 0; i < dim; i++) {
            result[i] = 0;
            for (int j = 0; j < dim; j++) {
                result[i] += gate[(i >> qubitIndex) & 1] * data[(i & ~(1 << qubitIndex)) | (j << qubitIndex)];
            }
        }
        *this = result;
    }

    // print the current state of the object
    void printState() const {
        for (int i = 0; i < dim; ++i) {
            if (imag(data[i]) >= 0)
                cout << real(data[i]) << "+" << imag(data[i]) << "i ";
            else
                cout << real(data[i]) << "-" << -imag(data[i]) << "i ";
        }
        cout << endl;
    }
};







int main() {
    // Initialize a simple two-qubit state |00>
    Tensor state(4);
    state[0] = 1;  // Coefficient for |00>

    printf("Initial state \n");
    state.printState();

    // Define Pauli X gate
    Tensor X(4);
    X[0] = 0; X[1] = 1;
    X[2] = 1; X[3] = 0;

    // Define Pauli Z gate
    Tensor Z(4);
    Z[0] = 1; Z[1] = 0;
    Z[2] = 0; Z[3] = -1;

    // Apply X gate to the first qubit
    state.applyGate(X, 0);

    printf("Apply X \n");
    state.printState();

    // Apply Z gate to the second qubit
    state.applyGate(Z, 1);
    printf("Apply Z \n");
    state.printState();

    // Print final state coefficients
    // cout << "State after applying X to qubit 1 and Z to qubit 2:" << endl;
    // for (int i = 0; i < state.dim; i++) {
    //     cout << state[i] << " ";
    // }
    // cout << endl;

    return 0;
}
