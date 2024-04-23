#include <iostream>
#include <complex>
#include <stdexcept>

using namespace std;

using Complex = complex<double>;

class Node {
private:
    Complex* data;   // Dynamic array to hold tensor data
    int* edges;      // Dynamic array to hold edges information
    size_t numEdges; // Number of edges
    size_t dataSize; // Size of the data array

public:
    // Constructor
    Node(size_t numEdges, size_t dataSize) : numEdges(numEdges), dataSize(dataSize) {
        data = new Complex[dataSize];
        edges = new int[numEdges];
        for (size_t i = 0; i < dataSize; i++) {
            data[i] = 0;  // Initialize data elements to zero
        }
    }

    // Destructor
    ~Node() {
        delete[] data;
        delete[] edges;
    }

    // Copy constructor for deep copy
    Node(const Node& other) : numEdges(other.numEdges), dataSize(other.dataSize) {
        data = new Complex[dataSize];
        edges = new int[numEdges];
        for (size_t i = 0; i < dataSize; i++) {
            data[i] = other.data[i];
        }
        for (size_t i = 0; i < numEdges; i++) {
            edges[i] = other.edges[i];
        }
    }

    // Assignment operator for deep copy
    Node& operator=(const Node& other) {
        if (this != &other) {
            delete[] data;
            delete[] edges;
            dataSize = other.dataSize;
            numEdges = other.numEdges;
            data = new Complex[dataSize];
            edges = new int[numEdges];
            for (size_t i = 0; i < dataSize; i++) {
                data[i] = other.data[i];
            }
            for (size_t i = 0; i < numEdges; i++) {
                edges[i] = other.edges[i];
            }
        }
        return *this;
    }

    // Set edges information
    void setEdges(const int* newEdges) {
        for (size_t i = 0; i < numEdges; i++) {
            edges[i] = newEdges[i];
        }
    }

    // Access and modify tensor data
    Complex& operator[](size_t index) {
        return data[index];
    }

    // Print the tensor data
    void print() {
        for (size_t i = 0; i < dataSize; i++) {
            cout << data[i] << " ";
        }
        cout << endl;
    }

    // Contract this node with another node
    static Node contract(const Node& a, const Node& b, int aIndex, int bIndex) {
        if (a.edges[aIndex] != b.edges[bIndex]) {
            throw runtime_error("Incompatible edges for contraction.");
        }

        // Calculate new data size and edges configuration
        size_t newNumEdges = a.numEdges + b.numEdges - 2;
        int* newEdges = new int[newNumEdges];
        size_t dataIndex = 0;
        for (size_t i = 0; i < a.numEdges; i++) {
            if (i != aIndex) {
                newEdges[dataIndex++] = a.edges[i];
            }
        }
        for (size_t i = 0; i < b.numEdges; i++) {
            if (i != bIndex) {
                newEdges[dataIndex++] = b.edges[i];
            }
        }

        // Determine the size of the new data array
        size_t newDataSize = 1;
        for (size_t i = 0; i < newNumEdges; i++) {
            newDataSize *= newEdges[i];
        }

        Node result(newNumEdges, newDataSize);
        result.setEdges(newEdges);
        delete[] newEdges;

        // Perform contraction (simple case for demonstration)
        for (size_t i = 0; i < a.dataSize; i++) {
            for (size_t j = 0; j < b.dataSize; j++) {
                if ((i % a.edges[aIndex]) == (j % b.edges[bIndex])) {
                    result.data[(i / a.edges[aIndex]) * (b.dataSize / b.edges[bIndex]) + (j / b.edges[bIndex])] +=
                        a.data[i] * b.data[j];
                }
            }
        }

        return result;
    }
};

int main() {
    // Create two nodes
    Node a(2, 4), b(2, 4);

    // print the 2 nodes
    a.print();
    b.print();

    int aEdges[2] = {1, 0}, bEdges[2] = {1, 0};
    a.setEdges(aEdges);
    b.setEdges(bEdges);

    // Initialize some data
    a[0] = 1; a[1] = 0; a[2] = 0; a[3] = -1;
    b[0] = 1; b[1] = 0; b[2] = 0; b[3] = 1;

    a.print();
    b.print();

    // Contract nodes on the second index of each
    Node c = Node::contract(a, b, 1, 1);

    cout << "Result of contraction: " << endl;
    c.print();

    return 0;
}
