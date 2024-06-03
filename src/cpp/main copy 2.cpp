#include <iostream>
#include <cstdlib>
// #include <complex>
#include <cmath>
#include "utils.h"
// #include "nodes.h"


// Define a complex number type for simplicity
// using Complex = std::complex<double>;


int main(int argc, char* argv[]) {

    // collect input args
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " n qubits<int>; marked state<int>; number of samples<int>; fileName<string>; verbose 0 or 1<int>" << std::endl;
        return 1;
    }


    int n = std::atoi(argv[1]);;
    int N = std::pow(2, n);
    int markedState = std::atoi(argv[2]);
    int numSamples = std::atoi(argv[3]);
    std::string fileName = argv[4];
    int verbose = std::atoi(argv[5]);

    // Assuming we have t = 1 solution in grover's algorithm
    // we have k = floor(pi/4 * sqrt(N))
    int k = std::floor(M_PI / 4 * std::sqrt(N));


    // Define the gates
    std::complex<double> H[2][2] = {
        {1 / std::sqrt(2.0), 1 / std::sqrt(2.0)},
        {1 / std::sqrt(2.0), -1 / std::sqrt(2.0)}
    };

    std::complex<double> I[2][2] = {
        {1, 0},
        {0, 1}
    };

    std::complex<double> Z[2][2] = {
        {1, 0},
        {0, -1}
    };

    std::complex<double> X[2][2] = {
        {0, 1},
        {1, 0}
    };

    // Dynamically allocate an array of size n with 2 dimensions each for each qubit
    int* shape = new int[n];
    for (int i = 0; i < n; ++i) {
        shape[i] = 2;
    }


    // Init a superposition of qubits
    std::complex<double>* state = new std::complex<double>[N];
    for (int i = 0; i < N; ++i) {
        state[i] = std::complex<double>(1 / std::sqrt(N), 0);
    }

    // Initialize the new state tensor
    std::complex<double> new_state[N];

    // if (verbose == 1) {
    //     printState(state, N, "Initial state");
    // }

    // Apply Grover's algorithm k iteration and then sample
    // if (verbose == 1) {
    //     std::cout << "Running " << k << " round(s)"  << std::endl;
    // }
    for (int i = 0; i < k; ++i) {
        if (verbose == 1) {
            std::cout << i << "/" << k << std::endl;
        }
        // Apply Oracle
        applyPhaseFlip(state, markedState);
        // if (verbose == 1) {

        //     printState(state, N, "Oracle applied");
        // }
        // Apply the diffusion operator ############################################
        applyDiffusionOperator(state, new_state, shape, H, X, Z, n, N);
        // if (verbose == 1) {
        //     printState(state, N, "After Diffusion");
        // }
    }


    double* averages = simulate(state, N, numSamples);
    std::cout << "Average frequency per position:" << std::endl;
        for (int i = 0; i < N; ++i) {
            std::cout << "Position " << i << ": " << averages[i] << std::endl;
        }

    // save the data
    saveArrayToCSV(averages, N, fileName);

    delete[] averages;

    return 0;

}