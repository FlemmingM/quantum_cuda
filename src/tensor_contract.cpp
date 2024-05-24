#include <iostream>
#include <complex>
#include <cmath>

void contract_tensor(const std::complex<double>* state,
                     const std::complex<double> gate[2][2],
                     int qubit,
                     std::complex<double>* new_state,
                     const int* shape, int n) {
    int total_elements = std::pow(2, n);

    // Zero out new_state
    for (int i = 0; i < total_elements; ++i) {
        new_state[i] = std::complex<double>(0, 0);
    }

    // Iterate over all possible indices of the state tensor
    for (int idx = 0; idx < total_elements; ++idx) {
        int new_idx[n];
        int old_idx[n];
        int temp = idx;

        // Compute the multi-dimensional index
        for (int i = n - 1; i >= 0; --i) {
            new_idx[i] = temp % shape[i];
            temp /= shape[i];
        }

        // Perform the tensor contraction manually for the specified qubit
        for (int j = 0; j < 2; ++j) {
            // Copy new_idx to old_idx
            for (int i = 0; i < n; ++i) {
                old_idx[i] = new_idx[i];
            }
            old_idx[qubit] = j;

            // Compute the linear index for old_idx
            int old_linear_idx = 0;
            int factor = 1;
            for (int i = n - 1; i >= 0; --i) {
                old_linear_idx += old_idx[i] * factor;
                factor *= shape[i];
            }

            new_state[idx] += gate[new_idx[qubit]][j] * state[old_linear_idx];
        }
    }
}


void printState(const std::complex<double>* state, int N, std::string message) {
    std::cout << message << std::endl;
    for (int i = 0; i < N; ++i) {
        std::cout << state[i] << " ";
    }
    std::cout << std::endl;
}


void applyPhaseFlip(std::complex<double>* state, int idx){
    state[idx] *= std::complex<double>(-1.0, 0.0);
}

void applyGateAllQubits(
    std::complex<double>* state,
    const std::complex<double> gate[2][2],
    std::complex<double>* new_state,
    const int* shape,
    int n,
    int N) {

    for (int i = 0; i < n; ++i){
        contract_tensor(state, gate, i, new_state, shape, n);
        // Update the state with the new state
        for (int j = 0; j < N; ++j) {
            state[j] = new_state[j];
        }
    }
}

void applyGateSingleQubit(
    std::complex<double>* state,
    const std::complex<double> gate[2][2],
    std::complex<double>* new_state,
    const int* shape,
    int n,
    int N,
    int idx
    ) {


    contract_tensor(state, gate, idx, new_state, shape, n);
    // Update the state with the new state
    for (int j = 0; j < N; ++j) {
        state[j] = new_state[j];
    }
}


int main() {
    // Define the shape of the state tensor
    // int shape[2] = {2, 2};  // For 2 qubits
    // int n = 2;

    int n = 2;
    int N = std::pow(2, n);
    int markedState = 3;


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

    printState(state, N, "Initial state");

    // Apply Oracle
    applyPhaseFlip(state, markedState);
    // state[idx] *= std::complex<double>(-1, 0);
    printState(state, N, "Oracle applied");

    // Apply the diffusion operator ############################################
    // H gates to all qubits


// const std::complex<double>* state,
//                         const std::complex<double> gate[2][2],
//                         std::complex<double>* new_state,
//                         const int* shape,
//                         int n,
//                         int N
    applyGateAllQubits(state, H, new_state, shape, n, N);
    applyGateAllQubits(state, X, new_state, shape, n, N);
    applyPhaseFlip(state, N);
    // applyGateSingleQubit(state, Z, new_state, shape, n, N, 0);
    // applyGateAllQubits(state, X, new_state, shape, n, N);
    // applyGateSingleQubit(state, Z, new_state, shape, n, N, 0);
    // applyGateAllQubits(state, H, new_state, shape, n, N);

    printState(state, N, "After Diffusion");



    // contract_tensor(state, H, 0, new_state, shape, n);
    // // Update the state with the new state
    // for (int i = 0; i < N; ++i) {
    //     state[i] = new_state[i];
    // }







    // Initialize the new state tensor
    // std::complex<double> new_state[N];

    // Apply the Hadamard gate to the first qubit
    // contract_tensor(state, H, 0, new_state, shape, n);

    // // Print the resulting state tensor
    // std::cout << "Resulting state tensor after applying H to the first qubit:" << std::endl;
    // for (int i = 0; i < N; ++i) {
    //     std::cout << new_state[i] << " ";
    // }
    // std::cout << std::endl;

    // // Update the state with the new state
    // for (int i = 0; i < N; ++i) {
    //     state[i] = new_state[i];
    // }

    // // Apply the Hadamard gate to the second qubit
    // contract_tensor(state, H, 1, new_state, shape, n);

    //     // Update the state with the new state
    // for (int i = 0; i < N; ++i) {
    //     state[i] = new_state[i];
    // }

    // // contract_tensor(state, H, 2, new_state, shape, n);

    // // Print the resulting state tensor
    // std::cout << "Resulting state tensor after applying H to the second qubit:" << std::endl;
    // for (int i = 0; i < N; ++i) {
    //     std::cout << new_state[i] << " ";
    // }
    // std::cout << std::endl;

    return 0;
}






// #include <iostream>
// #include <vector>
// #include <complex>
// #include <cmath>

// void contract_tensor(const std::vector<std::complex<double>>& state,
//                      const std::vector<std::vector<std::complex<double>>>& gate,
//                      int qubit,
//                      std::vector<std::complex<double>>& new_state,
//                      const std::vector<int>& shape) {
//     // Initialize a new state tensor to hold the result
//     std::fill(new_state.begin(), new_state.end(), std::complex<double>(0, 0));

//     int n = shape.size();
//     int total_elements = 1;
//     for (int dim : shape) {
//         total_elements *= dim;
//     }

//     std::cout << "num axes: " << n << std::endl;
//     std::cout << "total_elements: " << total_elements << std::endl;


//     // Iterate over all possible indices of the state tensor
//     for (int idx = 0; idx < total_elements; ++idx) {
//         std::vector<int> new_idx(n);
//         std::vector<int> old_idx(n);
//         int temp = idx;

//         // Compute the multi-dimensional index
//         for (int i = n - 1; i >= 0; --i) {
//             new_idx[i] = temp % shape[i];
//             temp /= shape[i];
//         }

//         // Perform the tensor contraction for the specified qubit
//         // We only iterate over 2 since all gates are 2x2 matrices
//         for (int j = 0; j < 2; ++j) {
//             old_idx = new_idx;
//             old_idx[qubit] = j;

//             // Compute the linear index for old_idx
//             int old_linear_idx = 0;
//             int factor = 1;
//             for (int i = n - 1; i >= 0; --i) {
//                 old_linear_idx += old_idx[i] * factor;
//                 std::cout << "old_linear_idx " << old_linear_idx << std::endl;
//                 factor *= shape[i];
//             }

//             new_state[idx] += gate[new_idx[qubit]][j] * state[old_linear_idx];
//         }
//     }
// }

// int main() {
//     // Define the shape of the state tensor
//     // std::vector<int> shape = {2, 2};  // For 2 qubits
//     std::vector<int> shape = {2, 2, 2};  // For 3 qubits

//     // Define the initial state tensor for 2 qubits (|00>)
//     // std::vector<std::complex<double>> state = {1, 0, 0, 0};
//     std::vector<std::complex<double>> state = {1, 0, 0, 0, 0, 0, 0, 0};

//     // Define the Hadamard gate
//     std::vector<std::vector<std::complex<double>>> H = {
//         {1 / std::sqrt(2.0), 1 / std::sqrt(2.0)},
//         {1 / std::sqrt(2.0), -1 / std::sqrt(2.0)}
//     };

//     // Initialize the new state tensor
//     std::vector<std::complex<double>> new_state(state.size());

//     // Apply the Hadamard gate to the first qubit
//     contract_tensor(state, H, 0, new_state, shape);

//     // Print the resulting state tensor
//     std::cout << "Resulting state tensor after applying H to the first qubit:" << std::endl;
//     for (const auto& val : new_state) {
//         std::cout << val << " ";
//     }
//     std::cout << std::endl;

//     // Update the state with the new state
//     state = new_state;

//     // Apply the Hadamard gate to the second qubit
//     contract_tensor(state, H, 1, new_state, shape);
//     state = new_state;
//     contract_tensor(state, H, 2, new_state, shape);

//     // Print the resulting state tensor
//     std::cout << "Resulting state tensor after applying H to the second qubit:" << std::endl;
//     for (const auto& val : new_state) {
//         std::cout << val << " ";
//     }
//     std::cout << std::endl;

//     return 0;
// }
