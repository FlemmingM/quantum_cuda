#include <iostream>
#include <random>
#include <complex>

typedef std::complex<double> Complex;

// Function to perform weighted random selection and calculate average selection frequency per position
double* weightedRandomAverage(const double* weights, int numElements, int numSamples) {
    if (numElements <= 0 || numSamples <= 0) {
        std::cerr << "Invalid input parameters." << std::endl;
        return nullptr;
    }

    int* counts = new int[numElements]();  // Array to count occurrences of each index, initialized to zero
    double* averages = new double[numElements](); // Array to store the average frequencies

    std::random_device rd;
    std::mt19937 gen(rd());
    std::discrete_distribution<> dist(weights, weights + numElements);

    for (int i = 0; i < numSamples; ++i) {
        int index = dist(gen);  // Generate a weighted index
        counts[index]++;        // Increment the count for this index
    }

    for (int i = 0; i < numElements; ++i) {
        averages[i] = static_cast<double>(counts[i]) / numSamples;  // Calculate the average frequency of selection
    }

    delete[] counts;  // Clean up the counts array
    return averages;
}

int main() {
    // Example weights
    int numElements = 2;
    double weights[] = {0.2, 0.8};
    int numSamples = 10000;  // Number of samples

    // Calculate average frequency per position
    double* averages = weightedRandomAverage(weights, numElements, numSamples);

    // Output the results
    if (averages) {
        std::cout << "Average frequency per position:" << std::endl;
        for (int i = 0; i < numElements; ++i) {
            std::cout << "Position " << i << ": " << averages[i] << std::endl;
        }

        delete[] averages;  // Don't forget to free the allocated memory
    }

    return 0;
}
