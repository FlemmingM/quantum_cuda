#include <iostream>
#include <random>
#include <complex>
#include <cmath> // For std::abs

typedef std::complex<double> Complex;

// Function to perform weighted random selection and calculate average selection frequency per position
double* simulate(const Complex* weights, int numElements, int numSamples) {
    if (numElements <= 0 || numSamples <= 0) {
        std::cerr << "Invalid input parameters." << std::endl;
        return nullptr;
    }

    // Array to count occurrences of each index
    int* counts = new int[numElements]();
    // Array to store the average frequencies
    double* averages = new double[numElements]();

    std::random_device rd;
    std::mt19937 gen(rd());

    // Prepare weights for the distribution by extracting their magnitudes
    double* magnitudes = new double[numElements];
    for (int i = 0; i < numElements; ++i) {
        magnitudes[i] = std::abs(weights[i]);
    }

    // Create a weighted distribution based on magnitudes of the complex weights
    std::discrete_distribution<> dist(magnitudes, magnitudes + numElements);

    for (int i = 0; i < numSamples; ++i) {
        int index = dist(gen);  // Generate a weighted index
        counts[index]++;        // Increment the count for this index
    }

    for (int i = 0; i < numElements; ++i) {
        averages[i] = static_cast<double>(counts[i]) / numSamples;  // Calculate the average frequency of selection
    }

    delete[] counts;          // Clean up the counts array
    delete[] magnitudes;      // Clean up the magnitudes array
    return averages;
}

int main() {
    // Example complex weights
    Complex weights[] = {Complex(0.8, 0.0), Complex(0.2, 0.0)};
    int numElements = 2;
    int numSamples = 10000;  // Number of samples

    // Calculate average frequency per position
    double* averages = simulate(weights, numElements, numSamples);

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
