// C++ program to convert a decimal
// number to binary number
#include <iostream>
using namespace std;

// Function to convert decimal
// to binary
void decToBinary(int n)
{
	// Array to store binary number
	int binaryNum[32];

	// Counter for binary array
	int i = 0;
	while (n > 0) {
		// Storing remainder in binary
		// array
		binaryNum[i] = n % 2;
		n = n / 2;
		i++;
	}

	// Printing binary array in reverse
	// order
	for (int j = i - 1; j >= 0; j--)
		cout << binaryNum[j];
}

// Driver code
int main()
{
	int n = 1;
	decToBinary(n);
	return 0;
}
