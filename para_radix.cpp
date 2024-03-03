#include <iostream>
#include <vector>
#include <ctime>
#include "mt19937-64.cpp"
#include "test.h"
#include <omp.h>

using namespace std;

void radixSort(vector<unsigned long long>& arr, int b) {
    const int n = arr.size();
    const int bucketCount = 1 << b;
    vector<unsigned long long> tempArray(n, 0);

    // Perform radix sort for each group of bits specified by 'b'
    for (int shift = 0; shift < 64; shift += b) {
        // Count array for each bucket
        vector<int> count(bucketCount, 0);

        // Count the occurrences of each digit using parallel for loop
    #pragma omp parallel
        {
        vector<int> localCount(bucketCount, 0);
    #pragma omp for
            for (int i = 0; i < n; ++i) {
                int digit = (arr[i] >> shift) & (bucketCount - 1);
                localCount[digit]++;
            }
    #pragma omp critical
        {
            for (int i = 0; i < bucketCount; i++) {
                count[i] += localCount[i];
            }
        }
        }

        // Calculate the prefix sum to determine the starting position of each bucket
        int total = 0;
        for (int i = 1; i < bucketCount; i++) {
            // int oldCount = count[i];
            // total += oldCount;
            count[i] += count[i - 1];
        }
        
    
        // Rearrange elements into tempArray based on the count and digit
        for (int i = n - 1; i >= 0; i--) {
            int digit = (arr[i] >> shift) & (bucketCount - 1);
            int newPos = --count[digit];
            tempArray[newPos] = arr[i];
        }

        // Synchronize threads before updating the main array
        // #pragma omp barrier

        // Copy tempArray back to arr using a single thread
        // #pragma omp single
        arr = tempArray;
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <number of omp threads>\n";
        return 1;
    }

    int numThreads = atoi(argv[1]);
    omp_set_num_threads(numThreads);


    for (const auto& testCase : testCases) {
        int n = testCase.first;
        int b = testCase.second;

        vector<unsigned long long> arr(n);
        mt19937_64 temp = init_genrand64_fromtime();
        for (int i = 0; i < n; ++i) {
            arr[i] = genrand64_int64(&temp);
        }

        double start = omp_get_wtime();
        radixSort(arr, numThreads);
        double end = omp_get_wtime();

        for (int i = 1; i < n; ++i) {
            // cout << arr[i] << endl;
            if (arr[i - 1] > arr[i]) {
                cout << "Test Case (" << n << ", " << b << ") - Not sorted\n";
                return 1;
            }
        }

        double elapsedTime = double(end - start);
        cout << "Test Case (" << n << ", " << b << ") - Sorting completed in " << elapsedTime << " seconds with " << numThreads << " threads.\n";
    }

    return 0;
}
