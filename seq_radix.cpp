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
    vector<unsigned long long> tempArray(n);

    for (int shift = 0; shift < 64; shift += b) {
        vector<int> count(bucketCount, 0);

        for (int i = 0; i < n; ++i) {
            int digit = (arr[i] >> shift) & (bucketCount - 1);
            // cout << "Digit: " << digit << ssendl;
            count[digit]++;
        }

        int total = 0;
        for (int i = 0; i < bucketCount; ++i) {
            int oldCount = count[i];
            count[i] = total;
            total += oldCount;
        }

        for (int i = 0; i < n; ++i) {
            int digit = (arr[i] >> shift) & (bucketCount - 1);
            tempArray[count[digit]++] = arr[i];
        }

        arr = tempArray;
    }
}

int main() {
    // vector<pair<int, int>> testCases = {{10, 4}};

    for (const auto& testCase : testCases) {
        int n = testCase.first;
        int b = testCase.second;

        vector<unsigned long long> arr(n);
        mt19937_64 temp = init_genrand64_fromtime();
        for (int i = 0; i < n; ++i) {
            arr[i] = genrand64_int64(&temp);
        }

        double start = omp_get_wtime();
        radixSort(arr, b);
        double end = omp_get_wtime();

        for (int i = 1; i < n; ++i) {
            if (arr[i - 1] > arr[i]) {
                cout << "Test Case (" << n << ", " << b << ") - Not sorted\n";
                return 1;
            }
        }

        double elapsedTime = double(end - start);
        cout << "Test Case (" << n << ", " << b << ") - Sorting completed in " << elapsedTime << " seconds.\n";
    }

    return 0;
}
