#include <iostream>
#include <vector>
#include <ctime>
#include "mt19937-64.cpp"
#include "test.h"
#include <omp.h>

using namespace std;

void sequentialRadixSort(vector<unsigned long long>& arr, int b) {
    const int n = arr.size();
    const int bucketCount = 1 << b;
    vector<unsigned long long> tempArray(n);

    for (int shift = 0; shift < 64; shift += b) {
        vector<int> count(bucketCount, 0);

        for (int i = 0; i < n; ++i) {
            int digit = (arr[i] >> shift) & (bucketCount - 1);
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

void parallelRadixSort(vector<unsigned long long>& arr, int b, int numThreads) {
    const int n = arr.size();
    const int bucketCount = 1 << b;
    vector<unsigned long long> tempArray(n, 0);

    for (int shift = 0; shift < 64; shift += b) {
        vector<int> count(bucketCount, 0);

    #pragma omp parallel
        {
            vector<int> localCount(bucketCount, 0);
    #pragma omp for nowait
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

        int total = 0;
        for (int i = 1; i < bucketCount; i++) {
            count[i] += count[i - 1];
        }

        for (int i = n - 1; i >= 0; i--) {
            int digit = (arr[i] >> shift) & (bucketCount - 1);
            int newPos = --count[digit];
            tempArray[newPos] = arr[i];
		}

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

        vector<unsigned long long> arrSequential, arrParallel;

        mt19937_64 temp = init_genrand64_fromtime();
        for (int i = 0; i < n; ++i) {
            arrSequential.push_back(genrand64_int64(&temp));
            arrParallel.push_back(arrSequential[i]);
        }

        cout << "Test Case (" << n << ", " << b << ")\n";

        double startSequential = omp_get_wtime();
        sequentialRadixSort(arrSequential, b);
        double endSequential = omp_get_wtime();
        double elapsedTimeSequential = double(endSequential - startSequential);

        cout << "Sequential Sorting: " << elapsedTimeSequential << " seconds\n";

        double startParallel = omp_get_wtime();
        parallelRadixSort(arrParallel, b, numThreads);
        double endParallel = omp_get_wtime();
        double elapsedTimeParallel = double(endParallel - startParallel);

        cout << "Parallel " << numThreads << " threads: " << elapsedTimeParallel << " seconds\n";

        for (int i = 0; i < n; ++i) {
            if (arrSequential[i] != arrParallel[i]) {
                cout << "Sorting mismatch between sequential and parallel versions.\n";
                return 1;
            }
        }

        double speedup = elapsedTimeSequential / elapsedTimeParallel;
        cout << "Speedup: " << speedup << "\n";

        cout << "-------------------------------------\n";
    }

    return 0;
}
