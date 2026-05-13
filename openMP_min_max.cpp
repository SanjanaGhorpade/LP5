#include <iostream>
#include <omp.h>
#include <climits>
#include <cstdlib>

using namespace std;

#define N 100000000

int main()
{
    int *arr = new int[N];

    // Initialize array
    for(int i = 0; i < N; i++)
        arr[i] = rand() % 100;

    // Sequential
    int min_seq = INT_MAX;
    int max_seq = INT_MIN;
    long long sum_seq = 0;

    double start_seq = omp_get_wtime();

    for(int i = 0; i < N; i++)
    {
        if(arr[i] < min_seq) min_seq = arr[i];
        if(arr[i] > max_seq) max_seq = arr[i];
        sum_seq += arr[i];
    }

    double end_seq = omp_get_wtime();

    double avg_seq = (double)sum_seq / N;

    // Parallel
    int min_par = INT_MAX;
    int max_par = INT_MIN;
    long long sum_par = 0;

    double start_par = omp_get_wtime();

    #pragma omp parallel for reduction(min:min_par) reduction(max:max_par) reduction(+:sum_par)
    for(int i = 0; i < N; i++)
    {
        if(arr[i] < min_par) min_par = arr[i];
        if(arr[i] > max_par) max_par = arr[i];
        sum_par += arr[i];
    }

    double end_par = omp_get_wtime();

    double avg_par = (double)sum_par / N;

    // Output
    cout << "\nSequential Results:\n";
    cout << "Min = " << min_seq
         << ", Max = " << max_seq
         << ", Sum = " << sum_seq
         << ", Avg = " << avg_seq << endl;

    cout << "Time = " << (end_seq - start_seq) << " sec\n";

    cout << "\nParallel Results:\n";
    cout << "Min = " << min_par
         << ", Max = " << max_par
         << ", Sum = " << sum_par
         << ", Avg = " << avg_par << endl;

    cout << "Time = " << (end_par - start_par) << " sec\n";

    cout << "\nSpeedup = "
         << (end_seq - start_seq) / (end_par - start_par)
         << endl;

    delete[] arr;

    return 0;
}