#include <iostream>
#include <sstream>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>
#include <omp.h>

#include "utilities.hpp"

using namespace std;

// Run the game of life experiment on the GPU
extern float GameOfLifeGPU_Experiment(unsigned char *grid, unsigned char *resultGrid, size_t M, size_t N, size_t B, size_t T, size_t iterations);

// Run the game of life experiment on the CPU
float GameOfLifeCPU_Experiment(unsigned char *grid, unsigned char *temporaryGrid, unsigned char *& resultGrid, size_t M, size_t N, size_t iterations)
{
    // Begin to record time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Perform the experiement
    for(int i = 0; i < iterations; i++)
    {
        // Perform the iteration
        GameOfLife(grid, temporaryGrid, M, N);

        // Swap the pointers
        unsigned char *temp = grid;
        grid = temporaryGrid;
        temporaryGrid = temp;
    }

    // GPU Execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Return
    resultGrid = grid;
    return ms;
}


int main (int argc, char** argv)
{
    // Initialize application
    srand(time(NULL));

    // Make sure the configuration was setup correctly
    if(argc < 6)
    {
        cerr << "Usage: " << argv[0] << " M N T B Iterations" << endl;
        return 1;
    }

    // Get the settings for the experiement
    //   We assume that T & B represent the length of a side
    //   of a cube.  We do not dispatch non-square blocks&grids
    size_t M = atoi(argv[1]);
    size_t N = atoi(argv[2]);
    size_t T = atoi(argv[3]);
    size_t B = atoi(argv[4]);
    size_t iterations = atoi(argv[5]);

    // Denote if a bad scenario is presented
    if(M > T * B || N > T * B)
    {
        cerr << "Error: Not enough thread blocks have been allocated to use single kernel launch" << endl;
        return 1;
    }

    // Allocate memory on CPU (grid holds initial, temporary holds results)
    unsigned char *grid = NULL;
    unsigned char *temporaryGrid = NULL;
    unsigned char *resultGrid = NULL;  // GPU result grid

    cudaMallocHost(&grid, M * N);
    cudaMallocHost(&temporaryGrid, M * N);
    cudaMallocHost(&resultGrid, M * N);

    // Generate the dataset
    for(size_t i = 0; i < M * N; i++)
    {
        grid[i] = rand() % 2;
    }

    // Perform the GPU experiement
    float gpuTime = GameOfLifeGPU_Experiment(grid, resultGrid, M, N, B, T, iterations);
    cout << " >> Elapsed Time (GPU): " << gpuTime << " ms" << endl;

    // Run on the CPU to verify correctness
    float cpuTime = GameOfLifeCPU_Experiment(grid, temporaryGrid, grid, M, N, iterations);
    cout << " >> Elapsed Time (CPU): " << cpuTime << " ms" << endl;

    // debug
    //PrintGrid(cout, resultGrid, M, N, "GPU Grid");
    //PrintGrid(cout, temporaryGrid, M, N, "CPU Grid");

    // Perform a comparison between the gpu result and the cpu result
    if(!memcmp((void*) grid, (void *) resultGrid, M * N))
    {
        unsigned int threads = 1;
        #pragma omp parallel
        #pragma omp master
        threads = omp_get_num_threads();

        cout << "Success: GPU speedup = " << cpuTime / gpuTime << "x over " << threads << " thread CPU" << endl;
    }
    else
    {
        cout << "Error: CPU and GPU experiment results differ" << endl;
    }

    // Cleanup
    cudaFreeHost(grid);
    cudaFreeHost(temporaryGrid);
    cudaFreeHost(resultGrid);

    return 0;
}
