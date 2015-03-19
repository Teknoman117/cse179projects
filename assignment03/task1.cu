#include "utilities.hpp"

// Task 1 kernel - simple, only uses global memory.
__global__ void GameOfLifeGPU(unsigned char *grid, unsigned char *resultGrid, size_t M, size_t N)
{
    // Get the global address of this thread
    ssize_t x = (blockIdx.x * blockDim.x) + threadIdx.x;
    ssize_t y = (blockIdx.y * blockDim.y) + threadIdx.y;

    // Compute the number of live neighbors
    short liveNeighbors = 0;

    // We manually check all of them for performance reasons
    ssize_t aX, aY;
    aX = x - 1; aY = y - 1;
    if(aX >= 0 && aY >= 0 && aX < M && aY < N)
        liveNeighbors += (grid[(aY * M) + aX]) ? 1 : 0;

    aX = x; aY = y - 1;
    if(aX >= 0 && aY >= 0 && aX < M && aY < N)
        liveNeighbors += (grid[(aY * M) + aX]) ? 1 : 0;

    aX = x + 1; aY = y - 1;
    if(aX >= 0 && aY >= 0 && aX < M && aY < N)
        liveNeighbors += (grid[(aY * M) + aX]) ? 1 : 0;

    aX = x - 1; aY = y;
    if(aX >= 0 && aY >= 0 && aX < M && aY < N)
        liveNeighbors += (grid[(aY * M) + aX]) ? 1 : 0;

    unsigned char localCellValue = grid[(y*M)+x];  // our value

    aX = x + 1; aY = y;
    if(aX >= 0 && aY >= 0 && aX < M && aY < N)
        liveNeighbors += (grid[(aY * M) + aX]) ? 1 : 0;

    aX = x - 1; aY = y + 1;
    if(aX >= 0 && aY >= 0 && aX < M && aY < N)
        liveNeighbors += (grid[(aY * M) + aX]) ? 1 : 0;

    aX = x; aY = y + 1;
    if(aX >= 0 && aY >= 0 && aX < M && aY < N)
        liveNeighbors += (grid[(aY * M) + aX]) ? 1 : 0;

    aX = x + 1; aY = y + 1;
    if(aX >= 0 && aY >= 0 && aX < M && aY < N)
        liveNeighbors += (grid[(aY * M) + aX]) ? 1 : 0;

    // Perform game of life logic
    if(localCellValue == CELL_STATUS_ALIVE)
    {
        // if we have one or two neighbors, we die from loneliness
        if(liveNeighbors < 2 || liveNeighbors > 3)
            localCellValue = CELL_STATUS_DEAD;
    }
    else
    {
        // If we have two or three neighbors, we LIVE
        if(liveNeighbors == 3)
            localCellValue = CELL_STATUS_ALIVE;
    }
    resultGrid[(y*M)+x] = localCellValue;
}

// Run the game of life experiment on the GPU
float GameOfLifeGPU_Experiment(unsigned char *grid, unsigned char *resultGrid, size_t M, size_t N, size_t B, size_t T, size_t iterations)
{
    // Allocate device memory for the experiment
    unsigned char *deviceGrid, *deviceGridResult;
    cudaMalloc(&deviceGrid, M*N);
    cudaMalloc(&deviceGridResult, M*N);
    cudaMemcpy(deviceGrid, grid, M*N, cudaMemcpyHostToDevice);

    // Compute block and grid sizes
    dim3 gridSize(B,B,1);
    dim3 blockSize(T,T,1);

    // Begin to record time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Perform all the iterations
    for(int i = 0; i < iterations; i++)
    {
        // Perform the iteration
        GameOfLifeGPU<<<gridSize,blockSize>>>(deviceGrid, deviceGridResult, M, N);

        // Swap the pointers
        unsigned char *temp = deviceGrid;
        deviceGrid = deviceGridResult;
        deviceGridResult = temp;
    }

    // GPU Execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Release memory
    cudaMemcpy(resultGrid, deviceGrid, M*N, cudaMemcpyDeviceToHost);
    cudaFree(deviceGrid);
    cudaFree(deviceGridResult);

    // Return
    return ms;
}
