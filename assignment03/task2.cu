#include "utilities.hpp"

// Task 1 kernel - simple, only uses global memory.
__global__ void GameOfLifeGPU(unsigned char *grid, unsigned char *resultGrid, size_t M, size_t N)
{
    // Shared memory
    extern __shared__ unsigned char buffer[];

    // Get the global address of this thread (account for padding)
    ssize_t x = (blockIdx.x * (blockDim.x-2)) + threadIdx.x - 1;
    ssize_t y = (blockIdx.y * (blockDim.y-2)) + threadIdx.y - 1;

    // Set our shared memory location to dead (to initialize)
    ssize_t sharedLocation = (threadIdx.y * blockDim.x) + threadIdx.x;
    ssize_t globalLocation = (y*M)+x;

    // Initialize shared memory of our position
    if(x >= 0 && y >= 0 && x < M && y < N)
        buffer[sharedLocation] = grid[globalLocation];
    else
        buffer[sharedLocation] = 0;

    // Synchronize threads
    __syncthreads();

    // If we are a thread that can store results
    if(threadIdx.x > 0 && threadIdx.x < blockDim.x-1 && threadIdx.y > 0 && threadIdx.y < blockDim.y-1
       && x >= 0 && x < M && y >= 0 && y < N)
    {
        // Compute the number of live neighbors
        unsigned char liveNeighbors = 0;

        // We manually check all of the neighbors for performance reasons
        size_t aX, aY;

        aX = threadIdx.x - 1; aY = threadIdx.y - 1;
        liveNeighbors += (buffer[aY*blockDim.x + aX]) ? 1 : 0;
        aX = threadIdx.x; aY = threadIdx.y - 1;
        liveNeighbors += (buffer[aY*blockDim.x + aX]) ? 1 : 0;
        aX = threadIdx.x + 1; aY = threadIdx.y - 1;
        liveNeighbors += (buffer[aY*blockDim.x + aX]) ? 1 : 0;



        aX = threadIdx.x - 1; aY = threadIdx.y;
        liveNeighbors += (buffer[aY*blockDim.x + aX]) ? 1 : 0;

        aX = threadIdx.x; aY = threadIdx.y;
        unsigned char localCellValue = (buffer[aY*blockDim.x + aX]) ? 1 : 0;

        aX = threadIdx.x + 1; aY = threadIdx.y;
        liveNeighbors += (buffer[aY*blockDim.x + aX]) ? 1 : 0;



        aX = threadIdx.x - 1; aY = threadIdx.y + 1;
        liveNeighbors += (buffer[aY*blockDim.x + aX]) ? 1 : 0;
        aX = threadIdx.x; aY = threadIdx.y + 1;
        liveNeighbors += (buffer[aY*blockDim.x + aX]) ? 1 : 0;
        aX = threadIdx.x + 1; aY = threadIdx.y + 1;
        liveNeighbors += (buffer[aY*blockDim.x + aX]) ? 1 : 0;


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
        resultGrid[globalLocation] = localCellValue;
    }
}

// Run the game of life experiment on the GPU
float GameOfLifeGPU_Experiment(unsigned char *grid, unsigned char *resultGrid, size_t M, size_t N, size_t B, size_t T, size_t iterations)
{
    // Allocate device memory for the experiment
    unsigned char *deviceGrid, *deviceGridResult;
    cudaMalloc(&deviceGrid, M*N);
    cudaMalloc(&deviceGridResult, M*N);
    cudaMemcpy(deviceGrid, grid, M*N, cudaMemcpyHostToDevice);

    // We need to amend the per block thread counts (2 more threads per dimension
    // to handle boundary conditions

    // Compute block and grid sizes
    dim3 gridSize(B,B,1);
    dim3 blockSize(T+2,T+2,1);

    // Compute the shared memory requirement
    size_t sharedMemorySize = (T+2)*(T+2);

    // Begin to record time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Perform all the iterations
    for(int i = 0; i < iterations; i++)
    {
        // Perform the iteration
        GameOfLifeGPU<<<gridSize,blockSize,sharedMemorySize>>>(deviceGrid, deviceGridResult, M, N);

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
