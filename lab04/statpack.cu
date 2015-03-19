#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

// Unrolled reduction operation for one warp
template <unsigned int blockSize>
__device__ void warpReduce(volatile float *mins, volatile float *maxs, volatile float *mean, volatile float *meansquares, unsigned int tid)
{
    if(blockSize >= 64) 
    {
        mins[tid] = (mins[tid] > mins[tid+32]) ? mins[tid+32] : mins[tid];
        maxs[tid] = (maxs[tid] < maxs[tid+32]) ? maxs[tid+32] : maxs[tid];
        mean[tid] += mean[tid+32];
        meansquares[tid] += meansquares[tid+32]*meansquares[tid+32];
    }
    if(blockSize >= 32) 
    {
        mins[tid] = (mins[tid] > mins[tid+16]) ? mins[tid+16] : mins[tid];
        maxs[tid] = (maxs[tid] < maxs[tid+16]) ? maxs[tid+16] : maxs[tid];
        mean[tid] += mean[tid+16];
        meansquares[tid] += meansquares[tid+16]*meansquares[tid+16];
    }
    if(blockSize >= 16) 
    {
        mins[tid] = (mins[tid] > mins[tid+8]) ? mins[tid+8] : mins[tid];
        maxs[tid] = (maxs[tid] < maxs[tid+8]) ? maxs[tid+8] : maxs[tid];
        mean[tid] += mean[tid+8];
        meansquares[tid] += meansquares[tid+8]*meansquares[tid+8];
    }
    if(blockSize >= 8) 
    {
        mins[tid] = (mins[tid] > mins[tid+4]) ? mins[tid+4] : mins[tid];
        maxs[tid] = (maxs[tid] < maxs[tid+4]) ? maxs[tid+4] : maxs[tid];
        mean[tid] += mean[tid+4];
        meansquares[tid] += meansquares[tid+4]*meansquares[tid+4];
    }
    if(blockSize >= 4) 
    {
        mins[tid] = (mins[tid] > mins[tid+2]) ? mins[tid+2] : mins[tid];
        maxs[tid] = (maxs[tid] < maxs[tid+2]) ? maxs[tid+2] : maxs[tid];
        mean[tid] += mean[tid+2];
        meansquares[tid] += meansquares[tid+2]*meansquares[tid+2];
    }
    if(blockSize >= 2) 
    {
        mins[tid] = (mins[tid] > mins[tid+1]) ? mins[tid+1] : mins[tid];
        maxs[tid] = (maxs[tid] < maxs[tid+1]) ? maxs[tid+1] : maxs[tid];
        mean[tid] += mean[tid+1];
        meansquares[tid] += meansquares[tid+1]*meansquares[tid+1];
    }
}

template <unsigned int blockSize>
__global__ void reduce(volatile float *inputMins, volatile float *inputMaxs, volatile float *inputMean, volatile float *meansquares, T *outputData, long N)
{
    // Shared data buffer for this block
    __shared__ float sharedData[sizeof(float)*blockSize];

    // Compute the
    unsigned int tid = threadIdx.x;
    unsigned long i = blockIdx.x*(blockSize*2) + tid;
    unsigned long gridSize = blockSize*2*gridDim.x;
    sharedData[tid] = 0;

    // All threads perform gather from global memory to into shared memory with an operation
    while(i < N)
    {
        float t = Operation(inputData[i], inputData[i+blockSize]);
        sharedData[tid] = Operation(sharedData[tid], t);
        i += gridSize;
    }
    __syncthreads();

    // Unrolled reduction (for greater than one warp active)
    if(blockSize >= 1024)
    {
        if(tid < 512) sharedData[tid] = Operation(sharedData[tid],sharedData[tid+512]);
        __syncthreads();
    }
    if(blockSize >= 512)
    {
        if(tid < 256) sharedData[tid] = Operation(sharedData[tid],sharedData[tid+256]);
        __syncthreads();
    }
    if(blockSize >= 256)
    {
        if(tid < 128) sharedData[tid] = Operation(sharedData[tid],sharedData[tid+128]);
        __syncthreads();
    }
    if(blockSize >= 128)
    {
        if(tid < 64) sharedData[tid] = Operation(sharedData[tid],sharedData[tid+64]);
        __syncthreads();
    }

    // Call unrolled reduction operation for a singular warp
    if(tid < 32) warpReduce<T,Operation,blockSize>(sharedData, tid);

    // Store result
    if(tid==0) outputData[blockIdx.x] = sharedData[0];
};

// Launch a reduction
template<typename T, T (*Operation)(const T a, const T b)>
void launchReduce (T *inputData, T *outputData, long N)
{
    // How many threads to we need
    long threadCount = (N/2 >= 1024) ? 1024 : N/2;

    // Compute the amount of elements per thread
    long elementCount = N / threadCount;

    // Limit the per thread element consumption to 1024
    long blockCount = 1;
    if(elementCount > 1024)
    {
        blockCount = elementCount / 1024;
        elementCount = 1024;
    }

    dim3 blockDim(blockCount);
    dim3 threadDim(threadCount);

    cout << "Lauching reduce<<<" << blockCount << "," << threadCount << ">>>(.,.," << N;
    cout << "), consuming " << elementCount << " elements per thread" << endl;

    // Launch
    switch (threadCount)
    {
        case 1024:
            reduce<T,Operation,1024><<<blockDim,threadDim>>>(inputData,outputData,N); break;
        case 512:
            reduce<T,Operation,512><<<blockDim,threadDim>>>(inputData,outputData,N); break;
        case 256:
            reduce<T,Operation,256><<<blockDim,threadDim>>>(inputData,outputData,N); break;
        case 128:
            reduce<T,Operation,128><<<blockDim,threadDim>>>(inputData,outputData,N); break;
        case 64:
            reduce<T,Operation,64><<<blockDim,threadDim>>>(inputData,outputData,N); break;
        case 32:
            reduce<T,Operation,32><<<blockDim,threadDim>>>(inputData,outputData,N); break;
        case 16:
            reduce<T,Operation,16><<<blockDim,threadDim>>>(inputData,outputData,N); break;
        case 8:
            reduce<T,Operation,8><<<blockDim,threadDim>>>(inputData,outputData,N); break;
        case 4:
            reduce<T,Operation,4><<<blockDim,threadDim>>>(inputData,outputData,N); break;
        case 2:
            reduce<T,Operation,2><<<blockDim,threadDim>>>(inputData,outputData,N); break;
        case 1:
            reduce<T,Operation,1><<<blockDim,threadDim>>>(inputData,outputData,N); break;
    }
}


template <typename T>
__device__ T sumOperation (T a, T b)
{
    return a + b;
};

template<typename T>
__device__ T gtOperation (T a, T b)
{
    return (a > b) ? a : b;
};

template<typename T>
__device__ T ltOperation (T a, T b)
{
    return (a < b) ? a : b;
};

// Finds the maximum element in a set of numbers (N must be power of two)
template<typename T>
T getMaximumElementCUDA(T* data, long N)
{
    // Copy the data to the GPU
    float delta = 0.0f;
    T *deviceData = NULL;
    T *deviceResults = NULL;
    long resultsN = N;
    cudaMalloc(&deviceData, N*sizeof(T));
    cudaMemcpy(deviceData, data, N*sizeof(T), cudaMemcpyHostToDevice);

    // Allocate a secondary buffer for storing results
    resultsN = (N / (1024*1024) > 1) ? N / (1024*1024) : 1;
    cudaMalloc(&deviceResults, resultsN*sizeof(T));

    // Perform reduction
    do
    {
        // Compute the number of results
        resultsN = (N / (1024*1024) > 1) ? N / (1024*1024) : 1;

        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        // Run the reduction
        launchReduce<T,gtOperation>(deviceData,deviceResults,N);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);
        delta += ms;
        printf(" >> Elapsed Time (GPU): %f ms\n", ms);

        // If resultsN is greater than one, swap arrays
        if(resultsN > 1)
        {
            T* temp = deviceResults;
            deviceResults = deviceData;
            deviceData = temp;
            N = resultsN;
        }

    } while (resultsN > 1);

    // Get the results
    int result = 0;
    cudaMemcpy(&result, deviceResults, sizeof(T), cudaMemcpyDeviceToHost);


    printf("Overall Elapsed Time (GPU): %f ms\n", delta);

    // Cleanup
    cudaFree(deviceData);
    cudaFree(deviceResults);
    return result;
}

template<typename T>
T getMaximumElementCPU(T* data, long N)
{
    T max = data[0];
    for(long i = 1; i < N; i++)
        if(data[i] > max) max = data[i];
    return max;
}


int main (int argc, char** argv)
{
    srand(time(NULL));

    int *local = NULL;
    long N = 536870912;
    cudaMallocHost(&local, N*sizeof(int));

    cout << "Generating Random Dataset" << endl;
    for(int i = 0; i < N; i++)
    {
        local[i] = rand();
    }
    cout << "Done" << endl << endl;


    cout << "Getting maximum on CPU" << endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    int m = getMaximumElementCPU(local, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cout << "Done: " << m  << " (elapsed: " << ms << " ms)" << endl << endl;

    cout << "Getting maximum on GPU" << endl;
    cout << "Done: " << getMaximumElementCUDA(local, N) << endl;

    cudaFreeHost(local);

    return 0;
}
