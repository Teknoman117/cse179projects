#include <iostream>
#include <cstdlib>

using namespace std;


// Perform

int main (int argc, char** argv)
{
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
    size_t interations = atoi(argv[5]);

    // Generate the dataset on the CPU
    unsigned char *grid = NULL;
    cudaError_t error = cudaMallocHost(&grid, M * N);
    for(size_t i = 0; i < M * N; i++)
    {
        grid[i] = rand() % 2;
    }
}
