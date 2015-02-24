#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <unistd.h>

#include <mpi.h>

#define MASTER_PROCESS     0
#define TAG_STATUS_MESSAGE 0z
#define CELL_STATUS_DEAD   0
#define CELL_STATUS_ALIVE  1

// Grid cell structure
struct GridCell
{
    int x;
    int y;
};

// Convert process id to grid cell
struct GridCell ConvertToGridCell(int processId, int gridWidth, int gridHeight)
{
    struct GridCell cell =
    {
        .x = processId % gridWidth,
        .y = processId / gridWidth
    };
    return cell;
}

// Convert grid cell to process id
int ConvertToProcessId(struct GridCell cell, int gridWidth, int gridHeight)
{
    return (cell.y * gridWidth) + cell.x;
}

// Print Matrix
void PrintGrid(char *grid, int gridWidth, int gridHeight, const char* name)
{
    // Generate a random starting grid
    printf("%s:\n------\n", name);
    for(int i = 0; i < gridHeight; i++)
    {
        for(int j = 0; j < gridWidth; j++)
        {
            printf("%d ", grid[i*gridWidth + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char *argv[])
{
    // Initialize application
    MPI_Init(&argc, &argv);
    srand(time(NULL));

    // Get application process dimensions
    int processCount   = 0;
    int processId      = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &processCount);
    MPI_Comm_rank(MPI_COMM_WORLD, &processId);

    // We require command line arguments
    if(argc < 5)
    {
        fprintf(stderr, "Error: Incorrect usage\n");
        fprintf(stderr, "  %s <grid width> <grid height> <iterations> <debug mode>\n", argv[0]);
        return 1;
    }

    // Load command line arguments
    int gridWidth      = atoi(argv[1]);
    int gridHeight     = atoi(argv[2]);
    int iterationCount = atoi(argv[3]);
    int debugMode      = atoi(argv[4]);

    // Enforce one process per cell
    if(processCount != gridWidth * gridHeight)
    {
        fprintf(stderr, "Error: Process count not equal to grid size\n");
        MPI_Finalize();
        return 1;
    }

    // Local cell information / grid state for master
    struct GridCell  localCell      = ConvertToGridCell(processId, gridWidth, gridHeight);
    char            *initialGrid    = NULL;
    char            *currentGrid    = NULL;
    char             localCellValue = 0;

    // Constuct initial grid in master process
    if(processId == MASTER_PROCESS)
    {
        // Allocate the storage grids.  initial is for the random starting state, current is for debug and final grid display
        initialGrid = (char *) calloc (gridWidth * gridHeight, sizeof(char));
        currentGrid = (char *) calloc (gridWidth * gridHeight, sizeof(char));
        for(int i = 0; i < gridWidth * gridHeight; i++)
            initialGrid[i] = rand() % 2;

        // Display the initial grid
        PrintGrid(initialGrid, gridWidth, gridHeight, "Start");
    }

    // Deploy the initial grid state to the nodes
    MPI_Scatter((void *) initialGrid, 1, MPI_CHAR, (void *) &localCellValue, 1, MPI_CHAR, MASTER_PROCESS, MPI_COMM_WORLD);

    // Compute a list of neighbors of this cell (their process ids)
    int  neighbors[8]     = {-1, -1, -1, -1, -1, -1, -1, -1};
    char neighborState[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int  neighborCount    = 0;
    for(int i = -1; i <= 1; i++)
    {
        for(int j = -1; j <= 1; j++)
        {
            // If this neighbor is located within the grid
            struct GridCell neighbor =
            {
                .x = localCell.x + j,
                .y = localCell.y + i
            };
            if(neighbor.x >= 0 && neighbor.x < gridWidth &&
               neighbor.y >= 0 && neighbor.y < gridHeight &&
               !(i == 0 && j == 0))
            {
                neighbors[neighborCount++] = ConvertToProcessId(neighbor, gridWidth, gridHeight);
            }
        }
    }

    // Perform the main iterations of the game of life simulation
    for(int iteration = 0; iteration < iterationCount; iteration++)
    {
        // Send our state to all the surrounding neighbors
        MPI_Request request;
        for(int i = 0; i < neighborCount; i++)
            MPI_Isend((void *) &localCellValue, 1, MPI_CHAR, neighbors[i], TAG_STATUS_MESSAGE, MPI_COMM_WORLD, &request);

        // Receive all surrounding cell values
        int liveNeighborCount = 0;
        for(int i = 0; i < neighborCount; i++)
        {
            MPI_Recv((void *) &neighborState[i], 1, MPI_CHAR, neighbors[i], TAG_STATUS_MESSAGE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            liveNeighborCount = (neighborState[i] == CELL_STATUS_ALIVE) ? liveNeighborCount + 1 : liveNeighborCount;
        }

        // Game of life logic
        if(localCellValue == CELL_STATUS_ALIVE)
        {
            // if we have one or two neighbors, we die from loneliness
            if(liveNeighborCount <= 1 || liveNeighborCount >= 4)
                localCellValue = CELL_STATUS_DEAD;
        }
        else
        {
            // If we have two or three neighbors, we LIVE
            if(liveNeighborCount == 2 || liveNeighborCount == 3)
                localCellValue = CELL_STATUS_ALIVE;
        }

        // Synchronize all threads
        MPI_Barrier(MPI_COMM_WORLD);

        // If we are in debug mode, gather the state of all threads
        if(debugMode)
        {
            // Get the current grid state
            MPI_Gather((void *) &localCellValue, 1, MPI_CHAR, (void *) currentGrid, 1, MPI_CHAR, MASTER_PROCESS, MPI_COMM_WORLD);
            if(processId == MASTER_PROCESS)
            {
                char title[256];
                sprintf((char *) title, "Iteration %d", iteration);
                PrintGrid(currentGrid, gridWidth, gridHeight, title);
            }
        }
    }

    // Display the state of the final grid (and cleanup)
    MPI_Gather((void *) &localCellValue, 1, MPI_CHAR, (void *) currentGrid, 1, MPI_CHAR, MASTER_PROCESS, MPI_COMM_WORLD);
    if(processId == MASTER_PROCESS)
    {
        PrintGrid(currentGrid, gridWidth, gridHeight, "Final");
        free(initialGrid);
        free(currentGrid);
    }

    MPI_Finalize();
}
