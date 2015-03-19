#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <time.h>
#include <unistd.h>

#include <omp.h>

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
    srand(time(NULL));

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

    // Allocate the storage grids.  initial is for the random starting state, current is the current state, temporary is the write buffer
    char *initialGrid = (char *) calloc (gridWidth * gridHeight, sizeof(char));
    char *currentGrid = (char *) calloc (gridWidth * gridHeight, sizeof(char));
    char *temporaryGrid = (char *) calloc (gridWidth * gridHeight, sizeof(char));

    // Initialize the grid to a random state
    for(int i = 0; i < gridWidth * gridHeight; i++)
        currentGrid[i] = initialGrid[i] = rand() % 2;

    // Display the initial grid
    PrintGrid(initialGrid, gridWidth, gridHeight, "Start");

    // Perform the main iterations of the game of life simulation
    for(int iteration = 0; iteration < iterationCount; iteration++)
    {
        // Perform iterations
        #pragma omp parallel for collapse(2)
        for(int i = 0; i < gridHeight; i++)
        {
            for(int j = 0; j < gridWidth; j++)
            {
                // Get the local cell position
                struct GridCell localCell =
                {
                    .x = j,
                    .y = i
                };
                int localCellId = ConvertToProcessId(localCell, gridWidth, gridHeight);

                // Get the neighbors are alive
                int  liveNeighborCount = 0;
                for(int k = -1; k <= 1; k++)
                {
                    for(int l = -1; l <= 1; l++)
                    {
                        // If this neighbor is located within the grid
                        struct GridCell neighbor =
                        {
                            .x = localCell.x + l,
                            .y = localCell.y + k
                        };
                        if(neighbor.x >= 0 && neighbor.x < gridWidth &&
                           neighbor.y >= 0 && neighbor.y < gridHeight &&
                           !(k == 0 && l == 0))
                        {
                            liveNeighborCount += currentGrid[ConvertToProcessId(neighbor, gridWidth, gridHeight)];
                        }
                    }
                }

                // Game of life logic
                char localCellValue = currentGrid[localCellId];
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
                temporaryGrid[localCellId] = localCellValue;
            }
        }

        // Swap the temporary grid and current grid pointers
        char *temp = temporaryGrid;
        temporaryGrid = currentGrid;
        currentGrid = temp;

        if(debugMode)
        {
            // Get the current grid state
            char title[256];
            sprintf((char *) title, "Iteration %d", iteration);
            PrintGrid(currentGrid, gridWidth, gridHeight, title);
        }
    }

    // Display the state of the final grid (and cleanup)
    PrintGrid(currentGrid, gridWidth, gridHeight, "Final");
    free(initialGrid);
    free(currentGrid);
    free(temporaryGrid);

    // Number of threads
    #pragma omp parallel
    {
        #pragma omp master
        printf("OpenMP Thread Count: %d\n", omp_get_num_threads());
    }
}
