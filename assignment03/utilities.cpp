#include <iostream>
#include "utilities.hpp"

using namespace std;

void PrintGrid(ostream& stream, unsigned char *grid, size_t gridWidth, size_t gridHeight, const char* name)
{
    // Generate a random starting grid
    stream << name << ":\n------\n";
    for(int i = 0; i < gridHeight; i++)
    {
        for(int j = 0; j < gridWidth; j++)
        {
            stream << (unsigned) grid[i*gridWidth + j] << " ";
        }
        stream << endl;
    }
    stream << endl;
}

// Run an iteration of game of life on the CPU (In parallel)
void GameOfLife(unsigned char * currentGrid, unsigned char * temporaryGrid, size_t M, size_t N)
{
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++)
    {
        for(int j = 0; j < M; j++)
        {
            // Get the local cell position
            GridCell localCell(j, i);
            int localCellId = localCell.GetProcessId(M);

            // Get the neighbors are alive
            int  liveNeighborCount = 0;
            for(int k = -1; k <= 1; k++)
            {
                for(int l = -1; l <= 1; l++)
                {
                    // If this neighbor is located within the grid
                    GridCell neighbor(localCell.x + l, localCell.y + k);
                    if(neighbor.x >= 0 && neighbor.x < M &&
                       neighbor.y >= 0 && neighbor.y < N &&
                       !(k == 0 && l == 0))
                    {
                        liveNeighborCount += currentGrid[neighbor.GetProcessId(M)];
                    }
                }
            }

            // Game of life logic
            char localCellValue = currentGrid[localCellId];
            if(localCellValue == CELL_STATUS_ALIVE)
            {
                // if we have one or two neighbors, we die from loneliness
                if(liveNeighborCount < 2 || liveNeighborCount > 3)
                    localCellValue = CELL_STATUS_DEAD;
            }
            else
            {
                // If we have two or three neighbors, we LIVE
                if(liveNeighborCount == 3)
                    localCellValue = CELL_STATUS_ALIVE;
            }
            temporaryGrid[localCellId] = localCellValue;
        }
    }
}
