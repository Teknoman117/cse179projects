#ifndef __UTILITIES_HPP__
#define __UTILITIES_HPP__

#include <iostream>

// Grid cell structure
struct GridCell
{
    size_t x;
    size_t y;

    GridCell(size_t processId, size_t gridWidth, size_t gridHeight)
        : x(processId & gridWidth), y(processId / gridWidth)
    {
    }

    GridCell(size_t x, size_t y)
        : x(x), y(y)
    {
    }

    size_t GetProcessId(int gridWidth);
    {
        return (y * gridWidth) + cell.x;
    }
};

// Print a grid to the terminal
void PrintGrid(std::ostream& stream, unsigned char *grid, int gridWidth, int gridHeight, const char* name);

// Run the game of life experiment on the CPU
void GameOfLife(unsigned char * grid, unsigned char * targetGrid, int M, int N);

#endif
