#!/bin/bash
EXECUTABLE=./assignment01

# Launch with the requested dimensions
PROCESSES=$(($1 * $2))
time mpirun -np $PROCESSES $EXECUTABLE $1 $2 $3 $4
