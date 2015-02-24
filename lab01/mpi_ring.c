#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) 
{
    int numprocs, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Get_processor_name(processor_name, &namelen);
    
    if(rank > 0)
    {
        // Receive the message from the preceding process
        int num = 0;
        MPI_Recv((void *) &num, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Process %d: Received message %d from Process %d\n", rank, num, rank - 1);
        num = num + 1;
        
        // If we are the last process, stop
        if(rank < numprocs - 1)
        {
            MPI_Send((void *) &num, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }
    
    else
    {
        // Derp
        MPI_Send((void *) &rank, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        printf("Process %d: Started comm\n", rank);
    }
    
    MPI_Finalize();
}
