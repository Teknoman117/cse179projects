#!/usr/bin/python
import numpy
import math
import time
import scipy.io
import scipy.optimize
import matplotlib.pyplot
import cProfile

from mpi4py import MPI

def main(comm, rank, size):
    #derp = numpy.empty((5, size))
    #derp[:, rank] = numpy.arange(5) * (rank + 1)

    #for i in range(size):
    #    derp[:, i] = comm.bcast(derp[:, i], root=i)
    derp = numpy.random.randint(30, size=(2,2))
    print("i am {processor}/{count}. i has a {herp}".format(processor=(rank+1), count=size, herp=derp))

    #derp = comm.gather(herp, root=0)
    derp = comm.reduce(derp, None, op=MPI.SUM, root=0)

    if rank == 0:
        print (derp)
        #print (numpy.sum(derp, axis=0));

# Call the main function
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    main(comm, rank, size)
