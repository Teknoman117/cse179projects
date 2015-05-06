Parallel Sparse Autoencoder
===========================

Original code from: https://github.com/siddharth-agrawal/Sparse-Autoencoder

Modified by Nathaniel R. Lewis (@Teknoman117) and Zachary Canann (@zcanann)

Changed to support parallel training using MPI.  Functions by
dividing the training sets across a collection of machines and
computes the local contribution to the function gradient on each.
These are then combined on the master using MPI reduce, and fed into
the original optimization function.  The new theta is then broadcast
to all the machines in the cluster.

Usage
-----
To launch a single process version, run 'python sparseAutoencoder.py',
otherwise run with your install of mpi4py with 'mpirun -np <num processes> python sparseAutoencoder.py'
