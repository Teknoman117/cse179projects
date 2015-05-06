Parallel Sparse Autoencoder
===========================

Original code from: https://github.com/siddharth-agrawal/Sparse-Autoencoder

Modified by Nathaniel R. Lewis (@Teknoman117) and Zachary Canann (@zcanann)

Changed to support parallel training operation using MPI.  Functions
by dividing the training sets across a collection of machines and
computes the local contribution to the function gradient.  These are
then combined on the master using MPI reduce, and fed into the
original optimization function.  The new theta is then broadcast
to all the machines in the cluster.
