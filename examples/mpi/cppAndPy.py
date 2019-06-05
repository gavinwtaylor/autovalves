from mpi4py import MPI
import numpy as np

#Initialize, get rank and size
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

print("Python program with rank ", rank)


