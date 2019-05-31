from mpi4py import MPI
import numpy as np

#Initialize, get rank and size
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

print("hello from the PY program, rank {rank} (of {size})".format(rank=rank, size=size))

#build an empty numpy array
a=np.empty(4)

#receive the array and print it out
comm.Recv(a,source=0,tag=0)
print("python program received",a)

#change an element
a[2]=10

#send it back
comm.Send(a,dest=0,tag=0)

#send it again, with a different tag, to show status stuff
comm.Send(a,dest=0,tag=1)
