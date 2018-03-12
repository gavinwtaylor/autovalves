from mpi4py import MPI
import numpy as np

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

print "python "+str(rank)+" of "+str(size)

status=MPI.Status()
state=np.zeros((2,))
comm.Recv(state,source=0,status=status)

while (status.Get_tag() != 2):
  print "controller received state ",state
  #feed state into NN, get back an action, for now, do this:
  action=np.random.rand(2)
  print "controller sending action ",action
  comm.Send(action,dest=0,tag=1)
  comm.Recv(state,source=0,status=status)

print "Python closing"
