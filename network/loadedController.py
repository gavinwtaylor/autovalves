from mpi4py import MPI
import numpy as np
import torch
from torch.autograd import Variable

NUMNODES=150
NUMLAYERS=4

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

print "python "+str(rank)+" of "+str(size)

dtype=torch.cuda.FloatTensor

layers=[]
layers.append(torch.nn.Linear(2,NUMNODES))
layers.append(torch.nn.ReLU())
for lay in range(1,NUMLAYERS):
  layers.append(torch.nn.Linear(NUMNODES,NUMNODES))
  layers.append(torch.nn.ReLU())
layers.append(torch.nn.Linear(NUMNODES,2))
model=torch.nn.Sequential(*layers)
model=model.cuda()
model=torch.nn.DataParallel(model)
model.load_state_dict(torch.load('/mnt/lustre/scratch/autoValveData/26528/nN150nL4e500lr1e-05bs3000.mod'))
model=model.cpu().cuda()

status=MPI.Status()
state=np.zeros((2,))
comm.Recv(state,source=0,status=status)

while (status.Get_tag() != 2):
  print "controller received state ",state
  state[0]=3*(state[0]-.75)
  state[1]=(state[1]-389)/21
  x=Variable(torch.from_numpy(state),requires_grad=False).type(dtype)
  action=model(x).data.cpu().numpy()
  action[0]=min(max((action[0]/3)+1,0),2)
  action[1]=min(max(np.exp(action[1]+4.5),0),20000)

  #feed state into NN, get back an action, for now, do this:
  print "controller sending action ",action
  comm.Send(action,dest=0,tag=1)
  comm.Recv(state,source=0,status=status)

print "Python closing"
