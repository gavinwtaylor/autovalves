from mpi4py import MPI
from trainNetwork import trainNet
from itertools import product
import argparse

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

parser=argparse.ArgumentParser()
parser.add_argument('--lr',action='append',type=float,help='learning rate')
parser.add_argument('--hidN',action='append',type=int,\
    help='number of hidden nodes')
parser.add_argument('--hidLay',action='append',type=int,\
    help='number of hidden layers')
parser.add_argument('--epochs',type=int,help='number of epochs to run each')
parser.add_argument('--batchSize',type=int,action='append',\
    help='batch size')
parser.add_argument('--trainFile',help='file path for training')
parser.add_argument('--testFile',help='file path for testing')
parser.add_argument('--outdir',help='directory for output')
args=parser.parse_args()

lrs=args.lr
hidNs=args.hidN
hidLayers=args.hidLay
numEpochs=args.epochs
batchSizes=args.batchSize

if rank==0:
  num_workers=size-1
  numDispatched=0
  for args in product(hidNs,hidLayers,lrs,batchSizes):
    if numDispatched<num_workers:
      comm.send(args,dest=numDispatched+1,tag=1)
    else:
      status=MPI.Status()
      back=comm.recv(status=status,source=MPI.ANY_SOURCE,tag=2)
      comm.send(args,dest=status.Get_source(),tag=1)
    numDispatched+=1

  stillRunning=min(num_workers,numDispatched)
  while stillRunning>0:
    status=MPI.Status()
    back=comm.recv(status=status,source=MPI.ANY_SOURCE,tag=2)
    comm.send(0,dest=status.Get_source(),tag=1)
    stillRunning-=1
else:
  while True:
    fargs=comm.recv(source=0,tag=1)
    if fargs==0:
      break
    trainNet(args.trainFile,fargs[0],fargs[1],numEpochs,fargs[2],fargs[3],testname=args.testFile,outdir=args.outdir)
    comm.send(0,dest=0,tag=2)
print str(rank)+' done!'
