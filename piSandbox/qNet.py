import h5py
import torch
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

def saveResults(outfile,trainX,trainY,testX,testY,model):
  import csv
  with open(outfile+'.csv','wb') as out:
    csvwriter=csv.writer(out,delimiter=' ')
    csvwriter.writerow(trainX)
    csvwriter.writerow(trainY)
    csvwriter.writerow(testX)
    csvwriter.writerow(testY)
  torch.save(model,outfile+'.mod')

def trainNet(trainname,numNodes,numLayers,epochs,lr,batchSize,numLoaders=1,\
    testname=None,outdir='.',gamma=.95):
  cuda=torch.cuda.is_available()
  dtype=torch.cuda.FloatTensor if cuda else torch.FloatTensor
  
  class ValveDataset(Dataset):
    def __init__(self, file, transform=None):
      f=h5py.File(file,'r')
      self.groupStates=[ f[group]['states'][:,:] for group in f.keys() ]
      self.groupActions=[ f[group]['actions'][:,:] for group in f.keys() ]
      self.groupRewards=[ f[group]['rewards'][:] for group in f.keys() ]
      sizes=[ g.shape[0]-1 for g in self.groupStates]
      self.csum=np.cumsum(sizes)
      self.transform=transform

    def __len__(self):
      return self.csum[-1]

    def __getitem__(self,idx):
      groupIdx=np.argmin(self.csum<=idx)
      if groupIdx!=0:
        idx=idx-self.csum[groupIdx-1]
      state=self.groupStates[groupIdx][idx,:]
      action=self.groupActions[groupIdx][idx,:]
      reward=self.groupRewards[groupIdx][idx]
      nextstate=self.groupStates[groupIdx][idx+1,:]
      nextaction=self.groupActions[groupIdx][idx+1,:]
      sample={'state':state,\
              'action':action,\
              'reward':reward,\
              'nextstate':nextstate,\
              'nextaction':nextaction}
      if self.transform:
        sample=self.transform(sample)
      return sample

  class ToTensor:
    def __call__(self,sample):
      s,a,r,sp,ap=sample['state'],sample['action'],sample['reward'],\
                sample['nextstate'],sample['nextaction']
      s[0]=3*(s[0]-.75)
      s[1]=(s[1]-389)/21
      a[0]=3*(a[0]-1)
      a[1]=np.log(max(a[1],.0001))-4.5
      sp[0]=3*(sp[0]-.75)
      sp[1]=(sp[1]-389)/21
      ap[0]=3*(ap[0]-1)
      ap[1]=np.log(max(ap[1],.0001))-4.5
      return {'state':torch.from_numpy(s),\
              'action':torch.from_numpy(a),\
              'reward':r,\
              'nextstate':torch.from_numpy(sp),\
              'nextaction':torch.from_numpy(ap)}

  layers=[]
  layers.append(torch.nn.Linear(4,numNodes))
  layers.append(torch.nn.ReLU())
  for lay in range(1,numLayers):
    layers.append(torch.nn.Linear(numNodes,numNodes))
    layers.append(torch.nn.ReLU())
  layers.append(torch.nn.Linear(numNodes,1))

  model=torch.nn.Sequential(*layers)
  if cuda:
    model=model.cuda()
    if torch.cuda.device_count()>1:
      model=torch.nn.DataParallel(model)

  optimizer = torch.optim.Adam(model.parameters(),lr=lr)

  dataset=ValveDataset(trainname,ToTensor())

  #num_workers>1 causes hdf5 errors
  dataloader = DataLoader(dataset,batch_size=batchSize,shuffle=True,num_workers=numLoaders)
  if testname is not None:
    testdataset=ValveDataset(testname,ToTensor())
    testdataloader = DataLoader(testdataset,batch_size=batchSize,shuffle=False,num_workers=numLoaders)

  loss_fn=torch.nn.MSELoss()
  
  trainX=[]
  trainY=[]

  testX=[]
  testY=[]

  for epoch in range(epochs):
    epochLoss=0
    for i_batch,sample_batched in enumerate(dataloader):
      sps=Variable(sample_batched['nextstate'],requires_grad=False).type(dtype)
      aps=Variable(sample_batched['nextaction'],requires_grad=False).type(dtype)
      rs=Variable(sample_batched['reward'],requires_grad=False).type(dtype)
      N=rs.shape[0]
      rs=rs.view(N,1)
      primes=torch.cat([sps,aps],dim=1)
      expectedNextvals=rs+gamma*model(primes)
      expectedNextvals=expectedNextvals.detach()

      ss=Variable(sample_batched['state'],requires_grad=False).type(dtype)
      ats=Variable(sample_batched['action'],requires_grad=False).type(dtype)
      orig=torch.cat([ss,ats],dim=1)
      vals=model(orig)

      loss=loss_fn(vals,expectedNextvals)
      epochLoss=epochLoss+loss.data[0]
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    trainX.append(epoch)
    trainY.append(epochLoss)
    print epoch,epochLoss
    if testname is not None and epoch>0 and (epoch%15==0 or epoch==epochs-1):
      epochLoss=0
      for i_batch, sample_batched in enumerate(testdataloader):
        sps=Variable(sample_batched['nextstate'],requires_grad=False).type(dtype)
        aps=Variable(sample_batched['nextaction'],requires_grad=False).type(dtype)
        rs=Variable(sample_batched['reward'],requires_grad=False).type(dtype)
        N=rs.shape[0]
        rs=rs.view(N,1)
        primes=torch.cat([sps,aps],dim=1)
        expectedNextvals=rs+gamma*model(primes)
        expectedNextvals=expectedNextvals.detach()

        ss=Variable(sample_batched['state'],requires_grad=False).type(dtype)
        ats=Variable(sample_batched['action'],requires_grad=False).type(dtype)
        orig=torch.cat([ss,ats],dim=1)
        vals=model(orig)

        loss=loss_fn(vals,expectedNextvals)
        epochLoss=epochLoss+loss.data[0]
      testX.append(epoch)
      testY.append(epochLoss)
  
  savefn=outdir+'/nN'+str(numNodes)+'nL'+str(numLayers)+'e'+str(epochs)+'lr'+str(lr)+'bs'+str(batchSize)
  saveResults(savefn,trainX,trainY,testX,testY,model)

if __name__ == "__main__":
  import argparse
  parser=argparse.ArgumentParser()
  parser.add_argument('--numHid',type=int,default=10,help='number of hidden nodes per layer')
  parser.add_argument('--numLayers',type=int,default=2,help='number of hidden layers')
  parser.add_argument('--trainData',default='/mnt/lustre/scratch/autoValveData/training.h5',help='filename of training data')
  parser.add_argument('--testData',help='filename of testing data')
  parser.add_argument('--epochs',type=int,default=500,help='epochs of training')
  parser.add_argument('--batchSize',type=int,default=500,help='training batch size')
  parser.add_argument('--lr',type=float,default=.1,help='learning rate')
  parser.add_argument('--numLoaders',type=int,default=1,help='data load threads')
  parser.add_argument('--outfile',default='.',help='out data goes here')

  args=parser.parse_args()
  trainNet(args.trainData,args.numHid,args.numLayers,args.epochs,\
      args.lr,args.batchSize,testname=args.testData,numLoaders=args.numLoaders,\
      outdir=args.outfile)
'''
trn='/home/taylor/autoValveData/training.h5'
nNs=200
nLs=5
es=100
lr=0.000001
bs=2500
trainNet(trn,nNs,nLs,es,lr,bs,numLoaders=8)
'''
