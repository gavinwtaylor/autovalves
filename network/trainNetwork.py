import h5py
import torch
from torch.autograd import Variable
from numpy import cumsum
from numpy import argmin
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def saveResults(outfile,trainX,trainY,testX,testY,model):
  import csv
  with open(outfile+'.csv','wb') as out:
    csvwriter=csv.writer(out,delimiter=' ')
    csvwriter.writerow(trainX)
    csvwriter.writerow(trainY)
    csvwriter.writerow(testX)
    csvwriter.writerow(testY)
  torch.save(model,outfile+'.mod')

def trainNet(trainname,numNodes,numLayers,epochs,lr,batchSize,numLoaders=1,testname=None,outdir='.'):
  cuda=torch.cuda.is_available()
  
  dtype=torch.cuda.FloatTensor if cuda else torch.FloatTensor
  class ValveDataset(Dataset):
    def __init__(self, file, transform=None):
      f=h5py.File(file,'r')
      self.groupStates=[ f[group]['states'][:,:] for group in f.keys() ]
      self.groupActions=[ f[group]['actions'][:,:] for group in f.keys() ]
      sizes=[ g.shape[0]-1 for g in self.groupStates]
      self.csum=cumsum(sizes)
      self.transform=transform

    def __len__(self):
      return self.csum[-1]

    def __getitem__(self,idx):
      groupIdx=argmin(self.csum<=idx)
      if groupIdx!=0:
        idx=idx-self.csum[groupIdx-1]
      state=self.groupStates[groupIdx][idx,:]
      action=self.groupActions[groupIdx][idx,:]
      sample={'state':state,'action':action}
      if self.transform:
        sample=self.transform(sample)
      return sample

  class ToTensor:
    def __call__(self,sample):
      s,a=sample['state'],sample['action']
      return {'state':torch.from_numpy(s),'action':torch.from_numpy(a)}

  layers=[]
  layers.append(torch.nn.Linear(2,numNodes))
  layers.append(torch.nn.ReLU())
  for lay in range(1,numLayers):
    layers.append(torch.nn.Linear(numNodes,numNodes))
    layers.append(torch.nn.ReLU())
  layers.append(torch.nn.Linear(numNodes,2))


  model=torch.nn.Sequential(*layers)
  if cuda:
    model=model.cuda()
    if torch.cuda.device_count()>1:
      model=torch.nn.DataParallel(model)

  loss_fn=torch.nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(),lr=lr)

  dataset=ValveDataset(trainname,ToTensor())

  #num_workers>1 causes hdf5 errors
  dataloader = DataLoader(dataset,batch_size=batchSize,shuffle=True,num_workers=numLoaders)
  if testname is not None:
    testdataset=ValveDataset(testname,ToTensor())
    testdataloader = DataLoader(testdataset,batch_size=batchSize,shuffle=False,num_workers=numLoaders)

  trainX=[]
  trainY=[]

  testX=[]
  testY=[]
  
  for epoch in range(epochs):
    print epoch
    epochLoss=0
    for i_batch, sample_batched in enumerate(dataloader):
      x=Variable(sample_batched['state'],requires_grad=False).type(dtype)
      y=Variable(sample_batched['action'],requires_grad=False).type(dtype)
      y_pred=model(x)
      loss=loss_fn(y_pred,y)
      epochLoss=epochLoss+loss.data[0]
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
    trainX.append(epoch)
    trainY.append(epochLoss)
    if testname is not None and epoch>0 and (epoch%10==0 or epoch==epochs-1):
      epochLoss=0
      for i_batch, sample_batched in enumerate(testdataloader):
        x=Variable(sample_batched['state'],requires_grad=False).type(dtype)
        y=Variable(sample_batched['action'],requires_grad=False).type(dtype)
        y_pred=model(x)
        loss=loss_fn(y_pred,y)
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
  parser.add_argument('--noCuda',help='No GPU')
  parser.add_argument('--batchSize',type=int,default=500,help='training batch size')
  parser.add_argument('--lr',type=float,default=.1,help='learning rate')
  parser.add_argument('--numLoaders',type=int,default=1,help='data load threads')
  parser.add_argument('--outfile',default='.',help='out data goes here')

  args=parser.parse_args()
  trainNet(args.trainData,args.numHid,args.numLayers,args.epochs,\
      args.lr,args.batchSize,testname=args.testData,numLoaders=args.numLoaders,\
      outdir=args.outfile)
