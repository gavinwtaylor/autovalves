import h5py
import torch
from torch.autograd import Variable
from numpy import cumsum
from numpy import argmin
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--numHid',type=int,default=10,help='number of hidden nodes per layer')
parser.add_argument('--numLayers',type=int,default=2,help='number of hidden layers')
parser.add_argument('--trainData',default='/mnt/lustre/scratch/autoValveData/training.h5',help='filename of training data')
parser.add_argument('--epochs',type=int,default=500,help='epochs of training')
parser.add_argument('--noCuda',help='No GPU')
parser.add_argument('--batchSize',type=int,default=500,help='training batch size')
parser.add_argument('--lr',type=float,default=.1,help='learning rate')

args=parser.parse_args()
NUMHIDDEN=args.numHid
NUMLAYERS=args.numLayers
EPOCHS=args.epochs

dtype=torch.FloatTensor
if not args.noCuda:
  dtype=torch.cuda.FloatTensor

class ValveDataset(Dataset):
  def __init__(self, file, transform=None):
    f=h5py.File(file,'r')
    self.groups=[ f[group] for group in f.keys() ]
    sizes=[ g['states'].shape[0]-1 for g in self.groups]
    self.csum=cumsum(sizes)
    self.transform=transform

  def __len__(self):
    return self.csum[-1]

  def __getitem__(self,idx):
    groupIdx=argmin(self.csum<=idx)
    if groupIdx!=0:
      idx=idx-self.csum[groupIdx-1]
    state=self.groups[groupIdx]['states'][idx,:]
    action=self.groups[groupIdx]['actions'][idx,:]
    sample={'state':state,'action':action}
    if self.transform:
      sample=self.transform(sample)
    return sample

class ToTensor:
  def __call__(self,sample):
    s,a=sample['state'],sample['action']
    return {'state':torch.from_numpy(s),'action':torch.from_numpy(a)}

layers=[]
layers.append(torch.nn.Linear(2,NUMHIDDEN))
layers.append(torch.nn.ReLU())
for lay in range(1,NUMLAYERS):
  layers.append(torch.nn.Linear(NUMHIDDEN,NUMHIDDEN))
  layers.append(torch.nn.ReLU())
layers.append(torch.nn.Linear(NUMHIDDEN,2))


model=torch.nn.Sequential(*layers)
if not args.noCuda:
  model=model.cuda()

loss_fn=torch.nn.MSELoss()
lr = 1e-5
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

dataset=ValveDataset(args.trainData,ToTensor())

#num_workers>1 causes hdf5 errors
dataloader = DataLoader(dataset,batch_size=args.batchSize,shuffle=True,num_workers=1)

for epoch in range(EPOCHS):
  for i_batch, sample_batched in enumerate(dataloader):
    x=Variable(sample_batched['state'],requires_grad=False).type(dtype)
    y=Variable(sample_batched['action'],requires_grad=False).type(dtype)
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print epoch,i_batch,loss.data[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
