import h5py
import torch
from torch.autograd import Variable
from numpy import cumsum
from numpy import argmin
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import argparse

dtype=torch.FloatTensor
#dtype=torch.cuda.FloatTensor

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
    return {'state':torch.from_numpy(s).type(dtype),'action':torch.from_numpy(a).type(dtype)}

parser=argparse.ArgumentParser()
parser.add_argument('--numHid',type=int,help='number of hidden nodes')
parser.add_argument('--trainData',help='filename of training data')

args=parser.parse_args()
NUMHIDDEN=args.numHid

model=torch.nn.Sequential(
    torch.nn.Linear(2,NUMHIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(NUMHIDDEN,2)
  )

loss_fn=torch.nn.MSELoss()
lr = 1e-5
optimizer = torch.optim.Adam(model.parameters(),lr=lr)

dataset=ValveDataset(args.trainData,ToTensor())

dataloader = DataLoader(dataset,batch_size=200,shuffle=True,num_workers=10)

for epoch in range(500):
  for i_batch, sample_batched in enumerate(dataloader):
    x=Variable(sample_batched['state'],requires_grad=False)
    y=Variable(sample_batched['action'],requires_grad=False)
    y_pred=model(x)
    loss=loss_fn(y_pred,y)
    print epoch,i_batch,loss.data[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
