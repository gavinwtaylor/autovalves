import h5py
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class ValveDataset(Dataset):
  def __init__(self, file, group, transform=None):
    f=h5py.File(file,'r')
    datagroup=f[group]
    self.states=datagroup['states']
    self.actions=datagroup['actions']
    self.transform=transform

  def __len__(self):
    return self.states.shape[0]

  def __getitem__(self,idx):
    state=self.states[idx,:]
    action=self.actions[idx,:]
    sample={'state':state,'action':action}
    if self.transform:
      sample=self.transform(sample)
    return sample

class ToTensor:
  def __call__(self,sample):
    s,a=sample['state'],sample['action']
    return {'state':torch.from_numpy(s),'action':torch.from_numpy(a)}

dataset=ValveDataset("/mnt/lustre/scratch/autoValveData/test.h5",'testGroup',ToTensor())

#this loop works correctly, and each printed sample is a dictionary
for i in range(len(dataset)):
  sample=dataset[i]
  print sample

#this does not work correctly: sample_batched is a list of tuples, where each
#tuple is the strings 'action' and 'state' repeated batch_size times
dataloader = DataLoader(dataset,batch_size=3)
for i_batch, sample_batched in enumerate(dataloader):
  print i_batch, sample_batched
  #print(i_batch,sample_batched['state'].size(),sample_batched['action'].size())
