import numpy as np
class Chain:
  def __init__(self):
    self.N=6
    self.SUC=.75
    self.r=np.zeros(self.N)
    self.r[1]=1
    self.r[4]=1
    self.state=np.random.randint(self.N)

  
  def step(self,action):
    reward=self.r[self.state]
    if np.random.random()<self.SUC:
      if action>0:
        self.state=min(self.state+1,self.N-1)
      else:
        self.state=max(self.state-1,0)
    return reward,self.state

c=Chain()
for i in range(10):
  a=np.random.rand()-.5
  print(a>0,c.step(a))
