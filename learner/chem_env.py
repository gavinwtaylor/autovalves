import sys, traceback
from mpi4py import MPI
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import argparse
#from chem_env import 
import numpy as np
from baselines import bench, logger
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
partner = rank - (size/2)
partner=int(partner)
      
os.environ["CUDA_VISIBLE_DEVICES"] =str(partner)
class ChemicalEnv(gym.Env, utils.EzPickle):    
    def __init__(self, comm):
      #1st dimension --> 0.1-1.0 and 2nd dimension 310 - 440i
      self.comm=comm
      self.rank=comm.Get_rank()
      self.size=comm.Get_size()
      self.partner= rank-(size/2)
      self.observation_space = spaces.Box(np.array([0.1,310]), np.array([1, 440]))      
      self.action_space = spaces.Box(np.array([0,0]), np.array([2, 20000])) 
      self.state = np.array([0.5, 350])
      
      exp=np.empty(4)
      
      self.comm.Recv(exp, source=self.partner, tag=0)    
      self.state = exp[:2] 
      self.reward = exp[2]
      self.done = exp[3]     
     
    def step(self, action):
      temp=np.empty(4)
      low=self.action_space.low
      high=self.action_space.high
      action=action*.5*(high-low)+.5*(high-low)
      for i in range(len(action)):
        if action[i]<low[i]:
          action[i]=low[i]
        if action[i]>high[i]:
          action[i]=high[i]
      action=action.astype(float)
      assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
      self.comm.Send(action, dest=self.partner, tag=0) #zero is the action tag
      self.comm.Recv(temp, source=self.partner, tag=0)
      self.state = temp[:2]
      self.reward=temp[2]
      self.done=temp[3]

      self.state[1]=(self.state[1]-310)/100
      return np.array(self.state), self.reward, self.done, {} 

    def __del__(self):
      pass
 
    def _configure_environment(self):
      pass
       
    def _start_viewer(self):
      pass
       
    def _take_action(self, action):
      pass
    
    def _get_reward(self):
      pass
       
    def reset(self):
      a = np.empty(2)
      a[0] = 1.0
      a[1] = 1.0
      
      s= np.empty(4)
           
       

      self.comm.Send(a, dest=self.partner, tag=1) #one is the reset tag
      self.comm.Recv(s, source=self.partner, tag=0)

      self.state = s[0:1]
      self.reward = s[2]
      self.done = s[3]
      return np.array(self.state)
       
    def _render(self, mode='human'):
      pass
        