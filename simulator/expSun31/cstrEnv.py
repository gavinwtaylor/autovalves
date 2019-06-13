import cstr
import sys, traceback
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import numpy as np
from baselines import bench, logger

class CSTREnvironment(gym.Env, utils.EzPickle):
  def __init__(self):
    self.observation_space = spaces.Box(np.array([0.1,310]), np.array([1, 440]))      
    self.action_space = spaces.Box(np.array([0,0]), np.array([2, 20000])) 
    self.realEnv=cstr.CSTREnv()

  def reset(self):
    self.realEnv.reset()

  def step(self,action):
    low=self.action_space.low
    high=self.action_space.high
    action=action*.5*(high-low)+.5*(high-low)

    '''
    action[0]=3*(action[0]-1)
    action[1]=np.log(max(action[1],.0001))-4.5
    '''
    for i in range(len(action)):
      if action[i]<low[i]:
        action[i]=low[i]
      if action[i]>high[i]:
        action[i]=high[i]
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    action=(action[0],action[1])
    state,reward,done=self.realEnv.step(action)
    return np.array(state),reward,done,{}

  def getrewardstuff(self):
    setpoint, x0scale,x1scale=self.realEnv.getrewardstuff()
    return np.array(setpoint),x0scale,x1scale

#ce=CSTREnvironment()
#sp,x0scale,x1scale=ce.getrewardstuff()
#print(sp,x0scale,x1scale)
#print(ce.step(np.array([0,0])))
#ce.reset()
#print(ce.step(np.array([0,0])))
#print(ce.step(np.array([0,0])))
