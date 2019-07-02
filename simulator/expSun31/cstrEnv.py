import cstr
import sys, traceback
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import numpy as np
from baselines import bench, logger

def action_net_to_env(action):
  action_space = spaces.Box(np.array([0,0]), np.array([2, 20000])) 
  low=action_space.low
  high=action_space.high
  action=action*.5*(high-low)+.5*(high-low) #scale actions

  for i in range(len(action)):
    if action[i]<low[i]:
      action[i]=low[i]
    if action[i]>high[i]:
      action[i]=high[i]
  return action

class CSTREnvironment(gym.Env, utils.EzPickle):
  '''
  observation_space and action_space are required
  '''
  def __init__(self):
    self.observation_space = spaces.Box(np.array([0.1,310]), np.array([1, 440]))      
    self.action_space = spaces.Box(np.array([0,0]), np.array([2, 20000])) 
    self.realEnv=cstr.CSTREnv()

  '''
  resets the environment to beginning state.
  MUST return a state - will get nans as actions without it
  '''
  def reset(self):
    self.state=np.array(self.realEnv.reset())
    return self.state

  '''
  takes in an action as a numpy array and returns a state (as a numpy array)
  a reward (a float), and done, a boolean
  '''
  def step(self,action):
    action=action_net_to_env(action)
    assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
    action=action.astype(float) #has to be double precision, or the C++ can't read it as a double
    action=(action[0],action[1])
    state,reward,done=self.realEnv.step(action) #action and state are tuples, here
    self.state=np.array(state)
    return self.state,reward,done,{}

  '''
  necessary getter to compute the reward of a state outside the simulator
  '''
  def getrewardstuff(self):
    setpoint, x0scale,x1scale=self.realEnv.getrewardstuff()
    return np.array(setpoint),x0scale,x1scale
