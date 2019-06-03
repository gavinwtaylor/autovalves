from mpi4py import MPI
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import numpy as np
from baselines import bench, logger
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size();
      

class ChemicalEnv(gym.Env, utils.EzPickle):    
    def __init__(self):
      #1st dimension --> 0.1-1.0 and 2nd dimension 310 - 440
      self.observation_space = spaces.Box(np.array([0.1,310]), np.array([1, 440]))      
      self.action_space = spaces.Box(np.array([0,0]), np.array([2, 20000])) 
      self.state = np.array([0.5, 350])
      
      state=np.empty(4)
      
      comm.Recv(state, source=0, tag=0)    
      print("Learner received the initial state")      

    def step(self, action):
      state = np.empty(4)
      low=self.action_space.low
      high=self.action_space.high
      action=action*.5*(high-low)+.5*(high-low)
      for i in range(len(action)):
        if action[i]<low[i]:
          action[i]=low[i]
        if action[i]>high[i]:
          action[i]=high[i]
      assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
      comm.Send(action, dest=0, tag=0) #zero is the action tag
      print("Before receive in learner step")
      comm.Recv(state, source=0, tag=0)
      print("After receive in learner step")
      
      return np.array(state[0:1]), state[2], state[3], {} 

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
      a = np.empty(4)
      a[0] = 0
      a[1] = 1
      a[2] = 0
      a[3] = 0    
      print("Sending reset in Learner reset")
      comm.Send(a, dest=0, tag=1) #one is the reset tag
      comm.Recv(a, source=0, tag=0)
      return np.array(a[0:1])
       
    def _render(self, mode='human'):
      pass
        
def train():
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common import set_global_seeds
    #from baselines.ppo2.policies import MlpPolicy
    from baselines.bench import Monitor
    from baselines.ppo2 import ppo2
    import sys
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    import multiprocessing
   
 
    ncpu = 1
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()

    env = DummyVecEnv([lambda:ChemicalEnv()])

    env = VecNormalize(env)
   # set_global_seeds(seed)
    policy = "mlp"
    model = ppo2.learn(network=policy, 
                       env=env, 
                       total_timesteps=int(1e12))
    return model, env

def main():
    logger._configure_default_logger()
    train()
if __name__ == '__main__':
    main()

