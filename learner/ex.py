import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding

import numpy as np
from baselines import bench, logger

class SoccerEnv(gym.Env, utils.EzPickle):    
    def __init__(self):
      #1st dimension --> 0.1-1.0 and 2nd dimension 310 - 440
      self.observation_space = spaces.Box(np.array([0.1,310]), np.array([1, 440]))     
      self.action_space = spaces.Box(np.array([0.1,310]), np.array([1, 440])) 
      self.state = None

    def step(self, action):
      #assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
      state = self.state
      return np.array(self.state), 1, "false", {} 

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
      return np.array(self.state)
       
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

    env = DummyVecEnv([lambda:SoccerEnv()])

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

