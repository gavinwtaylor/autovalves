import sys, traceback
from mpi4py import MPI
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import argparse

import numpy as np
from baselines import bench, logger
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
partner = rank - (size/2)
partner=int(partner)
      
os.environ["CUDA_VISIBLE_DEVICES"] =str(partner)

class ChemicalEnv(gym.Env, utils.EzPickle):    
    def __init__(self):
      #1st dimension --> 0.1-1.0 and 2nd dimension 310 - 440
      self.observation_space = spaces.Box(np.array([0.1,310]), np.array([1, 440]))      
      self.action_space = spaces.Box(np.array([0,0]), np.array([2, 20000])) 
      self.state = np.array([0.5, 350])
      
      exp=np.empty(4)
      
      comm.Recv(exp, source=partner, tag=0)    
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
      comm.Send(action, dest=partner, tag=0) #zero is the action tag
      comm.Recv(temp, source=partner, tag=0)
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
           
       

      comm.Send(a, dest=partner, tag=1) #one is the reset tag
      comm.Recv(s, source=partner, tag=0)

      self.state = s[0:1]
      self.reward = s[2]
      self.done = s[3]
      return np.array(self.state)
       
    def _render(self, mode='human'):
      pass
        
def train(lrnrt, timest, entr, valcoef, numlyrs, lyrsize):
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
    model = ppo2.learn(network=policy, env=env,total_timesteps=timest,ent_coef=entr,lr=lrnrt,vf_coef=valcoef,log_interval=500, num_layers=numlyrs, num_hidden=lyrsize)
    model.save(workdir+"/autovalves/learner/models/"+jobnumber)
  
    return model, env

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('lrs',help='comma-separated list of learning rates')
    parser.add_argument('tss', help='comma-separated list of timesteps')
    parser.add_argument('entps',help='comma-separated list of entropies')
    parser.add_argument('vcfs',help='comma-separated list of value coefficents')
    parser.add_argument('nlyrs', help='comma-separated list of layer numbers')
    parser.add_argument('slyrs', help='comma-separated list of layer sizes')
    args=parser.parse_args()
    lrs=[float(lr) for lr in args.lrs.split(',')]
    tss=[int(ts) for ts in args.tss.split(',')]
    entps=[float(ent) for ent in args.entps.split(',')]
    vcfs=[float(vcf) for vcf in args.vcfs.split(',')]
    nlyrs=[int(lyr) for lyr in args.nlyrs.split(',')]
    slyrs=[int(sz) for sz in args.slyrs.split(',')]
    workdir=os.getenv("WORKDIR")
    jobnumber=os.getenv("PBS_JOBID").split('.')[0]
    logger.configure(dir=workdir+"/autovalves/learner/logs", format_strs=['stdout','log'], log_suffix=jobnumber)
    logger.log("Job Number: ",jobnumber)
    logger.log("Rank: ",rank)
    logger.log("Learning Rate: ", lrs[partner])
    logger.log("Timestep: ", tss[partner])
    logger.log("Entropy: ", entps[partner])
    logger.log("Value Coefficent: ",vcfs[partner])
    logger.log("Number of Layers: ", nlyrs[partner])
    logger.log("Width of Layers: ", slyrs[partner])
    train(lrs[partner], tss[partner], entps[partner], vcfs[partner], nlyrs[partner], slyrs[partner])
    temp = np.array([0,1])
    temp = temp.astype(float)
    comm.Send(temp, dest=partner, tag=2) #two is the exit tag
      
