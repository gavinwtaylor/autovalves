import sys, traceback
import os
import gym
from gym import error, spaces
from gym import utils
from gym.utils import seeding
import argparse 
import numpy as np
from cstrEnv import CSTREnvironment
from baselines import bench, logger
from datetime import datetime
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)

def train(lrnrt, timest, entr, valcoef, numlyrs, lyrsize, jobnumber, numevs):
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common import set_global_seeds
    from baselines.bench import Monitor
    from baselines.ppo2 import ppo2
    import sys
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    #from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
    import multiprocessing
   
    from baselines.common.mpi_util import setup_mpi_gpus
    
    setup_mpi_gpus() #each process only sees one gpu
    env = DummyVecEnv([lambda:CSTREnvironment() for i in range(numevs)]) #used for multiple environments

    env = VecNormalize(env)
    policy = "mlp"
    model = ppo2.learn(network=policy, env=env,total_timesteps=timest,ent_coef=entr,lr=lrnrt,vf_coef=valcoef,log_interval=5, num_layers=numlyrs, num_hidden=lyrsize,comm=MPI.COMM_WORLD)
    if is_mpi_root: #only want one model saved
      model.save(workdir+"/autovalves/learner/models/"+str(jobnumber)) #only want to save one of the models
    return model, env

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('lrs',help='learning rate',type=float)
    parser.add_argument('tss', help='timestep', type=int)
    parser.add_argument('entps',help='entropy', type=float)
    parser.add_argument('vcfs',help='value coefficent',type=float)
    parser.add_argument('nlyrs', help='layer numbers', type=int)
    parser.add_argument('slyrs', help='layer size',type=int) 
    parser.add_argument('nevs', help='list of environments',type=int)
    args=parser.parse_args()
    workdir=os.getenv("WORKDIR")
    jobnumber=os.getenv("PBS_JOBID").split('.')[0]
    logger.configure(dir=workdir+"/autovalves/learner/logs", format_strs=['stdout','log'], log_suffix=jobnumber,comm=MPI.COMM_WORLD)
    if is_mpi_root: #only want one log file
      logger.log("Job Number: ",jobnumber)
      logger.log("Learning Rate: ", args.lrs)
      logger.log("Timestep: ", args.tss)
      logger.log("Entropy: ", args.entps)
      logger.log("Value Coefficent: ",args.vcfs)
      logger.log("Number of Layers: ", args.nlyrs)
      logger.log("Width of Layers: ", args.slyrs)
      logger.log(str(datetime.now()))
    train(args.lrs, args.tss, args.entps, args.vcfs, args.nlyrs, args.slyrs,jobnumber,args.nevs)
      
