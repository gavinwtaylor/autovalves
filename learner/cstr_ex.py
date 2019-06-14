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

def train(lrnrt, timest, entr, valcoef, numlyrs, lyrsize, jobnumber, numevs):
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common import set_global_seeds
    from baselines.bench import Monitor
    from baselines.ppo2 import ppo2
    import sys
    import gym
    import tensorflow as tf
    from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
    import multiprocessing
   
    ncpu = 10
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=ncpu,
                            inter_op_parallelism_threads=ncpu)
    tf.Session(config=config).__enter__()
    env = DummyVecEnv([lambda:CSTREnvironment() for i in range(numevs)])

    env = VecNormalize(env)
    policy = "mlp"
    model = ppo2.learn(network=policy, env=env,total_timesteps=timest,ent_coef=entr,lr=lrnrt,vf_coef=valcoef,log_interval=10, num_layers=numlyrs, num_hidden=lyrsize)
    model.save(workdir+"/autovalves/learner/models/"+str(jobnumber))
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
    logger.configure(dir=workdir+"/autovalves/learner/logs", format_strs=['stdout','log'], log_suffix=jobnumber)
    logger.log("Job Number: ",jobnumber)
    logger.log("Learning Rate: ", args.lrs)
    logger.log("Timestep: ", args.tss)
    logger.log("Entropy: ", args.entps)
    logger.log("Value Coefficent: ",args.vcfs)
    logger.log("Number of Layers: ", args.nlyrs)
    logger.log("Width of Layers: ", args.slyrs)
    train(args.lrs, args.tss, args.entps, args.vcfs, args.nlyrs, args.slyrs,jobnumber,args.nevs)
      
