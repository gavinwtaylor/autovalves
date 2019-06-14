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

def train(lrnrt, timest, entr, valcoef, numlyrs, lyrsize, jobnumber):
    from baselines.common.vec_env.vec_normalize import VecNormalize
    from baselines.common import set_global_seeds
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

    env = DummyVecEnv([lambda:CSTREnvironment()])

    env = VecNormalize(env)
    policy = "mlp"
    model = ppo2.learn(network=policy, env=env,total_timesteps=timest,ent_coef=entr,lr=lrnrt,vf_coef=valcoef,log_interval=10, num_layers=numlyrs, num_hidden=lyrsize)
    model.save(workdir+"/autovalves/learner/models/"+str(jobnumber))
  
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
    #logger.log("Rank: ",rank)
    partner=0
    rank=8
    logger.log("Learning Rate: ", lrs[partner])
    logger.log("Timestep: ", tss[partner])
    logger.log("Entropy: ", entps[partner])
    logger.log("Value Coefficent: ",vcfs[partner])
    logger.log("Number of Layers: ", nlyrs[partner])
    logger.log("Width of Layers: ", slyrs[partner])
    train(lrs[partner], tss[partner], entps[partner], vcfs[partner], nlyrs[partner], slyrs[partner],jobnumber)
    temp = np.array([0,1])
    temp = temp.astype(float)
      
