import os
import sys
import numpy as np
from baselines import logger
from collections import deque
from baselines.ppo2.runner import Runner
import argparse
from mpi4py import MPI
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from chem_env import ChemicalEnv
from baselines.ppo2.model import Model
from baselines.common.policies import build_policy

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
partner = rank - (size/2)
partner=int(partner)
 
if __name__ == '__main__':
  model_fn=Model
  gamma=0.99
  lam=0.99
  jobnumber=os.getenv("PBS_JOBID").split('.')[0]
  workdir=os.getenv("WORKDIR")
  logger.configure(dir=workdir+"/autovalves/learner/evaluate_logs", format_strs=['stdout', 'log'], log_suffix=jobnumber)
  parser=argparse.ArgumentParser() 
  parser.add_argument('rank',help='rank of log file')
  parser.add_argument('jobnm', help='jobnumber of log file')
  args=parser.parse_args()
  
  with open(workdir+"/autovalves/learner/logs/log"+args.jobnm+"-rank00"+args.rank+".txt") as f:
    for line in f:
      if "Number" in line and "Layers" in line:
        num_layers = line.split()[-1]
        num_layers=int(num_layers)
      if "Width" in line:
        layer_width=line.split()[-1]
        layer_width=int(layer_width)
      if "Entropy" in line:
        ent_coef=line.split()[-1]
        ent_coef=float(ent_coef)
      if "Value" in line:
        vf_coef=line.split()[-1] 
        vf_coef=float(vf_coef)
  
  env = DummyVecEnv([lambda:ChemicalEnv(comm)])
  env = VecNormalize(env)
  network = "mlp"
  ob_space=env.observation_space
  ac_space = env.action_space
  nevs = env.num_envs
  nsteps = 2048
  nbatch = nevs * nsteps
  nminibatches = 4
  nbatch_train = nbatch // nminibatches
  max_grad_norm=0.5
  print("about to build policy")
  policy=build_policy(env,network, num_layers=num_layers, num_hidden=layer_width)
  print("Policy built")
  model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nevs, nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=rank)
  print("Model built, about to laod it")
  model.load(workdir+"/autovalves/learner/models/"+args.jobnm+"-rank"+args.rank)
  print("Model loaded!")
  eval_runner=Runner(env=env, model=mod, nsteps=nsteps, gamma=gamma, lam=lam)
  eval_runner.run()
