import os
import os.path
import sys
import numpy as np
import h5py
import glob
import ntpath
from baselines import logger
from collections import deque
from baselines.ppo2.runner import Runner
import argparse
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from cstrEnv import *
from baselines.ppo2.model import Model
from baselines.common.policies import build_policy
import tensorflow as tf
from mpi4py import MPI
from baselines.common.mpi_util import setup_mpi_gpus

#setup MPI stuff
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

def calcReward(states, x0scaleinv, x1scaleinv, setpoint):
    n=states.shape[0]
    rewards=np.empty(n)
    for i in range(n):
        r = -(
              ((states[i][0] - setpoint[0]) * x0scaleinv) * 
              ((states[i][0] - setpoint[0]) * x0scaleinv) +
              ((states[i][1] - setpoint[1]) * x1scaleinv) * 
              ((states[i][1] - setpoint[1]) * x1scaleinv)
             )
        rewards[i] = r
    return rewards


if __name__ == '__main__':
  setup_mpi_gpus() #prevents GPU hogging
  model_fn=Model 
  gamma=0.99 #hardcoded in these values
  lam=0.99
  jobnumber=os.getenv("PBS_JOBID").split('.')[0]
  workdir=os.getenv("WORKDIR")
  loglist=glob.glob(workdir+"/autovalves/learner/logs/*") #all of the log files in the logs directory
  for l in loglist:  
     mname= (ntpath.basename(l)[3:]).split('.') 
     hname = workdir+"/autovalves/learner/hdf5/"+ mname[0]+".hdf5"
     if(os.path.isfile(hname)):
       loglist.remove(l) #only want to evaluate log files that haven't beeen evaluated yet
  fcount = len(loglist) 
  #divide up log list to each of the processes
  startIndex =rank*(fcount//(size//2))+min(rank,(fcount%(size//2))) 
  endIndex = ((rank+1)*(fcount//(size//2)))+min(rank,(fcount%(size//2)))
  if(fcount%(size//2)) > rank:
    endIndex+=1
  
  
  logger.configure(dir=workdir+"/autovalves/learner/evaluate_logs", format_strs=['stdout', 'log'], log_suffix=jobnumber)
  parser=argparse.ArgumentParser() 
  parser.add_argument('numtrue', help='number of completed runs') #argument for number of trajectories
  args=parser.parse_args()
  
  
  env = DummyVecEnv([lambda:CSTREnvironment()])
  setpoint, x0scale, x1scale = env.envs[0].getrewardstuff() #info needed from environment
  env = VecNormalize(env) 
  for name in loglist: #retrieving data from each of the log files
    with open(name) as f:
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

    #parameters needed such that the model can be rebuilt
    network = "mlp"
    ob_space=env.observation_space
    ac_space = env.action_space
    nevs = env.num_envs
    nsteps = 2048
    nbatch = nevs * nsteps
    nminibatches = 4
    nbatch_train = nbatch // nminibatches
    max_grad_norm=0.5
    with tf.Session(graph=tf.Graph()):
      policy=build_policy(env,network, num_layers=num_layers, num_hidden=layer_width)
      model = model_fn(policy=policy, ob_space=ob_space, ac_space=ac_space, nbatch_act=nevs, nbatch_train=nbatch_train, nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm)
      mname= (ntpath.basename(name)[3:]).split('.')
      con_name = mname[0]
      print("Model name: "+con_name)
      model.load(workdir+"/autovalves/learner/models/"+con_name)
      eval_runner=Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam) #load the model
      count = 0
      obs, returns, masks, actions, values, neglogpacs, states, epinfos = None, None, None, None, None, None,None, None 

      f = h5py.File(workdir+"/autovalves/learner/hdf5/"+mname[0]+".hdf5","w") 
      while(count < int(args.numtrue)):
         #evaluation metrics
        eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos =eval_runner.run()
   
        #obtaining data from completed runs (Trues)
        for index,i in enumerate(eval_masks):
          if i == True:
            count = count+1
            last_true = index
        if masks is None:
          obs = eval_obs
          returns = eval_returns
          masks = eval_masks
          actions=eval_actions
          values=eval_values
          neglogpacs=eval_neglogpacs
          states=eval_states
          epinfos=eval_epinfos
        else:
          last_true+=1
          obs= np.concatenate((obs,eval_obs[0:last_true,:]), axis=0)
          returns = np.concatenate((returns, eval_returns[0:last_true]), axis=0)
          masks= np.concatenate((masks,eval_masks[0:last_true]), axis=0)
          actions=np.concatenate((actions, eval_actions[0:last_true,:]), axis=0)
          values=np.concatenate((values,eval_values[0:last_true]), axis=0)
          neglogpacs=np.concatenate((neglogpacs,eval_neglogpacs[0:last_true]), axis=0)
          epinfos=np.concatenate((epinfos,eval_epinfos[0:last_true]), axis=0)
    actions=np.apply_along_axis(action_net_to_env,0,actions)
    x0scaleinv=1.0/x0scale
    x1scaleinv=1.0/x1scale
    rewards = calcReward(obs, x0scaleinv, x1scaleinv, setpoint)
    numruns = 0
    lastone=-1
    thisone=0
    
    #create the hdf5 file with the states, actions, and rewards that correspond to a particular log file
    for ind, j in enumerate(masks):
      if j == True:
        if(numruns == int(args.numtrue)):
          break	
        numruns = numruns+1
        thisone=ind
        grp = f.create_group("runnumber_"+str(thisone))
        dset1=grp.create_dataset("states",(thisone-lastone,2), dtype=obs.dtype)
        dset2=grp.create_dataset("actions",(thisone-lastone,2), dtype=actions.dtype)
        dset3=grp.create_dataset("rewards",(thisone-lastone, ), dtype=rewards.dtype)
        dset1[:]=obs[lastone+1:thisone+1,:]
        dset2[:]=actions[lastone+1:thisone+1,:]
        dset3[:]=rewards[lastone+1:thisone+1]
        lastone=ind
  
