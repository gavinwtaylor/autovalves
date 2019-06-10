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
from mpi4py import MPI
from baselines.common.vec_env.vec_normalize import VecNormalize
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from chem_env import ChemicalEnv
from baselines.ppo2.model import Model
from baselines.common.policies import build_policy
import tensorflow as tf

comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()
partner = rank - (size/2)
partner=int(partner)


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
  model_fn=Model
  gamma=0.99
  lam=0.99
  jobnumber=os.getenv("PBS_JOBID").split('.')[0]
  workdir=os.getenv("WORKDIR")
  loglist=glob.glob(workdir+"/autovalves/learner/logs/*")
  for l in loglist:
     mname= (ntpath.basename(l)[3:]).split('.')
     hname = workdir+"/autovalves/learner/hdf5/"+ mname[0]+".hdf5"
     if(os.path.isfile(hname)):
       loglist.remove(l)
  
  fcount = len(loglist)
  startIndex =partner*(fcount//(size//2))+min(partner,(fcount%(size//2))) 
  endIndex = ((partner+1)*(fcount//(size//2)))+min(partner,(fcount%(size//2)))
  if(fcount%(size//2)) > partner:
    endIndex+=1
  loglist=loglist[startIndex:endIndex]
  
  logger.configure(dir=workdir+"/autovalves/learner/evaluate_logs", format_strs=['stdout', 'log'], log_suffix=jobnumber)
  parser=argparse.ArgumentParser() 
  parser.add_argument('numtrue', help='number of completed runs')
  args=parser.parse_args()
  
  
  env = DummyVecEnv([lambda:ChemicalEnv(comm)])
  setpoint=env.envs[0].setpoint
  x0scaleinv=env.envs[0].x0scaleinv
  x1scaleinv=env.envs[0].x1scaleinv
  env = VecNormalize(env)
 #with open(workdir+"/autovalves/learner/logs/log"+args.jobnm+"-rank00"+args.rank+".txt") as f:
  for name in loglist:
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
      split_k = mname[0].split('k')
      num = str(int(split_k[1]))         
      con_name = split_k[0]+'k'+num
          
      model.load(workdir+"/autovalves/learner/models/"+con_name)
      print("I am about to work with model ", con_name)
      eval_runner=Runner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
      count = 0
      obs, returns, masks, actions, values, neglogpacs, states, epinfos = None, None, None, None, None, None,None, None 

      f = h5py.File(workdir+"/autovalves/learner/hdf5/"+mname[0]+".hdf5","w")
      print("Working with hdf5 file ", mname[0])
      while(count < int(args.numtrue)):
        eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos =eval_runner.run()

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
          #states=np.concatenate((states,eval_states[0:last_true]), axis=0)
          epinfos=np.concatenate((epinfos,eval_epinfos[0:last_true]), axis=0)

    rewards = calcReward(obs, x0scaleinv, x1scaleinv, setpoint)
    numruns = 0
    lastone=-1
    thisone=0
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
  
  temp = np.array([0,1])
  temp=temp.astype(float) 
  comm.Send(temp, dest=partner, tag=2)    
