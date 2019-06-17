import itertools
import subprocess

lrs = [1e-6, 1e-8, 1e-10]  #learning rate
numiters = [5000000000]    #timesteps
entropy = [0]              #value for randomness
value = [0.5]              #value coefficient
layers = [3]               #number of layers within neural net
perc = [64]                #width of neural net
envs = [10]                #number of environments
numNodes = 3               #number of nodes

def createScript(combos):
    cmd = 'mpiexec -np ' + str(4*numNodes) + ' python learner/cstr_ex.py ' 

    for combo in combos:
      selection = ' '.join([str(c) for c in combo])

      #assemble the command for each combo
      cmd2 = cmd + selection
      #assemble a script
      script = """#!/bin/bash

      #PBS -l select=""" + str(numNodes) + """:ncpus=20:mpiprocs=4
      #PBS -A MHPCC38870258
      #PBS -q standard
      #PBS -l walltime=36:00:00

      cd $WORKDIR/autovalves
      """ + cmd2
      #qsub the script for each combo
      subprocess.run(['qsub'],input=script,encoding='ascii')


#create every combination for the parameters above
combinations=[combo for combo in itertools.product(lrs,numiters,entropy, value,layers,perc, envs)]
while len(combinations) > 0: 
    #assign four per node 
    createScript(combinations[-4:])
    for i in range( min(4,len(combinations)) ):
        combinations.pop()
    
