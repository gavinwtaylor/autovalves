import itertools
import subprocess

lrs = [1e-2, 1e-4, 1e-6]
numiters = [5000]
entropy = [0]   
value = [0.5]
layers = [2]
perc = [64,128,256]
envs = [5]
numNodes = 5

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

combinations=[combo for combo in itertools.product(lrs,numiters,entropy, value,layers,perc, envs)]
while len(combinations) > 0: 
    createScript(combinations[-4:])
    for i in range( min(4,len(combinations)) ):
        combinations.pop()
    
