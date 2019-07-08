import itertools
import subprocess

lrs = [1e-4, 1e-6, 1e-10]  #learning rate
numiters = [25000000]    #timesteps
entropy = [0,10]              #value for randomness
value = [0.5,5]              #value coefficient
layers = [3]               #number of layers within neural net
perc = [64]                #width of neural net
envs = [10]                #number of environments
numNodes = 1               #number of nodes

def createScript(combos):
    cmd = 'mpiexec -np ' + str(4*numNodes) + ' python learner/cstr_ex.py ' 

    jobids=[]

    for combo in combos:
      selection = ' '.join([str(c) for c in combo])

      #assemble the command for each combo
      cmd2 = cmd + selection
      #assemble a script
      script = """#!/bin/bash

      #PBS -l select=""" + str(numNodes) + """:ncpus=20:mpiprocs=4
      #PBS -A MHPCC38870258
      #PBS -q standard
      #PBS -l walltime=30:00:00
      #PBS -N chemRL
      #PBS -j oe

      module purge
      module load pbs tensorflow/1.8.0 spectrum-mpi cuda/9.2
      export PYTHONPATH="$WORKDIR/autovalves/simulator/expSun31"
      cd $WORKDIR/autovalves
      """ + cmd2 +"""
      cd learner/logs
      """
      #qsub the script for each combo
      sp=subprocess.run(['qsub'],input=script,encoding='ascii',stdout=subprocess.PIPE)
      jobids.append(sp.stdout.split('.')[0])
    return jobids


#create every combination for the parameters above
combinations=[combo for combo in itertools.product(lrs,numiters,entropy, value,layers,perc, envs)]
jobids=createScript(combinations)
print('started',jobids)

script="""#!/bin/bash
#PBS -l select=2:ncpus=20:mpiprocs=4
#PBS -A MHPCC38870258
#PBS -q standard
#PBS -l walltime=30:00:00
#PBS -N evaluator
#PBS -j oe
#PBS -M taylor@usna.edu
#PBS -W depend=after:"""+':'.join(jobids)+"""

module purge
module load pbs tensorflow/1.8.0 spectrum-mpi cuda/9.2
export PYTHONPATH="$WORKDIR/autovalves/simulator/expSun31"
cd $WORKDIR/autovalves
mpiexec -np 8 python learner/cstr_evaluate.py 100
"""

subprocess.run(['qsub'],input=script,encoding='ascii')
