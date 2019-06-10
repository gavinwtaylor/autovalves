
#!/bin/bash

#PBS -l select=1:ncpus=30
#PBS -A MHPCC38870258
#PBS -q standard 
#PBS -l walltime=36:00:00

export PYTHONWARNINGS="ignore"
cd $WORKDIR/autovalves
mpiexec -np 4 ./simulator/expSun31/envSim : -np 4 python learner/evaluate.py 100


