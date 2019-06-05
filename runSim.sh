#!/bin/bash

#PBS -l select=1:ncpus=30
#PBS -A MHPCC38870258
#PBS -q standard 
#PBS -l walltime=03:00:00

cd $WORKDIR/autovalves
mpiexec -np 4 ./simulator/expSun31/envSim : -np 4 python learner/ex.py 1e-3,1e-4,1e-5,1e-6 500000,500000,500000,500000
