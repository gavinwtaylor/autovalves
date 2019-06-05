#!/bin/bash

#PBS -l select=1:ncpus=30
#PBS -A MHPCC38870258
#PBS -q standard 
#PBS -l walltime=00:30:00

cd autovalves
mpiexec -np 1 ./simulator/expSun31/envSim : -np 1 python learner/ex.py
