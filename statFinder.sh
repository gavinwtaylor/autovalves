#!/bin/bash

#PBS -l select=1:ncpus=30
#PBS -A MHPCC38870258
#PBS -q standard 
#PBS -l walltime=36:00:00
#PBS -W depend=afterok:65687

cd $WORKDIR/autovalves
mpiexec -np 20 python simulator/expSun31/extractData.py

