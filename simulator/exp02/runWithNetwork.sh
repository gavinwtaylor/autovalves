#!/bin/bash

#PBS -l select=2:ncpus=10
#PBS -l walltime=0:2:0
#PBS -j oe
#PBS -N trivValveInter
#PBS -q gpu

module load cray-tpsl-64 cray-hdf5
source ~/mypy/bin/activate
cd /mnt/lustre/scratch/$USER/autovalves/
aprun simulator/exp02/exp02 : python network/loadedController.py
