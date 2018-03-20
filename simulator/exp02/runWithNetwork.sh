#!/bin/bash

#PBS -l select=2:ncpus=10
#PBS -l walltime=0:2:0
#PBS -j oe
#PBS -N exp02
#PBS -q gpu

module load cray-tpsl-64 cray-hdf5
source ~/mypy/bin/activate
cd /mnt/lustre/scratch/$USER/autovalves/simulator/exp02
make
cd ../..
aprun simulator/exp02/exp02 : python network/loadedController.py
chmod -R 777 /mnt/lustre/scratch/autoValveData/*
