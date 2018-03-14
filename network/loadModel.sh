#!/bin/bash

#PBS -l select=2:ncpus=10
#PBS -l walltime=0:2:0
#PBS -j oe
#PBS -N trivValveInter
#PBS -q gpu

source ~/mypy/bin/activate
cd /mnt/lustre/scratch/$USER/autovalves/network/
CC trivMPISim.cpp -o trivMPISim
aprun ./trivMPISim : python loadedController.py
