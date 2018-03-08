#!/bin/bash

#PBS -l select=2:ncpus=24
#PBS -l walltime=0:2:0
#PBS -j oe
#PBS -N trivValveInter

source ~/mypy/bin/activate
cd /mnt/lustre/scratch/taylor/autovalves/network/
CC trivMPISim.cpp -o trivMPISim
aprun ./trivMPISim : python trivMPIControl.py
