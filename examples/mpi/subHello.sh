#!/bin/bash
#PBS -l select=2:ncpus=24

cd /mnt/lustre/scratch/$USER/autovalves/examples/mpi
aprun -n 48 hello
