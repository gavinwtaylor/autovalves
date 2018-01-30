#!/bin/bash
#PBS -l select=1:ncpus=24
#PBS -j oe

#the -j pbs flag puts the stdout and stderr in the same file

cd /mnt/lustre/scratch/$USER/autovalves/examples/hdf
aprun readOneRow
