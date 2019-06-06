import itertools
import subprocess

lrs = [1e-2]
numiters = [100000]
entropy = [0]
value = [0.25]
layers = [2]
perc = [64]

def createScript(combos):
    cmd = 'mpiexec -np ' + str(len(combos)) 
    cmd2 = ' ./simulator/expSun31/envSim : -np ' + str(len(combos))
    cmd3 = ' python learner/ex.py'

    rateList  = [str(combo[0]) for combo in combos]
    tstepList = [str(combo[1]) for combo in combos]
    entpsList = [str(combo[2]) for combo in combos]
    vcfsList  = [str(combo[3]) for combo in combos]
    nlyrsList = [str(combo[4]) for combo in combos]
    slyrsList = [str(combo[5]) for combo in combos]

    rates  = ','.join(rateList)
    tsteps = ','.join(tstepList)
    entps  = ','.join(entpsList)
    vcfs   = ','.join(vcfsList)
    nlyrs  = ','.join(nlyrsList)
    slyrs  = ','.join(slyrsList)


    cmd = cmd + cmd2 + cmd3 + ' ' + rates + ' ' + tsteps + ' ' + entps + ' ' + vcfs + ' ' + nlyrs + ' ' + slyrs
    
    script = """#!/bin/bash

    #PBS -l select=1:ncpus=30
    #PBS -A MHPCC38870258
    #PBS -q debug
    #PBS -l walltime=00:30:00

    cd $WORKDIR/autovalves
    """ + cmd
    print('\n\n',script)
    subprocess.run(['qsub'],input=script,encoding='ascii')


combinations=[combo for combo in itertools.product(lrs,numiters,entropy, value,layers,perc)]
while len(combinations) > 0: 
    createScript(combinations[-4:])
    for i in range( min(4,len(combinations)) ):
        combinations.pop()
    