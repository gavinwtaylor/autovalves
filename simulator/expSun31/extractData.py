import h5py
import glob
import os
from mpi4py import MPI

#MPI INFO
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

workDir = os.getenv('WORKDIR')
h5list = glob.glob(workDir + '/autovalves/learner/hdf5/*')
fCount = len(h5list)

startIndex = rank * (fCount//size) + min(rank, (fCount % size))
endIndex = ((rank + 1) * (fCount//size)) + min(rank, (fCount % size))

if (fCount % size) > rank:
    endIndex += 1

h5list=h5list[startIndex:endIndex]

for name in h5list:
        
    with h5py.File(name) as f:
        rewardAvg = 0
        stateAvg = 0
        numFiles = 0

        for page in f:
            #AVERAGE FOR REWARD IN ONE RUN
            l=f[page]['rewards']
            rewards=l[-10:]
            avg = 0
            for i in rewards:
                avg += i
            avg = avg / 10

            #NUMBER OF STATES IN ONE RUN
            l=f[page]['states']
            numStates = 0
            for whatever in l:
                numStates += 1

            rewardAvg += avg
            stateAvg += numStates
            numFiles += 1

        rewardAvg /= numFiles
        stateAvg /= numFiles

        print('File name = ', name)
        print('Average rewards = ', rewardAvg)
        print('Average # of states = ', stateAvg)
        print('Total # of files = ', numFiles, '\n\n')
