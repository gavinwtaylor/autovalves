import h5py
import glob
import os
from mpi4py import MPI

#MPI INFO
comm=MPI.COMM_WORLD
rank=comm.Get_rank()
size=comm.Get_size()

workDir = os.getenv('WORKDIR')
h5list = glob.glob(workDir + '/autovalves/learner/hdf5/*.hdf5')
fCount = len(h5list)

startIndex = rank * (fCount//size) + min(rank, (fCount % size))
endIndex = ((rank + 1) * (fCount//size)) + min(rank, (fCount % size))

if (fCount % size) > rank:
    endIndex += 1

h5list=h5list[startIndex:endIndex]

F = open("stats.txt", "a+")

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
        print("to the file!")

        #PRINT STATS TO THE FILE
        F.write('File name = ')
        F.write(name)
        F.write('\nTotal number of Files = ')
        F.write(str(numFiles))
        F.write('\nTotal average rewards in h5 File = ')
        F.write(str(rewardAvg))
        F.write('\nAverage number of states in h5 File = ')
        F.write(str(stateAvg))
        F.write('\n\n')

F.close()
