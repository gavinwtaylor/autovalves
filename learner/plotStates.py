import h5py
import random
import argparse
import os
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('fname', help='name of the hdf5 file')
args=parser.parse_args()
filename=args.fname

workDir = os.getenv('WORKDIR')
name = workDir + '/autovalves/learner/hdf5/' + filename

with h5py.File(name) as f:
    myList = list(f.keys())

    if (len(f.keys()) == 0):
        print('The hdf5 file is empty!')
        exit()

    #GET 3 RANDOM NUMBERS
    random.shuffle(myList)
    myList = myList[:3]

    #GET 3 RUNS BASED ON THE NUMBERS
    for run in myList:
        l=f[run]['states']
        states=l[:]

        i = 0;
        while i < len(states):
          states[i][1] = states[i][1]*100+310
          i += 1
        print(states)

        plt.plot(states[:,0],states[:,1],'.')
plt.show()



