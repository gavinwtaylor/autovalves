import h5py
import random
import argparse
import os
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser()
parser.add_argument('fname', help='name of the hdf5 file')
args=parser.parse_args()
filename=args.fname

name = 'hdf5/' + filename

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
        l=f[run]['actions']
        actions=l[:]
        l=f[run]['rewards']
        rewards=l[:]

        i = 0;

        plt.figure(0)
        plt.plot(states[:20,0],states[:20,1],'.')
        plt.figure(1)
        plt.plot(actions[:20,0],actions[:20,1],'.')
        plt.figure(2)
        plt.plot(rewards[:20])
plt.figure(0)
plt.title('states')
plt.plot([.57],[395.3],'ro')
plt.figure(1)
plt.title('actions')
plt.figure(2)
plt.title('rewards')
plt.show()



