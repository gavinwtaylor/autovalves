import h5py,os,glob,ntpath
import matplotlib.pyplot as plt
import numpy as np

sortOn=lambda run: np.average(run['rewards'][:])

learnerDir=os.path.dirname(os.path.realpath(__file__))
learnerDir=learnerDir+'/learner/'

def logfile(jobid):
  return learnerDir+'logs/log'+jobid+'.txt'

def h5file(jobid):
  return learnerDir+'hdf5/manystarts/'+jobid+'.hdf5'

def readLog(jobid):
  info=dict()
  with open(logfile(jobid)) as f:
    for line in f:
      if 'Learning Rate' in line:
        info['lr']=float(line.split()[-1])
      elif 'Timestep' in line:
        info['timesteps']=float(line.split()[-1])
      elif 'Entropy' in line:
        info['entropy']=float(line.split()[-1])
      elif 'Value' in line:
        info['value']=float(line.split()[-1])
      elif 'Number of Layers' in line:
        info['depth']=float(line.split()[-1])
      elif 'Width of Layers' in line:
        info['width']=float(line.split()[-1])
        break
  return info

def applyaverage(jobid,func):
  print(h5file(jobid))
  results=[]
  with h5py.File(h5file(jobid)) as f:
    for run in f:
      results.append(func(f[run]))
  return sum(results)/len(results)

jobids=[ntpath.basename(fn).split('.')[0] for fn in glob.glob(learnerDir+'hdf5/*.hdf5')]
if 'manystarts' in jobids:
  jobids.remove('manystarts')
if 'onestart' in jobids:
  jobids.remove('onestart')
if 'manystartmanytarg' in jobids:
  jobids.remove('manystartmanytarg')
if 'pid' in jobids:
  jobids.remove('pid')

info=dict()
for jobid in jobids:
  info[jobid]=readLog(jobid)
  try:
    info[jobid]['result']=applyaverage(jobid,sortOn)
  except:
    print('failed to load',jobid)
    del info[jobid]
info['orig']=dict()

info['orig']['result']=applyaverage('pid',sortOn)

sortedResults=sorted(info,key=lambda jobid: info[jobid]['result'])
for result in sortedResults:
  print(result,info[result])
