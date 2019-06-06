import h5py

with h5py.File('01-run03.h5') as f:
  l=f['Run-0099986']['rewards']
  actions=l[-10:]
  print(actions)
