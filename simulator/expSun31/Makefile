all:
	g++ Code_UpdatedSundialVersion.cpp -I/gpfs/pkgs/mhpcc/sundials-3.1.0/include -I/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/include -L/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/lib -lhdf5_cpp -lhdf5 -L/gpfs/pkgs/mhpcc/sundials-3.1.0/lib -lsundials_cvode

cv:
	g++ newcvode.cpp -I/gpfs/pkgs/mhpcc/sundials-3.1.0/include -I/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/include -L/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/lib -lhdf5_cpp -lhdf5 -L/gpfs/pkgs/mhpcc/sundials-3.1.0/lib -lsundials_cvode

envSim:
	g++ envSim.cpp -I/gpfs/pkgs/mhpcc/sundials-3.1.0/include -I/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/include -L/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/lib -lhdf5_cpp -lhdf5 -L/gpfs/pkgs/mhpcc/sundials-3.1.0/lib -lsundials_cvode -o envSim

chesley:
	g++ newSim.cpp -I/gpfs/pkgs/mhpcc/sundials-3.1.0/include -I/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/include -L/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/lib -lhdf5_cpp -lhdf5 -L/gpfs/pkgs/mhpcc/sundials-3.1.0/lib -lsundials_cvode -o newSim
