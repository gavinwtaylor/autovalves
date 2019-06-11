all: Code_UpdatedSundialVersion.cpp
	g++ Code_UpdatedSundialVersion.cpp -I/gpfs/pkgs/mhpcc/sundials-3.1.0/include -I/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/include -L/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/lib -lhdf5_cpp -lhdf5 -L/gpfs/pkgs/mhpcc/sundials-3.1.0/lib -lsundials_cvode

cv: newcvode.cpp
	g++ newcvode.cpp -g -I/gpfs/pkgs/mhpcc/sundials-3.1.0/include -I/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/include -L/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/lib -lhdf5_cpp -lhdf5 -L/gpfs/pkgs/mhpcc/sundials-3.1.0/lib -lsundials_cvode -o newcv

envSim: envSim.cpp
	g++ envSim.cpp -I/gpfs/pkgs/mhpcc/sundials-3.1.0/include -I/gpfs/pkgs/ibm/spectrum_mpi/10.1.0.3/include -L/gpfs/pkgs/ibm/spectrum_mpi/10.1.0.3/lib -L/gpfs/pkgs/mhpcc/sundials-3.1.0/lib -lsundials_cvode -lmpi_ibm -o envSim

newSim: newSim.cpp
	g++ newSim.cpp -I/gpfs/pkgs/mhpcc/sundials-3.1.0/include -I/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/include -L/gpfs/pkgs/mhpcc/hdf5-1.8.18/openmpi-2.1.1/lib -lhdf5_cpp -lhdf5 -L/gpfs/pkgs/mhpcc/sundials-3.1.0/lib -lsundials_cvode -o newSim


multEnvSim: multEnvSim.cpp
	g++ multEnvSim.cpp -g -I/gpfs/pkgs/mhpcc/sundials-3.1.0/include -I/gpfs/pkgs/ibm/spectrum_mpi/10.1.0.3/include -L/gpfs/pkgs/ibm/spectrum_mpi/10.1.0.3/lib -L/gpfs/pkgs/mhpcc/sundials-3.1.0/lib -lsundials_cvode -lmpi_ibm -o multEnvSim


