BOOST_ROOT := $(WORKDIR)/boost
PYTHON_ROOT := /gpfs/pkgs/mhpcc/anaconda3-5.0.1
CVODE_ROOT := /gpfs/pkgs/mhpcc/sundials-3.1.0

PYTHON_INC := $(PYTHON_ROOT)/include/python3.6m
PYTHON_LIB_LOCATION := $(PYTHON_ROOT)/lib
PYTHON_LIB_FILE := python3.6m

BOOST_INC := $(BOOST_ROOT)/include
BOOST_LIB_LOCATION := $(BOOST_ROOT)/lib
BOOST_LIB_FILE := boost_python36

CVODE_INC := $(CVODE_ROOT)/include
CVODE_LIB_LOCATION := $(CVODE_ROOT)/lib
CVODE_LIB_FILE := sundials_cvode

CC := gcc

CFLAGS := -fPIC

CLinkFlags = -shared -Wl,-soname,$@ -Wl,-rpath

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

chesSimEnv.so: chesSimEnv.cpp
	$(CC) chesSimEnv.cpp $(CFLAGS) $(CInc) -I$(BOOST_INC) -I$(PYTHON_INC) -I$(CVODE_INC) $(CLinkFlags) -L$(PYTHON_LIB_LOCATION) -L$(BOOST_LIB_LOCATION) -L$(CVODE_LIB_LOCATION) -l$(PYTHON_LIB_FILE) -l$(BOOST_LIB_FILE) -l$(CVODE_LIB_FILE) -o chesSimEnv.so

cstr.so: gtSimEnv.cpp gtSimEnv.h
	$(CC) gtSimEnv.cpp $(CFLAGS) $(CInc) -I$(BOOST_INC) -I$(PYTHON_INC) -I$(CVODE_INC) $(CLinkFlags) -L$(PYTHON_LIB_LOCATION) -L$(BOOST_LIB_LOCATION) -L$(CVODE_LIB_LOCATION) -l$(PYTHON_LIB_FILE) -l$(BOOST_LIB_FILE) -l$(CVODE_LIB_FILE) -o cstr.so
