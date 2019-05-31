all: hello status pt2pt collective

hello: hello.c
	CC hello.c -o hello

status: status.c
	CC status.c -o status

pt2pt: pt2pt.c
	CC pt2pt.c -o pt2pt

collective: collective.c
	CC collective.c -o collective

cppAndPy: cppAndPy.cpp
	g++ -I/gpfs/pkgs/ibm/spectrum_mpi/10.1.0.3/include -L/gpfs/pkgs/ibm/spectrum_mpi/10.1.0.3/lib -lmpi_ibm cppAndPy.cpp -o cppAndPy

clean:
	rm hello status pt2pt collective
