#include <iostream>
#include "mpi.h"

int main(int argc, char* argv[]) {
  int rank;
  int size;
  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

  cout << "C++ " << rank;

  MPI_Finalize();
}
