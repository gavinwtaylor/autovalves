#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"
#include <iostream>

int main(int argc, char *argv[])
{
  int rank;
  int numprocs;
  int i;

  /* Initialization stuff - get the rank and size */
  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
 
  std::cout<<"C++ program with rank "<<rank<<std::endl;

}
