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
 
  printf("hello from the CPP program, rank %d (of %d)\n"
    , rank
    , numprocs
	);

  //Build an array, and send it to the other process
  double* someInts=new double[4];
  for (int i = 0; i < 4; i++)
    someInts[i]=(double)i;
  /* pointer, number of elements, type of data, destination rank,
   * tag, MPI_COMM_WORLD */
  MPI_Send(someInts,4,MPI_DOUBLE,1,0,MPI_COMM_WORLD);

  //Receive the changed array
  /* pointer, number of elements, type of data, source rank, tag,
   * MPI_COMM_WORLD, status object*/
  MPI_Recv(someInts,4,MPI_DOUBLE,1,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  std::cout << "cpp program received ";
  for (int i = 0; i < 4; i++)
    std::cout << someInts[i]<<' '; //Look, it's changed
  std::cout <<std::endl;

  //Receive the array, and extract source and tag info
  MPI_Status status;
  //Can use either or both of MPI_ANY_SOURCE and MPI_ANY_TAG
  MPI_Recv(someInts,4,MPI_DOUBLE,MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD, &status);
  std::cout << "Received a message from sender "<<status.MPI_SOURCE
            <<" with tag "<<status.MPI_TAG<<std::endl;

  MPI_Finalize();
}
