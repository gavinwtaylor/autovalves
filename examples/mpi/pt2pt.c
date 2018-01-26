#include <stdio.h>
#include <string.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
  int rank;
  int numprocs;

  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

  if (rank!=0)
  {
    //ptr to data, count, type, destination, tag, communicator
    MPI_Send(&rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  } else 
  {
    int receivedRank;
    for (int sender=1; sender < numprocs; sender++) {
      //ptr to receiving buffer, count, type, source, tag, communicator,
      //status
      MPI_Recv(&receivedRank, 1, MPI_INT, sender, 0, MPI_COMM_WORLD,
          MPI_STATUS_IGNORE);
      printf("Rank %d received a %d from sender %d!\n",rank,receivedRank,sender);
    }
  }

  MPI_Finalize();
}
