#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include "mpi.h"

int main(int argc, char *argv[])
{
  int rank;
  int numprocs;
  int i;
  char hostnm[80];
  FILE *fp;

  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
 
  fp = popen("cat /proc/sys/kernel/hostname", "r");
  fscanf(fp, " %s", hostnm);
  fclose(fp);

  printf("hello from rank %d (number %d of %d) on host: %s\n"
    , rank
    , rank+1
    , numprocs
    , hostnm
	);

  MPI_Finalize();
}
