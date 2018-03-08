#include <iostream>
#include <stdlib.h>
#include "mpi.h"
using namespace std;

int main(int argc, char* argv[]) {

  int NUMRUNS=5;

  int rank;
  int size;
  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  cout << "C++ " << rank << " of " << size << endl;

  for (int i = 0; i < NUMRUNS; i++) {
    double state[2];
    state[0]=double(rand()); //instead of this, you would run your simulator
    state[1]=double(rand()); //to assign these values
    cout << "Sending " << state[0] << ' ' << state[1] << endl;

    MPI_Send(state,2,MPI_DOUBLE,1,0,MPI_COMM_WORLD);
    double action[2];
    MPI_Recv(action,2,MPI_DOUBLE,1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
  }

  MPI_Send(NULL,0,MPI_INT,1,2,MPI_COMM_WORLD);

  cout << "C++ closing" << endl;

  MPI_Finalize();
}
