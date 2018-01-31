#include <iostream>
using std::cout;
using std::cout;
using std::endl;
#include <string>
using std::string;
#include "H5Cpp.h"
using namespace H5;

/****************
  States is NxWhatever, where states[i] is a pointer to an array of all the
  state info for the i-th timestep

  actions is NxWhatever, where actions[i] is a point to an array showing the
  valve settings used in response to being in states[i]

  rewards is 1D of size N, and rewards[i] is the numerical value detailing how
  good it is to be in states[i]

  N is the number of timesteps recorded

  statesDim is the number of observations in one time step
  
  actionDim is the number of valves being set

  file is a pointer to the HDF5 file object

  groupname is the group to be written to within that file
 */
void writeData(double** states, double** actions, double* rewards,
               int N, int statesDim, int actionDim,
               H5File* file, string groupname);

int main() {
  string filename="/mnt/lustre/scratch/autoValveData/test.h5";
  H5File* file=new H5File(filename, H5F_ACC_EXCL);

  const int N=5, S_DIM=2, A_DIM=2;
  double** states=new double*[N];
  double** actions = new double*[N];
  double* rewards = new double[N];
  double sVal=0, aVal=10, rVal=100;
  for (int i = 0; i < N; i++){
    states[i] = new double[S_DIM];
    for (int j = 0; j < S_DIM; j++) {
      states[i][j]=sVal;
      sVal+=.1;
    }
    actions[i] = new double[A_DIM];
    for (int j = 0; j < A_DIM; j++) {
      actions[i][j]=aVal;
      aVal+=.1;
    }
    rewards[i]=rVal;
    rVal+=.1;
  }

  writeData(states, actions, rewards, N, S_DIM, A_DIM, file, "testGroup");

  for (int i = 0; i < N; i++) {
    delete [] states[i];
    delete [] actions[i];
  }
  delete [] states;
  delete [] actions;
  delete [] rewards;
  delete file;
}

void writeData(double** states, double** actions, double* rewards,
               int N, int statesDim, int actionDim,
               H5File* file, string groupname){
  Group group=file->createGroup("/"+groupname);

  //write states
  hsize_t S_DIMS[2]={N,statesDim};
  DataSpace stateDataspace(2,S_DIMS);
  DataSet dataset = 
    group.createDataSet("states",PredType::NATIVE_DOUBLE,stateDataspace);
  hsize_t M_DIMS[2]={1,statesDim};
  DataSpace memspace(2, M_DIMS);

  for (int row = 0; row < N; row++) {
    hsize_t offset[2]={row,0};
    stateDataspace.selectHyperslab(H5S_SELECT_SET,M_DIMS,offset);
    dataset.write(states[row],PredType::NATIVE_DOUBLE,memspace,stateDataspace,NULL);
  }

  //writeactions
  hsize_t A_DIMS[2]={N,actionDim};
  DataSpace actionDataspace(2,A_DIMS);
  dataset = 
    group.createDataSet("actions",PredType::NATIVE_DOUBLE,actionDataspace);
  M_DIMS[1]=actionDim;
  DataSpace actionmemspace(2, M_DIMS);

  for (int row = 0; row < N; row++) {
    hsize_t offset[2]={row,0};
    actionDataspace.selectHyperslab(H5S_SELECT_SET,M_DIMS,offset);
    dataset.write(actions[row],PredType::NATIVE_DOUBLE,actionmemspace,actionDataspace,NULL);
  }

  //writeRewards
  hsize_t R_DIMS=N;
  DataSpace rewardDataspace(1,&R_DIMS);
  dataset=
    group.createDataSet("rewards",PredType::NATIVE_DOUBLE,rewardDataspace);
  dataset.write(rewards,PredType::NATIVE_DOUBLE);

}

















