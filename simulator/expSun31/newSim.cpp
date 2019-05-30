#include <iostream>                                  
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <sstream>
#include <stdio.h>
using namespace std;

// Header files for CVODE
#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <cvode/cvode_direct.h>       /* prototype for CVDense */
#include <sundials/sundials_dense.h> /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h> /* definition of type realtype */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */

// Header files for HD5
#include "H5Cpp.h"
using namespace H5;

// prototype for the function we have
static int cstrfun2(realtype t, N_Vector x, N_Vector xp, void *user_data);

// prototype for its jacobian (DID NOT USE)

// prototype for HD5 storage function
void writeData(vector< vector<double> >& states, 
    vector< vector<double> >& actions,
    vector<double>& rewards,
    int N, int statesDim, int actionDim,
    H5File* file, string groupname);

// user_data passed through cv_ode that contains the persistant information for the controller
struct controller_data{
  vector<double> uprev; 
  vector<double> Kps; 
  vector<double> Kis; 
  vector<double>* tdat;
  vector< vector<double> >* xdat;
  vector<double> Integral; 
};

// prototype for the control action
//static int controller(realtype t, N_Vector x, N_Vector xsp, controller_data* cdata, vector<double>* u);

static void exit(N_Vector* x, N_Vector* abstol, void *cvode_mem);

static void reset(vector<double>* u, N_Vector* x, N_Vector* xsp, controller_data* cdata, vector<vector<double>>* xdat, vector<double>* rdat, vector<vector<double>>* udat, int* i, double* rad, double* x0scale, double* x1scale, void* cvode_mem);

static bool action(void* cvode_mem, realtype t, realtype tstep, N_Vector* x, int* runswitherrors, controller_data* cdata, vector<double>* tdat, vector<vector<double>>* xdat, vector<double>* rdat, vector<double>* x0, vector<vector<double>>* udat, vector<double> u0, N_Vector* xsp, double x0scaleinverse, double x1scaleinverse, int rewardcheck, double* rewardsum, double rewardtol);
 
int main(void) {
  //**MY VERSION**
  //
  int flag, i, p;
  clock_t clockin, clockout;
  
  // allocate memory for results storage and provide initial variables --> **init**
  vector<double> tdat;  // initial state and main memory object for t
  vector<double> x0(2, 0); // empty vector  
  vector< vector<double> > udat; // u state space. start with zeros [CAin, Q].
  vector< vector<double> > xdat;  // x state space history (reporting intervals)
  vector<double> rdat; // reward data
  vector<double> u0(2, 0); // the current control action.
  vector<double> clearvec(2,0); // something convenient
  controller_data cdata;
  cdata.uprev = u0;
  double rad; //place on oval



  // steady state checkers
  int j, k;  // storage for temporary sums used later

  //stay in simulator
  double rewardsum;
  int rewardcheck = 10; // how many timesteps to check?
  double rewardtol = 0.0001; // the reward tolerance for when we've reached our goal. 
  // Since it's scaled on the rnage, a reward tolernace of 0.0001 means within 0.01% of the range

  // this is our desired setpoint --> **leave in simulator...goal state**
  N_Vector xsp     = N_VNew_Serial(2);

  // CV ODE types for use with the integrator. Initial states and settings. --> **init**
  N_Vector x       = N_VNew_Serial(2); // create an x vector of two state variables

  N_Vector abstol  = N_VNew_Serial(2); // the vector of absolute tolerances
  NV_Ith_S(abstol, 0) = RCONST(1.0e-8);  // ok with CA (0...2) to the 1e-8      mol/m3
  NV_Ith_S(abstol, 1) = RCONST(1.0e-6);  // ok with T (300-400) to the 1e06

  realtype reltol  = RCONST(1.0e-4); // ok with relative error of 0.01%

  // Create the CV ODE memory space and provide settings.>
  void *cvode_mem;
  cvode_mem =CVodeCreate(CV_BDF,CV_NEWTON);             // specify using backwards difference methods (stiff)
  CVodeInit(cvode_mem, cstrfun2, RCONST(0.0), x);  // initialize at time zero
  CVodeSVtolerances(cvode_mem, reltol, abstol);    // specify the tolerances
  SUNDenseMatrix(2, 2);                           // specify the dense linear solver
  CVodeSetMaxNumSteps(cvode_mem, 5000);             // sets the maximum number of steps
  CVodeSetUserData(cvode_mem, &u0);                // sets the user data pointer

  realtype tstep = RCONST(0.01);
  realtype tfin = RCONST(10);
  realtype t;
  realtype maxit = tfin / tstep + RCONST(100);

  double rad = rand() *(2.0 * M_PI);

  double x0scale = 0.45;
  double x1scale = 65;
  double x0scaleinverse = 1/x0scale;
  double x1scaleinverse = 1/x1scale;

  int runswitherrors = 0;
     
  send(x); //send initial state
  bool proceed;

  while(true){
     u = receive();  
     if(u == exit){
        break;
     }
     else if(u == reset){
        send(x);
     }
     else{
       stop = action(cvode_mem,t,tstep,&x,&runswitherrors,cdata,&tdat,&xdat,&rdat,&x0,&udat, u0,&xsp,x0scaleinverse,x1scaleinverse,rewardcheck,&rewardsum,rewardtol);
       if(stop){
        break;
      }
    }
      
     //END OF MY VERSION

  // casual variables
   while (t < tfin && i < maxit) { // a little safety check on max iterations.
      // execute the control action for this step
      // **step**
       
  }

static void reset(vector<double>* u, N_Vector* x, N_Vector* xsp, controller_data* cdata, vector<vector<double>>* xdat, vector<double>* rdat, vector<vector<double>>* udat, int* i, double* rad, double* x0scale, double* x1scale, void* cvode_mem){
  (*u0)[0] = 0; 
  (*u0)[1] = 0; 
  (*cdata).Integral[0] = 0;
  (*cdata).Integral[1] = 0;  
  *i = 0; 
  (*xdat).clear();  
  (*rdat).clear(); 
  (*udat).clear();

  // initialize x in a circle surroudning the region of interestit** 
  // set rad to random number between 0 and 2 pi...would happen in initial set up and when you reset
  *rad = rand() %(2.0 * M_PI);
  NV_Ith_S(*x, 0)   = 0.55 + *x0scale * cos(*rad); // mol/m3
  NV_Ith_S(*x, 1)   = 375 + *x1scale * sin(*rad); // deg K

  // new setpoint
  // STANDARD SETPOINT
  NV_Ith_S(*xsp, 0)   = 0.57; // mol/m3
  NV_Ith_S(*xsp, 1)   = 395.3; // deg K

   // reinitialize the integrator --> **reset**
  CVodeReInit(cvode_mem, RCONST(0.0),*x);
}
static void exit(N_Vector* x, N_Vector* abstol, void *cvode_mem){
  // Free state space and abstol vectors 
  N_VDestroy_Serial(*x);
  N_VDestroy_Serial(*abstol);

  // Free integrator memory 
  CVodeFree(&cvode_mem);

 // cout << p << " runs completed (" << runswitherrors << " with errors). Program execution time: " << ((float)(clockout - clockin))/CLOCKS_PER_SEC << " seconds." << endl;

}

static bool action(void* cvode_mem, realtype t, realtype tstep, N_Vector* x, int* runswitherrors, controller_data* cdata, vector<double>* tdat, vector<vector<double>>* xdat, vector<double>* rdat, vector<double>* x0, vector<vector<double>>* udat, vector<double> u0, N_Vector* xsp, double x0scaleinverse, double x1scaleinverse, int rewardcheck, double* rewardsum, double rewardtol)
  bool done = false; 
  flag = CVode(cvode_mem, t + tstep,*x, &t, CV_NORMAL);
  if (flag < 0) {
     cout << "CVode error " << flag;
     *runswitherrors++;
     done ==  true;
  }
 
 *i++;
 cdata.uprev = u0;
 *tdat.push_back(t); // store the t at the beginning of the integratino step`
 (*x0)[0] = NV_Ith_S(x, 0);
 (*x0)[1] = NV_Ith_S(x, 1);
 (*xdat).push_back(*x0); // this is the state at the current t before integration.

 // compute the current reward function, the reward at the current state before integration --> **keep in simulator**
 rwd = 
        -(
          ((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse)*((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse) + 
          ((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)*((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)
         );
           
 (*rdat).push_back(rwd);
 (*udat).push_back(*u0); // this is the state at the beginning of the integration step 

 // check for steady state. Basically, if it hasn't deviated from its previous state by very much then it stops.
 // currently the reward is a deviation from the steady state so can just check the sum of the past few deviations
 if (*i >= rewardcheck)
 {
    *rewardsum = 0;
    for (j=(*rdat).size() - 1; j >= (*rdat).size() - rewardcheck; j--) {
          *rewardsum = *rewardsum + fabs((*rdat)[j]);
    }
    // stop if our reward is smaller than our cumulative tolerance over three iterations
    if (rewardtol > *rewardsum) { 
       done ==  true;
    }
 }

 send(*x);
 send(rwd);
 send(done);

}
// define model constants 
#define F    RCONST(0.1)     // m3/min
#define V    RCONST(0.1)     // m3
#define TIN  RCONST(310.0)   // K
#define DH   RCONST(-4.78e4) // kJ/kmol
#define k0   RCONST(72e9)    // 1/min
#define E    RCONST(8.314e4) // kJ/kmol
#define cp   RCONST(0.239)   // kJ / kg K
#define rho  RCONST(1000.0)   // kg/m3
#define R    RCONST(8.314)   // kJ/kmol-K

// THIS IS THE MAIN REACATOR FUNCTION
// State variables are x[0] = CA in kmol/m3
// x[1] is T in K
// data passed in via void* are the current controller outputs
// u[0] = CAIn in kmol/m3
// u[1] = Q in kJ/min

static int cstrfun2(realtype t, N_Vector x, N_Vector xp, void *user_data) { //**use this to update the state given the action given from our new controller

  // recast the user data pointer
  vector<double>* u = static_cast< vector<double>* >(user_data); 

  // Precalculate some common terms.
  realtype intermed = k0 * NV_Ith_S(x,0) * exp( -E / (R * NV_Ith_S(x,1)) );

  // CA' ==> since #defines are used, repeated calculations of F/V and cp*rho are done once during compiling so this takes no extra flops
  NV_Ith_S(xp,0) = (F/V) * ( (*u)[0] - NV_Ith_S(x,0)) - intermed; 

  // T'
  NV_Ith_S(xp,1) = (F/V) * (TIN - NV_Ith_S(x,1)) + ( DH/(cp * rho) ) * intermed + (*u)[1] / (cp * rho * V);

  return(0);
}




//static int controller(realtype t, N_Vector x, N_Vector xsp, controller_data* cdata, vector<double>* u) {

  //double h; 
  //int N;
  /*

  // classic PI control , 
  // this works really well with Kp= [5, 250] and Ki=[20, 2000]
  // perform integrals
  if ((*(*cdata).tdat).size() > 1) {
  N = (*(*cdata).tdat).size() - 1;
  // trapezoids
  h = ((*(*cdata).tdat)[N] - (*(*cdata).tdat)[N-1]);
  cdata->Integral[0] = cdata->Integral[0] + (NV_Ith_S(xsp,0) - 0.5*((*(*cdata).xdat)[N][0] + (*(*cdata).xdat)[N-1][0]) )*h;
  cdata->Integral[1] = cdata->Integral[1] + (NV_Ith_S(xsp,1) - 0.5*((*(*cdata).xdat)[N][1] + (*(*cdata).xdat)[N-1][1]) )*h;
  }

  (*u)[0] = cdata->Kps[0]*(NV_Ith_S(xsp,0) - NV_Ith_S(x,0)) + cdata->Kis[0] * cdata->Integral[0];
  (*u)[1] = cdata->Kps[1]*(NV_Ith_S(xsp,1) - NV_Ith_S(x,1)) + cdata->Kis[1] * cdata->Integral[1];
  */

  // crappy linear MPC
  // if we have sufficient prior data
  /*
  if ((*(*cdata).tdat).size() > 1) {
  N = (*(*cdata).tdat).size() - 1;
    h = ((*(*cdata).tdat)[N] - (*(*cdata).tdat)[N-1]);
    // Precalculate some common terms.
    realtype intermed = k0 * NV_Ith_S(x,0) * exp( -E / (R * NV_Ith_S(x,1)) );

    //THIS WAS VERY EFFECTIVE...
    (*u)[0] = (
        (NV_Ith_S(xsp,0) - NV_Ith_S(x,0))/h -    // xp approx
        ((F/V)*(-NV_Ith_S(x,0)) - intermed)     // f(x)
        ) / (F/V);                                 // inv(C) portion

    (*u)[1] = (
        (NV_Ith_S(xsp,1) - NV_Ith_S(x,1))/h -                              // xp approx
        ((F/V) * (TIN - NV_Ith_S(x,1)) + ( DH/(cp * rho) ) * intermed)     // f(x)
        ) * (cp * rho * V);                     // inv(C) portion



  } else { // use the PI controller as backup
    (*u)[0] = cdata->Kps[0]*(NV_Ith_S(xsp,0) - NV_Ith_S(x,0)) + cdata->Kis[0] * cdata->Integral[0];
    (*u)[1] = cdata->Kps[1]*(NV_Ith_S(xsp,1) - NV_Ith_S(x,1)) + cdata->Kis[1] * cdata->Integral[1];
  }

  // check valve saturation
  if ((*u)[0] < 0) {(*u)[0] = 0;} 
  if ((*u)[0] > 2) {(*u)[0] = 2;}
  if ((*u)[1] < 0) {(*u)[1] = 0;}  
  if ((*u)[1] > 20000) {(*u)[1] = 20000;}

  return (0);
}

*/
// Modified version of Gavin's example
// should use vectors now (hopefully it works, cross your fingers)
//**NOT worried about this right now**
//void writeData(vector< vector<double> >& states, 
  //  vector< vector<double> >& actions,
   // vector<double>& rewards,
   // int N, int statesDim, int actionDim,
   // H5File* file, string groupname){

  // TOM ADDITION
  // The new vector standard guarantees that the memory stored in the vector is contiguous just like ** arrays.
  // so I should be able to recast the memory address of each row as a double* and that should work...
  double* StatesRowPtr;
  double* ActionRowPtr;
  double* RewardsRowPtr;          

  Group group=file->createGroup("/"+groupname);

  //write states
  hsize_t S_DIMS[2]={N,statesDim};
  DataSpace stateDataspace(2,S_DIMS);
  DataSet dataset = group.createDataSet("states",PredType::NATIVE_DOUBLE,stateDataspace);
  hsize_t M_DIMS[2]={1,statesDim};
  DataSpace memspace(2, M_DIMS);

  for (int row = 0; row < N; row++) {
    hsize_t offset[2]={row,0};
    stateDataspace.selectHyperslab(H5S_SELECT_SET,M_DIMS,offset);
    //dataset.write(states[row],PredType::NATIVE_DOUBLE,memspace,stateDataspace,NULL);
    StatesRowPtr = &states[row][0];
    dataset.write(StatesRowPtr,PredType::NATIVE_DOUBLE,memspace,stateDataspace,0);
  }

  //writeactions
  hsize_t A_DIMS[2]={N,actionDim};
  DataSpace actionDataspace(2,A_DIMS);
  dataset = group.createDataSet("actions",PredType::NATIVE_DOUBLE,actionDataspace);
  M_DIMS[1]=actionDim;
  DataSpace actionmemspace(2, M_DIMS);

  for (int row = 0; row < N; row++) {
    hsize_t offset[2]={row,0};
    actionDataspace.selectHyperslab(H5S_SELECT_SET,M_DIMS,offset);
    //dataset.write(actions[row],PredType::NATIVE_DOUBLE,actionmemspace,actionDataspace,NULL);
    ActionRowPtr = &actions[row][0];
    dataset.write(ActionRowPtr,PredType::NATIVE_DOUBLE,actionmemspace,actionDataspace,0);
  }

  //writeRewards
  hsize_t R_DIMS=N;
  DataSpace rewardDataspace(1,&R_DIMS);
  dataset=
    group.createDataSet("rewards",PredType::NATIVE_DOUBLE,rewardDataspace);
  //dataset.write(rewards,PredType::NATIVE_DOUBLE);
  RewardsRowPtr = &rewards[0];
  dataset.write(RewardsRowPtr,PredType::NATIVE_DOUBLE);

}
