/* ========================================================================== 

   Experiment 1
   Toy Problem with Linear MPC
   Initialized a million times in a ring around the desired steady state
   No plotting on screen
   Writes to data file.
   Thomas A. Adams II
   Started Jan 31 2018

   SERIAL VERSION (NOT YET PARALLELIZED)
   Be sure to module load cray-tpsl-64
   module load cray-hdf5
   ========================================================================== */
#include <unistd.h>
#include <iostream>                                  
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include "mpi.h"
using namespace std;

// Header files for CVODE
#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <cvode/cvode_direct.h>       /* prototype for CVDense */
#include <sundials/sundials_dense.h> /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h> /* definition of type realtype */
#include <sunmatrix/sunmatrix_dense.h> /* access to dense SUNMatrix            */
#include <sunlinsol/sunlinsol_dense.h> /* access to dense SUNLinearSolver      */

// prototype for the function we have
static int cstrfun2(realtype t, N_Vector x, N_Vector xp, void *user_data);
void cleanUp(N_Vector& x, N_Vector& abstol, void* cvode_mem);
static void reset(vector<double>* u0, N_Vector& x, N_Vector& xsp, vector<double>* rdat, int* i, double* rad, double x0scale, double x1scale, void* cvode_mem, double* reward);
double calcReward(N_Vector x, N_Vector xsp, double x0scaleinverse,double
    x1scaleinverse);
bool steadyCheck(vector<double> rdat, int rewardcheck, double rewardtol,int i);

int main(void) {
  int rank, numprocs, k;
  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

	// casual variables
	int i, p;
	clock_t clockin, clockout;

	// allocate memory for results storage and provide initial variables
	vector<double> tdat;  // initial state and main memory object for t
	vector<double> x0(2, 0); // empty vector  
	vector< vector<double> > udat; // u state space. start with zeros [CAin, Q].
	vector< vector<double> > xdat;  // x state space history (reporting intervals)
	vector<double> rdat; // reward data
	vector<double> u0(2, 0); // the current control action.
	vector<double> clearvec(2,0); // something convenient

	// steady state checkers
	int j;  // storage for temporary sums used later
	double rewardsum;
	int rewardcheck = 10; // how many timesteps to check?
  double reward=0.0;
	double rewardtol = 0.0001; // the reward tolerance for when we've reached our goal. 
	// Since it's scaled on the rnage, a reward tolernace of 0.0001 means within 0.01% of the range

	// this is our desired setpoint
	N_Vector xsp     = N_VNew_Serial(2);

	// CV ODE types for use with the integrator. Initial states and settings.
	N_Vector x       = N_VNew_Serial(2); // create an x vector of two state variables

	N_Vector abstol  = N_VNew_Serial(2); // the vector of absolute tolerances
	NV_Ith_S(abstol, 0) = RCONST(1.0e-8);  // ok with CA (0...2) to the 1e-8      mol/m3
	NV_Ith_S(abstol, 1) = RCONST(1.0e-6);  // ok with T (300-400) to the 1e06

	realtype reltol  = RCONST(1.0e-4); // ok with relative error of 0.01%

	// Create the CV ODE memory space and provide settings.
	void *cvode_mem;
	cvode_mem =CVodeCreate(CV_BDF,CV_NEWTON);             // specify using backwards difference methods (stiff)
	CVodeInit(cvode_mem, cstrfun2, RCONST(0.0), x);  // initialize at time zero
	CVodeSVtolerances(cvode_mem, reltol, abstol);    // specify the tolerances
	SUNLinearSolver LS=SUNDenseLinearSolver(x, SUNDenseMatrix(2, 2));
        CVDlsSetLinearSolver(cvode_mem, reltol, abstol);                               // specify the dense linear solver
	CVodeSetMaxNumSteps(cvode_mem, 5000);             // sets the maximum number of steps
	CVodeSetUserData(cvode_mem, &u0);                // sets the user data pointer

	// TEMP
	// execute for 1 minute
	realtype tstep = RCONST(0.01); // the reporting interval / the time between updates to the control variabels.
	realtype tfin = RCONST(10); // the desired final time
	realtype t; // the time at the end of the integrator, which may be earlier than tout if it failed
	realtype maxit = tfin / tstep + RCONST(100);

	// start the timer
	clockin = clock();
	double rad = 0;

	double x0scale = 0.45;
	double x1scale = 65;
	double x0scaleinverse = 1/x0scale;
	double x1scaleinverse = 1/x1scale;

	int runswitherrors = 0;

	// reset variables to beginning 
	t = RCONST(0.000);
	xdat.clear();
	tdat.clear();
	rdat.clear();
	udat.clear();

  reset(&u0, x, xsp, &rdat, &i, &rad, x0scale, x1scale, cvode_mem, &reward);
  rdat.push_back(reward);
  MPI_Send(x, 2, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
  MPI_Send(&reward, 1, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
	
	while (t < tfin && i < maxit) { // a little safety check on max iterations.
    //get msg - if exit, call exit and break, if reset, call reset, else:
		//controller(t, x, xsp, &cdata, &u0); TODO get action
		// execute the ODE for one control step
		int flag = CVode(cvode_mem, t + tstep, x, &t, CV_NORMAL);
    reward=calcReward(x,xsp,x0scaleinverse,x1scaleinverse);
		rdat.push_back(reward);
		// check for steady state. Basically, if it hasn't deviated from its previous state by very much then it stops.
    bool done = steadyCheck(rdat,rewardcheck,rewardtol,i);
		// currently the reward is a deviation from the steady state so can just check the sum of the past few deviations
		if (flag < 0) {
			cout << "CVode error " << flag << " in loop " << p << ". Breaking." << endl;
      done=true;
		}
		i++;
	}
  cleanUp(x,abstol,cvode_mem);
	return(0);
}

bool steadyCheck(vector<double> rdat, int rewardcheck, double rewardtol,int i) {
  if (i >= rewardcheck) {
    double rewardsum = 0;
    for (int j=rdat.size() - 1; j >= rdat.size() - rewardcheck; j--)
      rewardsum = rewardsum + fabs(rdat[j]);
    // stop if our reward is smaller than our cumulative tolerance over three iterations
    if (rewardtol > rewardsum)
      return true;
  }
  return false;
}

double calcReward(N_Vector x, N_Vector xsp,
                double x0scaleinverse,double x1scaleinverse){
  double r=-(
      ((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse)*((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse) + 
      ((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)*((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)
      );
  return r;
}

static void reset(vector<double>* u0, N_Vector& x, N_Vector& xsp, vector<double>* rdat, int* i, double* rad, double x0scale, double x1scale, void* cvode_mem, double* reward){
  (*u0)[0] = 0; 
  (*u0)[1] = 0;   
  *i = 0; 
  (*rdat).clear(); 

  // initialize x in a circle surroudning the region of interestit** 
  // set rad to random number between 0 and 2 pi...would happen in initial set up and when you reset
  *rad = ((double)rand()/RAND_MAX) / (2.0 * M_PI);
  NV_Ith_S(x, 0)   = 0.55 + x0scale * cos(*rad); // mol/m3
  NV_Ith_S(x, 1)   = 375 + x1scale * sin(*rad); // deg K

  // new setpoint
  // STANDARD SETPOINT
  NV_Ith_S(xsp, 0)   = 0.57; // mol/m3
  NV_Ith_S(xsp, 1)   = 395.3; // deg K

   // reinitialize the integrator --> **reset**
  CVodeReInit(cvode_mem, RCONST(0.0),x);
  double x0inv = 1/x0scale;
  double x1inv = 1/x1scale;

  *reward=calcReward(x,xsp,x0inv,x1inv);
}

void cleanUp(N_Vector& x, N_Vector& abstol, void* cvode_mem) {
	// Free state space and abstol vectors 
	N_VDestroy_Serial(x);
	N_VDestroy_Serial(abstol);

	// Free integrator memory 
	CVodeFree(&cvode_mem);
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

static int cstrfun2(realtype t, N_Vector x, N_Vector xp, void *user_data) {

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
