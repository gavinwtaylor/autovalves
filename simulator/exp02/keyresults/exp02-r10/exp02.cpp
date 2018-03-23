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

#include <iostream>                                  
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string>
#include <sstream>
#include <stdio.h>
#include "mpi.h"
#include "termplot.h"
using namespace std;

// Header files for CVODE
#include <cvode/cvode.h>             /* prototypes for CVODE fcts., consts. */
#include <nvector/nvector_serial.h>  /* serial N_Vector types, fcts., macros */
#include <cvode/cvode_dense.h>       /* prototype for CVDense */
#include <sundials/sundials_dense.h> /* definitions DlsMat DENSE_ELEM */
#include <sundials/sundials_types.h> /* definition of type realtype */

// Header files for HD5
#include "H5Cpp.h"
using namespace H5;

// needed for usleep tests
#include <unistd.h>

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
struct controller_data
{
  vector<double> uprev; 
  vector<double> Kps; 
  vector<double> Kis; 
  vector<double>* tdat;
  vector< vector<double> >* xdat;
  vector<double> Integral; 
};

// prototype for the control action
static int controller(realtype t, N_Vector x, N_Vector xsp, controller_data* cdata, vector<double>* u);

int main(void) {

	// sim settings
	// int NRuns = 100000;
	int NRuns = 1;
	int SafetyMax = 50000000; // Max number of time steps per run, which is irrespective of timestep.  
	
  // MPI settings
  int rank;
  int size;
  MPI_Init(NULL,NULL);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  cout << "C++ " << rank << " of " << size << endl;
  double state[2];
  double action[2];
	
	// file settings
	string experimentnum = "02";
	string runnumber = "10";
	string groupname = "Run-";
	char buffer[40];
	string filename="/mnt/lustre/scratch/autoValveData/exp" + experimentnum + "-run" + runnumber + ".h5";
  H5File* file=new H5File(filename, H5F_ACC_EXCL);

  // casual variables
  int flag, i, p;
  clock_t clockin, clockout;

  // allocate memory for results storage and provide initial variables
  vector<double> tdat;  // initial state and main memory object for t
	vector<double> x0(2, 0); // empty vector  
  vector< vector<double> > udat; // u state space. start with zeros [CAin, Q].
  vector< vector<double> > xdat;  // x state space history (reporting intervals)
  vector<double> rdat; // reward data
  vector<double> u0(2, 0); // the current control action.
  vector<double> clearvec(2,0); // something convenient
  controller_data cdata;
  cdata.uprev = u0;
  
  // printing conveniences
  vector< vector<double> > u0vtime; // convenience for printing;
  vector< vector<double> > u1vtime; // convenience for printing;
 
  // PID settings... used for first run even in linear MPC
  cdata.Kps = clearvec;
  cdata.Kps[0] = 5;
  cdata.Kps[1] = 250;
  cdata.Kis = clearvec;
  cdata.Kis[0] = 20;
  cdata.Kis[1] = 2000;
  cdata.tdat = &tdat;
  cdata.xdat = &xdat;
  cdata.Integral = clearvec;
  
  // steady state checkers
  int j, k;  // storage for temporary sums used later
  double rewardsum;
  int rewardcheck = 10; // how many timesteps to check?
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
	cvode_mem = CVodeCreate(CV_BDF, CV_NEWTON);             // specify using backwards difference methods (stiff)
	CVodeInit(cvode_mem, cstrfun2, RCONST(0.0), x);  // initialize at time zero
	CVodeSVtolerances(cvode_mem, reltol, abstol);    // specify the tolerances
	CVDense(cvode_mem, 2);                           // specify the dense linear solver
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
	
	// master outer loop
	for (p=0; p<NRuns; p++) {
	
	  // reset variables to beginning 
	  u0[0] = 0;
	  u0[1] = 0;
	  cdata.Integral[0] = 0;
	  cdata.Integral[1] = 0;
	  t = RCONST(0.000);
	  i = 0;
	  xdat.clear();
	  tdat.clear();
	  rdat.clear();
	  udat.clear();
	  
	  // initialize x in a circle surroudning the region of interest; 
    rad = (double(p)/double(NRuns)) * 2.0 * M_PI;
		NV_Ith_S(x, 0)   = 0.55 + x0scale * cos(rad); // mol/m3
    NV_Ith_S(x, 1)   = 375 + x1scale * sin(rad); // deg K
	
    // new setpoint
    // STANDARD SETPOINT
	  NV_Ith_S(xsp, 0)   = 0.57; // mol/m3
    NV_Ith_S(xsp, 1)   = 395.3; // deg K
    
    //cout << "Loop " << p << " initial state with rad: " << rad << " time " << t << " at [" << NV_Ith_S(x, 0) << ", " << NV_Ith_S(x, 1) << "]" << endl;
    
    // reinitialize the integrator
		CVodeReInit(cvode_mem, RCONST(0.0), x);
	
	  while (t < tfin && i < maxit && i < SafetyMax) { // a little safety check on max iterations.
	    // execute the control action for this step
	    cdata.uprev = u0;
	    tdat.push_back(t); // store the t at the beginning of the integratino step
	    x0[0] = NV_Ith_S(x, 0);
	    x0[1] = NV_Ith_S(x, 1);
			xdat.push_back(x0); // this is the state at the current t before integration.
			
			// compute the current reward function, the reward at the current state before integration
		  rdat.push_back( 
			 -(
			    ((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse)*((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse) + 
			    ((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)*((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)
			  )
			);
									
			// controller section
			// OLD VERSION - This calls the controller function
			// controller(t, x, xsp, &cdata, &u0);
			
			////// NEW AI CONTROLLER BEGIN
			
			// send the current state to the AI controller on the other node
			// not sure if MPI will take a vector<double> so recast as double[]
			state[0] = x0[0]; 
      state[1] = x0[1]; 
      
      // send this state to the controller (AI)
      //cout << "Simulator sending state " << state[0] << ' ' << state[1] << endl;
      MPI_Send(state,2,MPI_DOUBLE,1,0,MPI_COMM_WORLD);
      
      // wait for the AI to give us an action to take.
      int err = MPI_Recv(action,2,MPI_DOUBLE,1,1,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
      
      //cout << "Simulator receiving action " << action[0] << ' ' << action[1] << " err code was " << err << endl;
      
      //if (MPI_SUCCESS == err) cout << "Error code was MPI_SUCCESS\n";
      //if (MPI_ERR_COMM == err) cout << "Error code was MPI_ERR_COMM\n";
      //if (MPI_ERR_TYPE == err) cout << "Error code was MPI_ERR_TYPE\n";
      //if (MPI_ERR_COUNT == err) cout << "Error code was MPI_ERR_COUNT\n";
      //if (MPI_ERR_TAG == err) cout << "Error code was MPI_ERR_TAG\n";
      //if (MPI_ERR_RANK == err) cout << "Error code was MPI_ERR_RANK\n";
	    
      // copy this action stored as double[] into our vector<double> we have set up already
      u0[0] = action[0];
      u0[1] = action[1];
      
      // double check for valve saturation in case external controller gave us crap
      if (u0[0] < 0) {u0[0] = 0;} 
    	if (u0[0] > 2) {u0[0] = 2;}
	    if (u0[1] < 0) {u0[1] = 0;}  
	    if (u0[1] > 20000) {u0[1] = 20000;}
	    
	    ///////// END AI CONTROLLER SECTION
      
      udat.push_back(u0); // this is the state at the beginning of the integration step 
      
      // more data stored in convenient format for plotting. This is still at the beginning of the integration step but after controller has given actions
		  x0[0] = t;
		  x0[1] = u0[0];
		  u0vtime.push_back(x0);
		  x0[1] = u0[1];
		  u1vtime.push_back(x0);
      
      // check for steady state. Basically, if it hasn't deviated from its previous state by very much then it stops.
		  // currently the reward is a deviation from the steady state so can just check the sum of the past few deviations
		  if (i >= rewardcheck)
		  {
		     rewardsum = 0;
		     for (j=rdat.size() - 1; j >= rdat.size() - rewardcheck; j--) {
				   rewardsum = rewardsum + fabs(rdat[j]);
				 }
		     // stop if our reward is smaller than our cumulative tolerance over three iterations
		     if (rewardtol > rewardsum) { 
		       break;
				 }
			}
	
	    // execute the ODE for one control step
		  flag = CVode(cvode_mem, t + tstep, x, &t, CV_NORMAL);
		  
		  if (flag < 0) {
		    cout << "CVode error " << flag << " in loop " << p << ". Breaking." << endl;
		    runswitherrors++;
		    break;
			}
		  		  

	    i++;
	  }
	  
	  sprintf(buffer, "%d", p);
	  groupname = "Run-";
		
		if (p < 1000000) groupname = groupname + "0";
		if (p < 100000) groupname = groupname + "0";
		if (p < 10000) groupname = groupname + "0";
		if (p < 1000) groupname = groupname + "0";
		if (p < 100) groupname = groupname + "0";
		if (p < 10) groupname = groupname + "0";
		groupname = groupname + buffer;
		writeData(xdat, udat, rdat, xdat.size(), xdat.back().size(), udat.back().size(), file, groupname);
		
		
		// temp
		//cout << "Final state at t=" << t << " [" << x0[0] << ", " << x0[1] << "] in " << groupname << endl;
	}
	
	// stop the timer
	clockout = clock();

  // Free state space and abstol vectors 
  N_VDestroy_Serial(x);
  N_VDestroy_Serial(abstol);

  // Free integrator memory 
  CVodeFree(&cvode_mem);
  
  
    // Plot the results
  vector<double> x0bounds;
	vector<double> x1bounds;
	vector<double> termsize(2, 50);   // pick a default for safety
	
	// set the terminal size since this is going to run from qsub
	termsize[0] = 40;
	termsize[1] = 80;
	
	cout << "CA (X-axis) v T (y-axis)" << endl;
	x0bounds.push_back(0.38);
	x0bounds.push_back(0.82	);
	x1bounds.push_back(420);
	x1bounds.push_back(375);	
	termplot(xdat, x0bounds, x1bounds, termsize);
	
	cout << "\nt (X-axis) v u0 (CAin) (y-axis)" << endl;
	x0bounds[0] = u0vtime[0][0];
	x0bounds[1] = u0vtime.back()[0];
	x1bounds[0] = 2;
	x1bounds[1] = -0.25;	
	termplot(u0vtime, x0bounds, x1bounds, termsize);
	
	cout << "\nt (X-axis) v u1 (Q) (y-axis)" << endl;
	x1bounds[0] = 11000;
	x1bounds[1] = -500;	
	termplot(u1vtime, x0bounds, x1bounds, termsize);
  
  
  
  
  
  
  
  // File cleanup
  delete file;
  
  // sends MPI signal that we're done now.
  MPI_Send(NULL,0,MPI_INT,1,2,MPI_COMM_WORLD);
  
  cout << p << " runs completed (" << runswitherrors << " with errors). Program execution time: " << ((float)(clockout - clockin))/CLOCKS_PER_SEC << " seconds." << endl;
  
  // close MPI now.
  MPI_Finalize();
  
  // goodbye! 
  return(0);
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




static int controller(realtype t, N_Vector x, N_Vector xsp, controller_data* cdata, vector<double>* u) {

	double h; 
	int N;
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


 // Modified version of Gavin's example
 // should use vectors now (hopefully it works, cross your fingers)

void writeData(vector< vector<double> >& states, 
               vector< vector<double> >& actions,
							 vector<double>& rewards,
               int N, int statesDim, int actionDim,
               H5File* file, string groupname){
               
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
  DataSet dataset = 
    group.createDataSet("states",PredType::NATIVE_DOUBLE,stateDataspace);
  hsize_t M_DIMS[2]={1,statesDim};
  DataSpace memspace(2, M_DIMS);

  for (int row = 0; row < N; row++) {
    hsize_t offset[2]={row,0};
    stateDataspace.selectHyperslab(H5S_SELECT_SET,M_DIMS,offset);
    //dataset.write(states[row],PredType::NATIVE_DOUBLE,memspace,stateDataspace,NULL);
    StatesRowPtr = &states[row][0];
    dataset.write(StatesRowPtr,PredType::NATIVE_DOUBLE,memspace,stateDataspace,NULL);
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
    //dataset.write(actions[row],PredType::NATIVE_DOUBLE,actionmemspace,actionDataspace,NULL);
    ActionRowPtr = &actions[row][0];
    dataset.write(ActionRowPtr,PredType::NATIVE_DOUBLE,actionmemspace,actionDataspace,NULL);
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
