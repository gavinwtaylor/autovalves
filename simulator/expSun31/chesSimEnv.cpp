#include "chesSimEnv.h"

CSTREnv::CSTREnv():x0(2,0),u0(2,0), clearvec(2,0){
	rewardcheck = 10; // how many timesteps to check?
	reward=0.0;
	rewardtol = 0.0001; // the reward tolerance for when we've reached our goal. 
	xsp = N_VNew_Serial(2);
	x = N_VNew_Serial(2); // create an x vector of two state variables
	abstol  = N_VNew_Serial(2); // the vector of absolute tolerances
	NV_Ith_S(abstol, 0) = RCONST(1.0e-8);  // ok with CA (0...2) to the 1e-8      mol/m3
	NV_Ith_S(abstol, 1) = RCONST(1.0e-6);  // ok with T (300-400) to the 1e06 
	reltol = RCONST(1.0e-4); // ok with relative error of 0.01%

	cvode_mem =CVodeCreate(CV_BDF,CV_NEWTON);             // specify using backwards difference methods (stiff)
	CVodeInit(cvode_mem, cstrfun2, RCONST(0.0), x);  // initialize at time zero
	CVodeSVtolerances(cvode_mem, reltol, abstol);    // specify the tolerances
	SUNLinearSolver LS=SUNDenseLinearSolver(x, SUNDenseMatrix(2, 2));
	CVDlsSetLinearSolver(cvode_mem, LS, SUNDenseMatrix(2,2)); // specify the dense linear solver
	CVodeSetMaxNumSteps(cvode_mem, 5000);             // sets the maximum number of steps
	CVodeSetUserData(cvode_mem, &u0);                // sets the user data pointer

	tstep = RCONST(0.01); // the reporting interval / the time between updates to the control variabels.
	tfin = RCONST(10); // the desired final time
	maxit = tfin / tstep + RCONST(100);

	clockin = clock();
	rad = 0;
	x0scale = 0.45;
	x1scale = 65;
	x0scaleinverse = 1/x0scale;
	x1scaleinverse = 1/x1scale;
	runswitherrors = 0;             
	t = RCONST(0.000);
	xdat.clear();
	tdat.clear();
	rdat.clear();
	udat.clear();

	refresh();
	rdat.push_back(reward);

}

bp::tuple CSTREnv::step(bp::numpy::ndarray action){
	bool done =false;
	u0[0]=bp::extract<double>(action[0]);
	u0[1]=bp::extract<double>(action[1]);

	int flag = CVode(cvode_mem, t + tstep, x, &t, CV_NORMAL);

	reward=calcReward();      
	if(!withinOval()){
		reward=reward*100;
		done = true;
	}

	rdat.push_back(reward);	
	done = steadyCheck();
	if (flag < 0) {
		cout << "CVode error"  << endl;
		done=true;
	}

	i++;

	if(i >= maxit){
		done = true;
	}

	return bp::make_tuple((bp::make_tuple(NV_Ith_S(x,0), NV_Ith_S(x,1))), reward, done); 

}

void CSTREnv::reset(){ 
	refresh();
}

bool CSTREnv::steadyCheck() {
	if (i >= rewardcheck) {
		double rewardsum = 0;
		for (int j=rdat.size() - 1; j >= rdat.size() - rewardcheck; j--)
			rewardsum = rewardsum + fabs(rdat[j]);
		// stop if our reward is smaller than our cumulative tolerance over three iterations
		if (rewardtol > rewardsum){
			cout << "YES! STEADY STATE"<<endl;
			return true;
		}
	}
	return false;
}

double CSTREnv::calcReward(){
	double r=-(
			((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse)*((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse) + 
			((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)*((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)
		  );
	return r;
}

bool CSTREnv::withinOval(){
	double centerx0=(NV_Ith_S(x,0)-.55)/x0scale;
	double centerx1=(NV_Ith_S(x,1)-375)/x1scale;
	double dist=centerx0*centerx0+centerx1*centerx1;
	return dist<1;
}

void CSTREnv::refresh(){
	u0[0] = 0; 
	u0[1] = 0;   
	i = 0; 
	rdat.clear(); 

	// initialize x in a circle surroudning the region of interestit** 
	// set rad to random number between 0 and 2 pi...would happen in initial set up and when you reset
	rad = ((double)rand()/RAND_MAX) / (2.0 * M_PI);
	rad = 2;
	NV_Ith_S(x, 0)   = 0.55 + x0scale * cos(rad); // mol/m3
	NV_Ith_S(x, 1)   = 375 + x1scale * sin(rad); // deg K

	// new setpoint
	// STANDARD SETPOINT
	NV_Ith_S(xsp, 0)   = 0.57; // mol/m3
	NV_Ith_S(xsp, 1)   = 395.3; // deg K

	// reinitialize the integrator --> **reset**
	CVodeReInit(cvode_mem, RCONST(0.0),x);
	double x0inv = 1/x0scale;
	double x1inv = 1/x1scale;

	reward=calcReward();
}

void CSTREnv::cleanUp() {
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
	std::vector<double>* u = static_cast< vector<double>* >(user_data); 

	// Precalculate some common terms.
	realtype intermed = k0 * NV_Ith_S(x,0) * exp( -E / (R * NV_Ith_S(x,1)) );

	// CA' ==> since #defines are used, repeated calculations of F/V and cp*rho are done once during compiling so this takes no extra flops
	NV_Ith_S(xp,0) = (F/V) * ( (*u)[0] - NV_Ith_S(x,0)) - intermed; 

	// T'
	NV_Ith_S(xp,1) = (F/V) * (TIN - NV_Ith_S(x,1)) + ( DH/(cp * rho) ) * intermed + (*u)[1] / (cp * rho * V);
	return(0);
}



bp::BOOST_PYTHON_MODULE(simulator)
{
	class_<Simulator>("Simulator", init())
		.def("reset", &Simulator::reset)
		.def("step", &Simulator::step)
		;
}
