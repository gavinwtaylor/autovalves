#include "cstrEnv.h"

/**
 * Constructor.  Defines initial state, setpoints, etc.  Depends upon reset()
 */
CSTREnv::CSTREnv():u0(2,0),numsteps(0),x0scale(0.45),x1scale(65) {
  srand(time(NULL));
  xsp=N_VNew_Serial(2);
  x=N_VNew_Serial(2);
  abstol=N_VNew_Serial(2);
  NV_Ith_S(abstol, 0) = RCONST(1.0e-8);  // ok with CA (0...2) to the 1e-8      mol/m3
  NV_Ith_S(abstol, 1) = RCONST(1.0e-6);  // ok with T (300-400) to the 1e06
  reltol  = RCONST(1.0e-4); // ok with relative error of 0.01%
  cvode_mem =CVodeCreate(CV_BDF,CV_NEWTON);             // specify using backwards difference methods (stiff)
  CVodeInit(cvode_mem, cstrfun2, RCONST(0.0), x);  // initialize at time zero
  CVodeSVtolerances(cvode_mem, reltol, abstol);    // specify the tolerances
  SUNLinearSolver LS=SUNDenseLinearSolver(x, SUNDenseMatrix(2, 2));
  CVDlsSetLinearSolver(cvode_mem, LS, SUNDenseMatrix(2,2)); // specify the dense linear solver
  CVodeSetMaxNumSteps(cvode_mem, 5000);             // sets the maximum number of steps
  CVodeSetUserData(cvode_mem, &u0);                // sets the user data pointer

  reset();
}

/**
 * Assumes a 2-element action and 2-element state.  Input is a 2-element tuple, representing the actions
 * assumed to have already been clipped.  Returns a 3-element tuple (state, reward, done), where state
 * is a 2-element tuple, reward is a double, and done is a boolean.
 */
boost::python::tuple CSTREnv::step(boost::python::tuple action){
  u0[0]=boost::python::extract<double>(action[0]);
  u0[1]=boost::python::extract<double>(action[1]);
  int flag = CVode(cvode_mem, t + tstep, x, &t, CV_NORMAL);
  double reward=calcReward();      
  bool done = false;
  if(!withinOval())
    done = true;
  rdat.push_back(reward);	
  done = steadyCheck();
  if (flag < 0) {
    cout << "CVode error"  << endl;
    done=true;
  }

  numsteps++;

  if(numsteps >= maxit)
    done = true;

  boost::python::tuple state=boost::python::make_tuple(NV_Ith_S(x,0),NV_Ith_S(x,1),NV_Ith_S(xsp,0),NV_Ith_S(xsp,1));
  boost::python::tuple retVal=boost::python::make_tuple(state,reward,done);
  return retVal;
}

/* Calculates the squared distance from the setpoint, scaled by x0scale, and x1scale */
double CSTREnv::calcReward(){
  double x0scaleinverse=1/x0scale;
  double x1scaleinverse=1/x1scale;
  double r=-(
      ((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse)*((NV_Ith_S(x, 0) - NV_Ith_S(xsp, 0))*x0scaleinverse) + 
      ((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)*((NV_Ith_S(x, 1) - NV_Ith_S(xsp, 1))*x1scaleinverse)
      );
  return r;
}

/* Resets without having to rebuild the object, required by baselines */
boost::python::tuple CSTREnv::reset(){
  tstep = RCONST(0.01); // the reporting interval / the time between updates to the control variabels.
  realtype tfin = RCONST(10); // the desired final time
  t = RCONST(0.000);
  maxit = tfin / tstep + RCONST(100);
  u0[0] = 0; 
  u0[1] = 0;   
  numsteps = 0; 
  rdat.clear(); 

  // initialize x in a circle surroudning the region of interestit** 
  // set rad to random number between 0 and 2 pi...would happen in initial set up and when you reset
  //double rad = 2;
  double rad = ((double)rand()/RAND_MAX) * (2.0 * M_PI);
  NV_Ith_S(x, 0)   = 0.55 + x0scale * cos(rad); // mol/m3
  NV_Ith_S(x, 1)   = 375 + x1scale * sin(rad); // deg K

  // new setpoint
  // STANDARD SETPOINT
  rad = ((double)rand()/RAND_MAX) * (2.0 * M_PI);
  double radius=(double)rand()/RAND_MAX;
  NV_Ith_S(xsp, 0)   = 0.55 + radius * x0scale * cos(rad); // mol/m3
  NV_Ith_S(xsp, 1)   = 375 + radius * x1scale * sin(rad); // deg K

  // reinitialize the integrator --> **reset**
  CVodeReInit(cvode_mem, RCONST(0.0),x);
  return boost::python::make_tuple(NV_Ith_S(x,0),NV_Ith_S(x,1),NV_Ith_S(xsp,0),NV_Ith_S(xsp,1));
}

/* Destructor, deallocating memory when the python garbage collector cleans it up*/
CSTREnv::~CSTREnv(){
  N_VDestroy_Serial(x);
  N_VDestroy_Serial(xsp);
  N_VDestroy_Serial(abstol);

  // Free integrator memory 
  CVodeFree(&cvode_mem);
}

/* Returns true if we've been close enough to the setpoint for long enough, as determined by
 * REWARDTOL and REWARDCHECK
 */
bool CSTREnv::steadyCheck(){
  if (numsteps >= REWARDCHECK) {
    double rewardsum = 0;
    for (int j=rdat.size() - 1; j >= rdat.size() - REWARDCHECK; j--)
      rewardsum = rewardsum + fabs(rdat[j]);
    // stop if our reward is smaller than our cumulative tolerance over three iterations
    if (REWARDTOL > rewardsum){
      cout << "YES! STEADY STATE"<<endl;
      return true;
    }
  }
  return false;
}

/* Returns if we're within the stable oval, or if we're out where there be dragons */
bool CSTREnv::withinOval(){
  double centerx0=(NV_Ith_S(x,0)-.55)/x0scale;
  double centerx1=(NV_Ith_S(x,1)-375)/x1scale;
  double dist=centerx0*centerx0+centerx1*centerx1;
  return dist<1;
}

/* Returns a tuple containing what's necessary to compute a reward: the setpoint (as a 2-element tuple),
 * the x0scale, and the x1scale. */
boost::python::tuple CSTREnv::getrewardstuff() {
  boost::python::tuple setpoint = boost::python::make_tuple(NV_Ith_S(xsp,0),NV_Ith_S(xsp,1));
  return boost::python::make_tuple(setpoint, x0scale, x1scale);
}

/* RHS to the integrator (did I use those words right?) */
static int cstrfun2(realtype t, N_Vector x, N_Vector xp, void *user_data) {
  // recast the user data pointer
  vector<double>* u = static_cast< vector<double>* >(user_data); 

  // Precalculate some common terms.
  realtype intermed = k0 * NV_Ith_S(x,0)
                      * exp( -E / (R * NV_Ith_S(x,1)) );

  // CA' ==> since #defines are used, repeated calculations of F/V and cp*rho are done once during compiling so this takes no extra flops
  NV_Ith_S(xp,0) = (F/V) * ( (*u)[0] - NV_Ith_S(x,0)) - intermed; 

  // T'
  NV_Ith_S(xp,1) = (F/V) * (TIN - NV_Ith_S(x,1)) + ( DH/(cp * rho) ) * intermed + (*u)[1] / (cp * rho * V);

  return(0);
}

/* Exposes the constructor, reset, and step functions for use as a Python module */
using namespace boost::python;
BOOST_PYTHON_MODULE(cstrMany){
  class_<CSTREnv>("CSTREnv")
    .def("reset", &CSTREnv::reset)
    .def("step", &CSTREnv::step)
    .def("getrewardstuff", &CSTREnv::getrewardstuff)
    ;
}
